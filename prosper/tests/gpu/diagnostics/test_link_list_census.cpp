// PROSPER_DRAW_LINKSCAN — the CPU-side census of the linked lists a graphics draw's scalar buffers
// contain.
//
// What is pinned here is the one modelling decision the whole instrument rests on, and it is the
// decision a reasonable implementer gets wrong: **an out-of-range scalar buffer read returns
// architectural ZERO, and zero is a link, not an exit.** A census that treated an out-of-range index
// as "the walk ended" would report a runaway list as terminating — a confident wrong answer about
// exactly the case the instrument exists to find (#3214: Astro Bot's world-map light-list walk,
// whose terminator is 0xffffffff, so every zero it reads is a jump back to record 0).
//
// The zero-pool case below is the counter-example built BY HAND, outside anything that produced a
// null, per the charter's positive-control rule: an unpopulated pool of zeros is not "an empty
// list", it is an infinite one, and no trip bound can end it.
#include "gpu/diagnostics/link_list_census.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

namespace {

constexpr uint32_t kTerm = 0xffffffffu;

// Astro Bot's shape: 8-byte records, the successor in the second dword, walked until 0xffffffff.
LinkListEncoding astro_encoding() {
    LinkListEncoding e;
    e.record_stride_dwords = 2;
    e.next_dword_offset = 1;
    e.terminator = kTerm;
    return e;
}

// records[i] = {tag, next}. Written this way so each test's LIST is readable as a list.
std::vector<uint32_t> pool(std::initializer_list<uint32_t> next_links) {
    std::vector<uint32_t> words;
    uint32_t tag = 0x70000000u;
    for (uint32_t next : next_links) { words.push_back(tag++); words.push_back(next); }
    return words;
}

}  // namespace

int main() {
    printf("== test_link_list_census ==\n");
    const auto enc = astro_encoding();

    // ---- an ordinary terminating chain -------------------------------------------------------
    {
        // 0 -> 1 -> 2 -> terminator, and record 3 is its own separate terminator.
        const auto words = pool({1u, 2u, kTerm, kTerm});
        const auto c = census_self_walk(words, enc);
        CHECK(c.records == 4 && c.starts == 4, "the self-walk starts once per record");
        CHECK(c.terminating == 4 && c.cyclic == 0, "a terminating pool has no cyclic starts");
        CHECK(c.cycle_nodes == 0, "a terminating pool has no cycle members");
        CHECK(c.longest == 3, "the longest chain is a DEPTH (0->1->2->term is three links)");
        CHECK(c.oob_starts == 0, "no walk left the buffer");
        CHECK(c.consistent(), "terminating + cyclic == starts");
    }

    // ---- THE ZERO POOL: an unpopulated pool is an INFINITE list, not an empty one -------------
    // This is the failure #3214 is about. Every link is zero; zero is a valid record index; record
    // zero's successor is zero. Nothing here is out of range and nothing here terminates.
    {
        const std::vector<uint32_t> words(64, 0u);
        const auto c = census_self_walk(words, enc);
        CHECK(c.records == 32 && c.starts == 32, "a 64-dword zero pool holds 32 records");
        CHECK(c.terminating == 0 && c.cyclic == 32,
              "EVERY start of an all-zero pool is cyclic -- zero is a link, not a terminator");
        CHECK(c.cycle_nodes == 1 && c.sample_count == 1 && c.sample_link[0] == 0 &&
                  c.sample_next[0] == 0,
              "the ring is the one-node self-loop at record 0");
        CHECK(c.oob_starts == 0, "the zero pool never leaves the buffer -- it loops inside it");
        CHECK(c.consistent(), "terminating + cyclic == starts");

        const auto h = histogram_words(words, kTerm);
        CHECK(h.words == 64 && h.zero == 64 && h.terminator == 0 && h.other == 0 &&
                  h.first_other_index == UINT32_MAX,
              "the histogram reports an all-zero buffer, with no counter-example dword");
    }

    // ---- a populated pool whose heads are all the terminator ---------------------------------
    // The other way to say "no lights in this tile", and the one that is CORRECT. It must not be
    // confusable with the zero pool above, which is why the histogram counts them separately.
    {
        const std::vector<uint32_t> heads(16, kTerm);
        const auto words = pool({kTerm, kTerm});
        const auto c = census_head_walk(words, heads, enc);
        CHECK(c.starts == 16 && c.terminating == 16 && c.cyclic == 0,
              "a head equal to the terminator is an empty list, and an empty list terminates");
        CHECK(c.longest == 0, "an empty list has depth zero");
        const auto h = histogram_words(heads, kTerm);
        CHECK(h.zero == 0 && h.terminator == 16 && h.other == 0,
              "the histogram separates a terminator-filled head table from a zero-filled one");
    }

    // ---- OUT OF RANGE READS ZERO, AND ZERO IS NOT AN EXIT ------------------------------------
    // The arm that fails if the walk models an out-of-range index as "the list ended". Record 0
    // terminates, so an out-of-range head lands on it and DOES terminate -- but only after leaving
    // the buffer, which oob_starts must record. The second half is the one that bites: an
    // out-of-range head into a pool whose record 0 does NOT terminate never comes back.
    {
        const auto words = pool({kTerm, kTerm});                      // 2 records, both terminate
        const std::vector<uint32_t> heads{99u};                       // far past the end
        const auto c = census_head_walk(words, heads, enc);
        CHECK(c.starts == 1 && c.terminating == 1 && c.cyclic == 0,
              "an out-of-range head reads zero, lands on record 0, and terminates there");
        CHECK(c.oob_starts == 1,
              "the walk is recorded as having left the buffer -- the discriminator between a "
              "cyclic guest list and a mis-resolved descriptor");

        const auto looping = pool({0u, kTerm});                       // record 0 points at itself
        const auto c2 = census_head_walk(looping, heads, enc);
        CHECK(c2.starts == 1 && c2.terminating == 0 && c2.cyclic == 1 && c2.oob_starts == 1,
              "an out-of-range head into a pool whose record 0 self-links never terminates");
    }

    // ---- a genuine cycle, constructed BY HAND ------------------------------------------------
    // The positive control for the cyclic verdict: a three-record ring plus a chain that leads into
    // it. Built here rather than drawn from any generator, so a census that can never report
    // `cyclic` at all is caught.
    {
        //  0 -> 1 -> 2 -> 0   (the ring)      3 -> 0   (leads in)      4 -> terminator
        const auto words = pool({1u, 2u, 0u, 0u, kTerm});
        const auto c = census_self_walk(words, enc);
        CHECK(c.records == 5 && c.starts == 5, "five records, five starts");
        CHECK(c.cyclic == 4 && c.terminating == 1,
              "the three ring members and the record leading into them are all cyclic");
        CHECK(c.cycle_nodes == 3,
              "only the ring members are ON the cycle -- the lead-in is not");
        CHECK(c.oob_starts == 0, "a ring inside the buffer never goes out of range");
        CHECK(c.consistent(), "terminating + cyclic == starts");
    }

    // ---- a self-loop is a different defect from a ring, and the samples must say so -----------
    {
        const auto words = pool({kTerm, 1u});      // record 1 points at itself
        const auto c = census_self_walk(words, enc);
        CHECK(c.cyclic == 1 && c.cycle_nodes == 1 && c.sample_count == 1 &&
                  c.sample_link[0] == 1u && c.sample_next[0] == 1u,
              "a self-loop is reported with its own index as both member and successor");
    }

    // ---- the histogram's counter-example ------------------------------------------------------
    {
        std::vector<uint32_t> words(8, 0u);
        words[5] = 0x1234u;
        words[6] = kTerm;
        const auto h = histogram_words(words, kTerm);
        CHECK(h.words == 8 && h.zero == 6 && h.terminator == 1 && h.other == 1,
              "the three histogram columns partition the buffer");
        CHECK(h.first_other_index == 5 && h.first_other_value == 0x1234u &&
                  h.min_other == 0x1234u && h.max_other == 0x1234u,
              "the first non-zero, non-terminator dword is reported so an all-zero verdict can be "
              "checked against a concrete counter-example");
    }

    // ---- the bounded single walk --------------------------------------------------------------
    {
        const auto words = pool({1u, 2u, kTerm});
        bool oob = false; uint32_t closing = 0;
        CHECK(walk_terminates(words, 0u, enc, 64, &closing, &oob) == LinkWalkVerdict::Terminates &&
                  !oob,
              "a bounded walk of a terminating chain terminates");
        const auto ring = pool({1u, 0u});
        CHECK(walk_terminates(ring, 0u, enc, 64, &closing, &oob) == LinkWalkVerdict::Cyclic,
              "a bounded walk of a ring exhausts its budget and reports cyclic");
    }

    // ---- settings parsing: a malformed setting disarms rather than changes the encoding --------
    {
        const auto d = parse_draw_link_scan_settings(nullptr, nullptr, nullptr, nullptr, nullptr,
                                                     nullptr);
        CHECK(d.valid && d.encoding.record_stride_dwords == 2 && d.encoding.next_dword_offset == 1 &&
                  d.encoding.terminator == 0xffffffffu && d.max_scans_per_buffer == 2 &&
                  d.heads_binding == UINT32_MAX && d.records_binding == UINT32_MAX,
              "the defaults are Astro Bot's shape, with the cross-buffer walk off");

        const auto ok = parse_draw_link_scan_settings("4", "3", "0", "1", "5", "9");
        CHECK(ok.valid && ok.encoding.record_stride_dwords == 4 &&
                  ok.encoding.next_dword_offset == 3 && ok.encoding.terminator == 0 &&
                  ok.max_scans_per_buffer == 1 && ok.heads_binding == 5 &&
                  ok.records_binding == 9,
              "every field is settable, terminator zero included");

        CHECK(!parse_draw_link_scan_settings("2", "2", nullptr, nullptr, nullptr, nullptr).valid,
              "a successor dword outside its own record is not an encoding");
        CHECK(!parse_draw_link_scan_settings("0", nullptr, nullptr, nullptr, nullptr, nullptr).valid,
              "a zero stride is rejected");
        CHECK(!parse_draw_link_scan_settings("2x", nullptr, nullptr, nullptr, nullptr, nullptr).valid,
              "trailing junk is rejected rather than silently truncated");
        CHECK(!parse_draw_link_scan_settings(nullptr, nullptr, nullptr, "0", nullptr, nullptr).valid,
              "a zero scan cap is rejected -- an instrument that scans nothing reports nothing");
        CHECK(!parse_draw_link_scan_settings(nullptr, nullptr, nullptr, nullptr, "5", nullptr).valid,
              "naming the head binding without the record binding is a half-armed cross walk");
        CHECK(!parse_draw_link_scan_settings(nullptr, nullptr, nullptr, nullptr, nullptr, "9").valid,
              "and so is naming the record binding without the head binding");
    }

    // ---- the selector: strict arming, and a per-BUFFER cap that reports its own exhaustion ----
    {
        using Configure = DrawLinkScanSelector::ConfigureResult;
        DrawLinkScanSelector s;
        CHECK(s.configure(nullptr) == Configure::Unset && !s.armed(),
              "an unset variable leaves the selector disarmed");
        CHECK(s.configure("21474836480") == Configure::Malformed && !s.armed(),
              "a bare decimal arms nothing -- the base-0 trap");
        CHECK(s.configure("0x5008f1400") == Configure::Armed && s.armed() && s.size() == 1,
              "a valid address arms the selector");
        CHECK(s.matches(0x5008efd00ull, 0, 0x5008f1400ull),
              "a draw whose PIXEL program is named matches");
        CHECK(!s.matches(0x5008efd00ull, 0, 0x5002af200ull),
              "a draw using neither named program does not match");

        bool exhausted = false;
        CHECK(s.should_scan(1, 1, 7, 2, &exhausted) == 1 && !exhausted,
              "the first scan of a binding is allowed and numbered");
        CHECK(s.should_scan(1, 1, 7, 2, &exhausted) == 2 && !exhausted,
              "the second is too");
        CHECK(s.should_scan(1, 1, 7, 2, &exhausted) == 0 && exhausted,
              "the third is refused and announces the cap ONCE");
        CHECK(s.should_scan(1, 1, 7, 2, &exhausted) == 0 && !exhausted,
              "later refusals are silent -- the cap must not masquerade as the rate");
        bool other = false;
        CHECK(s.should_scan(1, 1, 8, 2, &other) == 1,
              "the cap is per BINDING, so a different binding is scanned on its own budget");
        CHECK(s.should_scan(1, 0, 7, 2, &other) == 1,
              "and so is the same binding number in the other descriptor set");
        CHECK(s.should_scan(2, 1, 7, 2, &other) == 1,
              "and so is the same binding under a different program");
        CHECK(s.should_scan(1, 1, 7, 2, &other) == 0,
              "but a reallocated buffer does NOT buy a fresh budget -- an address-keyed cap bounds "
              "nothing on a title that reallocates every frame");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
