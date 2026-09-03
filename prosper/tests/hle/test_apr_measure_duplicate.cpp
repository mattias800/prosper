// test_apr_measure_duplicate — sceAmprMeasureCommandSizeReadFile's compatibility read (#3245).
//
// That call's firmware contract is pure sizing (20, or 24 for a wide file offset). prosper also
// performs the described read, on the recorded grounds that an older SDK wrapper consumes the
// destination from the measure and never records a ReadFile for it. The read used to fire for EVERY
// title whose file id resolved, so a title that also submits the read normally paid for the same
// bytes twice — 13,925 duplicate reads and 582 MB in a 3-minute Stray boot — and got them written
// into its destination before it had recorded the command.
//
// The two patterns this has to keep apart cannot be told apart AT the measure call: both pass a
// valid id, a real destination and a real size. What separates them is what happens next, so that
// is what the suppression keys on, and that is what the cases below drive:
//
//   * consume-from-the-measure: measure -> guest reads dst -> no submit.  Must still deliver.
//   * submit-what-it-measures:  measure -> submit of the IDENTICAL read.  Redundant from then on.
//
// The detector arms below double as the integrity check for the O(1) index that note_submit uses:
// an index that silently finds nothing would disable detection without failing any return value,
// and the "stays quiet" arm is what catches exactly that.
//
// The arms are named after the PATTERNS, not after titles, and deliberately so. An earlier version
// called the first one "the DOLL arm" — but replaying the pairing predicate over four of DOLL's own
// PROSPER_FILELOG boots shows DOLL pairs at submit #13 and has ZERO measured reads lacking an
// identical submit, i.e. DOLL is the SECOND pattern. Both titles this work is about turn out to be
// the second pattern; the first is the shape the suppression must never break if a title has it,
// which is a claim about the mechanism and not about anybody's traffic.
//
// Every delivery assertion pre-poisons the destination with a byte the fixture cannot produce at
// that offset, so "the callee wrote nothing" and "the callee wrote the right thing" are distinct
// observations rather than both reading as zero.
//
// Registered THREE times in CMake — default, PROSPER_APR_MEASURE_READ=always, and =never. The
// always-run asserts the OPPOSITE outcome for the submit-what-it-measures case, which is what shows
// the lever moved: a suppression that never suppressed would pass the always-run and fail the
// default one. The never-run covers the third mode, which nothing else executes.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fixtures/test_scratch.h"
#include "host/image/exec_image.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

namespace prosper {
uint32_t prosper_apr_register(const std::string& path, uint64_t size);
void prosper_apr_reset_for_test();
bool prosper_apr_mixed_pattern_warned_for_test();
void prosper_apr_index_coverage_for_test(size_t* tomb_steps, size_t* rebuilds);
size_t prosper_apr_index_bucket_for_test(uint32_t id, uint64_t dst, uint64_t offset, uint64_t size);
}
using namespace prosper;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { std::printf("  [ok]   %s\n", msg); } } while (0)

// The plain ReadFile takes its file offset in a stack slot, so it is entered the way the guest
// enters it. sysv_abi is a no-op on Linux and forces the guest convention on MinGW.
using GuestReadFile = uint64_t(__attribute__((sysv_abi)) *)(uint64_t, uint64_t, uint64_t, uint64_t,
                                                            uint64_t, uint64_t, uint64_t, uint64_t,
                                                            uint64_t);

static std::array<uint8_t, 4096> g_fixture{};

// One destination per case, always poisoned before use.
struct Dest {
    std::array<uint8_t, 1024> bytes{};
    void poison() { bytes.fill(0x5A); }
    uint64_t addr() { return (uint64_t)(uintptr_t)bytes.data(); }
    bool holds(size_t offset, size_t size) const {
        return std::memcmp(bytes.data(), g_fixture.data() + offset, size) == 0;
    }
    bool untouched(size_t size) const {
        for (size_t i = 0; i < size; ++i) if (bytes[i] != 0x5A) return false;
        return true;
    }
};

int main() {
    const char* mode_env = std::getenv("PROSPER_APR_MEASURE_READ");
    const bool forced_always = mode_env && std::strcmp(mode_env, "always") == 0;
    const bool forced_never  = mode_env && std::strcmp(mode_env, "never") == 0;
    std::printf("== test_apr_measure_duplicate (PROSPER_APR_MEASURE_READ=%s) ==\n",
                mode_env ? mode_env : "<unset, auto>");
    register_builtin_hle();

    HleFn measure = Hle::lookup("vWU-odnS+fU");
    CHECK(measure != nullptr, "sceAmprMeasureCommandSizeReadFile is registered");
    if (!measure) { std::printf("== FAIL ==\n"); return 1; }

    // ---- Fixtures: two distinct containers, so the per-container scoping can be exercised -------
    const std::string path_a_storage =
        prosper_test::test_scratch_file("prosper-test-apr-measure-dup-a.tmp");
    const std::string path_b_storage =
        prosper_test::test_scratch_file("prosper-test-apr-measure-dup-b.tmp");
    for (size_t i = 0; i < g_fixture.size(); ++i) g_fixture[i] = (uint8_t)(i * 61u + 17u);
    bool written = true;
    for (const std::string& p : { path_a_storage, path_b_storage }) {
        FILE* f = std::fopen(p.c_str(), "wb");
        if (!f) { written = false; continue; }
        written = std::fwrite(g_fixture.data(), 1, g_fixture.size(), f) == g_fixture.size() &&
                  std::fclose(f) == 0 && written;
    }
    CHECK(written, "wrote both APR container fixtures");

    std::string stub_error;
    const std::vector<ImportSlot> slots = {{"libSceAmpr", "mQ16-QdKv7k"}};
    CHECK(install_stubs(slots, 0x720000000ull, 96, &stub_error) && stub_error.empty(),
          "generated the executable AMPR ReadFile import stub");
    auto read_file_guest = reinterpret_cast<GuestReadFile>(static_cast<uintptr_t>(stub_addr(0)));

    prosper_apr_reset_for_test();
    const uint32_t id_a = prosper_apr_register(path_a_storage, g_fixture.size());
    const uint32_t id_b = prosper_apr_register(path_b_storage, g_fixture.size());
    CHECK(id_a != 0 && id_b != 0 && id_a != id_b, "registered two distinct APR containers");

    std::array<uint8_t, 0x48> request{};
    const uint64_t cb = (uint64_t)(uintptr_t)request.data();
    std::array<uint64_t, 3> completion{};
    const uint64_t record = (uint64_t)(uintptr_t)completion.data();

    // ---- The sizing answer is the contract, and it never changes ------------------------------
    CHECK(measure(0, 0, 0, 0xffffffffull, 0, 0) == 20,
          "an unregistered sizing query stays pure and answers 20 bytes");
    CHECK(measure(0, 0, 0, 0xffffffffffull, 0, 0) == 24,
          "a wide file offset answers 24 bytes");

    // ---- Unsubmitted-measure arm: a measure with no following submit still delivers -------------
    // This is a MECHANISM arm and is deliberately not named after a title. It used to be called the
    // "DOLL arm", which claimed authority it had not earned: replaying the pairing predicate over
    // four of DOLL's own PROSPER_FILELOG boots shows DOLL pairs at submit #13 and is a
    // submits-everything-it-measures title (0 unmatched measures across all four runs), not the
    // consume-from-the-measure shape this arm models. The shape is still the one the suppression
    // must never break, whichever title turns out to have it.
    //
    // Driven twice. The second call is the one that matters: if the suppression latched on anything
    // other than an actual submit pairing, this is where it breaks.
    constexpr size_t doll_off = 64, doll_size = 96;
    Dest doll1, doll2;
    doll1.poison();
    const uint64_t first_measure = measure(id_a, doll1.addr(), doll_size, doll_off, 0, 0);
    if (forced_never) {
        CHECK(first_measure == 20 && doll1.untouched(doll_size),
              "PROSPER_APR_MEASURE_READ=never: the compatibility read never fires");
    } else {
        CHECK(first_measure == 20 && doll1.holds(doll_off, doll_size),
              "unsubmitted-measure arm: a measure with no submit delivers the container's bytes");
        doll2.poison();
        CHECK(measure(id_a, doll2.addr(), doll_size, doll_off + 128, 0, 0) == 20 &&
                  doll2.holds(doll_off + 128, doll_size),
              "unsubmitted-measure arm: a SECOND unsubmitted measure still delivers (no spurious latch)");
    }
    if (forced_never) {
        // The rest of the file drives the pairing latch, which `never` bypasses entirely. Its one
        // contract -- the read never happens, the sizing answer is still right -- is asserted above.
        CHECK(measure(0, 0, 0, 0xffffffffffull, 0, 0) == 24,
              "PROSPER_APR_MEASURE_READ=never: the sizing answer is still the contract");
        prosper_apr_reset_for_test();
        std::remove(path_a_storage.c_str());
        std::remove(path_b_storage.c_str());
        std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
        return fails ? 1 : 0;
    }

    // ---- A submit that does NOT match the measured read must not latch either -------------------
    // Same container, same destination, a different range. If the pairing were keyed loosely -- on
    // the file id alone, say -- this would retire the compatibility read and the next arm would
    // read as a pass for the wrong reason.
    {
        constexpr size_t m_off = 512, m_size = 64;
        Dest near_miss;
        near_miss.poison();
        CHECK(measure(id_a, near_miss.addr(), m_size, m_off, 0, 0) == 20 &&
                  near_miss.holds(m_off, m_size),
              "near-miss arm: the measure delivered");
        // Submit the same destination and size at a DIFFERENT offset.
        CHECK(read_file_guest(cb, 0, record, id_a, near_miss.addr(), m_size, m_off + 8, 0, 0) == 0,
              "near-miss arm: a submit of a different range succeeds");
        Dest after;
        after.poison();
        CHECK(measure(id_a, after.addr(), m_size, m_off + 256, 0, 0) == 20 &&
                  after.holds(m_off + 256, m_size),
              "near-miss arm: a non-identical submit does NOT retire the compatibility read");
    }

    // ---- Stray arm: measure, then the IDENTICAL submit, retires the compatibility read ----------
    constexpr size_t stray_off = 1024, stray_size = 200;
    {
        Dest first;
        first.poison();
        CHECK(measure(id_a, first.addr(), stray_size, stray_off, 0, 0) == 20 &&
                  first.holds(stray_off, stray_size),
              "Stray arm: the first measure delivers (the one duplicate this costs)");
        CHECK(read_file_guest(cb, 0, record, id_a, first.addr(), stray_size, stray_off, 0, 0) == 0,
              "Stray arm: the identical submit re-delivers the same bytes");
    }
    {
        // A NEW read on the same container, measured after the pairing was observed.
        Dest later;
        later.poison();
        const uint64_t measured = measure(id_a, later.addr(), stray_size, stray_off + 512, 0, 0);
        CHECK(measured == 20, "Stray arm: the sizing answer is unchanged by the suppression");
        if (forced_always) {
            CHECK(later.holds(stray_off + 512, stray_size),
                  "Stray arm under PROSPER_APR_MEASURE_READ=always: the read still fires");
        } else {
            CHECK(later.untouched(stray_size),
                  "Stray arm: the compatibility read is suppressed once the container has paired");
        }
        // Whatever the mode, the submit path is untouched and remains the real delivery.
        CHECK(read_file_guest(cb, 0, record, id_a, later.addr(), stray_size, stray_off + 512, 0,
                              0) == 0 && later.holds(stray_off + 512, stray_size),
              "Stray arm: the submit delivers the bytes the measure no longer does");
    }

    // ---- Scoping: the suppression is per container, not global ---------------------------------
    {
        Dest other;
        other.poison();
        CHECK(measure(id_b, other.addr(), doll_size, doll_off, 0, 0) == 20 &&
                  other.holds(doll_off, doll_size),
              "a DIFFERENT container keeps its compatibility read after the first one paired");
    }

    // ---- The detector: the one case this heuristic gets wrong must announce itself ---------------
    // The suppression's DECISION is loud, but its CONSEQUENCE would be silent -- a container that
    // mixes submitted and unsubmitted measured reads gets an unwritten destination and, without
    // this, no message. So a suppressed measure that leaves the ring having never been claimed by a
    // submit warns once, loudly. Both arms are needed: the designed case must stay quiet, or the
    // warning is noise that will be ignored the one time it matters.
    //
    // Ring is 64 entries, so more than that many suppressed records are needed to evict the first.
    constexpr size_t kRing = 64, kEvict = kRing + 8;
    {
        // Arm A -- the DESIGNED case: suppressed measures that ARE each submitted. Must not warn.
        //
        // This arm is ALSO the (id, dst, offset, size) index's integrity oracle, and that is not a
        // coincidence worth leaving implicit. The lookup is O(1) and open-addressed, so the way it
        // fails is by finding NOTHING for a key that is present -- a bad hash, a probe chain
        // terminated early by a mishandled tombstone, a stale idx_pos after a rebuild. None of those
        // throws, and none of them changes a return value that anything else checks: pairing has
        // already happened, so a missed lookup just silently stops marking entries. The detector is
        // what turns that into a failure. An entry the index cannot find is never marked, ages out
        // of the ring suppressed-and-unmatched, and warns -- so a silently-degraded index makes
        // THIS arm go red.
        //
        // 600 iterations rather than 72 so the run spans many ring wraps and, at a 256-slot table
        // rebuilt whenever it passes half full, several index rebuilds. A rebuild is where a stale
        // idx_pos would bite, and 72 iterations would not have reached one.
        constexpr size_t kIndexStress = 600;
        CHECK(!prosper_apr_mixed_pattern_warned_for_test(),
              "detector: quiet before any suppressed measure goes unclaimed");
        Dest d;
        for (size_t i = 0; i < kIndexStress; ++i) {
            const size_t off = 2048 + i * 4, size = 32;
            d.poison();
            measure(id_a, d.addr(), size, off, 0, 0);
            read_file_guest(cb, 0, record, id_a, d.addr(), size, off, 0, 0);
        }
        CHECK(!prosper_apr_mixed_pattern_warned_for_test(),
              "detector quiet + index found every key across many wraps and rebuilds");
        // ...and PROVE the arm above reached the mechanism it claims to guard, rather than
        // assuming it did. Both of the index's silent-failure modes need a specific condition to
        // be reachable at all: a lookup must step OVER a tombstone (the classic open-addressing
        // bug is to terminate the chain there instead), and the table must be REBUILT (which is
        // where a stale slot pointer would bite). Neither is guaranteed by iteration count -- at a
        // 256-slot table holding at most 64 live keys, most lookups hit their home slot and never
        // touch either path. Without these two checks the arm above would pass against an index
        // that mishandles both, which is precisely the "passes for the wrong reason" failure this
        // file exists to avoid.
        size_t tomb_steps = 0, rebuilds = 0;
        prosper_apr_index_coverage_for_test(&tomb_steps, &rebuilds);
        char cov[192];
        std::snprintf(cov, sizeof cov,
                      "index coverage: the stress arm survived %zu rebuild(s) (it stepped over %zu "
                      "tombstone(s) -- ordinary traffic does not reach that path)",
                      rebuilds, tomb_steps);
        CHECK(rebuilds > 0, cov);   // holds in every mode: recording drives rebuilds, lookups do not
    }
    {
        // Arm B -- the MIXED pattern: suppressed measures that are never submitted. Must warn.
        //
        // Under PROSPER_APR_MEASURE_READ=always the correct outcome INVERTS, and that is the point
        // rather than an exemption: nothing is suppressed in that mode, so no measure can be lost
        // and the detector must stay silent. A detector that fired here would be reporting a
        // data-loss event on a run where every byte was delivered -- the false positive that would
        // teach the next reader to ignore the warning.
        Dest d;
        for (size_t i = 0; i < kEvict; ++i) {
            d.poison();
            measure(id_a, d.addr(), 32, 2560 + i * 4, 0, 0);
        }
        if (forced_always) {
            CHECK(!prosper_apr_mixed_pattern_warned_for_test(),
                  "detector: cannot false-positive under =always, where nothing is suppressed");
        } else {
            CHECK(prosper_apr_mixed_pattern_warned_for_test(),
                  "detector: warns when a suppressed measure is never submitted (the mixed pattern)");
        }
    }

    // ---- The probe chain, built BY HAND ---------------------------------------------------------
    // The stress arm above reports 0 tombstone steps, and that zero is real rather than a
    // measurement artefact: 64 live keys in a 256-slot table means almost every lookup resolves at
    // its home bucket, so ordinary traffic never crosses a deleted entry. A clean zero on the very
    // path most likely to be wrong is exactly the situation CLAUDE.md says to distrust -- construct
    // one positive instance by hand, outside whatever produced the null, before believing it.
    //
    // So: force the one state that makes the tombstone path load-bearing -- a LIVE key whose probe
    // chain passes through a DELETED entry -- and prove the lookup still finds it.
    //
    //   A and B are chosen to share a home bucket, so B is placed one past A.
    //   A is submitted (marked), then aged out of the 64-entry ring, leaving a TOMBSTONE at the
    //   shared home bucket while B is still live behind it.
    //   Looking B up must now step OVER that tombstone. Terminating the chain there instead -- the
    //   classic open-addressing bug -- loses B silently: no error, no wrong return value, just an
    //   entry that never gets marked and later warns as a phantom mixed pattern.
    {
        prosper_apr_reset_for_test();
        const uint32_t id = prosper_apr_register(path_a_storage, g_fixture.size());
        Dest d;
        constexpr uint64_t kSize = 32;
        const uint64_t dst = d.addr();

        // Pair the container first, so every later measure records without doing any file I/O.
        d.poison();
        measure(id, dst, kSize, 0, 0, 0);
        read_file_guest(cb, 0, record, id, dst, kSize, 0, 0, 0);

        // Search the key space for a colliding pair and for fillers that avoid their bucket.
        const auto bucket = [&](uint64_t off) {
            return prosper_apr_index_bucket_for_test(id, dst, off, kSize);
        };
        uint64_t off_a = 0, off_b = 0;
        bool found = false;
        for (uint64_t i = 1; i < 4000 && !found; ++i)
            for (uint64_t j = i + 1; j < 4000 && !found; ++j)
                if (bucket(i * 4) == bucket(j * 4)) { off_a = i * 4; off_b = j * 4; found = true; }
        CHECK(found, "probe-collision arm: found two keys sharing a home bucket");
        if (found) {
            const size_t home = bucket(off_a);
            std::vector<uint64_t> fillers;
            for (uint64_t k = 1; fillers.size() < 63 && k < 200000; ++k) {
                const uint64_t off = 8000 + k * 4;
                if (off != off_a && off != off_b && bucket(off) != home) fillers.push_back(off);
            }
            CHECK(fillers.size() == 63, "probe-collision arm: found 63 non-colliding fillers");

            size_t before = 0, dummy = 0;
            prosper_apr_index_coverage_for_test(&before, &dummy);

            // A: recorded and submitted, so its eviction cannot itself trip the detector.
            measure(id, dst, kSize, off_a, 0, 0);
            read_file_guest(cb, 0, record, id, dst, kSize, off_a, 0, 0);
            // B: recorded only. It lands one past A, behind the bucket A will vacate.
            measure(id, dst, kSize, off_b, 0, 0);
            // 62 fillers fill the ring to 64; the 63rd evicts A and tombstones the shared bucket.
            for (size_t i = 0; i < 63; ++i) {
                measure(id, dst, kSize, fillers[i], 0, 0);
                read_file_guest(cb, 0, record, id, dst, kSize, fillers[i], 0, 0);
            }

            size_t after = 0;
            prosper_apr_index_coverage_for_test(&after, &dummy);
            // Now the load-bearing lookup: B is live, its chain starts at the tombstoned bucket.
            read_file_guest(cb, 0, record, id, dst, kSize, off_b, 0, 0);
            size_t after_lookup = 0;
            prosper_apr_index_coverage_for_test(&after_lookup, &dummy);

            char msg[256];
            if (forced_always) {
                // Under =always, apr_measure_note_submit returns at its mode check BEFORE taking
                // the lock, so no lookup happens at all and the counter cannot move. Asserting that
                // is worth more than skipping: it is the same early return that makes the =always
                // arm a valid control for this whole index -- if the index could cost anything in
                // that mode, an A/B between the two arms would not isolate it.
                std::snprintf(msg, sizeof msg,
                              "probe-collision arm under =always: the index is never consulted, so "
                              "it cannot cost anything in that mode (tomb steps %zu -> %zu -> %zu)",
                              before, after, after_lookup);
                CHECK(after_lookup == before, msg);
            } else {
                std::snprintf(msg, sizeof msg,
                              "probe-collision arm: the lookup for B stepped over a tombstone "
                              "(tomb steps %zu -> %zu -> %zu across the arm)",
                              before, after, after_lookup);
                CHECK(after_lookup > after, msg);
            }

            // And it must have FOUND B, not merely probed past the tombstone. Flush the ring: if B
            // was marked, nothing warns; if the chain terminated early, B ages out suppressed and
            // unmatched and the detector fires.
            CHECK(!prosper_apr_mixed_pattern_warned_for_test(),
                  "probe-collision arm: quiet before the flush");
            for (size_t i = 0; i < 80; ++i) {
                const uint64_t off = 300000 + i * 4;
                measure(id, dst, kSize, off, 0, 0);
                read_file_guest(cb, 0, record, id, dst, kSize, off, 0, 0);
            }
            CHECK(!prosper_apr_mixed_pattern_warned_for_test(),
                  forced_always
                      ? "probe-collision arm under =always: nothing is suppressed, so nothing warns"
                      : "probe-collision arm: B was FOUND behind the tombstone, not lost silently");
        }
    }

    // ---- The test hook really does clear the latch ----------------------------------------------
    // Without this, a later case in this process would inherit id_a's pairing, and the arms above
    // would be order-dependent in a way nothing would report.
    prosper_apr_reset_for_test();
    {
        const uint32_t id_again = prosper_apr_register(path_a_storage, g_fixture.size());
        Dest reset_dest;
        reset_dest.poison();
        CHECK(measure(id_again, reset_dest.addr(), doll_size, doll_off, 0, 0) == 20 &&
                  reset_dest.holds(doll_off, doll_size),
              "prosper_apr_reset_for_test clears the pairing latch as well as the registry");
        CHECK(!prosper_apr_mixed_pattern_warned_for_test(),
              "prosper_apr_reset_for_test clears the detector's once-only latch too");
    }

    prosper_apr_reset_for_test();
    std::remove(path_a_storage.c_str());
    std::remove(path_b_storage.c_str());
    std::printf("== %s ==\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
