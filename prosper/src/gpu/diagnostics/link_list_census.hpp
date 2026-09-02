#pragma once
// `PROSPER_DRAW_LINKSCAN` — CPU-side census of the LINKED LISTS a graphics draw's scalar buffers
// contain, taken from the exact bytes prosper is about to hand the shader.
//
// This is the graphics counterpart of `PROSPER_COMPUTE_PARENTSCAN` (live_compute.cpp), and it
// exists for the same reason that one does. A shader that walks a guest linked list until it sees a
// terminator has exactly two ways to never stop: the list it was given is cyclic, or the list it was
// given is not the list the guest built. Both are properties of BYTES, not of control flow, so the
// cheapest correct instrument is a walk of those bytes on the CPU — no submit, no GPU state, nothing
// that can reorder the run it is measuring.
//
// The motivating case is Astro Bot's world-map pixel program `0x5008f1400` (#3214), whose per-tile
// light-list walk at guest pc 1911..2538 never reaches its `0xffffffff` terminator and takes the
// device down. Its hazard is general enough to deserve a general instrument:
//
//   **A bounded buffer load returns ZERO out of range, and zero is a valid link.** RDNA2's OOB
//   contract for a bounded V# is architectural zero, and `bounded_cbuf_load` reproduces it exactly.
//   So a walk whose terminator is anything other than zero converts EVERY out-of-range read into a
//   jump to record 0 — and if record 0's successor is also zero (an unpopulated pool is all zeros),
//   the walk is a one-node self-loop that no bound can end. There is no error, no reject, and no
//   diagnostic anywhere on that path. `walk_terminates()` models the zero substitution rather than
//   treating an out-of-range link as an exit, because the hardware does not treat it as an exit.
//
// The census answers three different questions and reports all three, because on a live title it is
// not known in advance which one is interesting:
//
//   * the WORD HISTOGRAM says whether the buffer was populated at all — an all-zero region and a
//     region full of terminators are both "no lights here", and they are produced by completely
//     different defects;
//   * the SELF-WALK census says whether the record pool is intrinsically cyclic, treating every
//     record as a start (GTA V's `0x413dc6700` was established this way, 22/22);
//   * the HEAD-WALK census says whether the heads the shader actually starts from terminate, which
//     is the question when the pool is fine and the index into it is not.
//
// The encoding is printed on every line, exactly as the compute census prints its shift/mask, so a
// wrong guess about a title's record layout is visible in the output rather than silently producing
// a confident wrong answer.

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace prosper::gpu {

// How a guest linked list is laid out inside one scalar buffer.
//
// The Astro Bot shape is `record_stride_dwords = 2, next_dword_offset = 1, terminator =
// 0xffffffff`: 8-byte records whose second dword is the successor link, walked until the link is
// all-ones. `s106 = s107 << 3` then `s_buffer_load_dword` is the stride; `s107 = (s107 << 1) + 1`
// then `s106 = s107 << 2` is the +1 dword.
struct LinkListEncoding {
    uint32_t record_stride_dwords = 2;
    uint32_t next_dword_offset = 1;
    uint32_t terminator = 0xffffffffu;

    bool valid() const {
        return record_stride_dwords != 0 && next_dword_offset < record_stride_dwords;
    }
};

// Raw dword histogram. Deliberately separate from the walk: "the buffer is all zeros" and "the walk
// does not terminate" are different findings, and the first explains the second rather than the
// other way round.
struct BufferWordHistogram {
    uint32_t words = 0;
    uint32_t zero = 0;
    uint32_t terminator = 0;
    uint32_t other = 0;
    uint32_t min_other = 0;
    uint32_t max_other = 0;
    // First dword that is neither zero nor the terminator, so an "all zeros" verdict can be checked
    // against a concrete counter-example rather than believed.
    uint32_t first_other_index = UINT32_MAX;
    uint32_t first_other_value = 0;
};

BufferWordHistogram histogram_words(std::span<const uint32_t> words, uint32_t terminator);

// One walk's verdict, modelling the hardware's out-of-range-reads-zero contract.
enum class LinkWalkVerdict : uint8_t {
    Terminates = 0,   // reached `terminator`
    Cyclic = 1,       // revisited a link, so it never will
};

struct LinkListCensus {
    uint32_t words = 0;         // dwords the scanned buffer holds
    uint32_t records = 0;       // records it can hold under this encoding
    uint32_t starts = 0;        // starts classified (== records for a self-walk)
    uint32_t terminating = 0;   // starts whose walk reaches the terminator
    uint32_t cyclic = 0;        // starts whose walk revisits a link
    uint32_t cycle_nodes = 0;   // distinct links that lie ON a cycle
    uint32_t longest = 0;       // longest terminating chain, as a DEPTH (visit-order independent)
    // How many starts left the buffer at least once and were answered with an architectural zero.
    // This is the count that separates "the guest's list is cyclic" from "prosper's bound is wrong":
    // a walk that never goes out of range cannot have been rescued by a correct bound.
    uint32_t oob_starts = 0;
    // Ring members, kept because a count says corruption happened and the shape says which one: a
    // self-loop, a two-cycle and a long ring are three different defects.
    uint32_t sample_count = 0;
    uint32_t sample_start[6]{};
    uint32_t sample_link[6]{};   // the link at which the walk closed its ring
    uint32_t sample_next[6]{};

    // `terminating + cyclic == starts` always. An earlier revision of the compute census broke out
    // of its walk before classifying the start itself, and the two columns silently summed to
    // starts-1 on every array; assert the identity rather than re-learn it.
    bool consistent() const { return terminating + cyclic == starts; }
};

// Walk from `start` under `enc`, treating an index at or past the end of `words` as reading zero.
// `visited_budget` bounds the walk; it must exceed the record count for the verdict to be sound, and
// the census below sizes it from the buffer.
LinkWalkVerdict walk_terminates(std::span<const uint32_t> words, uint32_t start,
                                const LinkListEncoding& enc, uint64_t visited_budget,
                                uint32_t* closing_link, bool* went_out_of_range);

// Classify every record of `words` as a start. This is the self-walk: it asks whether the pool is
// intrinsically cyclic, independent of which heads a shader happens to use.
LinkListCensus census_self_walk(std::span<const uint32_t> words, const LinkListEncoding& enc);

// Classify the links in `heads` as starts into `words`. This is the head-walk: it asks whether the
// heads a shader actually starts from terminate. Heads equal to the terminator are counted as
// terminating (an empty list is a correct list), which is what makes the two columns comparable
// against the head histogram's `terminator` count.
LinkListCensus census_head_walk(std::span<const uint32_t> words, std::span<const uint32_t> heads,
                                const LinkListEncoding& enc);

// ---------------------------------------------------------------------------------------------
// The live selector: which draws to scan, and under which encoding.
//
// Split from the analysis above on purpose. The census is a pure function of bytes and can be
// pinned by a unit test with hand-built arrays; the selector is environment-driven policy, and the
// two failing for different reasons must be distinguishable.

struct DrawLinkScanSettings {
    LinkListEncoding encoding;
    // Scans per distinct (program, set, binding, address). A full self-walk of a multi-megabyte pool
    // is O(records) but not free, and the interesting content is the FIRST sighting -- a later one
    // has already been shaped by whatever the earlier draws did.
    uint32_t max_scans_per_buffer = 2;
    // Optional cross-buffer walk: start from every dword of the buffer at `heads_binding` and walk
    // records in the buffer at `records_binding`. Unset (UINT32_MAX) leaves the per-buffer histogram
    // and self-walk, which is what a first run wants because the bindings are not yet known.
    uint32_t heads_binding = UINT32_MAX;
    uint32_t records_binding = UINT32_MAX;
    // False when any supplied value was rejected. A malformed setting disarms the whole scan rather
    // than silently scanning under a DIFFERENT encoding than the reader asked for -- a census under
    // the wrong stride is not a weaker answer, it is a wrong one.
    bool valid = true;
};

// Raw environment strings in, settings out. Null or empty means "use the default". Every field is
// range-checked; `valid` is false if any is not.
DrawLinkScanSettings parse_draw_link_scan_settings(const char* stride, const char* next,
                                                   const char* terminator, const char* max_scans,
                                                   const char* heads, const char* records);

class DrawLinkScanSelector {
public:
    enum class ConfigureResult { Unset, Armed, Malformed };

    // Parse the program list (the raw environment value). Arms only on a completely valid list, via
    // the same strict hex parser the skip selectors use, so a bare decimal or trailing junk arms
    // nothing and says so.
    ConfigureResult configure(const char* spec);

    bool armed() const { return !addresses_.empty(); }
    std::size_t size() const { return addresses_.size(); }
    bool matches(uint64_t vs_addr, uint64_t vs_chain_addr, uint64_t fs_addr) const;

    // Rate limit, keyed by (program, set, binding) and deliberately NOT by the buffer's guest
    // address. A title redraws one program thousands of times a frame, and reallocates its scratch
    // buffers every frame, so an address-keyed cap bounds nothing: each frame's fresh allocation
    // buys a fresh budget and the instrument scans forever. The binding is what "which buffer of
    // this shader" actually means, and it is stable. The address is still printed on every line.
    //
    // Returns the 1-based scan ordinal, or 0 when the cap is reached. The FIRST refusal per binding
    // returns 0 with `*exhausted` set, so the log records that the INSTRUMENT stopped rather than
    // that the buffer stopped appearing.
    uint32_t should_scan(uint64_t program, uint32_t set, uint32_t binding, uint32_t max_scans,
                         bool* exhausted);

private:
    std::vector<uint64_t> addresses_;
    mutable std::mutex mutex_;
    std::map<std::pair<uint64_t, uint64_t>, uint32_t> counts_;
};

// Process-wide instances, configured once from the environment on first use:
//   PROSPER_DRAW_LINKSCAN=0xADDR[,0xADDR...]      the programs whose draws are scanned
//   PROSPER_DRAW_LINKSCAN_STRIDE=N                dwords per record          (default 2)
//   PROSPER_DRAW_LINKSCAN_NEXT=N                  successor dword in record  (default 1)
//   PROSPER_DRAW_LINKSCAN_TERM=0xN                terminator link            (default 0xffffffff)
//   PROSPER_DRAW_LINKSCAN_MAX=N                   scans per buffer           (default 2)
//   PROSPER_DRAW_LINKSCAN_HEADS=B                 optional head-table binding
//   PROSPER_DRAW_LINKSCAN_RECORDS=B               optional record-pool binding
// Both print their arming state to stderr exactly once, including the encoding.
DrawLinkScanSelector& draw_link_scan_selector();
const DrawLinkScanSettings& draw_link_scan_settings();
bool draw_link_scan_enabled();

}  // namespace prosper::gpu
