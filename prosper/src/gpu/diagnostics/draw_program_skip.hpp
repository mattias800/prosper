#pragma once
// `PROSPER_SKIP_DRAW_PROGRAM` — decline GRAPHICS draws by shader PROGRAM identity, and the
// first-sighting census that supplies its input.
//
// This is the graphics counterpart of `PROSPER_COMPUTE_SKIP_PROGRAM` (live_compute.cpp). The
// motivating case is the same one that instrument made answerable on compute: a single draw can
// hang the GPU, RADV then cancels the whole context, and every later submit fails with
// `VK_ERROR_DEVICE_LOST` naming a *victim* rather than the cause. Astro Bot's world-map reset was
// misattributed to compute for exactly that reason; the RADV `hang` dump puts the last reached
// command-processor trace point immediately before a `DRAW_INDEX_2`. "Which draw is responsible?"
// therefore has to be askable directly.
//
// The pre-existing `PROSPER_SKIP_DRAW` cannot ask it. It selects by the submit-local semantic
// `draw_index`, which is an ordinal within one submit and is not stable from frame to frame, so it
// cannot name "every draw that runs this shader" — which is what a hanging *program* is.
//
// Contract, mirroring the compute selector deliberately:
//   * Default OFF. Unset, `evaluate()` returns "do not skip" after one `empty()` test and nothing
//     about the run changes.
//   * Impossible to arm by accident. The spec is parsed by the STRICT hex parser in
//     `watch_list.hpp`, so a bare decimal, a stray comma, trailing junk, an overflow or a zero
//     address arms NOTHING and says so. A selector whose null is meaningless is worse than none.
//   * It reports itself. The arming line names the variable and the count; every skip prints a
//     rate-limited `[draw-decline] ... reason=skipped-by-selector` line carrying its 1-based
//     ordinal (`diag_ratelimit.hpp`'s contract). A log from a run made with this set can never
//     later be read as a default run.
//
// FOUR LIMITS A READER OF THE RESULT CANNOT SEE IN THE OUTPUT. The compute selector documents two;
// the graphics analogue has more, because a draw's product is an image other draws then read.
//
//   1. **A skipped draw's TARGET still loses its contribution, and later passes sample the hole.**
//      The live renderer caches each pass's rendered pixels under `CB_COLOR0_BASE` and injects them
//      when a later draw samples a texture at that address. Decline the draw that fills a render
//      target and the composite that samples it does not fail — it succeeds against stale or empty
//      pixels. What you get is not "the frame minus that draw"; it is that frame minus the draw
//      minus every later draw's dependence on what it would have written.
//   2. **A pass whose every draw is declined renders NOTHING — not even its clear.**
//      `render_draw_pass_rgba` returns an empty image on an empty draw list, and the group's clear
//      colour is taken from the group's FIRST item regardless of whether that item was declined. So
//      naming a program that owns a whole pass removes the pass, and naming the first draw of a
//      pass does not change what the pass clears to.
//   3. **Only draws that reach the live renderer can be declined.** A draw whose shader the
//      recompiler rejected, or whose descriptor contract failed validation, never gets here, so
//      naming its program has no effect and produces no line. Silence is not proof the selector
//      matched — check the arming line and the census.
//   4. **An address names a PROGRAM, not a draw.** Every draw using that program is declined,
//      which on a real title is routinely thousands per frame across unrelated objects.
//
// And one that IS visible but is easy to misread: the decline is ordered LAST in the per-draw
// build, after resource resolution and after the `[render]`/`[rtt]` per-draw census lines
// (instrument trap 166 — a diagnostic skip ordered before the dump blinds every other instrument to
// the thing under suspicion). So a declined draw still costs its full CPU-side realization and is
// still fully observable; only the Vulkan draw call is withheld.

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

namespace prosper::gpu {

// Which of a draw's programs the selector matched. A guest address identifies one program, and a
// program is a vertex/ES program, its NGG main continuation, or a pixel program — never two of
// them — so the stage is reported rather than chosen by the caller.
enum class DrawProgramStage : uint8_t { None = 0, Vertex, VertexChain, Fragment };

// "vs", "vs-chain", "ps", or "none". Stable strings — they appear in logs that get grepped.
const char* draw_program_stage_name(DrawProgramStage stage);

struct DrawSkipDecision {
    bool skip = false;
    DrawProgramStage stage = DrawProgramStage::None;
    uint64_t address = 0;   // the matched address, not the draw's other programs
    // 1-based occurrence count for this (address, stage), carried on every printed line so the last
    // line's ordinal is a lower bound on the population. The line COUNT never is.
    uint64_t ordinal = 0;
    bool print = false;     // rate-limit verdict for this ordinal
};

class DrawProgramSkipSelector {
public:
    enum class ConfigureResult { Unset, Armed, Malformed };

    // Parse `spec` (the raw environment value; null or empty means "unset"). Arms only on a
    // completely valid list. On Malformed the selector is left disarmed — never partially armed.
    ConfigureResult configure(const char* spec);

    bool armed() const { return !addresses_.empty(); }
    std::size_t size() const { return addresses_.size(); }

    // The hot path. Cheap and lock-free when disarmed. When armed and matching, counts the skip and
    // returns the rate-limit verdict for the caller to print.
    DrawSkipDecision evaluate(uint64_t vs_addr, uint64_t vs_chain_addr, uint64_t fs_addr);

    uint64_t skipped_total() const;

private:
    std::vector<uint64_t> addresses_;
    mutable std::mutex mutex_;
    std::map<std::pair<uint64_t, uint8_t>, uint64_t> counts_;
    uint64_t total_ = 0;
};

// The census that supplies the selector's input: which graphics programs does this title actually
// draw with? Reports the FIRST sighting of each distinct (vs, vs-chain, ps) triple and then that
// triple's count at powers of two, so the number of lines is bounded by the number of distinct
// programs rather than by the number of draws. Deliberately not a teardown report: the run this
// exists for ends in a device loss, and a teardown report would not survive it.
class DrawProgramCensus {
public:
    struct Sighting {
        bool print = false;   // first sighting, or a power-of-two recurrence
        bool first = false;   // true only on the very first sighting of this triple
        uint64_t ordinal = 0; // 1-based count of draws with this triple
        uint64_t distinct = 0;// how many distinct triples have been seen, including this one
    };

    Sighting observe(uint64_t vs_addr, uint64_t vs_chain_addr, uint64_t fs_addr);
    uint64_t distinct_programs() const;

private:
    using Key = std::tuple<uint64_t, uint64_t, uint64_t>;
    mutable std::mutex mutex_;
    std::map<Key, uint64_t> counts_;
};

// Process-wide instances, configured once from the environment on first use:
//   PROSPER_SKIP_DRAW_PROGRAM=0xADDR[,0xADDR...]
//   PROSPER_DRAW_PROGRAM_CENSUS=1
// Both print their arming state to stderr exactly once.
DrawProgramSkipSelector& draw_program_skip_selector();
DrawProgramCensus& draw_program_census();
bool draw_program_census_enabled();

}  // namespace prosper::gpu
