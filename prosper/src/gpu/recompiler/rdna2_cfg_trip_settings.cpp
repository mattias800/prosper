// CFG trip-bound selector parsing and operation-scope state.
#include "gpu/recompiler/rdna2_cfg_trip_settings.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <atomic>
#include <cstdint>
#include <cstdlib>

namespace prosper::gpu {

// Arbitrary CFG fallback. Some UE4 volume-lighting kernels and Astro Bot material shaders contain
// nested EXEC loops plus scalar/VCC branches. They are valid reducible machine CFGs, but deliberately
// exceed the narrow pattern structurizer below. Lower them as a structured dispatcher loop: one
// switch case per basic block, with the emulated register file persisted in Function variables
// between iterations. Graphics stages can branch directly on this invocation's SCC/VCC/EXEC bit;
// compute needs the guest-wave reduction described below.
// VCCZ/EXECZ reduce over the dispatch's 32/64-lane hardware wave, not the host Vulkan subgroup.
// Native subgroup widths are implementation-defined (llvmpipe is commonly 8), so every invocation
// remains in the dispatcher until the whole workgroup is done and exchanges its vote at the common
// switch merge. Dedicated Workgroup scratch keeps multiple hardware waves independent. The fallback
// is gated by emit_body to complex compute CFGs without guest barriers. Ordinary LDS reads, writes,
// and atomics remain valid while waves visit different cases. V_MBCNT is split into a dedicated
// common phase so every workgroup invocation reaches its synthesized barriers in uniform control flow.
// PROSPER_CFG_TRIP_BOUND=N — diagnostic only. Bound the CFG DISPATCHER's loop at N iterations, on
// whichever back-edge it emits (direct or portable). PROSPER_CFG_TRIP_BOUND_PROGRAM selects one
// program; PROSPER_CFG_TRIP_BOUND_PHASE selects one phase and is REQUIRED — see
// emitted_loop_trip_bound for why a record spanning two phases has no true reading. A program that
// accesses GDS itself is refused, because the witness lives in guest-addressable storage.
//
// It does NOT cover the two structured loop emitters. An earlier revision of this comment claimed it
// did; it never called this helper from either, so the claim was false and would have made a null
// result from a structured-loop program read as evidence.
//
// A loop that never terminates hangs the GPU into a driver reset, which costs the whole process its
// compute backend and every later indirect draw — and from outside it is indistinguishable from a
// slow shader or from a defect anywhere else in the submit. Bounding the dispatcher's loop turns
// "is this a non-terminating loop at all?" into one run for the programs it covers.
//
// (This paragraph used to end "and covering all three emitters means the answer is not confined to
// one lowering path" — the very claim the paragraph above corrects. The correction landed two lines
// up and the boast survived underneath it, which is how a retracted statement keeps being read as
// current. `tests/shared/diagnostics/test_cfg_trip_bound.cpp` now pins both halves: a structured loop is unchanged when
// armed, a dispatcher loop is not.)
//
// Unset, nothing is emitted and modules are byte-identical. It is NOT a fix: truncating a guest
// program's control flow produces wrong results by construction, which is why it is opt-in and says
// so when it arms.
// See kComputeTripWitnessDword. Deliberately ignores the phase selector: if ANY phase of this program
// will be bounded, the witness destination has to be bound for the whole dispatch.
// The complete selector state, read fresh rather than cached in function-local statics.
//
// Two reasons it is a struct and not three scattered getenv sites. First, the shader cache is keyed
// on the program's CODE BYTES and never on its address, so a target and a non-target with the same
// body collide on one entry: whichever compiled first would serve the other, handing the bounded
// module to an excluded program or the unbounded module to the target. Either direction silently
// destroys the isolation that makes a bounded run evidence about ONE program. The cache key therefore
// mixes this whole struct in (see gpu_executor.cpp), and a struct is what makes that complete by
// construction — a future selector added here cannot be forgotten there.
//
// Second, function-local statics parse the environment exactly once per process, which makes the
// selectors untestable in-process: no test could arm a bound, assert, then disarm and assert again.
// So this parses on demand, and `TripBoundOperation` (rdna2_to_spirv.hpp) is what keeps one
// operation's reads coherent without freezing them for the process.
//
// Work counter for the operation scope. See trip_bound_parses().
std::atomic<uint64_t>& trip_bound_parse_counter() {
    static std::atomic<uint64_t> parses{0};
    return parses;
}

// The innermost TripBoundOperation's owned settings, or null when none is open on this thread.
// Per thread because compilation runs on whatever worker realized the draw.
const ComputeTripBoundSettings*& trip_bound_pin() {
    static thread_local const ComputeTripBoundSettings* pinned = nullptr;
    return pinned;
}

// THE COMMENT HERE USED TO SAY the re-reads happen "during RECOMPILATION only (never per draw or
// per dispatch — cf. #2214)", with CONFIDENCE: HIGH. That was wrong, and wrong in the way this
// project's charter warns about most: a true-sounding claim wearing a citation to the very issue
// about removing per-draw getenv, in a place nobody re-derives. Three of the callers are not in the
// recompiler at all — gpu_executor.cpp builds the shader-cache KEY from these settings, so they are
// read on every cache HIT as well as every miss, i.e. once or twice per draw. Measured on a 240 s
// routed Grand Theft Auto V window with the switch unset (#3714): 3,989,848 reads of
// PROSPER_CFG_TRIP_BOUND at `ec02a77f6`, ~16,600/s. That is nominally the largest remaining getenv
// name, but it leads PROSPER_GFXLOG by 0.39% while GFXLOG itself varies by 35,143 between two runs
// of the same route -- so read the two as jointly largest, not ranked. An unset name is the
// expensive case on glibc -- a full scan of environ -- so those are all misses.
//
// The body. `parse_trip_bound_settings()` and `compute_trip_bound_settings()` are declared in the
// public header and defined ONCE in rdna2_to_spirv.cpp on top of this. The reason is
// [basic.def.odr]/11: an `inline` function must be defined in every translation unit that uses it,
// and this header is internal to src/gpu/recompiler/, so a TU outside it -- the new
// `test_trip_bound_operation` -- cannot legally see a definition at all.
//
// Do NOT restate that as "no out-of-line copy survives". It does: on `ec02a77f6` gpu_executor.cpp
// calls `compute_trip_bound_settings()` through the public declaration WITHOUT including this
// header, and links. What was actually observed is narrower: linking test_trip_bound_operation
// against libprosper_core.a failed with `undefined reference to
// 'prosper::gpu::compute_trip_bound_settings()'` -- the PRE-EXISTING function, not any of the
// symbols this change adds, which is the only reason the observation bears on the inline question
// at all. The mechanism for it was not established and is deliberately not guessed at here.
ComputeTripBoundSettings parse_trip_bound_settings_impl() {
    trip_bound_parse_counter().fetch_add(1, std::memory_order_relaxed);
    ComputeTripBoundSettings settings;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): existing operation-scoped selector read, relocated unchanged.
    const char* spec = getenv("PROSPER_CFG_TRIP_BOUND");
    if (!spec || !*spec) return settings;
    char* end = nullptr;
    const unsigned long long parsed = strtoull(spec, &end, 0);
    if (!end || *end || !parsed || parsed > 0xffffffffull) return settings;
    settings.bound = static_cast<uint32_t>(parsed);

    // PROSPER_CFG_TRIP_BOUND_PROGRAM=0xADDR — restrict the bound to ONE program, leaving every other
    // recompiled module byte-identical. Without it, a bound low enough to be interesting truncates
    // MANY shaders, and any later change of behaviour has a second explanation: an earlier truncated
    // shader fed different data downstream. Targeting one program removes that alternative, which is
    // what turns "the run got further" into evidence about the program under test.
    // NOLINTNEXTLINE(concurrency-mt-unsafe): existing operation-scoped selector read, relocated unchanged.
    if (const char* only = getenv("PROSPER_CFG_TRIP_BOUND_PROGRAM"); only && *only) {
        char* only_end = nullptr;
        const unsigned long long wanted = strtoull(only, &only_end, 0);
        if (only_end && !*only_end) settings.only_program = wanted;
    }
    // PROSPER_CFG_TRIP_BOUND_PHASE=K — bound only the K-th dispatcher of the selected program.
    //
    // A barrier-phased compute program emits one dispatcher per barrier-free phase, each covering a
    // different guest pc range. Bounding them together answers "does some loop in this program run
    // away" but not which, and they are not equivalent: an acyclic phase that needs a bound is a CFG
    // state-transition defect, while the phase containing the guest's own loop might merely be slow.
    // NOLINTNEXTLINE(concurrency-mt-unsafe): existing operation-scoped selector read, relocated unchanged.
    if (const char* phase = getenv("PROSPER_CFG_TRIP_BOUND_PHASE"); phase && *phase) {
        char* phase_end = nullptr;
        const unsigned long long parsed_phase = strtoull(phase, &phase_end, 0);
        if (phase_end && !*phase_end && parsed_phase <= 0xfffffffeull)
            settings.only_phase = static_cast<uint32_t>(parsed_phase);
    }
    // PROSPER_CFG_TRIP_BOUND_ORDINAL=K — count only traversals about to dispatch ordinal K.
    // See ComputeTripBoundSettings::only_ordinal for why a per-loop cap is the thing that localizes
    // a runaway inside a program whose dispatcher serves several guest loops (#3193).
    // NOLINTNEXTLINE(concurrency-mt-unsafe): existing operation-scoped selector read, relocated unchanged.
    if (const char* ordinal = getenv("PROSPER_CFG_TRIP_BOUND_ORDINAL"); ordinal && *ordinal) {
        char* ordinal_end = nullptr;
        const unsigned long long parsed_ordinal = strtoull(ordinal, &ordinal_end, 0);
        if (ordinal_end && !*ordinal_end && parsed_ordinal <= 0xfffffffeull)
            settings.only_ordinal = static_cast<uint32_t>(parsed_ordinal);
    }
    return settings;
}

}   // namespace prosper::gpu
