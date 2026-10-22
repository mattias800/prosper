#include "gpu/recompiler/rdna2_cfg_divergent_detector.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
namespace {
// PROSPER_DBG: name WHICH rejection discards the loops. Every `return {}` here throws away
// EVERY loop in the shader, not just the one that failed. What the caller then does with an
// empty list is NOT fixed -- see the note on `caller_role` below -- but one of the outcomes is
// the whole-stream CFG dispatcher, which EMULATES the loop instead of emitting a real SPIR-V
// one. So one unhandled shape silently changes how an entire program executes. Static reading can
// narrow a symptom to this function in minutes and then stall, because the sites are
// indistinguishable from outside; this makes them distinguishable.
//
// Do NOT read an absent line as "this function was never called for that program". Absence has
// three other causes: the `out.empty()` early return (no eligible backward branch, so no site is
// reached), a caller that supplies no diagnostic context (suppressed below), and PROSPER_DBG
// itself, which desyncs a timing-dependent pad script badly enough that a routed run may never
// reach the dispatch you wanted to observe (see the note at the top of this file). Whether a
// given caller consults this function at all is a question for the CALL SITE, not for the log.
// Reported ONCE per (program, cause, pc). A program is recompiled or re-looked-up many times
// over a run, so an unfiltered print repeats the same rejection -- measured on a GTA V gameplay
// run, the busiest single site emitted 70 lines. A log whose line count far exceeds its finding
// count invites exactly one mistake: quoting the line count as a finding count. De-duplicating
// here means the raw count is the honest statistic.
//
// Reported ONLY when the caller supplied a real program address. Both graphics stages have no
// diagnostic parameter at all and so stay silent: they would report `program=0x0`, which is
// wrong twice over -- the address is not a program, and the de-dup key would merge genuinely
// different shaders rejecting at the same cause and pc into one line, making a count of
// distinct programs read LOW. An unattributed line is not a weaker finding, it is a wrong one.
//
// The message states ONLY what is certain at this point, and deliberately says nothing about
// what happens next. It is tempting to add "and the stream then falls back to the CFG
// dispatcher" -- an earlier version did, and so did the first attempt at THIS comment, which
// asserted the opposite outcome just as confidently and was equally wrong.
//
// The honest statement is that the outcome is not determined here, and this comment will not
// enumerate the ways it can go -- deliberately, because two drafts already got an enumeration
// wrong in opposite directions.
//
// What is checkable and sufficient: an empty list is ONE input among several to a decision taken
// in `emit_body`, whose branches carry predicates over values not visible here (whether the
// dispatcher is permitted at all for this sub-stream, whether a guest barrier made it unsafe,
// the branch count, the presence of particular ops). At least one caller withholds the
// dispatcher outright, so "refused loop" and "loop emulated" are not the same claim. That is as
// far as this function can see, and it is far enough to know the consequence is not its to
// state. Plumbing those values in purely to keep a sentence would be the wrong trade: the
// sentence is what should go.
//
// (Found reviewing #2695, which fixed the same over-assertion one level up. Counting both
// earlier drafts of this comment -- one asserting the dispatcher, one asserting outright
// rejection, one asserting a false universal over the entries -- the shared shape is that the
// clause gets written where the CAUSE is detected rather than where the EFFECT is decided, and
// those are different functions. Writing a *narrower* claim is not a fix if it is still a claim
// about the other function.)
//
// `caller_role` is in the message AND in the de-dup key, and both halves are load-bearing.
// Two callers reach this function with the same `safe` set and the same program, and their
// consequences are NOT the same:
//
//   emit            -- the recompile. Every loop in the shader is discarded; what the stream
//                      does next is decided by the caller, not here (see the note above).
//   multiwave-probe -- `compute_shader_prefers_native_multiwave`, which only tests
//                      `loops.empty()`. Nothing is discarded and no fallback happens; the
//                      preference analysis simply proceeds as though the shader were loop-free.
//
// Without the role in the KEY the probe silences the recompile, because the probe runs first
// (`gpu_executor.cpp` calls it before the recompile). Without the role in the MESSAGE the
// surviving line states the recompile's consequence over the probe's cause -- which reads as
// proof that a recompile discarded the loops when no recompile has happened yet. That is
// precisely the misreading this diagnostic exists to prevent, so it must not manufacture it.
std::vector<DivLoop> report_div_loop_rejection(const RecompileDiagnosticContext& diagnostic,
                                               const char*& caller_role, const char* why,
                                               uint32_t pc) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): existing diagnostic read, relocated unchanged.
    if (getenv("PROSPER_DBG") && diagnostic.program_address != 0) {
        static std::mutex seen_mu;
        // Keyed by string VALUE, not by pointer: every caller passes a literal today, so
        // pointer identity would happen to work, and would break silently the first time
        // someone passes a computed name.
        static std::set<std::tuple<uint64_t, std::string, uint32_t, std::string>> seen;
        std::lock_guard<std::mutex> lk(seen_mu);
        // The key space is (programs x causes x pcs x roles) and a long session keeps loading
        // new shaders, so leave nothing unbounded even on a diagnostic path. Clearing rather
        // than refusing to insert means the reporter degrades into REPEATING lines, never into
        // dropping them -- for a diagnostic, losing tidiness beats losing findings. Measured at
        // 14 tuples on a full routed GTA V run, so the cap is not expected to be reached.
        if (seen.size() >= 4096u) seen.clear();
        if (seen.emplace(diagnostic.program_address, why, pc, caller_role).second)
            std::fprintf(stderr, "[divloop-reject] program=0x%llx role=%s %s at pc=%u (%s)\n",
                         (unsigned long long)diagnostic.program_address, caller_role, why, pc,
                         std::strcmp(caller_role, "emit") == 0
                             ? "all loops discarded; what the stream does next is the caller's"
                             : "probe only: nothing discarded, analysis proceeds as loop-free");
    }
    return {};
}
}   // namespace

std::vector<DivLoop> detect_divergent_loops(const std::vector<Rdna2Inst>& ins,
                                            const std::unordered_set<uint32_t>& safe,
                                            bool exact_fragment_wave_breaks,
                                            const RecompileDiagnosticContext& diagnostic,
                                            const char* caller_role) {
    std::vector<DivLoop> out;
    auto divloop_reject = [&](const char* why, uint32_t pc) -> std::vector<DivLoop> {
        return report_div_loop_rejection(diagnostic, caller_role, why, pc);
    };
    uint32_t end_pc = UINT32_MAX;
    for (const auto& in : ins)
        if (in.is_end) {
            end_pc = in.pc;
            break;
        }
    // Pass 1: each backward s_branch / s_cbranch_execnz / s_cbranch_scc0/scc1 is a candidate
    // back-edge. A backward SCC branch closes a bottom-tested do-while unless it is a waterfall's
    // once-through branch, which the linearizer consumes (waterfall_branches).
    const std::unordered_set<uint32_t> waterfalls = waterfall_branches(ins);
    std::vector<uint32_t> backedge_op;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (!div_loop_backedge_candidate(in, safe, waterfalls)) continue;
        DivLoop L;
        if (const DivLoopRefusal r = div_loop_from_backedge(ins, in, end_pc, &L); r.why)
            return divloop_reject(r.why, r.pc);
        out.push_back(L);
        backedge_op.push_back(in.opcode);
    }
    if (out.empty()) return out;
    sort_div_loops_by_header(out, backedge_op);
    if (const DivLoopRefusal r =
            validate_div_loops(ins, safe, exact_fragment_wave_breaks, out, backedge_op);
        r.why)
        return divloop_reject(r.why, r.pc);
    return out;
}
}   // namespace prosper::gpu
