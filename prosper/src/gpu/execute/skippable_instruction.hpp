// skippable_instruction.hpp -- can a program finish without executing an instruction, by its
// scalar (wave-uniform) branch decisions alone?
//
// A descriptor slot that only such an instruction reads is one the guest may leave unwritten: the
// branch that skips it is decided by scalar state, typically a per-draw constant or a light count.
// Kena's lighting pixel program (PPSA01802, `ps b8e8f38ec3a5fc7f`) samples one of several T#s in a
// switch inside a loop over lights, plus one behind a feature branch; the slots of the arms a draw
// never takes hold stale descriptor-ring bytes that vary run to run, and refusing the program on them
// made admission non-deterministic on identical program bytes (#4796, #4775).
//
// WHY ONLY SCALAR BRANCHES (#4801 review). An alpha-test kill (`v_cmpx`, `s_cbranch_execz` to the
// end) or a divergent `if` also lets a wave skip a sample, but every visible pixel still executes it,
// so a bad descriptor there is not an unwritten slot. Kena measured the difference: with the broad
// "any path avoids it" predicate, default runs nulled descriptors in about twenty level-load programs
// whose tables had been recycled under prosper (the SDK<13 race, #1226/#2220); with the visibility
// contract forced on, only the lighting program's switch arms remained.
//
// The question is modelled as a two-player reachability game over direct branches. At s_cbranch_scc0
// and s_cbranch_scc1 (a scalar condition, uniform across the wave) a path may take either edge. At
// s_cbranch_vccz/vccnz/execz/execnz (a per-lane mask, or one the analysis cannot prove uniform) BOTH
// edges must avoid the instruction. The answer is true when some choice of scalar decisions reaches
// a program end without executing `use_pc`, whatever the lane masks do.
//
// What it does NOT prove: that THIS draw takes that path. A scalar condition can be constant for the
// draw (a uniform light count of zero skips the whole light loop, but a draw with lights runs it).
// The caller must only apply it to descriptor words that cannot be a T#. The full answer is a
// path-sensitive fold that evaluates the conditions for the draw; this is the structural interim.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace prosper::gpu {

// See the header comment. FAIL-CLOSED: false whenever the control-flow graph cannot be trusted --
// indirect control transfer (s_setpc/s_swappc/s_rfe/s_call), a branch target or fallthrough outside
// the decoded stream, an undecoded instruction, or a `use_pc` that names no instruction. A program end
// is s_endpgm, s_endpgm_saved or s_endpgm_ordered_ps_done.
bool program_may_skip_by_scalar_branch(const std::vector<Rdna2Inst>& instructions, uint32_t use_pc);

// The same question for a program the caller holds as words. `words` is the walked program (up to
// its first s_endpgm); `live` and `live_dwords` are the guest stream it came from, read only to
// append out-of-line tail blocks (a discard tail reached by a branch), without which such a branch
// names undecoded code and the answer is false. The decode cache's fold stream cannot be used: it
// keeps only the instructions the scalar fold reads, so its fallthrough edges are not the program's.
//
// Answers are cached per (decoded program, pc) in a bounded per-thread table keyed by `owner`, whose
// expiry invalidates its entries, so a draw that hits the same unusable slot every frame pays a
// lookup, not a re-walk (P5). Pass the decoded-program owner the words belong to.
class SkippableInstructionQuery {
public:
    SkippableInstructionQuery(std::shared_ptr<const void> owner, const std::vector<uint32_t>& words,
                              const uint32_t* live, size_t live_dwords)
        : owner_(std::move(owner)), words_(words), live_(live), live_dwords_(live_dwords) {}
    bool may_skip(uint32_t use_pc);

private:
    std::shared_ptr<const void> owner_;
    const std::vector<uint32_t>& words_;
    const uint32_t* live_;
    size_t live_dwords_;
    bool decoded_ = false;
    std::vector<Rdna2Inst> instructions_;
};

}   // namespace prosper::gpu
