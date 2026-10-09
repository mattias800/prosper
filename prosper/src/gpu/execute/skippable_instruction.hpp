// skippable_instruction.hpp -- can a program finish without executing one of its instructions?
//
// A descriptor slot that only a skippable instruction reads is one the guest is allowed to leave
// unwritten. Kena's lighting pixel program (PPSA01802, `ps b8e8f38ec3a5fc7f`) samples one of four
// T#s in a switch inside a loop over lights, plus one behind a feature branch; the table slots of the
// arms a draw never takes hold whatever stale bytes the ring allocation last held -- V#s, floats,
// half a previous T#. Whether those bytes happen to pass the T# screen changes from run to run, so
// refusing the program on them made admission non-deterministic on identical program bytes (#4775).
//
// The question this answers is purely structural: is there a path through the program's direct
// branches from its entry to a program end that does not pass through `use_pc`? It says nothing about
// whether a particular draw takes that path -- only that the program is built so some can.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <vector>

namespace prosper::gpu {

// True when a path from the first instruction (pc 0) to a program end avoids the instruction at
// `use_pc`, following direct SOPP branches (s_branch, s_cbranch_*) and fallthrough. A program end is
// s_endpgm, s_endpgm_saved or s_endpgm_ordered_ps_done.
//
// FAIL-CLOSED: false whenever the control-flow graph cannot be trusted -- indirect control transfer
// (s_setpc/s_swappc/s_rfe/s_call), a branch target or fallthrough that lands outside the decoded
// stream, an undecoded instruction, or a `use_pc` that names no instruction. A false answer keeps the
// caller on its fail-visible path. The converse limit: a branch whose condition is constant still
// counts as two edges, so a true answer means "skippable as encoded", not "skipped by some draw".
bool program_may_skip_instruction(const std::vector<Rdna2Inst>& instructions, uint32_t use_pc);

// The same question for a program the caller holds as words, decoded on first use and at most once:
// a descriptor-table build asks it only on the path that would otherwise refuse an image use, so the
// common case pays nothing. `words` is the walked program (up to its first s_endpgm); `live` and
// `live_dwords` are the guest stream it came from, read only to append out-of-line tail blocks (a
// discard tail reached by a branch), without which such a branch names undecoded code and the answer
// is false. The decode cache's fold stream cannot be used: it keeps only the instructions the
// scalar fold reads, so its fallthrough edges are not the program's.
class SkippableInstructionQuery {
public:
    SkippableInstructionQuery(const std::vector<uint32_t>& words, const uint32_t* live,
                              size_t live_dwords)
        : words_(words), live_(live), live_dwords_(live_dwords) {}
    bool may_skip(uint32_t use_pc);

private:
    const std::vector<uint32_t>& words_;
    const uint32_t* live_;
    size_t live_dwords_;
    bool decoded_ = false;
    std::vector<Rdna2Inst> instructions_;
};

}   // namespace prosper::gpu
