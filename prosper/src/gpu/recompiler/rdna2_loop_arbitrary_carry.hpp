// The loop-header half of the arbitrary property (#5185, the #5193 companion rule). A loop header
// assumes each carried word's back-edge value; LoopBlanketCheck (rdna2_to_spirv_internal.hpp)
// checks that assumption once the body exists. This file decides, per carried root, whether the
// back edge held it, held it only WEAKLY (it brought in arbitrary lanes where the header assumed
// defined ones), or violated it (it brought in a hard lane). Compiler facts only.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace prosper::gpu {

class LoopExecSubset;
struct DivLoop;
struct RegState;
struct ScalarSourceMarks;
struct SpirvCompute;

struct LoopRootVerdicts {
    // Roots whose back edge brought in a hard lane. A read that relied on one is refused.
    std::set<uint64_t> violated;
    // Roots a defined-preheader word assumed defined, whose back edge brought in only defined or
    // arbitrary lanes. Reads that needed no more than "no hard lane" stand; marks that carried the
    // assumption become arbitrary (weaken_vector_word_roots).
    std::set<uint64_t> weak;
};

// `carried`/`confined`/`arbitrary` are LoopBlanketCheck's, keyed by carried word; `weak_allowed`
// is loop_may_hold_weakly() for the loop. The scalar half of `carried` is never weak.
LoopRootVerdicts loop_root_verdicts(const SpirvCompute& b, const RegState& rs, uint32_t header_pc,
                                    const std::map<int, std::set<uint64_t>>& carried,
                                    const std::map<int, uint32_t>& confined,
                                    const std::set<int>& arbitrary, bool weak_allowed);
// A violated arbitrary-outside root (kArbitraryOutsideRootOffset) costs a mark its arbitrary
// property, not its owner's proof: drop it before retire_vector_word_roots sees the rest.
void strip_arbitrary_outside_roots(ScalarSourceMarks& marks, const std::set<uint64_t>& violated);
// A vector mark that relied on a weakly held root: every lane it could miss is defined or
// arbitrary, so it becomes an unconfined arbitrary mark that keeps its roots as obligations.
void weaken_vector_word_roots(ScalarSourceMarks& marks, const std::set<uint64_t>& weak);
// A weak hold is sound only while no vector value of the body reaches a scalar register: scalar
// data carries no arbitrary property (#5185), so a root read through a scalar needs DEFINED.
// V_READFIRSTLANE and V_READLANE are the vector-to-scalar data paths; a compare's mask carries no
// mark at all (rdna2_to_spirv_internal.hpp, the scalar write transfer).
bool loop_may_hold_weakly(const std::vector<Rdna2Inst>& ins, uint32_t lo, uint32_t hi);
// The dispatcher-retry gate of a structured EXEC loop. A compute loop without the narrowing
// certificate leaves its structured invocations one by one, so its wave-scalar state diverges
// from the wave's, and a refusal at its exit condition [header, exit branch) is what routes it to
// the exact dispatcher (closed_compute_store_loop_data_retry). For such a top-tested loop this
// returns the VGPRs that condition reads, whose headers keep no arbitrary property, and marks the
// condition strict (SpirvCompute::strict_loop_condition), where a fresh predicate may not rest
// on an arbitrary lane. Otherwise it returns nothing. A weak hold acts at the back edge, after the
// condition, and is unaffected. A live-out wave scalar is routed by what the loop is
// (refuse_wave_scalar_loop, #5201), so this gate is no longer what routes it. The gate stays because
// it also refuses the arbitrary variant of a loop that widens EXEC in its body, which the
// structured route miscompiles; the defined-data variant is #5206. Delete the gate with that rule.
std::set<int> retry_gate(SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                         const LoopExecSubset& proof);

}   // namespace prosper::gpu
