// Scalar-register clobber checks shared by the indirect-pointer proofs (moved out of
// rdna2_indirect_pointer_analysis.cpp so the descriptor-range proof stops keeping its own copy).
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <vector>

namespace prosper::gpu {

// Upper bound on the SGPRs one instruction's decoded destination may write. Over-counting only
// refuses a proof; under-counting would let one hide a clobber.
uint32_t conservative_scalar_write_width(const Rdna2Inst& instruction);

// True when no instruction strictly between `begin_pc` and `end_pc` may write any of the `count`
// SGPRs starting at `first_sgpr`.
bool scalar_range_unchanged(const std::vector<Rdna2Inst>& instructions,
                            uint32_t begin_pc, uint32_t end_pc,
                            uint32_t first_sgpr, uint32_t count);

}   // namespace prosper::gpu
