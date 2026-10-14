#pragma once
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
struct RegState;
struct SpirvCompute;
// A new logical predicate requires an actual reaching complete producer and genuine word.
// Numeric physical-word authority and complete saved-mask aliases are separate contracts.
void initialize_partial_vcc_word_proof(SpirvCompute&, const std::vector<Rdna2Inst>&,
                                       const uint32_t*, size_t, bool partial_entry);
uint32_t partial_vcc_word_predicate(SpirvCompute&, const RegState&, const Rdna2Inst&, uint32_t);
// The ordinary MOV's numeric value and descriptor provenance retain their existing transfer.
void publish_s_mov_b32_word(RegState&, const Rdna2Inst&, uint32_t, uint32_t);
} // namespace prosper::gpu
