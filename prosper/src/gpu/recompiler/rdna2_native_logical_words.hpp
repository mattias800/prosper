#pragma once

#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include <array>

namespace prosper::gpu {
// Native scalar logical DATA is independent of the lanes participating at this operation.
bool is_native_b64_logical(const SpirvCompute& b, const Rdna2Inst& in);
void initialize_native_logical_data_demand(SpirvCompute& b, const std::vector<Rdna2Inst>& original);
void initialize_native_logical_uniform_entry_proof(SpirvCompute& b,
                                                   const std::vector<Rdna2Inst>& original);
bool native_logical_producer_has_full_participation(const SpirvCompute& b, const Rdna2Inst& in);
bool native_logical_data_demanded(const SpirvCompute& b, const Rdna2Inst& in);
std::array<uint32_t, 2> native_logical_result_words(SpirvCompute& b, const RegState& rs,
                                                    const Rdna2Inst& in, uint32_t result_mask);
void record_native_logical_word_gaps(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in);
} // namespace prosper::gpu
