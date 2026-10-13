#pragma once

#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include <array>
#include <set>

namespace prosper::gpu {
struct SpilledMaskHalves;
void seed_native_logical_entry_marks(SpilledMaskHalves& marks, const RegState& initial);
void seed_native_logical_entry_domains(SpilledMaskHalves& marks, const RegState& initial,
                                       const std::set<int>& direct_descriptor_sregs,
                                       std::set<int>& masks, std::set<int>& scalar_words);
// Native scalar logical DATA is independent of the lanes participating at this operation.
bool native_logical_data_word(const RegState& rs, int word, uint32_t& value);
bool is_native_b64_logical(const SpirvCompute& b, const Rdna2Inst& in);
void initialize_native_logical_data_demand(SpirvCompute& b, const std::vector<Rdna2Inst>& original,
                                           const RegState* initial);
// Narrow an existing original fragment proof; never establish selected-entry authority.
void retain_fragment_copy_demand_intersection(SpirvCompute&, const std::vector<Rdna2Inst>& selected,
                                              const RegState* original_entry);
// Pure MUST provenance; it grants no emitter permit or physical-word authority.
std::set<uint32_t> fragment_survivor_mask_producers(const SpirvCompute&,
                                                    const std::vector<Rdna2Inst>&,
                                                    const RegState* entry);
void initialize_native_logical_uniform_entry_proof(SpirvCompute& b,
                                                   const std::vector<Rdna2Inst>& original,
                                                   const RegState* initial);
bool native_logical_producer_has_full_participation(const SpirvCompute& b, const Rdna2Inst& in);
bool native_logical_data_demanded(const SpirvCompute& b, const Rdna2Inst& in);
bool native_vcc_copy_entry_masks_match(const SpirvCompute& b, const RegState* initial);
bool native_vcc_copy_execution_domain_matches(const SpirvCompute& b);
bool native_vcc_copy_is_bool_only(const SpirvCompute& b, const Rdna2Inst& in);
bool fragment_saved_mask_is_bool_only(const SpirvCompute& b, const Rdna2Inst& in, int root);
std::array<uint32_t, 2> native_logical_result_words(SpirvCompute& b, const RegState& rs,
                                                    const Rdna2Inst& in, uint32_t result_mask);
void record_native_logical_word_gaps(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in);
} // namespace prosper::gpu
