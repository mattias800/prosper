#pragma once

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <utility>

namespace prosper::gpu {
struct SavedMaskDispatcherScope {
    SpirvCompute& builder;
    bool previous;
    explicit SavedMaskDispatcherScope(SpirvCompute& b)
        : builder(b), previous(std::exchange(b.saved_mask_in_dispatcher, true)) {}
    ~SavedMaskDispatcherScope() { builder.saved_mask_in_dispatcher = previous; }
    SavedMaskDispatcherScope(const SavedMaskDispatcherScope&) = delete;
    SavedMaskDispatcherScope& operator=(const SavedMaskDispatcherScope&) = delete;
};

// Native word retention is separate from the owned packet dispatcher's synchronized masks.
bool uses_native_saved_mask_words(const SpirvCompute& b);
void initialize_saved_mask_word_proof(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                      const uint32_t* code, size_t dwords, uint32_t initial_active);
bool saved_mask_producer_has_full_participation(const SpirvCompute& b, const Rdna2Inst& in);
bool saved_mask_spill_uses_data(const RegState& rs, int word);
bool validate_saved_mask_copy_words(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in);
bool saved_mask_write_retains_words(const SpirvCompute& b, const Rdna2Inst& in, int root);
bool saved_mask_mov_defines_words(const SpirvCompute& b, const Rdna2Inst& in, bool scalar_result,
                                  bool source_is_mask);
bool saved_mask_logical_defines_words(const SpirvCompute& b, const Rdna2Inst& in);
void load_missing_saved_mask_words(RegState& rs, const std::set<int>& roots,
                                   const std::set<int>& defined_words);
void retire_saved_mask_facts(const SpirvCompute& b, std::set<int>& masks, std::set<int>& ambiguous,
                             int base, uint32_t width);
void prepare_saved_mask_word_write(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in);
void retain_saved_mask_words(SpirvCompute& b, RegState& rs, const Rdna2Inst& in);
uint32_t refuse_unavailable_saved_mask_word(SpirvCompute& b, const Rdna2Inst& in, bool* ok);

} // namespace prosper::gpu
