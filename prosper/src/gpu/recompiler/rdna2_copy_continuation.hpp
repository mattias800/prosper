#pragma once

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <memory>

namespace prosper::gpu {
class SavedMaskCopyFrame;

// Only the actual body selector can manufacture this scoped lineage. Original full-program
// initialization remains public; a caller-supplied slice cannot manufacture a continuation.
class SavedMaskCopyBodyScope {
    friend bool emit_body(SpirvCompute&, RegState&, const std::vector<Rdna2Inst>&,
                          const std::unordered_set<uint32_t>&, const ShaderResourceTable*, bool,
                          bool, const std::function<bool(RegState&, const Rdna2Inst&)>&,
                          const uint32_t*, size_t, const std::unordered_set<uint32_t>*, bool,
                          uint32_t, bool, bool);

    SpirvCompute& builder_;
    RegState& state_;
    const std::vector<Rdna2Inst>& instructions_;
    const uint32_t* code_;
    size_t dwords_;
    std::shared_ptr<SavedMaskCopyFrame> frame_, previous_;

    SavedMaskCopyBodyScope(SpirvCompute&, RegState&, const std::vector<Rdna2Inst>&, const uint32_t*,
                           size_t, uint32_t);
    ~SavedMaskCopyBodyScope();
    SavedMaskCopyBodyScope(const SavedMaskCopyBodyScope&) = delete;
    SavedMaskCopyBodyScope& operator=(const SavedMaskCopyBodyScope&) = delete;
    SavedMaskCopyBodyScope(SavedMaskCopyBodyScope&&) = delete;
    SavedMaskCopyBodyScope& operator=(SavedMaskCopyBodyScope&&) = delete;

    bool emit_counted_postlude(uint32_t exit_pc, const std::unordered_set<uint32_t>& safe,
                               const ShaderResourceTable*, bool allow_exec_update, bool allow_smem,
                               const std::function<bool(RegState&, const Rdna2Inst&)>&);
};

// A one-shot initializer permit, never a reusable mask or participation fact.
bool consume_saved_mask_copy_continuation(SpirvCompute&, const std::vector<Rdna2Inst>&,
                                          const uint32_t*, size_t, uint32_t, const RegState*);
void revoke_saved_mask_copy_proof(SpirvCompute&);
} // namespace prosper::gpu
