// Bind the old store facts and VDATA reads to the shared IR; all access and proof stay in caller.
#include "gpu/recompiler/ir/lower/buffer_store_values.hpp"
#include "gpu/recompiler/ir/ops/buffer_store.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include <vector>
namespace prosper::gpu::ir {
namespace {
std::vector<uint32_t> emit(SpirvCompute& b, const BufferStoreBlock& block, const Rdna2Inst& in,
                           uint32_t index, const BufferStoreRead& vread) {
    return emit_buffer_store_values(b, block, [&](uint32_t what) {
        return what == 4 ? index : vread(in.dst.value + static_cast<int>(what));
    });
}
}   // namespace
BufferStoreWord lower_buffer_store_packed_dword(SpirvCompute& b, const Rdna2Inst& in, uint32_t idx,
                                                uint32_t d, uint32_t store_n, uint32_t comp_bytes,
                                                bool is_half, bool is_snorm, float norm,
                                                const BufferStoreRead& vread) {
    // norm is the caller's already-resolved standard field maximum; the width defines it exactly.
    (void)norm;
    const BufferStoreShape shape{is_half    ? BufferStoreKind::Half
                                 : is_snorm ? BufferStoreKind::Snorm
                                            : BufferStoreKind::Unorm,
                                 store_n, 4, comp_bytes};
    const auto ids = emit(b, build_buffer_store_dword(shape, d), in, idx, vread);
    return {ids[0], ids[1]};
}
uint32_t lower_buffer_store_packed_word(SpirvCompute& b, const Rdna2Inst& in, uint32_t store_n,
                                        uint32_t packed_word_ncomp, bool packed_10_11_11,
                                        bool is_uint, bool is_sint, bool is_snorm,
                                        const BufferStoreRead& vread) {
    const BufferStoreKind kind = packed_10_11_11        ? BufferStoreKind::Ufloat
                                 : (is_uint || is_sint) ? BufferStoreKind::Integer
                                 : is_snorm             ? BufferStoreKind::Snorm
                                                        : BufferStoreKind::Unorm;
    return emit(b, build_buffer_store_word({kind, store_n, packed_word_ncomp, 0}), in, 0, vread)[0];
}
uint32_t lower_buffer_store_index(SpirvCompute& b, uint32_t idx, uint32_t k) {
    return emit_buffer_store_values(b, build_buffer_store_index(k),
                                    [&](uint32_t) { return idx; })[0];
}
BufferStoreFields lower_buffer_store_fields(SpirvCompute& b, const Rdna2Inst& in, uint32_t addr,
                                            uint32_t k, uint32_t comp_bytes, uint32_t field_mask,
                                            const BufferStoreRead& vread) {
    const auto ids = emit(b, build_buffer_store_fields(k, comp_bytes, field_mask), in, addr, vread);
    return {ids[0], ids[1], ids[2]};
}
}   // namespace prosper::gpu::ir
