// Pure store values. The caller retains every access, predicate, coherence and proof.
#pragma once
#include <cstdint>
#include <functional>
namespace prosper::gpu {
struct SpirvCompute;
struct Rdna2Inst;
}   // namespace prosper::gpu
namespace prosper::gpu::ir {
using BufferStoreRead = std::function<uint32_t(int)>;
struct BufferStoreFields {
    uint32_t index, mask, value;
};
struct BufferStoreWord {
    uint32_t index, value;
};
BufferStoreFields lower_buffer_store_fields(SpirvCompute& b, const Rdna2Inst& in, uint32_t addr,
                                            uint32_t k, uint32_t comp_bytes, uint32_t field_mask,
                                            const BufferStoreRead& vread);
uint32_t lower_buffer_store_index(SpirvCompute& b, uint32_t idx, uint32_t k);
uint32_t lower_buffer_store_packed_word(SpirvCompute& b, const Rdna2Inst& in, uint32_t store_n,
                                        uint32_t packed_word_ncomp, bool packed_10_11_11,
                                        bool is_uint, bool is_sint, bool is_snorm,
                                        const BufferStoreRead& vread);
BufferStoreWord lower_buffer_store_packed_dword(SpirvCompute& b, const Rdna2Inst& in, uint32_t idx,
                                                uint32_t d, uint32_t store_n, uint32_t comp_bytes,
                                                bool is_half, bool is_snorm, float norm,
                                                const BufferStoreRead& vread);
}   // namespace prosper::gpu::ir
