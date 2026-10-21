// Buffer access routes relocated from emit_alu; descriptor/proof resolution stays at the call site.
#pragma once
#include "gpu/recompiler/ir/lower/lowering_context.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
namespace prosper::gpu::ir {
// Borrow only the disposition string. The original RAII logger remains the sole diagnostic owner.
struct BufferAccessDisposition {
    const char*& how;
};
struct BufferAccessInputs {
    DataFormat fmt;
    uint32_t n;
    uint32_t comp_bytes;
    uint32_t stride;
    uint32_t offset;
    uint32_t binding;
    uint32_t atomic_x2_record_count;
    uint32_t atomic_op;
    uint32_t fmt_ncomp;
    uint32_t raw_bits;
    StorageBufferTailSemantic one_record_tail_semantic;
    bool packed;
    bool packed_word;
    bool is_half;
    bool is_uint;
    bool is_sint;
    bool raw_subword;
    bool folded_vfetch;
    bool idxen;
    bool offen;
    bool is_store;
    bool int_subword;
    bool dyn_vfetch;
    bool instance_vfetch;
    bool indirect_pointer_descriptor_source;
    bool coherent_load;
    bool gta5_selected_sbuffer_consumer;
    bool is_atomic_x2;
    bool is_atomic;
    bool is_atomic_fminmax;
    bool atomic_fmin;
    bool coherent_store;
    bool is_format;
    bool packed_10_11_11;
    bool is_snorm;
    bool dst_sel_routed;
    bool one_record_16bit_tail;
    bool raw_signed;
    const uint32_t (&dst_sel)[4];
    BufferAccessDisposition buf_op;
};
struct BufferAccessRoutes : LoweringContext, BufferAccessInputs {
    bool lower();
    float normalization_range() const;
    bool finish();
};
}   // namespace prosper::gpu::ir
