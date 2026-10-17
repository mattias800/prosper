// AMD 70648 memory-format widths: 10_11_11 has low-to-high R11/G11/B10; 2_10_10_10
// has R10/G10/B10/A2. All operations below consume values, never memory/proof authority.
#include "gpu/recompiler/ir/ops/buffer_format.hpp"

#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace prosper::gpu::ir {

Value buffer_field(Builder& ir, Value word, const BufferFieldShape& s) {
    switch (s.kind) {
        case BufferFieldKind::Unsigned: return ir.bfe_u(word, ir.c32(s.offset), ir.c32(s.width));
        case BufferFieldKind::Signed: return ir.bfe_s(word, ir.c32(s.offset), ir.c32(s.width));
        case BufferFieldKind::Half: return ir.unpack_half(word, s.offset ? 1u : 0u);
        case BufferFieldKind::Unorm: {
            const Value field = ir.bfe_u(word, ir.c32(s.offset), ir.c32(s.width));
            return ir.normalize_field(field, s.norm, false);
        }
        case BufferFieldKind::Snorm: {
            const Value field = ir.bfe_s(word, ir.c32(s.offset), ir.c32(s.width));
            return ir.normalize_field(field, s.norm, true);
        }
        case BufferFieldKind::Ufloat: {
            const Value raw = ir.bfe_u(word, ir.c32(s.offset), ir.c32(s.width));
            const Value half = ir.shl(raw, ir.c32(s.width == 11 ? 4u : 5u));
            return ir.unpack_half(half, 0);
        }
    }
    return {};
}

BufferShiftValues buffer_raw_shift(Builder& ir, Value address, Value word) {
    const Value byte = ir.and_(address, ir.c32(3));
    const Value shift = ir.shl(byte, ir.c32(3));
    return {shift, ir.lshr(word, shift)};
}

Value buffer_low_shift(Builder& ir, Value word, Value shift) {
    return ir.lshr(word, shift);
}

Value buffer_high_join(Builder& ir, Value low, Value high, Value shift) {
    // Inverse shift is masked before use; shift zero must discard the adjacent word entirely.
    const Value inverse = ir.and_(ir.isub(ir.c32(32), shift), ir.c32(31));
    const Value upper = ir.shl(high, inverse);
    return ir.or_(low, ir.select(ir.ieq(shift, ir.c32(0)), ir.c32(0), upper));
}

BufferLocationValues buffer_component_location(Builder& ir, Value address, uint32_t component,
                                               uint32_t bytes) {
    const Value byte = component ? ir.iadd(address, ir.c32(component * bytes)) : address;
    const Value index = ir.lshr(byte, ir.c32(2));
    const Value shift = ir.shl(ir.and_(byte, ir.c32(3)), ir.c32(3));
    return {index, shift};
}

Value buffer_index_offset(Builder& ir, Value index, uint32_t dwords) {
    return dwords ? ir.iadd(index, ir.c32(dwords)) : index;
}

namespace {
template <size_t Outputs, size_t Inputs, class Build>
std::array<uint32_t, Outputs> emit_words(SpirvCompute& b, const std::array<uint32_t, Inputs>& bound,
                                         const Build& build) {
    Block block;
    Builder ir(block);
    std::array<Value, Inputs> inputs{};
    for (Value& value : inputs) value = ir.input(Type::U32);
    const std::array<Value, Outputs> outputs = build(ir, inputs);
    const auto ids = emit_spirv(b, block, bound);
    std::array<uint32_t, Outputs> result{};
    for (size_t i = 0; i < Outputs; ++i) result[i] = ids[outputs[i].index];
    return result;
}
} // namespace

uint32_t emit_buffer_field(SpirvCompute& b, uint32_t word, const BufferFieldShape& shape) {
    return emit_words<1>(b, std::array{word}, [&](Builder& ir, const auto& x) {
        return std::array{buffer_field(ir, x[0], shape)};
    })[0];
}
std::array<uint32_t, 2> emit_buffer_raw_shift(SpirvCompute& b, uint32_t address, uint32_t word) {
    return emit_words<2>(b, std::array{address, word}, [](Builder& ir, const auto& x) {
        const auto r = buffer_raw_shift(ir, x[0], x[1]);
        return std::array{r.shift, r.joined};
    });
}
uint32_t emit_buffer_low_shift(SpirvCompute& b, uint32_t word, uint32_t shift) {
    return emit_words<1>(b, std::array{word, shift}, [](Builder& ir, const auto& x) {
        return std::array{buffer_low_shift(ir, x[0], x[1])};
    })[0];
}
uint32_t emit_buffer_high_join(SpirvCompute& b, uint32_t low, uint32_t high, uint32_t shift) {
    return emit_words<1>(b, std::array{low, high, shift}, [](Builder& ir, const auto& x) {
        return std::array{buffer_high_join(ir, x[0], x[1], x[2])};
    })[0];
}
std::array<uint32_t, 2> emit_buffer_component_location(SpirvCompute& b, uint32_t address,
                                                       uint32_t component, uint32_t bytes) {
    return emit_words<2>(b, std::array{address}, [&](Builder& ir, const auto& x) {
        const auto r = buffer_component_location(ir, x[0], component, bytes);
        return std::array{r.index, r.shift};
    });
}
uint32_t emit_buffer_index_offset(SpirvCompute& b, uint32_t index, uint32_t dwords) {
    return emit_words<1>(b, std::array{index}, [&](Builder& ir, const auto& x) {
        return std::array{buffer_index_offset(ir, x[0], dwords)};
    })[0];
}
} // namespace prosper::gpu::ir
