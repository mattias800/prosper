// Buffer arithmetic and loaded fields: emitted SPIR-V versus eval and independent ISA truth.
// Distinct lanes and masked old destinations expose a discarded byte, sign or source component.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include <cstddef>
#include "gpu/recompiler/ir/buffer_format_cases.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ops/buffer_format.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "support/death_test.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace truth = prosper::test::ir::buffer_format;

namespace {
std::vector<uint32_t> execute(const Block& block, const std::vector<Value>& outputs,
                              const std::vector<std::vector<uint32_t>>& rows) {
    const auto n = static_cast<uint32_t>(rows.size());
    const uint32_t slots = block.input_count();
    SpirvCompute b;
    b.begin(slots, nullptr, n, 1, 1, n, 0, true, true);
    std::vector<uint32_t> inputs;
    inputs.reserve(slots);
    for (uint32_t slot = 0; slot < slots; ++slot) inputs.push_back(b.load_input(slot));
    const auto ids = emit_spirv(b, block, inputs);
    const auto count = static_cast<uint32_t>(outputs.size());
    for (uint32_t k = 0; k < count; ++k) b.store_output_word(ids[outputs[k].index], count, k);
    std::vector<uint32_t> buffer;
    for (const auto& row : rows) {
        EXPECT_EQ(row.size(), slots);
        buffer.insert(buffer.end(), row.begin(), row.end());
    }
    bpermute_oracle::Interpreter vm(b.finish());
    vm.half_rte_preserve_denorm = true; // selected interpreter profile; no device-mode claim
    auto out = vm.run_buffers(n, {{0, buffer}, {1, std::vector<uint32_t>(size_t{n} * count)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return out;
}

TEST(IrBufferFormatValues, EveryFieldMatchesIndependentTruthEvalAndSpirvPerLane) {
    uint32_t seed = 0x5384u;
    for (const auto& shape : truth::field_shapes()) {
        SCOPED_TRACE(::testing::Message() << "kind=" << int(shape.kind) << " offset="
                                          << shape.offset << " width=" << shape.width);
        Block block;
        Builder ir(block);
        const Value word = ir.input(Type::U32), active = ir.input(Type::U32);
        const Value old = ir.input(Type::U32);
        const Value field = buffer_field(ir, word, shape);
        const Value output = ir.select(ir.ine(active, ir.c32(0)), field, old);
        for (uint32_t wave : {32u, 64u}) {
            std::vector<std::vector<uint32_t>> rows;
            rows.reserve(64);
            for (uint32_t lane = 0; lane < wave; ++lane) {
                seed = seed * 1664525u + 1013904223u;
                uint32_t raw = lane < 4 ? std::array{0u, 1u, 0x80008000u, 0xffffffffu}[lane] : seed;
                // Keep both half components finite; either component may be unpacked by GLSL.
                if (shape.kind == BufferFieldKind::Half) raw &= 0xbfffbfffu;
                if (shape.kind == BufferFieldKind::Ufloat) raw &= 0x7fdffbffu;
                rows.push_back({raw, lane % 4 != 3, 0xd0000000u + lane});
            }
            const auto out = execute(block, {output}, rows);
            ASSERT_EQ(out.size(), wave);
            for (uint32_t lane = 0; lane < wave; ++lane) {
                const auto expected = truth::field_truth(rows[lane][0], shape);
                ASSERT_TRUE(expected);
                const std::array<Known, 3> inputs{rows[lane][0], rows[lane][1], rows[lane][2]};
                const auto values = evaluate(block, inputs);
                ASSERT_TRUE(values[field.index] && values[output.index]);
                EXPECT_EQ(*values[field.index], *expected);
                const uint32_t want = rows[lane][1] ? *expected : rows[lane][2];
                EXPECT_EQ(*values[output.index], want);
                EXPECT_EQ(out[lane], want) << "wave=" << wave << " lane=" << lane;
            }
        }
    }
}

TEST(IrBufferFormatValues, AddressesAndJoinsMatchEvalAndIndependentByteArithmetic) {
    for (uint32_t component = 0; component < 4; ++component)
        for (uint32_t bytes : {1u, 2u}) {
            Block block;
            Builder ir(block);
            const Value address = ir.input(Type::U32), word = ir.input(Type::U32);
            const Value high = ir.input(Type::U32), index = ir.input(Type::U32);
            const auto raw = buffer_raw_shift(ir, address, word);
            const auto location = buffer_component_location(ir, address, component, bytes);
            const std::vector<Value> outputs{raw.shift,
                                             raw.joined,
                                             buffer_high_join(ir, raw.joined, high, raw.shift),
                                             location.index,
                                             location.shift,
                                             buffer_index_offset(ir, index, component),
                                             buffer_low_shift(ir, high, raw.shift)};
            std::vector<std::vector<uint32_t>> rows;
            rows.reserve(64);
            for (uint32_t lane = 0; lane < 64; ++lane)
                rows.push_back({lane < 4 ? 0xfffffffcu + lane : lane * 33u + 3u,
                                0x8192a3b4u ^ lane * 0x10101u, 0xc5d6e7f8u ^ lane * 0x1020304u,
                                0xfffffffcu + lane});
            const auto out = execute(block, outputs, rows);
            ASSERT_EQ(out.size(), rows.size() * outputs.size());
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const auto& x = rows[lane];
                const uint32_t shift = (x[0] & 3) * 8, addr = x[0] + component * bytes;
                const std::array<uint32_t, 7> want{
                    shift,
                    x[1] >> shift,
                    static_cast<uint32_t>((uint64_t{x[2]} << 32 | x[1]) >> shift),
                    addr >> 2,
                    (addr & 3) * 8,
                    x[3] + component,
                    x[2] >> shift};
                const std::array<Known, 4> known{x[0], x[1], x[2], x[3]};
                const auto values = evaluate(block, known);
                for (size_t k = 0; k < outputs.size(); ++k) {
                    EXPECT_EQ(values[outputs[k].index], Known(want[k]));
                    EXPECT_EQ(out[size_t{lane} * outputs.size() + k], want[k]);
                }
            }
        }
}

TEST(IrBufferFormatValues, UnknownWordsNeverBecomeKnownComponents) {
    for (const auto& shape : truth::field_shapes()) {
        Block block;
        Builder ir(block);
        const Value value = buffer_field(ir, ir.input(Type::U32), shape);
        EXPECT_FALSE(evaluate(block, {})[value.index]);
    }
}

TEST(IrBufferFormatValues, NormalizationHasIndependentSignedAndUnsignedEndpoints) {
    const uint32_t signed_mode = 0x80000000u | std::bit_cast<uint32_t>(32767.0f);
    const uint32_t unsigned_mode = std::bit_cast<uint32_t>(65535.0f);
    EXPECT_EQ(evaluate_op(Opcode::NormalizeField, signed_mode, 0xffff8000u, 0, 0),
              Known(0xbf800000));
    EXPECT_EQ(evaluate_op(Opcode::NormalizeField, signed_mode, 0xffff8001u, 0, 0),
              Known(0xbf800000));
    EXPECT_EQ(evaluate_op(Opcode::NormalizeField, signed_mode, 32767, 0, 0), Known(0x3f800000));
    EXPECT_EQ(evaluate_op(Opcode::NormalizeField, unsigned_mode, 0, 0, 0), Known(0));
    EXPECT_EQ(evaluate_op(Opcode::NormalizeField, unsigned_mode, 65535, 0, 0), Known(0x3f800000));
    EXPECT_FALSE(evaluate_op(Opcode::NormalizeField, 0, 1, 0, 0));
    EXPECT_FALSE(evaluate_op(Opcode::NormalizeField, 0x7f800000, 1, 0, 0));
}

TEST(IrBufferFormatValues, MalformedDivisorsFailLoudly) {
    for (float divisor : {0.0f, -0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        EXPECT_DEATH_IF_SUPPORTED(
            {
                prosper::test::suppress_coredump();
                Block block;
                Builder ir(block);
                ir.normalize_field(ir.c32(1), divisor, false);
            },
            "normalized-field divisor must be finite and positive");
    }
}

TEST(IrBufferFormatValues, HalfAndMiniFloatNaNPayloadsStayUnknown) {
    for (const BufferFieldShape shape : {BufferFieldShape{BufferFieldKind::Half, 0, 16},
                                         BufferFieldShape{BufferFieldKind::Ufloat, 0, 11}}) {
        Block block;
        Builder ir(block);
        const Value value = buffer_field(ir, ir.input(Type::U32), shape);
        const std::array<Known, 1> words{shape.kind == BufferFieldKind::Half ? 0x7c01u : 0x7c1u};
        EXPECT_FALSE(evaluate(block, words)[value.index]);
    }
}
}   // namespace
