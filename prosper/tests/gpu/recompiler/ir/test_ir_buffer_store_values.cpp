// Pure store fields/indices execute per lane against eval and independently worked bytes.
// Masked lanes keep distinct old values; unknown/exceptional inputs never invent a known word.
#include "gpu/recompiler/ir/ops/buffer_store.hpp"
#include "gpu/recompiler/ir/buffer_store_cases.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "support/death_test.hpp"
#include "fixtures/buffer_store_ir.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <utility>
#include "gpu/recompiler/ir/ir.hpp"
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace fixture = prosper::test::ir::buffer_store;
namespace {
BufferStoreKind kind(fixture::Kind k) {
    switch (k) {
        case fixture::Kind::Integer: return BufferStoreKind::Integer;
        case fixture::Kind::Half: return BufferStoreKind::Half;
        case fixture::Kind::Unorm: return BufferStoreKind::Unorm;
        case fixture::Kind::Snorm: return BufferStoreKind::Snorm;
        case fixture::Kind::Ufloat: return BufferStoreKind::Ufloat;
    }
    std::abort();
}
using Row = std::array<uint32_t, 7>;
void check(BufferStoreBlock built, const std::vector<Row>& rows,
           const std::vector<std::vector<uint32_t>>& expected) {
    Builder ir(built.block);
    const auto append_input = [&](uint32_t what) {
        built.inputs.push_back(what);
        return ir.input(Type::U32);
    };
    const Value active = append_input(5), old = append_input(6);
    for (Value& output : built.outputs) output = ir.select(ir.ine(active, ir.c32(0)), output, old);
    const auto count = static_cast<uint32_t>(built.outputs.size());
    const auto lanes = static_cast<uint32_t>(rows.size());
    SpirvCompute b;
    b.begin(built.block.input_count(), nullptr, lanes, 1, 1, lanes, 0, true, true);
    std::vector<uint32_t> inputs;
    inputs.reserve(built.block.input_count());
    for (uint32_t i = 0; i < built.block.input_count(); ++i) inputs.push_back(b.load_input(i));
    const auto ids = emit_spirv(b, built.block, inputs);
    for (uint32_t i = 0; i < count; ++i) b.store_output_word(ids[built.outputs[i].index], count, i);
    std::vector<uint32_t> data;
    data.reserve(rows.size() * built.inputs.size());
    for (const auto& row : rows)
        for (uint32_t slot : built.inputs) data.push_back(row[slot]);
    bpermute_oracle::Interpreter vm(b.finish());
    vm.half_rte_preserve_denorm = true;
    const auto output =
        vm.run_buffers(lanes, {{0, data}, {1, std::vector<uint32_t>(rows.size() * count)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(output.size(), rows.size() * count);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        std::vector<Known> known;
        known.reserve(built.inputs.size());
        for (uint32_t slot : built.inputs) known.push_back(rows[lane][slot]);
        const auto values = evaluate(built.block, known);
        ASSERT_EQ(expected[lane].size(), count);
        for (uint32_t k = 0; k < count; ++k) {
            const uint32_t want = rows[lane][5] ? expected[lane][k] : rows[lane][6];
            EXPECT_EQ(values[built.outputs[k].index], Known(want))
                << "lane=" << lane << " output=" << k;
            EXPECT_EQ(output[lane * count + k], want) << "lane=" << lane << " output=" << k;
        }
    }
}
std::vector<Row> words(uint32_t wave) {
    std::vector<Row> out;
    out.reserve(wave);
    for (uint32_t lane = 0; lane < wave; ++lane)
        out.push_back({0x8192a3b4u ^ lane * 1234567u, 0xc5d6e7f8u ^ lane * 173u,
                       0x397c892fu ^ lane * 371u, 0xabcdef17u ^ lane * 197u,
                       lane < 4 ? 0xfffffffcu + lane : lane * 17u + 3u, fixture::active(lane, wave),
                       0xd0000000u + lane});
    return out;
}
TEST(IrBufferStoreValues, AtomicFieldIndicesMasksAndPayloadsKeepEveryByteAndWrap) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t bytes : {1u, 2u})
            for (uint32_t k = 0; k < 4; ++k) {
                const auto rows = words(wave);
                std::vector<std::vector<uint32_t>> expected;
                expected.reserve(rows.size());
                const uint32_t mask = bytes == 1 ? 255 : 65535;
                for (const auto& row : rows) {
                    const uint32_t addr = row[4] + k * bytes, shift = (addr % 4) * 8;
                    expected.push_back({addr / 4, mask << shift, (row[k] & mask) << shift});
                }
                check(build_buffer_store_fields(k, bytes, mask), rows, expected);
            }
}
TEST(IrBufferStoreValues, DwordComponentOffsetsAreModuloU32) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t k = 0; k < 4; ++k) {
            const auto rows = words(wave);
            std::vector<std::vector<uint32_t>> expected;
            expected.reserve(rows.size());
            for (const auto& row : rows) expected.push_back({row[4] + k});
            check(build_buffer_store_index(k), rows, expected);
        }
}
TEST(IrBufferStoreValues, PackedWordsMatchIndependentFieldsWithPartialExec) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t f = 9; f < fixture::formats.size(); ++f) {
            const fixture::Case c{f, wave, 0, false};
            const auto& fmt = fixture::formats[f];
            auto rows = words(wave);
            std::vector<std::vector<uint32_t>> expected;
            expected.reserve(rows.size());
            for (uint32_t lane = 0; lane < wave; ++lane) {
                uint32_t value = 0;
                for (uint32_t k = 0; k < 4; ++k) rows[lane][k] = fixture::input_word(c, lane, k);
                for (uint32_t k = 0; k < fmt.components; ++k)
                    value |= fixture::field(c, lane, k)
                             << (fmt.kind == fixture::Kind::Ufloat ? k * 11 : k * 10);
                expected.push_back({value});
            }
            check(build_buffer_store_word({kind(fmt.kind), 4, fmt.components, 0}), rows, expected);
        }
}
TEST(IrBufferStoreValues, NormalizedAndHalfDwordsMatchIndependentTightPacking) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t f : {4u, 5u, 6u, 7u, 8u}) {
            const fixture::Case c{f, wave, 0, false};
            const auto& fmt = fixture::formats[f];
            for (uint32_t count = 1; count <= 4; ++count)
                for (uint32_t d = 0; d < (count * fmt.bytes + 3) / 4; ++d) {
                    auto rows = words(wave);
                    std::vector<std::vector<uint32_t>> expected;
                    expected.reserve(rows.size());
                    for (uint32_t lane = 0; lane < wave; ++lane) {
                        uint32_t value = 0;
                        for (uint32_t k = 0; k < 4; ++k)
                            rows[lane][k] = fixture::input_word(c, lane, k);
                        for (uint32_t k = 0; k < count; ++k)
                            if (k * fmt.bytes / 4 == d)
                                value |= fixture::field(c, lane, k) << ((k * fmt.bytes % 4) * 8);
                        expected.push_back({rows[lane][4] + d, value});
                    }
                    check(build_buffer_store_dword({kind(fmt.kind), count, 4, fmt.bytes}, d), rows,
                          expected);
                }
        }
}
TEST(IrBufferStoreValues, MissingWordsAndExceptionalMiniFloatsStayUnknown) {
    for (auto kind : {BufferStoreKind::Integer, BufferStoreKind::Half, BufferStoreKind::Unorm,
                      BufferStoreKind::Snorm, BufferStoreKind::Ufloat}) {
        const auto built =
            kind == BufferStoreKind::Half
                ? build_buffer_store_dword({kind, 4, 4, 2}, 0)
                : build_buffer_store_word({kind, 4, kind == BufferStoreKind::Ufloat ? 3u : 4u, 0});
        for (Value output : built.outputs) EXPECT_FALSE(evaluate(built.block, {})[output.index]);
    }
    for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7fc01234u, 0x7f7fffffu})
        for (uint32_t mantissa : {5u, 6u})
            EXPECT_FALSE(evaluate_op(Opcode::PackUnsignedFloat, mantissa, bits, 0, 0));
    EXPECT_FALSE(evaluate_op(Opcode::PackNormalizedField, 8, 0x7fc01234, 0, 0));
}
TEST(IrBufferStoreValues, UnsignedFloatNormalSubnormalAndEvenTiesAreHandWorked) {
    // E5M6 and E5M5: one, minimum normal, minimum subnormal, half-minimum (even zero),
    // 1.5 subnormals (even two), maximum subnormal and maximum finite.
    const std::array<uint32_t, 7> e6{0x3f800000, 0x38800000, 0x35800000, 0x35000000,
                                     0x35c00000, 0x387c0000, 0x477e0000};
    const std::array<uint32_t, 7> e5{0x3f800000, 0x38800000, 0x36000000, 0x35800000,
                                     0x36400000, 0x38780000, 0x477c0000};
    const std::array<uint32_t, 7> r6{0x3c0, 0x40, 1, 0, 2, 63, 0x7bf};
    const std::array<uint32_t, 7> r5{0x1e0, 0x20, 1, 0, 2, 31, 0x3df};
    for (uint32_t wave : {32u, 64u})
        for (uint32_t mantissa : {5u, 6u}) {
            auto rows = words(wave);
            std::vector<std::vector<uint32_t>> expected;
            expected.reserve(rows.size());
            for (uint32_t lane = 0; lane < wave; ++lane) {
                rows[lane][0] = (mantissa == 6 ? e6 : e5)[lane % 7];
                expected.push_back({(mantissa == 6 ? r6 : r5)[lane % 7]});
            }
            BufferStoreBlock built;
            Builder ir(built.block);
            built.inputs.push_back(0);
            built.outputs.push_back(ir.pack_unsigned_float(ir.input(Type::U32), mantissa));
            check(std::move(built), rows, expected);
        }
}
TEST(IrBufferStoreValues, MalformedFieldMetadataAndTypesStopLoudly) {
    for (uint32_t width : {0u, 1u, 17u, 32u, 258u})
        EXPECT_DEATH_IF_SUPPORTED(
            {
                prosper::test::suppress_coredump();
                Block b;
                Builder ir(b);
                ir.pack_normalized_field(ir.c32(0), width, false);
            },
            "normalized pack width or mode out of range");
    for (uint32_t mantissa : {0u, 4u, 7u, 32u})
        EXPECT_DEATH_IF_SUPPORTED(
            {
                prosper::test::suppress_coredump();
                Block b;
                Builder ir(b);
                ir.pack_unsigned_float(ir.c32(0), mantissa);
            },
            "unsigned float mantissa out of range");
    EXPECT_DEATH_IF_SUPPORTED(
        {
            prosper::test::suppress_coredump();
            Block b;
            Builder ir(b);
            ir.pack_normalized_field(ir.ctrue(), 8, false);
        },
        "argument type mismatch");
}
TEST(IrBufferStoreValues, StrictValidationFixtureHasHandWorkedObservableValues) {
    std::vector<uint32_t> input;
    input.reserve(size_t{32} * 5);
    for (uint32_t lane = 0; lane < 32; ++lane)
        input.insert(input.end(), {0x3f800000, 0xbf800000, 0x3f000000, 0x3e800000, lane * 17 + 3});
    bpermute_oracle::Interpreter vm(prosper::test::buffer_store_ir::module());
    vm.half_rte_preserve_denorm = true;
    const auto out =
        vm.run_buffers(32, {{0, input}, {1, std::vector<uint32_t>(size_t{32} * 14)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 32u * 14);
    for (uint32_t lane = 0; lane < 32; ++lane) {
        const uint32_t index = lane * 17 + 3, addr = index + 6;
        const std::array<uint32_t, 14> expected{addr / 4,   65535u << ((addr % 4) * 8),
                                                0,          index + 3,
                                                0,          0x600003ff,
                                                0x100805ff, 0x700003c0,
                                                index + 1,  0x34003800,
                                                index,      0x408000ff,
                                                index + 1,  0x20004000};
        for (uint32_t k = 0; k < expected.size(); ++k)
            EXPECT_EQ(out[size_t{lane} * 14 + k], expected[k]) << "lane=" << lane << " field=" << k;
    }
}
} // namespace
