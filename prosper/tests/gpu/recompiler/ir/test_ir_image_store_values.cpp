// Storage word graphs run per lane against eval and independent bit patterns. Both mask
// edges and empty EXEC preserve old words; constants never borrow knowledge from another input.
#include "gpu/recompiler/ir/ops/image_store_values.hpp"
#include "gpu/recompiler/ir/image_store_values_cases.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
namespace cases = prosper::test::ir::image_store_values;
namespace {
using Row = std::array<uint32_t, 6>;
std::vector<Row> rows(uint32_t wave, uint32_t mask, bool packed_float = false) {
    std::vector<Row> out;
    out.reserve(wave);
    for (uint32_t lane = 0; lane < wave; ++lane) {
        Row r{};
        for (uint32_t c = 0; c < 4; ++c)
            r[c] = cases::word_value(packed_float ? cases::Kind::HalfFloat : cases::Kind::WordUint,
                                     lane, c);
        r[4] = cases::active(lane, mask) ? 1u : 0u;
        r[5] = 0xd0000000u | lane;
        out.push_back(r);
    }
    return out;
}
void check(ImageStorageValue graph, const std::vector<Row>& data,
           const std::vector<uint32_t>& expected) {
    Builder ir(graph.block);
    graph.sources.push_back(4);
    const Value active = ir.input(Type::U32);
    graph.sources.push_back(5);
    const Value old = ir.input(Type::U32);
    graph.value = ir.select(ir.ine(active, ir.c32(0)), graph.value, old);
    const auto lanes = static_cast<uint32_t>(data.size());
    SpirvCompute b;
    b.begin(graph.block.input_count(), nullptr, lanes, 1, 1, lanes, 0, true, true);
    std::vector<uint32_t> inputs;
    for (uint32_t slot = 0; slot < graph.block.input_count(); ++slot)
        inputs.push_back(b.load_input(slot));
    const auto ids = emit_spirv(b, graph.block, inputs);
    b.store_output_word(ids[graph.value.index], 1, 0);
    std::vector<uint32_t> buffer;
    buffer.reserve(data.size() * graph.sources.size());
    for (const auto& row : data)
        for (uint32_t source : graph.sources) buffer.push_back(row[source]);
    bpermute_oracle::Interpreter vm(b.finish());
    vm.half_rte_preserve_denorm = true;
    const auto output = vm.run_buffers(lanes, {{0, buffer}, {1, std::vector<uint32_t>(lanes)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(output.size(), data.size());
    ASSERT_EQ(expected.size(), data.size());
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        std::vector<Known> inputs_known;
        for (uint32_t source : graph.sources) inputs_known.push_back(data[lane][source]);
        const uint32_t want = data[lane][4] ? expected[lane] : data[lane][5];
        EXPECT_EQ(evaluate(graph.block, inputs_known)[graph.value.index], Known(want)) << lane;
        EXPECT_EQ(output[lane], want) << lane;
    }
}
TEST(IrImageStoreValues, EveryWordDefinitionExecutesWithDistinctAndMaskedLanes) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t mask : {0u, 1u, 2u})
            for (auto kind : {ImageStorageWordKind::Raw, ImageStorageWordKind::FloatHalf,
                              ImageStorageWordKind::UnsignedHalf})
                for (uint32_t half : {0u, 1u}) {
                    const auto data = rows(wave, mask, kind == ImageStorageWordKind::FloatHalf);
                    std::vector<uint32_t> expected;
                    for (uint32_t lane = 0; lane < wave; ++lane)
                        expected.push_back(kind == ImageStorageWordKind::Raw ? data[lane][0]
                                           : kind == ImageStorageWordKind::FloatHalf
                                               ? cases::half_value(lane, half).word
                                               : (data[lane][0] >> (half * 16u)) & 0xffffu);
                    check(build_image_storage_word(kind, half), data, expected);
                }
}
TEST(IrImageStoreValues, EveryChannelAndConstantExecutesWithDistinctAndMaskedLanes) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t mask : {0u, 1u, 2u})
            for (int source : {-1, 0, 1, 2, 3})
                for (uint32_t constant : {0u, 1u, 0x3f800000u}) {
                    const auto data = rows(wave, mask);
                    std::vector<uint32_t> expected;
                    for (const auto& row : data)
                        expected.push_back(source < 0 ? constant : row[source]);
                    check(build_image_storage_select(source, constant), data, expected);
                }
}
TEST(IrImageStoreValues, RawWordsAreIdentityAndUnknownsNeverAcquireValues) {
    for (auto kind : {ImageStorageWordKind::Raw, ImageStorageWordKind::FloatHalf,
                      ImageStorageWordKind::UnsignedHalf}) {
        const auto g = build_image_storage_word(kind, 1);
        const std::array<Known, 1> unknown{std::nullopt};
        EXPECT_FALSE(evaluate(g.block, unknown)[g.value.index]);
        if (kind == ImageStorageWordKind::Raw) EXPECT_EQ(g.block.size(), 1u);
    }
    const auto constant = build_image_storage_select(-1, 0x3f800000);
    EXPECT_TRUE(constant.sources.empty());
    EXPECT_EQ(evaluate(constant.block, {})[constant.value.index], Known(0x3f800000));
    const auto copied = build_image_storage_select(3, 0);
    EXPECT_EQ(copied.sources, (std::vector<uint32_t>{3}));
    const std::array<Known, 1> unknown{std::nullopt};
    EXPECT_FALSE(evaluate(copied.block, unknown)[copied.value.index]);
}
} // namespace
