// The MIMG spatial-coordinate definition (ir/ops/image_coordinates.cpp): ISA values worked by
// hand, the address slots each shape reads and in what order, and every shape's SPIR-V run per
// lane in the integer interpreter against its own evaluation.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/image_coordinates.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {

using Form = ImageCoordinateForm;
constexpr ImageCoordinateShape kLoad2d{2, Form::Separate, false, {}};
constexpr ImageCoordinateShape kLoad3d{3, Form::Separate, false, {}};
constexpr ImageCoordinateShape kLoadA16{2, Form::Packed16, false, {}};
constexpr ImageCoordinateShape kSample2dTexels{2, Form::Separate, true, {4, 8, 0}};
constexpr ImageCoordinateShape kSample3dTexels{3, Form::Separate, true, {2, 4, 16}};

uint32_t f(float x) {
    return std::bit_cast<uint32_t>(x);
}

TEST(IrImageCoordinates, HandWorkedCoordinates) {
    using A = std::array<Known, 3>;
    // Separate: each axis is its own address VGPR, unchanged (an integer texel for a load).
    EXPECT_EQ(eval_image_coordinates(kLoad2d, {5u, 7u, 99u}), (A{5u, 7u, std::nullopt}));
    EXPECT_EQ(eval_image_coordinates(kLoad3d, {5u, 7u, 0xffffffffu}), (A{5u, 7u, 0xffffffffu}));
    // A16: x = VADDR[15:0], y = VADDR[31:16], both zero-extended.
    EXPECT_EQ(eval_image_coordinates(kLoadA16, {0x00070005u, 99u, 99u}), (A{5u, 7u, std::nullopt}));
    EXPECT_EQ(eval_image_coordinates(kLoadA16, {0xffff8001u, 0u, 0u}),
              (A{0x8001u, 0xffffu, std::nullopt}));
    // FORCE_UNNORMALIZED: texel 3.0 of width 4 is 0.75; texel 2.0 of height 8 is 0.25.
    EXPECT_EQ(eval_image_coordinates(kSample2dTexels, {f(3.0f), f(2.0f), 0u}),
              (A{f(0.75f), f(0.25f), std::nullopt}));
    // 3D: depth 16 scales texel 8.0 to 0.5.
    EXPECT_EQ(eval_image_coordinates(kSample3dTexels, {f(1.0f), f(1.0f), f(8.0f)}),
              (A{f(0.5f), f(0.25f), f(0.5f)}));
    // The reciprocal is rounded to f32 once, then multiplied: 1.5 * f32(1/3) rounds to 0.5.
    EXPECT_EQ(eval_image_coordinates({2, Form::Separate, true, {3, 1, 0}}, {f(1.5f), f(-2.0f), 0u}),
              (A{f(0.5f), f(-2.0f), std::nullopt}));
    // An axis with extent 0 has no normalized value: the constant 0 (the lowering refuses it).
    EXPECT_EQ(eval_image_coordinates({2, Form::Separate, true, {0, 2, 0}}, {f(3.0f), f(1.0f), 0u}),
              (A{0u, f(0.5f), std::nullopt}));
}

TEST(IrImageCoordinates, EachShapeReadsItsSlotsInTheOldOrder) {
    using V = std::vector<uint32_t>;
    EXPECT_EQ(build_image_coordinates(kLoad2d).inputs, (V{0, 1}));
    EXPECT_EQ(build_image_coordinates(kLoad3d).inputs, (V{0, 1, 2}));
    EXPECT_EQ(build_image_coordinates(kLoadA16).inputs, (V{0}));
    // Normalizing reads nothing more: every read precedes the first reciprocal constant.
    EXPECT_EQ(build_image_coordinates(kSample3dTexels).inputs, (V{0, 1, 2}));
    const Block& block = build_image_coordinates(kSample3dTexels).block;
    ASSERT_GE(block.insts().size(), 4u);
    for (size_t k = 0; k < 3; ++k) EXPECT_EQ(block.insts()[k].op, Opcode::Input) << k;
}

TEST(IrImageCoordinates, AnUnknownWordLeavesOnlyItsAxisUnknown) {
    const auto values = eval_image_coordinates(kLoad3d, {3u, std::nullopt, 4u});
    EXPECT_EQ(values[0], Known{3u});
    EXPECT_FALSE(values[1].has_value());
    EXPECT_EQ(values[2], Known{4u});
    // A16 reads only slot 0: an unknown slot 1 cannot make it unknown.
    EXPECT_EQ(eval_image_coordinates(kLoadA16, {0x00020001u, std::nullopt, std::nullopt})[1],
              Known{2u});
}

// Per lane: the shape's Block emitted through SpirvCompute with every input loaded from the input
// buffer (so nothing folds), run in the interpreter; each lane stores its three coordinates.
std::vector<uint32_t> run_block(const ImageCoordinateShape& shape,
                                const std::vector<std::array<uint32_t, 3>>& lanes) {
    const ImageCoordinateBlock built = build_image_coordinates(shape);
    const uint32_t n = static_cast<uint32_t>(lanes.size());
    const uint32_t slots = built.block.input_count();
    SpirvCompute b;
    static uint64_t next_address = 0x1a6e0000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_address++};
    b.begin(slots, nullptr, n, 1u, 1u, n, 0u, true, true);
    std::vector<uint32_t> ids;
    ids.reserve(slots);
    for (uint32_t slot = 0; slot < slots; ++slot) ids.push_back(b.load_input(slot));
    const auto emitted = emit_spirv(b, built.block, ids);
    for (uint32_t axis = 0; axis < shape.axes; ++axis)
        b.store_output_word(emitted[built.coord[axis].index], 3u, axis);
    std::vector<uint32_t> buffer;
    for (const auto& lane : lanes)
        for (uint32_t slot = 0; slot < slots; ++slot) buffer.push_back(lane[built.inputs[slot]]);
    bpermute_oracle::Interpreter vm(b.finish());
    const size_t output_words = size_t{3} * n;
    auto out = vm.run_buffers(n, {{0u, buffer}, {1u, std::vector<uint32_t>(output_words, 0u)}}, 1u);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    out.resize(output_words);
    return out;
}

TEST(IrImageCoordinates, EverySpirvShapeMatchesItsEvaluationPerLane) {
    std::vector<std::array<uint32_t, 3>> lanes;
    uint32_t seed = 0x1a6eu;
    const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
    // Finite texel-space values (a NaN product has no evaluation), with integer edges in the
    // first lanes for the integer shapes; the float shapes read the same words as f32 bits.
    const uint32_t edges[] = {0u,          1u,          0xffffu,     0x10000u,
                              0x7f7fffffu, 0x3f800000u, 0xbf000000u, 0x00800000u};
    for (uint32_t lane = 0; lane < 64; ++lane) {
        if (lane < 8)
            lanes.push_back({edges[lane], edges[7 - lane], edges[(lane + 3) % 8]});
        else
            lanes.push_back({f(float(next() % 100000u) / 7.0f), f(float(next() % 4096u) - 2048.0f),
                             f(float(next() % 65536u) * 0.5f)});
    }
    const ImageCoordinateShape shapes[] = {
        kLoad2d,         kLoad3d,         kLoadA16,
        kSample2dTexels, kSample3dTexels, {2, Form::Separate, true, {3, 1920, 0}}};
    for (const auto& shape : shapes) {
        const auto out = run_block(shape, lanes);
        ASSERT_EQ(out.size(), 3 * lanes.size());
        for (size_t lane = 0; lane < lanes.size(); ++lane) {
            const auto& w = lanes[lane];
            const auto want = eval_image_coordinates(shape, {w[0], w[1], w[2]});
            for (uint32_t axis = 0; axis < shape.axes; ++axis) {
                const auto expected = want[axis];
                if (expected) {
                    EXPECT_EQ(out[3 * lane + axis], *expected)
                        << "axes " << shape.axes << " form " << int(shape.form) << " normalize "
                        << shape.normalize << " lane " << lane << " axis " << axis;
                } else {
                    FAIL() << "lane " << lane << " axis " << axis << " has no expected value";
                }
            }
        }
    }
}

}   // namespace
