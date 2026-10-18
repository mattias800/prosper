// AMD 70648 §12.12 DPP16: row XOR selects within each architectural 16-lane row;
// FI=0/BC=1 supplies zero when the selected source lane is inactive.
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/ops/vector_dpp_row_xor.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {
struct Built {
    Block block;
    Value result;
};

Built build(uint32_t stride) {
    Built out;
    Builder ir(out.block);
    const Value source = ir.input(Type::U32);
    const Value active = ir.input(Type::Bool);
    out.result = dpp_row_xor_source(ir, source, active, stride);
    return out;
}

std::vector<InvocationInputs> make_inputs(const Built& built,
                                          const std::vector<std::optional<uint32_t>>& source,
                                          const std::vector<bool>& active) {
    std::vector<InvocationInputs> result;
    for (std::size_t lane = 0; lane < source.size(); ++lane) {
        InvocationInputs row(built.block.input_count());
        row[0] = source[lane];
        row[1] = active[lane] ? 1u : 0u;
        result.push_back(std::move(row));
    }
    return result;
}

std::vector<uint32_t> emit_run(const Built& built, const std::vector<InvocationInputs>& inputs) {
    const uint32_t lanes = static_cast<uint32_t>(inputs.size());
    SpirvCompute b;
    b.begin(built.block.input_count(), nullptr, lanes, 1, 1, lanes, 0, true, true);
    const std::array ids{b.load_input(0), b.ucmp(Op_INotEqual, b.load_input(1), b.uconst(0))};
    const auto emitted = emit_spirv(b, built.block, ids);
    b.store_output_word(emitted[built.result.index], 1, 0);
    std::vector<uint32_t> buffer;
    for (const auto& row : inputs)
        for (const auto word : row) buffer.push_back(word.value_or(0));
    bpermute_oracle::Interpreter vm(b.finish());
    vm.native_subgroup_width = lanes;
    auto output = vm.run_buffers(lanes, {{0, buffer}, {1, std::vector<uint32_t>(lanes)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    output.resize(lanes);
    return output;
}

uint32_t source_lane(uint32_t lane, uint32_t stride) {
    return (lane & ~15u) | ((lane & 15u) ^ (stride & 15u));
}
}  // namespace

TEST(IrDppRowXorValues, HandWorkedDpp16RowTables) {
    constexpr std::array<uint32_t, 5> strides{0, 1, 4, 8, 15};
    constexpr std::array<std::array<uint32_t, 16>, 5> table{{
        {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
        {{1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14}},
        {{4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11}},
        {{8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7}},
        {{15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0}},
    }};
    std::vector<std::optional<uint32_t>> source(16);
    std::vector<bool> active(16, true);
    for (uint32_t lane = 0; lane < 16; ++lane) source[lane] = 0x100u + lane;
    for (std::size_t test = 0; test < strides.size(); ++test) {
        const Built built = build(strides[test]);
        const auto evaluated = evaluate_lanes(built.block, make_inputs(built, source, active));
        for (uint32_t lane = 0; lane < 16; ++lane)
            EXPECT_EQ(evaluated[lane][built.result.index], Known(0x100u + table[test][lane]))
                << "stride=" << strides[test] << " lane=" << lane;
    }
}

TEST(IrDppRowXorValues, EvaluationMatchesEmittedSpirvForEveryStrideAndExecMask) {
    for (uint32_t lanes : {32u, 64u}) {
        std::vector<std::optional<uint32_t>> source(lanes);
        std::vector<bool> active(lanes);
        for (uint32_t lane = 0; lane < lanes; ++lane) source[lane] = 0xA5000000u + lane * 0x101u;
        for (uint32_t mask = 0; mask < 3; ++mask) {
            for (uint32_t lane = 0; lane < lanes; ++lane)
                active[lane] = mask == 0 || (mask == 1 && lane % 3u != 1u);
            for (uint32_t stride = 0; stride < 16; ++stride) {
                const Built built = build(stride);
                const auto inputs = make_inputs(built, source, active);
                const auto evaluated = evaluate_lanes(built.block, inputs);
                const auto emitted = emit_run(built, inputs);
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    const uint32_t peer = source_lane(lane, stride);
                    const Known expected = active[peer] ? source[peer] : Known(0u);
                    EXPECT_EQ(evaluated[lane][built.result.index], expected)
                        << "eval lanes=" << lanes << " stride=" << stride << " lane=" << lane;
                    EXPECT_EQ(emitted[lane], expected.value_or(0u))
                        << "SPIR-V lanes=" << lanes << " stride=" << stride << " lane=" << lane;
                }
            }
        }
    }
}

// This pins the IR's 16-lane row mapping across a 64-invocation view; it makes no
// recompiler admission or native Wave32 execution claim.
TEST(IrDppRowXorValues, GuestWave32RowsStayInsideNative64Rows) {
    constexpr uint32_t lanes = 64, stride = 5;
    std::vector<std::optional<uint32_t>> source(lanes);
    std::vector<bool> active(lanes, true);
    for (uint32_t lane = 0; lane < lanes; ++lane) source[lane] = 0xD0000000u + lane;
    const Built built = build(stride);
    const auto inputs = make_inputs(built, source, active);
    const auto evaluated = evaluate_lanes(built.block, inputs);
    const auto emitted = emit_run(built, inputs);
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t peer = source_lane(lane, stride);
        EXPECT_EQ(evaluated[lane][built.result.index], source[peer]);
        ASSERT_TRUE(source[peer].has_value());
        EXPECT_EQ(emitted[lane], source[peer].value_or(0u));
    }
}

TEST(IrDppRowXorValues, UnknownSourceAndSingleInvocationStayUnknown) {
    constexpr uint32_t lanes = 32, stride = 1;
    std::vector<std::optional<uint32_t>> source(lanes);
    std::vector<bool> active(lanes, true);
    for (uint32_t lane = 0; lane < lanes; ++lane) source[lane] = lane + 1u;
    source[7] = std::nullopt;
    const Built built = build(stride);
    auto evaluated = evaluate_lanes(built.block, make_inputs(built, source, active));
    EXPECT_FALSE(evaluated[6][built.result.index].has_value());
    active[7] = false;
    evaluated = evaluate_lanes(built.block, make_inputs(built, source, active));
    // evaluate_lanes deliberately needs every Select argument known, even the unselected arm.
    // Preserve that conservative contract; runtime inactive-source zero is tested separately.
    EXPECT_FALSE(evaluated[6][built.result.index].has_value());
    source[7] = 0x12345678u;
    evaluated = evaluate_lanes(built.block, make_inputs(built, source, active));
    EXPECT_EQ(evaluated[6][built.result.index], Known(0u));
    EXPECT_EQ(emit_run(built, make_inputs(built, source, active))[6], 0u);
    const Known one_invocation[]{Known(7u), Known(1u)};
    EXPECT_FALSE(evaluate(built.block, one_invocation)[built.result.index].has_value());

    std::vector<std::optional<uint32_t>> short_source(8);
    std::vector<bool> short_active(8, true);
    for (uint32_t lane = 0; lane < 8; ++lane) short_source[lane] = lane + 1u;
    const Built wide_stride = build(8);
    const auto short_eval =
        evaluate_lanes(wide_stride.block, make_inputs(wide_stride, short_source, short_active));
    for (const auto& lane : short_eval) EXPECT_FALSE(lane[wide_stride.result.index].has_value());
}
