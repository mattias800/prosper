// DPP16's ISA truth is independent of the lane evaluator; emitted SPIR-V must agree with both.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_dpp_source.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace {
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
struct Built {
    Block block;
    Value result;
};
Built build(bool row, uint32_t control, bool bound) {
    Built out;
    Builder ir(out.block);
    const Value source = ir.input(Type::U32), active = ir.input(Type::Bool);
    const Value old = ir.input(Type::U32);
    Value word;
    if (row) {
        const auto result = dpp_row_shr_source(ir, source, active, control, bound);
        word = bound ? result.word : ir.select(result.in_bounds, result.word, old);
    } else {
        word = dpp_quad_source(ir, source, active, control);
    }
    out.result = ir.select(active, word, old);
    return out;
}
std::vector<InvocationInputs> inputs(uint32_t lanes, uint32_t mask) {
    std::vector<InvocationInputs> out;
    out.reserve(lanes);
    for (uint32_t lane = 0; lane < lanes; ++lane)
        out.push_back({0xa5000000u + lane * 0x101u,
                       uint32_t(mask == 0 || (mask == 1 && lane % 4 != 0)), 0xd0000000u + lane});
    return out;
}
std::vector<uint32_t> execute(const Built& built, const std::vector<InvocationInputs>& in) {
    const uint32_t lanes = static_cast<uint32_t>(in.size());
    SpirvCompute b;
    b.begin(3, nullptr, lanes, 1, 1, lanes, 0, true, true);
    const std::array ids{b.load_input(0), b.ucmp(Op_INotEqual, b.load_input(1), b.uconst(0)),
                         b.load_input(2)};
    const auto emitted = emit_spirv(b, built.block, ids);
    b.store_output_word(emitted[built.result.index], 1, 0);
    std::vector<uint32_t> buffer;
    for (const auto& lane : in)
        for (const auto value : lane) buffer.push_back(value.value_or(0));
    bpermute_oracle::Interpreter vm(b.finish());
    vm.native_subgroup_width = lanes;
    auto got = vm.run_buffers(lanes, {{0, buffer}, {1, std::vector<uint32_t>(lanes)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    got.resize(lanes);
    return got;
}
// AMD 70648 Tables90/91: row arithmetic or the selected entry in a four-element quad table.
uint32_t truth(const std::vector<InvocationInputs>& in, uint32_t lane, bool row, uint32_t control,
               bool bound) {
    const uint32_t old = in[lane][2].value_or(0);
    if (!in[lane][1].value_or(0)) return old;
    if (row && lane % 16 < control) return bound ? 0u : old;
    const std::array selectors{control & 3u, (control >> 2u) & 3u, (control >> 4u) & 3u,
                               (control >> 6u) & 3u};
    const uint32_t peer = row ? lane - control : lane / 4u * 4u + selectors[lane % 4u];
    return in[peer][1].value_or(0) ? in[peer][0].value_or(0) : 0u;
}
void check(bool row, uint32_t control, bool bound, uint32_t lanes, uint32_t mask, bool run) {
    const Built built = build(row, control, bound);
    const auto in = inputs(lanes, mask);
    const auto evaluated = evaluate_lanes(built.block, in);
    const auto got = run ? execute(built, in) : std::vector<uint32_t>{};
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t expected = truth(in, lane, row, control, bound);
        EXPECT_EQ(evaluated[lane][built.result.index], Known(expected))
            << row << ':' << control << ':' << bound << ':' << lanes << ':' << mask << ':' << lane;
        if (run) EXPECT_EQ(got[lane], expected);
    }
}
TEST(IrDppSourceValues, HandWorkedRowBoundaryAndInactiveSourceAreIndependent) {
    auto in = inputs(16, 0);
    for (uint32_t lane = 0; lane < 16; ++lane) in[lane] = {10u + lane, 1u, 100u + lane};
    in[0][1] = 0u;
    const std::array<uint32_t, 6> bounded{100, 0, 0, 11, 12, 13};
    const std::array<uint32_t, 6> unbounded{100, 101, 0, 11, 12, 13};
    for (const bool bound : {false, true}) {
        const Built built = build(true, 2, bound);
        const auto evaluated = evaluate_lanes(built.block, in);
        for (uint32_t lane = 0; lane < bounded.size(); ++lane)
            EXPECT_EQ(evaluated[lane][built.result.index],
                      Known(bound ? bounded[lane] : unbounded[lane]));
    }
}
TEST(IrDppSourceValues, HandWorkedQuadPairSwapReadsTheSelectedSourcesExec) {
    const std::vector<InvocationInputs> in{
        {10u, 0u, 100u}, {20u, 1u, 101u}, {30u, 1u, 102u}, {40u, 1u, 103u}};
    const Built built = build(false, 0xb1, true);
    const auto evaluated = evaluate_lanes(built.block, in);
    const std::array<uint32_t, 4> expected{100, 0, 40, 30};
    for (uint32_t lane = 0; lane < expected.size(); ++lane)
        EXPECT_EQ(evaluated[lane][built.result.index], Known(expected[lane]));
}
TEST(IrDppSourceValues, EveryRowShiftMatchesIsaAndEmittedSpirv) {
    for (uint32_t lanes : {32u, 64u})
        for (uint32_t shift = 1; shift < 16; ++shift)
            for (bool bound : {false, true})
                for (uint32_t mask : {0u, 1u, 2u}) check(true, shift, bound, lanes, mask, true);
}
TEST(IrDppSourceValues, EveryQuadTableMatchesIsaAndEmittedSpirv) {
    for (uint32_t lanes : {32u, 64u})
        for (uint32_t control = 0; control < 256; ++control)
            for (uint32_t mask : {0u, 1u, 2u}) check(false, control, true, lanes, mask, true);
}
TEST(IrDppSourceValues, RowsAndQuadsStayInsideEveryGuestWaveSlice) {
    // The mapping is independent of W: each 16-lane row and four-lane quad lies inside W=32/64.
    for (uint32_t lanes : {16u, 64u, 128u})
        for (uint32_t mask : {0u, 1u, 2u}) {
            check(true, 7, false, lanes, mask, false);
            check(true, 15, true, lanes, mask, false);
            check(false, 0x1b, true, lanes, mask, false);
        }
}
TEST(IrDppSourceValues, UnknownSourceAndOneInvocationNeverGainAValue) {
    for (bool row : {false, true}) {
        const Built built = build(row, row ? 1u : 0xb1u, true);
        auto in = inputs(32, 0);
        in[0][0] = std::nullopt;
        auto values = evaluate_lanes(built.block, in);
        EXPECT_FALSE(values[1][built.result.index].has_value());
        in[0][1] = 0u;
        values = evaluate_lanes(built.block, in);
        // Keep eval_lanes' conservative Select contract: an unselected unknown stays unknown.
        EXPECT_FALSE(values[1][built.result.index].has_value());
        const std::array<Known, 3> single{Known(7u), Known(1u), Known(9u)};
        EXPECT_FALSE(evaluate(built.block, single)[built.result.index].has_value());
    }
}
}   // namespace
