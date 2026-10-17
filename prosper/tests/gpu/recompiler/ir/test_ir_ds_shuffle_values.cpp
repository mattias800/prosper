// AMD 70648 10.4.4 and 12.13.1 give these byte-address and lane tables. Run the same IR graph
// per lane through evaluation and emitted SPIR-V; hand tables, not eval, supply the expected peer.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/ds_shuffle.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {
struct Graph {
    Block block;
    Value d;
    uint32_t bool_slot;
};
Graph bpermute_graph() {
    Graph out;
    Builder ir(out.block);
    const auto address = ir.input(Type::U32), source = ir.input(Type::U32);
    const auto active = ir.input(Type::Bool), offset = ir.input(Type::U32);
    out.d = ds_bpermute_b32(ir, address, source, active, offset);
    out.bool_slot = 2;
    return out;
}
Graph swizzle_graph(uint32_t control) {
    Graph out;
    Builder ir(out.block);
    const DsSwizzleInputs inputs{[&] { return ir.input(Type::U32); },
                                 [&] { return ir.input(Type::Bool); }};
    out.d = control & 0x8000u ? ds_swizzle_quad_b32(ir, control, inputs)
                              : ds_swizzle_mask_b32(ir, control, inputs);
    out.bool_slot = 1;
    return out;
}
std::vector<uint32_t> execute(const Graph& graph, const std::vector<InvocationInputs>& inputs) {
    const auto lanes = static_cast<uint32_t>(inputs.size());
    SpirvCompute b;
    b.begin(graph.block.input_count(), nullptr, lanes, 1, 1, lanes, 0, true, true);
    std::vector<uint32_t> ids, buffer;
    for (uint32_t slot = 0; slot < graph.block.input_count(); ++slot) {
        const auto word = b.load_input(slot);
        ids.push_back(slot == graph.bool_slot ? b.ucmp(Op_INotEqual, word, b.uconst(0)) : word);
    }
    const auto emitted = emit_spirv(b, graph.block, ids);
    b.store_output_word(emitted[graph.d.index], 1, 0);
    for (const auto& row : inputs)
        for (const auto word : row) buffer.push_back(static_cast<uint32_t>(word.value_or(0)));
    bpermute_oracle::Interpreter vm(b.finish());
    vm.native_subgroup_width = lanes;
    auto result = vm.run_buffers(lanes, {{0, buffer}, {1, std::vector<uint32_t>(lanes)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    result.resize(lanes);
    return result;
}
bool active(uint32_t lane, uint32_t mask) {
    return mask == 0 || (mask == 1 && lane % 4 != 0);
}
void check(const Graph& graph, const std::vector<InvocationInputs>& inputs,
           const std::vector<uint32_t>& expected) {
    const auto evaluated = evaluate_lanes(graph.block, inputs);
    const auto executed = execute(graph, inputs);
    ASSERT_EQ(executed.size(), expected.size());
    for (size_t lane = 0; lane < inputs.size(); ++lane) {
        EXPECT_EQ(evaluated[lane][graph.d.index], Known{expected[lane]}) << "lane=" << lane;
        EXPECT_EQ(executed[lane], expected[lane]) << "lane=" << lane;
    }
    EXPECT_FALSE(evaluate(graph.block, inputs.front())[graph.d.index].has_value());
}

TEST(IrDsShuffleValues, BpermuteUsesByteAdditionBeforeAlignmentAndKeepsBothHalves) {
    struct Address {
        uint32_t address, offset, peer;
    };
    // The sums at the last three boundaries are 0, 4 and 3 after uint32 wrap.
    constexpr Address addresses[] = {{0, 0, 0},
                                     {127, 0, 31},
                                     {128, 0, 0},
                                     {0xffffffff, 0, 31},
                                     {0xffffffff, 1, 0},
                                     {0xfffffffc, 8, 1},
                                     {0xffff0004, 0xffff, 0},
                                     {7, 1, 2}};
    for (uint32_t lanes : {32u, 64u})
        for (const auto& a : addresses)
            for (uint32_t mask = 0; mask < 3; ++mask) {
                SCOPED_TRACE(::testing::Message() << lanes << " address=" << a.address
                                                  << " offset=" << a.offset << " mask=" << mask);
                const auto graph = bpermute_graph();
                std::vector<InvocationInputs> inputs;
                std::vector<uint32_t> expected;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    inputs.push_back(
                        {a.address, 0x55000000u + lane, active(lane, mask) ? 1u : 0u, a.offset});
                    const auto peer = (lane / 32) * 32 + a.peer;
                    expected.push_back(active(peer, mask) ? 0x55000000u + peer : 0u);
                }
                check(graph, inputs, expected);
            }
}

TEST(IrDsShuffleValues, QuadSelectorsMatchTheHandWorkedFourLaneTables) {
    struct Quad {
        uint32_t control;
        std::array<uint32_t, 4> peers;
    };
    constexpr Quad quads[] = {{0x8000, {0, 0, 0, 0}},
                              {0x80e4, {0, 1, 2, 3}},
                              {0x80b1, {1, 0, 3, 2}},
                              {0x804e, {2, 3, 0, 1}},
                              {0x801b, {3, 2, 1, 0}}};
    for (uint32_t lanes : {32u, 64u})
        for (const auto& q : quads)
            for (uint32_t mask = 0; mask < 3; ++mask) {
                SCOPED_TRACE(::testing::Message()
                             << lanes << " control=" << q.control << " mask=" << mask);
                const auto graph = swizzle_graph(q.control);
                std::vector<InvocationInputs> inputs;
                std::vector<uint32_t> expected;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    inputs.push_back({0x66000000u + lane, active(lane, mask) ? 1u : 0u});
                    const auto peer = (lane / 4) * 4 + q.peers[lane % 4];
                    expected.push_back(active(peer, mask) ? 0x66000000u + peer : 0u);
                }
                check(graph, inputs, expected);
            }
}

TEST(IrDsShuffleValues, Group32MasksMatchIdentitySwapBroadcastReverseAndHalfTables) {
    constexpr std::array<uint32_t, 32> identity = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                                                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                                   22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    constexpr std::array<uint32_t, 32> swap = {1,  0,  3,  2,  5,  4,  7,  6,  9,  8,  11,
                                               10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                                               23, 22, 25, 24, 27, 26, 29, 28, 31, 30};
    constexpr std::array<uint32_t, 32> reverse = {31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
                                                  20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                                                  9,  8,  7,  6,  5,  4,  3,  2,  1,  0};
    constexpr std::array<uint32_t, 32> upper = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                                                27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21,
                                                22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
    constexpr std::array<uint32_t, 32> lower = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10,
                                                11, 12, 13, 14, 15, 0,  1,  2,  3,  4, 5,
                                                6,  7,  8,  9,  10, 11, 12, 13, 14, 15};
    for (uint32_t lanes : {32u, 64u})
        for (uint32_t control : {0x001fu, 0x041fu, 0x7c1fu, 0x03e0u, 0x020fu, 0x000fu})
            for (uint32_t mask = 0; mask < 3; ++mask) {
                SCOPED_TRACE(::testing::Message()
                             << lanes << " control=" << control << " mask=" << mask);
                const auto graph = swizzle_graph(control);
                std::vector<InvocationInputs> inputs;
                std::vector<uint32_t> expected;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    inputs.push_back({0x77000000u + lane, active(lane, mask) ? 1u : 0u});
                    const auto i = lane % 32;
                    const auto selected = control == 0x1f     ? identity[i]
                                          : control == 0x41f  ? swap[i]
                                          : control == 0x7c1f ? reverse[i]
                                          : control == 0x3e0  ? 31u
                                          : control == 0x20f  ? upper[i]
                                                              : lower[i];
                    const auto peer = (lane / 32) * 32 + selected;
                    expected.push_back(active(peer, mask) ? 0x77000000u + peer : 0u);
                }
                check(graph, inputs, expected);
            }
}
} // namespace
