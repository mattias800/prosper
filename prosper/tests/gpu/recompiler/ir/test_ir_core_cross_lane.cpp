// The cross-lane IR values (LaneId, Shuffle) and the v_readlane_b32 definition built on them:
// the one-invocation evaluator never knows them, the subgroup evaluator (eval_lanes.hpp) is pinned
// on hand-worked lane vectors, and every Block's SPIR-V runs per invocation in the interpreter's
// exact native subgroup against evaluate_lanes, including a Wave32 guest inside a 64-invocation
// subgroup.
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_cross_lane.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;

namespace {

// Lane l's data word: distinct in every lane, so a read from the wrong lane cannot pass.
uint32_t data(uint32_t lane) {
    return 0x5a000000u ^ (lane * 0x01010101u) ^ (lane << 20);
}

std::vector<InvocationInputs> lanes_of(uint32_t count, uint32_t selector) {
    std::vector<InvocationInputs> inputs;
    for (uint32_t lane = 0; lane < count; ++lane) inputs.push_back({data(lane), selector});
    return inputs;
}

struct Readlane {
    Block block;
    Value d;
};
Readlane readlane_block(uint32_t wave_size) {
    Readlane r;
    Builder ir(r.block);
    const Value s0 = ir.input(Type::U32), s1 = ir.input(Type::U32);
    r.d = readlane_b32(ir, s0, s1, wave_size);
    return r;
}

// One module: invocation l reads input slot k as inputs[l][k] (the input buffer, so nothing can
// fold), computes `block`, and stores `result`. Runs in the interpreter's exact native subgroup
// of `lanes` invocations.
std::vector<uint32_t> run_subgroup(const Block& block, Value result,
                                   const std::vector<InvocationInputs>& inputs) {
    const uint32_t lanes = static_cast<uint32_t>(inputs.size());
    const uint32_t slots = block.input_count();
    SpirvCompute b;
    static uint64_t next_address = 0x5c1a0000u;
    b.diagnostic = {RecompileDiagnosticStage::Compute, next_address++};
    b.begin(slots, nullptr, lanes, 1u, 1u, lanes, 0u, true, true);
    std::vector<uint32_t> ids;
    for (uint32_t slot = 0; slot < slots; ++slot) ids.push_back(b.load_input(slot));
    const auto emitted = emit_spirv(b, block, ids);
    b.store_output_word(emitted[result.index], 1u, 0u);
    std::vector<uint32_t> buffer;   // invocation l's slot k at l * slots + k
    for (const auto& lane : inputs)
        for (uint32_t slot = 0; slot < slots; ++slot) buffer.push_back(lane[slot].value_or(0u));
    bpermute_oracle::Interpreter vm(b.finish());
    vm.native_subgroup_width = lanes;
    auto out = vm.run_buffers(lanes, {{0u, buffer}, {1u, std::vector<uint32_t>(lanes, 0u)}}, 1u);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    out.resize(lanes);
    return out;
}

TEST(IrCoreCrossLane, OneInvocationNeverKnowsACrossLaneValue) {
    // The constant folder evaluates one invocation: it must never fold another lane's data.
    Block block;
    Builder ir(block);
    const Value id = ir.lane_id();
    const Value shuffled = ir.shuffle(ir.c32(7), ir.c32(0));
    const auto values = evaluate(block, {});
    EXPECT_FALSE(values[id.index].has_value());
    EXPECT_FALSE(values[shuffled.index].has_value());
    EXPECT_EQ(opcode_info(Opcode::LaneId).arity, 0u);
    EXPECT_EQ(opcode_info(Opcode::Shuffle).arity, 2u);
    EXPECT_EQ(opcode_info(Opcode::Shuffle).result, Type::U32);
}

TEST(IrCoreCrossLane, SubgroupEvaluationReadsTheNamedInvocation) {
    Block block;
    Builder ir(block);
    const Value value = ir.input(Type::U32), lane = ir.input(Type::U32);
    const Value id = ir.lane_id();
    const Value shuffled = ir.shuffle(value, lane);
    // Four invocations: values 10, 11, 12, 13; each names a different source, one out of range.
    const std::vector<InvocationInputs> inputs = {{10u, 3u}, {11u, 0u}, {12u, 2u}, {13u, 4u}};
    const auto values = evaluate_lanes(block, inputs);
    ASSERT_EQ(values.size(), 4u);
    for (uint32_t l = 0; l < 4; ++l) EXPECT_EQ(values[l][id.index], l);
    EXPECT_EQ(values[0][shuffled.index], 13u);
    EXPECT_EQ(values[1][shuffled.index], 10u);
    EXPECT_EQ(values[2][shuffled.index], 12u);
    EXPECT_FALSE(values[3][shuffled.index].has_value()) << "lane 4 is not an invocation";
    // An unknown source value stays unknown in every reader.
    const std::vector<InvocationInputs> unknown = {{std::nullopt, 1u}, {11u, 0u}};
    const auto partial = evaluate_lanes(block, unknown);
    EXPECT_FALSE(partial[1][shuffled.index].has_value());
    EXPECT_EQ(partial[0][shuffled.index], 11u);
}

// Hand-worked from AMD 70648 V_READLANE_B32: lane = S1 % W, EXEC ignored, and a guest wave is the
// aligned W-lane slice of the host subgroup.
uint32_t readlane_expected(uint32_t lane, uint32_t selector, uint32_t wave_size) {
    const uint32_t wave_base = lane / wave_size * wave_size;
    return data(wave_base + selector % wave_size);
}

TEST(IrCoreCrossLane, ReadlaneMatchesTheIsaInBothWaveSizes) {
    struct Case {
        uint32_t subgroup, wave;
    };
    // Wave64 in 64 invocations, Wave32 in 32, and Wave32 as two guest waves in one 64-invocation
    // subgroup (the second wave must read within its own slice).
    for (const Case c : {Case{64, 64}, Case{32, 32}, Case{64, 32}}) {
        const Readlane r = readlane_block(c.wave);
        for (const uint32_t selector : {0u, 5u, 31u, 32u, 37u, 63u, 64u, 69u, 0xffffffffu}) {
            const auto values = evaluate_lanes(r.block, lanes_of(c.subgroup, selector));
            for (uint32_t lane = 0; lane < c.subgroup; ++lane)
                EXPECT_EQ(values[lane][r.d.index], readlane_expected(lane, selector, c.wave))
                    << "subgroup " << c.subgroup << " wave " << c.wave << " selector " << selector
                    << " lane " << lane;
        }
    }
}

TEST(IrCoreCrossLane, SpirvRunsLikeTheSubgroupEvaluation) {
    std::set<Opcode> seen;
    struct Case {
        uint32_t subgroup, wave;
    };
    for (const Case c : {Case{64, 64}, Case{32, 32}, Case{64, 32}}) {
        const Readlane r = readlane_block(c.wave);
        for (const Inst& inst : r.block.insts()) seen.insert(inst.op);
        for (const uint32_t selector : {0u, 6u, 33u, 63u, 70u, 0xffffffffu}) {
            const auto inputs = lanes_of(c.subgroup, selector);
            const auto expected = evaluate_lanes(r.block, inputs);
            const auto actual = run_subgroup(r.block, r.d, inputs);
            ASSERT_EQ(actual.size(), c.subgroup);
            for (uint32_t lane = 0; lane < c.subgroup; ++lane)
                EXPECT_EQ(actual[lane], expected[lane][r.d.index])
                    << "subgroup " << c.subgroup << " wave " << c.wave << " selector " << selector
                    << " lane " << lane;
        }
    }
    // A per-invocation lane index: Shuffle itself, not only readlane's uniform selector.
    Block block;
    Builder ir(block);
    const Value value = ir.input(Type::U32), lane = ir.input(Type::U32);
    const Value shuffled = ir.shuffle(value, ir.and_(lane, ir.c32(63)));
    std::vector<InvocationInputs> inputs;
    for (uint32_t l = 0; l < 64; ++l) inputs.push_back({data(l), (l * 37u + 11u) ^ 0xc0u});
    const auto expected = evaluate_lanes(block, inputs);
    const auto actual = run_subgroup(block, shuffled, inputs);
    for (uint32_t l = 0; l < 64; ++l) EXPECT_EQ(actual[l], expected[l][shuffled.index]) << l;
    EXPECT_TRUE(seen.contains(Opcode::LaneId));
    EXPECT_TRUE(seen.contains(Opcode::Shuffle));
}

// WRITELANE's migrated arm requires an exact native subgroup. Unlike READLANE, this
// definition is never selected for a Wave32 guest in a 64-invocation subgroup.
TEST(IrCoreCrossLane, WritelaneMatchesHandSelectedLanesInBothExactNativeWidths) {
    for (uint32_t wave : {32u, 64u}) {
        Block block;
        Builder ir(block);
        const Value source = ir.input(Type::U32), selector = ir.input(Type::U32);
        const Value d =
            writelane_b32(ir, source, selector, wave, [&] { return ir.input(Type::U32); });
        EXPECT_EQ(block.input_count(), 3u);
        for (uint32_t selected : {0u, 1u, 31u, 32u, 63u, 64u, 69u, 0xffffffffu}) {
            std::vector<InvocationInputs> inputs;
            inputs.reserve(wave);
            for (uint32_t lane = 0; lane < wave; ++lane)
                inputs.push_back({0x5370abcd, selected, data(lane)});
            const auto evaluated = evaluate_lanes(block, inputs);
            const auto executed = run_subgroup(block, d, inputs);
            for (uint32_t lane = 0; lane < wave; ++lane) {
                const uint32_t expected = lane == selected % wave ? 0x5370abcd : data(lane);
                EXPECT_EQ(evaluated[lane][d.index], expected) << wave << " / " << lane;
                EXPECT_EQ(executed[lane], expected) << wave << " / " << lane;
            }
        }
        // The old destination is read AFTER the comparison, not while binding S0/S1.
        size_t old_position = 0, compare_position = 0;
        for (size_t i = 0; i < block.size(); ++i) {
            const Inst& inst = block.insts()[i];
            if (inst.op == Opcode::Input && inst.imm == 2u) old_position = i;
            if (inst.op == Opcode::IEq) compare_position = i;
        }
        EXPECT_GT(old_position, compare_position);
        EXPECT_EQ(block.insts().back().op, Opcode::Select);
    }
}

}   // namespace
