// Pin native row/nibble math independently, and label inherited FI/BC compatibility separately.
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>
#include "gpu/recompiler/ir/permlane_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_permlane.hpp"
#include "gpu/recompiler/ir/emit/spirv_emitter.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <optional>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::permlane;
namespace {
struct Built {
    Block block;
    Value d;
    std::optional<uint32_t> active_slot, old_slot;
};
Built build(bool across, bool fetch, bool bound) {
    Built out;
    Builder ir(out.block);
    const Value source = ir.input(Type::U32), lo = ir.input(Type::U32), hi = ir.input(Type::U32);
    const PermlaneInputs inputs{source, lo, hi,
                                [&] {
                                    out.active_slot = out.block.input_count();
                                    return ir.input(Type::Bool);
                                },
                                [&] {
                                    out.old_slot = out.block.input_count();
                                    return ir.input(Type::U32);
                                }};
    out.d = across ? permlanex16_b32(ir, inputs, fetch, bound)
                   : permlane16_b32(ir, inputs, fetch, bound);
    return out;
}
std::vector<InvocationInputs> inputs(const Built& b, const LaneInputs& in, const cases::Table& t) {
    std::vector<InvocationInputs> result;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        InvocationInputs row(b.block.input_count());
        row[0] = in.s0[lane];
        row[1] = t.lo;
        row[2] = t.hi;
        if (b.active_slot) row[*b.active_slot] = in.active[lane] ? 1u : 0u;
        if (b.old_slot) row[*b.old_slot] = in.old_d[lane];
        result.push_back(row);
    }
    return result;
}
std::vector<uint32_t> run(const Built& x, const std::vector<InvocationInputs>& inputs) {
    const uint32_t lanes = static_cast<uint32_t>(inputs.size());
    SpirvCompute b;
    b.begin(x.block.input_count(), nullptr, lanes, 1, 1, lanes, 0, true, true);
    std::vector<uint32_t> ids, buffer;
    for (uint32_t slot = 0; slot < x.block.input_count(); ++slot) {
        const uint32_t word = b.load_input(slot);
        ids.push_back(x.active_slot == slot ? b.ucmp(Op_INotEqual, word, b.uconst(0)) : word);
    }
    const auto emitted = emit_spirv(b, x.block, ids);
    b.store_output_word(emitted[x.d.index], 1, 0);
    for (const auto& row : inputs)
        for (const auto word : row) buffer.push_back(word.value_or(0));
    bpermute_oracle::Interpreter vm(b.finish());
    vm.native_subgroup_width = lanes;
    auto out = vm.run_buffers(lanes, {{0, buffer}, {1, std::vector<uint32_t>(lanes)}}, 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    out.resize(lanes);
    return out;
}
void check(bool compatibility) {
    for (uint32_t lanes : {16u, 32u, 64u})
        for (bool across : {false, true}) {
            if (across && lanes == 16) continue;
            const auto in = cases::inputs(lanes);
            for (bool fetch : {false, true})
                for (bool bound : {false, true}) {
                    if ((!fetch && !bound) != compatibility) continue;
                    const auto b = build(across, fetch, bound);
                    for (const auto& table : cases::kTables) {
                        const auto lanes_in = inputs(b, in, table);
                        const auto eval = evaluate_lanes(b.block, lanes_in);
                        // The interpreter's native builtin/shuffle contract is 32/64 only.
                        // A single 16-lane row still has independent hand/evaluator coverage.
                        const auto actual =
                            lanes == 16 ? std::vector<uint32_t>{} : run(b, lanes_in);
                        if (lanes != 16) ASSERT_EQ(actual.size(), lanes);
                        for (uint32_t lane = 0; lane < lanes; ++lane) {
                            const auto hand = cases::value(in, lane, table, across, fetch, bound);
                            ASSERT_EQ(eval[lane][b.d.index], Known(hand));
                            if (lanes != 16) ASSERT_EQ(actual[lane], hand) << lanes << ':' << lane;
                        }
                    }
                }
        }
}
}   // namespace
TEST(IrVectorPermlaneValues, IsaRowAndNibbleTruthMatchesEvalAndEmittedSpirv) {
    check(false);
}
TEST(IrVectorPermlaneValues, InheritedInactiveSourceOldDestinationChoiceStaysCompatible) {
    check(true);   // FI=0/BC=0 is compatibility, not an independently settled ISA interpretation.
}
TEST(IrVectorPermlaneValues, EachNativeRowPairRetainsItsOwnWave32Slice) {
    const auto in = cases::inputs(64);
    for (bool across : {false, true}) {
        const auto b = build(across, true, false);
        const auto lanes_in = inputs(b, in, cases::kTables[4]);
        const auto actual = run(b, lanes_in);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t peer = (lane / 32) * 32 + (across ? (lane % 32 < 16 ? 31 : 15)
                                                             : (lane % 32 < 16 ? 15 : 31));
            ASSERT_EQ(actual[lane], in.s0[peer]);
        }
    }
}
TEST(IrVectorPermlaneValues, UnknownAndAbsentPeerDataNeverBecomesKnownNumericData) {
    const auto b = build(true, true, false);
    auto in = inputs(b, cases::inputs(16), cases::kTables[0]);
    const auto too_small = evaluate_lanes(b.block, in);
    for (const auto& row : too_small) EXPECT_FALSE(row[b.d.index].has_value());
    in = inputs(b, cases::inputs(32), cases::kTables[3]);
    in[16][0] = std::nullopt;
    const auto missing = evaluate_lanes(b.block, in);
    for (uint32_t lane = 0; lane < 16; ++lane) EXPECT_FALSE(missing[lane][b.d.index].has_value());
    const Known fixed[] = {1, 2, 3};
    EXPECT_FALSE(evaluate(b.block, fixed)[b.d.index].has_value());
}
