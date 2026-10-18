// Compute DPP source values must retain FI0 zeroing and BC0 destination preservation.
#include "gpu/recompiler/ir/dpp16_source_cases.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/ops/vector_dpp_source.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <optional>

namespace {
using namespace prosper::gpu;
namespace cases = prosper::test::ir::dpp16_source;
namespace ir = prosper::gpu::ir;
void check(const cases::Case& c) {
    const auto p = cases::program(c);
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    ASSERT_FALSE(module.empty());
    bool shuffled = false;
    for (size_t pc = 5; pc < module.size();) {
        const uint32_t words = module[pc] >> 16u;
        ASSERT_NE(words, 0u);
        shuffled |= (module[pc] & 0xffffu) == Op_GroupNonUniformShuffle;
        pc += words;
    }
    EXPECT_TRUE(shuffled);
    for (uint32_t mask : {0u, 1u, 2u}) {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = c.wave;
        const auto in = cases::inputs(c, mask);
        auto got = vm.run_buffers(c.wave, {{3, cases::buffer(in)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        got.resize(c.wave);
        EXPECT_EQ(got, cases::expected(c, in));
        ir::Block block;
        ir::Builder builder(block);
        const auto source = builder.input(ir::Type::U32), active = builder.input(ir::Type::Bool);
        ir::Value transformed;
        std::optional<ir::Value> in_bounds;
        if (c.control >= 0x111) {
            const auto row =
                ir::dpp_row_shr_source(builder, source, active, c.control - 0x110, c.bound);
            transformed = row.word;
            in_bounds = row.in_bounds;
        } else {
            transformed = ir::dpp_quad_source(builder, source, active, c.control);
        }
        std::vector<ir::InvocationInputs> inputs;
        for (uint32_t lane = 0; lane < c.wave; ++lane)
            inputs.push_back({in.s0[lane], uint32_t(in.active[lane])});
        const auto evaluated = ir::evaluate_lanes(block, inputs);
        auto lowered = in;
        for (uint32_t lane = 0; lane < c.wave; ++lane) {
            ASSERT_TRUE(evaluated[lane][transformed.index].has_value());
            lowered.s0[lane] = evaluated[lane][transformed.index].value_or(0);
            if (in_bounds && !c.bound)
                lowered.active[lane] =
                    in.active[lane] && evaluated[lane][in_bounds->index].value_or(0);
        }
        // Identity DPP leaves only the hand-worked ALU combiner; SRC0 now comes from IR eval.
        const cases::Case identity{c.wave, 0xe4, c.op, true};
        EXPECT_EQ(got, cases::expected(identity, lowered));
    }
}
TEST(IrDppSourceExecution, RowShiftControlBeforeValueMigration) {
    for (uint32_t wave : {32u, 64u})
        for (bool bound : {false, true}) check({wave, 0x111, cases::Op::Mov, bound});
}
TEST(IrDppSourceExecution, QuadControlBeforeValueMigration) {
    for (uint32_t wave : {32u, 64u}) check({wave, 0xb1, cases::Op::Mov, true});
}
TEST(IrDppSourceExecution, EveryRowCombinerMatchesEvalAndIsa) {
    for (const auto& c : cases::cases()) {
        if (c.control < 0x111) continue;
        SCOPED_TRACE(::testing::Message()
                     << c.wave << ':' << c.control << ':' << int(c.op) << ':' << c.bound);
        check(c);
    }
}
TEST(IrDppSourceExecution, EveryQuadCombinerMatchesEvalAndIsa) {
    for (const auto& c : cases::cases()) {
        if (c.control >= 0x111) continue;
        SCOPED_TRACE(::testing::Message()
                     << c.wave << ':' << c.control << ':' << int(c.op) << ':' << c.bound);
        check(c);
    }
}
TEST(IrDppSourceExecution, FetchInactiveOneAndRowConversionRemainRefused) {
    for (uint32_t wave : {32u, 64u}) {
        for (uint32_t control : {0x111u, 0xb1u}) {
            const auto p = cases::program({wave, control, cases::Op::Mov, true}, true);
            EXPECT_TRUE(
                recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
        }
        const auto p = cases::program({wave, 0x111, cases::Op::Cvt, true});
        EXPECT_TRUE(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
    }
}
}   // namespace
