// Compute DPP source values must retain FI0 zeroing and BC0 destination preservation.
#include "gpu/recompiler/ir/dpp16_source_cases.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace cases = prosper::test::ir::dpp16_source;
void check(const cases::Case& c) {
    const auto p = cases::program(c);
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    ASSERT_FALSE(module.empty());
    for (uint32_t mask : {0u, 1u, 2u}) {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = c.wave;
        const auto in = cases::inputs(c, mask);
        auto got = vm.run_buffers(c.wave, {{3, cases::buffer(in)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        got.resize(c.wave);
        EXPECT_EQ(got, cases::expected(c, in));
    }
}
TEST(IrDppSourceExecution, RowShiftControlBeforeValueMigration) {
    for (uint32_t wave : {32u, 64u})
        for (bool bound : {false, true}) check({wave, 0x111, cases::Op::Mov, bound});
}
TEST(IrDppSourceExecution, QuadControlBeforeValueMigration) {
    for (uint32_t wave : {32u, 64u}) check({wave, 0xb1, cases::Op::Mov, true});
}
} // namespace
