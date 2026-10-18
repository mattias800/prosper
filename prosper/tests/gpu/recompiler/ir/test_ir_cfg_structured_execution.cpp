// CFG step2 must retain arm execution, inactive words and the structured SPIR-V shape; a
// dispatcher that happens to compute the right result does not exercise the region consumer.
#include "gpu/recompiler/ir/cfg_structured_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <system_error>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::cfg_structured;
namespace {
TEST(IrCfgStructuredExecution, EveryArmRunsPerLaneWithFullPartialAndEmptyExec) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t mask : {0u, 1u, 2u}) {
            const auto in = cases::inputs(mask);
            for (uint32_t shape = 0; shape < 5; ++shape)
                for (uint32_t choices = 0; choices < 4; ++choices) {
                    SCOPED_TRACE(testing::Message()
                                 << "wave=" << wave << " mask=" << mask << " shape=" << shape
                                 << " choices=" << choices);
                    const cases::Case c{shape, choices};
                    const auto result = run_vector_op_per_lane(cases::words(c), in, wave);
                    ASSERT_TRUE(result.error.empty()) << result.error;
                    ASSERT_EQ(result.d.size(), in.lanes());
                    for (uint32_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(result.d[lane], cases::expected(c, in, lane)) << lane;
                }
        }
}
TEST(IrCfgStructuredExecution, ProgramsReachStructuredSelectionsWithoutSwitchDispatch) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t shape = 0; shape < 5; ++shape) {
            const auto p = lane_program(cases::words({shape, 3}), 64, wave, true, true);
            const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
            ASSERT_FALSE(module.empty());
            uint32_t selections = 0, switches = 0;
            for (size_t at = 5; at < module.size();) {
                const uint32_t count = module[at] >> 16, op = module[at] & 0xffffu;
                ASSERT_GT(count, 0u);
                ASSERT_LE(at + count, module.size());
                selections += op == 247u;
                switches += op == 251u;
                at += count;
            }
            EXPECT_GT(selections, 0u);
            EXPECT_EQ(switches, 0u);
        }
}
TEST(IrCfgStructuredExecution, InvalidBranchTargetIsRefusedWithAnOtherwiseIdenticalSibling) {
    const auto valid = lane_program(cases::words({1, 1}), 64, 64, true, true);
    EXPECT_FALSE(recompile_compute(valid.code.data(), valid.code.size(), &valid.table, valid.config)
                     .empty());
    auto invalid = valid;
    bool changed = false;
    for (auto& word : invalid.code)
        if ((word & 0xffff0000u) == 0xbf840000u) {
            word = 0xbf847fffu;
            changed = true;
            break;
        }
    ASSERT_TRUE(changed);
    EXPECT_TRUE(
        recompile_compute(invalid.code.data(), invalid.code.size(), &invalid.table, invalid.config)
            .empty());
}
TEST(IrCfgStructuredWitness, ExportsEveryArmShapeAndWave) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable process launch configuration.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code ec;
        fs::remove_all(compare_witness_root("ir_cfg_structured"), ec);
        ASSERT_FALSE(ec);
    }
    ASSERT_FALSE(fs::exists(compare_witness_root("ir_cfg_structured")));
    const auto dump = compare_witness_dump("ir_cfg_structured");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (uint32_t wave : {32u, 64u})
        for (uint32_t shape = 0; shape < 5; ++shape)
            for (uint32_t choices = 0; choices < 4; ++choices) {
                write_compare_witness_program(
                    dump, 0x5d54240000ull + uint64_t(count) * 0x100u, count,
                    lane_program(cases::words({shape, choices}), 64, wave, true, true));
                ++count;
            }
    EXPECT_EQ(count, 40u);
}
} // namespace
