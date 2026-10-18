// CUBE sampling through the production compiler/backend. Independently worked face/texel tables
// distinguish axis reads, modifier-first slots, rounding, clamping and stacked-face scaling.
#include "gpu/recompiler/ir/image_cube_coordinates_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::image_cube;
namespace {
std::vector<uint32_t>& retain(std::vector<uint32_t> words) {
    static std::vector<std::unique_ptr<std::vector<uint32_t>>> owned;
    owned.push_back(std::make_unique<std::vector<uint32_t>>(std::move(words)));
    return *owned.back();
}
class IrImageCubeCoordinatesExecution : public testing::Test {
protected:
    void SetUp() override {
        prosper::frontend::execute_live_compute_items({});
        ASSERT_TRUE(prosper::frontend::live_compute_storage_image_device().initialized);
    }
};
void execute(cases::Arm arm, uint32_t wave, uint32_t mask, uint32_t index) {
    const auto in = cases::inputs(mask);
    auto& buffer = retain(cases::buffer_words(in));
    auto& image = retain(cases::texture_words());
    const auto p = cases::program(arm, wave, reinterpret_cast<uint64_t>(buffer.data()),
                                  reinterpret_cast<uint64_t>(image.data()));
    const uint64_t address = 0x5d54280000ull + uint64_t{index} * 0x100u;
    ComputeItem item;
    item.spirv = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                   {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(address);
    item.user_sgprs = p.config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(p.table);
    item.code_addr = address;
    item.dispatch_index = index;
    item.command_order = uint64_t{index} * 10u;
    item.launch.threads_x = item.launch.local_x = cases::kLanes;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    for (uint32_t lane = 0; lane < cases::kLanes; ++lane)
        EXPECT_EQ(buffer[lane], cases::expected(arm, in, lane)) << "lane=" << lane;
}
TEST_F(IrImageCubeCoordinatesExecution, EveryAcceptedFormReadsItsOwnFaceAndTexel) {
    uint32_t index = 1;
    for (uint32_t wave : {32u, 64u})
        for (const auto arm : cases::arms()) {
            SCOPED_TRACE(testing::Message()
                         << "opcode=" << arm.opcode << " nsa=" << arm.nsa << " wave=" << wave);
            execute(arm, wave, 0, index++);
        }
}
TEST_F(IrImageCubeCoordinatesExecution, PartialAndEmptyExecKeepOldDestination) {
    uint32_t index = 100;
    for (uint32_t wave : {32u, 64u})
        for (const auto arm : cases::arms())
            for (uint32_t mask : {1u, 2u}) {
                SCOPED_TRACE(testing::Message() << "opcode=" << arm.opcode << " nsa=" << arm.nsa
                                                << " wave=" << wave << " mask=" << mask);
                execute(arm, wave, mask, index++);
            }
}
TEST(IrImageCubeCoordinatesCases, UnnormalizedCubeStaysRefusedBesideNormalizedSuccess) {
    for (const auto arm : cases::arms()) {
        const auto good = cases::program(arm, 64);
        EXPECT_FALSE(recompile_compute(good.code.data(), good.code.size(), &good.table, good.config)
                         .empty());
        const auto bad = cases::program(arm, 64, 0, 0, true);
        EXPECT_TRUE(
            recompile_compute(bad.code.data(), bad.code.size(), &bad.table, bad.config).empty());
    }
}
TEST(IrImageCubeCoordinatesWitness, ExportsEveryExistingForm) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable process launch configuration.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code ec;
        fs::remove_all(compare_witness_root("ir_image_cube"), ec);
        ASSERT_FALSE(ec);
    }
    ASSERT_FALSE(fs::exists(compare_witness_root("ir_image_cube")));
    const auto dump = compare_witness_dump("ir_image_cube");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (uint32_t wave : {32u, 64u})
        for (const auto arm : cases::arms()) {
            write_compare_witness_program(dump, 0x5d54280000ull + uint64_t{count} * 0x100u, count,
                                          cases::program(arm, wave));
            ++count;
        }
    EXPECT_EQ(count, 20u);
}
} // namespace
