// Dynamic IMAGE_LOAD_MIP must clamp each lane's unsigned selector to the materialized image,
// preserving inactive VDST. A wrong clamp can request undefined SPIR-V Lod values or a wrong mip.
#include "gpu/recompiler/ir/image_mip_clamp_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/mip_chain_plan.hpp"
#include "gpu/texture/tile.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "shared/live/live_compute.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::image_mip_clamp;
namespace {

struct Allocation {
    std::vector<uint32_t> buffer;
    std::vector<uint8_t> image;
};

std::vector<uint32_t> run(bool nsa, uint32_t levels, uint32_t wave, const LaneInputs& in,
                          uint32_t width = cases::kWidth) {
    static std::vector<std::unique_ptr<Allocation>> allocations;
    auto allocation = std::make_unique<Allocation>();
    auto& buffer = allocation->buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    buffer.insert(buffer.end(), in.s2.begin(), in.s2.end());
    allocation->image = cases::image_bytes(width);
    const LaneProgram p =
        cases::program(nsa, levels, wave, reinterpret_cast<uint64_t>(buffer.data()),
                       reinterpret_cast<uint64_t>(allocation->image.data()), width);
    const uint64_t address = lane_harness_address();
    ComputeItem item;
    item.spirv = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                   {RecompileDiagnosticStage::Compute, address});
    if (item.spirv.empty()) {
        ADD_FAILURE() << last_terminal_reject_reason(address);
        return {};
    }
    item.user_sgprs = p.config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(p.table);
    item.code_addr = address;
    item.dispatch_index = static_cast<uint32_t>(allocations.size() + 1);
    item.command_order = uint64_t{item.dispatch_index} * 10;
    item.launch.threads_x = item.launch.local_x = cases::kLanes;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    if (!prosper::frontend::execute_live_compute_items({item})) {
        ADD_FAILURE() << "dynamic mip dispatch failed";
        return {};
    }
    std::vector<uint32_t> result(buffer.begin(), buffer.begin() + cases::kLanes);
    allocations.push_back(std::move(allocation));
    return result;
}

TEST(IrImageMipClampExecution, PerLaneUnsignedSelectorsAndPartialExec) {
    const LaneInputs in = cases::inputs();
    for (uint32_t wave : {64u, 32u})
        for (uint32_t levels : {3u, 7u})
            for (bool nsa : {false, true}) {
                SCOPED_TRACE(testing::Message()
                             << "wave " << wave << " levels " << levels << " nsa " << nsa);
                const auto out = run(nsa, levels, wave, in);
                ASSERT_EQ(out.size(), cases::kLanes);
                for (uint32_t lane = 0; lane < cases::kLanes; ++lane)
                    EXPECT_EQ(out[lane], cases::expected(in, lane, levels)) << "lane " << lane;
            }
}

// Independent geometry control: the same selectors, bytes per texel and expectations, with a
// 256x256 level zero that is outside the shared tail. Keep the failing all-tail arm above intact.
TEST(IrImageMipClampExecution, NonTailBaseControl) {
    const LaneInputs in = cases::inputs();
    for (uint32_t wave : {64u, 32u})
        for (uint32_t levels : {3u, 7u})
            for (bool nsa : {false, true}) {
                SCOPED_TRACE(testing::Message()
                             << "wave " << wave << " levels " << levels << " nsa " << nsa);
                const auto out = run(nsa, levels, wave, in, 256u);
                ASSERT_EQ(out.size(), cases::kLanes);
                for (uint32_t lane = 0; lane < cases::kLanes; ++lane)
                    EXPECT_EQ(out[lane], cases::expected(in, lane, levels)) << "lane " << lane;
            }
}

TEST(IrImageMipClampExecution, EmptyExecPreservesEveryOldWord) {
    const LaneInputs in = cases::inputs(true);
    for (uint32_t wave : {64u, 32u})
        for (bool nsa : {false, true}) EXPECT_EQ(run(nsa, 7u, wave, in), in.old_d);
}

TEST(IrImageMipClampAdmission, MissingMaterializationStillRefuses) {
    for (uint32_t wave : {64u, 32u})
        for (bool nsa : {false, true}) {
            auto p = cases::program(nsa, 7u, wave);
            EXPECT_EQ(shader_resource_compute_mip_chain_levels(p.table.resources[1]), 7u);
            ASSERT_FALSE(
                recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
            p.table.resources[1].mip_chain_bytes_per_block = 0;
            // The planner falls back to one materialized base level, never zero.
            EXPECT_EQ(shader_resource_compute_mip_chain_levels(p.table.resources[1]), 1u);
            EXPECT_TRUE(
                recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
        }
}

TEST(IrImageMipClampCases, TiledLevelsContainIndependentExpectedTexels) {
    const auto bytes = cases::image_bytes();
    for (uint32_t level = 0; level <= cases::kMaxMip; ++level) {
        const auto layout = tiled_mip_level_layout(cases::kWidth, cases::kWidth, 4, cases::kTile,
                                                   cases::kMaxMip, level);
        ASSERT_TRUE(layout.supported);
        const uint32_t width = cases::kWidth >> level;
        std::vector<uint32_t> texels(size_t{width} * width, 0);
        if (layout.in_tail)
            detile_surface_level(reinterpret_cast<uint8_t*>(texels.data()), bytes.data(),
                                 bytes.size(), width, width, cases::kTile, 4, layout.tail_x,
                                 layout.tail_y);
        else
            detile_surface(reinterpret_cast<uint8_t*>(texels.data()),
                           bytes.data() + layout.byte_offset, width, width, cases::kTile, 0, 4);
        EXPECT_EQ(texels, std::vector<uint32_t>(size_t{width} * width, 0x7100u + level))
            << "level " << level << " tail " << layout.in_tail << " x " << layout.tail_x << " y "
            << layout.tail_y;
    }
}

TEST(IrImageMipClampWitness, ExportsReplayablePrograms) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_image_mip_clamp_witness");
    if (!configured) fs::remove_all(root);
    ASSERT_FALSE(fs::exists(root));
    const fs::path dump = compare_witness_dump("ir_image_mip_clamp_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (uint32_t wave : {64u, 32u})
        for (uint32_t levels : {3u, 7u})
            for (bool nsa : {false, true}) {
                write_compare_witness_program(dump, 0x1a73000000ull + count * 0x100u, count,
                                              cases::program(nsa, levels, wave));
                ++count;
            }
    EXPECT_EQ(count, 8u);
}
}   // namespace
