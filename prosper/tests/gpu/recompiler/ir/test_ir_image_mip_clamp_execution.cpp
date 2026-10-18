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
#include <cstring>
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
                          const cases::Geometry& g = cases::kAllTail64R32) {
    static std::vector<std::unique_ptr<Allocation>> allocations;
    auto allocation = std::make_unique<Allocation>();
    auto& buffer = allocation->buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    buffer.insert(buffer.end(), in.s2.begin(), in.s2.end());
    allocation->image = cases::image_bytes(g);
    const LaneProgram p =
        cases::program(nsa, levels, wave, reinterpret_cast<uint64_t>(buffer.data()),
                       reinterpret_cast<uint64_t>(allocation->image.data()), g);
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
// 256x256 level zero that is outside the shared tail.
TEST(IrImageMipClampExecution, NonTailBaseControl) {
    const LaneInputs in = cases::inputs();
    for (uint32_t wave : {64u, 32u})
        for (uint32_t levels : {3u, 7u})
            for (bool nsa : {false, true}) {
                SCOPED_TRACE(testing::Message()
                             << "wave " << wave << " levels " << levels << " nsa " << nsa);
                const auto out = run(nsa, levels, wave, in, cases::kNonTail256R32);
                ASSERT_EQ(out.size(), cases::kLanes);
                for (uint32_t lane = 0; lane < cases::kLanes; ++lane)
                    EXPECT_EQ(out[lane], cases::expected(in, lane, levels)) << "lane " << lane;
            }
}

// #5429: two more all-tail chains, so the tail path is not proven on one element size and one
// extent. A 64x64 R16 chain (two-byte elements, a different in-tail origin for every level) and a
// 32x32 R32 chain that stops at level 5 (six declared levels, so the last level differs too).
TEST(IrImageMipClampExecution, AllTailChainsAcrossElementSizeAndExtent) {
    const LaneInputs in = cases::inputs();
    const struct {
        const char* name;
        cases::Geometry g;
        uint32_t levels[2];
    } shapes[] = {{"64x64 r16", cases::kAllTail64R16, {3u, 7u}},
                  {"32x32 r32", cases::kAllTail32R32, {3u, 6u}}};
    for (const auto& shape : shapes)
        for (uint32_t wave : {64u, 32u})
            for (uint32_t levels : shape.levels)
                for (bool nsa : {false, true}) {
                    SCOPED_TRACE(testing::Message() << shape.name << " wave " << wave << " levels "
                                                    << levels << " nsa " << nsa);
                    const auto out = run(nsa, levels, wave, in, shape.g);
                    ASSERT_EQ(out.size(), cases::kLanes);
                    for (uint32_t lane = 0; lane < cases::kLanes; ++lane)
                        EXPECT_EQ(out[lane], cases::expected(in, lane, levels)) << "lane " << lane;
                }
}

// #5429's root cause, pinned without a GPU: a tail-packed selected level keeps `gpu_addr` at the
// allocation base and carries its in-block byte origin separately. The origin is hand-derived:
// level zero is the FIRST tail level of a 64 KiB chain, i.e. tail slot m = 16 - 4 - 1 = 11, whose
// byte origin is 16 << 11 = 0x8000 for every element size. A fixture that added that origin to
// the base had the backend apply it twice and read foreign texels.
TEST(IrImageMipClampCases, TailBasePlacementFollowsTheDescriptorContract) {
    constexpr uint64_t kBase = 0x4000000000ull;
    for (const cases::Geometry& g :
         {cases::kAllTail64R32, cases::kAllTail64R16, cases::kAllTail32R32}) {
        SCOPED_TRACE(testing::Message() << g.width << "x" << g.width << " bpe " << g.texel_bytes);
        const ShaderResource t = cases::texture(g.max_mip + 1u, kBase, g);
        EXPECT_TRUE(t.in_mip_tail);
        EXPECT_EQ(t.gpu_addr, kBase);
        EXPECT_EQ(t.mip_tail_offset, 0x8000u);
        EXPECT_EQ(t.mip_tail_bytes, 0x10000u);
    }
    // The non-tail control is shifted onto its own disjoint level-zero range instead.
    const ShaderResource control = cases::texture(7u, kBase, cases::kNonTail256R32);
    EXPECT_FALSE(control.in_mip_tail);
    EXPECT_EQ(control.gpu_addr,
              kBase +
                  tiled_mip_level_layout(256, 256, 4, cases::kTile, cases::kMaxMip, 0).byte_offset);
    EXPECT_GT(control.gpu_addr, kBase);
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
    for (const cases::Geometry& g : {cases::kAllTail64R32, cases::kAllTail64R16,
                                     cases::kAllTail32R32, cases::kNonTail256R32}) {
        const auto bytes = cases::image_bytes(g);
        const uint32_t bpe = g.texel_bytes;
        for (uint32_t level = 0; level <= g.max_mip; ++level) {
            const auto layout =
                tiled_mip_level_layout(g.width, g.width, bpe, cases::kTile, g.max_mip, level);
            ASSERT_TRUE(layout.supported);
            // Every geometry but the control is tail-packed from level zero.
            if (g.width != 256u)
                EXPECT_TRUE(layout.in_tail) << "level " << level;
            else if (level == 0u)
                EXPECT_FALSE(layout.in_tail);
            const uint32_t width = g.width >> level;
            const size_t texels = size_t{width} * width;
            std::vector<uint8_t> linear(texels * bpe, 0);
            if (layout.in_tail)
                detile_surface_level(linear.data(), bytes.data(), bytes.size(), width, width,
                                     cases::kTile, bpe, layout.tail_x, layout.tail_y);
            else
                detile_surface(linear.data(), bytes.data() + layout.byte_offset, width, width,
                               cases::kTile, 0, bpe);
            for (size_t t = 0; t < texels; ++t) {
                uint32_t value = 0;
                std::memcpy(&value, linear.data() + t * bpe, bpe);
                ASSERT_EQ(value, 0x7100u + level)
                    << g.width << "x" << g.width << " bpe " << bpe << " level " << level
                    << " texel " << t << " tail " << layout.in_tail;
            }
        }
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
