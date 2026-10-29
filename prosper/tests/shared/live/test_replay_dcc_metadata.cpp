// Restore the real retained DCC association, then execute a hosted compute fill with no guest
// mapping. The next blended draw must read its start colour from the complete metadata write.
// Partial, nonuniform and unknown later writes must not reuse an earlier uniform-clear fact.
#include "fixtures/spirv_triangle.h"
#include "fixtures/render_runner.h"
#include "shared/live/live_renderer_internal.hpp"
#include "hle/dispatch/nid.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/submit_renderer/backend_draws.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace frontend = prosper::frontend;
namespace submit = frontend::submit_renderer;
using Draws = std::vector<prosper::test::BackendDraw>;
using Group = std::vector<const DrawItem*>;
constexpr uint32_t kWidth = 64, kHeight = 64;
constexpr uint64_t kMappedBytes = 0x20000, kMetadataOffset = 0x10000;
constexpr std::array<float, 4> kRed{1, 0, 0, 1};
frontend::RttCache* observed_cache = nullptr;
#define BUILD_DRAWS_ABI                                                                            \
    "_ZN7prosper8frontend15submit_renderer19build_backend_drawsERNS1_"                             \
    "18BackendDrawContextERKSt6vector"                                                             \
    "IPKNS_3gpu8DrawItemESaIS8_EEPNS_4test22BackendSubmissionBatchE"
} // namespace

Draws real_build_draws(submit::BackendDrawContext&, const Group&,
                       prosper::test::BackendSubmissionBatch*) asm("__real_" BUILD_DRAWS_ABI);
Draws wrapped_build_draws(submit::BackendDrawContext&, const Group&,
                          prosper::test::BackendSubmissionBatch*) asm("__wrap_" BUILD_DRAWS_ABI);
Draws wrapped_build_draws(submit::BackendDrawContext& context, const Group& group,
                          prosper::test::BackendSubmissionBatch* batch) {
    observed_cache = &context.draw_resource_ctx.g_rtt;
    return real_build_draws(context, group, batch);
}

namespace {
void clear_to(DrawItem& draw, uint32_t slot, const std::array<float, 4>& colour) {
    auto& target = draw.ps.color_targets[slot];
    target.has_clear = true;
    std::copy(colour.begin(), colour.end(), target.clear);
    if (slot == 0) {
        draw.ps.has_clear_color = true;
        std::copy(colour.begin(), colour.end(), draw.ps.clear_color);
    } else if (slot == 1) {
        draw.ps.has_clear_color1 = true;
        std::copy(colour.begin(), colour.end(), draw.ps.clear_color1);
    }
}
DrawItem draw_to(uint64_t base, uint32_t slot) {
    DrawItem draw;
    draw.vs.assign(std::begin(kTriVertSpv), std::end(kTriVertSpv));
    const uint32_t fragment[]{
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u, 0xf800180fu | (slot << 4u),
        0x03020100u, 0xbf810000u};
    draw.fs = recompile_fragment(fragment, std::size(fragment));
    draw.vrt = std::make_shared<ShaderResourceTable>();
    draw.prt = std::make_shared<ShaderResourceTable>();
    draw.vertex_count = 3;
    draw.ps.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    draw.ps.color_write_mask = slot == 0 ? 8u : 0u; // RGB must come from the pass's start.
    draw.ps.color1_write_mask = slot == 1 ? 8u : 0u;
    draw.color_targets[slot] = {base, kWidth, kHeight};
    draw.ps.color_targets[slot].format = VK_FORMAT_R8G8B8A8_UNORM;
    draw.ps.color_targets[slot].write_mask = 8;
    if (slot == 0) {
        draw.color0_base = base;
        draw.color0_width = kWidth;
        draw.color0_height = kHeight;
    } else if (slot == 1) {
        draw.color1_base = base;
        draw.color1_width = kWidth;
        draw.color1_height = kHeight;
        draw.ps.color1_format = VK_FORMAT_R8G8B8A8_UNORM;
    }
    clear_to(draw, slot, kRed);
    return draw;
}

using DccIdentity = std::tuple<uint64_t, uint64_t, uint32_t, bool, uint32_t, uint32_t>;
DccIdentity identity(const frontend::RttSurf& s) {
    return {s.dcc_metadata_addr,   s.dcc_metadata_bytes, s.dcc_num_components,
            s.dcc_alpha_is_on_msb, s.dcc_width,          s.dcc_height};
}
class ReplayDccMetadata : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        observed_cache = nullptr;
        prosper::register_builtin_hle();
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
        ASSERT_TRUE(map && unmap);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&mapped), kMappedBytes, 3, 0,
                      reinterpret_cast<uint64_t>("replay-dcc-metadata"), 0),
                  0u);
        target = mapped;
        ASSERT_NE(target, 0u);
        frontend::register_live_renderer("", false);
    }
    void TearDown() override {
        if (target) prosper::test::invalidate_persistent_color_target(target);
        if (observed_cache && target) observed_cache->erase(target);
        if (mapped) unmap(mapped, kMappedBytes, 0, 0, 0, 0);
    }
    void prepare_replay_surface() {
        const auto draw = draw_to(target, 0);
        ASSERT_FALSE(draw.fs.empty());
        (void)render_submit_items({draw}, kWidth, kHeight);
        ASSERT_NE(observed_cache, nullptr);
        ASSERT_TRUE(observed_cache->contains(target));
        ShaderResource sampled;
        sampled.cls = ResourceClass::Texture;
        sampled.gpu_addr = target;
        sampled.format = DataFormat::Unorm8;
        sampled.num_components = 4;
        sampled.alpha_is_on_msb = true;
        sampled.width = kWidth;
        sampled.height = kHeight;
        sampled.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
        sampled.compression_enabled = true;
        sampled.metadata_addr = mapped + kMetadataOffset;
        ASSERT_NE(gpu_capture_dcc_metadata_footprint(sampled), 0u);
        DrawItem declaration = draw;
        declaration.vertex_count = 0;
        declaration.ps.color_write_mask = 0;
        declaration.ps.color_targets[0].write_mask = 0;
        declaration.ps.has_clear_color = false;
        declaration.ps.color_targets[0].has_clear = false;
        declaration.prt = std::make_shared<ShaderResourceTable>();
        declaration.prt->resources.push_back(sampled);
        (void)render_submit_items({declaration}, kWidth, kHeight);
        ASSERT_TRUE(observed_cache->contains(target));
        expected = DccIdentity{sampled.metadata_addr,
                               gpu_capture_dcc_metadata_footprint(sampled),
                               4,
                               true,
                               kWidth,
                               kHeight};
        ASSERT_EQ(identity(observed_cache->at(target)), expected);
        GpuCaptureFile capture;
        capture.metadata.width = kWidth;
        capture.metadata.height = kHeight;
        std::string error;
        if (GetParam()) {
            ASSERT_TRUE(read_all_gpu_capture_rtt_seeds(capture.rtt_seeds, error)) << error;
        } else {
            GpuCaptureRttSeed seed;
            ASSERT_TRUE(read_gpu_capture_rtt_seed(target, seed, error)) << error;
            capture.rtt_seeds.push_back(std::move(seed));
        }
        ASSERT_FALSE(capture.rtt_seeds.empty());
        sampled.metadata_addr += 0x1000;
        declaration.prt->resources[0] = sampled;
        (void)render_submit_items({declaration}, kWidth, kHeight);
        ASSERT_NE(identity(observed_cache->at(target)), expected);
        std::vector<uint8_t> bytes;
        ASSERT_TRUE(serialize_gpu_capture(capture, bytes, error)) << error;
        GpuCaptureFile restored;
        ASSERT_TRUE(deserialize_gpu_capture(bytes, restored, error)) << error;
        observed_cache->erase(target);
        prosper::test::invalidate_persistent_color_target(target);
        ASSERT_EQ(unmap(mapped, kMappedBytes, 0, 0, 0, 0), 0u);
        mapped = 0; // replay owns no readable guest metadata
        ASSERT_TRUE(restore_gpu_replay_rtt_seeds(restored.rtt_seeds, error)) << error;
        ASSERT_TRUE(observed_cache->contains(target));
    }
    void fill_metadata(std::vector<uint8_t>& metadata, uint32_t value, uint32_t groups = 16) {
        const auto [address, bytes, components, alpha, width, height] = expected;
        ASSERT_EQ(bytes, 4096u);
        ASSERT_EQ(metadata.size(), bytes);
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.binding = 0;
        resource.sgpr_base = 0;
        resource.gpu_addr = address;
        resource.size = bytes;
        resource.stride = 4;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.host_data = metadata.data();
        resource.host_data_size = metadata.size();
        auto resources = std::make_shared<ShaderResourceTable>();
        resources->resources.push_back(resource);
        // global index = (TGID_X << 6) + localID; store the DCC clear code, one dword per worker.
        const uint32_t code[]{0x8f148604u, 0x4a000014u, 0x7e0202ffu, value,
                              0xe0702000u, 0x80000100u, 0xbf810000u};
        ComputeShaderConfig config;
        config.user_sgprs.resize(4);
        config.tgid_x_en = true;
        ComputeItem item;
        item.spirv = recompile_compute(code, std::size(code), resources.get(), config);
        ASSERT_FALSE(item.spirv.empty());
        item.resources = resources;
        item.user_sgprs = config.user_sgprs;
        item.launch.local_x = 64;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_x = groups;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.launch.threads_x = groups * 64;
        item.launch.threads_y = item.launch.threads_z = 1;
        const auto cpu_fills = frontend::live_compute_cpu_fill_dispatches();
        const auto submits = frontend::live_compute_queue_submit_attempts();
        ASSERT_TRUE(frontend::execute_live_compute_items({item}));
        EXPECT_EQ(frontend::live_compute_cpu_fill_dispatches(), cpu_fills);
        EXPECT_GT(frontend::live_compute_queue_submit_attempts(), submits);
    }
    void expect_start(const std::array<uint8_t, 4>& expected_pixel) {
        auto draw = draw_to(target, 0);
        draw.ps.color_write_mask = 15;
        draw.ps.blend_enable = true;
        draw.ps.src_color_blend_factor = VK_BLEND_FACTOR_ZERO;
        draw.ps.dst_color_blend_factor = VK_BLEND_FACTOR_ONE;
        draw.ps.src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
        draw.ps.dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
        auto& slot = draw.ps.color_targets[0];
        slot.write_mask = 15;
        slot.blend_enable = true;
        slot.src_color_blend_factor = VK_BLEND_FACTOR_ZERO;
        slot.dst_color_blend_factor = VK_BLEND_FACTOR_ONE;
        slot.src_alpha_blend_factor = VK_BLEND_FACTOR_ONE;
        slot.dst_alpha_blend_factor = VK_BLEND_FACTOR_ZERO;
        (void)render_submit_items({draw}, kWidth, kHeight);
        GpuCaptureRttSeed result;
        std::string error;
        ASSERT_TRUE(read_gpu_capture_rtt_seed(target, result, error)) << error;
        ASSERT_EQ(result.rgba.size(), kWidth * kHeight * 4u);
        for (size_t i = 0; i < result.rgba.size(); i += 4)
            for (size_t c = 0; c < 4; ++c)
                ASSERT_EQ(result.rgba[i + c], expected_pixel[c]) << i / 4 << ":" << c;
    }
    DccIdentity expected{};
    uint64_t mapped = 0, target = 0;
    prosper::HleFn unmap = nullptr;
};

TEST_P(ReplayDccMetadata, CaptureAndRestoreKeepTheExistingDescriptorAssociation) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    EXPECT_EQ(identity(observed_cache->at(target)), expected);
}
TEST_P(ReplayDccMetadata, ComputeMetadataFillDefinesTheFollowingRenderStart) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    std::vector<uint8_t> metadata(4096, 0xff);
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0x40404040));
    ASSERT_TRUE(std::all_of(metadata.begin(), metadata.end(), [](uint8_t b) { return b == 0x40; }));
    ASSERT_NO_FATAL_FAILURE(expect_start({0, 0, 0, 255}));
}
TEST_P(ReplayDccMetadata, AnotherCompleteCodeProducesWhiteRatherThanAConstantBlack) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    std::vector<uint8_t> metadata(4096, 0xff);
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0xc0c0c0c0));
    ASSERT_NO_FATAL_FAILURE(expect_start({255, 255, 255, 255}));
}
TEST_P(ReplayDccMetadata, EqualCompleteFillStillDefinesTheFollowingRenderStart) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    std::vector<uint8_t> metadata(4096, 0x40);   // this dispatch leaves the bytes unchanged
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0x40404040));
    ASSERT_NO_FATAL_FAILURE(expect_start({0, 0, 0, 255}));
}
TEST_P(ReplayDccMetadata, LaterPartialFillRevokesTheCompletePlaneFact) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    std::vector<uint8_t> metadata(4096, 0xff);
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0x40404040));
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0xc0c0c0c0, 1));
    ASSERT_EQ(metadata.front(), 0xc0);
    ASSERT_EQ(metadata.back(), 0x40);
    ASSERT_NO_FATAL_FAILURE(expect_start({255, 0, 0, 255}));
}
TEST_P(ReplayDccMetadata, LaterUnobservedWriteRevokesTheCompletePlaneFact) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    std::vector<uint8_t> metadata(4096, 0xff);
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0x40404040));
    metadata[16] = 0xff;
    notify_guest_gpu_write(std::get<0>(expected) + 16, 1);
    ASSERT_NO_FATAL_FAILURE(expect_start({255, 0, 0, 255}));
}
TEST_P(ReplayDccMetadata, LegacySeedPreservesTheDescriptorAlreadyLearnedForTheSameTarget) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    GpuCaptureRttSeed seed;
    std::string error;
    ASSERT_TRUE(read_gpu_capture_rtt_seed(target, seed, error)) << error;
    seed.dcc_metadata = {};
    ASSERT_TRUE(restore_gpu_replay_rtt_seeds({seed}, error)) << error;
    EXPECT_EQ(identity(observed_cache->at(target)), expected);
    std::vector<uint8_t> metadata(4096, 0xff);
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0x40404040));
    ASSERT_NO_FATAL_FAILURE(expect_start({0, 0, 0, 255}));
}
TEST_P(ReplayDccMetadata, QueuedFactCannotNameAReplacedMetadataAssociation) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    GpuCaptureRttSeed seed;
    std::string error;
    ASSERT_TRUE(read_gpu_capture_rtt_seed(target, seed, error)) << error;
    std::vector<uint8_t> metadata(4096, 0xff);
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0x40404040));
    seed.dcc_metadata.metadata_addr += 0x1000;
    ASSERT_TRUE(restore_gpu_replay_rtt_seeds({seed}, error)) << error;
    ASSERT_NO_FATAL_FAILURE(expect_start({255, 0, 0, 255}));
}
TEST_P(ReplayDccMetadata, FreshLegacySeedCannotInventAnAssociation) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    GpuCaptureRttSeed seed;
    std::string error;
    ASSERT_TRUE(read_gpu_capture_rtt_seed(target, seed, error)) << error;
    seed.dcc_metadata = {};
    observed_cache->erase(target);
    prosper::test::invalidate_persistent_color_target(target);
    ASSERT_TRUE(restore_gpu_replay_rtt_seeds({seed}, error)) << error;
    EXPECT_EQ(identity(observed_cache->at(target)), DccIdentity{});
    std::vector<uint8_t> metadata(4096, 0xff);
    ASSERT_NO_FATAL_FAILURE(fill_metadata(metadata, 0x40404040));
    ASSERT_NO_FATAL_FAILURE(expect_start({255, 0, 0, 255}));
}
TEST_P(ReplayDccMetadata, ChangedTargetCannotBorrowTheOldAssociationFromALegacySeed) {
    ASSERT_NO_FATAL_FAILURE(prepare_replay_surface());
    GpuCaptureRttSeed seed;
    std::string error;
    ASSERT_TRUE(read_gpu_capture_rtt_seed(target, seed, error)) << error;
    seed.dcc_metadata = {};
    seed.width /= 2;
    seed.rgba.resize(size_t{seed.width} * seed.height * 4);
    ASSERT_TRUE(restore_gpu_replay_rtt_seeds({seed}, error)) << error;
    EXPECT_EQ(identity(observed_cache->at(target)), DccIdentity{});
}
INSTANTIATE_TEST_SUITE_P(RegisteredReaders, ReplayDccMetadata, ::testing::Values(false, true),
                         [](const auto& info) { return info.param ? "All" : "Single"; });
}   // namespace
