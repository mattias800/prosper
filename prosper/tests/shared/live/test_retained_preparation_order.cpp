// Resource preparation may invalidate a retained input after the pass first sees its target.
// Exercise the registered per-target renderer and real resource builder/drain. Moving selection
// before that build must render the old red input instead of the programmed blue fallback.
#include "fixtures/spirv_triangle.h"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include "shared/live/submit_renderer/backend_draws.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
namespace frontend = prosper::frontend;
namespace submit = frontend::submit_renderer;
using Draws = std::vector<prosper::test::BackendDraw>;
using Group = std::vector<const DrawItem*>;
constexpr uint32_t kWidth = 16, kHeight = 16;
constexpr std::size_t kBytes = std::size_t{kWidth} * kHeight * 4;
constexpr std::array<float, 4> kRed{1, 0, 0, 1}, kBlue{0, 0, 1, 1};
struct Injection {
    uint64_t target;
    const std::vector<uint8_t>& replacement;
    bool observed = false, erased = false, built = false;
};
Injection* active = nullptr;
frontend::RttCache* observed_cache = nullptr;

// GNU linker wrapping intercepts the existing static-library call, without a shipping test hook.
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
    auto& resources = context.draw_resource_ctx;
    observed_cache = &resources.g_rtt;
    Injection* injection = active;
    active = nullptr;
    if (injection) {
        // Reproduce the already-admitted snapshot latch, then let the REAL builder drain.
        std::copy(injection->replacement.begin(), injection->replacement.end(),
                  reinterpret_cast<uint8_t*>(injection->target));
        resources.depth_array_snapshot_admitted = true;
        notify_guest_gpu_write(injection->target, injection->replacement.size());
    }
    auto draws = real_build_draws(context, group, batch);
    if (injection) {
        injection->observed = true;
        injection->erased = !resources.g_rtt.contains(injection->target);
        injection->built = !draws.empty();
    }
    return draws;
}

namespace {
struct InputCase {
    uint32_t slot;
    bool cpu;
};
std::vector<uint8_t> pixels(const std::array<float, 4>& colour) {
    std::vector<uint8_t> result(kBytes);
    for (std::size_t at = 0; at < result.size(); ++at)
        result[at] = static_cast<uint8_t>(colour[at % 4] * 255);
    return result;
}
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

class RetainedPreparationOrder : public ::testing::TestWithParam<InputCase> {
protected:
    void SetUp() override {
        observed_cache = nullptr;
        prosper::register_builtin_hle();
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
        ASSERT_TRUE(map && unmap);
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&base), 0x10000, 3, 0,
                      reinterpret_cast<uint64_t>("retained-preparation"), 0),
                  0u);
        ASSERT_NE(base, 0u);
        frontend::register_live_renderer("", false);
    }
    void TearDown() override {
        active = nullptr;
        if (base) prosper::test::invalidate_persistent_color_target(base);
        if (observed_cache && base) observed_cache->erase(base);
        if (base) unmap(base, 0x10000, 0, 0, 0, 0);
    }
    uint64_t base = 0;
    prosper::HleFn unmap = nullptr;
};

TEST_P(RetainedPreparationOrder, SelectionFollowsTheRealResourceBuildAndDrain) {
    const auto input = GetParam();
    auto draw = draw_to(base, input.slot);
    ASSERT_FALSE(draw.fs.empty());
    observed_cache = nullptr;
    (void)render_submit_items({draw}, kWidth, kHeight); // bootstrap and retain a red GPU image
    ASSERT_NE(observed_cache, nullptr);
    ASSERT_TRUE(observed_cache->contains(base));
    auto& surface = observed_cache->at(base);
    ASSERT_EQ(surface.w, kWidth);
    ASSERT_EQ(surface.h, kHeight);
    ASSERT_EQ(surface.format, VK_FORMAT_R8G8B8A8_UNORM);
    surface.gpu_valid = false;
    surface.has_uniform_color = !input.cpu;
    surface.uniform_color = kRed;
    surface.rgba = input.cpu ? std::make_shared<const std::vector<uint8_t>>(pixels(kRed)) : nullptr;
    prosper::test::invalidate_persistent_color_target(base);
    clear_to(draw, input.slot, kBlue);
    const auto blue = pixels(kBlue);
    Injection injection{base, blue};
    active = &injection;
    (void)render_submit_items({draw}, kWidth, kHeight);
    ASSERT_TRUE(injection.observed);
    ASSERT_TRUE(injection.built);
    ASSERT_TRUE(injection.erased);
    GpuCaptureRttSeed actual;
    std::string error;
    ASSERT_TRUE(read_gpu_capture_rtt_seed(base, actual, error)) << error;
    EXPECT_EQ(actual.rgba, blue); // pre-build selection retains red and fails this witness
}

INSTANTIATE_TEST_SUITE_P(CpuAndClear, RetainedPreparationOrder,
                         ::testing::Values(InputCase{0, true}, InputCase{1, true},
                                           InputCase{0, false}, InputCase{1, false},
                                           InputCase{2, false}),
                         [](const auto& info) {
                             return std::string(info.param.cpu ? "Cpu" : "Clear") +
                                    std::to_string(info.param.slot);
                         });
} // namespace
