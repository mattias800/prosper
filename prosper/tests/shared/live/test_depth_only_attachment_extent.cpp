// A depth-only pass's attachment extent comes from DB_DEPTH_SIZE_XY, not a disabled color target.
#include "fixtures/render_runner.h"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test;

static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

static bool depth_is(const std::vector<float>& values, uint32_t width, uint32_t height,
                     float left, float right) {
    if (values.size() != size_t(width) * height) return false;
    for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
        const float expected = x < width / 2 ? left : right;
        if (std::bit_cast<uint32_t>(values[y * width + x]) !=
            std::bit_cast<uint32_t>(expected)) return false;
    }
    return true;
}

static bool has_color_key(uint64_t base, uint32_t width, uint32_t height) {
    BackendPersistentResourceGuard guard;
    return find_persistent_color_target(base, width, height, VK_FORMAT_R8G8B8A8_UNORM) != nullptr;
}

static bool has_depth_key(uint64_t base, uint32_t width, uint32_t height) {
    BackendPersistentResourceGuard guard;
    for (const auto& [key, image] : persistent_ds_cache()) {
        (void)image;
        if ((key.dr == base || key.dw == base) && key.w == width && key.h == height &&
            key.slice == 0 && key.fmt == VK_FORMAT_D32_SFLOAT)
            return true;
    }
    return false;
}

int main() {
    prosper::register_builtin_hle();
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    uint64_t guest = 0;
    constexpr size_t GuestBytes = 0x200000;
    check(map && unmap && map(reinterpret_cast<uint64_t>(&guest), GuestBytes, 2, 0,
                             reinterpret_cast<uint64_t>("depth-only-attachment-extent"), 0) == 0 && guest,
          "fixture maps tracked guest addresses");
    if (!guest) return 1;

    constexpr uint32_t DepthW = 32, DepthH = 32, StaleColorW = 64, StaleColorH = 64;
    constexpr uint32_t DbDepth32 = (DepthW - 1) | ((DepthH - 1) << 16);
    constexpr uint32_t DbDepth16 = 15 | (15 << 16);
    const uint64_t MainDepth = guest;
    const uint64_t SplitDepth = guest + 0x40000;
    const uint64_t RawZeroDepth = guest + 0x80000;
    const uint64_t ActiveColorDepth1 = guest + 0xc0000;
    const uint64_t ActiveColorDepth2 = guest + 0xe0000;
    const uint64_t StaleColor = guest + 0x100000;
    const uint32_t vs_words[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    const uint32_t fs_words[]{0x7E0002F2u, 0x7E020280u, 0x7E040280u,
        0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u};
    const auto vs = recompile_vertex(vs_words, std::size(vs_words));
    const auto fs = recompile_fragment(fs_words, std::size(fs_words));
    check(!vs.empty() && !fs.empty(), "fullscreen depth shaders compile");
    if (vs.empty() || fs.empty()) return 1;
    prosper::frontend::register_live_renderer(".", false);

    auto depth_draw = [&](uint64_t base, uint32_t raw_extent, float z) {
        DrawItem draw;
        draw.vs = vs; draw.fs = fs; draw.vertex_count = 3;
        draw.ps.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        draw.ps.color_write_mask = 0;
        draw.ps.depth_test_enable = draw.ps.depth_write_enable = true;
        draw.ps.depth_compare_op = VK_COMPARE_OP_ALWAYS;
        draw.ps.depth_read_base = draw.ps.depth_write_base = base;
        draw.ps.db_depth_size_xy = raw_extent;
        draw.ps.has_viewport = true;
        draw.ps.viewport_w = float((raw_extent & 0x3fffu) + 1);
        draw.ps.viewport_h = float(((raw_extent >> 16) & 0x3fffu) + 1);
        draw.ps.min_depth = draw.ps.max_depth = z;
        return draw;
    };
    auto read = [&](uint64_t base, std::vector<float>& values) {
        std::string error;
        const auto status = read_persistent_ds_depth_array(base, DepthW, DepthH, 0, 1, values, error);
        if (status != PersistentDsDepthArrayStatus::Ready)
            std::fprintf(stderr, "depth read base=0x%llx: %s\n", (unsigned long long)base, error.c_str());
        return status == PersistentDsDepthArrayStatus::Ready;
    };

    // Callback 1: establish the real 32² retained attachment with non-clear contents.
    auto seed = depth_draw(MainDepth, DbDepth32, 0.25f);
    (void)render_submit_items({seed}, StaleColorW, StaleColorH);
    std::vector<float> values;
    check(read(MainDepth, values) && depth_is(values, DepthW, DepthH, 0.25f, 0.25f),
          "first depth-only callback writes the authoritative 32² attachment");

    // Establish actual 64² color authority at the stale CB address. The following depth-only
    // callback must neither overwrite nor re-publish that disabled attachment under a 32² identity.
    DrawItem color_seed = depth_draw(MainDepth, DbDepth32, 0.25f);
    color_seed.ps.depth_test_enable = color_seed.ps.depth_write_enable = false;
    color_seed.ps.color_write_mask = 0xf;
    color_seed.color0_base = StaleColor;
    color_seed.color0_width = StaleColorW; color_seed.color0_height = StaleColorH;
    (void)render_submit_items({color_seed}, StaleColorW, StaleColorH);
    std::vector<uint8_t> saved_color;
    std::string color_error;
    check(readback_persistent_color_target(StaleColor, StaleColorW, StaleColorH,
                                           VK_FORMAT_R8G8B8A8_UNORM, saved_color, color_error) &&
              saved_color.size() == size_t(StaleColorW) * StaleColorH * 4 &&
              saved_color[0] == 255 && saved_color[1] == 0 && saved_color[2] == 0,
          "fixture establishes a real 64² color target before its CB is disabled");

    // Callback 2 deliberately carries that obsolete disabled 64² color description. The DB extent,
    // viewport, and clear-writing VS all agree on 32². Keeping clear_value == VS depth means this
    // test does not depend on an out-of-band clear workaround: the clear and ordinary draw agree.
    auto clear = depth_draw(MainDepth, DbDepth32, 0.75f);
    clear.color0_base = StaleColor;
    clear.color0_width = StaleColorW; clear.color0_height = StaleColorH;
    clear.ps.depth_clear_enable = true;
    clear.ps.depth_clear_value = 0.75f;
    (void)render_submit_items({clear}, StaleColorW, StaleColorH);
    check(read(MainDepth, values) && depth_is(values, DepthW, DepthH, 0.75f, 0.75f),
          "disabled stale color target clears the pre-existing DB-sized depth image");
    check(has_depth_key(MainDepth, DepthW, DepthH) && !has_depth_key(MainDepth, StaleColorW, StaleColorH),
          "depth-only clear mints no persistent key from its disabled 64² color target");
    std::vector<uint8_t> retained_color;
    check(readback_persistent_color_target(StaleColor, StaleColorW, StaleColorH,
                                           VK_FORMAT_R8G8B8A8_UNORM, retained_color, color_error) &&
              retained_color == saved_color && !has_color_key(StaleColor, DepthW, DepthH),
          "disabled CB retains its old 64² color authority and mints no 32² alias");
    ShaderResource sampled_color{};
    sampled_color.cls = ResourceClass::Texture; sampled_color.format = DataFormat::Unorm8;
    sampled_color.num_components = 4; sampled_color.binding = 4; sampled_color.sgpr_base = 8;
    sampled_color.img_dim = 1; sampled_color.width = StaleColorW; sampled_color.height = StaleColorH;
    sampled_color.size = StaleColorW * StaleColorH * 4; sampled_color.gpu_addr = StaleColor;
    sampled_color.mag_filter = sampled_color.min_filter = 0;
    auto sample_table = std::make_shared<ShaderResourceTable>();
    sample_table->resources.push_back(sampled_color);
    const uint32_t sample_ps[]{0x100000ffu, std::bit_cast<uint32_t>(1.0f / StaleColorW),
        0x100202ffu, std::bit_cast<uint32_t>(1.0f / StaleColorH),
        0xf0800f08u, 0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u};
    const PixelSystemInputMapping sample_positions{0x300u, 0x300u};
    DrawItem color_consumer;
    color_consumer.vs = vs;
    color_consumer.fs = recompile_fragment(sample_ps, std::size(sample_ps), sample_table.get(),
                                            &sample_positions);
    color_consumer.vertex_count = 3; color_consumer.ps.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    color_consumer.ps.color_write_mask = 0xf; color_consumer.prt = sample_table;
    color_consumer.color0_base = guest + 0x140000;
    color_consumer.color0_width = StaleColorW; color_consumer.color0_height = StaleColorH;
    const auto sampled_pixels = render_submit_items({color_consumer}, StaleColorW, StaleColorH);
    check(sampled_pixels.size() == size_t(StaleColorW) * StaleColorH * 4 &&
              sampled_pixels[0] == 255 && sampled_pixels[1] == 0 && sampled_pixels[2] == 0,
          "frontend sampling retains the pre-existing disabled-CB color authority");

    auto partial = depth_draw(MainDepth, DbDepth32, 0.5f);
    partial.ps.has_scissor = true;
    partial.ps.scissor_left = 0; partial.ps.scissor_top = 0;
    partial.ps.scissor_right = DepthW / 2; partial.ps.scissor_bottom = DepthH;
    (void)render_submit_items({partial}, StaleColorW, StaleColorH);
    check(read(MainDepth, values) && depth_is(values, DepthW, DepthH, 0.5f, 0.75f),
          "partial writer preserves clear depth in untouched pixels of the same attachment");

    // Consecutive raw DB extents at the same base are separate physical attachment descriptions.
    // A single frontend callback must split them rather than letting the first draw's extent select
    // one cache key for both.
    auto extent32 = depth_draw(SplitDepth, DbDepth32, 0.125f);
    auto extent16 = depth_draw(SplitDepth, DbDepth16, 0.875f);
    (void)render_submit_items({extent32, extent16}, StaleColorW, StaleColorH);
    check(has_depth_key(SplitDepth, DepthW, DepthH) && has_depth_key(SplitDepth, 16, 16),
          "different nonzero raw DB extents split one depth-only callback into distinct attachments");

    // A raw zero DB size is unproven metadata. It retains the established disabled-CB/viewport
    // fallback rather than silently treating 1x1 as an authoritative depth allocation.
    auto raw_zero = depth_draw(RawZeroDepth, 0, 0.375f);
    raw_zero.ps.viewport_w = float(DepthW); raw_zero.ps.viewport_h = float(DepthH);
    raw_zero.color0_base = guest + 0x180000;
    raw_zero.color0_width = StaleColorW; raw_zero.color0_height = StaleColorH;
    (void)render_submit_items({raw_zero}, StaleColorW, StaleColorH);
    check(has_depth_key(RawZeroDepth, StaleColorW, StaleColorH) &&
          !has_depth_key(RawZeroDepth, DepthW, DepthH),
          "raw zero DB extent retains the established disabled-color fallback");

    // Test slots 1 and 2 independently with slot 0 masked. Slot 2 alone catches a predicate
    // that only inspects the legacy MRT0/MRT1 fields.
    auto active_mrt1 = depth_draw(ActiveColorDepth1, DbDepth32, 0.625f);
    active_mrt1.color0_base = guest + 0x1a0000;
    active_mrt1.color0_width = StaleColorW; active_mrt1.color0_height = StaleColorH;
    active_mrt1.ps.color_write_mask = 0;
    active_mrt1.color_targets[1] = {guest + 0x1b0000, StaleColorW, StaleColorH};
    active_mrt1.ps.color1_write_mask = 0xf;
    active_mrt1.ps.color1_format = VK_FORMAT_R8G8B8A8_UNORM;
    active_mrt1.ps.color_targets[1].write_mask = 0xf;
    active_mrt1.ps.color_targets[1].format = VK_FORMAT_R8G8B8A8_UNORM;
    (void)render_submit_items({active_mrt1}, StaleColorW, StaleColorH);
    check(has_depth_key(ActiveColorDepth1, StaleColorW, StaleColorH) &&
          !has_depth_key(ActiveColorDepth1, DepthW, DepthH),
          "active MRT1 preserves the color-target attachment extent with slot 0 masked");

    auto active_mrt2 = depth_draw(ActiveColorDepth2, DbDepth32, 0.875f);
    active_mrt2.color0_base = guest + 0x1c0000;
    active_mrt2.color0_width = StaleColorW; active_mrt2.color0_height = StaleColorH;
    active_mrt2.ps.color_write_mask = 0;
    active_mrt2.color_targets[2] = {guest + 0x1d0000, StaleColorW, StaleColorH};
    active_mrt2.ps.color_targets[2].write_mask = 0xf;
    active_mrt2.ps.color_targets[2].format = VK_FORMAT_R8G8B8A8_UNORM;
    (void)render_submit_items({active_mrt2}, StaleColorW, StaleColorH);
    check(has_depth_key(ActiveColorDepth2, StaleColorW, StaleColorH) &&
          !has_depth_key(ActiveColorDepth2, DepthW, DepthH),
          "active MRT2 preserves the color-target attachment extent with slot 0 masked");

    unmap(guest, GuestBytes, 0, 0, 0, 0);
    std::printf("depth-only attachment extent: %d failures\n", failures);
    return failures ? 1 : 0;
}
