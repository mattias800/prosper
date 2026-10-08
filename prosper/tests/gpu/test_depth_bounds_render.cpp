// test_depth_bounds_render -- the guest depth-bounds test (DB_DEPTH_CONTROL.DEPTH_BOUNDS_ENABLE)
// executes on the GPU against the retained depth surface, with the depth test itself OFF, and is
// applied ONLY against depth the guest produced (#4704 review B1).
//
// A producer writes depth 0.25 into a guest-identified persistent D32 surface. A consumer then
// draws a fullscreen red triangle with the depth test disabled and only the bounds test enabled:
//   * bounds [0.5, 1.0] exclude the stored 0.25, so NO pixel may change. Before the test was
//     implemented the bounds were never read, the pass bound no depth attachment, and the draw
//     covered the whole target -- which is how Kena's three shadow-cascade projections each wrote
//     every pixel (KENA_STATUS.md).
//   * bounds [0.0, 0.5] include it, so EVERY pixel turns red. This arm is the positive control:
//     it proves the draw ran and that the bounds compared the LOADED depth, since a freshly
//     created attachment would hold its far default (1.0) and fail this range too.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/render_runner.h"
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t W = 64, H = 64;
constexpr uint64_t kDepthBase = 0x20d6000000ull;

size_t red_pixels(const std::vector<uint8_t>& rgba) {
    size_t red = 0;
    for (size_t i = 0; i + 3 < rgba.size(); i += 4)
        red += rgba[i] > 200 && rgba[i + 1] < 40 && rgba[i + 2] < 40;
    return red;
}
}   // namespace

TEST(DepthBoundsRender, BoundsTestReadsTheRetainedDepthWithTheDepthTestOff) {
    // Fullscreen triangle at z = 0.25 (v7) and at z = 0 (the consumer), PARAM0 = (u, v, 0, 1).
    const uint32_t vs_z[] = {
        0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u,
        0x060606f3u, 0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e02ffu, 0x3e800000u,
        0x7e1002f2u, 0xf80008cfu, 0x08070403u, 0xf800020fu, 0x08070605u, 0xbf810000u,
    };
    const uint32_t vs[] = {
        0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u, 0x060606f3u,
        0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e0280u, 0x7e1002f2u, 0xf80008cfu, 0x08070403u,
        0xf800020fu, 0x08070605u, 0xbf810000u,
    };
    // exp mrt0 (1, 0, 0, 1): opaque red.
    const uint32_t ps_red[] = {
        0x7e0002f2u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800180fu, 0x03020100u, 0xbf810000u,
    };
    const std::vector<uint32_t> vert_z = recompile_vertex(vs_z, std::size(vs_z));
    const std::vector<uint32_t> vert = recompile_vertex(vs, std::size(vs));
    const std::vector<uint32_t> red = recompile_fragment(ps_red, std::size(ps_red), nullptr);
    ASSERT_FALSE(vert_z.empty() || vert.empty() || red.empty()) << "test shaders recompile";

    // The producer must create the persistent surface before the device context is queried.
    ResolvedPipelineState producer{};
    producer.topology = 3; producer.color_write_mask = 0xF;
    producer.depth_test_enable = true; producer.depth_write_enable = true;
    producer.depth_compare_op = 7;   // ALWAYS: store 0.25 everywhere
    producer.depth_read_base = kDepthBase; producer.depth_write_base = kDepthBase;
    prosper::test::BackendDraw pw;
    pw.vs = vert_z; pw.fs = red; pw.ps = &producer; pw.vcount = 3;
    const std::vector<uint8_t> produced = prosper::test::render_draws_rgba(
        {pw}, W, H, nullptr, nullptr, /*persist_depth_stencil=*/true);
    ASSERT_EQ(produced.size(), size_t(W) * H * 4) << "depth producer rendered";
    if (!prosper::test::render_vk_ctx().depth_bounds_enabled)
        GTEST_SKIP() << "device does not offer VkPhysicalDeviceFeatures::depthBounds";

    const auto consume = [&](float min, float max) {
        ResolvedPipelineState consumer{};
        consumer.topology = 3; consumer.color_write_mask = 0xF;
        consumer.depth_bounds_enable = true;   // depth test, write and clear all OFF
        consumer.depth_bounds_min = min; consumer.depth_bounds_max = max;
        consumer.depth_read_base = kDepthBase; consumer.depth_write_base = kDepthBase;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = red; draw.ps = &consumer; draw.vcount = 3;
        const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        return prosper::test::render_draws_rgba({draw}, W, H, nullptr, black,
                                                /*persist_depth_stencil=*/true);
    };

    const std::vector<uint8_t> excluded = consume(0.5f, 1.0f);
    const std::vector<uint8_t> included = consume(0.0f, 0.5f);
    ASSERT_EQ(excluded.size(), size_t(W) * H * 4);
    ASSERT_EQ(included.size(), size_t(W) * H * 4);
    std::printf("  red pixels: bounds [0.5,1] %zu, bounds [0,0.5] %zu of %u\n",
                red_pixels(excluded), red_pixels(included), W * H);
    EXPECT_EQ(red_pixels(excluded), 0u)
        << "stored depth 0.25 lies outside [0.5, 1]: the bounds test must discard every fragment";
    EXPECT_EQ(red_pixels(included), size_t(W) * H)
        << "stored depth 0.25 lies inside [0, 0.5]: every fragment passes (positive control)";

    auto& cache = prosper::test::persistent_ds_cache();
    for (auto it = cache.begin(); it != cache.end();)
        it = it->first.dr == kDepthBase ? cache.erase(it) : std::next(it);
}

// ---- #4704 review: the bounds are compared only against depth the guest produced. ----
namespace {
struct Shaders {
    std::vector<uint32_t> vert_z, vert, red, sample;
};
const Shaders& shaders() {
    static const Shaders s = [] {
        const uint32_t vs_z[] = {
            0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u,
            0x060606f3u, 0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e02ffu, 0x3e800000u,
            0x7e1002f2u, 0xf80008cfu, 0x08070403u, 0xf800020fu, 0x08070605u, 0xbf810000u,
        };
        const uint32_t vs[] = {
            0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x100602f6u, 0x100804f6u,
            0x060606f3u, 0x060808f3u, 0x100a02f4u, 0x100c04f4u, 0x7e0e0280u, 0x7e1002f2u,
            0xf80008cfu, 0x08070403u, 0xf800020fu, 0x08070605u, 0xbf810000u,
        };
        const uint32_t ps_red[] = {
            0x7e0002f2u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
            0xf800180fu, 0x03020100u, 0xbf810000u,
        };
        // Samples the T# at s[8:15] at (0.25, 0.25) and exports the texel: R = the stored depth.
        const uint32_t ps_sample[] = {
            0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
            0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        ShaderResourceTable rt;
        ShaderResource t{};
        t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1;
        t.width = W; t.height = H; t.sgpr_base = 8;
        rt.resources.push_back(t);
        return Shaders{recompile_vertex(vs_z, std::size(vs_z)), recompile_vertex(vs, std::size(vs)),
                       recompile_fragment(ps_red, std::size(ps_red), nullptr),
                       recompile_fragment(ps_sample, std::size(ps_sample), &rt)};
    }();
    return s;
}

ResolvedPipelineState bounds_only(uint64_t depth_base, float min, float max) {
    ResolvedPipelineState ps{};
    ps.topology = 3; ps.color_write_mask = 0xF;
    ps.depth_bounds_enable = true;   // depth test, write and clear all OFF
    ps.depth_bounds_min = min; ps.depth_bounds_max = max;
    ps.depth_read_base = depth_base; ps.depth_write_base = depth_base;
    return ps;
}

std::vector<uint8_t> render(const std::vector<prosper::test::BackendDraw>& draws) {
    const float black[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    return prosper::test::render_draws_rgba(draws, W, H, nullptr, black,
                                            /*persist_depth_stencil=*/true);
}

void forget(uint64_t depth_base) {
    auto& cache = prosper::test::persistent_ds_cache();
    for (auto it = cache.begin(); it != cache.end();)
        it = it->first.dr == depth_base ? cache.erase(it) : std::next(it);
}

bool depth_valid(uint64_t depth_base) {
    for (const auto& [key, image] : prosper::test::persistent_ds_cache())
        if (key.dr == depth_base && image.depth_valid) return true;
    return false;
}

bool device_has_depth_bounds() {
    // Bring the device up with a trivial draw before asking what it offers.
    ResolvedPipelineState plain{};
    plain.topology = 3; plain.color_write_mask = 0xF;
    prosper::test::BackendDraw draw;
    draw.vs = shaders().vert; draw.fs = shaders().red; draw.ps = &plain; draw.vcount = 3;
    (void)prosper::test::render_draws_rgba({draw}, W, H);
    return prosper::test::render_vk_ctx().depth_bounds_enabled;
}
}   // namespace

TEST(DepthBoundsRender, NoDepthSurfaceLeavesTheDrawUntested) {
    ASSERT_FALSE(shaders().vert.empty() || shaders().red.empty());
    if (!device_has_depth_bounds()) GTEST_SKIP() << "device lacks depthBounds";
    // Every DB base 0: hardware has no depth surface to compare. The draw must cover the target,
    // as it did before the test existed -- not be compared against a cleared 1.0 that [0, 0.5]
    // excludes.
    const ResolvedPipelineState ps = bounds_only(0, 0.0f, 0.5f);
    prosper::test::BackendDraw draw;
    draw.vs = shaders().vert; draw.fs = shaders().red; draw.ps = &ps; draw.vcount = 3;
    const auto out = render({draw});
    ASSERT_EQ(out.size(), size_t(W) * H * 4);
    EXPECT_EQ(red_pixels(out), size_t(W) * H);
}

TEST(DepthBoundsRender, NeverWrittenPlaneIsUntestedAndStaysInvalid) {
    ASSERT_FALSE(shaders().vert.empty() || shaders().red.empty());
    if (!device_has_depth_bounds()) GTEST_SKIP() << "device lacks depthBounds";
    constexpr uint64_t kFresh = 0x20d6100000ull;
    forget(kFresh);
    // Kena's near-cascade slice against a guest surface nothing has written: untested, so every
    // pixel is drawn...
    const ResolvedPipelineState ps = bounds_only(kFresh, 0.0005f, 0.0016f);
    prosper::test::BackendDraw draw;
    draw.vs = shaders().vert; draw.fs = shaders().red; draw.ps = &ps; draw.vcount = 3;
    const auto out = render({draw});
    ASSERT_EQ(out.size(), size_t(W) * H * 4);
    EXPECT_EQ(red_pixels(out), size_t(W) * H);
    // ...and the bounds-only pass, which only READS the plane, must not promote it to valid: a
    // later pass would LOAD, and the sampled-depth bridge serve, the value prosper cleared it to.
    EXPECT_FALSE(depth_valid(kFresh)) << "a bounds-only pass marked a never-written plane valid";
    forget(kFresh);
}

TEST(DepthBoundsRender, AnEarlierWriterInTheSamePassMakesTheDepthKnown) {
    ASSERT_FALSE(shaders().vert_z.empty() || shaders().red.empty());
    if (!device_has_depth_bounds()) GTEST_SKIP() << "device lacks depthBounds";
    constexpr uint64_t kSamePass = 0x20d6200000ull;
    for (const auto& [min, max, want] : {std::tuple{0.5f, 1.0f, size_t{0}},
                                         std::tuple{0.0f, 0.5f, size_t(W) * H}}) {
        forget(kSamePass);
        ResolvedPipelineState writer{};
        writer.topology = 3; writer.color_write_mask = 0;   // depth only
        writer.depth_test_enable = true; writer.depth_write_enable = true;
        writer.depth_compare_op = 7;   // ALWAYS: store 0.25
        writer.depth_read_base = kSamePass; writer.depth_write_base = kSamePass;
        const ResolvedPipelineState reader = bounds_only(kSamePass, min, max);
        prosper::test::BackendDraw w, r;
        w.vs = shaders().vert_z; w.fs = shaders().red; w.ps = &writer; w.vcount = 3;
        r.vs = shaders().vert; r.fs = shaders().red; r.ps = &reader; r.vcount = 3;
        const auto out = render({w, r});
        ASSERT_EQ(out.size(), size_t(W) * H * 4);
        EXPECT_EQ(red_pixels(out), want) << "bounds [" << min << ", " << max
                                         << "] against 0.25 written earlier in the same pass";
    }
    forget(kSamePass);
}

TEST(DepthBoundsRender, BoundsOnlyDrawSamplesItsOwnDepthPlane) {
    // UE4's CSM projection and light volumes sample SceneDepth while bounds-testing the same
    // surface, which now makes it the pass's attachment as well (#1186's read-only-depth path).
    ASSERT_FALSE(shaders().vert_z.empty() || shaders().sample.empty());
    if (!device_has_depth_bounds()) GTEST_SKIP() << "device lacks depthBounds";
    constexpr uint64_t kSelf = 0x20d6300000ull;
    forget(kSelf);
    ResolvedPipelineState producer{};
    producer.topology = 3; producer.color_write_mask = 0xF;
    producer.depth_test_enable = true; producer.depth_write_enable = true;
    producer.depth_compare_op = 7;
    producer.depth_read_base = kSelf; producer.depth_write_base = kSelf;
    prosper::test::BackendDraw pw;
    pw.vs = shaders().vert_z; pw.fs = shaders().red; pw.ps = &producer; pw.vcount = 3;
    ASSERT_EQ(render({pw}).size(), size_t(W) * H * 4);
    ASSERT_TRUE(depth_valid(kSelf));

    prosper::test::FrameResource own_depth;
    own_depth.binding = 4; own_depth.set = 1;
    own_depth.persistent_depth_target_id = kSelf;
    own_depth.tw = W; own_depth.th = H;
    const ResolvedPipelineState consumer = bounds_only(kSelf, 0.0f, 0.5f);
    prosper::test::BackendDraw draw;
    draw.vs = shaders().vert; draw.fs = shaders().sample; draw.ps = &consumer;
    draw.R = {own_depth}; draw.vcount = 3;
    const auto out = render({draw});
    ASSERT_EQ(out.size(), size_t(W) * H * 4);
    size_t sampled = 0;
    for (size_t i = 0; i < out.size(); i += 4) sampled += out[i] > 56 && out[i] < 72;
    std::printf("  self-sampling bounds-only draw: %zu of %u pixels read ~0.25\n", sampled, W * H);
    EXPECT_EQ(sampled, size_t(W) * H)
        << "every fragment passes [0, 0.5] and must read the retained 0.25 through its own plane";
    forget(kSelf);
}
