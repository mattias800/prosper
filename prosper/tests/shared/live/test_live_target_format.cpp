// test_live_target_format — every LiveTargetPixelFormat must survive the compute write-back
// notifier, not just the two that were mapped by hand.
//
// The failure this guards: a compute dispatch borrows a renderer-owned color target, writes it in
// place, and notifies the renderer. The notifier has to name the write's VkFormat so the
// mirror-identity check can prove that THIS image received THAT result. While that name came from a
// two-way ternary, an R11G11B10_FLOAT write was reported as VK_FORMAT_R8G8B8A8_UNORM, the identity
// check rejected it, and the notifier returned before re-authorizing the target. The ordinary
// guest-write invalidation for the same dispatch had already invalidated it, so the renderer's
// pixels were lost and every later consumer sampled guest bytes prosper never wrote — the whole 3D
// layer of Syberia: Remastered's profile-select menu, and #773 in a different format.
//
// The test drives the real registered notifier and asserts the observable effect
// (restore_persistent_color_target_after_mirrored_write re-authorizing the exact image) for ALL
// THREE enumerators. The RGBA8 and RGBA16F cases pass with or without the fix; the R11G11B10F case
// is the one that fails on the unpatched notifier, so the test proves the format gap rather than
// the general path. No Vulkan device is required: the persistent-target cache is seeded directly,
// exactly as tests/shared/live/test_shared_vulkan_device.cpp already does for the same restore contract.
#include "gpu/execute/gpu_execute.hpp"
#include <gtest/gtest.h>
#include "gpu/capture/gpu_capture.hpp"
#include "fixtures/render_runner.h"
#include "shared/live/live_renderer.hpp"
#include "shared/live/live_target_format.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <memory>
#include <vector>

static int fails = 0;
#define CHECK(cond, msg) EXPECT_TRUE(cond) << (msg)

namespace {

struct FormatCase {
    prosper::gpu::GpuCaptureColorFormat seed_format;
    prosper::gpu::LiveTargetPixelFormat live_format;
    VkFormat vk_format;
    uint32_t bytes_per_pixel;
    const char* label;
};

// Publish one renderer-owned target of `c` at `base`, then notify that a compute dispatch mirrored
// its result into it. Returns whether the notifier re-authorized that exact persistent image.
bool notifier_republishes(const FormatCase& c, uint64_t base, uint32_t width, uint32_t height) {
    auto& cache = prosper::test::persistent_color_target_cache();
    cache.clear();
    const prosper::test::PersistentColorTargetKey key{base, width, height, c.vk_format};
    // A non-null stand-in for a borrowed image, with the renderer's expected layout. Only cache
    // identity is under test; this is not a valid Vulkan handle and no device is needed.
    cache[key].image = reinterpret_cast<VkImage>(uintptr_t{1});
    cache[key].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    cache[key].valid = true;

    // Publish the matching RTT cache entry through the renderer's own seed path, so the identity the
    // notifier compares against is the one the renderer really stores for this format.
    prosper::gpu::GpuCaptureRttSeed seed;
    seed.guest_addr = base;
    seed.width = width;
    seed.height = height;
    seed.format = c.seed_format;
    seed.rgba.assign(static_cast<size_t>(width) * height * c.bytes_per_pixel, 0x40);
    std::string error;
    if (!prosper::gpu::restore_gpu_replay_rtt_seeds({seed}, error)) {
        printf("  [FAIL] %s: could not publish the renderer RTT entry (%s)\n", c.label,
               error.c_str());
        fails++;
        return false;
    }
    // Seeding invalidates the persistent image, exactly as a guest write would. Re-authorization is
    // therefore observable: only the notifier can set it back.
    if (cache.at(key).valid) {
        printf("  [FAIL] %s: seeding did not invalidate the persistent target first\n", c.label);
        fails++;
        return false;
    }

    prosper::gpu::LiveTargetImageWrite write;
    write.gpu_addr = base;
    write.width = width;
    write.height = height;
    write.format = c.live_format;
    prosper::gpu::notify_live_render_target_image_written(write);
    const bool republished = cache.at(key).valid;
    cache.clear();
    return republished;
}

// setenv() does not exist in the MinGW UCRT headers, so this file failed to COMPILE on the
// toolchain docs/platforms/WINDOWS_PORT_HANDOFF.md prescribes -- aborting `cmake --build all` and skipping
// every target after it (#2142's second instance; CI's MSYS2 gcc has the same gap and the job
// never reached this file because the SEH abort came first). Same shape as the helper
// test_apr_registry.cpp:38 already uses; kept local rather than shared because a two-line
// platform shim in a test is cheaper to read where it is used than to chase to a header.
static bool set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

} // namespace

TEST(LiveTargetFormat, Contract) {
    const size_t target_count_limit = prosper::test::persistent_color_target_count_limit();
    CHECK(prosper::test::persistent_color_target_count_ceiling(false) == target_count_limit &&
              prosper::test::persistent_color_target_count_ceiling(true) >= target_count_limit &&
              prosper::test::persistent_color_target_count_ceiling(true) - target_count_limit ==
                  64u,
          "an open submission batch gets bounded color-target admission headroom");

    // The renderer's replay seed writer is the only public way to publish an RTT identity, and it is
    // registered only when this is set at registration time.
    set_test_env("PROSPER_GPU_REPLAY_RTT_SEEDS", "1");
    prosper::frontend::register_live_renderer(std::string(), false);

    // The mapping itself. VK_FORMAT_UNDEFINED would mean "unnamed", which the notifier treats as a
    // decline; every real enumerator must name its exact backend format and round-trip back.
    using prosper::gpu::LiveTargetPixelFormat;
    const LiveTargetPixelFormat all[] = {LiveTargetPixelFormat::Rgba8Unorm,
                                         LiveTargetPixelFormat::Rgba16Float,
                                         LiveTargetPixelFormat::R11G11B10Float,
                                         LiveTargetPixelFormat::R8Unorm,
                                         LiveTargetPixelFormat::R32Uint,
                                         LiveTargetPixelFormat::R32Float,
                                         LiveTargetPixelFormat::Rg8Unorm,
                                         LiveTargetPixelFormat::Rgba32Float,
                                         LiveTargetPixelFormat::Rg16Float,
                                         LiveTargetPixelFormat::R16Float};
    bool mapped = true, round_trips = true;
    for (LiveTargetPixelFormat format : all) {
        const VkFormat vk = prosper::frontend::live_target_pixel_format_vk(format);
        mapped = mapped && vk != VK_FORMAT_UNDEFINED &&
                 prosper::frontend::live_target_pixel_format_bytes(format) ==
                     prosper::test::backend_color_bytes_per_pixel(vk) &&
                 prosper::test::backend_color_format(vk) == vk;
        LiveTargetPixelFormat back = LiveTargetPixelFormat::Rgba8Unorm;
        round_trips = round_trips &&
                      prosper::frontend::live_target_pixel_format_from_vk(vk, back) &&
                      back == format;
    }
    CHECK(mapped, "every LiveTargetPixelFormat names a backend format with a matching texel width");
    CHECK(round_trips, "every LiveTargetPixelFormat round-trips through its backend format");
    CHECK(prosper::frontend::live_target_pixel_format_vk(
              LiveTargetPixelFormat::R11G11B10Float) == VK_FORMAT_B10G11R11_UFLOAT_PACK32,
          "R11G11B10Float maps to VK_FORMAT_B10G11R11_UFLOAT_PACK32, not RGBA8");
    CHECK(prosper::frontend::renderer_mip_target_requires_submit_retention(
              VK_FORMAT_B10G11R11_UFLOAT_PACK32) &&
              !prosper::frontend::renderer_mip_target_requires_submit_retention(
                  VK_FORMAT_R8G8B8A8_UNORM) &&
              !prosper::frontend::renderer_mip_target_requires_submit_retention(
                  VK_FORMAT_R16G16B16A16_SFLOAT),
          "submit-lifetime mip retention selects packed HDR targets only");

    // The load-bearing assertions: the registered notifier, for each format the importer accepts.
    const FormatCase cases[] = {
        {prosper::gpu::GpuCaptureColorFormat::Rgba8Unorm,
         LiveTargetPixelFormat::Rgba8Unorm, VK_FORMAT_R8G8B8A8_UNORM, 4, "rgba8"},
        {prosper::gpu::GpuCaptureColorFormat::Rgba16Float,
         LiveTargetPixelFormat::Rgba16Float, VK_FORMAT_R16G16B16A16_SFLOAT, 8, "rgba16f"},
        {prosper::gpu::GpuCaptureColorFormat::R11G11B10Float,
         LiveTargetPixelFormat::R11G11B10Float, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4,
         "r11g11b10"},
        {prosper::gpu::GpuCaptureColorFormat::R8Unorm,
         LiveTargetPixelFormat::R8Unorm, VK_FORMAT_R8_UNORM, 1, "r8"},
        {prosper::gpu::GpuCaptureColorFormat::R32Uint,
         LiveTargetPixelFormat::R32Uint, VK_FORMAT_R32_UINT, 4, "r32ui"},
        {prosper::gpu::GpuCaptureColorFormat::R32Float,
         LiveTargetPixelFormat::R32Float, VK_FORMAT_R32_SFLOAT, 4, "r32f"},
        {prosper::gpu::GpuCaptureColorFormat::Rg8Unorm,
         LiveTargetPixelFormat::Rg8Unorm, VK_FORMAT_R8G8_UNORM, 2, "rg8"},
        {prosper::gpu::GpuCaptureColorFormat::Rgba32Float,
         LiveTargetPixelFormat::Rgba32Float, VK_FORMAT_R32G32B32A32_SFLOAT, 16, "rgba32f"},
        {prosper::gpu::GpuCaptureColorFormat::Rg16Float,
         LiveTargetPixelFormat::Rg16Float, VK_FORMAT_R16G16_SFLOAT, 4, "rg16f"},
        {prosper::gpu::GpuCaptureColorFormat::R16Float,
         LiveTargetPixelFormat::R16Float, VK_FORMAT_R16_SFLOAT, 2, "r16f"},
    };
    uint64_t base = 0x2110310000ull;
    for (const FormatCase& c : cases) {
        const bool republished = notifier_republishes(c, base, 64, 32);
        base += 0x100000ull;
        std::string message = "mirrored compute write to a ";
        message += c.label;
        message += " live target re-authorizes the renderer's image";
        CHECK(republished, message.c_str());
    }

    // A write whose identity does NOT match the published target must still be rejected: the fix
    // widens the set of nameable formats, it must not weaken the identity proof itself.
    const FormatCase& packed = cases[2];
    {
        auto& cache = prosper::test::persistent_color_target_cache();
        cache.clear();
        const uint64_t mismatch_base = base;
        const prosper::test::PersistentColorTargetKey key{
            mismatch_base, 64, 32, packed.vk_format};
        cache[key].image = reinterpret_cast<VkImage>(uintptr_t{1});
        cache[key].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cache[key].valid = true;
        prosper::gpu::GpuCaptureRttSeed seed;
        seed.guest_addr = mismatch_base;
        seed.width = 64;
        seed.height = 32;
        seed.format = packed.seed_format;
        seed.rgba.assign(64u * 32u * packed.bytes_per_pixel, 0x40);
        std::string error;
        CHECK(prosper::gpu::restore_gpu_replay_rtt_seeds({seed}, error),
              "publishing the packed target for the negative case succeeds");
        prosper::gpu::LiveTargetImageWrite write;
        write.gpu_addr = mismatch_base;
        write.width = 64;
        write.height = 16;             // a different shape at the same base
        write.format = packed.live_format;
        prosper::gpu::notify_live_render_target_image_written(write);
        CHECK(!cache.at(key).valid,
              "a mirrored write with a different extent does not re-authorize the target");
        cache.clear();
    }

    // ---- #3727: a NON-MIRRORED compute result must not lose to a stale device copy -----------
    //
    // A CPU-only result must replace the old pixels without authorizing the stale device image.
    // Both titles can have such an image after a graphics pass, even if it was absent on frame one.
    // These are metadata and CPU-consumer assertions: the synthetic handle stores no real pixels.
    // The capture reader exercises the authority decision; it does not execute a graphics LOAD.
    {
        printf("  -- non-mirrored publication (#3727) --\n");
        auto& cache = prosper::test::persistent_color_target_cache();
        cache.clear();
        const uint64_t base = 0x5150000000ull;
        const uint32_t w = 64, h = 32;
        const prosper::test::PersistentColorTargetKey key{base, w, h, VK_FORMAT_R8G8B8A8_UNORM};
        cache[key].image = reinterpret_cast<VkImage>(uintptr_t{1});
        cache[key].layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cache[key].valid = true;

        // Publish the previous CPU snapshot (colour A) and invalidate the stand-in device image.
        prosper::gpu::GpuCaptureRttSeed seed;
        seed.guest_addr = base;
        seed.width = w;
        seed.height = h;
        seed.format = prosper::gpu::GpuCaptureColorFormat::Rgba8Unorm;
        seed.rgba.assign(static_cast<size_t>(w) * h * 4u, 0x11);
        std::string error;
        CHECK(prosper::gpu::restore_gpu_replay_rtt_seeds({seed}, error),
              "an existing renderer target is published (colour A)");
        CHECK(!cache.at(key).valid, "...and seeding invalidated its device image, as a guest write would");

        // Compute publishes a distinct byte pattern without mirroring it into that image.
        std::vector<uint8_t> expected(static_cast<size_t>(w) * h * 4u);
        for (size_t i = 0; i < expected.size(); ++i)
            expected[i] = static_cast<uint8_t>(0x99u + 37u * i + i / 256u);
        auto pixels = std::make_shared<const std::vector<uint8_t>>(expected);
        prosper::gpu::LiveTargetImageWrite write;
        write.gpu_addr = base;
        write.width = w;
        write.height = h;
        write.format = prosper::gpu::LiveTargetPixelFormat::Rgba8Unorm;
        write.linear_pixels = pixels;
        prosper::gpu::notify_live_render_target_image_written(write);

        const bool stale_reauthorized = cache.at(key).valid;
        CHECK(!stale_reauthorized,
              "a non-mirrored result does NOT re-authorize the stale device image");

        // The pixel read is GUARDED on that first assertion, and the guard is the point rather than
        // caution. On the unfixed notifier the device copy is marked authoritative, so the consumer
        // reads back through the image handle -- which in this harness is a synthetic value, so the
        // process dies with SIGSEGV instead of reporting anything. A crash is still a red arm under
        // ctest, but it names nothing and looks like a broken test rather than a found defect. In
        // production the handle is real and the same routing returns STALE PIXELS silently, which is
        // the actual defect; the crash here is an artifact of the synthetic handle, not the bug.
        if (!stale_reauthorized) {
            prosper::gpu::GpuCaptureRttSeed observed;
            const bool read_ok = prosper::gpu::read_gpu_capture_rtt_seed(base, observed, error);
            CHECK(read_ok, "the published target is readable by a consumer");
            CHECK(read_ok && observed.guest_addr == base && observed.width == w &&
                  observed.height == h &&
                  observed.format == prosper::gpu::GpuCaptureColorFormat::Rgba8Unorm,
                  "...with the published address, extent and pixel format");
            CHECK(read_ok && observed.rgba == expected,
                  "...and the complete published payload reaches the consumer unchanged");
        } else {
            printf("  [skip] pixel read-back skipped: the device copy was re-authorized, so the "
                   "consumer would dereference this harness's synthetic image handle\n");
        }
        cache.clear();
    }

    EXPECT_EQ(fails, 0);
}

// #4291: a CB_COLOR ALT (BGRA) target that the renderer keeps as canonical RGBA8.
TEST(LiveTargetComponentOrder, OnlyBgraGuestOverRgbaHostIsSwapped) {
    using namespace prosper::frontend;
    EXPECT_TRUE(live_target_component_order_bgra(VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM));
    EXPECT_TRUE(live_target_component_order_bgra(VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM));
    EXPECT_FALSE(live_target_component_order_bgra(VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM));
    EXPECT_FALSE(live_target_component_order_bgra(VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM))
        << "a host image that kept BGRA needs no translation";
    EXPECT_FALSE(live_target_component_order_bgra(VK_FORMAT_UNDEFINED, VK_FORMAT_R8G8B8A8_UNORM));
}

TEST(LiveTargetComponentOrder, HostSelectorSwapsOnlyXAndZ) {
    using namespace prosper::frontend;
    // A BGRA descriptor's usual (Z,Y,X,W) becomes identity over the canonical image.
    EXPECT_EQ(live_target_host_selector(6, true), 4u);
    EXPECT_EQ(live_target_host_selector(5, true), 5u);
    EXPECT_EQ(live_target_host_selector(4, true), 6u);
    EXPECT_EQ(live_target_host_selector(7, true), 7u);
    EXPECT_EQ(live_target_host_selector(0, true), 0u);
    EXPECT_EQ(live_target_host_selector(1, true), 1u);
    for (uint32_t s = 0; s < 8; ++s) EXPECT_EQ(live_target_host_selector(s, false), s);
}

TEST(LiveTargetComponentOrder, SwapExchangesRedAndBlueOfEveryTexel) {
    std::vector<uint8_t> pixels{1, 2, 3, 4, 5, 6, 7, 8, 9};
    prosper::frontend::swap_rgba8_red_blue(pixels);
    EXPECT_EQ(pixels, (std::vector<uint8_t>{3, 2, 1, 4, 7, 6, 5, 8, 9}))
        << "G and A stay; a trailing partial texel is untouched";
}

// The graphics sampled path keeps its own copy of the composition. Pin the two to one answer, so
// a change to either cannot leave compute and graphics reading a BGRA target differently.
TEST(LiveTargetComponentOrder, GraphicsAndComputeCompositionsAgree) {
    for (VkFormat guest : {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM}) {
        for (uint32_t s = 0; s < 8; ++s) {
            if (s == 2 || s == 3) continue;
            prosper::test::FrameResource resource{};
            resource.render_target_guest_format = guest;
            resource.swizzle[0] = s; resource.swizzle[1] = 5; resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
            const auto graphics = prosper::test::backend_sampled_component_swizzle(resource);
            const bool bgra = prosper::frontend::live_target_component_order_bgra(
                guest, prosper::test::backend_color_format(guest));
            EXPECT_EQ(graphics[0], prosper::frontend::live_target_host_selector(s, bgra))
                << "guest format " << guest << " selector " << s;
        }
    }
}

// The composed decision the sampled-descriptor path asks (#4197): the AC Black Flag shape is a
// 1920x1080 RGBA8 view over a cached 1920x1080 R16_FLOAT target.
TEST(LiveTargetFormat, CachedTargetServesAViewOnlyWhenExtentAndTexelFootprintFit) {
    using prosper::frontend::live_rtt_serves_sampled_view;
    using prosper::frontend::live_target_vk_format_bytes;
    EXPECT_TRUE(live_target_vk_format_bytes(VK_FORMAT_R16_SFLOAT) == 2u);
    EXPECT_TRUE(live_target_vk_format_bytes(VK_FORMAT_R8G8B8A8_UNORM) == 4u);
    EXPECT_TRUE(live_target_vk_format_bytes(VK_FORMAT_UNDEFINED) == 0u);
    // Red-without-fix arm: the pre-fix rule (extent only) admitted this and served half-max texels.
    EXPECT_TRUE(prosper::frontend::rtt_sampled_extent_compatible(1920, 1080, 1920, 1080, 1, false));
    EXPECT_FALSE(live_rtt_serves_sampled_view(1920, 1080, 1920, 1080, 1, false,
                                        VK_FORMAT_R16_SFLOAT, 0u, /*RGBA8 view*/ 4u));
    // The same descriptor over a target that does hold four bytes per texel is served.
    EXPECT_TRUE(live_rtt_serves_sampled_view(1920, 1080, 1920, 1080, 1, false,
                                       VK_FORMAT_R8G8B8A8_UNORM, 0u, 4u));
    // A matching R16F view of the R16F target is served; a wrong extent still refuses.
    EXPECT_TRUE(live_rtt_serves_sampled_view(1920, 1080, 1920, 1080, 1, false,
                                       VK_FORMAT_R16_SFLOAT, 0u, 2u));
    EXPECT_FALSE(live_rtt_serves_sampled_view(1216, 684, 960, 540, 1, false,
                                        VK_FORMAT_R16_SFLOAT, 0u, 2u));
    // A volume and an unknown cached format keep their own contracts.
    EXPECT_TRUE(live_rtt_serves_sampled_view(32, 32, 32, 32, 1, false, VK_FORMAT_R16_SFLOAT, 32u, 4u));
    EXPECT_TRUE(live_rtt_serves_sampled_view(1920, 1080, 1920, 1080, 1, false,
                                       VK_FORMAT_UNDEFINED, 0u, 16u));
}
