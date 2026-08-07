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
// exactly as tests/test_shared_vulkan_device.cpp already does for the same restore contract.
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/gpu_capture.hpp"
#include "render_runner.h"
#include "../frontends/shared/live_renderer.hpp"
#include "../frontends/shared/live_target_format.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else         { printf("  [ok]   %s\n", msg); } } while (0)

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
    // A borrowed image the compute backend could have written: a real handle value and a layout the
    // renderer actually left it in. Only the identity is under test, so no device is needed.
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
// toolchain docs/WINDOWS_PORT_HANDOFF.md prescribes -- aborting `cmake --build all` and skipping
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

int main() {
    // The renderer's replay seed writer is the only public way to publish an RTT identity, and it is
    // registered only when this is set at registration time.
    set_test_env("PROSPER_GPU_REPLAY_RTT_SEEDS", "1");
    prosper::frontend::register_live_renderer(std::string(), false);

    // The mapping itself. VK_FORMAT_UNDEFINED would mean "unnamed", which the notifier treats as a
    // decline; every real enumerator must name its exact backend format and round-trip back.
    using prosper::gpu::LiveTargetPixelFormat;
    const LiveTargetPixelFormat all[] = {LiveTargetPixelFormat::Rgba8Unorm,
                                         LiveTargetPixelFormat::Rgba16Float,
                                         LiveTargetPixelFormat::R11G11B10Float};
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

    // The load-bearing assertions: the registered notifier, for each format the importer accepts.
    const FormatCase cases[] = {
        {prosper::gpu::GpuCaptureColorFormat::Rgba8Unorm,
         LiveTargetPixelFormat::Rgba8Unorm, VK_FORMAT_R8G8B8A8_UNORM, 4, "rgba8"},
        {prosper::gpu::GpuCaptureColorFormat::Rgba16Float,
         LiveTargetPixelFormat::Rgba16Float, VK_FORMAT_R16G16B16A16_SFLOAT, 8, "rgba16f"},
        {prosper::gpu::GpuCaptureColorFormat::R11G11B10Float,
         LiveTargetPixelFormat::R11G11B10Float, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4,
         "r11g11b10"},
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

    printf(fails ? "test_live_target_format: %d FAILURE(S)\n"
                 : "test_live_target_format: all ok\n", fails);
    return fails ? 1 : 0;
}
