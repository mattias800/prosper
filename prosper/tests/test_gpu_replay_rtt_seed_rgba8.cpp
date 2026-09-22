// RGBA8 picture of a render-target seed (tools/gpu_replay/rtt_seed_rgba8.hpp), #3765.
//
// The defect: the conversion handled five of GpuCaptureColorFormat's ten members and returned an
// empty vector for R32Float, Rgba32Float, Rg16Float, R16Float and Rg8Unorm, so
// `--output-target-after` armed on such a target and then failed with "conversion unavailable".
// The first arm below is the direct regression -- every member must produce texels*4 bytes -- and
// it is built from a hand-written seed per format, not from anything the executor produced.
// The others pin the channel conventions and prove the five formats that already worked still
// produce the same bytes they did.
#include "../tools/gpu_replay/rtt_seed_rgba8.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace prosper::gpu::replay_tool;
using F = prosper::gpu::GpuCaptureColorFormat;

static int failures = 0;

static void check(const char* what, bool ok) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static prosper::gpu::GpuCaptureRttSeed seed_of(F format, uint32_t w, uint32_t h,
                                                std::vector<uint8_t> payload) {
    prosper::gpu::GpuCaptureRttSeed seed;
    seed.width = w;
    seed.height = h;
    seed.format = format;
    seed.rgba = std::move(payload);
    return seed;
}

template <typename T>
static std::vector<uint8_t> bytes_of(const std::vector<T>& values) {
    std::vector<uint8_t> out(values.size() * sizeof(T));
    std::memcpy(out.data(), values.data(), out.size());
    return out;
}

static bool pixel_is(const std::vector<uint8_t>& rgba, size_t texel,
                     uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    return rgba.size() >= texel * 4 + 4 && rgba[texel * 4] == r && rgba[texel * 4 + 1] == g &&
           rgba[texel * 4 + 2] == b && rgba[texel * 4 + 3] == a;
}

int main() {
    std::printf("gpu_replay RTT seed -> RGBA8\n");

    const F all[] = {F::Rgba8Unorm, F::Rgba16Float, F::R11G11B10Float, F::R8Unorm, F::R32Uint,
                     F::R32Float, F::Rg8Unorm, F::Rgba32Float, F::Rg16Float, F::R16Float};

    // 1. THE REGRESSION: every member converts. A 3x2 extent so a stride bug cannot hide.
    {
        bool every = true;
        for (F f : all) {
            const uint32_t bytes = rtt_seed_texel_bytes(f);
            const auto rgba = rtt_seed_to_rgba8(seed_of(f, 3, 2, std::vector<uint8_t>(6 * bytes, 0)));
            if (!bytes || rgba.size() != 6 * 4) {
                std::printf("    format %u (%s): %zu bytes\n", static_cast<unsigned>(f),
                            rtt_seed_format_name(f), rgba.size());
                every = false;
            }
        }
        check("every GpuCaptureColorFormat member converts to texels*4 bytes", every);
    }

    // 2. The five formats that used to be dropped, with values that pin their channel layout.
    {
        const auto r32f = rtt_seed_to_rgba8(seed_of(F::R32Float, 2, 1, bytes_of<float>({0.5f, 2.0f})));
        check("R32Float replicates into gray and clamps above 1",
              pixel_is(r32f, 0, 128, 128, 128, 255) && pixel_is(r32f, 1, 255, 255, 255, 255));

        const auto rgba32f = rtt_seed_to_rgba8(
            seed_of(F::Rgba32Float, 1, 1, bytes_of<float>({0.25f, 1.0f, -3.0f, 0.5f})));
        check("Rgba32Float converts each channel at its own offset, alpha included",
              pixel_is(rgba32f, 0, 64, 255, 0, 128));

        // binary16: 0x3800 = 0.5, 0x3C00 = 1.0, 0x7C00 = +inf.
        const auto rg16f = rtt_seed_to_rgba8(
            seed_of(F::Rg16Float, 2, 1, bytes_of<uint16_t>({0x3800, 0x3C00, 0x3C00, 0x3800})));
        check("Rg16Float writes R and G, B = 0, A = 255",
              pixel_is(rg16f, 0, 128, 255, 0, 255) && pixel_is(rg16f, 1, 255, 128, 0, 255));

        const auto r16f = rtt_seed_to_rgba8(seed_of(F::R16Float, 2, 1, bytes_of<uint16_t>({0x3800, 0x0000})));
        check("R16Float replicates into gray",
              pixel_is(r16f, 0, 128, 128, 128, 255) && pixel_is(r16f, 1, 0, 0, 0, 255));

        const auto rg8 = rtt_seed_to_rgba8(seed_of(F::Rg8Unorm, 2, 1, {10, 200, 255, 0}));
        check("Rg8Unorm passes R and G through, B = 0, A = 255",
              pixel_is(rg8, 0, 10, 200, 0, 255) && pixel_is(rg8, 1, 255, 0, 0, 255));
    }

    // 3. The five that already worked keep their exact bytes (the conversion moved to a header).
    {
        const std::vector<uint8_t> rgba8 = {1, 2, 3, 4, 5, 6, 7, 8};
        check("Rgba8Unorm is passed through verbatim",
              rtt_seed_to_rgba8(seed_of(F::Rgba8Unorm, 2, 1, rgba8)) == rgba8);
        // 1.0 / 2.0 / 0.5 packed R-low, G-middle, B-high (the convention native_texels shares).
        const auto packed = rtt_seed_to_rgba8(seed_of(F::R11G11B10Float, 1, 1, bytes_of<uint32_t>({0x702003c0u})));
        check("R11G11B10Float keeps its channel order and clamp", pixel_is(packed, 0, 255, 255, 128, 255));
        const auto r8 = rtt_seed_to_rgba8(seed_of(F::R8Unorm, 1, 1, {77}));
        check("R8Unorm replicates into gray", pixel_is(r8, 0, 77, 77, 77, 255));
        const auto r32ui = rtt_seed_to_rgba8(seed_of(F::R32Uint, 2, 1, bytes_of<uint32_t>({300u, 7u})));
        check("R32Uint saturates at 255", pixel_is(r32ui, 0, 255, 255, 255, 255) &&
              pixel_is(r32ui, 1, 7, 7, 7, 255));
        // The long-standing rule: every non-finite value -- +inf included -- maps to 0.
        const auto rgba16f = rtt_seed_to_rgba8(
            seed_of(F::Rgba16Float, 1, 1, bytes_of<uint16_t>({0x3800, 0x7C00, 0x7E00, 0x3C00})));
        check("Rgba16Float keeps the non-finite -> 0 rule", pixel_is(rgba16f, 0, 128, 0, 0, 255));
    }

    // 4. Never a wrong-sized image: a payload that disagrees with its extent is refused, per format.
    {
        bool every = true;
        for (F f : all) {
            const uint32_t bytes = rtt_seed_texel_bytes(f);
            every = every && rtt_seed_to_rgba8(seed_of(f, 2, 2, std::vector<uint8_t>(4 * bytes - 1, 0))).empty();
        }
        check("a short payload is refused for every format", every);
        const auto bogus = static_cast<F>(200);
        check("an out-of-range format value is refused",
              rtt_seed_to_rgba8(seed_of(bogus, 1, 1, {0, 0, 0, 0})).empty() &&
              rtt_seed_texel_bytes(bogus) == 0);
        check("...and is named unknown", std::string(rtt_seed_format_name(bogus)) == "unknown");
    }

    // 5. Log names are distinct: the old ternary printed "rgba8" for every format it did not list.
    {
        std::set<std::string> names;
        for (F f : all) names.insert(rtt_seed_format_name(f));
        check("every member has its own log name", names.size() == sizeof(all) / sizeof(all[0]) &&
              !names.count("unknown"));
    }

    std::printf("%s\n", failures ? "FAILURES PRESENT" : "all passed");
    return failures ? 1 : 0;
}
