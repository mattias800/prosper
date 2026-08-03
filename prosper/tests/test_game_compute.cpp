#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "../src/gpu/tile.hpp"
#include "../src/host/guest_write_watch.hpp"
#include "live_compute.hpp"
#include "seed_reprove.hpp"
#include "test_scratch.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#if defined(__linux__)
#include <sys/mman.h>
#endif

using namespace prosper::gpu;

static int fails = 0;
// #1690: this binary was excluded from CI for months, and the cost was argued from a *static* grep
// of CHECK sites because nothing reported how many actually ran. Count the executed ones so the
// coverage this registration contributes is a measured number in the run's own log, and so a gate
// that quietly stops executing assertions shows up as a drop rather than as unchanged green.
static int checks = 0;
#define CHECK(c, msg) do { ++checks; if (!(c)) { std::printf("FAIL: %s\n", msg); fails++; } } while (0)

static int file_descriptor(FILE* file) {
#ifdef _WIN32
    return _fileno(file);
#else
    return fileno(file);
#endif
}

static int duplicate_descriptor(int fd) {
#ifdef _WIN32
    return _dup(fd);
#else
    return dup(fd);
#endif
}

static int replace_descriptor(int from, int to) {
#ifdef _WIN32
    return _dup2(from, to);
#else
    return dup2(from, to);
#endif
}

static void close_descriptor(int fd) {
#ifdef _WIN32
    _close(fd);
#else
    close(fd);
#endif
}

int main() {
#ifdef _WIN32
    _putenv_s("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB", "0");
    _putenv_s("PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB", "1");
#else
    setenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB", "0", 1);
    setenv("PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB", "1", 1);
#endif
    const bool adaptive_storage_result_validation_enabled =
        std::getenv("PROSPER_NO_ADAPTIVE_STORAGE_RESULT_VALIDATION") == nullptr;
    const bool native_2d_compute_transfer_enabled =
        std::getenv("PROSPER_NO_NATIVE_2D_COMPUTE_TRANSFER") == nullptr;
    const bool native_2d_compute_transfer_available =
        native_2d_compute_transfer_enabled &&
        adaptive_storage_result_validation_enabled;
    const bool cold_storage_snapshot_deferral_enabled =
        adaptive_storage_result_validation_enabled;
    using prosper::frontend::ComputeImageCacheClass;
    CHECK(prosper::frontend::compute_image_cache_default_minimum_bytes(
              ComputeImageCacheClass::sampled) == 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_minimum_bytes(
              ComputeImageCacheClass::storage) == 4ull * 1024ull,
          "sampled and storage images retain independent default cache crossovers");
    CHECK(!prosper::frontend::compute_image_cache_default_eligible(
              1024ull * 1024ull - 1, ComputeImageCacheClass::sampled) &&
          prosper::frontend::compute_image_cache_default_eligible(
              1024ull * 1024ull, ComputeImageCacheClass::sampled) &&
          !prosper::frontend::compute_image_cache_default_eligible(
              4ull * 1024ull - 1, ComputeImageCacheClass::storage) &&
          prosper::frontend::compute_image_cache_default_eligible(
              4ull * 1024ull, ComputeImageCacheClass::storage),
          "image residency policy includes each crossover exactly without caching smaller inputs");
    CHECK(prosper::frontend::compute_image_cache_default_limit_bytes(0) ==
              512ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(
              4ull * 1024ull * 1024ull * 1024ull) == 512ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(
              8ull * 1024ull * 1024ull * 1024ull) == 1024ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(
              16ull * 1024ull * 1024ull * 1024ull) == 2048ull * 1024ull * 1024ull &&
          prosper::frontend::compute_image_cache_default_limit_bytes(UINT64_MAX) ==
              2048ull * 1024ull * 1024ull,
          "image cache limit scales with local memory and retains bounded floor and ceiling");
    CHECK(prosper::frontend::compute_sampled_guest_prepare_required(false, false, false),
          "guest-backed sampled image prepares its source");
    CHECK(!prosper::frontend::compute_sampled_guest_prepare_required(false, true, false),
          "renderer-owned sampled image uses renderer authority instead of guest backing");
    CHECK(!prosper::frontend::compute_sampled_guest_prepare_required(false, false, true),
          "directly imported depth image does not validate unused guest backing");
    CHECK(prosper::frontend::compute_sampled_guest_prepare_required(
              false, false, true, true),
          "imported-image recovery switch restores guest validation");
    CHECK(!prosper::frontend::compute_sampled_guest_prepare_required(true, false, false),
          "storage image retains its independent seed/writeback path");
    CHECK(prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 27, false, false),
          "exact-width tiled storage can feed mapped bytes directly to the tiler");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              false, 27, false, false),
          "converted storage retains its mutable packed buffer");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 0, false, false),
          "linear guest storage still copies mapped bytes into guest memory");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 27, true, false),
          "poison proving retains a mutable copy for untouched-texel restoration");
    CHECK(!prosper::frontend::storage_writeback_can_tile_mapped_bytes(
              true, 27, false, true),
          "the recovery switch restores the copied writeback path");

    bool half_luts_match = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        const float value = half_to_float(half);
        uint32_t float_bits = 0;
        std::memcpy(&float_bits, &value, sizeof(float_bits));
        half_luts_match &= prosper::frontend::storage_unpack_float16_bits(half) == float_bits;

        float normalized = value;
        if (std::isnan(normalized)) normalized = 0.0f;
        normalized = normalized < 0.0f ? 0.0f : (normalized > 1.0f ? 1.0f : normalized);
        const uint8_t reference = static_cast<uint8_t>(std::lround(normalized * 255.0f));
        half_luts_match &= prosper::frontend::sampled_float16_to_unorm8(half) == reference;
    }
    CHECK(half_luts_match,
          "binary16 lookup conversions are exhaustive bit-exact matches for storage and sampling");

    std::vector<uint8_t> half_source(65536u * sizeof(uint16_t));
    std::vector<uint8_t> half_rgba(65536u * 4u);
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        std::memcpy(half_source.data() + bits * sizeof(half), &half, sizeof(half));
    }
    prosper::frontend::sampled_float16_to_unorm8_range(
        half_source.data(), 1, 65536u, half_rgba.data());
    bool half_range_matches = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        half_range_matches &=
            half_rgba[bits * 4] == prosper::frontend::sampled_float16_to_unorm8(
                                       static_cast<uint16_t>(bits)) &&
            half_rgba[bits * 4 + 1] == 0 && half_rgba[bits * 4 + 2] == 0 &&
            half_rgba[bits * 4 + 3] == 255;
    }
    CHECK(half_range_matches,
          "parallel binary16 sampled range matches every scalar value and fills (R,0,0,1)");

    // Exercise the packed RGBA fast path with every binary16 bit pattern. On x86 this runtime-
    // dispatches through F16C/AVX2 when available; elsewhere it proves the identical scalar fallback.
    std::vector<uint8_t> half_source_x4(65536u * sizeof(uint16_t));
    std::vector<uint8_t> half_rgba_x4(65536u);
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits) {
        const uint16_t half = static_cast<uint16_t>(bits);
        std::memcpy(half_source_x4.data() + bits * sizeof(half), &half, sizeof(half));
    }
    prosper::frontend::sampled_float16_to_unorm8_range(
        half_source_x4.data(), 4, 65536u / 4u, half_rgba_x4.data());
    bool half_range_x4_matches = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits)
        half_range_x4_matches &= half_rgba_x4[bits] ==
            prosper::frontend::sampled_float16_to_unorm8(static_cast<uint16_t>(bits));
    CHECK(half_range_x4_matches,
          "packed RGBA binary16 sampled range matches every scalar lookup value");

    // Compute's ordinary guest-backed 2D RGBA16F sampled path uses an RGBA8 staging image. A
    // uniform embedded DCC clear is authoritative over the compressed base allocation, so prove
    // that the narrow fast-clear gate produces exactly the same sampled bytes as an explicitly
    // materialized uncompressed FP16 image. DCC_CLEAR_0001 is especially important: its alpha-one
    // value is absent from a zero-filled compressed base and is consumed by Plucky's composite.
    {
        constexpr uint32_t clear_width = 64;
        constexpr size_t clear_texels = clear_width;
        ShaderResource clear_resource{};
        clear_resource.cls = ResourceClass::Texture;
        clear_resource.format = DataFormat::Float16;
        clear_resource.num_components = 4;
        clear_resource.img_dim = 1;
        clear_resource.width = clear_width;
        clear_resource.height = clear_resource.depth = 1;
        clear_resource.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
        clear_resource.size = static_cast<uint32_t>(
            tiled_surface_bytes(clear_width, 1, clear_resource.tile_mode, 0, 8));
        clear_resource.compression_enabled = true;
        clear_resource.meta_pipe_aligned = true;
        clear_resource.alpha_is_on_msb = true;
        clear_resource.metadata_addr = 0x301758d000ull;
        const size_t clear_metadata_bytes = gpu_capture_dcc_metadata_footprint(clear_resource);
        std::vector<uint8_t> clear_metadata(clear_metadata_bytes, 0x40);
        std::vector<uint8_t> clear_rgba(clear_texels * 4, 0xa5);
        uint8_t clear_code = 0;
        CHECK(clear_metadata_bytes &&
              prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, clear_rgba.data(), clear_texels,
                  clear_metadata.data(), clear_metadata.size(), &clear_code) &&
              clear_code == 0x40,
              "compute recognizes a complete uniform RGBA16F DCC_CLEAR_0001 plane");

        std::vector<uint8_t> explicit_fp16(clear_texels * 8, 0);
        const uint16_t half_one = float_to_half(1.0f);
        for (size_t texel = 0; texel < clear_texels; ++texel)
            std::memcpy(explicit_fp16.data() + texel * 8 + 6,
                        &half_one, sizeof(half_one));
        std::vector<uint8_t> explicit_rgba(clear_texels * 4, 0);
        prosper::frontend::sampled_float16_to_unorm8_range(
            explicit_fp16.data(), 4, clear_texels, explicit_rgba.data());
        CHECK(clear_rgba == explicit_rgba,
              "DCC_CLEAR_0001 matches explicitly materialized FP16 (0,0,0,1)");
        bool clear_channels_match = true;
        for (size_t texel = 0; texel < clear_texels; ++texel)
            clear_channels_match &= clear_rgba[texel * 4 + 0] == 0 &&
                                    clear_rgba[texel * 4 + 1] == 0 &&
                                    clear_rgba[texel * 4 + 2] == 0 &&
                                    clear_rgba[texel * 4 + 3] == 255;
        CHECK(clear_channels_match,
              "MSB alpha placement preserves the DCC alpha-one sampled channel");

        ShaderResource lsb_alpha = clear_resource;
        lsb_alpha.alpha_is_on_msb = false;
        uint8_t lsb_pixel[4]{};
        CHECK(prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  lsb_alpha, true, false, false, lsb_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              std::vector<uint8_t>(lsb_pixel, lsb_pixel + 4) ==
                  std::vector<uint8_t>({255, 0, 0, 0}),
              "LSB alpha placement routes DCC_CLEAR_0001 to component zero");

        std::vector<uint8_t> rejected = clear_metadata;
        rejected.back() = 0x00;
        uint8_t rejected_pixel[4]{};
        CHECK(!prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, rejected_pixel, 1,
                  rejected.data(), rejected.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size() - 1) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, false, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, true, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, true, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()),
              "mixed, incomplete, non-guest, arrayed, and rollback DCC states retain the old path");
        std::fill(rejected.begin(), rejected.end(), 0xff);
        ShaderResource wrong_shape = clear_resource;
        wrong_shape.declared_mip_levels = 2;
        ShaderResource wrong_format = clear_resource;
        wrong_format.format = DataFormat::Unorm8;
        CHECK(!prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  clear_resource, true, false, false, rejected_pixel, 1,
                  rejected.data(), rejected.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  wrong_shape, true, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()) &&
              !prosper::frontend::compute_sampled_dcc_fast_clear_rgba8(
                  wrong_format, true, false, false, rejected_pixel, 1,
                  clear_metadata.data(), clear_metadata.size()),
              "uncompressed metadata, mip chains, and non-FP16 views fail closed");
    }

    std::vector<uint32_t> half_storage_x4(65536u);
    prosper::frontend::storage_unpack_float16x4_range(
        half_source_x4.data(), 65536u / 4u, half_storage_x4.data());
    bool half_storage_x4_matches = true;
    for (uint32_t bits = 0; bits <= 0xffffu; ++bits)
        half_storage_x4_matches &= half_storage_x4[bits] ==
            prosper::frontend::storage_unpack_float16_bits(static_cast<uint16_t>(bits));
    CHECK(half_storage_x4_matches,
          "packed RGBA16F storage range preserves every scalar float bit pattern");

    // The storage writeback fast path converts two RGBA32F texels per F16C instruction. Compare a
    // million deterministic float bit patterns, including explicit overflow/subnormal/NaN payload
    // edges, against the established scalar round-to-nearest-even converter.
    constexpr size_t pack_channels_count = 1u << 20;
    std::vector<uint32_t> pack_channels(pack_channels_count);
    const uint32_t pack_edges[] = {
        0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u,
        0x33000000u, 0x33000001u, 0x387fc000u, 0x38800000u,
        0x477fe000u, 0x477ff000u, 0x47800000u, 0xc7800000u,
        0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u,
        0x7fffffffu, 0xff800001u, 0x3f800000u, 0xbf800000u,
    };
    size_t pack_at = 0;
    for (uint32_t bits : pack_edges) pack_channels[pack_at++] = bits;
    uint32_t pack_random = 0x91e10da5u;
    while (pack_at < pack_channels.size()) {
        pack_random ^= pack_random << 13;
        pack_random ^= pack_random >> 17;
        pack_random ^= pack_random << 5;
        pack_channels[pack_at++] = pack_random;
    }
    std::vector<uint8_t> packed_half(pack_channels_count * sizeof(uint16_t));
    prosper::frontend::storage_pack_float16x4_range(
        pack_channels.data(), pack_channels_count / 4, packed_half.data());
    bool storage_half_range_matches = true;
    for (size_t i = 0; i < pack_channels.size(); ++i) {
        float value;
        std::memcpy(&value, &pack_channels[i], sizeof(value));
        const uint16_t expected = prosper::gpu::float_to_half(value);
        uint16_t actual = 0;
        std::memcpy(&actual, packed_half.data() + i * sizeof(actual), sizeof(actual));
        if (actual != expected) {
            std::printf("Float16 pack mismatch index=%zu f32=%08x expected=%04x actual=%04x\n",
                        i, pack_channels[i], expected, actual);
            storage_half_range_matches = false;
            break;
        }
    }
    CHECK(storage_half_range_matches,
          "packed RGBA32F storage range matches scalar Float16 rounding and NaN payloads");

    using prosper::frontend::direct_sampled_rtt_compatible;
    CHECK(direct_sampled_rtt_compatible(DataFormat::Unorm8, 4,
                                        LiveTargetPixelFormat::Rgba8Unorm, false) &&
          direct_sampled_rtt_compatible(DataFormat::Float16, 4,
                                        LiveTargetPixelFormat::Rgba16Float, false) &&
          direct_sampled_rtt_compatible(DataFormat::Float10_11_11, 3,
                                        LiveTargetPixelFormat::R11G11B10Float, false),
          "renderer RTT direct bind accepts exact RGBA8, RGBA16F, and R11G11B10 views");
    CHECK(!direct_sampled_rtt_compatible(DataFormat::Float16, 4,
                                         LiveTargetPixelFormat::Rgba8Unorm, true) &&
          !direct_sampled_rtt_compatible(DataFormat::Unorm8, 4,
                                         LiveTargetPixelFormat::Rgba16Float, true) &&
          !direct_sampled_rtt_compatible(DataFormat::Float16, 2,
                                         LiveTargetPixelFormat::Rgba16Float, true),
          "renderer RTT direct bind rejects format conversion and component aliases");
    CHECK(direct_sampled_rtt_compatible(DataFormat::Unorm16, 4,
                                        LiveTargetPixelFormat::Rgba8Unorm, true) &&
          !direct_sampled_rtt_compatible(DataFormat::Unorm16, 4,
                                         LiveTargetPixelFormat::Rgba8Unorm, false) &&
          !direct_sampled_rtt_compatible(DataFormat::Unorm16, 2,
                                         LiveTargetPixelFormat::Rgba8Unorm, true),
          "float RGBA16-UNORM sampled values may reuse RGBA8 without widening texels");
    // A renderer-owned target is stored canonically as RGBA8 or RGBA16F, while a later compute
    // descriptor can alias it as packed R11G11B10. Reconstruct the descriptor-visible words rather
    // than sampling stale guest backing or dropping the dispatch.
    {
        auto rgba8 = std::make_shared<std::vector<uint8_t>>(std::initializer_list<uint8_t>{
            255, 0, 128, 17, 0, 255, 64, 99,
        });
        LiveTargetSnapshot snapshot8{2, 1, LiveTargetPixelFormat::Rgba8Unorm, rgba8};
        std::vector<uint8_t> packed8(8);
        CHECK(prosper::frontend::pack_live_target_r11g11b10(
                  snapshot8, packed8.data(), packed8.size()),
              "RGBA8 renderer target reconstructs as two packed R11G11B10 texels");
        uint32_t words8[2]{};
        if (packed8.size() == sizeof(words8)) std::memcpy(words8, packed8.data(), sizeof(words8));
        CHECK((words8[0] & 0x7ffu) == float_to_f11(1.0f) &&
              ((words8[0] >> 11) & 0x7ffu) == float_to_f11(0.0f) &&
              ((words8[0] >> 22) & 0x3ffu) == float_to_f10(128.0f / 255.0f),
              "RGBA8 reconstruction quantizes RGB through the unsigned 11/11/10 float contract");

        auto rgba16 = std::make_shared<std::vector<uint8_t>>(8, 0);
        const float values[4] = {4.0f, 0.5f, 2.0f, 1.0f};
        for (uint32_t c = 0; c < 4; ++c) {
            const uint16_t half = float_to_half(values[c]);
            std::memcpy(rgba16->data() + c * 2, &half, sizeof(half));
        }
        LiveTargetSnapshot snapshot16{1, 1, LiveTargetPixelFormat::Rgba16Float, rgba16};
        std::vector<uint8_t> packed16(4);
        CHECK(prosper::frontend::pack_live_target_r11g11b10(
                  snapshot16, packed16.data(), packed16.size()),
              "RGBA16F renderer target reconstructs as packed R11G11B10");
        uint32_t word16 = 0;
        if (packed16.size() == sizeof(word16)) std::memcpy(&word16, packed16.data(), sizeof(word16));
        CHECK((word16 & 0x7ffu) == float_to_f11(4.0f) &&
              ((word16 >> 11) & 0x7ffu) == float_to_f11(0.5f) &&
              ((word16 >> 22) & 0x3ffu) == float_to_f10(2.0f),
              "RGBA16F reconstruction preserves HDR RGB while discarding alpha");

        auto native = std::make_shared<std::vector<uint8_t>>(std::initializer_list<uint8_t>{
            0x78, 0x56, 0x34, 0x12,
        });
        LiveTargetSnapshot native_snapshot{
            1, 1, LiveTargetPixelFormat::R11G11B10Float, native};
        std::vector<uint8_t> native_copy(4);
        CHECK(prosper::frontend::pack_live_target_r11g11b10(
                  native_snapshot, native_copy.data(), native_copy.size()) &&
              native_copy == *native,
              "native R11G11B10 renderer target remains bit-exact through CPU fallback");

        snapshot16.pixels = std::make_shared<std::vector<uint8_t>>(7, 0);
        CHECK(!prosper::frontend::pack_live_target_r11g11b10(
                  snapshot16, packed16.data(), packed16.size()),
              "R11G11B10 reconstruction rejects a malformed renderer snapshot");
    }

    // #1127: the seed-skip re-prove counter (seed_reprove.hpp). interval 0 disables (old prove-once);
    // interval N fires on the Nth fast-skip and resets, so a Full-cached data-dependent shader is
    // re-proven within N fast-skips instead of trusting a stale "covers every texel" verdict forever.
    {
        using prosper::frontend::seed_reprove_due;
        using prosper::frontend::dispatch_has_enough_threads_for_texels;
        CHECK(dispatch_has_enough_threads_for_texels(15360, 135, 1, 1920, 1080, 1),
              "vectorized/swizzled dispatch with one invocation per texel can prove coverage");
        CHECK(!dispatch_has_enough_threads_for_texels(1919, 1080, 1, 1920, 1080, 1),
              "dispatch with fewer total invocations than texels cannot prove coverage");
        CHECK(!dispatch_has_enough_threads_for_texels(0, 1080, 1, 1920, 1080, 1),
              "degenerate dispatch cannot prove coverage");
        uint32_t s = 0;
        CHECK(!seed_reprove_due(s, 0) && !seed_reprove_due(s, 0) && s == 0,
              "interval 0 never re-proves and leaves the counter untouched (prove-once)");
        s = 0;
        bool a = seed_reprove_due(s, 3), b = seed_reprove_due(s, 3), c = seed_reprove_due(s, 3);
        CHECK(!a && !b && c && s == 0, "interval 3 fires on the 3rd fast-skip and resets the counter");
        CHECK(!seed_reprove_due(s, 3) && !seed_reprove_due(s, 3) && seed_reprove_due(s, 3),
              "the re-prove cycle repeats after a reset");
        s = 0;
        CHECK(seed_reprove_due(s, 1) && seed_reprove_due(s, 1),
              "interval 1 re-proves every fast-skip (maximum soundness)");

        // #1127: the interval env-parse must fail SAFE -- garbage/overflow keeps the default rather
        // than silently disabling the safety (which an atol-style parse would, returning 0 = off).
        using prosper::frontend::seed_reprove_interval_from_env;
        CHECK(seed_reprove_interval_from_env(nullptr, 256) == 256, "unset env -> default");
        CHECK(seed_reprove_interval_from_env("", 256) == 256, "empty env -> default");
        CHECK(seed_reprove_interval_from_env("foo", 256) == 256, "non-numeric env -> default (fail-safe, not 0)");
        CHECK(seed_reprove_interval_from_env("256x", 256) == 256, "trailing junk -> default");
        CHECK(seed_reprove_interval_from_env("-4", 256) == 256, "negative -> default");
        CHECK(seed_reprove_interval_from_env("4294967296", 256) == 256, "overflow (2^32) -> default, not truncated to 0");
        CHECK(seed_reprove_interval_from_env("0", 256) == 0, "explicit 0 honored (intentionally disables re-proving)");
        CHECK(seed_reprove_interval_from_env("64", 256) == 64, "exact in-range value overrides default");
        CHECK(seed_reprove_interval_from_env("4294967295", 256) == 4294967295u, "max uint32 accepted");
    }

    // MinGW's lround dominates full-HD storage-image writeback. Prove the bounded integer path is
    // identical to the previous conversion across all half-float values, every UNORM threshold and
    // adjacent float, plus a deterministic million-value float32 sample.
    auto reference_unorm8 = [](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (std::isnan(value)) value = 0.0f;
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<uint8_t>(std::lround(value * 255.0f));
    };
    auto check_unorm8 = [&](float value) {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return prosper::frontend::storage_pack_unorm8(bits) == reference_unorm8(bits);
    };
    bool unorm8_matches = true;
    const float edge_values[] = {
        -std::numeric_limits<float>::infinity(), -1.0f, -0.0f, 0.0f,
        std::numeric_limits<float>::denorm_min(), 1.0f,
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(),
    };
    for (float value : edge_values) unorm8_matches &= check_unorm8(value);
    for (uint32_t i = 0; i < 255; ++i) {
        const float threshold = (static_cast<float>(i) + 0.5f) / 255.0f;
        unorm8_matches &= check_unorm8(std::nextafter(threshold, 0.0f));
        unorm8_matches &= check_unorm8(threshold);
        unorm8_matches &= check_unorm8(std::nextafter(threshold, 1.0f));
    }
    for (uint32_t bits = 0; bits <= 0xffff; ++bits) {
        const float value = prosper::gpu::half_to_float(static_cast<uint16_t>(bits));
        unorm8_matches &= check_unorm8(value);
    }
    uint32_t random_bits = 0x6d2b79f5u;
    for (uint32_t i = 0; i < 1000000; ++i) {
        random_bits ^= random_bits << 13;
        random_bits ^= random_bits >> 17;
        random_bits ^= random_bits << 5;
        if (prosper::frontend::storage_pack_unorm8(random_bits) !=
            reference_unorm8(random_bits)) {
            unorm8_matches = false;
            break;
        }
    }
    CHECK(unorm8_matches, "fast UNORM8 pack is equivalent to the lround reference");

    auto reference_unorm16 = [](uint32_t bits) {
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        if (std::isnan(value)) value = 0.0f;
        value = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
        return static_cast<uint16_t>(std::lround(value * 65535.0f));
    };
    bool unorm16_matches = true;
    for (uint32_t raw = 0; raw <= 0xffffu; ++raw) {
        const float value = raw / 65535.0f;
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        if (prosper::frontend::storage_pack_unorm16(bits) != raw) {
            unorm16_matches = false;
            break;
        }
    }
    for (uint32_t bits : pack_channels) {
        if (prosper::frontend::storage_pack_unorm16(bits) != reference_unorm16(bits)) {
            unorm16_matches = false;
            break;
        }
    }
    CHECK(unorm16_matches,
          "UNORM16 pack preserves all quantized channels and matches lround for float inputs");

    bool storage_unorm8_range_matches = true;
    constexpr size_t unorm_texels = pack_channels_count / 4;
    for (uint32_t components : {1u, 2u, 3u, 4u}) {
        std::vector<uint8_t> packed_unorm8(unorm_texels * components);
        prosper::frontend::storage_pack_unorm8_range(
            pack_channels.data(), components, unorm_texels, packed_unorm8.data());
        for (size_t texel = 0; texel < unorm_texels; ++texel) {
            for (uint32_t channel = 0; channel < components; ++channel) {
                const size_t source = texel * 4 + channel;
                const uint8_t expected =
                    prosper::frontend::storage_pack_unorm8(pack_channels[source]);
                const uint8_t actual = packed_unorm8[texel * components + channel];
                if (actual == expected) continue;
                std::printf("UNORM8 pack mismatch n=%u texel=%zu channel=%u f32=%08x "
                            "expected=%02x actual=%02x\n",
                            components, texel, channel, pack_channels[source], expected, actual);
                storage_unorm8_range_matches = false;
                break;
            }
            if (!storage_unorm8_range_matches) break;
        }
        if (!storage_unorm8_range_matches) break;
    }
    CHECK(storage_unorm8_range_matches,
          "packed one-, two-, three-, and four-channel UNORM8 ranges match scalar clamp and rounding");

    // Dead Cells' bound startup fill kernel, copied verbatim from eboot.elf at runtime address
    // 0x401aec200. It stores s4-s7 to record `(TGID_X << 6) + local_id_x` through the V# in s0-s3.
    static const uint32_t code[] = {
        0xd7460004, 0x04010c08, 0x7e000204, 0x7e020205, 0x7e040206,
        0x7e060207, 0xe01c2000, 0x80000004, 0xbf810000,
    };

    constexpr uint32_t records = 130;
    constexpr uint32_t launched_records = 3 * 64;
    std::vector<uint32_t> result(launched_records * 4, 0xcccccccc);
    ShaderResourceTable rt;
    ShaderResource buffer;
    // Runtime metadata classifies compute direct type-1 V#s as ConstantBuffer even though MUBUF
    // format stores use them; both classes lower to storage buffers.
    buffer.cls = ResourceClass::ConstantBuffer;
    buffer.format = DataFormat::Uint32;
    buffer.num_components = 4;
    buffer.binding = 2;
    buffer.gpu_addr = (uint64_t)(uintptr_t)result.data();
    // Deliberately expose the padded Vulkan lanes as valid writable storage. Only the generated
    // exact-thread guard can keep records 130..191 unchanged; robust buffer access cannot hide a bug.
    buffer.size = launched_records * 4 * sizeof(uint32_t);
    buffer.stride = 4 * sizeof(uint32_t);
    buffer.sgpr_base = 0;
    rt.resources.push_back(buffer);

    ComputeShaderConfig config;
    config.user_sgprs = {
        0, 0, 0, 0,
        0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00,
    };
    config.local_x = 64;
    config.exact_thread_extent = true;
    config.threads_x = records;
    config.threads_y = config.threads_z = 1;
    config.tidig_comp_cnt = 0;
    config.tgid_x_en = true;

    std::vector<uint32_t> spirv = recompile_compute(
        code, sizeof(code) / sizeof(code[0]), &rt, config);
    CHECK(!spirv.empty(), "real Dead Cells compute kernel recompiles");
    if (spirv.empty()) return 1;
    CHECK(classify_compute_cpu_fast_path(code, std::size(code)) ==
              ComputeCpuFastPath::FillSgprUvec4 &&
          classify_compute_cpu_fast_path(code, std::size(code) - 1) ==
              ComputeCpuFastPath::None,
          "buffer-fill CPU fast path requires the complete exact RDNA2 program");
    std::array<uint32_t, std::size(code)> altered_fill_code{};
    std::copy_n(code, std::size(code), altered_fill_code.begin());
    altered_fill_code[6] ^= 1u;
    CHECK(classify_compute_cpu_fast_path(altered_fill_code.data(), altered_fill_code.size()) ==
              ComputeCpuFastPath::None,
          "buffer-fill CPU fast path rejects a one-bit instruction change");
    ComputeShaderConfig alternate_config = config;
    alternate_config.user_sgprs[4] ^= 0xffffffffu;
    alternate_config.user_sgprs[7] ^= 0x13579bdfu;
    CHECK(recompile_compute(code, sizeof(code) / sizeof(code[0]), &rt, alternate_config) == spirv,
          "per-dispatch user SGPR values do not specialize the reusable SPIR-V module");

    auto report = validate_spirv_descriptor_interface(
        spirv, &rt, 0, SpirvShaderStage::Compute);
    for (const auto& issue : report.issues) {
        if (issue.error)
            std::printf("descriptor error: %s binding=%u expected=%s actual=%s\n",
                        descriptor_issue_name(issue.code), issue.binding,
                        spirv_descriptor_kind_name(issue.expected),
                        spirv_descriptor_kind_name(issue.actual));
    }
    CHECK(report.ok(), "real compute descriptor interface validates");

    ComputeItem item;
    item.spirv = spirv;
    item.user_sgprs = config.user_sgprs;
    item.resources = std::make_shared<ShaderResourceTable>(rt);
    item.launch.threads_x = records;
    item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_x = 64;
    item.launch.groups_x = 3;
    item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x401aec200;
    item.dispatch_index = 7;
    item.submit_no = 11;
    item.command_order = 70;
    item.recompile_config = config;
    item.recompile_config_available = true;
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production live backend executes the game kernel");
    CHECK(result.size() == launched_records * 4, "compute resource retains its padded declared size");
    const uint32_t expected[4] = {0x11223344, 0x55667788, 0x99aabbcc, 0xddeeff00};
    bool all_filled = result.size() == launched_records * 4;
    for (uint32_t record = 0; record < records; record++) {
        for (uint32_t component = 0; component < 4; component++) {
            if (result[record * 4 + component] != expected[component]) {
                std::printf("FAIL: record %u component %u = %08x, expected %08x\n",
                            record, component, result[record * 4 + component], expected[component]);
                all_filled = false;
                break;
            }
        }
        if (!all_filled) break;
    }
    CHECK(all_filled, "all 130 records are filled across three workgroups");
    bool padded_lanes_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            padded_lanes_untouched &= result[record * 4 + component] == 0xcccccccc;
    CHECK(padded_lanes_untouched,
          "partial workgroup suppresses all 62 padded invocations without a guest bounds check");

    std::fill(result.begin(), result.end(), 0xddddddddu);
    ComputeItem cpu_fill_item = item;
    cpu_fill_item.cpu_fast_path = ComputeCpuFastPath::FillSgprUvec4;
    uint32_t cpu_fill_write_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
        if (addr == buffer.gpu_addr && size == buffer.size)
            ++cpu_fill_write_notifications;
    });
    const uint64_t cpu_fills_before =
        prosper::frontend::live_compute_cpu_fill_dispatches();
    CHECK(prosper::frontend::execute_live_compute_items({cpu_fill_item}),
          "exact buffer-fill program executes through the CPU fast path");
    set_guest_gpu_write_observer({});
    bool cpu_fill_matches = true;
    for (uint32_t record = 0; record < records && cpu_fill_matches; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            cpu_fill_matches &= result[record * 4 + component] == expected[component];
    bool cpu_fill_padding_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            cpu_fill_padding_untouched &= result[record * 4 + component] == 0xddddddddu;
    CHECK(cpu_fill_matches && cpu_fill_padding_untouched &&
              cpu_fill_write_notifications == 1 &&
              prosper::frontend::live_compute_cpu_fill_dispatches() ==
                  cpu_fills_before + 1,
          "CPU fill matches Vulkan, preserves padded lanes, and invalidates the declared range");

    ShaderResourceTable narrow_stride_rt = rt;
    narrow_stride_rt.resources[0].stride = 8;
    const std::vector<uint32_t> narrow_stride_spirv = recompile_compute(
        code, std::size(code), &narrow_stride_rt, config);
    CHECK(!narrow_stride_spirv.empty(), "alternate-stride fill kernel recompiles");
    if (!narrow_stride_spirv.empty()) {
        std::fill(result.begin(), result.end(), 0xabababab);
        ComputeItem narrow_stride_item = cpu_fill_item;
        narrow_stride_item.spirv = narrow_stride_spirv;
        narrow_stride_item.resources =
            std::make_shared<ShaderResourceTable>(narrow_stride_rt);
        const uint64_t narrow_stride_fills_before =
            prosper::frontend::live_compute_cpu_fill_dispatches();
        CHECK(prosper::frontend::execute_live_compute_items({narrow_stride_item}),
              "noncanonical fill descriptor falls back to Vulkan");
        CHECK(result[300] == 0xabababab &&
                  prosper::frontend::live_compute_cpu_fill_dispatches() ==
                      narrow_stride_fills_before,
              "CPU fast path does not replace the descriptor's eight-byte stride semantics");
    }

    ComputeShaderConfig extra_user_config = config;
    extra_user_config.user_sgprs.push_back(0);
    const std::vector<uint32_t> extra_user_spirv = recompile_compute(
        code, std::size(code), &rt, extra_user_config);
    CHECK(!extra_user_spirv.empty(), "alternate user-SGPR fill kernel recompiles");
    if (!extra_user_spirv.empty()) {
        std::fill(result.begin(), result.end(), 0xcdcdcdcdu);
        ComputeItem extra_user_item = cpu_fill_item;
        extra_user_item.spirv = extra_user_spirv;
        extra_user_item.user_sgprs = extra_user_config.user_sgprs;
        extra_user_item.recompile_config = extra_user_config;
        const uint64_t extra_user_fills_before =
            prosper::frontend::live_compute_cpu_fill_dispatches();
        CHECK(prosper::frontend::execute_live_compute_items({extra_user_item}),
              "shifted TGID input fill falls back to Vulkan");
        CHECK(result[100 * 4] == 0xcdcdcdcdu &&
                  prosper::frontend::live_compute_cpu_fill_dispatches() ==
                      extra_user_fills_before,
              "CPU fast path requires TGID.x to occupy the shader's exact s8 input");
    }

    uint32_t unchanged_write_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
        if (addr == buffer.gpu_addr && size == buffer.size)
            unchanged_write_notifications++;
    });
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production live backend repeats an idempotent buffer dispatch");
    set_guest_gpu_write_observer({});
    CHECK(unchanged_write_notifications == 1,
          "idempotent compute writes still invalidate divergent renderer-resident state");

    // Astro Bot's world-map visibility kernel lowers IMAGE_ATOMIC_ADD on a tiled R32_UINT image
    // to a linear SSBO atomic. The descriptor's declared size is the logical texel count, while a
    // Sw64KbRX backing includes whole-tile padding and can therefore be larger. Replay provides that
    // physical capture through host_data_size; rejecting it against the logical size skipped the
    // entire visibility pass.
    {
        static const uint32_t image_atomic_add[] = {
            0x7e000280u, 0x7e020280u, 0x7e120281u,
            0xf0442108u, 0x00000900u, 0xbf810000u,
        };
        constexpr uint32_t atomic_width = 2048;
        constexpr uint32_t atomic_height = 64;
        constexpr uint32_t atomic_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);
        constexpr size_t atomic_logical_bytes =
            static_cast<size_t>(atomic_width) * atomic_height * sizeof(uint32_t);
        const size_t atomic_guest_bytes = tiled_surface_bytes(
            atomic_width, atomic_height, atomic_tile, 0, sizeof(uint32_t));
        CHECK(atomic_guest_bytes > atomic_logical_bytes,
              "world-map atomic image fixture includes physical tile padding");
        std::vector<uint8_t> atomic_guest(atomic_guest_bytes, 0);

        ShaderResourceTable atomic_rt;
        ShaderResource atomic_image;
        atomic_image.cls = ResourceClass::StorageImage;
        atomic_image.format = DataFormat::Uint32;
        atomic_image.num_components = 1;
        atomic_image.binding = 4;
        atomic_image.sgpr_base = 0;
        atomic_image.img_dim = 1;
        atomic_image.width = atomic_width;
        atomic_image.height = atomic_height;
        atomic_image.depth = 1;
        atomic_image.tile_mode = atomic_tile;
        atomic_image.size = static_cast<uint32_t>(atomic_logical_bytes);
        atomic_image.gpu_addr = reinterpret_cast<uint64_t>(atomic_guest.data());
        atomic_image.host_data = atomic_guest.data();
        atomic_image.host_data_size = atomic_guest.size();
        atomic_rt.resources.push_back(atomic_image);

        ComputeShaderConfig atomic_config;
        atomic_config.local_x = 1;
        const std::vector<uint32_t> atomic_spirv = recompile_compute(
            image_atomic_add, std::size(image_atomic_add), &atomic_rt, atomic_config);
        CHECK(!atomic_spirv.empty(), "world-map tiled image-atomic kernel recompiles");
        if (!atomic_spirv.empty()) {
            ComputeItem atomic_item;
            atomic_item.spirv = atomic_spirv;
            atomic_item.resources = std::make_shared<ShaderResourceTable>(atomic_rt);
            atomic_item.launch.threads_x = atomic_item.launch.threads_y =
                atomic_item.launch.threads_z = 1;
            atomic_item.launch.local_x = atomic_item.launch.local_y =
                atomic_item.launch.local_z = 1;
            atomic_item.launch.groups_x = atomic_item.launch.groups_y =
                atomic_item.launch.groups_z = 1;
            atomic_item.code_addr = 0x500525200;

            uint64_t atomic_notified_bytes = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t bytes) {
                if (addr == atomic_image.gpu_addr) atomic_notified_bytes = bytes;
            });
            CHECK(prosper::frontend::execute_live_compute_items({atomic_item}),
                  "world-map tiled image-atomic dispatch accepts its physical backing");
            set_guest_gpu_write_observer({});

            std::vector<uint32_t> atomic_linear(atomic_width * atomic_height, 0);
            detile_surface(reinterpret_cast<uint8_t*>(atomic_linear.data()), atomic_guest.data(),
                           atomic_width, atomic_height, atomic_tile, 0, sizeof(uint32_t));
            CHECK(atomic_linear[0] == 1,
                  "world-map tiled image-atomic dispatch publishes its atomic result");
            CHECK(atomic_notified_bytes == atomic_guest_bytes,
                  "world-map tiled image-atomic write invalidates the full physical backing");
        }
    }

    // Plucky Squire binds the same 32 MiB writable lighting buffer to consecutive kernels even
    // when their output is byte-identical to the previous frame. Exercise the production cache at
    // its one-MiB retention threshold: the second dispatch can use the exact GPU-side baseline,
    // while a later external guest mutation must defeat that shortcut and be repaired normally.
    {
        constexpr uint32_t large_buffer_bytes = 1u << 20;
        constexpr size_t large_words = large_buffer_bytes / sizeof(uint32_t);
        std::vector<uint32_t> large_result_storage;
        uint32_t* large_result = nullptr;
#if defined(__linux__)
        void* large_mapping = mmap(nullptr, large_buffer_bytes, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(large_mapping != MAP_FAILED, "map large writable compute-buffer regression range");
        if (large_mapping == MAP_FAILED) return fails ? fails : 1;
        large_result = static_cast<uint32_t*>(large_mapping);
        prosper::host::guest_write_watch_set_fault_onstack(true);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(large_result), large_buffer_bytes, 0x6b0000,
            0x3 /* SCE CPU_READ|CPU_WRITE */);
#else
        large_result_storage.resize(large_words);
        large_result = large_result_storage.data();
#endif
#if defined(__linux__)
        // The ordinary CPU-comparison fallback is used below the persistent-buffer threshold. It
        // also has an exact unchanged result and must use the byte-preserving notification, not
        // merely the retained GPU-comparator path exercised below.
        std::fill_n(large_result, buffer.size / sizeof(uint32_t), 0xccccccccu);
        ShaderResource watched_small_buffer = buffer;
        watched_small_buffer.gpu_addr = reinterpret_cast<uint64_t>(large_result);
        ShaderResourceTable watched_small_rt;
        watched_small_rt.resources.push_back(watched_small_buffer);
        ComputeItem watched_small_item = item;
        watched_small_item.resources = std::make_shared<ShaderResourceTable>(watched_small_rt);
        CHECK(prosper::frontend::execute_live_compute_items({watched_small_item}),
              "small writable buffer establishes its CPU-comparison result");
        auto unchanged_small_watch = prosper::host::GuestWriteWatch::create(
            reinterpret_cast<uint64_t>(large_result), buffer.size);
        CHECK(prosper::frontend::execute_live_compute_items({watched_small_item}),
              "small writable buffer repeats through CPU comparison");
        CHECK(static_cast<bool>(unchanged_small_watch) &&
                  unchanged_small_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
              "CPU-identical production dispatch keeps guest-byte watches clean");
        unchanged_small_watch.reset();
#endif
        std::fill_n(large_result, large_words, 0xababababu);
        ShaderResource large_buffer = buffer;
        large_buffer.gpu_addr = reinterpret_cast<uint64_t>(large_result);
        large_buffer.size = large_buffer_bytes;
        ShaderResourceTable large_rt;
        large_rt.resources.push_back(large_buffer);
        ComputeItem large_item = item;
        large_item.resources = std::make_shared<ShaderResourceTable>(large_rt);
        large_item.code_addr = 0x401aec210;

        CHECK(prosper::frontend::execute_live_compute_items({large_item}),
              "large writable buffer dispatch establishes an exact retained result");
        const std::vector<uint32_t> large_expected(large_result, large_result + large_words);
        const uint64_t buffer_skips_before =
            prosper::frontend::live_compute_buffer_gpu_result_skips();
#if defined(__linux__)
        auto unchanged_result_watch = prosper::host::GuestWriteWatch::create(
            reinterpret_cast<uint64_t>(large_result), large_buffer_bytes);
        CHECK(static_cast<bool>(unchanged_result_watch) &&
                  unchanged_result_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
              "arm guest-byte watch around retained compute-buffer result");
#endif
        uint32_t large_repeat_notifications = 0;
        set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
            if (addr == large_buffer.gpu_addr && size == large_buffer.size)
                ++large_repeat_notifications;
        });
        CHECK(prosper::frontend::execute_live_compute_items({large_item}),
              "large writable buffer repeats against its exact GPU baseline");
        set_guest_gpu_write_observer({});
        CHECK(std::equal(large_result, large_result + large_words, large_expected.begin()) &&
                  large_repeat_notifications == 1,
              "GPU-identical buffer output preserves bytes and architectural invalidation");
        CHECK(prosper::frontend::live_compute_buffer_gpu_result_skips() > buffer_skips_before,
              "large idempotent buffer takes the exact GPU result-comparison fast path");
#if defined(__linux__)
        CHECK(unchanged_result_watch.query() ==
                  prosper::host::GuestWriteWatchQuery::Unchanged,
              "GPU-identical production dispatch keeps guest-byte watches clean");
        unchanged_result_watch.reset();
#endif

        large_result[0] ^= 0xffffffffu;
        uint32_t large_repair_notifications = 0;
        set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
            if (addr == large_buffer.gpu_addr && size == large_buffer.size)
                ++large_repair_notifications;
        });
        CHECK(prosper::frontend::execute_live_compute_items({large_item}),
              "externally changed buffer reruns with exact source validation");
        set_guest_gpu_write_observer({});
        CHECK(std::equal(large_result, large_result + large_words, large_expected.begin()) &&
                  large_repair_notifications == 1,
              "external buffer mutation forces exact guest repair and invalidation");
#if defined(__linux__)
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(large_result), large_buffer_bytes);
        prosper::host::guest_write_watch_set_fault_onstack(false);
        munmap(large_result, large_buffer_bytes);
#endif
    }

    std::fill(result.begin(), result.end(), 0xeeeeeeee);
    item.user_sgprs = alternate_config.user_sgprs;
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "cached compute pipeline executes with updated user SGPR push constants");
    const uint32_t alternate_expected[4] = {
        alternate_config.user_sgprs[4], alternate_config.user_sgprs[5],
        alternate_config.user_sgprs[6], alternate_config.user_sgprs[7],
    };
    bool alternate_filled = true;
    for (uint32_t record = 0; record < records && alternate_filled; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            alternate_filled &= result[record * 4 + component] == alternate_expected[component];
    CHECK(alternate_filled,
          "cached compute pipeline observes per-dispatch user SGPR values");
    item.user_sgprs = config.user_sgprs;

    std::vector<uint32_t> replay_owned(launched_records * 4, 0xdddddddd);
    ShaderResource replay_buffer = buffer;
    replay_buffer.gpu_addr = 1; // Deliberately unreadable: replay must never dereference this identity.
    replay_buffer.host_data = reinterpret_cast<uint8_t*>(replay_owned.data());
    replay_buffer.host_data_size = replay_owned.size() * sizeof(uint32_t);
    item.resources = std::make_shared<ShaderResourceTable>();
    item.resources->resources.push_back(replay_buffer);
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "production compute backend executes against replay-owned bytes");
    bool replay_filled = true;
    for (uint32_t record = 0; record < records && replay_filled; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            replay_filled &= replay_owned[record * 4 + component] == expected[component];
    CHECK(replay_filled, "compute writeback updates owned backing for a later replay operation");
    bool replay_padding_untouched = true;
    for (uint32_t record = records; record < launched_records; ++record)
        for (uint32_t component = 0; component < 4; ++component)
            replay_padding_untouched &= replay_owned[record * 4 + component] == 0xdddddddd;
    CHECK(replay_padding_untouched, "captured/replay-owned backing preserves padded invocations");

    // Terminator 2D's startup uses device-global GDS. Capture its initial host-backed state, then
    // execute two ordered replay dispatches against the same materialized 64 KiB instance. The
    // first store wraps its byte address to 16 bits; the second narrows EXEC to lane zero. If the
    // store ignores EXEC, lanes 1..3 overwrite the first dispatch's dword at wrapped address 4.
    static const uint32_t gds_wrap_store[] = {
        0x7e0002ffu, 0x00010004u,  // v0 = byte address 0x10004 -> GDS byte address 4
        0x7e0202ffu, 0xdeadbeefu,  // v1 = value
        0xd8360000u, 0x00000100u,  // ds_write_b32 v0, v1 gds
        0xbf810000u,
    };
    static const uint32_t gds_exec_store[] = {
        0x34000082u,               // v_lshlrev_b32 v0, 2, v0 (one dword per lane)
        0x7e0202ffu, 0xcafebabeu,  // v1 = value
        0x7da40080u,               // v_cmpx_eq_u32 0, v0
        0xd8360000u, 0x00000100u,  // only lane zero stores at byte address 0
        0xbf810000u,
    };
    std::array<uint8_t, 64 * 1024> gds_initial{};
    ShaderResource gds_resource;
    gds_resource.cls = ResourceClass::ConstantBuffer;
    gds_resource.format = DataFormat::Uint32;
    gds_resource.num_components = 1;
    gds_resource.binding = kComputeInternalGdsBinding;
    gds_resource.size = gds_initial.size();
    gds_resource.stride = 4;
    gds_resource.host_data = gds_initial.data();
    gds_resource.host_data_size = gds_initial.size();
    auto gds_table = std::make_shared<ShaderResourceTable>();
    gds_table->resources.push_back(gds_resource);
    ComputeShaderConfig gds_config;
    auto make_gds_item = [&](const uint32_t* code_words, size_t word_count,
                             uint32_t threads, uint64_t order) {
        ComputeItem gds_item;
        gds_item.spirv = recompile_compute(code_words, word_count, gds_table.get(), gds_config);
        gds_item.resources = gds_table;
        gds_item.launch.threads_x = threads;
        gds_item.launch.local_x = threads;
        gds_item.launch.local_y = gds_item.launch.local_z = 1;
        gds_item.launch.groups_x = gds_item.launch.groups_y = gds_item.launch.groups_z = 1;
        gds_item.dispatch_index = order;
        gds_item.command_order = order;
        return gds_item;
    };
    ComputeItem gds_first = make_gds_item(gds_wrap_store, std::size(gds_wrap_store), 1, 10);
    ComputeItem gds_second = make_gds_item(gds_exec_store, std::size(gds_exec_store), 4, 20);
    CHECK(!gds_first.spirv.empty() && !gds_second.spirv.empty(),
          "compute GDS wrap and EXEC kernels recompile");
    GpuCaptureFile gds_capture;
    GpuCaptureMetadata gds_metadata;
    std::string gds_error;
    const std::vector<SubmitOperation> gds_operations = {
        {SubmitOperationKind::Dispatch, 10, 10},
        {SubmitOperationKind::Dispatch, 20, 20},
    };
    // Deferred runtime capture finishes after the backend mutates this shared table. Its replay
    // input must nevertheless be the pre-submit GDS bytes, paired with the exact realized compute
    // list obtained after execution.
    constexpr uint32_t kGdsPreSubmitSentinel = 0x10293847u;
    constexpr uint32_t kGdsPostSubmitSentinel = 0xfedcba98u;
    std::memcpy(gds_initial.data(), &kGdsPreSubmitSentinel,
                sizeof(kGdsPreSubmitSentinel));
    // Per-process path: this binary runs as two concurrent ctest cases, so a fixed name made both
    // read back the other run's bytes (#1613).
    const std::string deferred_gds_capture_path =
        prosper_test::test_scratch_file("prosper_deferred_gds_capture.prgcap");
    auto deferred_gds_pending = std::make_unique<PendingGpuCapture>();
    deferred_gds_pending->materialized = false;
    deferred_gds_pending->path = deferred_gds_capture_path;
    snapshot_pending_gpu_capture_compute_gds(
        deferred_gds_pending.get(), gds_initial.data(), gds_initial.size());
    std::memcpy(gds_initial.data(), &kGdsPostSubmitSentinel,
                sizeof(kGdsPostSubmitSentinel));
    const std::vector<DrawItem> deferred_gds_draws;
    const std::vector<ComputeItem> deferred_gds_computes = {gds_first, gds_second};
    CHECK(finish_requested_gpu_capture(
              std::move(deferred_gds_pending), {}, gds_error, &deferred_gds_draws,
              &deferred_gds_computes, &gds_operations),
          "deferred capture writes exact compute operations with snapshotted GDS input");
    GpuCaptureFile deferred_gds_capture;
    GpuReplayFrame deferred_gds_replay;
    const bool deferred_gds_reopened = read_gpu_capture(
        deferred_gds_capture_path, deferred_gds_capture, gds_error) &&
        materialize_gpu_replay(deferred_gds_capture, deferred_gds_replay, gds_error);
    const uint32_t* deferred_gds_words = deferred_gds_reopened &&
        !deferred_gds_replay.computes.empty() &&
        deferred_gds_replay.computes[0].resources &&
        !deferred_gds_replay.computes[0].resources->resources.empty()
            ? reinterpret_cast<const uint32_t*>(
                  deferred_gds_replay.computes[0].resources->resources[0].host_data)
            : nullptr;
    CHECK(deferred_gds_words && deferred_gds_words[0] == kGdsPreSubmitSentinel,
          "deferred capture replay restores pre-submit persistent GDS, not post-submit bytes");
    std::remove(deferred_gds_capture_path.c_str());
    gds_initial.fill(0);
    CHECK(capture_submit_items({}, {gds_first, gds_second}, gds_operations, gds_metadata,
                               [](uint64_t, uint8_t*, size_t) { return size_t{0}; },
                               gds_capture, gds_error),
          "capture records host-backed compute GDS without reading guest address zero");
    std::vector<uint8_t> gds_capture_bytes;
    GpuCaptureFile gds_loaded;
    GpuReplayFrame gds_replay;
    CHECK(serialize_gpu_capture(gds_capture, gds_capture_bytes, gds_error) &&
          deserialize_gpu_capture(gds_capture_bytes, gds_loaded, gds_error) &&
          materialize_gpu_replay(gds_loaded, gds_replay, gds_error) &&
          gds_replay.computes.size() == 2 &&
          gds_replay.computes[0].resources->resources[0].host_data ==
              gds_replay.computes[1].resources->resources[0].host_data,
          "capture v22 materializes one persistent GDS instance across ordered dispatches");
    uint32_t gds_notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t, uint64_t) { ++gds_notifications; });
    CHECK(prosper::frontend::execute_live_compute_items(gds_replay.computes),
          "production live backend executes captured GDS stores");
    set_guest_gpu_write_observer({});
    const auto* gds_words = reinterpret_cast<const uint32_t*>(
        gds_replay.computes[0].resources->resources[0].host_data);
    CHECK(gds_words && gds_words[0] == 0xcafebabeu && gds_words[1] == 0xdeadbeefu &&
              gds_words[2] == 0 && gds_words[3] == 0,
          "GDS stores honor 16-bit address wrap, EXEC predication, and cross-dispatch persistence");
    CHECK(gds_notifications == 0,
          "GPU-internal GDS writeback does not invalidate guest address zero");

    // Astro Bot's four indirect-list counters prove the GDS M0 layout independently of the LDS
    // append tests: M0[31:16] is the byte base and M0[15:0] is the allocation size. The exact live
    // value 0x0c600020 plus offset 0x10 must append at 0xc70, not at the old low-half address 0x30.
    // A second kernel crosses the 64 KiB boundary, proving that base+offset wraps in the GDS domain.
    static const uint32_t gds_m0_high_append[] = {
        0xbefc03ffu, 0x0c600020u,  // s_mov_b32 m0, {base=0xc60,size=0x20}
        0xd8fa0010u, 0x00000000u,  // ds_append v0 offset:0x10 gds -> 0xc70
        0xbf810000u,
    };
    static const uint32_t gds_m0_wrap_append[] = {
        0xbefc03ffu, 0xfff00020u,  // s_mov_b32 m0, {base=0xfff0,size=0x20}
        0xd8fa0030u, 0x00000000u,  // ds_append v0 offset:0x30 gds -> 0x20 after wrap
        0xbf810000u,
    };
    gds_initial.fill(0);
    const auto* gds_m0_words = reinterpret_cast<const uint32_t*>(gds_initial.data());
    CHECK(gds_m0_words[0] == 0 && gds_m0_words[0x20 / 4] == 0 &&
              gds_m0_words[0x30 / 4] == 0 && gds_m0_words[0x50 / 4] == 0 &&
              gds_m0_words[0xc70 / 4] == 0,
          "GDS M0 fixture starts with zeroed authoritative host backing");
    // Match Astro's 1x1x1 producer. The earlier store fixture intentionally used the default
    // 64-lane workgroup; leaving that config in place here would append 64 active lanes while the
    // launch metadata claimed one, making the address assertion depend on an unrelated count.
    gds_config.local_x = 1;
    ComputeItem gds_high_append = make_gds_item(
        gds_m0_high_append, std::size(gds_m0_high_append), 1, 30);
    ComputeItem gds_wrap_append = make_gds_item(
        gds_m0_wrap_append, std::size(gds_m0_wrap_append), 1, 40);
    CHECK(!gds_high_append.spirv.empty() && !gds_wrap_append.spirv.empty(),
          "GDS append kernels with nonzero M0 base recompile");
    CHECK(prosper::frontend::execute_live_compute_items({gds_high_append, gds_wrap_append}),
          "production compute backend executes GDS M0 base and wrap kernels");
    printf("  GDS M0 words [0]=%u [0x20]=%u [0x30]=%u [0x50]=%u [0xc70]=%u\n",
           gds_m0_words[0], gds_m0_words[0x20 / 4], gds_m0_words[0x30 / 4],
           gds_m0_words[0x50 / 4], gds_m0_words[0xc70 / 4]);
    CHECK(gds_m0_words[0xc70 / 4] == 1 && gds_m0_words[0x20 / 4] == 1 &&
              gds_m0_words[0x30 / 4] == 0 && gds_m0_words[0x50 / 4] == 0,
          "GDS append uses M0 high-half base, wraps at 64 KiB, and rejects low-half aliases");

    // --- #590: the live backend's storage-IMAGE path. The same 1D image-copy kernel that
    // test_storage_image_copy proves against the raw harness, executed through the PRODUCTION
    // backend with Unorm8x4 guest-style backing — exercising the full chain: channel unpack
    // (bytes -> float-bit uvec4 texels), the R32G32B32A32_UINT image contract, and the pack-back
    // writeback (float bits -> clamped bytes). Unorm8 pack(unpack(b)) == b exactly, so a bit-exact
    // dst==src is the correctness assertion.
    static const uint32_t image_copy[] = {
        0x7E080300u, 0xF0000F00u, 0x00000004u, 0xBF8C3F70u, 0xF0200F00u, 0x00020004u, 0xBF810000u,
    };
    static const uint32_t image_copy_2d[] = {
        0x7E080300u,             // v4 = x from the shell input
        0x7E0A0280u,             // v5 = y = 0
        0xF0000F08u, 0x00000004u, 0xBF8C3F70u,
        0xF0200F08u, 0x00020004u, 0xBF810000u,
    };
    static const uint32_t image_copy_3d[] = {
        0x7E080300u,             // v4 = x from the shell input
        0x7E0A0280u,             // v5 = y = 0
        0x7E0C0280u,             // v6 = z = 0
        // MIMG.DIM lives in bits [5:3] (rdna2_decode.cpp: `(w >> 3) & 0x7`), so DIM=3D is 0x10 —
        // 0x08 is DIM=2D, exactly what image_copy_2d above encodes. This fixture prepares a third
        // address register and calls itself 3D, so the DIM field has to agree: with 0x08 the
        // recompiler faithfully emitted a 2D SPIR-V image addressed by two coordinates while the
        // backend bound the 3D view its 3D resource descriptor calls for, and a 3D view bound to a
        // descriptor whose shader declares Dim 2D is undefined in Vulkan (#1690).
        0xF0000F10u, 0x00000004u, 0xBF8C3F70u,
        0xF0200F10u, 0x00020004u, 0xBF810000u,
    };
    static const uint32_t image_copy_2d_array[] = {
        0x7E080300u,             // v4 = x from the shell input
        0x7E0A0280u,             // v5 = y = 0
        0x7E0C0282u,             // v6 = array layer 2
        0xF0000F28u, 0x00000004u, 0xBF8C3F70u,
        0xF0200F28u, 0x00020004u, 0xBF810000u,
    };
    const uint32_t W = 64;
    std::vector<uint32_t> lane_index(W);
    for (uint32_t i = 0; i < W; i++) lane_index[i] = i;      // shell input: v0 = input[gid] = gid
    std::vector<uint32_t> dummy(4, 0);
    std::vector<uint8_t> img_src(W * 4), img_dst(W * 4, 0xEE);
    for (uint32_t i = 0; i < W * 4; i++) img_src[i] = (uint8_t)(i * 37 + 5);
    ShaderResourceTable irt;
    auto add_buffer = [&](uint32_t binding, void* data, uint32_t size) {
        ShaderResource b{};
        b.cls = ResourceClass::ConstantBuffer;
        b.binding = binding;
        b.gpu_addr = (uint64_t)(uintptr_t)data;
        b.size = size;
        irt.resources.push_back(b);
    };
    add_buffer(0, lane_index.data(), W * sizeof(uint32_t));
    add_buffer(1, dummy.data(), 16);
    add_buffer(2, dummy.data(), 16);
    add_buffer(3, dummy.data(), 16);
    auto add_image = [&](uint32_t binding, uint32_t sgpr, void* data, uint32_t size) {
        ShaderResource im{};
        im.cls = ResourceClass::StorageImage;
        im.img_dim = 0;                     // 1D
        im.binding = binding;
        im.sgpr_base = sgpr;
        im.format = DataFormat::Unorm8;
        im.num_components = 4;
        im.width = W; im.height = 1;
        im.gpu_addr = (uint64_t)(uintptr_t)data;
        im.size = size;
        irt.resources.push_back(im);
    };
    add_image(4, 0, img_src.data(), W * 4);
    add_image(5, 8, img_dst.data(), W * 4);
    std::vector<uint32_t> image_spirv = recompile_valu(
        image_copy, sizeof(image_copy) / sizeof(image_copy[0]), 1, 0, &irt);
    CHECK(!image_spirv.empty(), "storage-image copy kernel recompiles against the game-style table");
    if (!image_spirv.empty()) {
        ComputeItem image_item;
        image_item.spirv = image_spirv;
        image_item.resources = std::make_shared<ShaderResourceTable>(irt);
        image_item.launch.threads_x = W;
        image_item.launch.local_x = 64;
        image_item.launch.groups_x = 1;
        image_item.launch.local_y = image_item.launch.local_z = 1;
        image_item.launch.groups_y = image_item.launch.groups_z = 1;
        image_item.code_addr = 0x590590;
        CHECK(prosper::frontend::execute_live_compute_items({image_item}),
              "live backend executes the storage-image copy dispatch (#590)");
        uint32_t bad = 0;
        for (uint32_t i = 0; i < W * 4; i++) bad += img_dst[i] != img_src[i];
        if (bad) std::printf("  image copy mismatched bytes = %u/%u (b0 src=%02x dst=%02x)\n",
                             bad, W * 4, img_src[0], img_dst[0]);
        CHECK(bad == 0, "Unorm8 unpack -> uvec4 copy -> pack writeback is byte-exact (#590)");
    }

    // --- #590: the same production storage-image copy across the INTEGER and packed formats DOLL's
    // UE4 post-process writes its color-grading LUT / exposure / volume dispatches as (a 32x32x32
    // Uint8/Unorm2_10_10_10 3D LUT, a 1x1x1 exposure, a 16x16x16 Uint16 volume). Each dispatch was
    // skipped before storage_(un)pack gained these formats, leaving the tonemap sampling an
    // unproduced surface. A raw uvec4 copy through the production backend must round-trip the guest
    // bytes bit-exact: integer formats widen/truncate, and any quantized 10/10/10/2 word is stable
    // under unpack->pack.
    auto run_format_copy = [&](DataFormat fmt, uint32_t ncomp, uint32_t texel_bytes,
                               const std::vector<uint8_t>& src) -> std::vector<uint8_t> {
        std::vector<uint8_t> dst(src.size(), 0x5A);
        ShaderResourceTable frt;
        frt.resources = irt.resources;   // reuse constant buffers 0..3 (binding 0 -> v0 = gid)
        frt.resources.erase(std::remove_if(frt.resources.begin(), frt.resources.end(),
            [](const ShaderResource& r) { return r.cls == ResourceClass::StorageImage; }),
            frt.resources.end());
        auto add_fmt_image = [&](uint32_t binding, uint32_t sgpr, void* data) {
            ShaderResource im{};
            im.cls = ResourceClass::StorageImage; im.img_dim = 0; im.binding = binding;
            im.sgpr_base = sgpr; im.format = fmt; im.num_components = ncomp;
            im.width = W; im.height = 1;
            im.gpu_addr = (uint64_t)(uintptr_t)data; im.size = W * texel_bytes;
            frt.resources.push_back(im);
        };
        add_fmt_image(4, 0, const_cast<uint8_t*>(src.data()));
        add_fmt_image(5, 8, dst.data());
        std::vector<uint32_t> spv = recompile_valu(
            image_copy, sizeof(image_copy) / sizeof(image_copy[0]), 1, 0, &frt);
        if (spv.empty()) return {};
        ComputeItem it;
        it.spirv = spv;
        it.resources = std::make_shared<ShaderResourceTable>(frt);
        it.launch.threads_x = W; it.launch.local_x = 64; it.launch.groups_x = 1;
        it.launch.local_y = it.launch.local_z = 1; it.launch.groups_y = it.launch.groups_z = 1;
        it.code_addr = 0x590591;
        if (!prosper::frontend::execute_live_compute_items({it})) return {};
        return dst;
    };
    {   // Uint8 x4 (fmt=11): raw integer channels, exact round-trip
        std::vector<uint8_t> s(W * 4);
        for (uint32_t i = 0; i < s.size(); i++) s[i] = (uint8_t)(i * 53 + 7);
        CHECK(run_format_copy(DataFormat::Uint8, 4, 4, s) == s,
              "Uint8x4 storage copy round-trips guest bytes bit-exact (#590)");
    }
    {   // Unorm8 x2: native RG8 storage keeps its two-byte texel width and conversion contract
        std::vector<uint8_t> s(W * 2);
        for (uint32_t i = 0; i < s.size(); i++) s[i] = (uint8_t)(i * 71 + 11);
        CHECK(run_format_copy(DataFormat::Unorm8, 2, 2, s) == s,
              "Unorm8x2 storage copy round-trips native RG8 texels bit-exact (#590)");
    }
    {   // Astro Bot's title composite writes a tiled 256x64 RGBA16-UNORM surface.
        std::vector<uint8_t> s(W * 8);
        for (uint32_t t = 0; t < W * 4; ++t) {
            const uint16_t value = static_cast<uint16_t>(t * 251u + 17u);
            s[t * 2] = static_cast<uint8_t>(value);
            s[t * 2 + 1] = static_cast<uint8_t>(value >> 8);
        }
        CHECK(run_format_copy(DataFormat::Unorm16, 4, 8, s) == s,
              "Unorm16x4 storage copy round-trips guest channels bit-exact (#590)");
    }
    {   // Uint16 x1 (fmt=7): 16-bit integer widen on load, low-16-bit truncate on store
        std::vector<uint8_t> s(W * 2);
        for (uint32_t i = 0; i < s.size(); i++) s[i] = (uint8_t)(i * 29 + 3);
        CHECK(run_format_copy(DataFormat::Uint16, 1, 2, s) == s,
              "Uint16x1 storage copy round-trips guest bytes bit-exact (#590)");
    }
    {   // Unorm16 x1 (fmt=4): normalized conversion preserves every quantized source value
        std::vector<uint8_t> s(W * 2);
        for (uint32_t t = 0; t < W; ++t) {
            const uint16_t value = static_cast<uint16_t>(t * 1040u);
            std::memcpy(&s[t * 2], &value, sizeof(value));
        }
        CHECK(run_format_copy(DataFormat::Unorm16, 1, 2, s) == s,
              "Unorm16x1 storage copy round-trips normalized guest values");
    }
    {   // Snorm8 x1 (fmt=9): -128 and -127 both represent -1 and write back canonically as -127
        const int8_t values[] = {-128, -127, -96, -64, -1, 0, 1, 63, 96, 127};
        std::vector<uint8_t> s(W), expected(W);
        for (uint32_t t = 0; t < W; ++t) {
            const int8_t value = values[t % std::size(values)];
            s[t] = static_cast<uint8_t>(value);
            expected[t] = static_cast<uint8_t>(value == -128 ? -127 : value);
        }
        CHECK(run_format_copy(DataFormat::Snorm8, 1, 1, s) == expected,
              "Snorm8x1 storage copy normalizes and canonicalizes the negative endpoint");
    }
    {   // Snorm16 x1 (fmt=5): The Plucky Squire uses this storage surface in its title path
        const int16_t values[] = {-32768, -32767, -24576, -16384, -1, 0, 1, 16383, 24576, 32767};
        std::vector<uint8_t> s(W * 2), expected(W * 2);
        for (uint32_t t = 0; t < W; ++t) {
            const int16_t value = values[t % std::size(values)];
            const int16_t canonical = value == -32768 ? -32767 : value;
            std::memcpy(&s[t * 2], &value, sizeof(value));
            std::memcpy(&expected[t * 2], &canonical, sizeof(canonical));
        }
        CHECK(run_format_copy(DataFormat::Snorm16, 1, 2, s) == expected,
              "Snorm16x1 storage copy normalizes and canonicalizes the negative endpoint");
    }
    {   // Unorm2_10_10_10 x4 (fmt=21): packed 10/10/10/2, quantized word stable under unpack->pack
        std::vector<uint8_t> s(W * 4);
        for (uint32_t t = 0; t < W; t++) { uint32_t p = t * 2654435761u; std::memcpy(&s[t * 4], &p, 4); }
        CHECK(run_format_copy(DataFormat::Unorm2_10_10_10, 4, 4, s) == s,
              "Unorm2_10_10_10 storage copy round-trips packed guest word bit-exact (#590)");
    }
    {   // R11G11B10 UFLOAT: native float storage conversion must preserve quantized finite texels
        std::vector<uint8_t> s(W * 4);
        for (uint32_t t = 0; t < W; ++t) {
            const float r = static_cast<float>((t * 17u) % 97u) / 13.0f;
            const float g = static_cast<float>((t * 29u) % 89u) / 11.0f;
            const float b = static_cast<float>((t * 43u) % 83u) / 9.0f;
            const uint32_t packed = static_cast<uint32_t>(float_to_f11(r)) |
                                    (static_cast<uint32_t>(float_to_f11(g)) << 11) |
                                    (static_cast<uint32_t>(float_to_f10(b)) << 22);
            std::memcpy(&s[t * 4], &packed, sizeof(packed));
        }
        CHECK(run_format_copy(DataFormat::Float10_11_11, 3, 4, s) == s,
              "R11G11B10 storage copy round-trips native packed texels bit-exact (#590)");
    }
    {   // No native R11 storage: shader-side R32_UINT packing must preserve every f11 code.
        constexpr uint32_t PACKED_W = 2048;
        std::vector<uint32_t> indices(PACKED_W), src(PACKED_W), dst(PACKED_W, 0x5a5a5a5au);
        for (uint32_t t = 0; t < PACKED_W; ++t) {
            indices[t] = t;
            src[t] = t | (((t * 1031u) & 0x7ffu) << 11) | ((t & 0x3ffu) << 22);
        }
        ShaderResourceTable packed_rt = irt;
        for (ShaderResource& resource : packed_rt.resources) {
            if (resource.binding == 0) {
                resource.gpu_addr = reinterpret_cast<uint64_t>(indices.data());
                resource.size = static_cast<uint32_t>(indices.size() * sizeof(uint32_t));
            }
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.img_dim = 1;
            resource.format = DataFormat::Float10_11_11;
            resource.num_components = 3;
            resource.width = PACKED_W;
            resource.height = resource.depth = 1;
            resource.tile_mode = 0;
            resource.gpu_addr = reinterpret_cast<uint64_t>(
                resource.binding == 4 ? src.data() : dst.data());
            resource.size = PACKED_W * sizeof(uint32_t);
        }
        const std::vector<uint32_t> packed_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &packed_rt);
        const auto packed_report = validate_spirv_descriptor_interface(
            packed_spirv, &packed_rt, 0, SpirvShaderStage::Compute, false);
        CHECK(!packed_spirv.empty() && packed_report.ok() &&
                  packed_report.descriptors.size() == 4 &&
                  packed_report.descriptors[2].storage_image_format == kSpirvImageFormatR32ui &&
                  packed_report.descriptors[3].storage_image_format == kSpirvImageFormatR32ui,
              "R11G11B10 fallback recompiles both 2D storage views as typed R32ui");
        if (!packed_spirv.empty()) {
            ComputeItem packed_item;
            packed_item.spirv = packed_spirv;
            packed_item.resources = std::make_shared<ShaderResourceTable>(packed_rt);
            packed_item.launch.threads_x = PACKED_W;
            packed_item.launch.local_x = 64;
            packed_item.launch.groups_x = PACKED_W / 64;
            packed_item.launch.local_y = packed_item.launch.local_z = 1;
            packed_item.launch.groups_y = packed_item.launch.groups_z = 1;
            packed_item.code_addr = 0x590b10;
            const bool packed_executed =
                prosper::frontend::execute_live_compute_items({packed_item});

            // #1681. Two separate corrections live here.
            //
            // The INSTRUMENT: the old report used std::mismatch, which returns only the FIRST
            // differing element while the assertion compared the whole vector — so its printout was
            // byte-identical whether 1 texel differed or 92, and a breadth claim read off it had no
            // evidence behind it. The census below counts and classifies every differing texel.
            //
            // The CONTRACT: the sweep covers all 2048 f11 codes, which necessarily includes the NaN
            // encodings, and it required bit-exact equality over them. NaN *payload* propagation is
            // not something any driver owes us — IEEE 754 leaves it unspecified and SPIR-V/Vulkan
            // follow suit — and prosper's unpack routes each code through GLSL UnpackHalf2x16, a
            // genuine f16->f32 conversion. RADV preserves the payload; lavapipe quiets signalling
            // NaNs (x86 VCVTPH2PS sets the quiet bit), and both are conformant. So require
            // bit-exactness for every finite and infinite code — the values that actually carry
            // rendered colour — and require only that a NaN stays a NaN. This keeps all 2048 codes
            // under test and drops only the bits no implementation guarantees.
            struct Field { uint32_t shift, width; };
            const Field fields[3] = {{0, 11}, {11, 11}, {22, 10}};  // R f11, G f11, B f10
            auto chan = [&](uint32_t texel, int c) {
                return (texel >> fields[c].shift) & ((1u << fields[c].width) - 1u);
            };
            auto mant_bits = [&](int c) { return fields[c].width - 5u; };
            auto is_nan = [&](uint32_t v, int c) {
                const uint32_t m = mant_bits(c);
                return (v >> m) == 0x1fu && (v & ((1u << m) - 1u)) != 0u;
            };
            auto is_inf = [&](uint32_t v, int c) {
                const uint32_t m = mant_bits(c);
                return (v >> m) == 0x1fu && (v & ((1u << m) - 1u)) == 0u;
            };
            auto is_snan = [&](uint32_t v, int c) {
                const uint32_t m = mant_bits(c);
                return is_nan(v, c) && (v & (1u << (m - 1u))) == 0u;
            };
            // A channel round-trips when it is bit-exact, or when a NaN source stayed some NaN.
            auto channel_round_trips = [&](uint32_t s, uint32_t d, int c) {
                const uint32_t sv = chan(s, c), dv = chan(d, c);
                return is_nan(sv, c) ? is_nan(dv, c) : sv == dv;
            };

            // Pin the narrowed predicate itself on the CPU, with no GPU involved (#1681). The live
            // sweep can only demonstrate the cases the driver happens to produce, so it cannot show
            // that the exemption stops at NaN *payload* — and a narrowing that quietly also accepted
            // Inf<->NaN or NaN->finite would still pass every run. These assertions fix the contract
            // in both directions and would fail if a later edit widened it.
            {
                bool accepts_payload = true, rejects_everything_else = true;
                for (int c = 0; c < 3; ++c) {
                    const uint32_t m = mant_bits(c), sh = fields[c].shift;
                    auto tex = [&](uint32_t v) { return v << sh; };
                    const uint32_t inf = 0x1fu << m;                 // exp all ones, mantissa 0
                    const uint32_t snan = inf | 1u;                  // mantissa MSB clear
                    const uint32_t qnan = inf | (1u << (m - 1u));    // mantissa MSB set
                    const uint32_t sub = 1u;                         // exp 0, mantissa 1
                    const uint32_t norm = (15u << m) | 3u;           // ordinary finite value
                    // Payload movement between NaNs is permitted, in either direction.
                    accepts_payload &= channel_round_trips(tex(snan), tex(qnan), c);
                    accepts_payload &= channel_round_trips(tex(qnan), tex(snan), c);
                    accepts_payload &= channel_round_trips(tex(snan), tex(snan), c);
                    accepts_payload &= channel_round_trips(tex(inf), tex(inf), c);
                    accepts_payload &= channel_round_trips(tex(sub), tex(sub), c);
                    accepts_payload &= channel_round_trips(tex(norm), tex(norm), c);
                    // Nothing else is. A NaN may not decay, and a non-NaN may not become one.
                    rejects_everything_else &= !channel_round_trips(tex(snan), tex(inf), c);
                    rejects_everything_else &= !channel_round_trips(tex(qnan), tex(inf), c);
                    rejects_everything_else &= !channel_round_trips(tex(snan), tex(0), c);
                    rejects_everything_else &= !channel_round_trips(tex(snan), tex(norm), c);
                    rejects_everything_else &= !channel_round_trips(tex(inf), tex(snan), c);
                    rejects_everything_else &= !channel_round_trips(tex(inf), tex(norm), c);
                    rejects_everything_else &= !channel_round_trips(tex(norm), tex(norm + 1u), c);
                    rejects_everything_else &= !channel_round_trips(tex(sub), tex(0), c);
                    rejects_everything_else &= !channel_round_trips(tex(0), tex(sub), c);
                }
                CHECK(accepts_payload,
                      "R11G11B10 round-trip contract permits NaN payload movement in both directions");
                CHECK(rejects_everything_else,
                      "R11G11B10 round-trip contract still rejects NaN<->Inf, NaN->finite, and any "
                      "finite or subnormal change in every channel");
            }

            size_t payload_only = 0, real_diffs = 0;
            for (size_t t = 0; t < src.size(); ++t) {
                if (src[t] == dst[t]) continue;
                bool ok = true;
                for (int c = 0; c < 3; ++c)
                    if (!channel_round_trips(src[t], dst[t], c)) ok = false;
                if (ok) ++payload_only; else ++real_diffs;
            }
            if (packed_executed && dst != src) {
                size_t total = 0, all_finite = 0, any_snan = 0, any_nan = 0, any_inf = 0;
                size_t quiet_exact = 0, canon_exact = 0;
                long long first_finite = -1, first_any = -1;
                for (size_t t = 0; t < src.size(); ++t) {
                    if (src[t] == dst[t]) continue;
                    ++total;
                    if (first_any < 0) first_any = static_cast<long long>(t);
                    bool finite = true, snan = false, nan = false, inf = false;
                    uint32_t quieted = 0, canonical = 0;
                    for (int c = 0; c < 3; ++c) {
                        const uint32_t v = chan(src[t], c), m = mant_bits(c);
                        if (is_nan(v, c)) { nan = true; finite = false; }
                        if (is_inf(v, c)) { inf = true; finite = false; }
                        if (is_snan(v, c)) snan = true;
                        // Quieting model: an sNaN gains the mantissa MSB; everything else is kept.
                        const uint32_t q = is_snan(v, c) ? (v | (1u << (m - 1u))) : v;
                        // Canonicalisation model: every NaN collapses to exp=31, mantissa MSB only.
                        const uint32_t k = is_nan(v, c) ? ((0x1fu << m) | (1u << (m - 1u))) : v;
                        quieted |= q << fields[c].shift;
                        canonical |= k << fields[c].shift;
                    }
                    if (finite) {
                        ++all_finite;
                        if (first_finite < 0) first_finite = static_cast<long long>(t);
                    }
                    if (snan) ++any_snan;
                    if (nan) ++any_nan;
                    if (inf && !nan) ++any_inf;
                    if (dst[t] == quieted) ++quiet_exact;
                    if (dst[t] == canonical) ++canon_exact;
                    if (total <= 8 || finite)
                        std::printf("  packed R11 diff texel=%zu src=%08x dst=%08x xor=%08x%s\n",
                                    t, src[t], dst[t], src[t] ^ dst[t],
                                    finite ? "  ALL-FINITE" : "");
                }
                std::printf("  packed R11 census: total=%zu all-finite=%zu any-sNaN=%zu "
                            "any-NaN=%zu inf-only=%zu quiet-model-exact=%zu canon-model-exact=%zu "
                            "first=%lld first-finite=%lld nan-payload-only=%zu real=%zu\n",
                            total, all_finite, any_snan, any_nan, any_inf, quiet_exact,
                            canon_exact, first_any, first_finite, payload_only, real_diffs);
            }
            CHECK(packed_executed && real_diffs == 0,
                  "shader-side R11G11B10 pack/unpack round-trips every finite and infinite f11 "
                  "code exactly, and preserves NaN-ness (payload is implementation-defined)");
        }

        // Also compare arbitrary raw f32 channel bits against the CPU's exact RNE oracle. The
        // representable-code round-trip above cannot exercise values on either side of a packing
        // boundary, negative clamping, f32 underflow, overflow, or payload truncation.
        std::vector<uint32_t> float_channels(PACKED_W * 4), expected(PACKED_W);
        const uint32_t edge_bits[] = {
            0x00000000u, 0x00000001u, 0x007fffffu, 0x00800000u,
            0x80000000u, 0xbf800000u, 0x3f800000u, 0x3f810000u,
            0x477fe000u, 0x477ff000u, 0x47800000u, 0x7f800000u,
            0xff800000u, 0x7f800001u, 0x7fc12345u, 0xffc54321u,
        };
        auto bits_float = [](uint32_t bits) {
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            return value;
        };
        for (uint32_t t = 0; t < PACKED_W; ++t) {
            for (uint32_t c = 0; c < 4; ++c)
                float_channels[t * 4 + c] = t < std::size(edge_bits)
                    ? edge_bits[(t + c * 5u) % std::size(edge_bits)]
                    : (t * 2654435761u + c * 2246822519u);
            expected[t] = static_cast<uint32_t>(float_to_f11(
                              bits_float(float_channels[t * 4 + 0]))) |
                          (static_cast<uint32_t>(float_to_f11(
                              bits_float(float_channels[t * 4 + 1]))) << 11) |
                          (static_cast<uint32_t>(float_to_f10(
                              bits_float(float_channels[t * 4 + 2]))) << 22);
        }
        std::fill(dst.begin(), dst.end(), 0x5a5a5a5au);
        ShaderResourceTable conversion_rt = packed_rt;
        for (ShaderResource& resource : conversion_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.gpu_addr = reinterpret_cast<uint64_t>(
                resource.binding == 4 ? float_channels.data() : dst.data());
            if (resource.binding == 4) {
                resource.format = DataFormat::Float32;
                resource.num_components = 4;
                resource.size = static_cast<uint32_t>(float_channels.size() * sizeof(uint32_t));
            }
        }
        const std::vector<uint32_t> conversion_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &conversion_rt);
        if (!conversion_spirv.empty()) {
            ComputeItem conversion_item;
            conversion_item.spirv = conversion_spirv;
            conversion_item.resources = std::make_shared<ShaderResourceTable>(conversion_rt);
            conversion_item.launch.threads_x = PACKED_W;
            conversion_item.launch.local_x = 64;
            conversion_item.launch.groups_x = PACKED_W / 64;
            conversion_item.launch.local_y = conversion_item.launch.local_z = 1;
            conversion_item.launch.groups_y = conversion_item.launch.groups_z = 1;
            conversion_item.code_addr = 0x590b11;
            CHECK(prosper::frontend::execute_live_compute_items({conversion_item}) &&
                      dst == expected,
                  "shader-side R11G11B10 packing matches the CPU oracle for arbitrary f32 bits");
        } else {
            CHECK(false, "mixed Float32-to-R11G11B10 storage conversion recompiles");
        }
    }

    // The backend publishes ordinary tiled texels, not hardware-compressed blocks. Prove that a
    // DCC-enabled storage destination atomically becomes the uncompressed (0xff) metadata state,
    // including replay-owned metadata that a later sampled descriptor shares by logical address.
    const uint32_t dcc_tile = static_cast<uint32_t>(TileMode::Sw64KbRX);

    // The production sampled-image path must agree with a semantic uncompressed reference, not with
    // the old compressed-base fallback. A zero base plus DCC_CLEAR_0001 logically contains
    // FP16 (0,0,0,1); copy both representations through the same generated shader and require the
    // resulting guest RGBA8 storage bytes and hashes to match exactly.
    {
        const size_t clear_source_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 8);
        const size_t clear_metadata_bytes =
            gfx10_dcc_metadata_bytes(W, 1, 1, dcc_tile, 8, true);
        std::vector<uint8_t> compressed_base(clear_source_bytes, 0);
        std::vector<uint8_t> clear_metadata(clear_metadata_bytes, 0x40);
        std::vector<uint8_t> reference_linear(W * 8, 0);
        const uint16_t one = float_to_half(1.0f);
        for (uint32_t texel = 0; texel < W; ++texel)
            std::memcpy(reference_linear.data() + texel * 8 + 6, &one, sizeof(one));
        std::vector<uint8_t> reference_tiled(clear_source_bytes, 0);
        tile_surface(reference_tiled.data(), reference_linear.data(), W, 1,
                     dcc_tile, 0, 8);
        std::vector<uint8_t> clear_dst(W * 4, 0xa5);
        std::vector<uint8_t> reference_dst(W * 4, 0x5a);

        auto sampled_clear_table = [&](std::vector<uint8_t>& source,
                                       std::vector<uint8_t>& destination,
                                       bool compressed) {
            ShaderResourceTable table = irt;
            for (ShaderResource& resource : table.resources) {
                if (resource.binding != 4 && resource.binding != 5) continue;
                resource.img_dim = 1;
                resource.width = W;
                resource.height = resource.depth = 1;
                resource.declared_mip_levels = 1;
                resource.in_mip_tail = false;
                resource.layer_stride_bytes = resource.layer_mip_offset_bytes = 0;
                if (resource.binding == 4) {
                    resource.cls = ResourceClass::Texture;
                    resource.format = DataFormat::Float16;
                    resource.num_components = 4;
                    resource.tile_mode = dcc_tile;
                    resource.gpu_addr = reinterpret_cast<uint64_t>(source.data());
                    resource.size = static_cast<uint32_t>(source.size());
                    resource.swizzle[0] = 4;
                    resource.swizzle[1] = 5;
                    resource.swizzle[2] = 6;
                    resource.swizzle[3] = 7;
                    resource.compression_enabled = compressed;
                    resource.meta_pipe_aligned = compressed;
                    resource.alpha_is_on_msb = compressed;
                    resource.metadata_addr = compressed ? 0x301758d000ull : 0;
                    resource.dcc_metadata_size = compressed ? clear_metadata.size() : 0;
                    resource.dcc_metadata_host_data = compressed ? clear_metadata.data() : nullptr;
                    resource.dcc_metadata_host_data_size = compressed ? clear_metadata.size() : 0;
                } else {
                    resource.cls = ResourceClass::StorageImage;
                    resource.format = DataFormat::Unorm8;
                    resource.num_components = 4;
                    resource.tile_mode = 0;
                    resource.gpu_addr = reinterpret_cast<uint64_t>(destination.data());
                    resource.size = static_cast<uint32_t>(destination.size());
                }
            }
            return table;
        };
        ShaderResourceTable clear_rt = sampled_clear_table(
            compressed_base, clear_dst, true);
        ShaderResourceTable reference_rt = sampled_clear_table(
            reference_tiled, reference_dst, false);
        const std::vector<uint32_t> clear_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &clear_rt);
        CHECK(!clear_spirv.empty(),
              "sampled DCC fast-clear semantic-reference kernel recompiles");
        auto run_clear = [&](ShaderResourceTable& table, uint64_t code_addr) {
            if (clear_spirv.empty()) return false;
            ComputeItem item;
            item.spirv = clear_spirv;
            item.resources = std::make_shared<ShaderResourceTable>(table);
            item.launch.threads_x = W;
            item.launch.local_x = 64;
            item.launch.groups_x = 1;
            item.launch.local_y = item.launch.local_z = 1;
            item.launch.groups_y = item.launch.groups_z = 1;
            item.code_addr = code_addr;
            return prosper::frontend::execute_live_compute_items({item});
        };
        const bool clear_ok = run_clear(clear_rt, 0x301758d1) &&
                              run_clear(reference_rt, 0x301758d2);
        std::vector<uint8_t> expected(W * 4, 0);
        for (uint32_t texel = 0; texel < W; ++texel) expected[texel * 4 + 3] = 255;
        const uint64_t clear_hash = gpu_capture_hash(clear_dst.data(), clear_dst.size());
        const uint64_t reference_hash =
            gpu_capture_hash(reference_dst.data(), reference_dst.size());
        const uint64_t expected_hash = gpu_capture_hash(expected.data(), expected.size());
        CHECK(clear_ok && clear_dst == reference_dst && clear_dst == expected &&
                  clear_hash == reference_hash && clear_hash == expected_hash,
              "sampled DCC clear output matches explicit uncompressed FP16 (0,0,0,1)");
    }

    const size_t tiled_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 4);
    const size_t metadata_bytes = gfx10_dcc_metadata_bytes(W, 1, 1, dcc_tile, 4, true);
    std::vector<uint8_t> tiled_src(tiled_bytes, 0), tiled_dst(tiled_bytes, 0);
    std::vector<uint8_t> dcc_metadata(metadata_bytes, 0x40);
    tile_surface(tiled_src.data(), img_src.data(), W, 1, dcc_tile, 0, 4);
    ShaderResourceTable dcc_rt = irt;
    for (ShaderResource& resource : dcc_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.tile_mode = dcc_tile;
        resource.size = static_cast<uint32_t>(tiled_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? tiled_src.data() : tiled_dst.data());
        if (resource.binding == 5) {
            resource.compression_enabled = true;
            resource.write_compress_enabled = true;
            resource.meta_pipe_aligned = true;
            resource.metadata_addr = 0x207cef0000ull;
            resource.dcc_metadata_size = metadata_bytes;
            resource.dcc_metadata_host_data = dcc_metadata.data();
            resource.dcc_metadata_host_data_size = dcc_metadata.size();
        }
    }
    std::vector<uint32_t> dcc_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0, &dcc_rt);
    CHECK(!dcc_spirv.empty(), "2D storage copy recompiles for a DCC-enabled tiled destination");
    if (!dcc_spirv.empty()) {
        ComputeItem dcc_item;
        dcc_item.spirv = dcc_spirv;
        dcc_item.resources = std::make_shared<ShaderResourceTable>(dcc_rt);
        dcc_item.launch.threads_x = W;
        dcc_item.launch.local_x = 64;
        dcc_item.launch.groups_x = 1;
        dcc_item.launch.local_y = dcc_item.launch.local_z = 1;
        dcc_item.launch.groups_y = dcc_item.launch.groups_z = 1;
        dcc_item.code_addr = 0x719dcc;
        CHECK(prosper::frontend::execute_live_compute_items({dcc_item}),
              "live backend writes the tiled DCC storage image");
        std::vector<uint8_t> dcc_linear(W * 4, 0);
        detile_surface(dcc_linear.data(), tiled_dst.data(), W, 1, dcc_tile, 0, 4);
        CHECK(dcc_linear == img_src,
              "DCC storage writeback preserves the producer's tiled base texels");
        CHECK(std::all_of(dcc_metadata.begin(), dcc_metadata.end(),
                          [](uint8_t code) { return code == 0xff; }),
              "DCC storage writeback publishes uniform uncompressed metadata");
    }

    // Two storage views may share base texels while only one carries DCC state. They are not safe
    // Vulkan-image aliases: collapsing the compressed view onto an uncompressed owner drops its
    // separate metadata writeback obligation and leaves later sampled users reading stale DCC.
    std::vector<uint8_t> mixed_alias_base = tiled_src;
    std::vector<uint8_t> mixed_alias_metadata(metadata_bytes, 0x40);
    ShaderResourceTable mixed_alias_rt = dcc_rt;
    for (ShaderResource& resource : mixed_alias_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.gpu_addr = reinterpret_cast<uint64_t>(mixed_alias_base.data());
        if (resource.binding == 5) {
            resource.metadata_addr = 0x207cf00000ull;
            resource.dcc_metadata_host_data = mixed_alias_metadata.data();
            resource.dcc_metadata_host_data_size = mixed_alias_metadata.size();
        }
    }
    const std::vector<uint32_t> mixed_alias_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &mixed_alias_rt);
    CHECK(!mixed_alias_spirv.empty(), "mixed DCC storage-alias kernel recompiles");
    if (!mixed_alias_spirv.empty()) {
        ComputeItem mixed_alias_item;
        mixed_alias_item.spirv = mixed_alias_spirv;
        mixed_alias_item.resources = std::make_shared<ShaderResourceTable>(mixed_alias_rt);
        mixed_alias_item.launch.threads_x = W;
        mixed_alias_item.launch.local_x = 64;
        mixed_alias_item.launch.groups_x = 1;
        mixed_alias_item.launch.local_y = mixed_alias_item.launch.local_z = 1;
        mixed_alias_item.launch.groups_y = mixed_alias_item.launch.groups_z = 1;
        mixed_alias_item.code_addr = 0x719dcd;
        CHECK(prosper::frontend::execute_live_compute_items({mixed_alias_item}),
              "mixed compressed/uncompressed storage views execute independently");
        CHECK(std::all_of(mixed_alias_metadata.begin(), mixed_alias_metadata.end(),
                          [](uint8_t code) { return code == 0xff; }),
              "compressed storage alias retains its DCC writeback obligation");
    }

    // Exercise the same production path with a tiled 2D guest surface. The kernel copies row zero;
    // every untouched row must survive upload/readback/retiling exactly. This guards the inverse
    // address mapping used by live tiled StorageImage dispatches, not only the pure tile helper.
    constexpr uint32_t TILED_H = 19;
    constexpr uint32_t tiled2d_mode = static_cast<uint32_t>(TileMode::Sw4KbS);
    std::vector<uint8_t> tiled2d_src_linear(W * TILED_H * 4);
    std::vector<uint8_t> tiled2d_dst_initial(W * TILED_H * 4, 0x5a);
    for (size_t i = 0; i < tiled2d_src_linear.size(); ++i)
        tiled2d_src_linear[i] = static_cast<uint8_t>(i * 41 + 7);
    const size_t tiled2d_bytes = tiled_surface_bytes(W, TILED_H, tiled2d_mode, 0, 4);
    std::vector<uint8_t> tiled2d_src(tiled2d_bytes, 0), tiled2d_dst(tiled2d_bytes, 0);
    tile_surface(tiled2d_src.data(), tiled2d_src_linear.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);
    tile_surface(tiled2d_dst.data(), tiled2d_dst_initial.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);

    ShaderResourceTable tiled2d_rt = irt;
    for (ShaderResource& resource : tiled2d_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.width = W;
        resource.height = TILED_H;
        resource.depth = 1;
        resource.tile_mode = tiled2d_mode;
        resource.size = static_cast<uint32_t>(tiled2d_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? tiled2d_src.data() : tiled2d_dst.data());
    }
    const std::vector<uint32_t> tiled2d_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &tiled2d_rt);
    CHECK(!tiled2d_spirv.empty(), "tiled 2D storage-image copy kernel recompiles");
    if (!tiled2d_spirv.empty()) {
        ComputeItem tiled2d_item;
        tiled2d_item.spirv = tiled2d_spirv;
        tiled2d_item.resources = std::make_shared<ShaderResourceTable>(tiled2d_rt);
        tiled2d_item.launch.threads_x = W;
        tiled2d_item.launch.local_x = 64;
        tiled2d_item.launch.groups_x = 1;
        tiled2d_item.launch.local_y = tiled2d_item.launch.local_z = 1;
        tiled2d_item.launch.groups_y = tiled2d_item.launch.groups_z = 1;
        tiled2d_item.code_addr = 0x590592;
        CHECK(prosper::frontend::execute_live_compute_items({tiled2d_item}),
              "production backend executes a tiled 2D storage-image dispatch");

        std::vector<uint8_t> tiled2d_result(W * TILED_H * 4, 0);
        detile_surface(tiled2d_result.data(), tiled2d_dst.data(), W, TILED_H,
                       tiled2d_mode, 0, 4);
        std::vector<uint8_t> tiled2d_expected = tiled2d_dst_initial;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, tiled2d_expected.begin());
        CHECK(tiled2d_result == tiled2d_expected,
              "tiled 2D storage-image writeback matches the linear reference byte-exactly");
    }

    // Some descriptors declare a one-layer 2D array while the MIMG instruction itself uses DIM=2D.
    // Preserve that established byte-identical lowering instead of creating a 2D-array Vulkan view
    // that disagrees with the generated SPIR-V image type.
    std::fill(tiled2d_dst.begin(), tiled2d_dst.end(), 0);
    tile_surface(tiled2d_dst.data(), tiled2d_dst_initial.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);
    ShaderResourceTable single_layer_array_rt = tiled2d_rt;
    for (ShaderResource& resource : single_layer_array_rt.resources)
        if (resource.binding == 4 || resource.binding == 5) resource.img_dim = 5;
    const std::vector<uint32_t> single_layer_array_spirv = recompile_valu(
        image_copy_2d, std::size(image_copy_2d), 1, 0, &single_layer_array_rt);
    CHECK(!single_layer_array_spirv.empty(),
          "DIM=2D kernel recompiles over a byte-identical one-layer array descriptor");
    if (!single_layer_array_spirv.empty()) {
        ComputeItem item;
        item.spirv = single_layer_array_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(single_layer_array_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x590599;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "one-layer array descriptor executes through its DIM=2D Vulkan view");
        std::vector<uint8_t> actual(W * TILED_H * 4, 0);
        detile_surface(actual.data(), tiled2d_dst.data(), W, TILED_H, tiled2d_mode, 0, 4);
        std::vector<uint8_t> expected = tiled2d_dst_initial;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, expected.begin());
        CHECK(actual == expected,
              "one-layer array descriptor retains byte-exact 2D storage writeback");
    }

    // A guest allocation may be cube-shaped while the instruction and generated SPIR-V deliberately
    // view only face zero as a plain 2D texture. This is a view contract, not a request to upload all
    // six faces into a 2D Vulkan image. Plucky's lighting setup uses exactly this form.
    std::vector<uint8_t> cube_faces(tiled2d_bytes * 6u, 0xa7);
    std::copy(tiled2d_src.begin(), tiled2d_src.end(), cube_faces.begin());
    std::fill(tiled2d_dst.begin(), tiled2d_dst.end(), 0);
    tile_surface(tiled2d_dst.data(), tiled2d_dst_initial.data(), W, TILED_H,
                 tiled2d_mode, 0, 4);
    ShaderResourceTable cube_face_rt = tiled2d_rt;
    for (ShaderResource& resource : cube_face_rt.resources) {
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.img_dim = 3;
            resource.depth = 6;
            resource.size = static_cast<uint32_t>(cube_faces.size());
            resource.gpu_addr = reinterpret_cast<uint64_t>(cube_faces.data());
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else if (resource.binding == 5) {
            resource.gpu_addr = reinterpret_cast<uint64_t>(tiled2d_dst.data());
        }
    }
    const std::vector<uint32_t> cube_face_spirv = recompile_valu(
        image_copy_2d, std::size(image_copy_2d), 1, 0, &cube_face_rt);
    CHECK(!cube_face_spirv.empty(), "DIM=2D kernel recompiles over a cube allocation");
    const auto cube_face_report = validate_spirv_descriptor_interface(
        cube_face_spirv, &cube_face_rt, 0, SpirvShaderStage::Compute);
    const auto cube_face_descriptor = std::find_if(
        cube_face_report.descriptors.begin(), cube_face_report.descriptors.end(),
        [](const SpirvDescriptorBinding& descriptor) { return descriptor.binding == 4; });
    CHECK(cube_face_report.ok() && cube_face_descriptor != cube_face_report.descriptors.end() &&
              cube_face_descriptor->image_dim == 1 && !cube_face_descriptor->image_arrayed,
          "generated shader reflects the cube allocation binding as a non-array 2D image");
    if (!cube_face_spirv.empty()) {
        ComputeItem item;
        item.spirv = cube_face_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(cube_face_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x59059a;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "production backend binds cube face zero through the shader-declared 2D view");
        std::vector<uint8_t> actual(W * TILED_H * 4, 0);
        detile_surface(actual.data(), tiled2d_dst.data(), W, TILED_H, tiled2d_mode, 0, 4);
        std::vector<uint8_t> expected = tiled2d_dst_initial;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, expected.begin());
        CHECK(actual == expected,
              "cube face zero reaches the 2D shader view without reading later faces");
    }

    // Plucky's lighting producer writes a true 3D SW_64KB_S RGBA16 surface. Exercise the full
    // Vulkan storage-image path, including S3 detile/upload and readback/retile, while preserving
    // every voxel outside the row touched by this small copy kernel.
    constexpr uint32_t VOLUME_W = W, VOLUME_H = 32, VOLUME_D = 32;
    constexpr uint32_t volume_mode = static_cast<uint32_t>(TileMode::Sw64KbS);
    constexpr uint32_t volume_bpe = 8;
    const size_t volume_linear_bytes =
        static_cast<size_t>(VOLUME_W) * VOLUME_H * VOLUME_D * volume_bpe;
    std::vector<uint8_t> volume_src_linear(volume_linear_bytes);
    std::vector<uint8_t> volume_dst_initial(volume_linear_bytes);
    for (size_t i = 0; i < volume_linear_bytes / 2; ++i) {
        const uint16_t src_value = static_cast<uint16_t>((i * 313u) & 0xffffu);
        const uint16_t dst_value = static_cast<uint16_t>((i * 197u + 17u) & 0xffffu);
        std::memcpy(volume_src_linear.data() + i * 2, &src_value, 2);
        std::memcpy(volume_dst_initial.data() + i * 2, &dst_value, 2);
    }
    const size_t volume_tiled_bytes = tiled_volume_bytes(
        VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
    std::vector<uint8_t> volume_src(volume_tiled_bytes, 0);
    std::vector<uint8_t> volume_dst(volume_tiled_bytes, 0);
    tile_volume(volume_src.data(), volume_src.size(), volume_src_linear.data(),
                VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
    tile_volume(volume_dst.data(), volume_dst.size(), volume_dst_initial.data(),
                VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
    ShaderResourceTable volume_rt = tiled2d_rt;
    for (ShaderResource& resource : volume_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 2;
        resource.format = DataFormat::Unorm16;
        resource.num_components = 4;
        resource.width = VOLUME_W;
        resource.height = VOLUME_H;
        resource.depth = VOLUME_D;
        resource.tile_mode = volume_mode;
        resource.size = static_cast<uint32_t>(volume_tiled_bytes);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? volume_src.data() : volume_dst.data());
    }
    const std::vector<uint32_t> volume_spirv = recompile_valu(
        image_copy_3d, std::size(image_copy_3d), 1, 0, &volume_rt);
    CHECK(!volume_spirv.empty(), "SW_64KB_S3 storage-volume copy kernel recompiles");
    if (!volume_spirv.empty()) {
        ComputeItem volume_item;
        volume_item.spirv = volume_spirv;
        volume_item.resources = std::make_shared<ShaderResourceTable>(volume_rt);
        volume_item.launch.threads_x = VOLUME_W;
        volume_item.launch.local_x = 64;
        volume_item.launch.groups_x = 1;
        volume_item.launch.local_y = volume_item.launch.local_z = 1;
        volume_item.launch.groups_y = volume_item.launch.groups_z = 1;
        volume_item.code_addr = 0x306a150000;
        CHECK(prosper::frontend::execute_live_compute_items({volume_item}),
              "production backend executes an SW_64KB_S3 storage-volume dispatch");
        std::vector<uint8_t> volume_result(volume_linear_bytes, 0);
        detile_volume(volume_result.data(), volume_dst.data(), volume_dst.size(),
                      VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
        std::vector<uint8_t> volume_expected = volume_dst_initial;
        std::copy_n(volume_src_linear.begin(), VOLUME_W * volume_bpe,
                    volume_expected.begin());
        // #1681: this pair failed on the CI runner's Mesa 25.2.8 lavapipe while passing on Mesa
        // 26.1.4 lavapipe and on RADV, and a bare vector compare cannot say why. Report whether the
        // difference lands in the one row the dispatch writes or in the voxels it must preserve —
        // "wrote the wrong data" and "failed to preserve untouched data" are different defects.
        const size_t volume_written_bytes = VOLUME_W * volume_bpe;
        auto report_volume_diff = [&](const char* tag) {
            if (volume_result == volume_expected) return;
            size_t diff = 0, in_written = 0, first = volume_result.size();
            for (size_t i = 0; i < volume_result.size() && i < volume_expected.size(); ++i) {
                if (volume_result[i] == volume_expected[i]) continue;
                ++diff;
                if (i < volume_written_bytes) ++in_written;
                if (first == volume_result.size()) first = i;
            }
            std::printf("  volume diff [%s]: bytes=%zu/%zu in-written-row=%zu untouched=%zu "
                        "first=%zu got=%02x want=%02x\n",
                        tag, diff, volume_result.size(), in_written, diff - in_written, first,
                        first < volume_result.size() ? volume_result[first] : 0,
                        first < volume_expected.size() ? volume_expected[first] : 0);
        };
        report_volume_diff("S3 writeback");
        CHECK(volume_result == volume_expected,
              "S3 writeback updates one row and preserves all untouched 3D voxels");

        // Renderer ownership is a concrete 2D image identity, not merely an address. Simulate a
        // stale/recycled 2D cache entry at the volume destination and require the layered descriptor
        // to use its valid guest backing without trying to read an impossible 2D snapshot.
        tile_volume(volume_dst.data(), volume_dst.size(), volume_dst_initial.data(),
                    VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
        const uint64_t volume_dst_addr = reinterpret_cast<uint64_t>(volume_dst.data());
        bool layered_reader_called = false;
        set_live_target_query(
            [volume_dst_addr](uint64_t addr) { return addr == volume_dst_addr; });
        set_live_target_reader(
            [&](uint64_t, LiveTargetSnapshot&) {
                layered_reader_called = true;
                return false;
            });
        CHECK(prosper::frontend::execute_live_compute_items({volume_item}),
              "layered resource executes when its base aliases a cached 2D render target");
        std::fill(volume_result.begin(), volume_result.end(), 0);
        detile_volume(volume_result.data(), volume_dst.data(), volume_dst.size(),
                      VOLUME_W, VOLUME_H, VOLUME_D, volume_mode, volume_bpe);
        report_volume_diff("3D descriptor");
        CHECK(!layered_reader_called && volume_result == volume_expected,
              "3D descriptor rejects address-only 2D ownership and preserves guest-backed voxels");
        set_live_target_reader({});
        set_live_target_query({});
    }

    // A selected mip of a 2D array is not stored as selected-level0, selected-level1, ... . Every
    // array slice owns a complete tail-first mip chain. Exercise a real DIM=2D_ARRAY Vulkan image,
    // seed all three selected subresources through their stride/offset, modify layer two, then prove
    // writeback preserves the other layers and every byte outside the selected level.
    constexpr uint32_t ARRAY_LAYERS = 3;
    constexpr uint32_t ARRAY_BASE_W = W * 2;
    constexpr uint32_t ARRAY_BASE_H = TILED_H * 2;
    constexpr uint32_t ARRAY_MAX_MIP = 4;
    constexpr uint32_t ARRAY_LEVEL = 1;
    const TiledMipLevelLayout array_level = tiled_mip_level_layout(
        ARRAY_BASE_W, ARRAY_BASE_H, 4, tiled2d_mode, ARRAY_MAX_MIP, ARRAY_LEVEL);
    const size_t array_stride = tiled_mip_chain_bytes(
        ARRAY_BASE_W, ARRAY_BASE_H, 4, tiled2d_mode, ARRAY_MAX_MIP);
    const size_t array_selected_bytes = tiled_surface_bytes(W, TILED_H, tiled2d_mode, 0, 4);
    std::vector<uint8_t> array_src(array_stride * ARRAY_LAYERS, 0x31);
    std::vector<uint8_t> array_dst(array_stride * ARRAY_LAYERS, 0xA7);
    std::vector<std::vector<uint8_t>> array_src_linear(
        ARRAY_LAYERS, std::vector<uint8_t>(W * TILED_H * 4));
    std::vector<std::vector<uint8_t>> array_dst_initial(
        ARRAY_LAYERS, std::vector<uint8_t>(W * TILED_H * 4));
    for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
        for (size_t i = 0; i < array_src_linear[layer].size(); ++i) {
            array_src_linear[layer][i] = static_cast<uint8_t>(i * 17 + layer * 53 + 5);
            array_dst_initial[layer][i] = static_cast<uint8_t>(i * 29 + layer * 71 + 11);
        }
        tile_surface(array_src.data() + layer * array_stride + array_level.byte_offset,
                     array_src_linear[layer].data(), W, TILED_H, tiled2d_mode, 0, 4);
        tile_surface(array_dst.data() + layer * array_stride + array_level.byte_offset,
                     array_dst_initial[layer].data(), W, TILED_H, tiled2d_mode, 0, 4);
    }
    const std::vector<uint8_t> array_dst_outside_before = array_dst;
    ShaderResourceTable array_rt = tiled2d_rt;
    for (ShaderResource& resource : array_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 5;
        resource.depth = ARRAY_LAYERS;
        resource.size = W * TILED_H * ARRAY_LAYERS * 4;
        resource.layer_stride_bytes = static_cast<uint32_t>(array_stride);
        resource.layer_mip_offset_bytes = static_cast<uint32_t>(array_level.byte_offset);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? array_src.data() : array_dst.data());
    }
    const std::vector<uint32_t> array_spirv = recompile_valu(
        image_copy_2d_array, std::size(image_copy_2d_array), 1, 0, &array_rt);
    CHECK(array_level.supported && !array_level.in_tail && array_stride != 0 &&
              !array_spirv.empty(),
          "selected-mip 2D-array storage copy recompiles with a proven slice layout");
    if (!array_spirv.empty()) {
        ComputeItem array_item;
        array_item.spirv = array_spirv;
        array_item.resources = std::make_shared<ShaderResourceTable>(array_rt);
        array_item.launch.threads_x = W;
        array_item.launch.local_x = 64;
        array_item.launch.groups_x = 1;
        array_item.launch.local_y = array_item.launch.local_z = 1;
        array_item.launch.groups_y = array_item.launch.groups_z = 1;
        array_item.code_addr = 0x590597;
        CHECK(prosper::frontend::execute_live_compute_items({array_item}),
              "production backend executes a selected-mip 2D-array storage dispatch");
        bool selected_levels_exact = true;
        for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
            std::vector<uint8_t> actual(W * TILED_H * 4, 0);
            detile_surface(actual.data(),
                           array_dst.data() + layer * array_stride + array_level.byte_offset,
                           W, TILED_H, tiled2d_mode, 0, 4);
            std::vector<uint8_t> expected = array_dst_initial[layer];
            if (layer == 2)
                std::copy_n(array_src_linear[2].begin(), W * 4, expected.begin());
            selected_levels_exact &= actual == expected;
        }
        CHECK(selected_levels_exact,
              "array writeback updates addressed layer two and preserves sibling selected mips");
        bool outside_unchanged = true;
        for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
            const size_t selected_begin = layer * array_stride + array_level.byte_offset;
            const size_t selected_end = selected_begin + array_selected_bytes;
            for (size_t i = layer * array_stride; i < (layer + 1) * array_stride; ++i)
                if ((i < selected_begin || i >= selected_end) &&
                    array_dst[i] != array_dst_outside_before[i])
                    outside_unchanged = false;
        }
        CHECK(outside_unchanged,
              "array writeback leaves sibling mips and inter-level padding byte-exact");
    }

    // Linear mip chains use a 256-byte-aligned row pitch independently inside every slice. Use a
    // 65-pixel selected level (260 tight bytes, 512-byte guest pitch) so a bulk tight memcpy would
    // visibly cross both row and layer boundaries. The exact-backing comparison also proves that
    // row padding and sibling levels survive writeback.
    constexpr uint32_t LINEAR_W = W + 1;
    const TiledMipLevelLayout linear_array_level = tiled_mip_level_layout(
        LINEAR_W * 2, ARRAY_BASE_H, 4, 0, ARRAY_MAX_MIP, ARRAY_LEVEL);
    const size_t linear_array_stride = tiled_mip_chain_bytes(
        LINEAR_W * 2, ARRAY_BASE_H, 4, 0, ARRAY_MAX_MIP);
    const size_t linear_array_pitch = linear_sampled_row_pitch(LINEAR_W, 4);
    std::vector<uint8_t> linear_array_src(linear_array_stride * ARRAY_LAYERS, 0x43);
    std::vector<uint8_t> linear_array_dst(linear_array_stride * ARRAY_LAYERS, 0xB9);
    std::vector<uint8_t> linear_array_expected = linear_array_dst;
    for (uint32_t layer = 0; layer < ARRAY_LAYERS; ++layer) {
        for (uint32_t y = 0; y < TILED_H; ++y) {
            uint8_t* src_row = linear_array_src.data() + layer * linear_array_stride +
                linear_array_level.byte_offset + y * linear_array_pitch;
            uint8_t* dst_row = linear_array_dst.data() + layer * linear_array_stride +
                linear_array_level.byte_offset + y * linear_array_pitch;
            for (uint32_t x = 0; x < LINEAR_W * 4; ++x) {
                src_row[x] = static_cast<uint8_t>(x * 17 + y * 31 + layer * 59 + 3);
                dst_row[x] = static_cast<uint8_t>(x * 23 + y * 37 + layer * 67 + 9);
            }
        }
    }
    linear_array_expected = linear_array_dst;
    const size_t linear_array_layer_two = 2u * linear_array_stride;
    std::memcpy(linear_array_expected.data() + linear_array_layer_two +
                    linear_array_level.byte_offset,
                linear_array_src.data() + linear_array_layer_two +
                    linear_array_level.byte_offset,
                W * 4);
    ShaderResourceTable linear_array_rt = tiled2d_rt;
    for (ShaderResource& resource : linear_array_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 5;
        resource.width = LINEAR_W;
        resource.height = TILED_H;
        resource.depth = ARRAY_LAYERS;
        resource.tile_mode = 0;
        resource.size = LINEAR_W * TILED_H * ARRAY_LAYERS * 4;
        resource.layer_stride_bytes = static_cast<uint32_t>(linear_array_stride);
        resource.layer_mip_offset_bytes = static_cast<uint32_t>(linear_array_level.byte_offset);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? linear_array_src.data() : linear_array_dst.data());
    }
    const std::vector<uint32_t> linear_array_spirv = recompile_valu(
        image_copy_2d_array, std::size(image_copy_2d_array), 1, 0, &linear_array_rt);
    CHECK(linear_array_level.supported && linear_array_stride != 0 &&
              linear_array_pitch == 512 && !linear_array_spirv.empty(),
          "linear selected-mip array exposes its aligned per-row and per-slice layout");
    if (!linear_array_spirv.empty()) {
        ComputeItem linear_array_item;
        linear_array_item.spirv = linear_array_spirv;
        linear_array_item.resources = std::make_shared<ShaderResourceTable>(linear_array_rt);
        linear_array_item.launch.threads_x = W;
        linear_array_item.launch.local_x = 64;
        linear_array_item.launch.groups_x = 1;
        linear_array_item.launch.local_y = linear_array_item.launch.local_z = 1;
        linear_array_item.launch.groups_y = linear_array_item.launch.groups_z = 1;
        linear_array_item.code_addr = 0x590598;
        CHECK(prosper::frontend::execute_live_compute_items({linear_array_item}),
              "production backend executes a selected-mip linear 2D-array dispatch");
        CHECK(linear_array_dst == linear_array_expected,
              "linear array writeback preserves aligned rows, sibling slices, mips, and padding");
    }

    // Exercise the wider element mappings that become default-on with tiled storage. Float16 uses
    // finite values so half->float->half is exact; Float32 follows the backend's raw channel contract.
    auto wide_tiled_roundtrip = [&](DataFormat format, uint32_t bytes_per_texel,
                                    uint64_t code_addr) {
        const size_t linear_bytes = static_cast<size_t>(W) * TILED_H * bytes_per_texel;
        std::vector<uint8_t> src_linear(linear_bytes, 0);
        std::vector<uint8_t> dst_initial(linear_bytes, 0);
        if (format == DataFormat::Float16) {
            for (size_t i = 0; i < linear_bytes / 2; ++i) {
                const uint16_t src_half = float_to_half(
                    static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f);
                const uint16_t dst_half = float_to_half(
                    static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f);
                std::memcpy(src_linear.data() + i * 2, &src_half, sizeof(src_half));
                std::memcpy(dst_initial.data() + i * 2, &dst_half, sizeof(dst_half));
            }
        } else {
            for (size_t i = 0; i < linear_bytes / 4; ++i) {
                const uint32_t src_word = static_cast<uint32_t>(i * 2654435761u + 0x1020304u);
                const uint32_t dst_word = static_cast<uint32_t>(i * 2246822519u + 0x5060708u);
                std::memcpy(src_linear.data() + i * 4, &src_word, sizeof(src_word));
                std::memcpy(dst_initial.data() + i * 4, &dst_word, sizeof(dst_word));
            }
        }
        const size_t tiled_bytes_wide = tiled_surface_bytes(
            W, TILED_H, tiled2d_mode, 0, bytes_per_texel);
        std::vector<uint8_t> src_tiled(tiled_bytes_wide, 0);
        std::vector<uint8_t> dst_tiled(tiled_bytes_wide, 0);
        tile_surface(src_tiled.data(), src_linear.data(), W, TILED_H,
                     tiled2d_mode, 0, bytes_per_texel);
        tile_surface(dst_tiled.data(), dst_initial.data(), W, TILED_H,
                     tiled2d_mode, 0, bytes_per_texel);
        ShaderResourceTable wide_rt = tiled2d_rt;
        for (ShaderResource& resource : wide_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.format = format;
            resource.num_components = 4;
            resource.size = static_cast<uint32_t>(tiled_bytes_wide);
            resource.gpu_addr = reinterpret_cast<uint64_t>(
                resource.binding == 4 ? src_tiled.data() : dst_tiled.data());
        }
        const std::vector<uint32_t> wide_spirv = recompile_valu(
            image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
            &wide_rt);
        if (wide_spirv.empty()) return false;
        ComputeItem wide_item;
        wide_item.spirv = wide_spirv;
        wide_item.resources = std::make_shared<ShaderResourceTable>(wide_rt);
        wide_item.launch.threads_x = W;
        wide_item.launch.local_x = 64;
        wide_item.launch.groups_x = 1;
        wide_item.launch.local_y = wide_item.launch.local_z = 1;
        wide_item.launch.groups_y = wide_item.launch.groups_z = 1;
        wide_item.code_addr = code_addr;
        if (!prosper::frontend::execute_live_compute_items({wide_item})) return false;
        std::vector<uint8_t> result(linear_bytes, 0);
        detile_surface(result.data(), dst_tiled.data(), W, TILED_H,
                       tiled2d_mode, 0, bytes_per_texel);
        std::vector<uint8_t> expected = dst_initial;
        std::copy_n(src_linear.begin(), static_cast<size_t>(W) * bytes_per_texel,
                    expected.begin());
        return result == expected;
    };
    CHECK(wide_tiled_roundtrip(DataFormat::Float16, 8, 0x590595),
          "tiled Float16x4 storage writeback is byte-exact");
    CHECK(wide_tiled_roundtrip(DataFormat::Float32, 16, 0x590596),
          "tiled Float32x4 storage writeback is byte-exact");

    ShaderResourceTable unsupported_tiled_rt = tiled2d_rt;
    for (ShaderResource& resource : unsupported_tiled_rt.resources)
        if (resource.binding == 4 || resource.binding == 5) resource.tile_mode = 6;
    const std::vector<uint32_t> unsupported_tiled_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &unsupported_tiled_rt);
    CHECK(!unsupported_tiled_spirv.empty(), "unsupported tiled storage kernel still recompiles");
    if (!unsupported_tiled_spirv.empty()) {
        ComputeItem unsupported_tiled_item;
        unsupported_tiled_item.spirv = unsupported_tiled_spirv;
        unsupported_tiled_item.resources =
            std::make_shared<ShaderResourceTable>(unsupported_tiled_rt);
        unsupported_tiled_item.launch.threads_x = W;
        unsupported_tiled_item.launch.local_x = 64;
        unsupported_tiled_item.launch.groups_x = 1;
        unsupported_tiled_item.launch.local_y = unsupported_tiled_item.launch.local_z = 1;
        unsupported_tiled_item.launch.groups_y = unsupported_tiled_item.launch.groups_z = 1;
        unsupported_tiled_item.code_addr = 0x590594;
        CHECK(!prosper::frontend::execute_live_compute_items({unsupported_tiled_item}),
              "unknown nonzero tile mode is rejected before storage writeback");
    }

    // A renderer-owned color target is newer than its guest backing. Recompile the same copy kernel
    // with a sampled source, publish deliberately different live pixels, and prove the production
    // backend imports the immutable renderer snapshot instead of either skipping or reading stale RAM.
    std::vector<uint8_t> stale_rtt(W * 4, 0);
    auto live_rtt = std::make_shared<std::vector<uint8_t>>(W * 4);
    std::vector<uint8_t> live_dst(W * 4, 0xEE);
    for (uint32_t i = 0; i < W * 4; ++i) (*live_rtt)[i] = static_cast<uint8_t>(i * 29 + 11);
    ShaderResourceTable live_rt = irt;
    for (ShaderResource& resource : live_rt.resources) {
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.img_dim = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(stale_rtt.data());
            resource.host_data = nullptr;
            resource.host_data_size = 0;
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else if (resource.binding == 5) {
            resource.img_dim = 1;
            resource.gpu_addr = reinterpret_cast<uint64_t>(live_dst.data());
            resource.host_data = nullptr;
            resource.host_data_size = 0;
        }
    }
    const uint64_t live_addr = reinterpret_cast<uint64_t>(stale_rtt.data());
    set_live_target_query([live_addr](uint64_t addr) { return addr == live_addr; });
    set_live_target_reader(
        [live_addr, live_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != live_addr) return false;
            snapshot.width = W;
            snapshot.height = 1;
            snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = live_rtt;
            return true;
        });
    std::vector<uint32_t> live_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &live_rt);
    CHECK(!live_spirv.empty(), "sampled-image copy kernel recompiles for renderer RTT import");
    if (!live_spirv.empty()) {
        ComputeItem live_item;
        live_item.spirv = live_spirv;
        live_item.resources = std::make_shared<ShaderResourceTable>(live_rt);
        live_item.launch.threads_x = W;
        live_item.launch.local_x = 64;
        live_item.launch.groups_x = 1;
        live_item.launch.local_y = live_item.launch.local_z = 1;
        live_item.launch.groups_y = live_item.launch.groups_z = 1;
        live_item.code_addr = 0x590591;
        CHECK(prosper::frontend::execute_live_compute_items({live_item}),
              "live backend executes a dispatch sampling a renderer-owned RTT");
        CHECK(live_dst == *live_rtt,
              "sampled renderer RTT pixels reach storage-image writeback byte-exactly");
        CHECK(live_dst != stale_rtt, "renderer RTT import does not use stale guest backing");
    }

    // A renderer target can become a writable storage image before its pixels have been materialized
    // in guest RAM. Seed the dispatch from the immutable renderer snapshot, modify row zero, and
    // prove writeback preserves every untouched row while publishing a cache-invalidation write.
    std::vector<uint8_t> writable_rtt_guest(tiled2d_bytes, 0);
    auto writable_rtt = std::make_shared<std::vector<uint8_t>>(W * TILED_H * 4);
    for (size_t i = 0; i < writable_rtt->size(); ++i)
        (*writable_rtt)[i] = static_cast<uint8_t>(i * 17 + 3);
    ShaderResourceTable writable_rt = tiled2d_rt;
    for (ShaderResource& resource : writable_rt.resources) {
        if (resource.binding == 5) {
            resource.gpu_addr = reinterpret_cast<uint64_t>(writable_rtt_guest.data());
            resource.size = static_cast<uint32_t>(writable_rtt_guest.size());
        }
    }
    const uint64_t writable_addr = reinterpret_cast<uint64_t>(writable_rtt_guest.data());
    set_live_target_query([writable_addr](uint64_t addr) { return addr == writable_addr; });
    set_live_target_reader(
        [writable_addr, writable_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != writable_addr) return false;
            snapshot.width = W;
            snapshot.height = TILED_H;
            snapshot.format = LiveTargetPixelFormat::Rgba8Unorm;
            snapshot.pixels = writable_rtt;
            return true;
        });
    bool writable_rtt_published = false;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
        writable_rtt_published |= addr == writable_addr && size == writable_rtt_guest.size();
    });
    const std::vector<uint32_t> writable_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0,
        &writable_rt);
    CHECK(!writable_spirv.empty(), "writable renderer RTT copy kernel recompiles");
    if (!writable_spirv.empty()) {
        ComputeItem writable_item;
        writable_item.spirv = writable_spirv;
        writable_item.resources = std::make_shared<ShaderResourceTable>(writable_rt);
        writable_item.launch.threads_x = W;
        writable_item.launch.local_x = 64;
        writable_item.launch.groups_x = 1;
        writable_item.launch.local_y = writable_item.launch.local_z = 1;
        writable_item.launch.groups_y = writable_item.launch.groups_z = 1;
        writable_item.code_addr = 0x590593;
        CHECK(prosper::frontend::execute_live_compute_items({writable_item}),
              "live backend executes a dispatch writing a renderer-owned RTT");
        std::vector<uint8_t> writable_result(W * TILED_H * 4, 0);
        detile_surface(writable_result.data(), writable_rtt_guest.data(), W, TILED_H,
                       tiled2d_mode, 0, 4);
        std::vector<uint8_t> writable_expected = *writable_rtt;
        std::copy_n(tiled2d_src_linear.begin(), W * 4, writable_expected.begin());
        CHECK(writable_result == writable_expected,
              "writable renderer RTT seeds from live pixels and preserves untouched rows");
        CHECK(writable_rtt_published,
              "writable renderer RTT publishes guest writeback for cache invalidation");
    }
    set_guest_gpu_write_observer({});
    set_live_target_reader({});
    set_live_target_query({});

    // The exact packed fallback must obey the same authority rule. Keep stale zeroes in guest RAM,
    // publish distinct R11G11B10 renderer pixels, overwrite only row zero, and require every other
    // row to survive from the snapshot rather than the stale allocation.
    std::vector<uint32_t> packed_rtt_source(W * TILED_H);
    std::vector<uint32_t> packed_rtt_guest(W * TILED_H, 0);
    auto packed_rtt = std::make_shared<std::vector<uint8_t>>(W * TILED_H * sizeof(uint32_t));
    std::vector<uint32_t> packed_rtt_expected(W * TILED_H);
    for (size_t t = 0; t < packed_rtt_source.size(); ++t) {
        packed_rtt_source[t] = static_cast<uint32_t>(float_to_f11((t % 19u) * 0.25f)) |
            (static_cast<uint32_t>(float_to_f11((t % 23u) * 0.375f)) << 11) |
            (static_cast<uint32_t>(float_to_f10((t % 29u) * 0.5f)) << 22);
        const uint32_t live_word = static_cast<uint32_t>(float_to_f11((t % 31u) * 0.125f)) |
            (static_cast<uint32_t>(float_to_f11((t % 37u) * 0.1875f)) << 11) |
            (static_cast<uint32_t>(float_to_f10((t % 41u) * 0.3125f)) << 22);
        std::memcpy(packed_rtt->data() + t * sizeof(live_word), &live_word, sizeof(live_word));
        packed_rtt_expected[t] = live_word;
    }
    std::copy_n(packed_rtt_source.begin(), W, packed_rtt_expected.begin());
    ShaderResourceTable packed_writable_rt = tiled2d_rt;
    for (ShaderResource& resource : packed_writable_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.format = DataFormat::Float10_11_11;
        resource.num_components = 3;
        resource.tile_mode = 0;
        resource.size = W * TILED_H * sizeof(uint32_t);
        resource.gpu_addr = reinterpret_cast<uint64_t>(
            resource.binding == 4 ? packed_rtt_source.data() : packed_rtt_guest.data());
    }
    const uint64_t packed_writable_addr =
        reinterpret_cast<uint64_t>(packed_rtt_guest.data());
    set_live_target_query(
        [packed_writable_addr](uint64_t addr) { return addr == packed_writable_addr; });
    set_live_target_reader(
        [packed_writable_addr, packed_rtt](uint64_t addr, LiveTargetSnapshot& snapshot) {
            if (addr != packed_writable_addr) return false;
            snapshot.width = W;
            snapshot.height = TILED_H;
            snapshot.format = LiveTargetPixelFormat::R11G11B10Float;
            snapshot.pixels = packed_rtt;
            return true;
        });
    const std::vector<uint32_t> packed_writable_spirv = recompile_valu(
        image_copy_2d, std::size(image_copy_2d), 1, 0, &packed_writable_rt);
    CHECK(!packed_writable_spirv.empty(),
          "partial R11G11B10 renderer RTT copy kernel recompiles");
    if (!packed_writable_spirv.empty()) {
        ComputeItem item;
        item.spirv = packed_writable_spirv;
        item.resources = std::make_shared<ShaderResourceTable>(packed_writable_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = 0x590b12;
        CHECK(prosper::frontend::execute_live_compute_items({item}),
              "live backend partially writes an authoritative packed renderer RTT");
        CHECK(packed_rtt_guest == packed_rtt_expected,
              "packed renderer RTT seeds from live pixels and preserves every untouched row");
        CHECK(std::any_of(packed_rtt_guest.begin() + W, packed_rtt_guest.end(),
                          [](uint32_t word) { return word != 0; }),
              "packed renderer RTT never falls back to its stale zeroed guest backing");
    }
    set_live_target_reader({});
    set_live_target_query({});

    // UE4's exposure chain samples tiny tiled Float32 surfaces. Preserve those values natively:
    // normalizing through RGBA8 would clamp negative and HDR channels before the compute shader sees
    // them. Copy a tiled Float32x4 source through the production sampled-image path into the existing
    // raw-channel Float32 storage path and require exact bits at the guest destination.
    std::vector<float> float_src(W * 4), float_dst(W * 4, 0.0f);
    for (uint32_t i = 0; i < W * 4; ++i)
        float_src[i] = (static_cast<int32_t>(i % 17) - 8) * 0.375f;
    const size_t float_tiled_bytes = tiled_surface_bytes(W, 1, dcc_tile, 0, 16);
    std::vector<uint8_t> float_tiled(float_tiled_bytes, 0);
    tile_surface(float_tiled.data(), reinterpret_cast<const uint8_t*>(float_src.data()),
                 W, 1, dcc_tile, 0, 16);
    ShaderResourceTable float_rt = irt;
    for (ShaderResource& resource : float_rt.resources) {
        if (resource.binding != 4 && resource.binding != 5) continue;
        resource.img_dim = 1;
        resource.format = DataFormat::Float32;
        resource.num_components = 4;
        resource.width = W;
        resource.height = resource.depth = 1;
        if (resource.binding == 4) {
            resource.cls = ResourceClass::Texture;
            resource.tile_mode = dcc_tile;
            resource.gpu_addr = reinterpret_cast<uint64_t>(float_tiled.data());
            resource.size = static_cast<uint32_t>(float_tiled.size());
            resource.swizzle[0] = 4;
            resource.swizzle[1] = 5;
            resource.swizzle[2] = 6;
            resource.swizzle[3] = 7;
        } else {
            resource.tile_mode = 0;
            resource.gpu_addr = reinterpret_cast<uint64_t>(float_dst.data());
            resource.size = static_cast<uint32_t>(float_dst.size() * sizeof(float));
        }
    }
    std::vector<uint32_t> float_spirv = recompile_valu(
        image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0, &float_rt);
    CHECK(!float_spirv.empty(), "sampled-image copy kernel recompiles for tiled Float32 input");
    if (!float_spirv.empty()) {
        ComputeItem float_item;
        float_item.spirv = float_spirv;
        float_item.resources = std::make_shared<ShaderResourceTable>(float_rt);
        float_item.launch.threads_x = W;
        float_item.launch.local_x = 64;
        float_item.launch.groups_x = 1;
        float_item.launch.local_y = float_item.launch.local_z = 1;
        float_item.launch.groups_y = float_item.launch.groups_z = 1;
        float_item.code_addr = 0x590f32;
        CHECK(prosper::frontend::execute_live_compute_items({float_item}),
              "live backend executes a dispatch sampling a tiled Float32 image");
        CHECK(float_dst == float_src,
              "tiled Float32 sampled values preserve negative and HDR channels byte-exactly");
    }

    // Sonic's full-screen compute resolve reads two unsigned tiled views that were previously
    // rejected by the live uploader: R32_UINT and RGBA16_UINT.  Keep these as native integer Vulkan
    // images so image_load returns exact component bits; routing them through RGBA8 would truncate
    // the former and reinterpret the latter as normalized color.
    auto sampled_uint_roundtrip = [&](DataFormat format, uint32_t components,
                                      uint32_t bytes_per_texel, uint64_t code_addr) {
        std::vector<uint8_t> linear_src(W * bytes_per_texel);
        std::vector<uint8_t> linear_dst(W * bytes_per_texel, 0xA5);
        for (size_t i = 0; i < linear_src.size(); ++i)
            linear_src[i] = static_cast<uint8_t>(i * 73 + 19);
        const size_t tiled_size = tiled_surface_bytes(W, 1, dcc_tile, 0, bytes_per_texel);
        std::vector<uint8_t> tiled_src(tiled_size, 0);
        tile_surface(tiled_src.data(), linear_src.data(), W, 1, dcc_tile, 0, bytes_per_texel);
        ShaderResourceTable uint_rt = irt;
        for (ShaderResource& resource : uint_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.img_dim = 1;
            resource.format = format;
            resource.num_components = components;
            resource.width = W;
            resource.height = resource.depth = 1;
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.tile_mode = dcc_tile;
                resource.gpu_addr = reinterpret_cast<uint64_t>(tiled_src.data());
                resource.size = static_cast<uint32_t>(tiled_src.size());
                resource.swizzle[0] = 4;
                resource.swizzle[1] = 5;
                resource.swizzle[2] = 6;
                resource.swizzle[3] = 7;
            } else {
                resource.tile_mode = 0;
                resource.gpu_addr = reinterpret_cast<uint64_t>(linear_dst.data());
                resource.size = static_cast<uint32_t>(linear_dst.size());
            }
        }
        const std::vector<uint32_t> spv = recompile_valu(
            image_copy_2d, sizeof(image_copy_2d) / sizeof(image_copy_2d[0]), 1, 0, &uint_rt);
        if (spv.empty()) return false;
        ComputeItem item;
        item.spirv = spv;
        item.resources = std::make_shared<ShaderResourceTable>(uint_rt);
        item.launch.threads_x = W;
        item.launch.local_x = 64;
        item.launch.groups_x = 1;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.code_addr = code_addr;
        return prosper::frontend::execute_live_compute_items({item}) && linear_dst == linear_src;
    };
    CHECK(sampled_uint_roundtrip(DataFormat::Uint32, 1, 4, 0x590320),
          "tiled R32_UINT sampled values reach storage writeback byte-exactly");
    CHECK(sampled_uint_roundtrip(DataFormat::Uint16, 4, 8, 0x590164),
          "tiled RGBA16_UINT sampled values reach storage writeback byte-exactly");

    // GPU captures preserve descriptor addresses but materialize resource bytes in owned host
    // arrays. Warm replay must retain those sampled images just like the live guest-backed path,
    // while validating by exact bytes because host_data mutations never enter the guest journal or
    // page write-watch system. Prove both the unchanged hit and a direct unreported mutation.
    {
        constexpr uint32_t host_texel_bytes = 4;
        std::vector<uint8_t> linear_src(W * host_texel_bytes);
        std::vector<uint8_t> linear_dst(W * host_texel_bytes, 0xA5);
        for (size_t i = 0; i < linear_src.size(); ++i)
            linear_src[i] = static_cast<uint8_t>(i * 41 + 13);
        const size_t tiled_size = tiled_surface_bytes(
            W, 1, dcc_tile, 0, host_texel_bytes);
        std::vector<uint8_t> capture_owned_src(tiled_size, 0);
        tile_surface(capture_owned_src.data(), linear_src.data(), W, 1,
                     dcc_tile, 0, host_texel_bytes);

        ShaderResourceTable host_rt = irt;
        for (ShaderResource& resource : host_rt.resources) {
            if (resource.binding != 4 && resource.binding != 5) continue;
            resource.img_dim = 1;
            resource.format = DataFormat::Uint8;
            resource.num_components = 4;
            resource.width = W;
            resource.height = resource.depth = 1;
            if (resource.binding == 4) {
                resource.cls = ResourceClass::Texture;
                resource.tile_mode = dcc_tile;
                resource.gpu_addr = 0x71c0000000ull;
                resource.size = static_cast<uint32_t>(capture_owned_src.size());
                resource.host_data = capture_owned_src.data();
                resource.host_data_size = capture_owned_src.size();
                resource.swizzle[0] = 4;
                resource.swizzle[1] = 5;
                resource.swizzle[2] = 6;
                resource.swizzle[3] = 7;
            } else {
                resource.tile_mode = 0;
                resource.gpu_addr = reinterpret_cast<uint64_t>(linear_dst.data());
                resource.size = static_cast<uint32_t>(linear_dst.size());
            }
        }
        const std::vector<uint32_t> host_spirv = recompile_valu(
            image_copy_2d, std::size(image_copy_2d), 1, 0, &host_rt);
        CHECK(!host_spirv.empty(),
              "capture-owned sampled-image copy kernel recompiles");
        if (!host_spirv.empty()) {
            ComputeItem host_item;
            host_item.spirv = host_spirv;
            host_item.resources = std::make_shared<ShaderResourceTable>(host_rt);
            host_item.launch.threads_x = W;
            host_item.launch.local_x = 64;
            host_item.launch.groups_x = 1;
            host_item.launch.local_y = host_item.launch.local_z = 1;
            host_item.launch.groups_y = host_item.launch.groups_z = 1;
            host_item.code_addr = 0x590ca9;
            CHECK(prosper::frontend::execute_live_compute_items({host_item}) &&
                      linear_dst == linear_src,
                  "capture-owned sampled image establishes an exact retained source");

            const uint64_t skips_before =
                prosper::frontend::live_compute_sampled_image_upload_skips();
            CHECK(prosper::frontend::execute_live_compute_items({host_item}) &&
                      prosper::frontend::live_compute_sampled_image_upload_skips() >
                          skips_before,
                  "unchanged capture-owned sampled image skips its warm-replay upload");

            linear_src[0] ^= 0xff;
            tile_surface(capture_owned_src.data(), linear_src.data(), W, 1,
                         dcc_tile, 0, host_texel_bytes);
            const uint64_t skips_before_mutation =
                prosper::frontend::live_compute_sampled_image_upload_skips();
            CHECK(prosper::frontend::execute_live_compute_items({host_item}) &&
                      linear_dst == linear_src &&
                      prosper::frontend::live_compute_sampled_image_upload_skips() ==
                          skips_before_mutation,
                  "direct capture-owned mutation refreshes the retained sampled image exactly");
        }
    }

    // --- #1122 seed-skip coverage proof. A write-only storage image whose dispatch fully covers the
    // extent can skip the (expensive) seed -- BUT ONLY after proving the write actually stores every
    // texel. covers_extent is necessary, not sufficient: a shader that stores a SUBSET of a covering
    // grid leaves the rest undefined, and skipping the seed there packs pool garbage to the guest.
    // The backend proves coverage per (shader,binding) on first sight (poison the seed, require zero
    // survivors) and only fast-skips proven-full shaders; a partial write is detected and always
    // seeds, restoring untouched texels. These two cases pin both halves of that contract.

    // (a) FULL coverage: a 1D write-only fill (no image load) over threads_x == width. The seed is
    // unobservable, so the poison-proving run and every seed-skipped run must agree byte-for-byte and
    // differ from the pre-run guest content (the kernel really wrote every texel).
    {
        static const uint32_t fill_1d[] = {
            0x7E080300u,             // v4 = v0 (x coord from the shell input)
            0x7E0A0301u,             // v5 = v1 (y local ID; zero for the one-row launch)
            0x7E040280u,             // v2 = 0 (stored B)
            0x7E060280u,             // v3 = 0 (stored A)
            0xF0200F08u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5) -- NO preceding IMAGE_LOAD
            0xBF810000u,             // s_endpgm
        };
        // A typed RGBA16F storage image exercises the same native-storage retention path as the
        // game's 4K post-process output. Height one keeps the coverage proof compact.
        const size_t fill_guest_bytes = W * 8;
        const size_t fill_mapping_bytes = 4096;
        std::vector<uint8_t> fill_guest_storage;
        uint8_t* fill_guest = nullptr;
#if defined(__linux__)
        void* fill_mapping = mmap(nullptr, fill_mapping_bytes, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(fill_mapping != MAP_FAILED, "map typed storage-image regression range");
        if (fill_mapping == MAP_FAILED) return fails ? fails : 1;
        fill_guest = static_cast<uint8_t*>(fill_mapping);
        prosper::host::guest_write_watch_set_fault_onstack(true);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(fill_guest), fill_mapping_bytes, 0x7c0000,
            0x3 /* SCE CPU_READ|CPU_WRITE */);
#else
        fill_guest_storage.resize(fill_guest_bytes);
        fill_guest = fill_guest_storage.data();
#endif
        std::fill_n(fill_guest, fill_guest_bytes, 0xC3);   // distinctive pre-run content
        const std::vector<uint8_t> fill_original(fill_guest, fill_guest + fill_guest_bytes);
        auto fill_equals = [&](const std::vector<uint8_t>& expected) {
            return expected.size() == fill_guest_bytes &&
                   std::equal(fill_guest, fill_guest + fill_guest_bytes, expected.begin());
        };
        ShaderResourceTable fill_rt;
        ShaderResource fdst{};
        fdst.cls = ResourceClass::StorageImage; fdst.img_dim = 1; fdst.binding = 5; fdst.sgpr_base = 8;
        fdst.format = DataFormat::Float16; fdst.num_components = 4; fdst.width = W; fdst.height = 1;
        fdst.depth = 1; fdst.gpu_addr = (uint64_t)(uintptr_t)fill_guest;
        fdst.size = static_cast<uint32_t>(fill_guest_bytes);
        fill_rt.resources.push_back(fdst);
        ComputeShaderConfig fill_config;
        fill_config.user_sgprs.resize(16);
        fill_config.local_x = W;
        fill_config.local_y = fill_config.local_z = 1;
        fill_config.tidig_comp_cnt = 1;
        fill_config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Float16, 4);
        std::vector<uint32_t> fill_spirv = recompile_compute(
            fill_1d, sizeof(fill_1d) / sizeof(fill_1d[0]), &fill_rt, fill_config);
        CHECK(!fill_spirv.empty(), "write-only 1D fill kernel recompiles");
        if (!fill_spirv.empty()) {
            ComputeItem it; it.spirv = fill_spirv;
            it.resources = std::make_shared<ShaderResourceTable>(fill_rt);
            it.launch.threads_x = W; it.launch.local_x = 64; it.launch.groups_x = 1;
            it.launch.threads_y = it.launch.threads_z = 1;
            it.launch.local_y = it.launch.local_z = 1;
            it.launch.groups_y = it.launch.groups_z = 1;
            it.code_addr = 0x1122f11du;   // fresh code -> first run proves, second run seed-skips
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "seed-skip proving run (poison-seeded) executes a full-coverage 1D fill");
            const std::vector<uint8_t> after_prove(fill_guest, fill_guest + fill_guest_bytes);
            CHECK(!fill_equals(fill_original),
                  "full-coverage fill overwrites the guest content (kernel ran)");
            std::fill_n(fill_guest, fill_guest_bytes, 0x00);  // scrub between runs
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "seed-skip fast run (seed skipped) executes the proven full-coverage fill");
            CHECK(fill_equals(after_prove),
                  "seed-skipped run is byte-identical to the poison-proven run (seed is unobserved)");

#if defined(__linux__)
            auto repeated_write_watch = prosper::host::GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(fill_guest), fill_guest_bytes);
            const uint64_t repeated_snapshots_before =
                prosper::frontend::live_compute_storage_result_snapshot_bytes();
#endif
            uint32_t repeated_write_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
                if (addr == fdst.gpu_addr && size == fdst.size)
                    ++repeated_write_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "retained full-coverage image repeats an identical dispatch");
            set_guest_gpu_write_observer({});
            CHECK(fill_equals(after_prove),
                  "identical retained output leaves the exact guest result intact");
            CHECK(repeated_write_notifications == 1,
                  "identical retained output invalidates renderer aliases without rewriting bytes");
#if defined(__linux__)
            CHECK(static_cast<bool>(repeated_write_watch) && repeated_write_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
                  "identical retained output leaves guest-byte dirty tracking clean");
            CHECK(prosper::frontend::live_compute_storage_result_snapshot_bytes() ==
                      repeated_snapshots_before,
                  "identical retained output does not recopy its guest-byte baseline");
            repeated_write_watch.reset();
#endif

            // The live draw immediately following a compute producer is inside the same ordered
            // guest submit. Its mutation journal is an exact authority even before a cross-submit
            // page watch has accumulated enough stable validations to be promoted.
            std::vector<uint8_t> journal_guest(W * 8, 0x51);
            ShaderResourceTable journal_rt = fill_rt;
            ShaderResource& journal_dst = journal_rt.resources.back();
            journal_dst.gpu_addr = reinterpret_cast<uint64_t>(journal_guest.data());
            ComputeItem journal_item = it;
            journal_item.resources = std::make_shared<ShaderResourceTable>(journal_rt);
            journal_item.dispatch_index = 31;
            journal_item.command_order = 10;
            DrawItem journal_consumer;
            journal_consumer.draw_index = 47;
            journal_consumer.command_order = 20;
            bool journal_imported = false;
            const OrderedSubmitResult journal_result = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, journal_item.dispatch_index,
                  journal_item.command_order},
                 {SubmitOperationKind::Draw, journal_consumer.draw_index,
                  journal_consumer.command_order}},
                {journal_consumer}, {journal_item},
                [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                    ShaderResource sampled = journal_dst;
                    sampled.cls = ResourceClass::Texture;
                    prosper::frontend::LiveComputeImageImport compute_import;
                    journal_imported = prosper::frontend::import_live_compute_storage_image(
                        sampled, journal_dst.size, compute_import) && compute_import.valid();
                    return RenderedFrame{};
                },
                [&](const std::vector<ComputeItem>& items) {
                    return prosper::frontend::execute_live_compute_items(items);
                },
                1, 1);
            CHECK(journal_result.compute_executed && journal_result.render_spans == 1 &&
                      journal_imported,
                  "same-submit journal authorizes the first retained compute-image consumer");

            // Syberia's save-warning pass dispatches eleven shrinking rectangles over one native
            // Float32x1 atlas in a single ordered guest submit. Its producer uses a real one-layer
            // DIM=2D_ARRAY storage image while the next dispatch samples an ordinary DIM=2D view of
            // the same guest allocation. Keep the sampled and writable images distinct
            // (read-old/write-new), but seed the sampled image with a device-local copy of the
            // retained arrayed storage result instead of detiling and uploading the complete guest
            // mirror. This fixture uses the submit journal, rather than a cross-submit page watch,
            // as authority. Pin format equality separately so a future sampled-format policy change
            // cannot silently turn the production check into the guest fallback.
            CHECK(prosper::frontend::compute_native_2d_transfer_format_compatible(
                      DataFormat::Float32, 1) &&
                      !prosper::frontend::compute_native_2d_transfer_format_compatible(
                          DataFormat::Float16, 1),
                  "native 2D transfer candidate requires exact sampled/storage format equality");
            static const uint32_t transfer_fill_2d_array[] = {
                0x7E080300u,             // v4 = x coordinate
                0x7E0A0280u,             // v5 = y = 0
                0x7E0C0280u,             // v6 = array layer 0
                0x7E0002F2u,             // v0 = 1.0f
                0x7E020280u,             // v1 = 0.0f
                0x7E0402F2u,             // v2 = 1.0f
                0x7E060280u,             // v3 = 0.0f
                0xF0200F28u, 0x00020004u,// IMAGE_STORE RGBA at (v4,v5,v6), DIM=2D_ARRAY
                0xBF810000u,
            };
            const size_t transfer_guest_bytes = W * sizeof(float);
            std::vector<uint8_t> transfer_proof_guest(transfer_guest_bytes, 0x37);
            std::vector<uint8_t> transfer_source_guest(transfer_guest_bytes, 0x52);
            std::vector<uint8_t> transfer_copy_guest(transfer_guest_bytes, 0xa9);
            ShaderResource transfer_producer_dst{};
            transfer_producer_dst.cls = ResourceClass::StorageImage;
            // Syberia's producer uses a real one-layer DIM=2D_ARRAY storage instruction, while
            // the consumer reads the byte-identical base slice through an ordinary non-arrayed
            // DIM=2D sampled instruction.
            transfer_producer_dst.img_dim = 5;
            transfer_producer_dst.binding = 5;
            transfer_producer_dst.sgpr_base = 8;
            transfer_producer_dst.format = DataFormat::Float32;
            transfer_producer_dst.num_components = 1;
            transfer_producer_dst.width = W;
            transfer_producer_dst.height = transfer_producer_dst.depth = 1;
            transfer_producer_dst.gpu_addr =
                reinterpret_cast<uint64_t>(transfer_proof_guest.data());
            transfer_producer_dst.size = static_cast<uint32_t>(transfer_guest_bytes);
            ShaderResource transfer_multilayer = transfer_producer_dst;
            transfer_multilayer.depth = 2;
            CHECK(shader_resource_uses_ordinary_2d_image(
                      transfer_producer_dst, true, false, false) &&
                      !shader_resource_uses_ordinary_2d_image(
                          transfer_producer_dst, true, true, false) &&
                      shader_resource_uses_native_2d_storage_image(
                          transfer_producer_dst, true, true, false) &&
                      !shader_resource_uses_ordinary_2d_image(
                          transfer_producer_dst, true, false, true) &&
                      !shader_resource_uses_native_2d_storage_image(
                          transfer_producer_dst, true, true, true) &&
                      !shader_resource_uses_native_2d_storage_image(
                          transfer_multilayer, true, true, false),
                  "single-layer 2D-array native storage requires one reflected array layer");
            ShaderResourceTable transfer_producer_rt;
            transfer_producer_rt.resources.push_back(transfer_producer_dst);
            ComputeShaderConfig transfer_config;
            transfer_config.user_sgprs.resize(16);
            transfer_config.local_x = W;
            transfer_config.local_y = transfer_config.local_z = 1;
            transfer_config.tidig_comp_cnt = 1;
            transfer_config.native_storage_format_support =
                native_storage_format_support_bit(DataFormat::Float32, 1);
            const std::vector<uint32_t> transfer_producer_spirv = recompile_compute(
                transfer_fill_2d_array, std::size(transfer_fill_2d_array),
                &transfer_producer_rt, transfer_config);
            const DescriptorValidationReport transfer_producer_report =
                validate_spirv_descriptor_interface(
                    transfer_producer_spirv, &transfer_producer_rt, 0,
                    SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* transfer_producer_binding =
                find_spirv_descriptor_binding(
                    transfer_producer_report, 0, transfer_producer_dst.binding);
            CHECK(!transfer_producer_spirv.empty() && transfer_producer_report.ok() &&
                      transfer_producer_binding && transfer_producer_binding->storage_float &&
                      transfer_producer_binding->image_dim == 1 &&
                      transfer_producer_binding->image_arrayed &&
                      !transfer_producer_binding->image_multisampled,
                  "single-layer 2D-array producer reflects arrayed exact typed storage");
            if (!transfer_producer_spirv.empty() && transfer_producer_report.ok() &&
                transfer_producer_binding && transfer_producer_binding->storage_float) {
                ComputeItem transfer_proof = it;
                transfer_proof.spirv = transfer_producer_spirv;
                transfer_proof.resources =
                    std::make_shared<ShaderResourceTable>(transfer_producer_rt);
                transfer_proof.code_addr = 0x1122f13du;
                CHECK(prosper::frontend::execute_live_compute_items({transfer_proof}),
                      "native Float32x1 arrayed producer proves complete storage coverage");

                transfer_producer_dst.gpu_addr =
                    reinterpret_cast<uint64_t>(transfer_source_guest.data());
                transfer_producer_rt.resources.back() = transfer_producer_dst;
                ComputeItem transfer_producer = transfer_proof;
                transfer_producer.resources =
                    std::make_shared<ShaderResourceTable>(transfer_producer_rt);
                transfer_producer.dispatch_index = 41;
                transfer_producer.command_order = 10;

                ShaderResourceTable transfer_consumer_rt;
                ShaderResource transfer_sampled = transfer_producer_dst;
                transfer_sampled.cls = ResourceClass::Texture;
                transfer_sampled.binding = 4;
                transfer_sampled.sgpr_base = 0;
                transfer_sampled.swizzle[0] = 4;
                transfer_sampled.swizzle[1] = 5;
                transfer_sampled.swizzle[2] = 6;
                transfer_sampled.swizzle[3] = 7;
                transfer_consumer_rt.resources.push_back(transfer_sampled);
                ShaderResource transfer_copy_dst = transfer_producer_dst;
                transfer_copy_dst.gpu_addr =
                    reinterpret_cast<uint64_t>(transfer_copy_guest.data());
                transfer_consumer_rt.resources.push_back(transfer_copy_dst);
                const std::vector<uint32_t> transfer_consumer_spirv = recompile_compute(
                    image_copy_2d, std::size(image_copy_2d),
                    &transfer_consumer_rt, transfer_config);
                const DescriptorValidationReport transfer_consumer_report =
                    validate_spirv_descriptor_interface(
                        transfer_consumer_spirv, &transfer_consumer_rt, 0,
                        SpirvShaderStage::Compute, false);
                const SpirvDescriptorBinding* transfer_sampled_binding =
                    find_spirv_descriptor_binding(
                        transfer_consumer_report, 0, transfer_sampled.binding);
                const SpirvDescriptorBinding* transfer_copy_binding =
                    find_spirv_descriptor_binding(
                        transfer_consumer_report, 0, transfer_copy_dst.binding);
                CHECK(!transfer_consumer_spirv.empty() && transfer_consumer_report.ok() &&
                          transfer_sampled_binding && transfer_copy_binding &&
                          transfer_sampled_binding->image_dim == 1 &&
                          !transfer_sampled_binding->image_arrayed &&
                          !transfer_sampled_binding->image_multisampled &&
                          transfer_copy_binding->storage_float &&
                          transfer_copy_binding->image_dim == 1 &&
                          !transfer_copy_binding->image_arrayed &&
                          !transfer_copy_binding->image_multisampled,
                      "single-layer 2D-array consumer reflects ordinary sampled/storage views");
                if (!transfer_consumer_spirv.empty() && transfer_consumer_report.ok() &&
                    transfer_sampled_binding && transfer_copy_binding) {
                    ComputeItem transfer_consumer = transfer_proof;
                    transfer_consumer.spirv = transfer_consumer_spirv;
                    transfer_consumer.resources =
                        std::make_shared<ShaderResourceTable>(transfer_consumer_rt);
                    transfer_consumer.code_addr = 0x1122f14du;
                    transfer_consumer.dispatch_index = 42;
                    transfer_consumer.command_order = 20;
                    const uint64_t transfer_seeds_before =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    const OrderedSubmitResult transfer_result = execute_ordered_items(
                        {{SubmitOperationKind::Dispatch, transfer_producer.dispatch_index,
                          transfer_producer.command_order},
                         {SubmitOperationKind::Dispatch, transfer_consumer.dispatch_index,
                          transfer_consumer.command_order}},
                        {}, {transfer_producer, transfer_consumer},
                        [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                            return RenderedFrame{};
                        },
                        [&](const std::vector<ComputeItem>& items) {
                            return prosper::frontend::execute_live_compute_items(items);
                        },
                        1, 1);
                    const uint64_t transfer_seeds_after =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    CHECK(transfer_result.compute_executed &&
                              transfer_copy_guest == transfer_source_guest,
                          "arrayed producer to ordinary sampled consumer preserves every Float32 texel");
                    CHECK(!native_2d_compute_transfer_available ||
                              transfer_seeds_after > transfer_seeds_before,
                          "single-layer arrayed native storage producer seeds ordinary 2D "
                          "sampled consumer on-GPU");
                    CHECK(native_2d_compute_transfer_available ||
                              transfer_seeds_after == transfer_seeds_before,
                          "disabled native 2D transfer or authority validation keeps the exact "
                          "guest fallback");
                }
            }

#if defined(__linux__)
            // Use a dedicated mapping so page protection cannot alias unrelated malloc metadata.
            // Outside an ordered submit there is no submit-local journal, so a successful import
            // here specifically proves the cross-submit write-watch authority path.
            constexpr size_t import_mapping_bytes = 4096;
            auto* import_guest = static_cast<uint8_t*>(mmap(
                nullptr, import_mapping_bytes, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            CHECK(import_guest != MAP_FAILED,
                  "allocate dedicated typed-storage import mapping");
            if (import_guest != MAP_FAILED) {
                std::memset(import_guest, 0x37, import_mapping_bytes);
                prosper::host::guest_write_watch_set_fault_onstack(true);
                prosper::host::guest_write_watch_notify_direct_mapping_added(
                    reinterpret_cast<uint64_t>(import_guest), import_mapping_bytes,
                    0x7e0000, 0x3 /* SCE CPU_READ|CPU_WRITE */);
                ShaderResourceTable import_rt = fill_rt;
                ShaderResource& import_dst = import_rt.resources.back();
                import_dst.gpu_addr = reinterpret_cast<uint64_t>(import_guest);
                ComputeItem import_item = it;
                import_item.resources = std::make_shared<ShaderResourceTable>(import_rt);
                CHECK(prosper::frontend::execute_live_compute_items({import_item}) &&
                          prosper::frontend::execute_live_compute_items({import_item}) &&
                          prosper::frontend::execute_live_compute_items({import_item}) &&
                          prosper::frontend::execute_live_compute_items({import_item}),
                      "repeated typed-storage output promotes its exact source watch");
                ShaderResource sampled_fill = import_dst;
                sampled_fill.cls = ResourceClass::Texture;
                prosper::frontend::LiveComputeImageImport compute_import;
                CHECK(prosper::frontend::import_live_compute_storage_image(
                          sampled_fill, import_dst.size, compute_import) &&
                          compute_import.valid() && compute_import.width == W &&
                          compute_import.height == 1 && compute_import.depth == 1 &&
                          compute_import.native_format && compute_import.layout,
                      "validated typed-storage result can be leased by an exact sampled descriptor");
                prosper::frontend::LiveComputeImageImport rejected_import;
                ShaderResource mismatched_fill = sampled_fill;
                ++mismatched_fill.width;
                const bool rejected_extent =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                ++mismatched_fill.tile_mode;
                const bool rejected_layout =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                mismatched_fill.host_data = import_guest;
                mismatched_fill.host_data_size = import_dst.size;
                const bool rejected_replay =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                mismatched_fill.srgb = true;
                const bool rejected_srgb =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                mismatched_fill = sampled_fill;
                mismatched_fill.declared_mip_levels = 2;
                const bool rejected_mips =
                    !prosper::frontend::import_live_compute_storage_image(
                        mismatched_fill, import_dst.size, rejected_import);
                CHECK(rejected_extent && rejected_layout && rejected_replay && rejected_srgb &&
                          rejected_mips,
                      "compute-image import requires full identity and keeps replay/mip/sRGB fallback");
                compute_import = {};
                // This binary does not install the emulator's SIGSEGV write-fault handler. Mark the
                // same page dirty through the DMA/GPU side of the watch API; the importer must not
                // trust the retained image after any architectural writer invalidates its bytes.
                prosper::host::guest_write_watch_notify_gpu_write(
                    reinterpret_cast<uint64_t>(import_guest), import_dst.size);
                CHECK(!prosper::frontend::import_live_compute_storage_image(
                          sampled_fill, import_dst.size, compute_import),
                      "guest mutation revokes a cross-submit compute-image import");
                prosper::host::guest_write_watch_notify_direct_mapping_removed(
                    reinterpret_cast<uint64_t>(import_guest), import_mapping_bytes);
                prosper::host::guest_write_watch_set_fault_onstack(false);
                munmap(import_guest, import_mapping_bytes);
            }
#endif

            // Prove a second full writer of the same format/extent on a different target, then use
            // it on this still-valid guest mirror. The persistent image key intentionally does not
            // contain the shader: different post-processes may reuse one target. Its GPU comparison
            // must report the changed word exactly and force the ordinary writeback path.
            static const uint32_t changed_fill_1d[] = {
                0x7E080300u,             // v4 = v0 (x)
                0x7E0A0301u,             // v5 = v1 (y)
                0x7E0402F2u,             // v2 = 1.0f (changed stored B)
                0x7E060280u,             // v3 = 0
                0xF0200F08u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5)
                0xBF810000u,
            };
            std::vector<uint8_t> changed_proof_guest(W * 8, 0x71);
            ShaderResourceTable changed_proof_rt = fill_rt;
            changed_proof_rt.resources.back().gpu_addr =
                reinterpret_cast<uint64_t>(changed_proof_guest.data());
            std::vector<uint32_t> changed_spirv = recompile_compute(
                changed_fill_1d,
                sizeof(changed_fill_1d) / sizeof(changed_fill_1d[0]),
                &changed_proof_rt, fill_config);
            CHECK(!changed_spirv.empty(), "second full-coverage fill kernel recompiles");
            if (!changed_spirv.empty()) {
                ComputeItem changed = it;
                changed.spirv = changed_spirv;
                changed.code_addr = 0x1122f12du;
                changed.resources = std::make_shared<ShaderResourceTable>(changed_proof_rt);
                CHECK(prosper::frontend::execute_live_compute_items({changed}),
                      "changed fill proves full coverage on an independent target");
                const std::vector<uint8_t> changed_expected = changed_proof_guest;
                changed.resources = std::make_shared<ShaderResourceTable>(fill_rt);

                prosper::frontend::live_compute_fail_next_storage_readback_for_test();
                CHECK(!prosper::frontend::execute_live_compute_items({changed}),
                      "injected post-submit storage readback failure is reported");
                CHECK(fill_equals(after_prove),
                      "failed storage readback does not publish newer image bytes to the guest");
                uint32_t changed_write_notifications = 0;
                set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
                    if (addr == fdst.gpu_addr && size == fdst.size)
                        ++changed_write_notifications;
                });
#if defined(__linux__)
                // The nested compute-import fixture temporarily disables the synthetic fault
                // handler when it tears down. Restore the outer mapping's emulator contract before
                // exercising this target again.
                prosper::host::guest_write_watch_set_fault_onstack(true);
                const uint64_t storage_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
#endif
                CHECK(prosper::frontend::execute_live_compute_items({changed}),
                      "storage image retries after a failed post-submit readback");
                set_guest_gpu_write_observer({});
                CHECK(fill_equals(changed_expected) && !fill_equals(after_prove),
                      "retry invalidates the stale baseline and publishes the changed output");
                CHECK(changed_write_notifications == 1,
                      "recovered storage output forces guest writeback and invalidation");
#if defined(__linux__)
                const uint64_t storage_snapshots_after =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                CHECK(storage_snapshots_after >=
                          storage_snapshots_before + fill_guest_bytes,
                      "post-failure repair replaces its invalidated exact result baseline");

                auto recovered_write_watch = prosper::host::GuestWriteWatch::create(
                    reinterpret_cast<uint64_t>(fill_guest), fill_guest_bytes);
                const uint64_t recovered_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
#endif
                uint32_t recovered_repeat_notifications = 0;
                set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
                    if (addr == fdst.gpu_addr && size == fdst.size)
                        ++recovered_repeat_notifications;
                });
                CHECK(prosper::frontend::execute_live_compute_items({changed}),
                      "recovered storage image repeats an identical dispatch");
                set_guest_gpu_write_observer({});
                CHECK(fill_equals(changed_expected),
                      "post-recovery identical dispatch preserves the repaired guest result");
                CHECK(recovered_repeat_notifications == 1,
                      "post-recovery identical result still invalidates renderer aliases");
#if defined(__linux__)
                CHECK(static_cast<bool>(recovered_write_watch) &&
                          recovered_write_watch.query() ==
                              prosper::host::GuestWriteWatchQuery::Unchanged,
                      "post-recovery identical result does not rewrite guest bytes");
                CHECK(prosper::frontend::live_compute_storage_result_snapshot_bytes() ==
                          recovered_snapshots_before,
                      "post-recovery identical result does not recopy its repaired baseline");
                recovered_write_watch.reset();
#endif
            }

            // Guest memory no longer matches the retained source. Exact source validation must
            // force a writeback rather than trusting only the GPU-side output baseline.
#if defined(__linux__)
            // The production emulator installs the SIGSEGV write-fault handler; this compact test
            // does not. Model its architectural CPU write through the ordinary pre-write hook so
            // the armed page is made writable and marked Dirty before touching it.
            prosper::host::guest_write_watch_notify_host_write(
                reinterpret_cast<uint64_t>(fill_guest), 1);
#endif
            fill_guest[0] ^= 0xff;
            uint32_t repaired_write_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
                if (addr == fdst.gpu_addr && size == fdst.size)
                    ++repaired_write_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({it}),
                  "externally changed guest mirror reruns the retained full-coverage dispatch");
            set_guest_gpu_write_observer({});
            CHECK(fill_equals(after_prove),
                  "retained output repairs an externally changed guest mirror");
            CHECK(repaired_write_notifications == 1,
                  "externally changed guest mirror forces writeback and invalidation");

#if defined(__linux__)
            // Alternate two proven-full writers and verify that a changing target carries no exact
            // source snapshot: its old seed is unobservable, and the next GPU result comparison is
            // independently collision-free.
            if (!changed_spirv.empty()) {
                ComputeItem alternating_changed = it;
                alternating_changed.spirv = changed_spirv;
                alternating_changed.code_addr = 0x1122f12du;
                alternating_changed.resources = std::make_shared<ShaderResourceTable>(fill_rt);
                const uint64_t dynamic_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                CHECK(prosper::frontend::execute_live_compute_items({alternating_changed}) &&
                          prosper::frontend::execute_live_compute_items({it}),
                      "alternating full-coverage storage results execute");
                const uint64_t dynamic_snapshots_after =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                if (adaptive_storage_result_validation_enabled) {
                    CHECK(dynamic_snapshots_after == dynamic_snapshots_before,
                          "dynamic full-overwrite result avoids redundant snapshots");
                } else {
                    CHECK(dynamic_snapshots_after >=
                              dynamic_snapshots_before + 2 * fill_guest_bytes,
                          "disabled adaptive policy preserves exact dynamic snapshots");
                }
            }
#endif

            // The first dispatch to a new target is the cache-churn shape seen in full-resolution
            // post-processing: the shader/extent already has a Full proof, but this address has no
            // retained image yet. A dedicated invocation lowers the production crossover to zero so
            // this reduced target exercises the real deferring branch without a 64 MiB allocation;
            // the ordinary and disable-switch invocations retain the previous immediate baseline.
            std::vector<uint8_t> cold_guest(W * 8, 0x6b);
            ShaderResourceTable cold_rt = fill_rt;
            cold_rt.resources.back().gpu_addr =
                reinterpret_cast<uint64_t>(cold_guest.data());
            ComputeItem cold_item = it;
            cold_item.resources = std::make_shared<ShaderResourceTable>(cold_rt);
            const uint64_t cold_source_snapshots_before =
                prosper::frontend::live_compute_storage_result_snapshot_bytes();
            const uint64_t cold_result_snapshots_before =
                prosper::frontend::live_compute_image_result_snapshot_bytes();
            prosper::frontend::live_compute_zero_next_cold_storage_snapshot_minimum_for_test();
            CHECK(prosper::frontend::execute_live_compute_items({cold_item}),
                  "proven-full writer executes on a cold retained target");
            const uint64_t cold_source_snapshots_after =
                prosper::frontend::live_compute_storage_result_snapshot_bytes();
            const uint64_t cold_result_snapshots_after =
                prosper::frontend::live_compute_image_result_snapshot_bytes();
            if (cold_storage_snapshot_deferral_enabled) {
                CHECK(cold_source_snapshots_after == cold_source_snapshots_before,
                      "deferring policy admits a cold proven-full target without a source copy");

                // A later standalone backend invocation has no same-submit journal authority. The
                // first repeat therefore cannot trust the deferred source: it must take ordinary
                // writeback and establish the exact baseline used by later source validation.
                const std::vector<uint8_t> cold_expected = cold_guest;
                const uint64_t repeat_snapshots_before =
                    prosper::frontend::live_compute_storage_result_snapshot_bytes();
                CHECK(prosper::frontend::execute_live_compute_items({cold_item}),
                      "first invalidated repeat repairs a deferred cold target");
                CHECK(cold_guest == cold_expected &&
                          prosper::frontend::live_compute_storage_result_snapshot_bytes() >=
                              repeat_snapshots_before + fill_guest_bytes,
                      "first invalidated repeat establishes exact source authority");

                cold_guest[0] ^= 0xff;
                CHECK(cold_guest != cold_expected,
                      "external mutation changes the deferred target fixture");
                CHECK(prosper::frontend::execute_live_compute_items({cold_item}) &&
                          cold_guest == cold_expected,
                      "external mutation forces deferred target writeback and exact repair");
            } else {
                CHECK(cold_source_snapshots_after >=
                          cold_source_snapshots_before + fill_guest_bytes,
                      "default or disabled policy retains the cold source immediately");
            }
            CHECK(cold_result_snapshots_after == cold_result_snapshots_before,
                  "cold exact target retains its GPU baseline without a transient host copy");
            CHECK(prosper::frontend::cold_storage_result_snapshot_can_defer(
                      false, true, 64u << 20, 16u << 20) &&
                      !prosper::frontend::cold_storage_result_snapshot_can_defer(
                          true, true, 64u << 20, 16u << 20) &&
                      !prosper::frontend::cold_storage_result_snapshot_can_defer(
                          false, false, 64u << 20, 16u << 20) &&
                      !prosper::frontend::cold_storage_result_snapshot_can_defer(
                          false, true, 8u << 20, 16u << 20),
                  "cold snapshot deferral is limited to large proven-full guest targets");

            // Model the exact stale-fallback sequence from the review: transient setup stores host
            // result A, result B reaches guest memory but GPU-baseline ownership fails, then A runs
            // again. The failed B ownership attempt must invalidate host A; otherwise the final A
            // can be misclassified as repeated and leave B in architectural guest memory.
            if (!changed_spirv.empty()) {
                std::vector<uint8_t> fallback_guest(W * 8, 0x42);
                ShaderResourceTable fallback_rt = fill_rt;
                fallback_rt.resources.back().gpu_addr =
                    reinterpret_cast<uint64_t>(fallback_guest.data());
                ComputeItem fallback_a = it;
                fallback_a.resources = std::make_shared<ShaderResourceTable>(fallback_rt);
                ComputeItem fallback_b = fallback_a;
                fallback_b.spirv = changed_spirv;
                fallback_b.code_addr = 0x1122f12du;
                fallback_b.dispatch_index = 51;
                fallback_b.command_order = 10;
                fallback_a.dispatch_index = 52;
                fallback_a.command_order = 20;

                const uint64_t fallback_snapshots_before =
                    prosper::frontend::live_compute_image_result_snapshot_bytes();
                prosper::frontend::live_compute_force_next_image_result_host_fallback_for_test();
                CHECK(prosper::frontend::execute_live_compute_items({fallback_a}),
                      "transient baseline setup fallback publishes result A");
                const uint64_t fallback_snapshots_after_a =
                    prosper::frontend::live_compute_image_result_snapshot_bytes();
                CHECK(std::equal(fallback_guest.begin(), fallback_guest.end(),
                                 after_prove.begin()) &&
                          fallback_snapshots_after_a >=
                              fallback_snapshots_before + fill_guest_bytes,
                      "transient setup fallback retains exact host result A");

                prosper::frontend::live_compute_fail_next_image_result_buffer_retain_for_test();
                size_t fallback_dispatches = 0;
                bool fallback_dispatches_ok = true;
                bool failed_retain_published_b = false;
                const OrderedSubmitResult fallback_submit = execute_ordered_items(
                    {{SubmitOperationKind::Dispatch, fallback_b.dispatch_index,
                      fallback_b.command_order},
                     {SubmitOperationKind::Dispatch, fallback_a.dispatch_index,
                      fallback_a.command_order}},
                    {}, {fallback_b, fallback_a},
                    [](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                        return RenderedFrame{};
                    },
                    [&](const std::vector<ComputeItem>& items) {
                        const bool ok = prosper::frontend::execute_live_compute_items(items);
                        fallback_dispatches_ok &= ok;
                        if (fallback_dispatches++ == 0)
                            failed_retain_published_b = std::equal(
                                fallback_guest.begin(), fallback_guest.end(),
                                changed_proof_guest.begin()) &&
                                prosper::frontend::live_compute_image_result_snapshot_bytes() ==
                                    fallback_snapshots_after_a;
                        return ok;
                    },
                    1, 1);
                CHECK(fallback_submit.compute_executed && fallback_dispatches_ok &&
                          fallback_dispatches == 2 && failed_retain_published_b,
                      "result B survives injected GPU-baseline ownership failure");
                CHECK(std::equal(fallback_guest.begin(), fallback_guest.end(),
                                 after_prove.begin()),
                      "stale host result A cannot suppress required A-after-B writeback");
            }
        }
#if defined(__linux__)
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(fill_guest), fill_mapping_bytes);
        prosper::host::guest_write_watch_set_fault_onstack(false);
        munmap(fill_guest, fill_mapping_bytes);
#endif
    }

    // A proven-full write-only raw storage image may retain its RGBA32_UINT interchange result even
    // though that representation is not canonical guest data: the old raw texels are unobservable.
    // This is the Plucky Squire lighting-grid shape (3D tiled FP16), reduced to one 4x4x4 workgroup.
    // The third identical dispatch must compare the raw transfer on-GPU and omit pack/retile and the
    // guest invalidation, while an external guest-memory change must still force exact writeback.
    {
        static const uint32_t fill_3d[] = {
            0x7E080300u,             // v4 = v0 (x)
            0x7E0A0301u,             // v5 = v1 (y)
            0x7E0C0302u,             // v6 = v2 (z)
            0xF0200F10u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5,v6), dim:3D, no load
            0xBF810000u,
        };
        constexpr uint32_t RW = 4, RH = 4, RD = 4;
        constexpr uint32_t raw_mode = static_cast<uint32_t>(TileMode::Sw64KbS);
        const size_t raw_guest_bytes = tiled_volume_bytes(RW, RH, RD, raw_mode, 8);
        std::vector<uint8_t> raw_guest_storage;
        uint8_t* raw_guest = nullptr;
#if defined(__linux__)
        void* raw_mapping = mmap(nullptr, raw_guest_bytes, PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(raw_mapping != MAP_FAILED, "map raw storage-image regression range");
        if (raw_mapping == MAP_FAILED) return fails ? fails : 1;
        raw_guest = static_cast<uint8_t*>(raw_mapping);
        prosper::host::guest_write_watch_set_fault_onstack(true);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes, 0x7d0000,
            0x3 /* SCE CPU_READ|CPU_WRITE */);
#else
        raw_guest_storage.resize(raw_guest_bytes);
        raw_guest = raw_guest_storage.data();
#endif
        std::fill_n(raw_guest, raw_guest_bytes, 0x9d);
        ShaderResourceTable raw_rt;
        ShaderResource raw_dst{};
        raw_dst.cls = ResourceClass::StorageImage;
        raw_dst.img_dim = 2;
        raw_dst.binding = 5;
        raw_dst.sgpr_base = 8;
        raw_dst.format = DataFormat::Float16;
        raw_dst.num_components = 4;
        raw_dst.width = RW;
        raw_dst.height = RH;
        raw_dst.depth = RD;
        raw_dst.tile_mode = raw_mode;
        raw_dst.gpu_addr = reinterpret_cast<uint64_t>(raw_guest);
        raw_dst.size = static_cast<uint32_t>(raw_guest_bytes);
        raw_rt.resources.push_back(raw_dst);
        ComputeShaderConfig raw_config;
        raw_config.user_sgprs.resize(16);
        raw_config.local_x = RW;
        raw_config.local_y = RH;
        raw_config.local_z = RD;
        raw_config.tidig_comp_cnt = 2;
        // Keep the raw fallback used by true 3D images even on devices supporting typed FP16.
        raw_config.native_storage_format_support = 0;
        const std::vector<uint32_t> raw_spirv = recompile_compute(
            fill_3d, std::size(fill_3d), &raw_rt, raw_config);
        CHECK(!raw_spirv.empty(), "write-only raw 3D fill kernel recompiles");
        if (!raw_spirv.empty()) {
            ComputeItem raw_item;
            raw_item.spirv = raw_spirv;
            raw_item.resources = std::make_shared<ShaderResourceTable>(raw_rt);
            raw_item.launch.threads_x = RW;
            raw_item.launch.threads_y = RH;
            raw_item.launch.threads_z = RD;
            raw_item.launch.local_x = RW;
            raw_item.launch.local_y = RH;
            raw_item.launch.local_z = RD;
            raw_item.launch.groups_x = raw_item.launch.groups_y = raw_item.launch.groups_z = 1;
            raw_item.code_addr = 0x1122f13du;
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "raw 3D fill proves complete write coverage");
            const std::vector<uint8_t> raw_expected(raw_guest,
                                                    raw_guest + raw_guest_bytes);
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "raw 3D fill establishes a retained exact result baseline");
#if defined(__linux__)
            auto raw_repeat_watch = prosper::host::GuestWriteWatch::create(
                reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes);
#endif
            uint32_t raw_repeat_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
                if (addr == raw_dst.gpu_addr && size == raw_guest_bytes)
                    ++raw_repeat_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "retained raw 3D fill repeats an identical dispatch");
            set_guest_gpu_write_observer({});
            CHECK(std::equal(raw_guest, raw_guest + raw_guest_bytes, raw_expected.begin()) &&
                      raw_repeat_notifications == 1,
                  "GPU-identical raw 3D output skips pack/retile but invalidates renderer aliases");
#if defined(__linux__)
            CHECK(static_cast<bool>(raw_repeat_watch) && raw_repeat_watch.query() ==
                      prosper::host::GuestWriteWatchQuery::Unchanged,
                  "GPU-identical raw 3D output leaves guest-byte dirty tracking clean");
            raw_repeat_watch.reset();
#endif

            raw_guest[0] ^= 0xff;
            uint32_t raw_repair_notifications = 0;
            set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size) {
                if (addr == raw_dst.gpu_addr && size == raw_guest_bytes)
                    ++raw_repair_notifications;
            });
            CHECK(prosper::frontend::execute_live_compute_items({raw_item}),
                  "externally changed raw 3D mirror reruns the retained dispatch");
            set_guest_gpu_write_observer({});
            CHECK(std::equal(raw_guest, raw_guest + raw_guest_bytes, raw_expected.begin()) &&
                      raw_repair_notifications == 1,
                  "external raw 3D mirror change forces exact writeback and invalidation");
        }

        // The same 3D FP16 resource can stay at its exact eight-byte native width when the device
        // advertises dimension-specific support. This is Astro Bot's 240x135x64 ping-pong shape;
        // reflection must select float storage rather than the sixteen-byte raw-uvec4 interchange.
        ComputeShaderConfig native_3d_config = raw_config;
        native_3d_config.native_storage_format_support =
            native_storage_3d_format_support_bit(DataFormat::Float16, 4);
        const std::vector<uint32_t> native_3d_spirv = recompile_compute(
            fill_3d, std::size(fill_3d), &raw_rt, native_3d_config);
        CHECK(!native_3d_spirv.empty(), "native typed 3D fill kernel recompiles");
        if (!native_3d_spirv.empty()) {
            const DescriptorValidationReport native_3d_report =
                validate_spirv_descriptor_interface(
                    native_3d_spirv, &raw_rt, 0, SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* native_3d_binding =
                find_spirv_descriptor_binding(native_3d_report, 0, raw_dst.binding);
            CHECK(native_3d_report.ok() && native_3d_binding &&
                      native_3d_binding->kind == SpirvDescriptorKind::StorageImage &&
                      native_3d_binding->storage_float && native_3d_binding->image_dim == 2,
                  "dimension-capable FP16 volume reflects exact float 3D storage");

#if defined(__linux__)
            const bool native_3d_runtime_supported =
                prosper::frontend::live_compute_native_storage_3d_supported(
                    DataFormat::Float16, 4, RW, RH, RD);
#else
            constexpr bool native_3d_runtime_supported = false;
#endif
            if (native_3d_runtime_supported &&
                adaptive_storage_result_validation_enabled) {
#if defined(__linux__)
                prosper::host::guest_write_watch_notify_host_write(
                    reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes);
#endif
                std::fill_n(raw_guest, raw_guest_bytes, 0x3a);
                ComputeItem native_3d_item;
                native_3d_item.spirv = native_3d_spirv;
                native_3d_item.resources = std::make_shared<ShaderResourceTable>(raw_rt);
                native_3d_item.launch.threads_x = RW;
                native_3d_item.launch.threads_y = RH;
                native_3d_item.launch.threads_z = RD;
                native_3d_item.launch.local_x = RW;
                native_3d_item.launch.local_y = RH;
                native_3d_item.launch.local_z = RD;
                native_3d_item.launch.groups_x = native_3d_item.launch.groups_y =
                    native_3d_item.launch.groups_z = 1;
                native_3d_item.code_addr = 0x1122f14du;
                CHECK(prosper::frontend::execute_live_compute_items({native_3d_item}),
                      "native typed 3D fill executes at exact guest width");
                const std::vector<uint8_t> native_3d_expected(
                    raw_guest, raw_guest + raw_guest_bytes);
                CHECK(std::any_of(native_3d_expected.begin(), native_3d_expected.end(),
                                  [](uint8_t byte) { return byte != 0x3a; }),
                      "native typed 3D fill overwrites its guest volume");
                CHECK(prosper::frontend::execute_live_compute_items({native_3d_item}) &&
                          std::equal(raw_guest, raw_guest + raw_guest_bytes,
                                     native_3d_expected.begin()),
                      "retained native 3D result is byte-exact across repeated dispatches");
#if defined(__linux__)
                // The first successful dispatch proves full coverage and the second retains the
                // image. Force one exact repair so the retained result establishes the cross-submit
                // page watch that authorizes the following device-local sampled copy.
                raw_guest[0] ^= 0x5a;
                CHECK(prosper::frontend::execute_live_compute_items({native_3d_item}) &&
                          std::equal(raw_guest, raw_guest + raw_guest_bytes,
                                     native_3d_expected.begin()),
                      "native 3D result repairs an external mirror write before transfer");
#endif

                ShaderResource sampled_native_3d = raw_dst;
                sampled_native_3d.cls = ResourceClass::Texture;
                prosper::frontend::LiveComputeImageImport graphics_import;
                CHECK(prosper::frontend::import_live_compute_storage_image(
                          sampled_native_3d, raw_guest_bytes, graphics_import) &&
                          graphics_import.valid() && graphics_import.width == RW &&
                          graphics_import.height == RH && graphics_import.depth == RD,
                      "validated native 3D storage result can be leased by graphics");
                ShaderResource mismatched_native_3d = sampled_native_3d;
                --mismatched_native_3d.depth;
                prosper::frontend::LiveComputeImageImport rejected_graphics_import;
                CHECK(!prosper::frontend::import_live_compute_storage_image(
                          mismatched_native_3d, raw_guest_bytes, rejected_graphics_import),
                      "graphics volume import requires exact 3D descriptor identity");
                graphics_import = {};

                static const uint32_t sampled_copy_3d[] = {
                    0x7E080300u,             // v4 = v0 (x)
                    0x7E0A0301u,             // v5 = v1 (y)
                    0x7E0C0302u,             // v6 = v2 (z)
                    0xF0000F10u, 0x00000004u,// IMAGE_LOAD v0..v3 at (v4,v5,v6), dim:3D
                    0xBF8C3F70u,             // s_waitcnt vmcnt(0)
                    0xF0200F10u, 0x00020004u,// IMAGE_STORE to a distinct 3D destination
                    0xBF810000u,
                };
                std::vector<uint8_t> native_copy_guest(raw_guest_bytes, 0x71);
                ShaderResourceTable native_copy_rt;
                ShaderResource native_copy_src = raw_dst;
                native_copy_src.cls = ResourceClass::Texture;
                native_copy_src.binding = 4;
                native_copy_src.sgpr_base = 0;
                native_copy_rt.resources.push_back(native_copy_src);
                ShaderResource native_copy_dst = raw_dst;
                native_copy_dst.gpu_addr =
                    reinterpret_cast<uint64_t>(native_copy_guest.data());
                native_copy_rt.resources.push_back(native_copy_dst);
                const std::vector<uint32_t> native_copy_spirv = recompile_compute(
                    sampled_copy_3d, std::size(sampled_copy_3d),
                    &native_copy_rt, native_3d_config);
                CHECK(!native_copy_spirv.empty(),
                      "sampled-to-storage native 3D copy kernel recompiles");
                if (!native_copy_spirv.empty()) {
                    ComputeItem native_copy_item = native_3d_item;
                    native_copy_item.spirv = native_copy_spirv;
                    native_copy_item.resources =
                        std::make_shared<ShaderResourceTable>(native_copy_rt);
                    native_copy_item.code_addr = 0x1122f15du;
                    const uint64_t transfer_seeds_before =
                        prosper::frontend::live_compute_storage_transfer_seeds();
                    CHECK(prosper::frontend::execute_live_compute_items({native_copy_item}),
                          "sampled 3D consumer executes from retained native storage output");
                    CHECK(prosper::frontend::live_compute_storage_transfer_seeds() >
                              transfer_seeds_before,
                          "sampled 3D consumer seeds on-GPU without a guest upload");
                    std::vector<uint8_t> native_source_linear(RW * RH * RD * 8u);
                    std::vector<uint8_t> native_copy_linear(native_source_linear.size());
                    CHECK(detile_volume(native_source_linear.data(), raw_guest,
                                        raw_guest_bytes, RW, RH, RD, raw_mode, 8) &&
                              detile_volume(native_copy_linear.data(),
                                            native_copy_guest.data(), raw_guest_bytes,
                                            RW, RH, RD, raw_mode, 8) &&
                              native_copy_linear == native_source_linear,
                          "device-local native 3D seed preserves every logical FP16 voxel");
                }
            }
        }
#if defined(__linux__)
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(raw_guest), raw_guest_bytes);
        prosper::host::guest_write_watch_set_fault_onstack(false);
        munmap(raw_guest, raw_guest_bytes);
#endif
    }

    // Syberia's colour-grade LUT is a tiled R11G11B10F volume written by one compute dispatch and
    // sampled by the immediately following dispatch in the same guest submit (#1790). Existing
    // coverage separates that contract into 2D packed-R11 conversion, 3D FP16 storage, and a 2D
    // same-submit graphics import. Keep the exact missing cross-product together here: a real 3D
    // SAMPLE_LZ consumer, the guest's SW_64KB_R_X layout, exact storage with and without a native
    // typed-format capability bit, and a quantitative f11/f10 oracle over every logical voxel.
    {
        static const uint32_t r11_volume_producer[] = {
            0xF0000F10u, 0x00000400u, // IMAGE_LOAD v[4:7], v[0:2], s[0:7], dim:3D
            0xBF8C3F70u,              // s_waitcnt vmcnt(0)
            0xF0200F10u, 0x00020400u, // IMAGE_STORE v[4:7], v[0:2], s[8:15], dim:3D
            0xBF810000u,
        };
        static const uint32_t r11_volume_consumer[] = {
            0x7E080300u,              // v4 = v0 (integer x retained in v0)
            0x7E0A0301u,              // v5 = v1 (integer y retained in v1)
            0x7E0C0302u,              // v6 = v2 (integer z retained in v2)
            0x7E080D04u,              // v_cvt_f32_u32 v4, v4
            0x7E0A0D05u,              // v_cvt_f32_u32 v5, v5
            0x7E0C0D06u,              // v_cvt_f32_u32 v6, v6
            0x060808F0u,              // v4 = v4 + 0.5
            0x060A0AF0u,              // v5 = v5 + 0.5
            0x060C0CF0u,              // v6 = v6 + 0.5
            0x100808FFu, 0x3E800000u, // v4 *= 0.25 (literal)
            0x100A0AFFu, 0x3E800000u, // v5 *= 0.25 (literal)
            0x100C0CFFu, 0x3E800000u, // v6 *= 0.25 (literal)
            0xF09C0F10u, 0x00400C04u, // IMAGE_SAMPLE_LZ v[12:15], v[4:6], s[0:7], s[8:11], 3D
            0xBF8C3F70u,              // s_waitcnt vmcnt(0)
            0xF0200F10u, 0x00040C00u, // IMAGE_STORE v[12:15], v[0:2], s[16:23], dim:3D
            0xBF810000u,
        };

        constexpr uint32_t LUT_W = 4, LUT_H = 4, LUT_D = 4;
        constexpr size_t LUT_TEXELS = LUT_W * LUT_H * LUT_D;
        constexpr uint32_t LUT_TILE = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const size_t lut_guest_bytes =
            tiled_volume_bytes(LUT_W, LUT_H, LUT_D, LUT_TILE, sizeof(uint32_t));
        CHECK(lut_guest_bytes != 0 && tile_mode_supports_volume(LUT_TILE),
              "Syberia reduced LUT uses the implemented tiled 3D SW_64KB_R_X layout");

        auto float_bits = [](float value) {
            uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return bits;
        };
        std::vector<uint32_t> source_bits(LUT_TEXELS * 4u);
        std::vector<uint32_t> expected_bits(LUT_TEXELS * 4u);
        std::vector<uint32_t> expected_packed(LUT_TEXELS);
        bool has_sub_one = false, has_hdr = false, has_quantized_value = false;
        for (uint32_t z = 0; z < LUT_D; ++z) {
            for (uint32_t y = 0; y < LUT_H; ++y) {
                for (uint32_t x = 0; x < LUT_W; ++x) {
                    const size_t texel = (static_cast<size_t>(z) * LUT_H + y) * LUT_W + x;
                    const float values[4] = {
                        0.003f + x * 0.537f + y * 0.019f + z * 0.007f,
                        0.125f + x * 0.013f + y * 0.911f + z * 0.203f,
                        0.250f + x * 0.017f + y * 0.071f + z * 2.003f,
                        1.0f,
                    };
                    const float quantized[4] = {
                        f11_to_float(float_to_f11(values[0])),
                        f11_to_float(float_to_f11(values[1])),
                        f10_to_float(float_to_f10(values[2])),
                        1.0f,
                    };
                    for (uint32_t channel = 0; channel < 4; ++channel) {
                        source_bits[texel * 4 + channel] = float_bits(values[channel]);
                        expected_bits[texel * 4 + channel] = float_bits(quantized[channel]);
                        if (channel < 3) {
                            has_sub_one |= values[channel] > 0.0f && values[channel] < 1.0f;
                            has_hdr |= values[channel] > 1.0f;
                            has_quantized_value |= float_bits(values[channel]) !=
                                                   float_bits(quantized[channel]);
                        }
                    }
                    expected_packed[texel] =
                        static_cast<uint32_t>(float_to_f11(values[0])) |
                        (static_cast<uint32_t>(float_to_f11(values[1])) << 11) |
                        (static_cast<uint32_t>(float_to_f10(values[2])) << 22);
                }
            }
        }
        CHECK(has_sub_one && has_hdr && has_quantized_value,
              "Syberia reduced LUT oracle spans sub-one, HDR, and real packing boundaries");

        auto spirv_has_opcode = [](const std::vector<uint32_t>& spirv, uint16_t wanted) {
            for (size_t offset = 5; offset < spirv.size();) {
                const uint32_t word_count = spirv[offset] >> 16;
                if (!word_count || offset + word_count > spirv.size()) return false;
                if (static_cast<uint16_t>(spirv[offset]) == wanted) return true;
                offset += word_count;
            }
            return false;
        };

        auto run_r11_volume_handoff = [&](uint32_t native_support,
                                           bool native_capability_present) {
            const char* arm = native_capability_present ? "capability-present" : "portable";
            std::vector<uint8_t> lut_guest(lut_guest_bytes, 0xA5);
            const uint32_t sentinel_word = static_cast<uint32_t>(float_to_f11(0.0625f)) |
                (static_cast<uint32_t>(float_to_f11(0.125f)) << 11) |
                (static_cast<uint32_t>(float_to_f10(0.25f)) << 22);
            std::vector<uint32_t> sentinel_linear(LUT_TEXELS, sentinel_word);
            const bool sentinel_tiled = tile_volume(
                lut_guest.data(), lut_guest.size(),
                reinterpret_cast<const uint8_t*>(sentinel_linear.data()),
                LUT_W, LUT_H, LUT_D, LUT_TILE, sizeof(uint32_t));
            CHECK(sentinel_tiled,
                  native_capability_present
                      ? "native-capability R11 LUT sentinel tiles across XYZ"
                      : "portable R11 LUT sentinel tiles across XYZ");

            ShaderResource producer_src{};
            producer_src.cls = ResourceClass::StorageImage;
            producer_src.img_dim = 2;
            producer_src.binding = 4;
            producer_src.sgpr_base = 0;
            producer_src.format = DataFormat::Float32;
            producer_src.num_components = 4;
            producer_src.width = LUT_W;
            producer_src.height = LUT_H;
            producer_src.depth = LUT_D;
            producer_src.gpu_addr = reinterpret_cast<uint64_t>(source_bits.data());
            producer_src.size = static_cast<uint32_t>(source_bits.size() * sizeof(uint32_t));

            ShaderResource producer_dst{};
            producer_dst.cls = ResourceClass::StorageImage;
            producer_dst.img_dim = 2;
            producer_dst.binding = 5;
            producer_dst.sgpr_base = 8;
            producer_dst.format = DataFormat::Float10_11_11;
            producer_dst.num_components = 3;
            producer_dst.width = LUT_W;
            producer_dst.height = LUT_H;
            producer_dst.depth = LUT_D;
            producer_dst.tile_mode = LUT_TILE;
            producer_dst.gpu_addr = reinterpret_cast<uint64_t>(lut_guest.data());
            producer_dst.size = static_cast<uint32_t>(lut_guest.size());

            ShaderResourceTable producer_rt;
            producer_rt.resources = {producer_src, producer_dst};
            ComputeShaderConfig producer_config;
            producer_config.user_sgprs.resize(16);
            producer_config.local_x = LUT_W;
            producer_config.local_y = LUT_H;
            producer_config.local_z = LUT_D;
            producer_config.threads_x = LUT_W;
            producer_config.threads_y = LUT_H;
            producer_config.threads_z = LUT_D;
            producer_config.tidig_comp_cnt = 2;
            producer_config.native_storage_format_support = native_support;
            producer_config.packed_r11_storage = true;
            const std::vector<uint32_t> producer_spirv = recompile_compute(
                r11_volume_producer, std::size(r11_volume_producer),
                &producer_rt, producer_config);
            const DescriptorValidationReport producer_report =
                validate_spirv_descriptor_interface(
                    producer_spirv, &producer_rt, 0, SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* reflected_lut =
                find_spirv_descriptor_binding(producer_report, 0, producer_dst.binding);
            const bool producer_shape_ok = !producer_spirv.empty() && producer_report.ok() &&
                reflected_lut && reflected_lut->kind == SpirvDescriptorKind::StorageImage &&
                reflected_lut->image_dim == 2 && reflected_lut->writable;
            CHECK(producer_shape_ok,
                  native_capability_present
                      ? "native-capability R11 producer reflects writable 3D storage"
                      : "portable R11 producer reflects writable 3D storage");
            const bool exact_storage = producer_shape_ok && !reflected_lut->storage_float &&
                reflected_lut->storage_image_format == kSpirvImageFormatR32ui;
            CHECK(exact_storage,
                  native_capability_present
                      ? "native R11 capability preserves exact packed R32ui 3D storage"
                      : "disabled native capability selects exact packed R32ui 3D storage");
            if (!exact_storage) return;

            std::vector<uint32_t> output_bits(LUT_TEXELS * 4u, 0xCDCDCDCDu);
            ShaderResource sampled_lut = producer_dst;
            sampled_lut.cls = ResourceClass::Texture;
            sampled_lut.binding = 4;
            sampled_lut.sgpr_base = 0;
            sampled_lut.sampler_sgpr_base = 8;
            sampled_lut.mag_filter = 1;
            sampled_lut.min_filter = 1;
            sampled_lut.mip_filter = 0;

            ShaderResource consumer_dst{};
            consumer_dst.cls = ResourceClass::StorageImage;
            consumer_dst.img_dim = 2;
            consumer_dst.binding = 5;
            consumer_dst.sgpr_base = 16;
            consumer_dst.format = DataFormat::Float32;
            consumer_dst.num_components = 4;
            consumer_dst.width = LUT_W;
            consumer_dst.height = LUT_H;
            consumer_dst.depth = LUT_D;
            consumer_dst.gpu_addr = reinterpret_cast<uint64_t>(output_bits.data());
            consumer_dst.size = static_cast<uint32_t>(output_bits.size() * sizeof(uint32_t));

            ShaderResourceTable consumer_rt;
            consumer_rt.resources = {sampled_lut, consumer_dst};
            ComputeShaderConfig consumer_config;
            consumer_config.user_sgprs.resize(24);
            consumer_config.local_x = LUT_W;
            consumer_config.local_y = LUT_H;
            consumer_config.local_z = LUT_D;
            consumer_config.threads_x = LUT_W;
            consumer_config.threads_y = LUT_H;
            consumer_config.threads_z = LUT_D;
            consumer_config.tidig_comp_cnt = 2;
            consumer_config.native_storage_format_support = native_support;
            const std::vector<uint32_t> consumer_spirv = recompile_compute(
                r11_volume_consumer, std::size(r11_volume_consumer),
                &consumer_rt, consumer_config);
            const DescriptorValidationReport consumer_report =
                validate_spirv_descriptor_interface(
                    consumer_spirv, &consumer_rt, 0, SpirvShaderStage::Compute, false);
            const SpirvDescriptorBinding* reflected_sample =
                find_spirv_descriptor_binding(consumer_report, 0, sampled_lut.binding);
            constexpr uint16_t OpImageSampleExplicitLod = 88;
            const bool consumer_shape_ok = !consumer_spirv.empty() && consumer_report.ok() &&
                reflected_sample &&
                reflected_sample->kind == SpirvDescriptorKind::CombinedImageSampler &&
                reflected_sample->image_dim == 2 && reflected_sample->sampled_float &&
                reflected_sample->normalized_sampling &&
                spirv_has_opcode(consumer_spirv, OpImageSampleExplicitLod);
            CHECK(consumer_shape_ok,
                  native_capability_present
                      ? "native-capability arm retains explicit-LOD normalized 3D R11 sampling"
                      : "portable arm retains explicit-LOD normalized 3D R11 sampling");
            if (!consumer_shape_ok) return;

            ComputeItem producer_item;
            producer_item.spirv = producer_spirv;
            producer_item.resources = std::make_shared<ShaderResourceTable>(producer_rt);
            producer_item.dispatch_index = native_capability_present ? 218 : 118;
            producer_item.command_order = 10;
            producer_item.code_addr = native_capability_present ? 0x1790a118u : 0x1790b118u;
            producer_item.launch.threads_x = LUT_W;
            producer_item.launch.threads_y = LUT_H;
            producer_item.launch.threads_z = LUT_D;
            producer_item.launch.local_x = LUT_W;
            producer_item.launch.local_y = LUT_H;
            producer_item.launch.local_z = LUT_D;
            producer_item.launch.groups_x = producer_item.launch.groups_y =
                producer_item.launch.groups_z = 1;

            ComputeItem consumer_item = producer_item;
            consumer_item.spirv = consumer_spirv;
            consumer_item.resources = std::make_shared<ShaderResourceTable>(consumer_rt);
            consumer_item.dispatch_index = native_capability_present ? 219 : 119;
            consumer_item.command_order = 20;
            consumer_item.code_addr = native_capability_present ? 0x1790a119u : 0x1790b119u;

            // Prime the sampled-image cache with the old guest LUT. The ordered run must still see
            // the producer's new result; otherwise a first-use upload would accidentally make the
            // test green without exercising same-submit invalidation/authority.
            std::vector<uint32_t> warm_bits(LUT_TEXELS * 4u, 0xABABABABu);
            ShaderResourceTable warm_rt = consumer_rt;
            warm_rt.resources.back().gpu_addr = reinterpret_cast<uint64_t>(warm_bits.data());
            ComputeItem warm_item = consumer_item;
            warm_item.resources = std::make_shared<ShaderResourceTable>(warm_rt);
            warm_item.dispatch_index += 1000;
            warm_item.command_order = 1;
            const bool warm_executed =
                prosper::frontend::execute_live_compute_items({warm_item});
            const uint32_t warm_expected[4] = {
                float_bits(f11_to_float(static_cast<uint16_t>(sentinel_word))),
                float_bits(f11_to_float(static_cast<uint16_t>(sentinel_word >> 11))),
                float_bits(f10_to_float(static_cast<uint16_t>(sentinel_word >> 22))),
                float_bits(1.0f),
            };
            size_t warm_mismatches = 0;
            for (size_t texel = 0; texel < LUT_TEXELS; ++texel)
                for (uint32_t channel = 0; channel < 4; ++channel)
                    warm_mismatches += warm_bits[texel * 4 + channel] !=
                                       warm_expected[channel];
            CHECK(warm_executed && warm_mismatches == 0,
                  native_capability_present
                      ? "native-capability sampled cache primes from the old tiled R11 sentinel"
                      : "portable sampled cache primes from the old tiled R11 sentinel");

            std::array<uint32_t, 2> callback_dispatches = {UINT32_MAX, UINT32_MAX};
            size_t callback_count = 0;
            bool callbacks_ok = true;
            const OrderedSubmitResult submit = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, producer_item.dispatch_index,
                  producer_item.command_order},
                 {SubmitOperationKind::Dispatch, consumer_item.dispatch_index,
                  consumer_item.command_order}},
                {}, {producer_item, consumer_item},
                [](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                    return RenderedFrame{};
                },
                [&](const std::vector<ComputeItem>& items) {
                    const bool singleton = items.size() == 1;
                    if (callback_count < callback_dispatches.size() && singleton)
                        callback_dispatches[callback_count] = items[0].dispatch_index;
                    ++callback_count;
                    const bool ok = singleton &&
                        prosper::frontend::execute_live_compute_items(items);
                    callbacks_ok &= ok;
                    return ok;
                },
                1, 1);
            CHECK(submit.compute_executed && callbacks_ok && callback_count == 2,
                  native_capability_present
                      ? "native-capability ordered submit executes both dispatches exactly once"
                      : "portable ordered submit executes both dispatches exactly once");
            CHECK(callback_dispatches[0] == producer_item.dispatch_index &&
                      callback_dispatches[1] == consumer_item.dispatch_index,
                  native_capability_present
                      ? "native-capability submit preserves producer-before-consumer order"
                      : "portable submit preserves producer-before-consumer order");

            std::vector<uint32_t> produced_packed(LUT_TEXELS, 0xDEADBEEFu);
            const bool produced_detiled = detile_volume(
                reinterpret_cast<uint8_t*>(produced_packed.data()), lut_guest.data(),
                lut_guest.size(), LUT_W, LUT_H, LUT_D, LUT_TILE, sizeof(uint32_t));
            CHECK(produced_detiled,
                  native_capability_present
                      ? "native-capability producer detiles for direct packed-word census"
                      : "portable producer detiles for direct packed-word census");
            const uint32_t field_shift[3] = {0, 11, 22};
            const uint32_t field_mask[3] = {0x7ffu, 0x7ffu, 0x3ffu};
            std::array<size_t, 3> packed_channel_mismatches{};
            size_t packed_mismatches = 0, packed_lower = 0, packed_higher = 0;
            for (size_t texel = 0; produced_detiled && texel < LUT_TEXELS; ++texel) {
                if (produced_packed[texel] == expected_packed[texel]) continue;
                ++packed_mismatches;
                for (uint32_t channel = 0; channel < 3; ++channel) {
                    const uint32_t actual =
                        (produced_packed[texel] >> field_shift[channel]) & field_mask[channel];
                    const uint32_t expected =
                        (expected_packed[texel] >> field_shift[channel]) & field_mask[channel];
                    if (actual == expected) continue;
                    ++packed_channel_mismatches[channel];
                    packed_lower += actual < expected;
                    packed_higher += actual > expected;
                }
                if (packed_mismatches <= 8) {
                    const size_t xy = LUT_W * LUT_H;
                    const uint32_t z = static_cast<uint32_t>(texel / xy);
                    const uint32_t y = static_cast<uint32_t>((texel % xy) / LUT_W);
                    const uint32_t x = static_cast<uint32_t>(texel % LUT_W);
                    std::printf("  %s R11 producer packed diff xyz=(%u,%u,%u) expected=%08x "
                                "actual=%08x xor=%08x\n", arm, x, y, z,
                                expected_packed[texel], produced_packed[texel],
                                expected_packed[texel] ^ produced_packed[texel]);
                }
            }

            size_t sentinel_survivors = 0, mismatches = 0;
            std::array<size_t, 4> output_channel_mismatches{};
            size_t first_texel = SIZE_MAX;
            uint32_t first_channel = UINT32_MAX;
            for (size_t texel = 0; texel < LUT_TEXELS; ++texel) {
                for (uint32_t channel = 0; channel < 4; ++channel) {
                    const size_t index = texel * 4 + channel;
                    sentinel_survivors += output_bits[index] == 0xCDCDCDCDu;
                    if (output_bits[index] == expected_bits[index]) continue;
                    if (first_texel == SIZE_MAX) {
                        first_texel = texel;
                        first_channel = channel;
                    }
                    ++output_channel_mismatches[channel];
                    ++mismatches;
                }
            }
            if (packed_mismatches || mismatches) {
                std::printf("  %s R11 producer census: packed-texels=%zu/%zu channels=(%zu,%zu,%zu) "
                            "lower=%zu higher=%zu\n", arm, packed_mismatches, LUT_TEXELS,
                            packed_channel_mismatches[0], packed_channel_mismatches[1],
                            packed_channel_mismatches[2], packed_lower, packed_higher);
            }
            if (mismatches) {
                std::printf("  %s tiled R11 3D census: mismatches=%zu/%zu first-texel=%zu "
                            "channel=%u channels=(%zu,%zu,%zu,%zu) expected=%08x actual=%08x "
                            "sentinel-survivors=%zu\n",
                            arm, mismatches, expected_bits.size(), first_texel, first_channel,
                            output_channel_mismatches[0], output_channel_mismatches[1],
                            output_channel_mismatches[2], output_channel_mismatches[3],
                            expected_bits[first_texel * 4 + first_channel],
                            output_bits[first_texel * 4 + first_channel], sentinel_survivors);
            }
            CHECK(sentinel_survivors == 0,
                  native_capability_present
                      ? "native-capability 3D consumer overwrites every output channel"
                      : "portable 3D consumer overwrites every output channel");
            CHECK(packed_mismatches == 0,
                  native_capability_present
                      ? "native-capability producer matches every packed R11 voxel"
                      : "portable producer matches every packed R11 voxel");
            CHECK(mismatches == 0,
                  native_capability_present
                      ? "native-capability exact R11 handoff matches every CPU f11/f10 voxel"
                      : "portable exact R11 handoff matches every CPU f11/f10 voxel");
        };

        run_r11_volume_handoff(0, false);
        run_r11_volume_handoff(
            native_storage_3d_format_support_bit(DataFormat::Float10_11_11, 3),
            true);
    }

    // (b) PARTIAL store under a FULL grid (reviewer B1): a write-only 2D kernel that always stores to
    // row 0 (v5 hard-zero), dispatched over a W x 2 grid so covers_extent is TRUE while row 1 is never
    // written. The proof must detect the survivor, NOT fast-skip, and preserve row 1's prior content.
    // Before the fix this row was undefined pool memory packed to the guest -- silent corruption.
    {
        static const uint32_t store_row0_2d[] = {
            0x7E080300u,             // v4 = v0 (x)
            0x7E0A0280u,             // v5 = 0  (y) -- ALWAYS row 0, regardless of the y invocation
            0xF0200F08u, 0x00020004u,// IMAGE_STORE v0..v3 at (v4,v5) to binding 5 -- write-only
            0xBF810000u,             // s_endpgm
        };
        const uint32_t H2 = 2;
        std::vector<uint32_t> p_index(W);
        for (uint32_t i = 0; i < W; ++i) p_index[i] = i;   // x == gid: row 0 fully written, row 1 never
        std::vector<uint32_t> pdummy(4, 0);
        std::vector<uint8_t> part_guest(W * H2 * 4);
        for (size_t i = 0; i < part_guest.size(); ++i) part_guest[i] = (uint8_t)(i * 29 + 13);
        const std::vector<uint8_t> part_original = part_guest;
        ShaderResourceTable part_rt;
        auto padd_buf = [&](uint32_t b, void* d, uint32_t s) {
            ShaderResource r{}; r.cls = ResourceClass::ConstantBuffer; r.binding = b;
            r.gpu_addr = (uint64_t)(uintptr_t)d; r.size = s; part_rt.resources.push_back(r); };
        padd_buf(0, p_index.data(), W * sizeof(uint32_t));
        padd_buf(1, pdummy.data(), 16); padd_buf(2, pdummy.data(), 16); padd_buf(3, pdummy.data(), 16);
        ShaderResource pdst{};
        pdst.cls = ResourceClass::StorageImage; pdst.img_dim = 1; pdst.binding = 5; pdst.sgpr_base = 8;
        pdst.format = DataFormat::Unorm8; pdst.num_components = 4; pdst.width = W; pdst.height = H2;
        pdst.depth = 1; pdst.gpu_addr = (uint64_t)(uintptr_t)part_guest.data(); pdst.size = W * H2 * 4;
        part_rt.resources.push_back(pdst);
        std::vector<uint32_t> part_spirv = recompile_valu(
            store_row0_2d, sizeof(store_row0_2d) / sizeof(store_row0_2d[0]), 1, 0, &part_rt);
        CHECK(!part_spirv.empty(), "write-only 2D row-0 store kernel recompiles");
        if (!part_spirv.empty()) {
            ComputeItem it; it.spirv = part_spirv;
            it.resources = std::make_shared<ShaderResourceTable>(part_rt);
            it.launch.threads_x = W; it.launch.local_x = 64; it.launch.groups_x = 1;
            it.launch.threads_y = H2; it.launch.local_y = 1; it.launch.groups_y = H2;
            it.launch.threads_z = 1; it.launch.local_z = 1; it.launch.groups_z = 1;
            it.code_addr = 0x1122f22du;
            const size_t row_bytes = (size_t)W * 4;
            // Two runs: the first proves (poison) and detects partial coverage; the second uses the
            // cached "Partial" verdict and seeds normally. Row 1 must survive on BOTH.
            for (int run = 0; run < 2; ++run) {
                CHECK(prosper::frontend::execute_live_compute_items({it}),
                      run == 0 ? "partial-store proving run executes"
                               : "partial-store cached (always-seed) run executes");
                const bool row1_preserved = std::memcmp(part_guest.data() + row_bytes,
                                                        part_original.data() + row_bytes, row_bytes) == 0;
                CHECK(row1_preserved,
                      run == 0 ? "B1: full-grid partial store preserves untouched row on the proving run"
                               : "B1: full-grid partial store preserves untouched row once proven partial");
                const bool row0_written = std::memcmp(part_guest.data(),
                                                      part_original.data(), row_bytes) != 0;
                CHECK(row0_written, "partial store did write the covered row (kernel ran)");
            }
        }
    }

    // Keep this last: the production contract deliberately latches a lost VkDevice for the rest of
    // the process. Inject one queue-submit loss into a two-item batch, then call the backend again.
    // Both checks measure the lever directly: disabling either the in-batch break or the persistent
    // short-circuit causes another queue-submit attempt and makes a named assertion fail.
    {
        FILE* diagnostic_file = std::tmpfile();
        int saved_stderr = -1;
        bool capturing = false;
        if (diagnostic_file) {
            std::fflush(stderr);
            saved_stderr = duplicate_descriptor(file_descriptor(stderr));
            capturing = saved_stderr >= 0 &&
                        replace_descriptor(file_descriptor(diagnostic_file),
                                           file_descriptor(stderr)) >= 0;
        }
        CHECK(capturing, "device-loss diagnostic capture initialized");

        const uint64_t attempts_before =
            prosper::frontend::live_compute_queue_submit_attempts();
        prosper::frontend::live_compute_force_next_queue_submit_device_lost_for_test();
        const bool batch_result =
            prosper::frontend::execute_live_compute_items({item, item});
        const uint64_t attempts_after_loss =
            prosper::frontend::live_compute_queue_submit_attempts();
        const bool later_result = prosper::frontend::execute_live_compute_items({item});
        const uint64_t attempts_after_later_call =
            prosper::frontend::live_compute_queue_submit_attempts();

        std::string diagnostic;
        if (capturing) {
            std::fflush(stderr);
            if (replace_descriptor(saved_stderr, file_descriptor(stderr)) < 0)
                capturing = false;
            close_descriptor(saved_stderr);
            saved_stderr = -1;
            std::rewind(diagnostic_file);
            char buffer[512];
            while (const size_t bytes = std::fread(buffer, 1, sizeof(buffer), diagnostic_file))
                diagnostic.append(buffer, bytes);
        } else if (saved_stderr >= 0) {
            close_descriptor(saved_stderr);
        }
        if (diagnostic_file) std::fclose(diagnostic_file);

        CHECK(!batch_result && attempts_after_loss == attempts_before + 1,
              "device loss aborts the current batch before another queue submit");
        CHECK(!later_result && attempts_after_later_call == attempts_after_loss,
              "latched device loss rejects later callbacks before queue submission");
        CHECK(capturing &&
                  diagnostic.find("result=VK_ERROR_DEVICE_LOST(-4)") != std::string::npos &&
                  diagnostic.find("program=0x401aec200 submit=11 dispatch=7 order=70") !=
                      std::string::npos &&
                  diagnostic.find("result=VK_ERROR_DEVICE_LOST(-4)",
                                  diagnostic.find("result=VK_ERROR_DEVICE_LOST(-4)") + 1) ==
                      std::string::npos,
              "device loss is unconditional and identifies the first observed guest dispatch");
    }

    if (fails) {
        std::printf("== FAIL: %d == (%d assertions executed)\n", fails, checks);
        return 1;
    }
    std::printf("== PASS == (%d assertions executed)\n", checks);
    return 0;
}
