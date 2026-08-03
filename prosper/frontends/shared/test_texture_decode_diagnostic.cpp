#include "texture_decode_diagnostic.hpp"
#include "texture_decode_cache_policy.hpp"
#include "gpu/gpu_capture.hpp"

#include <cstdio>

using prosper::frontend::should_report_texture_decode_miss;
using prosper::frontend::block_compressed_cube_source_size;
using prosper::frontend::texture_decode_cache_candidate;
using prosper::frontend::texture_decode_miss_reason;
using prosper::frontend::texture_decode_miss_is_expensive_block;
using prosper::frontend::TextureDecodeMissReason;

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)
#define CHECK_NAMED(name, cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s: %s\n", name, #cond); ++failures; } } while (0)

int main() {
    CHECK(texture_decode_cache_candidate(false, false, false, 1u, true, true, false));
    CHECK(texture_decode_cache_candidate(false, false, false, 2u, true, true, false));
    CHECK(texture_decode_cache_candidate(false, false, false, 5u, true, true, false));
    CHECK_NAMED("syberia_bc6_cube_candidate",
                texture_decode_cache_candidate(false, false, false, 3u, true, true, true));
    CHECK(!texture_decode_cache_candidate(false, false, false, 3u, true, true, false));
    CHECK(!texture_decode_cache_candidate(false, true, false, 1u, true, true, false));
    CHECK(!texture_decode_cache_candidate(true, false, false, 1u, true, true, false));

    constexpr uint64_t syberia_cube_address = 0x212bd41000ull;
    constexpr uint64_t syberia_cube_footprint = 33570816ull;
    prosper::gpu::ShaderResource syberia_cube{};
    syberia_cube.cls = prosper::gpu::ResourceClass::Texture;
    syberia_cube.gpu_addr = syberia_cube_address;
    syberia_cube.size = 25165824u;
    syberia_cube.format = prosper::gpu::DataFormat::Bc6;
    syberia_cube.num_components = 3u;
    syberia_cube.img_dim = 3u;
    syberia_cube.width = syberia_cube.height = 2048u;
    syberia_cube.depth = 6u;
    syberia_cube.tile_mode = 5u;
    syberia_cube.layer_stride_bytes = 5595136u;
    syberia_cube.layer_mip_offset_bytes = 1400832u;
    const uint64_t computed_syberia_footprint =
        prosper::gpu::gpu_capture_resource_footprint(syberia_cube);
    CHECK_NAMED("syberia_bc6_cube_descriptor_footprint",
                computed_syberia_footprint == syberia_cube_footprint);
    CHECK_NAMED("syberia_bc6_cube_source_range",
                block_compressed_cube_source_size(
                    true, syberia_cube_address, computed_syberia_footprint) ==
                    syberia_cube_footprint);
    CHECK(block_compressed_cube_source_size(
              false, syberia_cube_address, syberia_cube_footprint) == 0);
    CHECK(block_compressed_cube_source_size(true, syberia_cube_address, 0) == 0);
    CHECK_NAMED("syberia_bc6_cube_source_overflow",
                block_compressed_cube_source_size(
                    true, UINT64_MAX - syberia_cube_footprint + 1u,
                    syberia_cube_footprint) == 0);

    constexpr size_t retained_bytes = 16u << 20;
    CHECK(texture_decode_miss_reason(
              false, true, true, retained_bytes, false, 1u << 30,
              /*matching_cache_entry=*/true, /*cache_eligible=*/true) ==
              TextureDecodeMissReason::ContentInvalidated);
    CHECK(texture_decode_miss_reason(
              false, true, true, retained_bytes, false, 1u << 30,
              /*matching_cache_entry=*/false, /*cache_eligible=*/true) ==
              TextureDecodeMissReason::ColdOrEvicted);
    CHECK(texture_decode_miss_reason(
              false, true, true, retained_bytes, true, 1u << 30,
              /*matching_cache_entry=*/false, /*cache_eligible=*/false) ==
              TextureDecodeMissReason::CacheDisabled);
    CHECK(texture_decode_miss_reason(
              false, true, false, retained_bytes, false, 1u << 30,
              /*matching_cache_entry=*/false, /*cache_eligible=*/false) ==
              TextureDecodeMissReason::UnsupportedCompression);

    // The exact defect shape: a once-per-second BC6H miss can be the first event for its address
    // after thousands of cheaper misses. It must be printed immediately, not after 3,000 repeats.
    const bool ineligible_large_bc6 = texture_decode_miss_is_expensive_block(
        /*expensive_codec=*/true, /*persistent_source_size=*/0,
        /*fallback_source_size=*/32u << 20);
    CHECK(ineligible_large_bc6);
    CHECK(!texture_decode_miss_is_expensive_block(
              /*expensive_codec=*/false, 0, 24u << 20));
    CHECK(should_report_texture_decode_miss(
              /*global_ordinal=*/4001, /*address_ordinal=*/1,
              ineligible_large_bc6));
    CHECK(should_report_texture_decode_miss(4009, 8, true));
    CHECK(should_report_texture_decode_miss(4017, 16, true));
    CHECK(!should_report_texture_decode_miss(4010, 9, true));
    CHECK(!should_report_texture_decode_miss(4001, 1, false));
    CHECK(should_report_texture_decode_miss(6000, 125, false));

    if (!failures) std::printf("texture_decode_diagnostic: OK\n");
    return failures ? 1 : 0;
}
