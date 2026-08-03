#include "../tools/gpu_replay/post_compute_resource.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

static uint32_t pack_r11(float r, float g, float b) {
    return static_cast<uint32_t>(gpu::float_to_f11(r)) |
           (static_cast<uint32_t>(gpu::float_to_f11(g)) << 11) |
           (static_cast<uint32_t>(gpu::float_to_f10(b)) << 22);
}

static gpu::ShaderResource r11_storage(uint8_t* bytes, size_t byte_count,
                                       uint32_t width, uint32_t height,
                                       uint32_t depth, uint32_t tile_mode) {
    gpu::ShaderResource resource;
    resource.binding = 14;
    resource.cls = gpu::ResourceClass::StorageImage;
    resource.format = gpu::DataFormat::Float10_11_11;
    resource.num_components = 3;
    resource.width = width;
    resource.height = height;
    resource.depth = depth;
    resource.img_dim = depth > 1 ? 2u : 1u;
    resource.tile_mode = tile_mode;
    resource.host_data = bytes;
    resource.host_data_size = byte_count;
    return resource;
}

int main() {
    std::printf("== test_gpu_replay_post_compute_resource ==\n");

    const std::array<uint32_t, 2> numeric_words = {
        pack_r11(0.0f, 1.0f, 2.0f),
        pack_r11(std::numeric_limits<float>::infinity(),
                 std::numeric_limits<float>::quiet_NaN(), 0.5f),
    };
    std::vector<uint8_t> numeric_bytes(sizeof(numeric_words));
    std::memcpy(numeric_bytes.data(), numeric_words.data(), numeric_bytes.size());
    gpu::ShaderResourceTable numeric_table;
    numeric_table.resources.push_back(
        r11_storage(numeric_bytes.data(), numeric_bytes.size(), 2, 1, 1, 0));
    tools::PostComputeResourceSnapshot numeric;
    std::string error;
    CHECK(tools::snapshot_post_compute_resource(&numeric_table, 14, numeric, error) &&
              numeric.raw == numeric_bytes && numeric.linear == numeric_bytes,
          "linear storage snapshot preserves exact descriptor-visible words");
    CHECK(numeric.has_r11_stats && numeric.r11.texels == 2 &&
              numeric.r11.channels == 6 && numeric.r11.finite == 4 &&
              numeric.r11.zero == 1 && numeric.r11.above_one == 1 &&
              numeric.r11.positive_infinity == 1 && numeric.r11.nan == 1 &&
              std::fabs(numeric.r11.finite_min - 0.0) < 1e-9 &&
              std::fabs(numeric.r11.finite_max - 2.0) < 1e-9 &&
              std::fabs(numeric.r11.finite_mean - 0.875) < 1e-9,
          "R11 analysis separates finite, zero, HDR, infinity, and NaN channels");

    constexpr uint32_t width = 4, height = 4;
    std::vector<uint8_t> surface_linear(width * height * sizeof(uint32_t));
    for (uint32_t texel = 0; texel < width * height; ++texel) {
        const uint32_t word = pack_r11(static_cast<float>(texel) / 8.0f,
                                       static_cast<float>(texel + 1) / 4.0f,
                                       static_cast<float>(texel + 2) / 2.0f);
        std::memcpy(surface_linear.data() + texel * sizeof(word), &word, sizeof(word));
    }
    std::vector<uint8_t> surface_tiled(gpu::tiled_surface_bytes(
        width, height, static_cast<uint32_t>(gpu::TileMode::Sw64KbRX), 0, 4));
    gpu::tile_surface(surface_tiled.data(), surface_linear.data(), width, height,
                      static_cast<uint32_t>(gpu::TileMode::Sw64KbRX), 0, 4);
    gpu::ShaderResourceTable surface_table;
    surface_table.resources.push_back(r11_storage(
        surface_tiled.data(), surface_tiled.size(), width, height, 1,
        static_cast<uint32_t>(gpu::TileMode::Sw64KbRX)));
    tools::PostComputeResourceSnapshot surface_before;
    error.clear();
    CHECK(tools::snapshot_post_compute_resource(
              &surface_table, 14, surface_before, error) &&
              surface_before.linear == surface_linear,
          "tiled 2D post-compute storage detiles to the exact linear texel order");

    uint32_t brighter = pack_r11(8.0f, 4.0f, 2.0f);
    std::memcpy(surface_linear.data(), &brighter, sizeof(brighter));
    gpu::tile_surface(surface_tiled.data(), surface_linear.data(), width, height,
                      static_cast<uint32_t>(gpu::TileMode::Sw64KbRX), 0, 4);
    tools::PostComputeResourceSnapshot surface_after;
    error.clear();
    CHECK(tools::snapshot_post_compute_resource(
              &surface_table, 14, surface_after, error) &&
              surface_after.linear == surface_linear &&
              surface_after.raw_hash != surface_before.raw_hash &&
              surface_after.linear_hash != surface_before.linear_hash &&
              tools::post_compute_change_evidence(
                  surface_before, surface_before, surface_after).selected_changed &&
              surface_after.r11.above_one != surface_before.r11.above_one,
          "a changed producer word moves raw hash, linear hash, and the HDR histogram lever");

    tools::PostComputeResourceSnapshot selected_noop_after;
    error.clear();
    CHECK(tools::snapshot_post_compute_resource(
              &surface_table, 14, selected_noop_after, error) &&
              surface_before.linear_hash != surface_after.linear_hash &&
              tools::post_compute_change_evidence(
                  surface_before, surface_after, selected_noop_after).prefix_changed &&
              !tools::post_compute_change_evidence(
                  surface_before, surface_after, selected_noop_after).selected_changed,
          "prefix changed the resource but a selected no-op does not satisfy the change gate");

    std::vector<uint8_t> padding_only_tiled = surface_tiled;
    padding_only_tiled.back() ^= 1u;
    gpu::ShaderResourceTable padding_only_table;
    padding_only_table.resources.push_back(r11_storage(
        padding_only_tiled.data(), padding_only_tiled.size(), width, height, 1,
        static_cast<uint32_t>(gpu::TileMode::Sw64KbRX)));
    tools::PostComputeResourceSnapshot padding_only_after;
    error.clear();
    CHECK(tools::snapshot_post_compute_resource(
              &padding_only_table, 14, padding_only_after, error) &&
              padding_only_after.raw_hash != surface_after.raw_hash &&
              padding_only_after.linear_hash == surface_after.linear_hash &&
              !tools::post_compute_change_evidence(
                  surface_after, surface_after, padding_only_after).selected_changed,
          "tiled padding changes raw provenance but not descriptor-visible change evidence");

    constexpr uint32_t depth = 4;
    std::vector<uint8_t> volume_linear(width * height * depth * sizeof(uint32_t));
    for (uint32_t z = 0; z < depth; ++z)
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const uint32_t word = pack_r11(
                    static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
                const size_t texel = (static_cast<size_t>(z) * height + y) * width + x;
                std::memcpy(volume_linear.data() + texel * sizeof(word), &word, sizeof(word));
            }
    std::vector<uint8_t> volume_tiled(gpu::tiled_volume_bytes(
        width, height, depth, static_cast<uint32_t>(gpu::TileMode::Sw64KbRX), 4));
    CHECK(gpu::tile_volume(volume_tiled.data(), volume_tiled.size(), volume_linear.data(),
                           width, height, depth,
                           static_cast<uint32_t>(gpu::TileMode::Sw64KbRX), 4),
          "3D fixture tiles through the production volume layout");
    gpu::ShaderResourceTable volume_table;
    volume_table.resources.push_back(r11_storage(
        volume_tiled.data(), volume_tiled.size(), width, height, depth,
        static_cast<uint32_t>(gpu::TileMode::Sw64KbRX)));
    tools::PostComputeResourceSnapshot volume;
    error.clear();
    CHECK(tools::snapshot_post_compute_resource(&volume_table, 14, volume, error) &&
              volume.linear == volume_linear && volume.r11.texels == width * height * depth,
          "tiled 3D post-compute storage detiles every voxel in XYZ order");

    gpu::ShaderResourceTable invalid = surface_table;
    invalid.resources.push_back(invalid.resources.front());
    tools::PostComputeResourceSnapshot rejected;
    error.clear();
    CHECK(!tools::snapshot_post_compute_resource(&invalid, 14, rejected, error) &&
              error == "selected compute binding is duplicated",
          "ambiguous selectors fail visibly instead of choosing one resource");

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
