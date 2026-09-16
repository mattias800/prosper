// Exercise preparation ordering through the production live renderer, including rejected
// optimizations. Pixel checks alone cannot detect discarded metadata reads or source copies.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}
int main(int argc, char** argv) {
    const bool control = argc == 2 && std::strcmp(argv[1], "--control") == 0;
    prosper::register_builtin_hle();
    // Valid texels cross the logical byte count in mode 5; truncating tile padding must
    // change visible pixels, not merely remove bytes that no valid coordinate addresses.
    constexpr uint32_t W = 20, H = 17;
    const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    const uint32_t ps_rdna[] = {
        // POS_X/Y_FLOAT occupy v0/v1. Scale pixel centers to normalized UVs so every
        // source texel is observed, rather than broadcasting one constant texture sample.
        0x100000ffu, std::bit_cast<uint32_t>(1.0f / W),
        0x100202ffu, std::bit_cast<uint32_t>(1.0f / H),
        0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResource source{};
    source.cls = ResourceClass::Texture;
    source.format = DataFormat::Unorm8;
    source.num_components = 4; source.binding = 4; source.sgpr_base = 8;
    source.img_dim = 1; source.depth = 1;
    source.width = source.height = 2; source.size = 16;
    source.linear_row_pitch_bytes = 8;
    source.mag_filter = source.min_filter = 0;
    source.gpu_addr = 0x71000000;
    std::vector<uint8_t> color(16, 0);
    for (size_t i = 0; i < color.size(); i += 4) { color[i] = 255; color[i+3] = 255; }
    source.host_data = color.data(); source.host_data_size = color.size();
    auto table = std::make_shared<ShaderResourceTable>();
    table->resources.push_back(source);
    DrawItem producer;
    producer.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
    const PixelSystemInputMapping positions{0x300u, 0x300u};
    producer.fs = recompile_fragment(ps_rdna, std::size(ps_rdna), table.get(), &positions);
    producer.prt = table; producer.vertex_count = 3;
    producer.ps.topology = 3; producer.ps.color_write_mask = 15;
    producer.color0_base = 0x72000000;
    producer.color0_width = W; producer.color0_height = H;
    check(!producer.vs.empty() && !producer.fs.empty(), "fixture shaders compile");
    prosper::frontend::register_live_renderer(".", false);
    auto render = [&](const DrawItem& draw) { return render_submit_items({draw}, W, H); };
    auto center = [&](const std::vector<uint8_t>& image, unsigned r, unsigned g, unsigned b) {
        if (image.size() != W * H * 4u) return false;
        const size_t at = (H / 2 * W + W / 2) * 4u;
        return image[at] == r && image[at+1] == g && image[at+2] == b;
    };
    check(center(render(producer), 255, 0, 0), "real producer renders red");
    DrawItem consumer = producer;
    consumer.color0_base = 0x73000000;
    auto consumer_table = std::make_shared<ShaderResourceTable>(*table);
    auto& sampled = consumer_table->resources[0];
    sampled.gpu_addr = producer.color0_base;
    sampled.width = W; sampled.height = H; sampled.size = W * H * 4;
    sampled.host_data = nullptr; sampled.host_data_size = 0;
    sampled.linear_row_pitch_bytes = 0; sampled.tile_mode = 27;
    sampled.compression_enabled = true; sampled.metadata_addr = 0x74000000;
    std::vector<uint8_t> metadata(gpu_capture_dcc_metadata_footprint(sampled), 0xc0);
    check(!metadata.empty(), "DCC positive control has a real metadata footprint");
    sampled.dcc_metadata_host_data = metadata.data();
    sampled.dcc_metadata_host_data_size = metadata.size();
    consumer.prt = consumer_table;
    using prosper::frontend::reset_texture_decode_scope_stats;
    using prosper::frontend::texture_decode_scope_stats;
    reset_texture_decode_scope_stats();
    check(center(render(consumer), 255, 0, 0), "GPU RTT wins over conflicting guest DCC clear");
    auto stats = texture_decode_scope_stats();
    check(stats.dcc_metadata_read_attempts == (control ? 1u : 0u) &&
          stats.dcc_metadata_read_bytes == (control ? metadata.size() : 0u),
          "GPU winner eliminates metadata reads, with an eager control");
    check(stats.compute_import_span_queries == (control ? 1u : 0u),
          "GPU winner does not derive a rejected compute import span");
    for (size_t i = 0; i < color.size(); i += 4) { color[i] = 0; color[i+1] = 255; }
    render(producer);
    check(center(render(consumer), 0, 255, 0), "renderer-only rewrite replaces the sampled generation");
    // The observed descriptor associates this metadata range with the retained target. An ordered
    // metadata write invalidates its GPU pixels and publishes a uniform clear before preparation.
    notify_guest_gpu_write(sampled.metadata_addr, metadata.size());
    reset_texture_decode_scope_stats();
    check(center(render(consumer), 255, 255, 255), "metadata write publishes a uniform-only RTT clear");
    stats = texture_decode_scope_stats();
    check(stats.dcc_metadata_read_attempts == (control ? 1u : 0u) &&
          stats.compute_import_span_queries == (control ? 1u : 0u),
          "uniform winner avoids rereading DCC metadata and deriving a rejected import span");
    render(producer);
    notify_guest_gpu_write(sampled.gpu_addr, sampled.size);
    reset_texture_decode_scope_stats();
    check(center(render(consumer), 255, 255, 255), "invalidation restores the guest DCC clear path");
    check(texture_decode_scope_stats().dcc_metadata_read_attempts == 1 &&
          texture_decode_scope_stats().dcc_metadata_read_bytes == metadata.size(),
          "guest fallback still reads complete DCC metadata");

    // Keep the destination pool warm and vary the source prefix, including a prefix that covers
    // logical bytes but lacks tile padding. The CPU oracle reproduces the historical staged input.
    sampled.compression_enabled = false; sampled.metadata_addr = 0;
    sampled.dcc_metadata_host_data = nullptr; sampled.dcc_metadata_host_data_size = 0;
    sampled.gpu_addr = 0x75000000; sampled.tile_mode = 5;
    const size_t logical = W * H * 4;
    const size_t padded = tiled_surface_bytes(W, H, sampled.tile_mode, 0, 4);
    check(padded > logical, "short-padding control differs from a short logical source");
    std::vector<uint8_t> tiled(padded);
    for (size_t i = 0; i < tiled.size(); ++i) tiled[i] = static_cast<uint8_t>(i * 13 + 19);
    sampled.host_data = tiled.data();
    std::vector<uint8_t> full_expected;
    for (size_t prefix : {padded, logical - 1, logical + 1, padded}) {
        sampled.host_data_size = prefix;
        std::vector<uint8_t> staged(padded, 0), linear(logical, 0);
        std::copy_n(tiled.data(), prefix, staged.data());
        detile_surface(linear.data(), staged.data(), W, H, sampled.tile_mode, 0, 4);
        auto oracle_table = std::make_shared<ShaderResourceTable>(*consumer_table);
        auto& oracle_resource = oracle_table->resources[0];
        oracle_resource.gpu_addr = 0x76000000; oracle_resource.tile_mode = 0;
        oracle_resource.host_data = linear.data(); oracle_resource.host_data_size = linear.size();
        oracle_resource.linear_row_pitch_bytes = W * 4;
        DrawItem oracle = consumer; oracle.prt = oracle_table; oracle.color0_base = 0x77000000;
        const auto expected = render(oracle);
        check(expected == linear, "pixel-center shader observes every texel of the CPU oracle");
        if (prefix == padded) full_expected = expected;
        else check(expected != full_expected, "short backing changes visible oracle pixels");
        reset_texture_decode_scope_stats();
        const auto actual = render(consumer);
        stats = texture_decode_scope_stats();
        check(!actual.empty() && actual == expected, "full/short source matches the independent staged pixel oracle");
        const size_t expected_copy = control || prefix < logical ? std::min(prefix, logical) : 0;
        check(stats.generic_source_copied_bytes == expected_copy &&
              stats.generic_source_copy_deferrals == (control ? 0u : 1u),
              "copy is omitted only when complete detiling supersedes it; short recovery remains");
    }
    // A viable guest-backed import must still derive its exact span. A test containing only
    // renderer winners and host mirrors would also pass if every import query were deleted.
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    uint64_t guest = 0;
    check(map && unmap && map(reinterpret_cast<uint64_t>(&guest), 0x10000, 2, 0,
                             reinterpret_cast<uint64_t>("texture-preparation"), 0) == 0 && guest,
          "positive import control maps real guest backing");
    if (guest) {
        std::memcpy(reinterpret_cast<void*>(guest), tiled.data(), tiled.size());
        const auto expected = render(consumer);
        sampled.host_data = nullptr; sampled.host_data_size = 0; sampled.gpu_addr = guest;
        reset_texture_decode_scope_stats();
        check(render(consumer) == expected, "guest-backed eligible import preserves exact sampled pixels");
        check(texture_decode_scope_stats().compute_import_span_queries == 1,
              "eligible import still resolves its complete span");
        unmap(guest, 0x10000, 0, 0, 0, 0);
    }
    std::printf("texture preparation: %d failures\n", failures);
    return failures ? 1 : 0;
}
