// Real-device regression for destination-only renderer RTT publication.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"
#include "fixtures/render_runner.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL: %s\n", msg); ++fails; } } while (0)

static int run_destination_mirror_regression() {
    // This mode registers and renders before the first compute dispatch, so compute adopts the
    // renderer device. The ordinary suite deliberately keeps its independent-device coverage.
    constexpr uint32_t W = 8, H = 4;
#ifdef _WIN32
    _putenv_s("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB", "0");
#else
    setenv("PROSPER_COMPUTE_IMAGE_CACHE_MIN_KB", "0", 1);
#endif
    std::vector<uint8_t> destination(W * H * 4, 0x9d);
    const uint64_t address = reinterpret_cast<uint64_t>(destination.data());
    prosper::frontend::register_live_renderer(".", false);

    const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u,
        0x10020B01u, 0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u,
        0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    const uint32_t ps_rdna[] = {
        0x100000ffu, std::bit_cast<uint32_t>(1.0f / W),
        0x100202ffu, std::bit_cast<uint32_t>(1.0f / H),
        0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResource source{};
    source.cls = ResourceClass::Texture;
    source.format = DataFormat::Unorm8;
    source.num_components = 4;
    source.binding = 4;
    source.sgpr_base = 8;
    source.img_dim = 1;
    source.width = source.height = 2;
    source.depth = 1;
    source.size = 16;
    source.linear_row_pitch_bytes = 8;
    source.mag_filter = source.min_filter = 0;
    std::vector<uint8_t> red(16, 0);
    for (size_t i = 0; i < red.size(); i += 4) { red[i] = red[i + 3] = 255; }
    source.host_data = red.data(); source.host_data_size = red.size();
    auto producer_table = std::make_shared<ShaderResourceTable>();
    producer_table->resources.push_back(source);
    DrawItem producer;
    producer.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
    const PixelSystemInputMapping positions{0x300u, 0x300u};
    producer.fs = recompile_fragment(ps_rdna, std::size(ps_rdna), producer_table.get(), &positions);
    producer.prt = producer_table; producer.vertex_count = 3;
    producer.ps.topology = 3; producer.ps.color_write_mask = 15;
    producer.ps.color0_format = VK_FORMAT_R8G8B8A8_UNORM;
    producer.color0_base = address; producer.color0_width = W; producer.color0_height = H;
    CHECK(!producer.vs.empty() && !producer.fs.empty(), "RTT mirror producer shaders compile");
    const auto render = [&](const DrawItem& draw) { return render_submit_items({draw}, W, H); };
    const auto initial = render(producer);
    CHECK(initial.size() == W * H * 4 && initial[0] == 255,
          "RTT mirror producer materializes a red persistent renderer target");

    // An invalid renderer target must never act as an input seed, while it remains eligible as an
    // exact destination for a full overwrite.
    // Preserve the renderer registry entry but make its device image non-authoritative. This is
    // the CPU-newer state a destination lease may overwrite; a queued write drain would erase the
    // entry and test only its absence rather than destination-only admission.
    auto cpu_newer = std::make_shared<std::vector<uint8_t>>(destination.size(), 0x11);
    notify_live_render_target_image_written(
        {address, W, H, LiveTargetPixelFormat::Rgba8Unorm, std::move(cpu_newer)});
    LiveTargetImageImport source_import;
    LiveTargetImageRequest source_request{};
    source_request.width = W; source_request.height = H;
    CHECK(!import_live_render_target_image(address, source_request, source_import),
          "invalid renderer RTT is refused as a strict sampled source before destination admission");
    LiveTargetImageImport wrong_destination;
    CHECK(!borrow_live_render_target_image_destination(
              address, {W + 1, H, LiveTargetPixelFormat::Rgba8Unorm}, wrong_destination) &&
          !borrow_live_render_target_image_destination(
              address, {W, H, LiveTargetPixelFormat::R11G11B10Float}, wrong_destination),
          "destination lease requires exact renderer shape and format identity");

    static const uint32_t store_black[] = {
        0x7E080300u, 0x7E0A0301u, // v4=x, v5=y
        0x7E000280u, 0x7E020280u, 0x7E040280u, // RGB=0
        0x7E0602F2u, // A=1
        0xF0200F08u, 0x00020004u, // image_store v0..v3 at v4,v5 through s[8:15]
        0xBF810000u,
    };
    ShaderResource output{};
    output.cls = ResourceClass::StorageImage; output.format = DataFormat::Unorm8;
    output.num_components = 4; output.binding = 5; output.sgpr_base = 8;
    output.img_dim = 1; output.width = W; output.height = H; output.depth = 1;
    output.gpu_addr = address; output.size = static_cast<uint32_t>(destination.size());
    ShaderResourceTable table; table.resources.push_back(output);
    ComputeShaderConfig config;
    config.user_sgprs.resize(16); config.local_x = W; config.local_y = config.local_z = 1;
    config.threads_x = W; config.threads_y = H; config.threads_z = 1; config.tidig_comp_cnt = 1;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Unorm8, 4);
    const auto spirv = recompile_compute(store_black, std::size(store_black), &table, config);
    CHECK(!spirv.empty(), "RTT mirror full-overwrite storage shader recompiles");
    ComputeItem item;
    item.spirv = spirv; item.resources = std::make_shared<ShaderResourceTable>(table);
    item.launch.threads_x = W; item.launch.threads_y = H; item.launch.threads_z = 1;
    item.launch.local_x = W; item.launch.local_y = H; item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    item.code_addr = 0x37310001u;

    const auto before_failure = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    CHECK(!spirv.empty() && !prosper::frontend::execute_live_compute_items({item}),
          "failed storage completion rejects destination mirror publication");
    const auto after_failure = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(after_failure.borrowed == before_failure.borrowed + 1 &&
              after_failure.failed == before_failure.failed + 1 &&
              after_failure.recorded == before_failure.recorded + 1 &&
              after_failure.published == before_failure.published &&
              !import_live_render_target_image(address, source_request, source_import),
          "failed destination write never restores readable renderer authority");

    const auto before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!spirv.empty() && prosper::frontend::execute_live_compute_items({item}),
          "full-overwrite storage dispatch completes into invalid renderer destination");
    const auto after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(after.candidates == before.candidates + 1 && after.borrowed == before.borrowed + 1 &&
              after.recorded == before.recorded + 1 && after.published == before.published + 1 &&
              after.failed == before.failed,
          "destination mirror records its device copy and publishes only after writeback");
    CHECK(import_live_render_target_image(address, source_request, source_import) && source_import.valid(),
          "completed destination mirror restores the exact renderer image as a readable source");
    release_live_render_target_image(address);

    auto consumer_table = std::make_shared<ShaderResourceTable>(*producer_table);
    ShaderResource& sampled = consumer_table->resources[0];
    sampled.gpu_addr = address; sampled.width = W; sampled.height = H; sampled.size = destination.size();
    sampled.host_data = nullptr; sampled.host_data_size = 0; sampled.linear_row_pitch_bytes = 0;
    DrawItem consumer = producer;
    consumer.prt = consumer_table; consumer.color0_base = address + 0x100000;
    const auto mirrored = render(consumer);
    CHECK(mirrored.size() == W * H * 4,
          "renderer can sample the completed device-mirrored storage result");

    // Reissue identical CPU-newer bytes: this deliberately invalidates GPU authority without
    // adding a readable source image, so the repeat can exercise the unchanged-result completion
    // path without the same-image source/destination collision.
    auto same_cpu_result = std::make_shared<std::vector<uint8_t>>(destination);
    notify_live_render_target_image_written(
        {address, W, H, LiveTargetPixelFormat::Rgba8Unorm, std::move(same_cpu_result)});
    const auto warm_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(prosper::frontend::execute_live_compute_items({item}),
          "unchanged full-overwrite result completes after destination publication");
    const auto warm_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(warm_after.recorded == warm_before.recorded + 1 &&
              warm_after.published == warm_before.published + 1 &&
              import_live_render_target_image(address, source_request, source_import),
          "unchanged result records and publishes after identical CPU-newer invalidation");
    release_live_render_target_image(address);
    // Compare the device-mirror sample with the ordinary CPU publication of exactly its guest
    // bytes. This proves the first copy's pixels, without assuming a particular shader packing.
    auto cpu_result = std::make_shared<std::vector<uint8_t>>(destination);
    notify_live_render_target_image_written(
        {address, W, H, LiveTargetPixelFormat::Rgba8Unorm, std::move(cpu_result)});
    const auto fallback = render(consumer);
    CHECK(mirrored == fallback,
          "device destination mirror pixels equal the independent CPU publication fallback");

    // A renderer image may be the source of a partial write and the destination of its completed
    // result in one ordered dispatch. The guest bytes are still black from the prior full write;
    // repainting the renderer target red makes rows below the partial write prove the GPU seed.
    const auto red_seed = render(producer);
    CHECK(red_seed.size() == destination.size() &&
              !std::equal(red_seed.begin() + W * 4, red_seed.end(),
                          destination.begin() + W * 4),
          "partial-write source differs from stale guest rows");
    const auto* color_target = prosper::test::find_persistent_color_target(
        address, W, H, VK_FORMAT_R8G8B8A8_UNORM);
    const auto source_layout = color_target ? color_target->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    ComputeItem partial = item;
    partial.launch.threads_y = partial.launch.local_y = partial.launch.groups_y = 1;
    partial.code_addr = 0x37310002u;
    const auto partial_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(prosper::frontend::execute_live_compute_items({partial}),
          "partial storage writer completes from the renderer image");
    const auto partial_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(partial_after.recorded == partial_before.recorded + 1 &&
              partial_after.published == partial_before.published + 1,
          "same-image source and destination publish only after completed writeback");
    CHECK(std::equal(destination.begin() + W * 4, destination.end(),
                     red_seed.begin() + W * 4),
          "partial writer preserves untouched rows from the renderer image");
    CHECK(import_live_render_target_image(address, source_request, source_import) &&
              source_import.valid(),
          "completed partial write leaves its exact renderer image readable");
    release_live_render_target_image(address);
    std::vector<uint8_t> mirrored_partial_pixels;
    std::string mirror_error;
    CHECK(prosper::test::readback_persistent_color_target(
              address, W, H, VK_FORMAT_R8G8B8A8_UNORM,
              mirrored_partial_pixels, mirror_error) &&
              mirrored_partial_pixels == destination,
          "same-image destination has the exact completed guest pixels");
    color_target = prosper::test::find_persistent_color_target(
        address, W, H, VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(color_target && color_target->pin_count == 0 && color_target->layout == source_layout,
          "source and destination leases release both pins and restore the saved layout");

    // The next partial dispatch must seed from the just-published image even with identical
    // shader inputs. This does not assert that the cached unchanged-writeback shortcut ran.
    const auto repeat_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(prosper::frontend::execute_live_compute_items({partial}),
          "second partial writer consumes the first mirrored result");
    const auto repeat_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(repeat_after.recorded == repeat_before.recorded + 1 &&
              repeat_after.published == repeat_before.published + 1 &&
              std::equal(destination.begin() + W * 4, destination.end(),
                         red_seed.begin() + W * 4),
          "repeat partial result preserves untouched rows and remains published");
    color_target = prosper::test::find_persistent_color_target(
        address, W, H, VK_FORMAT_R8G8B8A8_UNORM);
    CHECK(color_target && color_target->pin_count == 0 && color_target->layout == source_layout,
          "repeat partial write releases both leases at the saved layout");

    const auto failed_partial_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    CHECK(!prosper::frontend::execute_live_compute_items({partial}),
          "failed same-image partial completion is reported");
    const auto failed_partial_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(failed_partial_after.recorded == failed_partial_before.recorded + 1 &&
              failed_partial_after.failed == failed_partial_before.failed + 1 &&
              failed_partial_after.published == failed_partial_before.published &&
              !import_live_render_target_image(address, source_request, source_import),
          "failed same-image completion revokes renderer authority without publication");
    color_target = prosper::test::find_persistent_color_target(
        address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false);
    CHECK(color_target && color_target->pin_count == 0 &&
              color_target->layout == source_layout,
          "failed same-image completion releases both leases at the saved layout");

    // Packed R11 is the primary destination-mirror shape. The first complete dispatch removes the
    // CPU snapshot; the second writes only row zero, so rows one through H-1 can survive only if
    // the new renderer-image -> canonical R32_UINT seed supplied their exact packed words.
    std::vector<uint32_t> r11_words(W * H, 0xdeadbeefu);
    const uint64_t r11_address = reinterpret_cast<uint64_t>(r11_words.data());
    DrawItem r11_producer = producer;
    r11_producer.color0_base = r11_address;
    r11_producer.ps.color0_format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    CHECK(!render(r11_producer).empty(), "R11 destination mirror producer materializes a renderer target");
    auto r11_cpu_newer = std::make_shared<std::vector<uint8_t>>(W * H * sizeof(uint32_t), 0x33);
    notify_live_render_target_image_written(
        {r11_address, W, H, LiveTargetPixelFormat::R11G11B10Float, std::move(r11_cpu_newer)});
    LiveTargetImageImport r11_source;
    CHECK(!import_live_render_target_image(r11_address, source_request, r11_source),
          "CPU-newer R11 target is refused as a strict source before its first mirror");
    static const uint32_t store_coordinates[] = {
        0x7E080300u, 0x7E0A0301u, // preserve x/y in v0/v1; copy to coordinates v4/v5
        0x7E040280u, 0x7E0602F2u, // B=0, A=1
        0xF0200F08u, 0x00020004u, 0xBF810000u,
    };
    ShaderResource r11_output{};
    r11_output.cls = ResourceClass::StorageImage; r11_output.format = DataFormat::Float10_11_11;
    r11_output.num_components = 3; r11_output.binding = 5; r11_output.sgpr_base = 8;
    r11_output.img_dim = 1; r11_output.width = W; r11_output.height = H; r11_output.depth = 1;
    r11_output.gpu_addr = r11_address; r11_output.size = static_cast<uint32_t>(W * H * sizeof(uint32_t));
    ShaderResourceTable r11_table; r11_table.resources.push_back(r11_output);
    ComputeShaderConfig r11_config = config;
    r11_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Float10_11_11, 3);
    const auto r11_full_spirv = recompile_compute(
        store_coordinates, std::size(store_coordinates), &r11_table, r11_config);
    CHECK(!r11_full_spirv.empty(), "packed R11 full-overwrite producer recompiles");
    ComputeItem r11_full = item;
    r11_full.spirv = r11_full_spirv;
    r11_full.resources = std::make_shared<ShaderResourceTable>(r11_table);
    r11_full.code_addr = 0x37310011u;
    CHECK(!r11_full_spirv.empty() && prosper::frontend::execute_live_compute_items({r11_full}),
          "packed R11 first dispatch mirrors its exact words and removes CPU fallback authority");
    const std::vector<uint32_t> first_r11 = r11_words;
    CHECK(std::any_of(first_r11.begin(), first_r11.end(), [](uint32_t word) { return word != 0; }) &&
              import_live_render_target_image(r11_address, source_request, r11_source),
          "first packed R11 mirror publishes an exact readable renderer result");
    release_live_render_target_image(r11_address);
    const auto r11_partial_spirv = recompile_compute(store_black, std::size(store_black),
                                                      &r11_table, r11_config);
    CHECK(!r11_partial_spirv.empty(), "packed R11 partial writer recompiles");
    ComputeItem r11_partial = r11_full;
    r11_partial.spirv = r11_partial_spirv;
    r11_partial.resources = std::make_shared<ShaderResourceTable>(r11_table);
    r11_partial.launch.threads_y = 1; r11_partial.launch.local_y = 1; r11_partial.launch.groups_y = 1;
    r11_partial.code_addr = 0x37310012u;
    const auto r11_seed_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!r11_partial_spirv.empty() && prosper::frontend::execute_live_compute_items({r11_partial}),
          "packed R11 partial consumer executes from the renderer GPU seed");
    const auto r11_seed_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(r11_seed_after.r11_source_seed_recorded == r11_seed_before.r11_source_seed_recorded + 1,
          "packed R11 partial writer records the exact renderer-image source seed");
    CHECK(std::equal(r11_words.begin() + W, r11_words.end(), first_r11.begin() + W),
          "packed R11 partial writer preserves every untouched canonical word from GPU source");
    return fails ? 1 : 0;
}

int main() {
    return run_destination_mirror_regression();
}
