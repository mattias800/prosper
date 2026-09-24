// Real-device regression for destination-only renderer RTT publication.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"
#include "shared/live/gpu_retile.hpp"
#include "fixtures/render_runner.h"
#include "gpu/texture/tile.hpp"

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

    // Native RGBA16F is a recurring renderer-publication format in GTA V. Exercise the exact
    // eight-byte representation: the full write proves completed publication, while the partial
    // write can preserve the lower rows only by seeding from the renderer's current GPU image.
    const bool rgba16_mirror_disabled =
        std::getenv("PROSPER_NO_RGBA16_COMPUTE_RTT_MIRROR") != nullptr;
    std::vector<uint16_t> rgba16_words(W * H * 4, 0x7bffu);
    const uint64_t rgba16_address = reinterpret_cast<uint64_t>(rgba16_words.data());
    DrawItem rgba16_producer = producer;
    rgba16_producer.color0_base = rgba16_address;
    rgba16_producer.ps.color0_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    const uint16_t rgba16_sample[] = {0x4000u, 0xbc00u, 0x3800u, 0x3c00u};
    static const uint32_t rgba16_ps_rdna[] = {
        0x7E0002FFu, 0x40000000u, // v0 = 2.0
        0x7E0202FFu, 0xBF800000u, // v1 = -1.0
        0x7E0402FFu, 0x3F000000u, // v2 = 0.5
        0x7E0602F2u,              // v3 = 1.0
        0xF800000Fu, 0x03020100u, 0xBF810000u,
    };
    auto rgba16_producer_table = std::make_shared<ShaderResourceTable>();
    rgba16_producer.prt = rgba16_producer_table;
    rgba16_producer.fs = recompile_fragment(
        rgba16_ps_rdna, std::size(rgba16_ps_rdna), rgba16_producer_table.get(), &positions);
    CHECK(!rgba16_producer.fs.empty(), "RGBA16F renderer producer recompiles");
    CHECK(!render(rgba16_producer).empty(),
          "RGBA16F destination mirror producer materializes a renderer target");
    auto rgba16_cpu_newer =
        std::make_shared<std::vector<uint8_t>>(W * H * 8u, 0x53);
    notify_live_render_target_image_written(
        {rgba16_address, W, H, LiveTargetPixelFormat::Rgba16Float,
         std::move(rgba16_cpu_newer)});
    LiveTargetImageImport rgba16_source;
    CHECK(!import_live_render_target_image(rgba16_address, source_request, rgba16_source),
          "CPU-newer RGBA16F target is refused as a strict source before completion");

    ShaderResource rgba16_output{};
    rgba16_output.cls = ResourceClass::StorageImage;
    rgba16_output.format = DataFormat::Float16;
    rgba16_output.num_components = 4;
    rgba16_output.binding = 5;
    rgba16_output.sgpr_base = 8;
    rgba16_output.img_dim = 1;
    rgba16_output.width = W;
    rgba16_output.height = H;
    rgba16_output.depth = 1;
    rgba16_output.gpu_addr = rgba16_address;
    rgba16_output.size = static_cast<uint32_t>(rgba16_words.size() * sizeof(uint16_t));
    ShaderResourceTable rgba16_table;
    rgba16_table.resources.push_back(rgba16_output);
    ComputeShaderConfig rgba16_config = config;
    rgba16_config.local_y = H;
    rgba16_config.native_storage_format_support =
        native_storage_format_support_bit(DataFormat::Float16, 4);
    const auto rgba16_spirv = recompile_compute(
        store_black, std::size(store_black), &rgba16_table, rgba16_config);
    CHECK(!rgba16_spirv.empty(), "native RGBA16F storage writer recompiles");
    ComputeItem rgba16_full = item;
    rgba16_full.spirv = rgba16_spirv;
    rgba16_full.resources = std::make_shared<ShaderResourceTable>(rgba16_table);
    rgba16_full.code_addr = 0x37310021u;
    const auto rgba16_full_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!rgba16_spirv.empty() &&
              prosper::frontend::execute_live_compute_items({rgba16_full}),
          "native RGBA16F full write completes");
    const auto rgba16_full_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    const uint16_t rgba16_black[] = {0u, 0u, 0u, 0x3c00u};
    bool rgba16_expected = true;
    for (size_t i = 0; i < rgba16_words.size(); ++i)
        rgba16_expected &= rgba16_words[i] == rgba16_black[i % 4u];
    CHECK(rgba16_expected,
          "native RGBA16F guest writeback contains exact half-float black and alpha one");
    if (rgba16_mirror_disabled) {
        CHECK(rgba16_full_after.candidates == rgba16_full_before.candidates &&
                  rgba16_full_after.borrowed == rgba16_full_before.borrowed &&
                  rgba16_full_after.recorded == rgba16_full_before.recorded &&
                  rgba16_full_after.published == rgba16_full_before.published &&
                  !import_live_render_target_image(
                      rgba16_address, source_request, rgba16_source),
              "disabled RGBA16F admission uses ordinary CPU publication");
        LiveTargetSnapshot rgba16_snapshot;
        CHECK(read_live_render_target(rgba16_address, rgba16_snapshot) &&
                  rgba16_snapshot.format == LiveTargetPixelFormat::Rgba16Float &&
                  rgba16_snapshot.pixels &&
                  rgba16_snapshot.pixels->size() == rgba16_words.size() * sizeof(uint16_t) &&
                  std::memcmp(rgba16_snapshot.pixels->data(), rgba16_words.data(),
                              rgba16_snapshot.pixels->size()) == 0,
              "disabled RGBA16F path publishes exact fallback bytes");
    } else {
        CHECK(rgba16_full_after.candidates == rgba16_full_before.candidates + 1 &&
                  rgba16_full_after.borrowed == rgba16_full_before.borrowed + 1 &&
                  rgba16_full_after.recorded == rgba16_full_before.recorded + 1 &&
                  rgba16_full_after.published == rgba16_full_before.published + 1 &&
                  import_live_render_target_image(
                      rgba16_address, source_request, rgba16_source) &&
                  rgba16_source.valid() &&
                  rgba16_source.format == LiveTargetPixelFormat::Rgba16Float &&
                  rgba16_source.native_format == VK_FORMAT_R16G16B16A16_SFLOAT,
              "completed RGBA16F write publishes the exact renderer image");
        release_live_render_target_image(rgba16_address);
        std::vector<uint8_t> rgba16_full_pixels;
        std::string rgba16_readback_error;
        CHECK(prosper::test::readback_persistent_color_target(
                  rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                  rgba16_full_pixels, rgba16_readback_error) &&
                  rgba16_full_pixels.size() == rgba16_words.size() * sizeof(uint16_t) &&
                  std::memcmp(rgba16_full_pixels.data(), rgba16_words.data(),
                              rgba16_full_pixels.size()) == 0,
              "RGBA16F renderer destination equals guest writeback bit for bit");
    }

    // A plain 2D T# can be written through a one-layer DIM=2D_ARRAY instruction.
    // This is the reflected shape of Outer Wilds' full-resolution Float16 output:
    // the private storage view must be arrayed, but the renderer and guest surface
    // remain ordinary 2D. Repaint first so an unchanged-result skip cannot satisfy
    // the publication assertion without recording this writer.
    CHECK(!render(rgba16_producer).empty(),
          "RGBA16F renderer repaints before the one-layer array writer");
    static const uint32_t store_black_array[] = {
        0x7E080300u, 0x7E0A0301u, 0x7E0C0280u, // v4=x, v5=y, v6=layer zero
        0x7E000280u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF0200F28u, 0x00020004u, 0xBF810000u,
    };
    const auto rgba16_array_spirv = recompile_compute(
        store_black_array, std::size(store_black_array), &rgba16_table, rgba16_config);
    CHECK(!rgba16_array_spirv.empty(),
          "one-layer array instruction recompiles for a 2D Float16 descriptor");
    ComputeItem rgba16_array_full = rgba16_full;
    rgba16_array_full.spirv = rgba16_array_spirv;
    rgba16_array_full.code_addr = 0x37310023u;
    const auto rgba16_array_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!rgba16_array_spirv.empty() &&
              prosper::frontend::execute_live_compute_items({rgba16_array_full}),
          "one-layer array writer completes into the ordinary 2D guest surface");
    const auto rgba16_array_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    bool rgba16_array_expected = true;
    for (size_t i = 0; i < rgba16_words.size(); ++i)
        rgba16_array_expected &= rgba16_words[i] == rgba16_black[i % 4u];
    CHECK(rgba16_array_expected,
          "one-layer array writer publishes exact half-float guest bytes");
    if (!rgba16_mirror_disabled) {
        CHECK(rgba16_array_after.candidates == rgba16_array_before.candidates + 1 &&
                  rgba16_array_after.recorded == rgba16_array_before.recorded + 1 &&
                  rgba16_array_after.published == rgba16_array_before.published + 1,
              "completed one-layer array writer publishes through the renderer mirror");
        std::vector<uint8_t> rgba16_array_pixels;
        std::string rgba16_array_error;
        CHECK(prosper::test::readback_persistent_color_target(
                  rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                  rgba16_array_pixels, rgba16_array_error) &&
                  rgba16_array_pixels.size() == rgba16_words.size() * sizeof(uint16_t) &&
                  std::memcmp(rgba16_array_pixels.data(), rgba16_words.data(),
                              rgba16_array_pixels.size()) == 0,
              "one-layer array mirror pixels equal exact guest half-float writeback");
    }

    CHECK(!render(rgba16_producer).empty(),
          "RGBA16F renderer repaints its image before the partial write");
    std::vector<uint8_t> rgba16_seed;
    std::string rgba16_seed_error;
    const bool rgba16_seed_valid = prosper::test::readback_persistent_color_target(
        rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
        rgba16_seed, rgba16_seed_error) &&
        rgba16_seed.size() == rgba16_words.size() * sizeof(uint16_t);
    bool rgba16_seed_expected = rgba16_seed_valid;
    if (rgba16_seed_valid) {
        std::vector<uint16_t> seed_words(rgba16_words.size());
        std::memcpy(seed_words.data(), rgba16_seed.data(), rgba16_seed.size());
        for (size_t i = 0; i < rgba16_words.size(); ++i)
            rgba16_seed_expected &= seed_words[i] == rgba16_sample[i % 4u];
    }
    CHECK(rgba16_seed_expected,
          "RGBA16F renderer seed preserves negative and greater-than-one half floats");
    const auto* rgba16_target = prosper::test::find_persistent_color_target(
        rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT);
    const VkImage rgba16_image = rgba16_target ? rgba16_target->image : VK_NULL_HANDLE;
    const VkImageLayout rgba16_layout =
        rgba16_target ? rgba16_target->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    const uint32_t rgba16_renderer_pins = rgba16_target ? rgba16_target->pin_count : 0;
    ComputeShaderConfig rgba16_partial_config = rgba16_config;
    rgba16_partial_config.local_y = 1;
    const auto rgba16_partial_spirv = recompile_compute(
        store_black, std::size(store_black), &rgba16_table, rgba16_partial_config);
    CHECK(!rgba16_partial_spirv.empty(), "native RGBA16F partial writer recompiles");
    ComputeItem rgba16_partial = rgba16_full;
    rgba16_partial.spirv = rgba16_partial_spirv;
    rgba16_partial.launch.threads_y = 1;
    rgba16_partial.launch.local_y = 1;
    rgba16_partial.launch.groups_y = 1;
    rgba16_partial.code_addr = 0x37310022u;
    const auto rgba16_partial_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(prosper::frontend::execute_live_compute_items({rgba16_partial}),
          "native RGBA16F partial write completes from renderer pixels");
    const auto rgba16_partial_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(rgba16_seed_valid &&
              std::memcmp(reinterpret_cast<const uint8_t*>(rgba16_words.data()) + W * 8u,
                          rgba16_seed.data() + W * 8u,
                          rgba16_seed.size() - W * 8u) == 0,
          "RGBA16F partial writer preserves exact untouched renderer rows");
    if (rgba16_mirror_disabled) {
        CHECK(rgba16_partial_after.candidates == rgba16_partial_before.candidates &&
                  rgba16_partial_after.borrowed == rgba16_partial_before.borrowed &&
                  rgba16_partial_after.recorded == rgba16_partial_before.recorded &&
                  rgba16_partial_after.published == rgba16_partial_before.published &&
                  rgba16_partial_after.rgba16_source_seed_recorded ==
                      rgba16_partial_before.rgba16_source_seed_recorded,
              "disabled RGBA16F partial path refuses destination admission");
        LiveTargetSnapshot rgba16_snapshot;
        CHECK(read_live_render_target(rgba16_address, rgba16_snapshot) &&
                  rgba16_snapshot.format == LiveTargetPixelFormat::Rgba16Float &&
                  rgba16_snapshot.pixels &&
                  rgba16_snapshot.pixels->size() == rgba16_words.size() * sizeof(uint16_t) &&
                  std::memcmp(rgba16_snapshot.pixels->data(), rgba16_words.data(),
                              rgba16_snapshot.pixels->size()) == 0,
              "disabled RGBA16F partial path publishes exact fallback bytes");
    } else {
        CHECK(rgba16_partial_after.candidates == rgba16_partial_before.candidates + 1 &&
                  rgba16_partial_after.borrowed == rgba16_partial_before.borrowed + 1 &&
                  rgba16_partial_after.recorded == rgba16_partial_before.recorded + 1 &&
                  rgba16_partial_after.published == rgba16_partial_before.published + 1 &&
                  rgba16_partial_after.rgba16_source_seed_recorded ==
                      rgba16_partial_before.rgba16_source_seed_recorded + 1,
              "RGBA16F partial write seeds and publishes the same renderer image");
        std::vector<uint8_t> rgba16_partial_pixels;
        CHECK(prosper::test::readback_persistent_color_target(
                  rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                  rgba16_partial_pixels, rgba16_seed_error) &&
                  rgba16_partial_pixels.size() == rgba16_words.size() * sizeof(uint16_t) &&
                  std::memcmp(rgba16_partial_pixels.data(), rgba16_words.data(),
                              rgba16_partial_pixels.size()) == 0,
              "RGBA16F partial destination equals guest writeback bit for bit");
        rgba16_target = prosper::test::find_persistent_color_target(
            rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT);
        CHECK(rgba16_target && rgba16_target->image == rgba16_image &&
                  rgba16_target->pin_count == rgba16_renderer_pins &&
                  rgba16_target->layout == rgba16_layout,
              "RGBA16F source and destination release their leases at the saved layout");

        const auto rgba16_failure_before =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        prosper::frontend::live_compute_fail_next_storage_readback_for_test();
        CHECK(!prosper::frontend::execute_live_compute_items({rgba16_partial}),
              "failed RGBA16F partial completion is reported");
        const auto rgba16_failure_after =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(rgba16_failure_after.borrowed == rgba16_failure_before.borrowed + 1 &&
                  rgba16_failure_after.recorded == rgba16_failure_before.recorded + 1 &&
                  rgba16_failure_after.failed == rgba16_failure_before.failed + 1 &&
                  rgba16_failure_after.published == rgba16_failure_before.published &&
                  rgba16_failure_after.rgba16_source_seed_recorded ==
                      rgba16_failure_before.rgba16_source_seed_recorded + 1 &&
                  !import_live_render_target_image(
                      rgba16_address, source_request, rgba16_source),
              "failed RGBA16F completion revokes authority without publication");
        rgba16_target = prosper::test::find_persistent_color_target(
            rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, false);
        CHECK(rgba16_target && rgba16_target->image == rgba16_image &&
                  rgba16_target->pin_count == rgba16_renderer_pins &&
                  rgba16_target->layout == rgba16_layout,
              "failed RGBA16F completion releases both leases at the saved layout");
    }

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
    CHECK(!render(r11_producer).empty(),
          "packed R11 renderer repaints its target before the partial dispatch");
    std::vector<uint8_t> r11_seed_pixels;
    std::string r11_readback_error;
    CHECK(prosper::test::readback_persistent_color_target(
              r11_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
              r11_seed_pixels, r11_readback_error) &&
              r11_seed_pixels.size() == r11_words.size() * sizeof(uint32_t),
          "packed R11 renderer seed has exact readable words");
    std::vector<uint32_t> r11_seed_words(r11_words.size());
    if (r11_seed_pixels.size() == r11_seed_words.size() * sizeof(uint32_t))
        std::memcpy(r11_seed_words.data(), r11_seed_pixels.data(), r11_seed_pixels.size());
    CHECK(!std::equal(r11_seed_words.begin() + W, r11_seed_words.end(),
                      first_r11.begin() + W),
          "packed R11 renderer seed differs from stale guest words below the written row");
    const auto* r11_target = prosper::test::find_persistent_color_target(
        r11_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    const VkImage r11_image = r11_target ? r11_target->image : VK_NULL_HANDLE;
    const VkImageLayout r11_layout = r11_target ? r11_target->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    // Packed HDR mip producers keep an independent renderer retention pin until sampled.
    // Compute must return to that baseline after releasing its source and destination leases.
    const uint32_t r11_renderer_pins = r11_target ? r11_target->pin_count : 0;
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
    CHECK(r11_seed_after.r11_source_seed_recorded == r11_seed_before.r11_source_seed_recorded + 1 &&
              r11_seed_after.borrowed == r11_seed_before.borrowed + 1 &&
              r11_seed_after.recorded == r11_seed_before.recorded + 1 &&
              r11_seed_after.published == r11_seed_before.published + 1 &&
              r11_seed_after.failed == r11_seed_before.failed,
          "packed R11 partial writer seeds and publishes the same renderer image");
    CHECK(std::equal(r11_words.begin() + W, r11_words.end(), r11_seed_words.begin() + W) &&
              !std::equal(r11_words.begin(), r11_words.begin() + W, r11_seed_words.begin()),
          "packed R11 partial writer changes row zero and preserves GPU-seeded rows");
    CHECK(import_live_render_target_image(r11_address, source_request, r11_source) &&
              r11_source.valid(),
          "packed R11 partial result remains a strict renderer source");
    release_live_render_target_image(r11_address);
    std::vector<uint8_t> r11_mirrored_pixels;
    CHECK(prosper::test::readback_persistent_color_target(
              r11_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
              r11_mirrored_pixels, r11_readback_error) &&
              r11_mirrored_pixels.size() == r11_words.size() * sizeof(uint32_t) &&
              std::memcmp(r11_mirrored_pixels.data(), r11_words.data(), r11_mirrored_pixels.size()) == 0,
          "packed R11 renderer image contains the exact completed guest words");
    r11_target = prosper::test::find_persistent_color_target(
        r11_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    CHECK(r11_target && r11_target->image == r11_image &&
              r11_target->pin_count == r11_renderer_pins &&
              r11_target->layout == r11_layout,
          "packed R11 source and destination release their pins at the saved layout");

    const std::vector<uint32_t> first_partial_r11 = r11_words;
    static const uint32_t store_red_r11[] = {
        0x7E080300u, 0x7E0A0301u, // v4=x, v5=y
        0x7E0002F2u, 0x7E020280u, 0x7E040280u, // R=1, G/B=0
        0x7E0602F2u, 0xF0200F08u, 0x00020004u, 0xBF810000u,
    };
    const auto r11_red_spirv = recompile_compute(
        store_red_r11, std::size(store_red_r11), &r11_table, r11_config);
    CHECK(!r11_red_spirv.empty(), "changed packed R11 partial writer recompiles");
    ComputeItem r11_partial_changed = r11_partial;
    r11_partial_changed.spirv = r11_red_spirv;
    r11_partial_changed.code_addr = 0x37310014u;
    const auto r11_repeat_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(prosper::frontend::execute_live_compute_items({r11_partial_changed}),
          "second packed R11 partial dispatch changes the published renderer result");
    const auto r11_repeat_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(r11_repeat_after.r11_source_seed_recorded == r11_repeat_before.r11_source_seed_recorded + 1 &&
              r11_repeat_after.recorded == r11_repeat_before.recorded + 1 &&
              r11_repeat_after.published == r11_repeat_before.published + 1 &&
              !std::equal(r11_words.begin(), r11_words.begin() + W,
                          first_partial_r11.begin()) &&
              std::equal(r11_words.begin() + W, r11_words.end(), r11_seed_words.begin() + W),
          "second packed R11 partial result changes only its written row and remains published");
    r11_target = prosper::test::find_persistent_color_target(
        r11_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    CHECK(r11_target && r11_target->image == r11_image &&
              r11_target->pin_count == r11_renderer_pins &&
              r11_target->layout == r11_layout,
          "repeat packed R11 dispatch preserves allocation, layout, and released leases");

    const auto r11_failure_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    CHECK(!prosper::frontend::execute_live_compute_items({r11_partial}),
          "failed packed R11 partial completion is reported");
    const auto r11_failure_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(r11_failure_after.r11_source_seed_recorded ==
                  r11_failure_before.r11_source_seed_recorded + 1 &&
              r11_failure_after.borrowed == r11_failure_before.borrowed + 1 &&
              r11_failure_after.recorded == r11_failure_before.recorded + 1 &&
              r11_failure_after.failed == r11_failure_before.failed + 1 &&
              r11_failure_after.published == r11_failure_before.published &&
              !import_live_render_target_image(r11_address, source_request, r11_source),
          "failed packed R11 completion revokes renderer authority");
    r11_target = prosper::test::find_persistent_color_target(
        r11_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32, false);
    CHECK(r11_target && r11_target->image == r11_image &&
              r11_target->pin_count == r11_renderer_pins &&
              r11_target->layout == r11_layout,
          "failed packed R11 completion releases both leases at the saved layout");

    // A tiled guest result produces canonical packed words in staging via either the direct
    // retile shader or the image-transfer plus buffer-retile route. Both feed the same renderer
    // destination copy; run this test once with each route selected by the CTest environment.
    constexpr uint32_t tiled_mode = uint32_t(TileMode::Sw64KbS);
    const size_t tiled_bytes = tiled_surface_bytes(W, H, tiled_mode, 0, sizeof(uint32_t));
    std::vector<uint8_t> tiled_guest(tiled_bytes, 0xcd);
    const uint64_t tiled_address = reinterpret_cast<uint64_t>(tiled_guest.data());
    DrawItem tiled_producer = r11_producer;
    tiled_producer.color0_base = tiled_address;
    CHECK(!render(tiled_producer).empty(), "tiled R11 producer creates an exact renderer seed");
    std::vector<uint8_t> tiled_seed;
    CHECK(prosper::test::readback_persistent_color_target(
              tiled_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
              tiled_seed, r11_readback_error) && tiled_seed.size() == W * H * sizeof(uint32_t),
          "tiled R11 renderer seed has exact packed words");
    const auto* tiled_target = prosper::test::find_persistent_color_target(
        tiled_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    const VkImage tiled_image = tiled_target ? tiled_target->image : VK_NULL_HANDLE;
    const VkImageLayout tiled_layout = tiled_target ? tiled_target->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    const uint32_t tiled_renderer_pins = tiled_target ? tiled_target->pin_count : 0;
    ShaderResource tiled_output = r11_output;
    tiled_output.gpu_addr = tiled_address;
    tiled_output.size = static_cast<uint32_t>(tiled_bytes);
    tiled_output.tile_mode = tiled_mode;
    ShaderResourceTable tiled_table;
    tiled_table.resources.push_back(tiled_output);
    const auto tiled_spirv = recompile_compute(
        store_black, std::size(store_black), &tiled_table, r11_config);
    CHECK(!tiled_spirv.empty(), "tiled packed R11 partial writer recompiles");
    ComputeItem tiled_partial = r11_partial;
    tiled_partial.spirv = tiled_spirv;
    tiled_partial.resources = std::make_shared<ShaderResourceTable>(tiled_table);
    tiled_partial.code_addr = 0x37310013u;
    const auto tiled_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    const auto direct_before = prosper::frontend::gpu_direct_retile_recordings().load();
    CHECK(!tiled_spirv.empty() && prosper::frontend::execute_live_compute_items({tiled_partial}),
          "tiled packed R11 partial writer completes from renderer seed");
    const auto tiled_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    const bool direct_expected = std::getenv("PROSPER_NO_DIRECT_IMAGE_RETILE") == nullptr;
    CHECK(tiled_after.r11_source_seed_recorded == tiled_before.r11_source_seed_recorded + 1 &&
              tiled_after.recorded == tiled_before.recorded + 1 &&
              tiled_after.published == tiled_before.published + 1 &&
              prosper::frontend::gpu_direct_retile_recordings().load() ==
                  direct_before + (direct_expected ? 1u : 0u),
          "tiled packed R11 publishes through the selected staging producer");
    std::vector<uint8_t> tiled_linear(W * H * sizeof(uint32_t), 0);
    detile_surface(tiled_linear.data(), tiled_guest.data(), W, H, tiled_mode, 0,
                   sizeof(uint32_t));
    CHECK(tiled_seed.size() == tiled_linear.size() &&
              std::equal(tiled_linear.begin() + W * sizeof(uint32_t), tiled_linear.end(),
                         tiled_seed.begin() + W * sizeof(uint32_t)) &&
              !std::equal(tiled_linear.begin(), tiled_linear.begin() + W * sizeof(uint32_t),
                          tiled_seed.begin()),
          "tiled R11 guest writeback changes row zero and preserves renderer-seeded rows");
    std::vector<uint8_t> tiled_mirrored;
    CHECK(prosper::test::readback_persistent_color_target(
              tiled_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
              tiled_mirrored, r11_readback_error) && tiled_mirrored == tiled_linear,
          "tiled R11 renderer destination equals detiled guest result bit for bit");
    tiled_target = prosper::test::find_persistent_color_target(
        tiled_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    CHECK(tiled_target && tiled_target->image == tiled_image &&
              tiled_target->layout == tiled_layout &&
              tiled_target->pin_count == tiled_renderer_pins,
          "tiled R11 destination retains image and layout and releases compute leases");

    // A different binding importing the same renderer image must keep the destination lease
    // collision closed. The storage binding's own packed seed is allowed, but this sampled import
    // is a second reader whose ordering and layout cannot be justified by the self-seed exception.
    std::vector<uint32_t> collision_words(W * H, 0xdeadbeefu);
    const uint64_t collision_address = reinterpret_cast<uint64_t>(collision_words.data());
    DrawItem collision_producer = r11_producer;
    collision_producer.color0_base = collision_address;
    CHECK(!render(collision_producer).empty(),
          "packed R11 collision producer creates a renderer image");
    const auto* collision_target = prosper::test::find_persistent_color_target(
        collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32);
    const VkImage collision_image = collision_target ? collision_target->image : VK_NULL_HANDLE;
    const VkImageLayout collision_layout = collision_target
        ? collision_target->layout : VK_IMAGE_LAYOUT_UNDEFINED;
    const uint32_t collision_renderer_pins = collision_target ? collision_target->pin_count : 0;
    std::vector<uint8_t> collision_seed;
    CHECK(prosper::test::readback_persistent_color_target(
              collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
              collision_seed, r11_readback_error) &&
              collision_seed.size() == W * H * sizeof(uint32_t),
          "packed R11 collision source has readable renderer pixels");
    LiveTargetImageImport collision_destination;
    CHECK(borrow_live_render_target_image_destination(
              collision_address, {W, H, LiveTargetPixelFormat::R11G11B10Float},
              collision_destination) && collision_destination.valid() &&
              collision_destination.image == collision_image,
          "packed R11 collision destination itself remains borrowable");
    release_live_render_target_image(collision_address);
    ShaderResource collision_sampled = r11_output;
    collision_sampled.cls = ResourceClass::Texture;
    collision_sampled.binding = 4;
    collision_sampled.sgpr_base = 0;
    collision_sampled.gpu_addr = collision_address;
    ShaderResource collision_output = r11_output;
    collision_output.gpu_addr = collision_address;
    ShaderResourceTable collision_table;
    collision_table.resources = {collision_sampled, collision_output};
    static const uint32_t image_copy_r11[] = {
        0x7E080300u, 0x7E0A0280u, // v4=x, v5=0
        0xF0000F08u, 0x00000004u, 0xBF8C3F70u, // load sampled binding 4
        0xF0200F08u, 0x00020004u, 0xBF810000u, // store binding 5
    };
    const auto collision_spirv = recompile_compute(
        image_copy_r11, std::size(image_copy_r11), &collision_table, r11_config);
    CHECK(!collision_spirv.empty(), "packed R11 sampled/storage collision recompiles");
    ComputeItem collision_item = r11_partial;
    collision_item.spirv = collision_spirv;
    collision_item.resources = std::make_shared<ShaderResourceTable>(collision_table);
    collision_item.code_addr = 0x37310015u;
    const auto collision_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!collision_spirv.empty() && prosper::frontend::execute_live_compute_items({collision_item}),
          "packed R11 sampled/storage collision completes through fallback");
    const auto collision_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(collision_after.r11_source_seed_recorded ==
                  collision_before.r11_source_seed_recorded + 1 &&
              collision_after.candidates == collision_before.candidates + 1 &&
              collision_after.borrowed == collision_before.borrowed &&
              collision_after.recorded == collision_before.recorded &&
              collision_after.published == collision_before.published,
          "second imported R11 binding rejects an otherwise borrowable destination");
    CHECK(collision_seed.size() == W * H * sizeof(uint32_t) &&
              std::memcmp(collision_words.data(), collision_seed.data(),
                          collision_seed.size()) == 0,
          "sampled R11 binding uses renderer pixels while guest fallback writes exact words");
    CHECK(!import_live_render_target_image(collision_address, source_request, r11_source),
          "rejected R11 destination does not restore strict renderer authority");
    collision_target = prosper::test::find_persistent_color_target(
        collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32, false);
    CHECK(collision_target && collision_target->image == collision_image &&
              collision_target->layout == collision_layout &&
              collision_target->pin_count == collision_renderer_pins,
          "rejected R11 destination releases imports and preserves image layout");
    return fails ? 1 : 0;
}

int main() {
    return run_destination_mirror_regression();
}
