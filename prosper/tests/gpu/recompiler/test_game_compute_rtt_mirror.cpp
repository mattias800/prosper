// Real-device regression for destination-only renderer RTT publication.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/live/live_renderer.hpp"
#include "shared/live/gpu_retile.hpp"
#include "shared/live/unorm10_snapshot.hpp"
#include "shared/compute/storage_image_alias_plan.hpp"
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
    // 64 texels: every fixture's rows (RGBA8, R32 and R11 at 256 bytes, RGBA16F at 512) are already
    // multiples of GFX10's 256-byte linear pitch, so the guest buffers below ARE the hardware layout
    // of these images. An 8-wide image would occupy 256-byte rows in guest memory (#4586); its tight
    // W*H*bpp vectors would be overrun by the compute writeback.
    constexpr uint32_t W = 64, H = 4;
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
          "RGBA16F renderer repaints before the partial one-layer array write");
    std::vector<uint8_t> rgba16_array_seed;
    std::string rgba16_array_seed_error;
    const bool rgba16_array_seed_valid = prosper::test::readback_persistent_color_target(
        rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
        rgba16_array_seed, rgba16_array_seed_error) &&
        rgba16_array_seed.size() == rgba16_words.size() * sizeof(uint16_t);
    bool rgba16_array_seed_expected = rgba16_array_seed_valid;
    if (rgba16_array_seed_valid) {
        for (size_t i = 0; i < rgba16_words.size(); ++i) {
            uint16_t pixel;
            std::memcpy(&pixel, rgba16_array_seed.data() + i * sizeof(pixel), sizeof(pixel));
            rgba16_array_seed_expected &= pixel == rgba16_sample[i % 4u] &&
                                         rgba16_words[i] == rgba16_black[i % 4u];
        }
    }
    CHECK(rgba16_array_seed_expected,
          "partial-array fixture has nonblack renderer pixels and stale black guest bytes");
    ComputeShaderConfig rgba16_array_partial_config = rgba16_config;
    rgba16_array_partial_config.local_y = 1;
    const auto rgba16_array_partial_spirv = recompile_compute(
        store_black_array, std::size(store_black_array), &rgba16_table,
        rgba16_array_partial_config);
    CHECK(!rgba16_array_partial_spirv.empty(),
          "partial one-layer array writer recompiles");
    ComputeItem rgba16_array_partial = rgba16_array_full;
    rgba16_array_partial.spirv = rgba16_array_partial_spirv;
    rgba16_array_partial.launch.threads_y = 1;
    rgba16_array_partial.launch.local_y = 1;
    rgba16_array_partial.launch.groups_y = 1;
    rgba16_array_partial.code_addr = 0x37310024u;
    const auto rgba16_array_partial_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!rgba16_array_partial_spirv.empty() &&
              prosper::frontend::execute_live_compute_items({rgba16_array_partial}),
          "partial one-layer array writer completes");
    const auto rgba16_array_partial_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    const size_t rgba16_row_bytes = W * 4u * sizeof(uint16_t);
    bool rgba16_array_first_row_black = true;
    for (size_t i = 0; i < W * 4u; ++i)
        rgba16_array_first_row_black &= rgba16_words[i] == rgba16_black[i % 4u];
    CHECK(rgba16_array_seed_valid &&
              rgba16_array_first_row_black &&
              std::memcmp(reinterpret_cast<const uint8_t*>(rgba16_words.data()) + rgba16_row_bytes,
                          rgba16_array_seed.data() + rgba16_row_bytes,
                          rgba16_array_seed.size() - rgba16_row_bytes) == 0,
          "partial array writer preserves untouched renderer rows in guest writeback");
    if (!rgba16_mirror_disabled) {
        CHECK(rgba16_array_partial_after.rgba16_source_seed_recorded ==
                  rgba16_array_partial_before.rgba16_source_seed_recorded + 1 &&
                  rgba16_array_partial_after.published ==
                      rgba16_array_partial_before.published + 1,
              "partial array writer seeds from and publishes to the renderer image");
        std::vector<uint8_t> rgba16_array_partial_pixels;
        CHECK(prosper::test::readback_persistent_color_target(
                  rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                  rgba16_array_partial_pixels, rgba16_array_seed_error) &&
                  rgba16_array_partial_pixels.size() == rgba16_array_seed.size() &&
                  std::memcmp(rgba16_array_partial_pixels.data(), rgba16_words.data(),
                              rgba16_array_partial_pixels.size()) == 0,
              "partial array renderer pixels equal exact guest writeback");
    }

    // A single guest image may be named by both 2D and 2D_ARRAY storage
    // declarations. Reusing either sibling's VkImageView for the other is invalid;
    // making two writable images would lose one store at whole-image writeback.
    // The separate-address control proves this shader otherwise executes.
    static const uint32_t mixed_array_store[] = {
        0x7E080300u, 0x7E0A0301u, 0x7E0C0280u,
        0x7E000280u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF0200F28u, 0x00020004u, // 2D_ARRAY store through s[8:15]
        0xF0200F08u, 0x00040004u, // ordinary 2D store through s[16:23]
        0xBF810000u,
    };
    std::vector<uint16_t> distinct_words(rgba16_words.size(), 0x3555u);
    for (bool reverse : {false, true}) {
        std::fill(rgba16_words.begin(), rgba16_words.end(), 0x3555u);
        std::fill(distinct_words.begin(), distinct_words.end(), 0x3555u);
        CHECK(!render(rgba16_producer).empty(),
              "mixed-view positive control repaints the first renderer destination");
        ShaderResourceTable mixed_table = rgba16_table;
        ShaderResource sibling = rgba16_output;
        sibling.binding = 6;
        sibling.sgpr_base = reverse ? 8 : 16;
        sibling.gpu_addr = reinterpret_cast<uint64_t>(distinct_words.data());
        if (reverse) mixed_table.resources[0].sgpr_base = 16;
        mixed_table.resources.push_back(sibling);
        ComputeShaderConfig mixed_config = rgba16_config;
        mixed_config.user_sgprs.resize(24);
        const auto mixed_spirv = recompile_compute(
            mixed_array_store, std::size(mixed_array_store), &mixed_table, mixed_config);
        const auto mixed_report = validate_spirv_descriptor_interface(
            mixed_spirv, &mixed_table, 0, SpirvShaderStage::Compute, false);
        const auto* first = find_spirv_descriptor_binding(mixed_report, 0, 5);
        const auto* second = find_spirv_descriptor_binding(mixed_report, 0, 6);
        CHECK(!mixed_spirv.empty() && mixed_report.ok() && first && second &&
                  first->kind == SpirvDescriptorKind::StorageImage &&
                  second->kind == SpirvDescriptorKind::StorageImage &&
                  first->image_arrayed != second->image_arrayed &&
                  first->writable && second->writable,
              "mixed-view fixture reflects two distinct writable Vulkan view types");
        if (mixed_spirv.empty() || !mixed_report.ok() || !first || !second) continue;
        ComputeItem mixed_item = rgba16_array_full;
        mixed_item.spirv = mixed_spirv;
        mixed_item.resources = std::make_shared<ShaderResourceTable>(mixed_table);
        mixed_item.code_addr = reverse ? 0x37310026u : 0x37310025u;
        CHECK(prosper::frontend::execute_live_compute_items({mixed_item}),
              "mixed-view shader executes when its two outputs have distinct guest addresses");
        bool separate_outputs_black = true;
        for (size_t i = 0; i < distinct_words.size(); ++i)
            separate_outputs_black &= rgba16_words[i] == rgba16_black[i % 4u] &&
                                      distinct_words[i] == rgba16_black[i % 4u];
        std::vector<uint8_t> mixed_renderer_pixels;
        std::string mixed_renderer_error;
        const bool first_renderer_black = prosper::test::readback_persistent_color_target(
            rgba16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
            mixed_renderer_pixels, mixed_renderer_error) &&
            mixed_renderer_pixels.size() == rgba16_words.size() * sizeof(uint16_t) &&
            std::memcmp(mixed_renderer_pixels.data(), rgba16_words.data(),
                        mixed_renderer_pixels.size()) == 0;
        CHECK(separate_outputs_black && (rgba16_mirror_disabled || first_renderer_black),
              "both distinct-view stores change their respective output images to black");
        mixed_table.resources[1].gpu_addr = rgba16_address;
        const auto mixed_plan = prosper::frontend::plan_storage_image_aliases(
            mixed_report.descriptors, mixed_table);
        CHECK(!mixed_plan.valid && mixed_plan.decline_reason &&
                  std::strcmp(mixed_plan.decline_reason,
                              "storage-alias-mixed-array-view") == 0,
              "same-address mixed storage views name the intended early decline");
        const auto before_mixed_decline = rgba16_words;
        mixed_item.resources = std::make_shared<ShaderResourceTable>(mixed_table);
        CHECK(!prosper::frontend::execute_live_compute_items({mixed_item}) &&
                  rgba16_words == before_mixed_decline,
              "same-address mixed-view shader declines without changing guest pixels");
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
    // This is Astro's hot shape: compute writes the packed image before any graphics pass
    // registers the address. A full overwrite may reserve an image, but an ordinary borrow or
    // failed completion must never make its uninitialized pixels readable.
    std::vector<uint32_t> first_compute_r11(W * H, 0xdeadbeefu);
    const uint64_t first_compute_address =
        reinterpret_cast<uint64_t>(first_compute_r11.data());
    ShaderResourceTable first_compute_table = r11_table;
    first_compute_table.resources[0].gpu_addr = first_compute_address;
    ComputeItem first_compute_full = r11_full;
    first_compute_full.resources =
        std::make_shared<ShaderResourceTable>(first_compute_table);
    first_compute_full.code_addr = 0x37310018u;
    LiveTargetImageImport first_compute_borrow;
    CHECK(!borrow_live_render_target_image_destination(
              first_compute_address,
              {W, H, LiveTargetPixelFormat::R11G11B10Float}, first_compute_borrow) &&
              first_compute_borrow.refusal == LiveTargetImageImport::Refusal::NoRttEntry,
          "unregistered packed R11 address rejects an ordinary destination borrow");
    const auto first_compute_failed_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    CHECK(!prosper::frontend::execute_live_compute_items({first_compute_full}),
          "unregistered packed R11 destination reports failed completion");
    const auto first_compute_failed_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(first_compute_failed_after.borrowed == first_compute_failed_before.borrowed + 1 &&
              first_compute_failed_after.failed == first_compute_failed_before.failed + 1 &&
              first_compute_failed_after.published == first_compute_failed_before.published &&
              !import_live_render_target_image(first_compute_address, source_request,
                                               first_compute_borrow) &&
              std::all_of(first_compute_r11.begin(), first_compute_r11.end(),
                          [](uint32_t word) { return word == 0xdeadbeefu; }),
          "failed first producer grants no authority and leaves guest bytes untouched");
    const auto first_compute_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(prosper::frontend::execute_live_compute_items({first_compute_full}),
          "unregistered packed R11 full overwrite completes");
    const auto first_compute_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(first_compute_after.borrowed == first_compute_before.borrowed + 1 &&
              first_compute_after.published == first_compute_before.published + 1 &&
              import_live_render_target_image(first_compute_address, source_request,
                                              first_compute_borrow),
          "completed first producer publishes a readable packed renderer image");
    if (first_compute_borrow.valid()) release_live_render_target_image(first_compute_address);
    std::vector<uint8_t> first_compute_pixels;
    std::string first_compute_error;
    CHECK(prosper::test::readback_persistent_color_target(
              first_compute_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
              first_compute_pixels, first_compute_error) &&
              first_compute_pixels.size() == first_compute_r11.size() * sizeof(uint32_t) &&
              std::memcmp(first_compute_pixels.data(), first_compute_r11.data(),
                          first_compute_pixels.size()) == 0,
          "first compute image equals completed packed guest words bit for bit");
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

    // The sampled binding reads the old renderer image during the dispatch. The separate output
    // lease may publish into that same image only after the shader and readback staging are done.
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
        0x100000FFu, 0x3F000000u, // red = sampled red * 0.5, unlike the old image
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
          "packed R11 sampled/storage alias completes");
    const auto collision_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    const bool read_alias_disabled = std::getenv("PROSPER_NO_COMPUTE_RTT_READ_ALIAS") != nullptr;
    CHECK(collision_after.r11_source_seed_recorded ==
                  collision_before.r11_source_seed_recorded + 1 &&
              collision_after.candidates == collision_before.candidates + 1 &&
              collision_after.borrowed == collision_before.borrowed + !read_alias_disabled &&
              collision_after.recorded == collision_before.recorded + !read_alias_disabled &&
              collision_after.published == collision_before.published + !read_alias_disabled,
          "read-only R11 alias mirrors after shader use; explicit control declines it");
    std::vector<uint32_t> collision_expected(W * H);
    if (collision_seed.size() == collision_expected.size() * sizeof(uint32_t))
        std::memcpy(collision_expected.data(), collision_seed.data(), collision_seed.size());
    CHECK(std::all_of(collision_expected.begin(), collision_expected.begin() + W,
                      [](uint32_t word) { return (word & 0x7ffu) == 0x3c0u; }),
          "R11 sampled seed has unit red before the shader halves it");
    for (size_t x = 0; x < W; ++x)
        collision_expected[x] = (collision_expected[x] & ~0x7ffu) | 0x380u;
    CHECK(collision_words == collision_expected,
          "R11 sampled old pixels produce a distinct exact guest result");
    const bool collision_importable = import_live_render_target_image(
        collision_address, source_request, r11_source);
    CHECK(collision_importable == !read_alias_disabled,
          "successful R11 alias publication restores strict renderer authority");
    if (collision_importable) release_live_render_target_image(collision_address);
    if (!read_alias_disabled) {
        std::vector<uint8_t> mirrored;
        CHECK(prosper::test::readback_persistent_color_target(
                  collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                  mirrored, r11_readback_error) &&
                  mirrored.size() == collision_words.size() * sizeof(uint32_t) &&
                  std::memcmp(mirrored.data(), collision_expected.data(), mirrored.size()) == 0,
              "read-only R11 alias copies the changed completed result into the renderer image");
    }
    collision_target = prosper::test::find_persistent_color_target(
        collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32, false);
    CHECK(collision_target && collision_target->image == collision_image &&
              collision_target->layout == collision_layout &&
              collision_target->pin_count == collision_renderer_pins,
          "R11 read alias releases all three leases and preserves the renderer layout");

    // Astro's resource order is the opposite: the output's seed is prepared before the sampled
    // alias transitions their shared renderer image. A fresh draw makes guest bytes stale again.
    CHECK(!render(collision_producer).empty(),
          "reverse-order R11 source is repainted independently of the prior guest result");
    std::vector<uint8_t> reverse_seed;
    CHECK(prosper::test::readback_persistent_color_target(
              collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
              reverse_seed, r11_readback_error) && reverse_seed == collision_seed,
          "reverse-order R11 arm starts from the same distinct old renderer pixels");
    ShaderResourceTable reverse_table;
    ShaderResource reverse_output = collision_output;
    ShaderResource reverse_sampled = collision_sampled;
    reverse_output.binding = 4;  // output is now the first reflected descriptor
    reverse_sampled.binding = 5; // shader SGPR bases still address their original resources
    reverse_table.resources = {reverse_output, reverse_sampled};
    const auto reverse_spirv = recompile_compute(
        image_copy_r11, std::size(image_copy_r11), &reverse_table, r11_config);
    const auto reverse_reflection = validate_spirv_descriptor_interface(
        reverse_spirv, &reverse_table, 0, SpirvShaderStage::Compute, false);
    const auto* reverse_write = find_spirv_descriptor_binding(reverse_reflection, 0, 4);
    const auto* reverse_read = find_spirv_descriptor_binding(reverse_reflection, 0, 5);
    CHECK(!reverse_spirv.empty() && reverse_reflection.ok() && reverse_write &&
              reverse_read && reverse_write->writable && !reverse_read->writable,
          "reverse-order R11 fixture reflects output before sampled input");
    ComputeItem reverse_item = collision_item;
    reverse_item.spirv = reverse_spirv;
    reverse_item.resources = std::make_shared<ShaderResourceTable>(reverse_table);
    reverse_item.code_addr = 0x3731001bu;
    const auto reverse_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!reverse_spirv.empty() &&
              prosper::frontend::execute_live_compute_items({reverse_item}),
          "reverse-order R11 alias dispatch completes");
    const auto reverse_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(reverse_after.r11_source_seed_recorded ==
                  reverse_before.r11_source_seed_recorded + 1 &&
              reverse_after.candidates == reverse_before.candidates + 1 &&
              reverse_after.borrowed == reverse_before.borrowed + !read_alias_disabled &&
              reverse_after.recorded == reverse_before.recorded + !read_alias_disabled &&
              reverse_after.published == reverse_before.published + !read_alias_disabled,
          "output-before-sampled R11 alias obeys the same controlled admission");
    CHECK(collision_words == collision_expected,
          "reverse-order alias reads repainted old pixels and writes the distinct expected result");
    if (!read_alias_disabled) {
        std::vector<uint8_t> mirrored;
        CHECK(prosper::test::readback_persistent_color_target(
                  collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                  mirrored, r11_readback_error) &&
                  mirrored.size() == collision_expected.size() * sizeof(uint32_t) &&
                  std::memcmp(mirrored.data(), collision_expected.data(), mirrored.size()) == 0,
              "reverse-order R11 mirror contains the changed output");
    }
    collision_target = prosper::test::find_persistent_color_target(
        collision_address, W, H, VK_FORMAT_B10G11R11_UFLOAT_PACK32, false);
    CHECK(collision_target && collision_target->image == collision_image &&
              collision_target->layout == collision_layout &&
              collision_target->pin_count == collision_renderer_pins,
          "reverse-order R11 alias restores the original renderer ownership");

    std::vector<uint16_t> alias16_words;
    if (!rgba16_mirror_disabled) {
        alias16_words.assign(W * H * 4, 0x7777u);
        const uint64_t alias16_address = reinterpret_cast<uint64_t>(alias16_words.data());
        DrawItem alias16_producer = rgba16_producer;
        alias16_producer.color0_base = alias16_address;
        CHECK(!render(alias16_producer).empty(),
              "RGBA16F read-alias fixture creates a renderer source");
        std::vector<uint8_t> alias16_seed;
        std::string alias16_error;
        CHECK(prosper::test::readback_persistent_color_target(
                  alias16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                  alias16_seed, alias16_error) &&
                  alias16_seed.size() == alias16_words.size() * sizeof(uint16_t),
              "RGBA16F read alias starts from known renderer pixels");
        std::vector<uint16_t> alias16_expected(alias16_words.size());
        if (alias16_seed.size() == alias16_expected.size() * sizeof(uint16_t))
            std::memcpy(alias16_expected.data(), alias16_seed.data(), alias16_seed.size());
        CHECK(std::all_of(alias16_expected.begin(), alias16_expected.begin() + W * 4,
                          [channel = size_t{0}](uint16_t word) mutable {
                              constexpr uint16_t expected[] = {0x4000u, 0xbc00u, 0x3800u, 0x3c00u};
                              return word == expected[(channel++) % 4];
                          }),
              "RGBA16F sampled seed has the expected nonuniform channel values");
        for (size_t x = 0; x < W; ++x) alias16_expected[x * 4] = 0x3c00u;
        const auto* alias16_target = prosper::test::find_persistent_color_target(
            alias16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT);
        const VkImage alias16_image = alias16_target ? alias16_target->image : VK_NULL_HANDLE;
        const VkImageLayout alias16_layout = alias16_target
            ? alias16_target->layout : VK_IMAGE_LAYOUT_UNDEFINED;
        const uint32_t alias16_pins = alias16_target ? alias16_target->pin_count : 0;
        ShaderResource alias16_sampled = rgba16_output;
        alias16_sampled.cls = ResourceClass::Texture;
        alias16_sampled.binding = 4;
        alias16_sampled.sgpr_base = 0;
        alias16_sampled.gpu_addr = alias16_address;
        ShaderResource alias16_output = rgba16_output;
        alias16_output.gpu_addr = alias16_address;
        alias16_output.binding = 4;
        alias16_sampled.binding = 5;
        ShaderResourceTable alias16_table;
        // Reflection sorts by descriptor binding, not resource-table insertion. Output-first is
        // the missing path: sampled-first uses the older seed_from_imported mirror, whereas Astro
        // prepares its standalone seed before importing the sampled binding.
        alias16_table.resources = {alias16_output, alias16_sampled};
        const auto alias16_spirv = recompile_compute(
            image_copy_r11, std::size(image_copy_r11), &alias16_table, rgba16_config);
        const auto alias16_reflection = validate_spirv_descriptor_interface(
            alias16_spirv, &alias16_table, 0, SpirvShaderStage::Compute, false);
        const auto* alias16_write = find_spirv_descriptor_binding(alias16_reflection, 0, 4);
        const auto* alias16_read = find_spirv_descriptor_binding(alias16_reflection, 0, 5);
        CHECK(!alias16_spirv.empty() && alias16_reflection.ok() && alias16_write &&
                  alias16_read && alias16_write->writable && !alias16_read->writable,
              "RGBA16F fixture reflects output before sampled input");
        ComputeItem alias16_item = rgba16_full;
        alias16_item.spirv = alias16_spirv;
        alias16_item.resources = std::make_shared<ShaderResourceTable>(alias16_table);
        alias16_item.launch.threads_y = alias16_item.launch.local_y = 1;
        alias16_item.code_addr = 0x3731001au;
        const auto alias16_before =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(!alias16_spirv.empty() &&
                  prosper::frontend::execute_live_compute_items({alias16_item}),
              "RGBA16F read alias executes with old sampled pixels and partial output");
        const auto alias16_after =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(alias16_after.rgba16_source_seed_recorded ==
                  alias16_before.rgba16_source_seed_recorded + 1 &&
                  alias16_after.candidates == alias16_before.candidates + 1 &&
                  alias16_after.borrowed == alias16_before.borrowed + !read_alias_disabled &&
                  alias16_after.recorded == alias16_before.recorded + !read_alias_disabled &&
                  alias16_after.published == alias16_before.published + !read_alias_disabled,
              "RGBA16F read alias mirrors after shader use; control keeps CPU authority");
        CHECK(alias16_words == alias16_expected,
              "RGBA16F sampled old pixels produce a distinct exact guest result");
        LiveTargetImageImport alias16_import;
        const bool alias16_gpu_valid = import_live_render_target_image(
            alias16_address, source_request, alias16_import);
        CHECK(alias16_gpu_valid == !read_alias_disabled,
              "RGBA16F read alias publishes strict GPU authority only when enabled");
        if (alias16_gpu_valid) release_live_render_target_image(alias16_address);
        if (!read_alias_disabled) {
            std::vector<uint8_t> mirrored;
            CHECK(prosper::test::readback_persistent_color_target(
                      alias16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                      mirrored, alias16_error) &&
                      mirrored.size() == alias16_expected.size() * sizeof(uint16_t) &&
                      std::memcmp(mirrored.data(), alias16_expected.data(), mirrored.size()) == 0,
                  "RGBA16F mirror contains the changed sampled result");
        }
        alias16_target = prosper::test::find_persistent_color_target(
            alias16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, false);
        // This is the renderer owner's bookkeeping, not a query of VkImage's actual layout.
        // The preceding readback is also exercised under strict Vulkan validation, which checks
        // the compute submit's final GENERAL -> saved-layout transition before that readback.
        CHECK(alias16_target && alias16_target->image == alias16_image &&
                  alias16_target->layout == alias16_layout &&
                  alias16_target->pin_count == alias16_pins,
              "RGBA16F read alias retains renderer image ownership and layout bookkeeping");

        // Astro binds the same sampled renderer image twice after its writable destination.
        // Both descriptors acquire pins, but the second is folded onto the first Vulkan view.
        // The result copy follows the sampled dispatch and must restore GPU authority.
        CHECK(!render(alias16_producer).empty(),
              "folded read-alias fixture restores known renderer pixels");
        ShaderResource folded_sampled = alias16_sampled;
        folded_sampled.binding = 6;
        folded_sampled.sgpr_base = 16;
        ShaderResourceTable folded_table;
        folded_table.resources = {alias16_output, alias16_sampled, folded_sampled};
        static const uint32_t two_sampled_reads[] = {
            0x7E080300u, 0x7E0A0280u,                    // v4=x, v5=0
            0xF0000F08u, 0x00000004u, 0xBF8C3F70u,      // first sampled descriptor
            0xF0000F08u, 0x00040004u, 0xBF8C3F70u,      // duplicate sampled descriptor
            0x100000FFu, 0x3F000000u,                    // red *= 0.5
            0xF0200F08u, 0x00020004u, 0xBF810000u,      // store to renderer image
        };
        ComputeShaderConfig folded_config = rgba16_config;
        folded_config.user_sgprs.resize(24);
        const auto folded_spirv = recompile_compute(
            two_sampled_reads, std::size(two_sampled_reads), &folded_table, folded_config);
        const auto folded_reflection = validate_spirv_descriptor_interface(
            folded_spirv, &folded_table, 0, SpirvShaderStage::Compute, false);
        const auto* folded_write = find_spirv_descriptor_binding(folded_reflection, 0, 4);
        const auto* folded_read_a = find_spirv_descriptor_binding(folded_reflection, 0, 5);
        const auto* folded_read_b = find_spirv_descriptor_binding(folded_reflection, 0, 6);
        CHECK(!folded_spirv.empty() && folded_reflection.ok() && folded_write &&
                  folded_read_a && folded_read_b && folded_write->writable &&
                  !folded_read_a->writable && !folded_read_b->writable,
              "folded fixture retains both read-only sampled descriptors");
        ComputeItem folded_item = alias16_item;
        folded_item.spirv = folded_spirv;
        folded_item.resources = std::make_shared<ShaderResourceTable>(folded_table);
        folded_item.code_addr = 0x3731001bu;
        std::vector<uint8_t> folded_seed;
        CHECK(prosper::test::readback_persistent_color_target(
                  alias16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                  folded_seed, alias16_error) &&
                  folded_seed.size() == alias16_words.size() * sizeof(uint16_t),
              "folded read alias starts from completed renderer pixels");
        std::vector<uint16_t> folded_expected(alias16_words.size());
        if (folded_seed.size() == folded_expected.size() * sizeof(uint16_t))
            std::memcpy(folded_expected.data(), folded_seed.data(), folded_seed.size());
        for (size_t x = 0; x < W; ++x) folded_expected[x * 4] = 0x3c00u;
        const auto folded_before =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(!folded_spirv.empty() &&
                  prosper::frontend::execute_live_compute_items({folded_item}),
              "folded sampled aliases execute before destination copy");
        const auto folded_after =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        const bool folded_read_alias_disabled = read_alias_disabled ||
            std::getenv("PROSPER_NO_FOLDED_COMPUTE_RTT_READ_ALIAS") != nullptr;
        CHECK(folded_after.candidates == folded_before.candidates + 1 &&
                  folded_after.borrowed == folded_before.borrowed + !folded_read_alias_disabled &&
                  folded_after.recorded == folded_before.recorded + !folded_read_alias_disabled &&
                  folded_after.published == folded_before.published + !folded_read_alias_disabled,
              "folded sampled aliases publish GPU authority after the sampled dispatch");
        CHECK(alias16_words == folded_expected,
              "folded sampled aliases preserve the exact guest writeback");
        LiveTargetImageImport folded_import;
        const bool folded_gpu_valid = import_live_render_target_image(
            alias16_address, source_request, folded_import);
        CHECK(folded_gpu_valid == !folded_read_alias_disabled,
              "folded sampled aliases retain strict GPU authority only when enabled");
        if (folded_gpu_valid) release_live_render_target_image(alias16_address);
        if (!folded_read_alias_disabled) {
            std::vector<uint8_t> folded_pixels;
            CHECK(prosper::test::readback_persistent_color_target(
                      alias16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                      folded_pixels, alias16_error) &&
                      folded_pixels.size() == folded_seed.size() &&
                      std::memcmp(folded_pixels.data(), folded_expected.data(),
                                  folded_pixels.size()) == 0,
                  "folded sampled aliases mirror their completed result exactly");
        }
        alias16_target = prosper::test::find_persistent_color_target(
            alias16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, false);
        CHECK(alias16_target && alias16_target->image == alias16_image &&
                  alias16_target->layout == alias16_layout &&
                  alias16_target->pin_count == alias16_pins,
              "folded sampled aliases release every renderer pin");
    }

    // A compute-owned output can reach the renderer first as CPU pixels, without any graphics
    // pass having allocated its image. A full overwrite must be able to create that destination;
    // failure must leave it unreadable, and only completed guest writeback may publish it.
    std::vector<uint8_t> cold_guest(W * H * 4, 0x39);
    const uint64_t cold_address = reinterpret_cast<uint64_t>(cold_guest.data());
    auto cold_cpu = std::make_shared<std::vector<uint8_t>>(cold_guest);
    notify_live_render_target_image_written(
        {cold_address, W, H, LiveTargetPixelFormat::Rgba8Unorm, std::move(cold_cpu)});
    CHECK(!prosper::test::find_persistent_color_target(
              cold_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false),
          "CPU-only RTT has no renderer image before compute destination admission");
    LiveTargetImageImport cold_borrow;
    CHECK(!borrow_live_render_target_image_destination(
              cold_address, {W, H, LiveTargetPixelFormat::Rgba8Unorm}, cold_borrow) &&
              cold_borrow.refusal == LiveTargetImageImport::Refusal::NoPersistentImage,
          "ordinary destination request cannot create an image implicitly");
    ShaderResourceTable cold_table = table;
    cold_table.resources[0].gpu_addr = cold_address;
    ComputeItem cold_item = item;
    cold_item.resources = std::make_shared<ShaderResourceTable>(cold_table);
    cold_item.code_addr = 0x37310016u;
    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    const auto cold_failed_before =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(!prosper::frontend::execute_live_compute_items({cold_item}),
          "newly allocated destination reports failed compute completion");
    const auto cold_failed_after =
        prosper::frontend::live_compute_rtt_destination_mirror_counters();
    const auto* cold_target = prosper::test::find_persistent_color_target(
        cold_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false);
    CHECK(cold_failed_after.recorded == cold_failed_before.recorded + 1 &&
              cold_failed_after.published == cold_failed_before.published &&
              cold_target && cold_target->image && !cold_target->valid &&
              cold_target->layout == VK_IMAGE_LAYOUT_UNDEFINED && !cold_target->pin_count &&
              !import_live_render_target_image(cold_address, source_request, source_import),
          "failed first overwrite retains no GPU read authority or lease");
    CHECK(cold_target && cold_target->compute_overwrite_uninitialized,
          "only the compute allocator tags its uninitialized destination for retry");
    if (cold_target) {
        auto* mutable_target = prosper::test::find_persistent_color_target(
            cold_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false);
        mutable_target->compute_overwrite_uninitialized = false;
        LiveTargetImageImport unrelated_undefined;
        CHECK(!borrow_live_render_target_image_destination(
                  cold_address, {W, H, LiveTargetPixelFormat::Rgba8Unorm, true,
                                 static_cast<void*>(prosper::test::render_vk_ctx().dev)},
                  unrelated_undefined) &&
                  unrelated_undefined.refusal ==
                      LiveTargetImageImport::Refusal::NoPersistentImage,
              "creation permission does not borrow an unrelated undefined-layout image");
        mutable_target->compute_overwrite_uninitialized = true;
    }
    const VkImage cold_image = cold_target ? cold_target->image : VK_NULL_HANDLE;
    const auto cold_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    CHECK(prosper::frontend::execute_live_compute_items({cold_item}),
          "retry overwrites the reserved renderer image from UNDEFINED");
    const auto cold_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    cold_target = prosper::test::find_persistent_color_target(
        cold_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false);
    CHECK(cold_after.recorded == cold_before.recorded + 1 &&
              cold_after.published == cold_before.published + 1 &&
              cold_target && cold_target->image == cold_image && cold_target->valid &&
              cold_target->layout == VK_IMAGE_LAYOUT_GENERAL && !cold_target->pin_count &&
              import_live_render_target_image(cold_address, source_request, source_import),
          "completed first overwrite publishes the exact device image and releases its lease");
    if (source_import.valid()) release_live_render_target_image(cold_address);
    std::vector<uint8_t> cold_pixels;
    std::string cold_error;
    CHECK(prosper::test::readback_persistent_color_target(
              cold_address, W, H, VK_FORMAT_R8G8B8A8_UNORM,
              cold_pixels, cold_error) && cold_pixels == cold_guest,
          "new renderer image has the same completed texels as guest writeback");

    std::vector<uint8_t> budget_guest(W * H * 4, 0x62);
    const uint64_t budget_address = reinterpret_cast<uint64_t>(budget_guest.data());
    auto budget_cpu = std::make_shared<std::vector<uint8_t>>(budget_guest);
    notify_live_render_target_image_written(
        {budget_address, W, H, LiveTargetPixelFormat::Rgba8Unorm, std::move(budget_cpu)});
    ShaderResourceTable budget_table = table;
    budget_table.resources[0].gpu_addr = budget_address;
    ComputeItem budget_item = item;
    budget_item.resources = std::make_shared<ShaderResourceTable>(budget_table);
    budget_item.code_addr = 0x37310017u;
    LiveTargetImageImport wrong_device_borrow;
    CHECK(!borrow_live_render_target_image_destination(
              budget_address, {W, H, LiveTargetPixelFormat::Rgba8Unorm, true,
                               reinterpret_cast<void*>(uintptr_t{1})},
              wrong_device_borrow) &&
              wrong_device_borrow.refusal == LiveTargetImageImport::Refusal::DeviceMismatch &&
              !prosper::test::find_persistent_color_target(
                  budget_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false),
          "separate compute device refuses before allocating renderer residency");
    VkDeviceSize& resident_bytes = prosper::test::persistent_color_target_bytes();
    const VkDeviceSize actual_resident_bytes = resident_bytes;
    resident_bytes = prosper::test::persistent_color_target_limit();
    // Creation at a bound now evicts under the graphics admission predicate (#3873), which would
    // make room here by evicting this test's real targets against the forced byte gauge. A pending
    // graphics batch is the state in which refusal remains the contract, so hold one: this arm is
    // about what compute does AFTER a refusal. Recorded, never submitted, discarded below.
    prosper::test::BackendSubmissionBatch budget_pending_batch;
    budget_pending_batch.enqueue(VK_NULL_HANDLE);
    LiveTargetImageImport budget_borrow;
    CHECK(!borrow_live_render_target_image_destination(
              budget_address, {W, H, LiveTargetPixelFormat::Rgba8Unorm, true,
                               static_cast<void*>(prosper::test::render_vk_ctx().dev)},
              budget_borrow) &&
              budget_borrow.refusal ==
                  LiveTargetImageImport::Refusal::DestinationCreationRefused,
          "budget refusal identifies image creation as the fallback gate");
    const auto budget_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    const bool budget_completed = prosper::frontend::execute_live_compute_items({budget_item});
    const auto budget_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
    budget_pending_batch.discard();
    resident_bytes = actual_resident_bytes;
    CHECK(budget_completed, "residency refusal retains successful compute execution");
    CHECK(std::equal(budget_guest.begin(), budget_guest.begin() + W * 4,
                     cold_guest.begin()) &&
              std::all_of(budget_guest.begin() + W * 4, budget_guest.end(),
                          [](uint8_t value) { return value == 0x62; }),
          "residency refusal writes the covered row and preserves untouched guest rows");
    CHECK(budget_after.candidates == budget_before.candidates + 1,
          "residency refusal still reaches the destination admission check");
    CHECK(budget_after.borrowed == budget_before.borrowed,
          "residency refusal does not borrow a destination image");
    CHECK(!prosper::test::find_persistent_color_target(
              budget_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false),
          "residency refusal leaves no renderer image");
    LiveTargetSnapshot budget_snapshot;
    CHECK(read_live_render_target(budget_address, budget_snapshot) &&
              budget_snapshot.pixels &&
              *budget_snapshot.pixels == budget_guest,
          "residency refusal publishes the exact final guest bytes as a CPU RTT");

    // Count admission is independent of the byte gauge. Occupy only this test's cache keys so
    // reverting to deferred-graphics headroom would incorrectly admit the next compute image.
    std::vector<uint8_t> count_guest(W * H * 4, 0x42);
    const uint64_t count_address = reinterpret_cast<uint64_t>(count_guest.data());
    notify_live_render_target_image_written(
        {count_address, W, H, LiveTargetPixelFormat::Rgba8Unorm,
         std::make_shared<std::vector<uint8_t>>(count_guest)});
    auto& color_cache = prosper::test::persistent_color_target_cache();
    std::vector<prosper::test::PersistentColorTargetKey> count_keys;
    const size_t count_limit = prosper::test::persistent_color_target_count_limit();
    CHECK(resident_bytes < prosper::test::persistent_color_target_limit() / 2,
          "count refusal fixture has independent renderer byte-budget headroom");
    CHECK(count_limit <= 4096, "focused count-bound fixture needs a bounded configured limit");
    if (count_limit <= 4096) {
        for (uint64_t i = 0; color_cache.size() < count_limit; ++i) {
            const prosper::test::PersistentColorTargetKey key{
                0xf000000000000000ull + i, W, H, VK_FORMAT_R8G8B8A8_UNORM};
            if (color_cache.try_emplace(key).second) count_keys.push_back(key);
        }
        {
            // While a graphics batch is pending nothing may be evicted, so the nominal count
            // (no deferred-graphics headroom) still refuses.
            prosper::test::BackendSubmissionBatch count_pending_batch;
            count_pending_batch.enqueue(VK_NULL_HANDLE);
            LiveTargetImageImport count_borrow;
            CHECK(!borrow_live_render_target_image_destination(
                      count_address, {W, H, LiveTargetPixelFormat::Rgba8Unorm, true,
                                      static_cast<void*>(prosper::test::render_vk_ctx().dev)},
                      count_borrow) &&
                      count_borrow.refusal ==
                          LiveTargetImageImport::Refusal::DestinationCreationRefused &&
                      !prosper::test::find_persistent_color_target(
                          count_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, false) &&
                      color_cache.size() == count_limit,
                  "compute-only destination obeys the nominal image count without batch headroom "
                  "while a graphics batch is pending, and evicts nothing");
            count_pending_batch.discard();
        }
        // With nothing pending, the same creation evicts the least recently used target -- one
        // of this fixture's never-used keys -- and stays within the nominal count (#3873).
        LiveTargetImageImport count_borrow;
        const bool count_created = borrow_live_render_target_image_destination(
            count_address, {W, H, LiveTargetPixelFormat::Rgba8Unorm, true,
                            static_cast<void*>(prosper::test::render_vk_ctx().dev)},
            count_borrow);
        size_t fixture_keys_left = 0;
        for (const auto& key : count_keys) fixture_keys_left += color_cache.count(key);
        CHECK(count_created && count_borrow.valid() &&
                  fixture_keys_left + 1 == count_keys.size() &&
                  color_cache.size() == count_limit,
              "with no batch pending, a compute destination at the count bound evicts one "
              "least-recently-used entry instead of refusing");
        if (count_created) release_live_render_target_image(count_address);
        for (const auto& key : count_keys) color_cache.erase(key);
    }

    if (!rgba16_mirror_disabled) {
        // This is the 4K Outer Wilds format: a compute-written RGBA16F surface with no prior
        // graphics allocation. The next partial dispatch must import the newly created image.
        std::vector<uint16_t> cold16_words(W * H * 4, 0x5555u);
        const uint64_t cold16_address = reinterpret_cast<uint64_t>(cold16_words.data());
        auto cold16_cpu = std::make_shared<std::vector<uint8_t>>(
            cold16_words.size() * sizeof(uint16_t), 0x55);
        notify_live_render_target_image_written(
            {cold16_address, W, H, LiveTargetPixelFormat::Rgba16Float,
             std::move(cold16_cpu)});
        ShaderResourceTable cold16_table = rgba16_table;
        cold16_table.resources[0].gpu_addr = cold16_address;
        ComputeItem cold16_full = rgba16_full;
        cold16_full.spirv = recompile_compute(
            store_red_r11, std::size(store_red_r11), &cold16_table, rgba16_config);
        CHECK(!cold16_full.spirv.empty(), "nonuniform cold RGBA16F producer recompiles");
        cold16_full.resources = std::make_shared<ShaderResourceTable>(cold16_table);
        cold16_full.code_addr = 0x37310018u;
        const auto cold16_before =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(prosper::frontend::execute_live_compute_items({cold16_full}),
              "cold RGBA16F storage output completes");
        const auto cold16_after =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        const auto* cold16_target = prosper::test::find_persistent_color_target(
            cold16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, false);
        CHECK(cold16_after.recorded == cold16_before.recorded + 1 &&
                  cold16_after.published == cold16_before.published + 1 &&
                  cold16_target && cold16_target->valid &&
                  cold16_target->layout == VK_IMAGE_LAYOUT_GENERAL &&
                  import_live_render_target_image(
                      cold16_address, source_request, rgba16_source),
              "cold RGBA16F result becomes a readable exact renderer image");
        if (rgba16_source.valid()) release_live_render_target_image(cold16_address);
        std::vector<uint16_t> cold16_expected(W * H * 4);
        for (size_t pixel = 0; pixel < W * H; ++pixel) {
            cold16_expected[pixel * 4] = 0x3c00u; // red=1 from the complete first dispatch
            cold16_expected[pixel * 4 + 3] = 0x3c00u;
        }
        CHECK(cold16_words == cold16_expected,
              "cold RGBA16F first write establishes distinct untouched rows");
        ComputeItem cold16_partial = cold16_full;
        cold16_partial.spirv = rgba16_partial_spirv;
        cold16_partial.launch.threads_y = cold16_partial.launch.local_y = 1;
        cold16_partial.code_addr = 0x37310019u;
        const auto cold16_seed_before =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(prosper::frontend::execute_live_compute_items({cold16_partial}),
              "partial RGBA16F writer consumes the newly created renderer image");
        const auto cold16_seed_after =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(cold16_seed_after.rgba16_source_seed_recorded ==
                  cold16_seed_before.rgba16_source_seed_recorded + 1 &&
                  cold16_seed_after.published == cold16_seed_before.published + 1,
              "cold RGBA16F target completes the GPU seed and destination publication loop");
        for (size_t x = 0; x < W; ++x) cold16_expected[x * 4] = 0;
        CHECK(cold16_words == cold16_expected,
              "cold RGBA16F partial write preserves the completed red rows below row zero");
        std::vector<uint8_t> cold16_pixels;
        std::string cold16_error;
        CHECK(prosper::test::readback_persistent_color_target(
                  cold16_address, W, H, VK_FORMAT_R16G16B16A16_SFLOAT,
                  cold16_pixels, cold16_error) &&
                  cold16_pixels.size() == cold16_expected.size() * sizeof(uint16_t) &&
                  std::memcmp(cold16_pixels.data(), cold16_expected.data(),
                              cold16_pixels.size()) == 0,
              "cold RGBA16F renderer image retains the exact nonuniform partial result");
    }
    {
        // #3929: a one-layer 2D_ARRAY T# (img_dim 5, depth 1) is Sonic Frontiers' post-process
        // output shape. Its single subresource is byte-identical to a plain 2D surface, so a full
        // overwrite must reach the renderer's device image instead of a CPU snapshot. This is a
        // first producer at a fresh address, so admission also exercises destination creation.
        const bool one_layer_array_disabled =
            std::getenv("PROSPER_NO_COMPUTE_RTT_ONE_LAYER_ARRAY_DEST") != nullptr;
        std::vector<uint8_t> array1(W * H * 4, 0x5a);
        const uint64_t array1_address = reinterpret_cast<uint64_t>(array1.data());
        ShaderResource array1_output = output;
        array1_output.img_dim = 5;
        array1_output.gpu_addr = array1_address;
        ShaderResourceTable array1_table;
        array1_table.resources.push_back(array1_output);
        ComputeShaderConfig array1_config = config;
        array1_config.local_y = H;  // every row: a full overwrite, not the one-row partial shape
        const auto array1_spirv = recompile_compute(
            store_black_array, std::size(store_black_array), &array1_table, array1_config);
        CHECK(!array1_spirv.empty(), "one-layer 2D_ARRAY RGBA8 storage writer recompiles");
        ComputeItem array1_item = item;
        array1_item.spirv = array1_spirv;
        array1_item.resources = std::make_shared<ShaderResourceTable>(array1_table);
        array1_item.code_addr = 0x3731002au;
        const auto array1_before =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(!array1_spirv.empty() &&
                  prosper::frontend::execute_live_compute_items({array1_item}),
              "one-layer 2D_ARRAY RGBA8 full overwrite completes");
        const auto array1_after =
            prosper::frontend::live_compute_rtt_destination_mirror_counters();
        bool array1_black = true;
        for (size_t i = 0; i < array1.size(); i += 4)
            array1_black &= array1[i] == 0 && array1[i + 1] == 0 && array1[i + 2] == 0 &&
                            array1[i + 3] == 255;
        if (!array1_black)
            std::printf("  one-layer array guest texel0=%u,%u,%u,%u last=%u,%u,%u,%u\n",
                        array1[0], array1[1], array1[2], array1[3], array1[array1.size() - 4],
                        array1[array1.size() - 3], array1[array1.size() - 2], array1.back());
        CHECK(array1_black, "one-layer 2D_ARRAY writer still writes back exact guest bytes");
        if (!one_layer_array_disabled) {
            CHECK(array1_after.candidates == array1_before.candidates + 1 &&
                      array1_after.borrowed == array1_before.borrowed + 1 &&
                      array1_after.recorded == array1_before.recorded + 1 &&
                      array1_after.published == array1_before.published + 1 &&
                      array1_after.failed == array1_before.failed,
                  "one-layer 2D_ARRAY result publishes through the renderer device mirror");
            std::vector<uint8_t> array1_pixels;
            std::string array1_error;
            CHECK(prosper::test::readback_persistent_color_target(
                      array1_address, W, H, VK_FORMAT_R8G8B8A8_UNORM, array1_pixels,
                      array1_error) &&
                      array1_pixels == array1,
                  "one-layer 2D_ARRAY mirror pixels equal the exact guest writeback");
        } else {
            CHECK(array1_after.candidates == array1_before.candidates &&
                      array1_after.published == array1_before.published,
                  "control: the 2D-only shape rule keeps one-layer arrays on CPU publication");
        }
    }
    // Packed R10G10B10A2 UNORM: graphics holds this guest colour format as an RGBA8 image, while
    // compute stores it natively as A2B10G10R10. The exact-result mirror converts on the GPU with the
    // CPU path's integer rounding (PackedRttConversion::record_packed10_to_rgba8) instead of unpacking
    // every texel on the CPU (publish_unorm10_as_rgba8). Before it these results were declined as
    // "format-unmapped". Every 10-bit value appears once per channel -- a DIFFERENT permutation in each
    // colour channel (R = t, G = t ^ 0x155, B = 1023 - t), so a channel swap cannot pass -- and every
    // alpha value in a row. Each output byte is compared with the CPU table: a float conversion (a
    // blit) rounds 48 of the 1,024 values one step low on RADV, which a single-value check cannot see.
    {
        constexpr uint32_t PW = 256, PH = 4;
        std::vector<uint8_t> p10_guest(size_t{PW} * PH * 4, 0x5a);
        const uint64_t p10_address = reinterpret_cast<uint64_t>(p10_guest.data());
        DrawItem p10_producer = producer;
        p10_producer.color0_base = p10_address;
        p10_producer.color0_width = PW;
        p10_producer.color0_height = PH;
        p10_producer.ps.color0_format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        CHECK(!render_submit_items({p10_producer}, PW, PH).empty(),
              "R10G10B10A2 producer materializes a renderer target");
        static const uint32_t store_p10[] = {
            0x7E080300u, 0x7E0A0301u,   // v4=x, v5=y
            0x340C0A88u,   // v6 = y << 8
            0x4A0C0906u,   // v6 = v6 + x: every value 0..1023 once
            0x3A0E0CFFu, 0x155u,   // v7 = v6 ^ 0x155
            0x4C100CFFu, 0x3FFu,   // v8 = 1023 - v6
            0x7E000D06u, 0x7E020D07u, 0x7E040D08u,   // v0, v1, v2 = float(v6, v7, v8)
            0x100000FFu, 0x3A802008u,   // v0 *= 1/1023
            0x100202FFu, 0x3A802008u,   // v1 *= 1/1023
            0x100404FFu, 0x3A802008u,   // v2 *= 1/1023
            0x7E060D05u,   // v3 = float(y)
            0x100606FFu, 0x3EAAAAABu,   // v3 *= 1/3: alpha 0..3 by row
            0xF0200F08u, 0x00020004u,   // image_store v[0:3] (dmask RGBA) at v4,v5 through s[8:15]
            0xBF810000u,
        };
        ShaderResource p10_output{};
        p10_output.cls = ResourceClass::StorageImage;
        p10_output.format = DataFormat::Unorm2_10_10_10;
        p10_output.num_components = 4;
        p10_output.binding = 5;
        p10_output.sgpr_base = 8;
        p10_output.img_dim = 1;
        p10_output.width = PW;
        p10_output.height = PH;
        p10_output.depth = 1;
        p10_output.gpu_addr = p10_address;
        p10_output.size = static_cast<uint32_t>(p10_guest.size());
        ShaderResourceTable p10_table;
        p10_table.resources.push_back(p10_output);
        ComputeShaderConfig p10_config;
        p10_config.user_sgprs.resize(16);
        p10_config.local_x = PW;
        p10_config.local_y = PH;
        p10_config.local_z = 1;
        p10_config.threads_x = PW;
        p10_config.threads_y = PH;
        p10_config.threads_z = 1;
        p10_config.tidig_comp_cnt = 1;
        p10_config.native_storage_format_support =
            native_storage_format_support_bit(DataFormat::Unorm2_10_10_10, 4);
        const auto p10_spirv =
            recompile_compute(store_p10, std::size(store_p10), &p10_table, p10_config);
        CHECK(!p10_spirv.empty(), "R10G10B10A2 storage writer recompiles");
        ComputeItem p10_full;
        p10_full.spirv = p10_spirv;
        p10_full.resources = std::make_shared<ShaderResourceTable>(p10_table);
        p10_full.launch.threads_x = PW;
        p10_full.launch.threads_y = PH;
        p10_full.launch.threads_z = 1;
        p10_full.launch.local_x = PW;
        p10_full.launch.local_y = PH;
        p10_full.launch.local_z = 1;
        p10_full.launch.groups_x = p10_full.launch.groups_y = p10_full.launch.groups_z = 1;
        p10_full.code_addr = 0x37310031u;

        const auto p10_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(prosper::frontend::execute_live_compute_items({p10_full}),
              "R10G10B10A2 full-overwrite dispatch completes");
        const auto p10_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(p10_after.candidates == p10_before.candidates + 1 &&
                  p10_after.borrowed == p10_before.borrowed + 1 &&
                  p10_after.recorded == p10_before.recorded + 1 &&
                  p10_after.published == p10_before.published + 1 &&
                  p10_after.failed == p10_before.failed,
              "R10G10B10A2 result is mirrored on the GPU and publishes after writeback");
        // Guest bytes stay the architectural packed words: R low 10 bits, A top 2 bits. The pixel check
        // below is tied to these stored words, not to what the shader meant to store.
        std::vector<uint32_t> p10_words(size_t{PW} * PH);
        std::memcpy(p10_words.data(), p10_guest.data(), p10_guest.size());
        size_t covered = 0;
        for (size_t t = 0; t < p10_words.size(); ++t)
            covered +=
                (p10_words[t] & 0x3ffu) == t && ((p10_words[t] >> 10) & 0x3ffu) == (t ^ 0x155u) &&
                ((p10_words[t] >> 20) & 0x3ffu) == 1023u - t && (p10_words[t] >> 30) == t / PW;
        CHECK(covered == p10_words.size(),
              "R10G10B10A2 guest writeback holds every 10-bit value and every alpha");
        LiveTargetImageRequest p10_request{};
        p10_request.width = PW;
        p10_request.height = PH;
        LiveTargetImageImport p10_import;
        CHECK(import_live_render_target_image(p10_address, p10_request, p10_import) &&
                  p10_import.valid() && p10_import.format == LiveTargetPixelFormat::Rgba8Unorm &&
                  p10_import.native_format == VK_FORMAT_R8G8B8A8_UNORM,
              "completed R10G10B10A2 mirror leaves a readable RGBA8 renderer image");
        release_live_render_target_image(p10_address);
        std::vector<uint8_t> p10_pixels;
        std::string p10_error;
        const bool read = prosper::test::readback_persistent_color_target(p10_address, PW, PH,
                                                                          VK_FORMAT_R8G8B8A8_UNORM,
                                                                          p10_pixels, p10_error) &&
                          p10_pixels.size() == size_t{PW} * PH * 4;
        size_t mismatches[4] = {};
        for (size_t t = 0; read && t < p10_words.size(); ++t) {
            const uint32_t w = p10_words[t];
            const uint8_t* px = p10_pixels.data() + t * 4;
            mismatches[0] += px[0] != prosper::frontend::kUnorm10To8[w & 0x3ffu];
            mismatches[1] += px[1] != prosper::frontend::kUnorm10To8[(w >> 10) & 0x3ffu];
            mismatches[2] += px[2] != prosper::frontend::kUnorm10To8[(w >> 20) & 0x3ffu];
            mismatches[3] += px[3] != prosper::frontend::kUnorm2To8[w >> 30];
        }
        if (mismatches[0] || mismatches[1] || mismatches[2] || mismatches[3])
            std::fprintf(stderr, "R10G10B10A2 mismatches R=%zu G=%zu B=%zu A=%zu of %zu\n",
                         mismatches[0], mismatches[1], mismatches[2], mismatches[3],
                         p10_words.size());
        CHECK(read && !mismatches[0] && !mismatches[1] && !mismatches[2] && !mismatches[3],
              "R10G10B10A2 renderer image equals the CPU path's conversion byte for byte");
        // A packed-10 sampled binding is never a raw import of the RGBA8 renderer image, so the
        // raw-copy seed and result mirror (seed_from_imported) can never reach this format.
        CHECK(!prosper::frontend::direct_sampled_rtt_compatible(
                  DataFormat::Unorm2_10_10_10, 4, LiveTargetPixelFormat::Rgba8Unorm, true) &&
                  !prosper::frontend::direct_sampled_rtt_compatible(
                      DataFormat::Unorm2_10_10_10, 4, LiveTargetPixelFormat::Rgba8Unorm, false),
              "packed R10G10B10A2 never imports the RGBA8 renderer image raw");
    }
    // R8Unorm (one 8-bit channel) is the same destination-mirror shape as RGBA8: the shader writes a
    // native R8_UNORM storage image and the renderer's R8_UNORM image seeds and receives it by an
    // exact copy, so a one-channel result no longer round-trips through the CPU (it used to be
    // declined as "format-no-seed-path"). Rows are 256 texels = 256 bytes, GFX10's linear pitch.
    {
        constexpr uint32_t RW = 256, RH = 4;
        std::vector<uint8_t> r8_guest(RW * RH, 0x9d);
        const uint64_t r8_address = reinterpret_cast<uint64_t>(r8_guest.data());
        DrawItem r8_producer = producer;
        r8_producer.color0_base = r8_address;
        r8_producer.color0_width = RW; r8_producer.color0_height = RH;
        r8_producer.ps.color0_format = VK_FORMAT_R8_UNORM;
        CHECK(!render_submit_items({r8_producer}, RW, RH).empty(),
              "R8 destination mirror producer materializes a renderer target");
        auto r8_cpu_newer = std::make_shared<std::vector<uint8_t>>(RW * RH, 0x11);
        notify_live_render_target_image_written(
            {r8_address, RW, RH, LiveTargetPixelFormat::R8Unorm, std::move(r8_cpu_newer)});
        LiveTargetImageRequest r8_request{};
        r8_request.width = RW; r8_request.height = RH;
        LiveTargetImageImport r8_import;
        CHECK(!import_live_render_target_image(r8_address, r8_request, r8_import),
              "CPU-newer R8 target is refused as a strict source before its first mirror");

        static const uint32_t store_zero_r8[] = {
            0x7E080300u, 0x7E0A0301u, // v4=x, v5=y
            0x7E000280u,              // R=0
            0xF0200108u, 0x00020004u, // image_store v0 (dmask R) at v4,v5 through s[8:15]
            0xBF810000u,
        };
        static const uint32_t store_half_r8[] = {
            0x7E080300u, 0x7E0A0301u,
            0x7E0002F0u,              // R=0.5
            0xF0200108u, 0x00020004u,
            0xBF810000u,
        };
        ShaderResource r8_output{};
        r8_output.cls = ResourceClass::StorageImage; r8_output.format = DataFormat::Unorm8;
        r8_output.num_components = 1; r8_output.binding = 5; r8_output.sgpr_base = 8;
        r8_output.img_dim = 1; r8_output.width = RW; r8_output.height = RH; r8_output.depth = 1;
        r8_output.gpu_addr = r8_address; r8_output.size = static_cast<uint32_t>(r8_guest.size());
        ShaderResourceTable r8_table; r8_table.resources.push_back(r8_output);
        ComputeShaderConfig r8_config;
        r8_config.user_sgprs.resize(16); r8_config.local_x = RW; r8_config.local_y = RH; r8_config.local_z = 1;
        r8_config.threads_x = RW; r8_config.threads_y = RH; r8_config.threads_z = 1;
        r8_config.tidig_comp_cnt = 1;
        r8_config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Unorm8, 1);
        const auto r8_full_spirv = recompile_compute(
            store_zero_r8, std::size(store_zero_r8), &r8_table, r8_config);
        // The partial writer's workgroup is one row tall (the shader's local size comes from its config).
        ComputeShaderConfig r8_row_config = r8_config;
        r8_row_config.local_y = 1; r8_row_config.threads_y = 1;
        const auto r8_half_spirv = recompile_compute(
            store_half_r8, std::size(store_half_r8), &r8_table, r8_row_config);
        CHECK(!r8_full_spirv.empty() && !r8_half_spirv.empty(), "R8 storage writers recompile");
        ComputeItem r8_full;
        r8_full.spirv = r8_full_spirv;
        r8_full.resources = std::make_shared<ShaderResourceTable>(r8_table);
        r8_full.launch.threads_x = RW; r8_full.launch.threads_y = RH; r8_full.launch.threads_z = 1;
        r8_full.launch.local_x = RW; r8_full.launch.local_y = RH; r8_full.launch.local_z = 1;
        r8_full.launch.groups_x = r8_full.launch.groups_y = r8_full.launch.groups_z = 1;
        r8_full.code_addr = 0x37310021u;

        // 1. a failed completion must not leave readable authority (same contract as RGBA8).
        const auto r8_fail_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        prosper::frontend::live_compute_fail_next_storage_readback_for_test();
        CHECK(!prosper::frontend::execute_live_compute_items({r8_full}),
              "failed R8 completion is reported");
        const auto r8_fail_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(r8_fail_after.borrowed == r8_fail_before.borrowed + 1 &&
                  r8_fail_after.failed == r8_fail_before.failed + 1 &&
                  r8_fail_after.published == r8_fail_before.published &&
                  !import_live_render_target_image(r8_address, r8_request, r8_import),
              "failed R8 destination write never restores readable renderer authority");

        // 2. a full overwrite is mirrored GPU-side and equals the guest bytes bit for bit.
        const auto r8_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(prosper::frontend::execute_live_compute_items({r8_full}),
              "R8 full-overwrite dispatch completes into an invalid renderer destination");
        const auto r8_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(r8_after.candidates == r8_before.candidates + 1 &&
                  r8_after.borrowed == r8_before.borrowed + 1 &&
                  r8_after.published == r8_before.published + 1 &&
                  r8_after.failed == r8_before.failed,
              "R8 result is a destination-mirror candidate and publishes after writeback");
        CHECK(std::all_of(r8_guest.begin(), r8_guest.end(), [](uint8_t b) { return b == 0; }),
              "R8 full overwrite wrote 0 to every guest byte");
        CHECK(import_live_render_target_image(r8_address, r8_request, r8_import) && r8_import.valid() &&
                  r8_import.format == LiveTargetPixelFormat::R8Unorm &&
                  r8_import.native_format == VK_FORMAT_R8_UNORM,
              "completed R8 mirror leaves a readable R8_UNORM renderer image");
        release_live_render_target_image(r8_address);
        std::vector<uint8_t> r8_pixels;
        std::string r8_error;
        CHECK(prosper::test::readback_persistent_color_target(
                  r8_address, RW, RH, VK_FORMAT_R8_UNORM, r8_pixels, r8_error) &&
                  r8_pixels == r8_guest,
              "R8 renderer image equals the completed guest bytes");

        // 3. a partial writer is seeded from the renderer image: repaint the renderer target
        // differently from the stale guest bytes, write only row zero, rows below must be the seed.
        CHECK(!render_submit_items({r8_producer}, RW, RH).empty(),
              "R8 renderer repaints its target before the partial writer");
        std::vector<uint8_t> r8_seed;
        CHECK(prosper::test::readback_persistent_color_target(
                  r8_address, RW, RH, VK_FORMAT_R8_UNORM, r8_seed, r8_error) &&
                  r8_seed.size() == r8_guest.size() &&
                  !std::equal(r8_seed.begin() + RW, r8_seed.end(), r8_guest.begin() + RW),
              "R8 renderer seed differs from the stale guest rows");
        ComputeItem r8_partial = r8_full;
        r8_partial.spirv = r8_half_spirv;
        r8_partial.launch.threads_y = r8_partial.launch.local_y = r8_partial.launch.groups_y = 1;
        r8_partial.code_addr = 0x37310022u;
        const auto r8_seed_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(prosper::frontend::execute_live_compute_items({r8_partial}),
              "R8 partial writer completes from the renderer GPU seed");
        const auto r8_seed_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(r8_seed_after.r8_source_seed_recorded == r8_seed_before.r8_source_seed_recorded + 1 &&
                  r8_seed_after.published == r8_seed_before.published + 1 &&
                  r8_seed_after.failed == r8_seed_before.failed,
              "R8 partial writer records an exact GPU seed copy and publishes");
        CHECK(std::equal(r8_guest.begin() + RW, r8_guest.end(), r8_seed.begin() + RW) &&
                  std::all_of(r8_guest.begin(), r8_guest.begin() + RW,
                              [](uint8_t b) { return b == 127 || b == 128; }),
              "R8 partial writer writes 0.5 to row zero and preserves the GPU-seeded rows");
        CHECK(prosper::test::readback_persistent_color_target(
                  r8_address, RW, RH, VK_FORMAT_R8_UNORM, r8_pixels, r8_error) &&
                  r8_pixels == r8_guest,
              "R8 renderer image equals the guest bytes after the partial result");

        // 4. control: a shader that does NOT use native R8 storage must keep the CPU path. The
        // mirror gate is "native storage AND R8", so this is declined before any borrow.
        ComputeShaderConfig raw_config = r8_config;
        raw_config.native_storage_format_support = 0;
        const auto r8_raw_spirv = recompile_compute(
            store_zero_r8, std::size(store_zero_r8), &r8_table, raw_config);
        CHECK(!r8_raw_spirv.empty(), "raw (non-native) R8 storage writer recompiles");
        ComputeItem r8_raw = r8_full;
        r8_raw.spirv = r8_raw_spirv;
        r8_raw.code_addr = 0x37310023u;
        const auto r8_raw_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(prosper::frontend::execute_live_compute_items({r8_raw}),
              "raw R8 storage writer still completes through the CPU path");
        const auto r8_raw_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(r8_raw_after.borrowed == r8_raw_before.borrowed &&
                  r8_raw_after.published == r8_raw_before.published &&
                  std::all_of(r8_guest.begin(), r8_guest.end(), [](uint8_t b) { return b == 0; }),
              "raw R8 storage is not mirrored but its guest bytes are still correct");
    }

    // R8 at a width that is not a multiple of 256: GFX10 pads each guest row to 256 bytes (#4586), but
    // the renderer image is tight (width*height). The mirror must map padded guest rows <-> tight rows
    // and never touch the guest padding between rows.
    {
        constexpr uint32_t PW = 192, PH = 4, PITCH = 256, SENT = 0xA5;
        std::vector<uint8_t> pg(static_cast<size_t>(PITCH) * PH, SENT);
        const uint64_t p_address = reinterpret_cast<uint64_t>(pg.data());
        DrawItem p_producer = producer;
        p_producer.color0_base = p_address;
        p_producer.color0_width = PW; p_producer.color0_height = PH;
        p_producer.ps.color0_format = VK_FORMAT_R8_UNORM;
        CHECK(!render_submit_items({p_producer}, PW, PH).empty(),
              "padded R8 producer materializes a renderer target");
        auto p_cpu_newer = std::make_shared<std::vector<uint8_t>>(PW * PH, 0x11);
        notify_live_render_target_image_written(
            {p_address, PW, PH, LiveTargetPixelFormat::R8Unorm, std::move(p_cpu_newer)});
        LiveTargetImageRequest p_request{};
        p_request.width = PW; p_request.height = PH;
        LiveTargetImageImport p_import;

        static const uint32_t p_store_zero[] = {
            0x7E080300u, 0x7E0A0301u, 0x7E000280u, 0xF0200108u, 0x00020004u, 0xBF810000u,
        };
        static const uint32_t p_store_half[] = {
            0x7E080300u, 0x7E0A0301u, 0x7E0002F0u, 0xF0200108u, 0x00020004u, 0xBF810000u,
        };
        ShaderResource p_output{};
        p_output.cls = ResourceClass::StorageImage; p_output.format = DataFormat::Unorm8;
        p_output.num_components = 1; p_output.binding = 5; p_output.sgpr_base = 8;
        p_output.img_dim = 1; p_output.width = PW; p_output.height = PH; p_output.depth = 1;
        p_output.gpu_addr = p_address; p_output.size = static_cast<uint32_t>(pg.size());
        ShaderResourceTable p_table; p_table.resources.push_back(p_output);
        ComputeShaderConfig p_config;
        p_config.user_sgprs.resize(16); p_config.local_x = PW; p_config.local_y = PH; p_config.local_z = 1;
        p_config.threads_x = PW; p_config.threads_y = PH; p_config.threads_z = 1;
        p_config.tidig_comp_cnt = 1;
        p_config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Unorm8, 1);
        const auto p_full_spirv = recompile_compute(
            p_store_zero, std::size(p_store_zero), &p_table, p_config);
        ComputeShaderConfig p_row_config = p_config;
        p_row_config.local_y = 1; p_row_config.threads_y = 1;
        const auto p_half_spirv = recompile_compute(
            p_store_half, std::size(p_store_half), &p_table, p_row_config);
        CHECK(!p_full_spirv.empty() && !p_half_spirv.empty(), "padded R8 storage writers recompile");
        ComputeItem p_full;
        p_full.spirv = p_full_spirv;
        p_full.resources = std::make_shared<ShaderResourceTable>(p_table);
        p_full.launch.threads_x = PW; p_full.launch.threads_y = PH; p_full.launch.threads_z = 1;
        p_full.launch.local_x = PW; p_full.launch.local_y = PH; p_full.launch.local_z = 1;
        p_full.launch.groups_x = p_full.launch.groups_y = p_full.launch.groups_z = 1;
        p_full.code_addr = 0x37310031u;

        auto texels_are_zero = [&]() {
            for (uint32_t y = 0; y < PH; ++y)
                for (uint32_t x = 0; x < PW; ++x)
                    if (pg[static_cast<size_t>(y) * PITCH + x] != 0) return false;
            return true;
        };
        auto padding_is_sentinel = [&]() {
            for (uint32_t y = 0; y < PH; ++y)
                for (uint32_t x = PW; x < PITCH; ++x)
                    if (pg[static_cast<size_t>(y) * PITCH + x] != SENT) return false;
            return true;
        };
        // tight renderer image vs the padded guest, row by row.
        auto tight_matches_guest = [&](const std::vector<uint8_t>& tight) {
            if (tight.size() != static_cast<size_t>(PW) * PH) return false;
            for (uint32_t y = 0; y < PH; ++y)
                for (uint32_t x = 0; x < PW; ++x)
                    if (tight[static_cast<size_t>(y) * PW + x] != pg[static_cast<size_t>(y) * PITCH + x])
                        return false;
            return true;
        };

        // 1. full overwrite is mirrored; guest texels written, padding preserved, image tight.
        const auto p_before = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(prosper::frontend::execute_live_compute_items({p_full}),
              "padded R8 full-overwrite dispatch completes");
        const auto p_after = prosper::frontend::live_compute_rtt_destination_mirror_counters();
        CHECK(p_after.borrowed == p_before.borrowed + 1 &&
                  p_after.published == p_before.published + 1 &&
                  p_after.failed == p_before.failed,
              "padded R8 result is mirrored (borrowed and published +1)");
        CHECK(texels_are_zero(), "padded R8 guest texels are 0 row by row");
        CHECK(padding_is_sentinel(), "padded R8 guest row padding bytes are preserved");
        CHECK(import_live_render_target_image(p_address, p_request, p_import) && p_import.valid() &&
                  p_import.format == LiveTargetPixelFormat::R8Unorm,
              "padded R8 mirror leaves a readable R8_UNORM renderer image");
        release_live_render_target_image(p_address);
        std::vector<uint8_t> p_pixels;
        std::string p_error;
        CHECK(prosper::test::readback_persistent_color_target(
                  p_address, PW, PH, VK_FORMAT_R8_UNORM, p_pixels, p_error) &&
                  p_pixels.size() == static_cast<size_t>(PW) * PH && tight_matches_guest(p_pixels),
              "padded R8 renderer image is tight (width*height) and equals the guest texels");

        // 2. partial writer: rows 1.. come from the tight renderer seed (distinct from the stale guest)
        // and land at padded guest offsets with the padding untouched.
        CHECK(!render_submit_items({p_producer}, PW, PH).empty(),
              "padded R8 renderer repaints before the partial writer");
        std::vector<uint8_t> p_seed;
        CHECK(prosper::test::readback_persistent_color_target(
                  p_address, PW, PH, VK_FORMAT_R8_UNORM, p_seed, p_error) &&
                  p_seed.size() == static_cast<size_t>(PW) * PH &&
                  !std::all_of(p_seed.begin() + PW, p_seed.end(), [](uint8_t b) { return b == 0; }),
              "padded R8 renderer seed is tight and differs from the stale guest rows");
        ComputeItem p_partial = p_full;
        p_partial.spirv = p_half_spirv;
        p_partial.launch.threads_y = p_partial.launch.local_y = p_partial.launch.groups_y = 1;
        p_partial.code_addr = 0x37310032u;
        CHECK(prosper::frontend::execute_live_compute_items({p_partial}),
              "padded R8 partial writer completes");
        bool seed_rows_ok = true;
        for (uint32_t y = 1; y < PH; ++y)
            for (uint32_t x = 0; x < PW; ++x)
                seed_rows_ok &= pg[static_cast<size_t>(y) * PITCH + x] == p_seed[static_cast<size_t>(y) * PW + x];
        bool row0_ok = true;
        for (uint32_t x = 0; x < PW; ++x) row0_ok &= pg[x] == 127 || pg[x] == 128;
        CHECK(seed_rows_ok && row0_ok, "padded R8 guest rows 1.. equal the tight seed, row 0 is 0.5");
        CHECK(padding_is_sentinel(), "padded R8 partial result preserves guest padding");
        CHECK(prosper::test::readback_persistent_color_target(
                  p_address, PW, PH, VK_FORMAT_R8_UNORM, p_pixels, p_error) &&
                  tight_matches_guest(p_pixels),
              "padded R8 renderer image is tight and equals the guest texels after the partial result");
    }

    return fails ? 1 : 0;
}

int main() {
    return run_destination_mirror_regression();
}
