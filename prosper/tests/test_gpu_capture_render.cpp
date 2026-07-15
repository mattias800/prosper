#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../frontends/shared/live_renderer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

static void set_descriptor_mode(const char* value) {
#ifdef _WIN32
    _putenv_s("PROSPER_DESCRIPTOR_VALIDATE", value);
#else
    setenv("PROSPER_DESCRIPTOR_VALIDATE", value, 1);
#endif
}

int main() {
    std::printf("== test_gpu_capture_render ==\n");
    constexpr uint32_t W = 64, H = 64;

    const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    const uint32_t ps_rdna[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResourceTable compile_rt;
    ShaderResource tex{}; tex.cls = ResourceClass::Texture; tex.binding = 4; tex.img_dim = 1;
    tex.width = 2; tex.height = 2; tex.size = 16; tex.sgpr_base = 8; tex.gpu_addr = 0x100000;
    tex.mag_filter = tex.min_filter = 0; compile_rt.resources.push_back(tex);
    DrawItem item; item.vs = recompile_vertex(vs_rdna, std::size(vs_rdna));
    item.fs = recompile_fragment(ps_rdna, std::size(ps_rdna), &compile_rt);
    item.prt = std::make_shared<ShaderResourceTable>(compile_rt); item.vertex_count = 3;
    item.ps.topology = 3; item.ps.color_write_mask = 0xF; item.color0_base = 0x200000;
    CHECK(!item.vs.empty() && !item.fs.empty(), "test shaders compile to SPIR-V");

    const std::vector<uint8_t> texture = {
        255, 0, 0, 255,   0, 255, 0, 255,
        0, 0, 255, 255,   255, 255, 255, 255,
    };
    auto reader = [&](uint64_t addr, uint8_t* dst, size_t n) -> size_t {
        if (addr != tex.gpu_addr) return 0; n = std::min(n, texture.size()); std::memcpy(dst, texture.data(), n); return n;
    };
    GpuCaptureMetadata meta; meta.width = W; meta.height = H; meta.revision = "render-test";
    GpuCaptureFile capture; std::string error;
    CHECK(capture_draw_items({item}, meta, reader, capture, error), "textured draw captures");
    GpuReplayFrame replay;
    CHECK(materialize_gpu_replay(capture, replay, error), "textured draw materializes with owned bytes");
    CHECK(replay.items[0].prt->resources[0].gpu_addr == tex.gpu_addr &&
          replay.items[0].prt->resources[0].host_data_size == texture.size(),
          "replay keeps logical VA and exposes exact host backing size");

#ifdef _WIN32
    _putenv_s("PROSPER_GPU_CAPTURE", "1");
    _putenv_s("PROSPER_GPU_REPLAY_RTT_SEEDS", "1");
    _putenv_s("PROSPER_GPU_REPLAY_DS_SEEDS", "1");
    _putenv_s("PROSPER_GPU_REPLAY_EXPORT_DS", "1");
#else
    setenv("PROSPER_GPU_CAPTURE", "1", 1);
    setenv("PROSPER_GPU_REPLAY_RTT_SEEDS", "1", 1);
    setenv("PROSPER_GPU_REPLAY_DS_SEEDS", "1", 1);
    setenv("PROSPER_GPU_REPLAY_EXPORT_DS", "1", 1);
#endif
    prosper::frontend::register_live_renderer(".", false);
    std::vector<uint8_t> pixels = render_submit_items(replay.items, W, H);
    CHECK(pixels.size() == static_cast<size_t>(W) * H * 4, "replayed draw renders through live backend");
    if (!pixels.empty()) {
        const uint8_t* center = &pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
        CHECK(center[0] > 0xC0 && center[1] < 0x40 && center[2] < 0x40,
              "replay samples captured red texel from owned backing");
    }

    // Texture decode scratch survives between renderer callbacks and reuses slots from the front.
    // Grow it to two slots, then make a packed texture the sole decode in the next callback. Converting
    // texstore.back() here used to mutate the stale second slot while uploading the first slot's raw
    // R10G10B10A2 bytes, so a logical half-red texel arrived almost black.
    {
        uint8_t filler_a[16] = {
            255, 0, 0, 255, 255, 0, 0, 255,
            255, 0, 0, 255, 255, 0, 0, 255,
        };
        uint8_t filler_b[16] = {
            0, 255, 0, 255, 0, 255, 0, 255,
            0, 255, 0, 255, 0, 255, 0, 255,
        };
        auto texture_draw = [&](uint64_t texture_addr, uint64_t target_addr,
                                uint8_t* host_data, DataFormat format) {
            DrawItem draw = replay.items[0];
            draw.color0_base = target_addr;
            auto table = std::make_shared<ShaderResourceTable>(*draw.prt);
            ShaderResource& resource = table->resources[0];
            resource.gpu_addr = texture_addr;
            resource.size = 16;
            resource.width = resource.height = 2;
            resource.depth = 1;
            resource.tile_mode = 0;
            resource.format = format;
            resource.num_components = 4;
            resource.host_data = host_data;
            resource.host_data_size = 16;
            draw.prt = std::move(table);
            return draw;
        };
        DrawItem filler_draw_a = texture_draw(
            0xd10000, 0xd30000, filler_a, DataFormat::Unorm8);
        DrawItem filler_draw_b = texture_draw(
            0xd20000, 0xd30000, filler_b, DataFormat::Unorm8);
        std::vector<uint8_t> filler = render_submit_items(
            {filler_draw_a, filler_draw_b}, W, H);
        CHECK(!filler.empty(), "renderer grows reusable texture decode scratch beyond one slot");

        uint32_t packed_texels[4] = {
            0xC0000200u, 0xC0000200u, 0xC0000200u, 0xC0000200u,
        };  // logical RGBA = (128,0,0,255); raw little-endian bytes = (0,2,0,192)
        DrawItem packed_draw = texture_draw(
            0xd40000, 0xd50000, reinterpret_cast<uint8_t*>(packed_texels),
            DataFormat::Unorm2_10_10_10);
        std::vector<uint8_t> packed = render_submit_items({packed_draw}, W, H);
        bool packed_red = packed.size() == static_cast<size_t>(W) * H * 4;
        if (packed_red) {
            const uint8_t* center = &packed[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
            packed_red = center[0] >= 120 && center[0] <= 136 &&
                         center[1] < 8 && center[2] < 8 && center[3] > 240;
        }
        CHECK(packed_red,
              "reused scratch converts and uploads the current packed R10G10B10A2 texture");
    }

    // A non-VideoOut producer must render at CB_COLOR0_ATTRIB2's extent, be cached under its
    // target address, and feed a later pass. Before #526 both passes used the global 64x64 extent.
    DrawItem producer = replay.items[0];
    producer.color0_base = 0x300000;
    producer.color0_width = 32;
    producer.color0_height = 2;

    DrawItem consumer = replay.items[0];
    consumer.color0_base = 0x400000;
    consumer.color0_width = 16;
    consumer.color0_height = 16;
    auto consumer_rt = std::make_shared<ShaderResourceTable>(*consumer.prt);
    consumer_rt->resources[0].gpu_addr = producer.color0_base;
    consumer_rt->resources[0].width = producer.color0_width;
    consumer_rt->resources[0].height = producer.color0_height;
    consumer_rt->resources[0].host_data = nullptr;
    consumer_rt->resources[0].host_data_size = 0;
    consumer.prt = std::move(consumer_rt);

    std::vector<uint8_t> chained = render_submit_items({producer, consumer}, W, H);
    CHECK(chained.size() == static_cast<size_t>(16) * 16 * 4,
          "per-target chain returns the last pass at its declared 16x16 extent");
    if (chained.size() == static_cast<size_t>(16) * 16 * 4) {
        const uint8_t* center = &chained[(8u * 16u + 8u) * 4];
        CHECK(center[0] > 0xC0 && center[1] < 0x40 && center[2] < 0x40,
              "later pass samples pixels cached from the 32x2 producer target");
    }

    // A simultaneous MRT producer must publish its second attachment under CB_COLOR1_BASE so a later
    // pass can sample it in the same submit. This is DOLL's missing temporal scene dependency (#719).
    const uint32_t dual_ps_rdna[] = {
        0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u, // RED -> MRT0
        0xF800180Fu, 0x03020100u,
        0x7E080280u, 0x7E0A02F2u, 0x7E0C0280u, 0x7E0E02F2u, // GREEN -> MRT1
        0xF800181Fu, 0x07060504u, 0xBF810000u,
    };
    DrawItem mrt_producer = replay.items[0];
    mrt_producer.fs = recompile_fragment(dual_ps_rdna, std::size(dual_ps_rdna));
    mrt_producer.fs_identity = 0; mrt_producer.prt.reset();
    mrt_producer.color0_base = 0xa00000;
    mrt_producer.color0_width = 32; mrt_producer.color0_height = 32;
    mrt_producer.color1_base = 0xb00000;
    mrt_producer.color1_width = 32; mrt_producer.color1_height = 32;
    mrt_producer.ps.color_write_mask = 0xf;
    mrt_producer.ps.color1_write_mask = 0xf;

    DrawItem mrt_consumer = replay.items[0];
    mrt_consumer.color0_base = 0xc00000;
    mrt_consumer.color0_width = 16; mrt_consumer.color0_height = 16;
    mrt_consumer.color1_base = 0; mrt_consumer.color1_width = mrt_consumer.color1_height = 0;
    mrt_consumer.ps.color1_write_mask = 0;
    auto mrt_consumer_rt = std::make_shared<ShaderResourceTable>(*mrt_consumer.prt);
    mrt_consumer_rt->resources[0].gpu_addr = mrt_producer.color1_base;
    mrt_consumer_rt->resources[0].width = mrt_producer.color1_width;
    mrt_consumer_rt->resources[0].height = mrt_producer.color1_height;
    mrt_consumer_rt->resources[0].host_data = nullptr;
    mrt_consumer_rt->resources[0].host_data_size = 0;
    mrt_consumer.prt = std::move(mrt_consumer_rt);
    std::vector<uint8_t> mrt_chain = render_submit_items({mrt_producer, mrt_consumer}, W, H);
    CHECK(!mrt_producer.fs.empty() && mrt_chain.size() == 16u * 16u * 4u,
          "MRT1 producer and consumer execute through the per-target live renderer");
    if (mrt_chain.size() == 16u * 16u * 4u) {
        const uint8_t* center = &mrt_chain[(8u * 16u + 8u) * 4];
        CHECK(center[1] > 0xc0 && center[0] < 0x40 && center[2] < 0x40,
              "later pass samples the GREEN pixels retained from color attachment 1");
    }

    render_submit_items({producer}, W, H);
    std::vector<uint8_t> cross_submit = render_submit_items({consumer}, W, H);
    CHECK(cross_submit.size() == static_cast<size_t>(16) * 16 * 4,
          "renderer retains a producer target across separate replay submits");
    if (cross_submit.size() == static_cast<size_t>(16) * 16 * 4) {
        const uint8_t* center = &cross_submit[(8u * 16u + 8u) * 4];
        CHECK(center[0] > 0xC0 && center[1] < 0x40 && center[2] < 0x40,
              "later replay submit samples the retained producer target");
    }

    // UE depth prepasses can use a tiny dummy color target with writes disabled while their viewport
    // and depth surface are full-size. The lighting pass then changes color targets but keeps the exact
    // DS identity and performs EQUAL testing. Persistent DS must follow the viewport extent, not the
    // dummy color extent, or the lighting pass receives a fresh clear image and rejects every pixel.
    DrawItem depth_prepass = replay.items[0];
    depth_prepass.color0_base = 0x910000;
    depth_prepass.color0_width = 8; depth_prepass.color0_height = 8;
    depth_prepass.ps.color_write_mask = 0;
    depth_prepass.ps.depth_test_enable = true;
    depth_prepass.ps.depth_write_enable = true;
    depth_prepass.ps.depth_compare_op = 7; // ALWAYS: populate the persistent surface
    depth_prepass.ps.depth_clear_value = 1.0f;
    depth_prepass.ps.has_viewport = true;
    depth_prepass.ps.viewport_x = 0.0f; depth_prepass.ps.viewport_y = static_cast<float>(H);
    depth_prepass.ps.viewport_w = static_cast<float>(W); depth_prepass.ps.viewport_h = -static_cast<float>(H);
    depth_prepass.ps.depth_read_base = depth_prepass.ps.depth_write_base = 0x920000;
    depth_prepass.ps.htile_data_base = 0x930000;

    DrawItem equal_lighting = depth_prepass;
    equal_lighting.color0_base = 0x940000;
    equal_lighting.color0_width = W; equal_lighting.color0_height = H;
    equal_lighting.ps.color_write_mask = 0xF;
    equal_lighting.ps.depth_write_enable = false;
    equal_lighting.ps.depth_compare_op = 2; // EQUAL: same VS depth must match the prepass
    std::vector<uint8_t> depth_reused = render_submit_items({depth_prepass, equal_lighting}, W, H);
    CHECK(depth_reused.size() == static_cast<size_t>(W) * H * 4,
          "full-size lighting pass follows a dummy-color depth prepass");
    if (depth_reused.size() == static_cast<size_t>(W) * H * 4) {
        const uint8_t* center = &depth_reused[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
        CHECK(center[0] > 0xC0 && center[1] < 0x40 && center[2] < 0x40,
              "EQUAL lighting reuses viewport-sized depth written behind the tiny color attachment");
    }

    // A translated viewport is raster state, not proof that the attachment itself is larger than the
    // presentation surface. Promoting its far edge used to allocate and retain enormous depth images
    // (up to 16384x16384) during Dead Cells level loading. Keep this deliberately out-of-bounds
    // prepass on its declared dummy extent; the full-size EQUAL pass must not find invented depth.
    DrawItem translated_prepass = depth_prepass;
    translated_prepass.color0_base = 0x950000;
    translated_prepass.ps.viewport_x = static_cast<float>(W * 2);
    translated_prepass.ps.depth_read_base = translated_prepass.ps.depth_write_base = 0x960000;
    translated_prepass.ps.htile_data_base = 0x970000;
    DrawItem translated_lighting = equal_lighting;
    translated_lighting.color0_base = 0x980000;
    translated_lighting.ps.depth_read_base = translated_lighting.ps.depth_write_base = 0x960000;
    translated_lighting.ps.htile_data_base = 0x970000;
    std::vector<uint8_t> translated = render_submit_items(
        {translated_prepass, translated_lighting}, W, H);
    CHECK(translated.size() == static_cast<size_t>(W) * H * 4,
          "oversized viewport does not inflate the persistent depth attachment");
    if (translated.size() == static_cast<size_t>(W) * H * 4) {
        const uint8_t* center = &translated[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
        CHECK(center[0] < 0x40,
              "lighting does not reuse depth invented from a translated viewport far edge");
    }

    // A captured consumer may depend on a producer from an earlier submit. Restore its serialized
    // host RTT surface before replaying draw zero; guest memory has no copy of these rendered pixels.
    GpuCaptureRttSeed temporal_seed;
    temporal_seed.guest_addr = 0x600000; temporal_seed.width = 2; temporal_seed.height = 2;
    temporal_seed.rgba = {
        255, 0, 0, 255, 255, 0, 0, 255,
        255, 0, 0, 255, 255, 0, 0, 255,
    };
    CHECK(restore_gpu_replay_rtt_seeds({temporal_seed}, error),
          "replay restores a producer surface captured from an earlier submit");
    DrawItem temporal_consumer = replay.items[0];
    temporal_consumer.color0_base = 0x700000;
    temporal_consumer.color0_width = 8; temporal_consumer.color0_height = 8;
    auto temporal_rt = std::make_shared<ShaderResourceTable>(*temporal_consumer.prt);
    temporal_rt->resources[0].gpu_addr = temporal_seed.guest_addr;
    temporal_rt->resources[0].width = temporal_seed.width;
    temporal_rt->resources[0].height = temporal_seed.height;
    temporal_rt->resources[0].host_data = nullptr;
    temporal_rt->resources[0].host_data_size = 0;
    temporal_consumer.prt = std::move(temporal_rt);
    std::vector<uint8_t> temporal = render_submit_items({temporal_consumer}, W, H);
    CHECK(temporal.size() == 8u * 8u * 4u, "temporal consumer renders at its native target extent");
    if (temporal.size() == 8u * 8u * 4u) {
        const uint8_t* center = &temporal[(4u * 8u + 4u) * 4];
        CHECK(center[0] > 0xC0 && center[1] < 0x40 && center[2] < 0x40,
              "temporal consumer samples restored host RTT pixels, not guest-memory fallback");
    }

    // A retained CPU RTT is only authoritative until compute/DMA writes its guest backing. Dead
    // Cells clears this FP16 lighting history with compute before drawing the next frame; keeping the
    // older CPU seed defeated that clear and fed brightness back until the scene became white/yellow.
    GpuCaptureRttSeed fp16_history;
    fp16_history.guest_addr = 0x610000;
    fp16_history.width = fp16_history.height = 2;
    fp16_history.format = GpuCaptureColorFormat::Rgba16Float;
    fp16_history.rgba.assign(2u * 2u * 8u, 0x5a);
    CHECK(restore_gpu_replay_rtt_seeds({fp16_history}, error),
          "replay restores a native FP16 temporal history surface");
    GpuCaptureRttSeed retained_history;
    CHECK(read_gpu_capture_rtt_seed(fp16_history.guest_addr, retained_history, error) &&
          retained_history.format == GpuCaptureColorFormat::Rgba16Float,
          "restored FP16 history is visible to the capture reader");
    notify_guest_gpu_write(fp16_history.guest_addr + fp16_history.rgba.size(), 1);
    CHECK(read_gpu_capture_rtt_seed(fp16_history.guest_addr, retained_history, error),
          "guest write at the native FP16 range end does not invalidate history");
    notify_guest_gpu_write(fp16_history.guest_addr + fp16_history.rgba.size() - 1, 1);
    CHECK(!read_gpu_capture_rtt_seed(fp16_history.guest_addr, retained_history, error) &&
          error == "render target is absent from live cache",
          "overlapping guest GPU write invalidates stale CPU FP16 history");

    GpuCaptureDsSeed ds_seed;
    ds_seed.depth_read_base = ds_seed.depth_write_base = 0x810000;
    ds_seed.stencil_read_base = ds_seed.stencil_write_base = 0x820000;
    ds_seed.htile_data_base = 0x800000;
    ds_seed.width = 4; ds_seed.height = 3;
    ds_seed.format = GpuCaptureDsFormat::D32FloatS8;
    ds_seed.depth_valid = true; ds_seed.stencil_valid = true;
    ds_seed.depth.resize(4u * 3u * 4u);
    ds_seed.stencil.resize(4u * 3u);
    for (size_t i = 0; i < ds_seed.depth.size(); ++i)
        ds_seed.depth[i] = static_cast<uint8_t>(i * 17u + 3u);
    for (size_t i = 0; i < ds_seed.stencil.size(); ++i)
        ds_seed.stencil[i] = static_cast<uint8_t>(i * 11u + 1u);
    CHECK(restore_gpu_replay_ds_seeds({ds_seed}, error),
          "replay uploads exact persistent Vulkan depth and stencil planes");
    std::vector<GpuCaptureDsSeed> ds_snapshot;
    const bool ds_read = read_all_gpu_capture_ds_seeds(ds_snapshot, error);
    const auto restored = std::find_if(ds_snapshot.begin(), ds_snapshot.end(), [&](const auto& seed) {
        return seed.depth_read_base == ds_seed.depth_read_base &&
               seed.stencil_read_base == ds_seed.stencil_read_base;
    });
    CHECK(ds_read && restored != ds_snapshot.end() && restored->depth == ds_seed.depth &&
          restored->stencil == ds_seed.stencil && restored->depth_valid && restored->stencil_valid,
          "persistent Vulkan DS snapshot reads back byte-exact checkpoint contents");

    DrawItem missing_texture = replay.items[0];
    missing_texture.prt = std::make_shared<ShaderResourceTable>();
    missing_texture.color0_base = 0x500000;
    missing_texture.color0_width = 8;
    missing_texture.color0_height = 8;
    set_descriptor_mode("poison");
    std::vector<uint8_t> poisoned = render_submit_items({missing_texture}, W, H);
    set_descriptor_mode("off");
    CHECK(poisoned.size() == static_cast<size_t>(8) * 8 * 4,
          "poison mode keeps a missing-descriptor draw executable at its target extent");
    if (poisoned.size() == static_cast<size_t>(8) * 8 * 4) {
        size_t rgb_nonblack = 0;
        for (size_t i = 0; i < poisoned.size(); i += 4)
            rgb_nonblack += poisoned[i] || poisoned[i + 1] || poisoned[i + 2];
        CHECK(rgb_nonblack == 64,
              "missing sampled image receives a full-surface nonblack poison texture instead of zero");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
