#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/videoout_present.hpp"
#include "../src/hle/dispatch.hpp"
#include "../frontends/shared/live_renderer.hpp"
#include "render_runner.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

using namespace prosper::gpu;

using Hle8Fn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t,
                            uint64_t, uint64_t, uint64_t, uint64_t);

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

static void set_env(const char* name, const std::string& value) {
#ifdef _WIN32
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

static void unset_env(const char* name) {
#ifdef _WIN32
    _putenv_s(name, "");
#else
    unsetenv(name);
#endif
}

int main() {
    // buffer_upload_bytes caches this override on first use. Reset it before the renderer can build
    // any buffer resource so the test always exercises the production default ceiling.
    unset_env("PROSPER_MAX_BUFFER_UPLOAD_MB");
    std::printf("== test_gpu_capture_render ==\n");
    constexpr uint32_t W = 64, H = 64;

    // Give the renderer a 128x128 guest presentation surface while rendering at 64x64. DrawItems
    // entering the live callback have therefore already had viewport/scissor coordinates scaled by
    // one half, exactly like a PROSPER_RENDER_SCALE=2 game run.
    prosper::register_builtin_hle();
    present_reset();
    auto open = prosper::Hle::lookup(prosper::nid_hash("sceVideoOutOpen"));
    auto setba2 = reinterpret_cast<Hle8Fn>(
        prosper::Hle::lookup("PjS5uASwcV8"));
    auto regb2 = prosper::Hle::lookup("rKBUtgRrtbk");
    constexpr uint32_t PRESENT_W = 128, PRESENT_H = 128;
    std::vector<uint8_t> scanout0(PRESENT_W * PRESENT_H * 4u);
    std::vector<uint8_t> scanout1(PRESENT_W * PRESENT_H * 4u);
    std::vector<uint8_t> scanout2(PRESENT_W * PRESENT_H * 4u);
    uint8_t scanout_attr[0x50]{};
    struct VideoBuffer { const void* data; const void* metadata; const void* reserved[2]; };
    VideoBuffer scanouts[3] = {
        {scanout0.data(), nullptr, {nullptr, nullptr}},
        {scanout1.data(), nullptr, {nullptr, nullptr}},
        {scanout2.data(), nullptr, {nullptr, nullptr}},
    };
    const uint64_t video_handle = open ? open(0, 0, 0, 0, 0, 0) : 0;
    if (setba2)
        setba2(reinterpret_cast<uint64_t>(scanout_attr), 0x8000000000000000ull,
               0, PRESENT_W, PRESENT_H, 0, 0, 0);
    const uint64_t register_result = regb2
        ? regb2(video_handle, 0, 0, reinterpret_cast<uint64_t>(scanouts), 3,
                reinterpret_cast<uint64_t>(scanout_attr))
        : UINT64_MAX;
    CHECK(open && setba2 && regb2 && register_result == 0 &&
              present_width() == PRESENT_W && present_height() == PRESENT_H,
          "live-render test configures a reduced-resolution presentation surface");

    const uint32_t vs_rdna[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    const uint32_t ps_rdna[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResourceTable compile_rt;
    ShaderResource tex{}; tex.cls = ResourceClass::Texture; tex.format = DataFormat::Unorm8;
    tex.num_components = 4; tex.binding = 4; tex.img_dim = 1;
    tex.width = 2; tex.height = 2; tex.size = 16; tex.sgpr_base = 8; tex.gpu_addr = 0x100000;
    tex.linear_row_pitch_bytes = 8;  // synthetic reader publishes tightly packed RGBA8 rows
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

    // NV12 chroma arrives as an RG8 sampled texture. The narrow upload path used to broadcast only
    // its first byte, turning (U,V) into (U,U) and giving every decoded movie a green/purple cast.
    {
        uint8_t rg_texels[8] = {32, 224, 32, 224, 32, 224, 32, 224};
        DrawItem rg_draw = replay.items[0];
        rg_draw.color0_base = 0x210000;
        auto rg_table = std::make_shared<ShaderResourceTable>(*rg_draw.prt);
        ShaderResource& rg = rg_table->resources[0];
        rg.gpu_addr = 0x110000;
        rg.format = DataFormat::Unorm8;
        rg.num_components = 2;
        rg.width = rg.height = 2;
        rg.size = sizeof(rg_texels);
        rg.linear_row_pitch_bytes = 4;
        rg.host_data = rg_texels;
        rg.host_data_size = sizeof(rg_texels);
        rg.swizzle[0] = 4; rg.swizzle[1] = 5; rg.swizzle[2] = 0; rg.swizzle[3] = 1;
        rg_draw.prt = rg_table;
        const std::vector<uint8_t> generic_rg_pixels =
            render_submit_items({rg_draw}, W, H);
        bool generic_rg_broadcast =
            generic_rg_pixels.size() == static_cast<size_t>(W) * H * 4;
        if (generic_rg_broadcast) {
            const uint8_t* center =
                &generic_rg_pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
            for (uint32_t channel = 0; channel < 4; ++channel)
                generic_rg_broadcast &= center[channel] >= 24 && center[channel] <= 40;
        }
        CHECK(generic_rg_broadcast,
              "ordinary narrow RG8 uploads retain the established coverage broadcast");

        uint8_t luma_texels[16] = {};
        ShaderResource luma = rg;
        luma.binding = 5;
        luma.gpu_addr = rg.gpu_addr - sizeof(luma_texels);
        luma.num_components = 1;
        luma.width = luma.height = 4;
        luma.size = sizeof(luma_texels);
        luma.linear_row_pitch_bytes = 4;
        luma.host_data = luma_texels;
        luma.host_data_size = sizeof(luma_texels);
        rg_table->resources.push_back(luma);
        rg_draw.prt = std::move(rg_table);
        const std::vector<uint8_t> rg_pixels = render_submit_items({rg_draw}, W, H);
        bool rg_preserved = rg_pixels.size() == static_cast<size_t>(W) * H * 4;
        if (rg_preserved) {
            const uint8_t* center =
                &rg_pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
            rg_preserved = center[0] >= 24 && center[0] <= 40 &&
                           center[1] >= 216 && center[1] <= 232 &&
                           center[2] < 8 && center[3] > 248;
        }
        CHECK(rg_preserved,
              "narrow RG8 uploads preserve both channels and apply the descriptor swizzle");
    }

    // R8 masks historically broadcast coverage to every channel and deliberately ignore a T#
    // remap. Keep that independent behavior while the exact AvPlayer RG8 contract preserves U/V.
    {
        uint8_t r_texels[4] = {96, 96, 96, 96};
        DrawItem r_draw = replay.items[0];
        r_draw.color0_base = 0x220000;
        auto r_table = std::make_shared<ShaderResourceTable>(*r_draw.prt);
        ShaderResource& r = r_table->resources[0];
        r.gpu_addr = 0x120000;
        r.format = DataFormat::Unorm8;
        r.num_components = 1;
        r.width = r.height = 2;
        r.size = sizeof(r_texels);
        r.linear_row_pitch_bytes = 2;
        r.host_data = r_texels;
        r.host_data_size = sizeof(r_texels);
        r.swizzle[0] = 0; r.swizzle[1] = 1; r.swizzle[2] = 0; r.swizzle[3] = 1;
        r_draw.prt = std::move(r_table);
        const std::vector<uint8_t> r_pixels = render_submit_items({r_draw}, W, H);
        bool r_broadcast = r_pixels.size() == static_cast<size_t>(W) * H * 4;
        if (r_broadcast) {
            const uint8_t* center =
                &r_pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
            for (uint32_t channel = 0; channel < 4; ++channel)
                r_broadcast &= center[channel] >= 88 && center[channel] <= 104;
        }
        CHECK(r_broadcast,
              "narrow R8 uploads retain coverage broadcast independently of descriptor swizzle");
    }

    {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        const std::filesystem::path dump_dir = std::filesystem::temp_directory_path() /
            ("prosper-rtgroup-rgba8-" + std::to_string(nonce));
        std::filesystem::create_directories(dump_dir);
        set_env("PROSPER_FRAME_DIR", dump_dir.string());
        set_env("PROSPER_DUMP_RTGROUPS_RGBA", "1");

        DrawItem rgba8_dump = replay.items[0];
        rgba8_dump.color0_base = 0xda700000;
        rgba8_dump.color0_width = 8;
        rgba8_dump.color0_height = 8;
        rgba8_dump.draw_index = (uint64_t{1} << 32) + 7;
        rgba8_dump.ps.color0_format = VK_FORMAT_R8G8B8A8_UNORM;
        render_submit_items({rgba8_dump}, 8, 8);

        const std::string rgba8_prefix = "rtgrp_da700000_8x8_";
        const std::string draw_suffix = "_d4294967303-4294967303.rgba";
        bool found_rgba8 = false;
        for (const auto& entry : std::filesystem::directory_iterator(dump_dir)) {
            const std::string name = entry.path().filename().string();
            if (name.starts_with(rgba8_prefix) && name.ends_with(draw_suffix)) {
                found_rgba8 = std::filesystem::file_size(entry.path()) == 8u * 8u * 4u;
            }
        }
        CHECK(found_rgba8, "RT-group RGBA8 dump preserves alpha bytes and 64-bit draw indices");

        DrawItem fp16_dump = rgba8_dump;
        fp16_dump.color0_base = 0xdb700000;
        fp16_dump.ps.color0_format = VK_FORMAT_R16G16B16A16_SFLOAT;
        render_submit_items({fp16_dump}, 8, 8);
        bool found_fp16 = false;
        for (const auto& entry : std::filesystem::directory_iterator(dump_dir)) {
            found_fp16 |= entry.path().filename().string().starts_with("rtgrp_db700000_8x8_");
        }
        CHECK(!found_fp16, "RT-group RGBA8 dump skips native FP16 targets");

        unset_env("PROSPER_DUMP_RTGROUPS_RGBA");
        unset_env("PROSPER_FRAME_DIR");
        std::filesystem::remove_all(dump_dir);
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

        float float_texels[16] = {
            0.5f, 0.25f, 0.125f, 1.0f, 0.5f, 0.25f, 0.125f, 1.0f,
            0.5f, 0.25f, 0.125f, 1.0f, 0.5f, 0.25f, 0.125f, 1.0f,
        };
        DrawItem float_draw = replay.items[0];
        float_draw.color0_base = 0xd60000;
        auto float_table = std::make_shared<ShaderResourceTable>(*float_draw.prt);
        ShaderResource& float_resource = float_table->resources[0];
        float_resource.gpu_addr = 0xd70000;
        float_resource.size = sizeof(float_texels);
        float_resource.width = float_resource.height = 2;
        float_resource.depth = 1;
        float_resource.img_dim = 1;
        float_resource.tile_mode = 0;
        float_resource.linear_row_pitch_bytes = 2u * 4u * sizeof(float);
        float_resource.format = DataFormat::Float32;
        float_resource.num_components = 4;
        float_resource.host_data = reinterpret_cast<uint8_t*>(float_texels);
        float_resource.host_data_size = sizeof(float_texels);
        float_draw.prt = std::move(float_table);
        const std::vector<uint8_t> float_pixels = render_submit_items({float_draw}, W, H);
        bool float_sampled = float_pixels.size() == static_cast<size_t>(W) * H * 4;
        if (float_sampled) {
            const uint8_t* center =
                &float_pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
            float_sampled = center[0] >= 120 && center[0] <= 136 &&
                            center[1] >= 56 && center[1] <= 72 &&
                            center[2] >= 24 && center[2] <= 40 && center[3] > 240;
        }
        CHECK(float_sampled,
              "sampled Float32x4 texture preserves ordinary values instead of raw RGBA8 bytes");
    }

#if defined(_WIN32) || defined(__linux__)
    // Evergate's dominant sampled resources are guest-backed BC textures. The frontend must retain
    // their decoded pixels under an exact source-content version so the backend can retain the
    // uploaded image too. A second unchanged callback previously decoded and uploaded BC again.
    {
        auto map_flexible = prosper::Hle::lookup(
            prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
        auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
        constexpr uint64_t source_mapping_size = 0x10000;
        uint64_t source_va = 0;
        const uint64_t source_name = reinterpret_cast<uint64_t>("bc-cache-test");
        CHECK(map_flexible && unmap &&
                  map_flexible(reinterpret_cast<uint64_t>(&source_va), source_mapping_size,
                               0x2 /* RW */, 0, source_name, 0) == 0 && source_va,
              "persistent BC test maps guest-readable source memory");
        if (source_va) {
            const uint8_t bc3_red[16] = {
                0xff, 0x00, 0x88, 0xc6, 0xfa, 0x88, 0xc6, 0xfa,
                0x00, 0xf8, 0x1f, 0x00, 0x00, 0x00, 0x00, 0x00,
            };
            std::memcpy(reinterpret_cast<void*>(source_va), bc3_red, sizeof(bc3_red));

            DrawItem bc_draw = replay.items[0];
            bc_draw.color0_base = 0xd60000;
            auto bc_table = std::make_shared<ShaderResourceTable>(*bc_draw.prt);
            ShaderResource& bc_resource = bc_table->resources[0];
            bc_resource.gpu_addr = source_va;
            bc_resource.host_data = nullptr;
            bc_resource.host_data_size = 0;
            bc_resource.size = sizeof(bc3_red);
            bc_resource.width = bc_resource.height = 4;
            bc_resource.depth = 1;
            bc_resource.img_dim = 1;
            bc_resource.tile_mode = 0;
            bc_resource.format = DataFormat::Bc3;
            bc_resource.num_components = 4;
            bc_resource.compression_enabled = false;
            bc_draw.prt = std::move(bc_table);

            const std::vector<uint8_t> first_bc = render_submit_items({bc_draw}, W, H);
            const auto first_bc_stats = prosper::test::backend_texture_upload_stats();
            const std::vector<uint8_t> reused_bc = render_submit_items({bc_draw}, W, H);
            const auto reused_bc_stats = prosper::test::backend_texture_upload_stats();
            CHECK(first_bc_stats.persistent_misses == 1 &&
                      first_bc_stats.unique_uploads == 1,
                  "first exact-validated BC version enters the persistent backend cache");
            CHECK(reused_bc_stats.persistent_hits == 1 &&
                      reused_bc_stats.unique_uploads == 0 && reused_bc_stats.upload_bytes == 0,
                  "unchanged guest-backed BC texture skips its next decode/upload callback");
            CHECK(!first_bc.empty() && first_bc == reused_bc,
                  "persistent BC reuse remains pixel-identical to the initial decode");
#ifdef __linux__
            // The cache write-watch may protect this page after the first validation. A real guest
            // write must fault/rearm normally, invalidate the content version, and upload new pixels.
            auto* mutable_bc = reinterpret_cast<uint8_t*>(source_va);
            mutable_bc[8] = 0xe0;
            mutable_bc[9] = 0x07;  // BC3 color endpoint 0: red -> green
            const std::vector<uint8_t> changed_bc = render_submit_items({bc_draw}, W, H);
            const auto changed_bc_stats = prosper::test::backend_texture_upload_stats();
            CHECK(changed_bc_stats.persistent_misses == 1 &&
                      changed_bc_stats.unique_uploads == 1,
                  "guest BC mutation invalidates the decoded and uploaded content version");
            CHECK(!changed_bc.empty() && changed_bc != reused_bc,
                  "guest BC mutation publishes newly decoded pixels instead of stale cache data");
#endif

            // Cobra repeatedly binds guest-backed Float32 post-process inputs. They narrow to
            // RGBA16F, but an unchanged version should retain both that conversion and its backend
            // image just like BC/fp16 textures do.
            float* float_source = reinterpret_cast<float*>(source_va + 0x1000);
            const float float_values[16] = {
                0.5f, 0.25f, 0.125f, 1.0f, 0.5f, 0.25f, 0.125f, 1.0f,
                0.5f, 0.25f, 0.125f, 1.0f, 0.5f, 0.25f, 0.125f, 1.0f,
            };
            std::memcpy(float_source, float_values, sizeof(float_values));
            DrawItem cached_float_draw = replay.items[0];
            cached_float_draw.color0_base = 0xd80000;
            auto cached_float_table =
                std::make_shared<ShaderResourceTable>(*cached_float_draw.prt);
            ShaderResource& cached_float_resource = cached_float_table->resources[0];
            cached_float_resource.gpu_addr = reinterpret_cast<uint64_t>(float_source);
            cached_float_resource.host_data = nullptr;
            cached_float_resource.host_data_size = 0;
            cached_float_resource.size = sizeof(float_values);
            cached_float_resource.width = cached_float_resource.height = 2;
            cached_float_resource.depth = 1;
            cached_float_resource.img_dim = 1;
            cached_float_resource.tile_mode = 0;
            cached_float_resource.linear_row_pitch_bytes = 2u * 4u * sizeof(float);
            cached_float_resource.format = DataFormat::Float32;
            cached_float_resource.num_components = 4;
            cached_float_resource.compression_enabled = false;
            cached_float_draw.prt = std::move(cached_float_table);

            const std::vector<uint8_t> first_float =
                render_submit_items({cached_float_draw}, W, H);
            const auto first_float_stats = prosper::test::backend_texture_upload_stats();
            const std::vector<uint8_t> reused_float =
                render_submit_items({cached_float_draw}, W, H);
            const auto reused_float_stats = prosper::test::backend_texture_upload_stats();
            CHECK(first_float_stats.persistent_misses == 1 &&
                      first_float_stats.unique_uploads == 1,
                  "first guest-backed Float32 version enters the persistent texture cache");
            CHECK(reused_float_stats.persistent_hits == 1 &&
                      reused_float_stats.unique_uploads == 0 &&
                      reused_float_stats.upload_bytes == 0,
                  "unchanged Float32 texture skips its next narrow/decode/upload callback");
            CHECK(!first_float.empty() && first_float == reused_float,
                  "persistent Float32 reuse preserves the native RGBA16F sampling result");

            // Graphics lowers a sampled 2D array to its base slice today. The descriptor address is
            // the layer allocation base, while layer_mip_offset selects the level that must be
            // decoded. Exercise both the offset and the persistent decoded/uploaded reuse contract.
            float* array_base = reinterpret_cast<float*>(source_va + 0x2000);
            float* array_selected = reinterpret_cast<float*>(source_va + 0x2100);
            const float base_values[16] = {
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
            };
            std::memcpy(array_base, base_values, sizeof(base_values));
            constexpr size_t array_row_pitch = 0x100;
            constexpr size_t array_tight_row = 2u * 4u * sizeof(float);
            std::memcpy(array_selected, float_values, array_tight_row);
            std::memcpy(reinterpret_cast<uint8_t*>(array_selected) + array_row_pitch,
                        reinterpret_cast<const uint8_t*>(float_values) + array_tight_row,
                        array_tight_row);
            DrawItem array_draw = cached_float_draw;
            array_draw.color0_base = 0xd90000;
            auto array_table = std::make_shared<ShaderResourceTable>(*array_draw.prt);
            ShaderResource& array_resource = array_table->resources[0];
            array_resource.gpu_addr = reinterpret_cast<uint64_t>(array_base);
            array_resource.size = 4u * 0x1000u;
            array_resource.img_dim = 5;
            array_resource.depth = 4;
            array_resource.layer_stride_bytes = 0x1000;
            array_resource.layer_mip_offset_bytes = 0x100;
            array_resource.linear_row_pitch_bytes = array_row_pitch;
            array_draw.prt = std::move(array_table);

            const std::vector<uint8_t> first_array =
                render_submit_items({array_draw}, W, H);
            const auto first_array_stats = prosper::test::backend_texture_upload_stats();
            const std::vector<uint8_t> reused_array =
                render_submit_items({array_draw}, W, H);
            const auto reused_array_stats = prosper::test::backend_texture_upload_stats();
            bool array_selected_level =
                first_array.size() == static_cast<size_t>(W) * H * 4;
            if (array_selected_level) {
                const uint8_t* center =
                    &first_array[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
                array_selected_level = center[0] >= 120 && center[0] <= 136 &&
                    center[1] >= 56 && center[1] <= 72 &&
                    center[2] >= 24 && center[2] <= 40 && center[3] > 240;
            }
            CHECK(array_selected_level,
                  "2D-array base-slice decode reads the selected mip offset, not allocation base");
            CHECK(first_array_stats.persistent_misses == 1 &&
                      first_array_stats.unique_uploads == 1 &&
                      reused_array_stats.persistent_hits == 1 &&
                      reused_array_stats.unique_uploads == 0,
                  "unchanged 2D-array base slice reuses its decoded and uploaded image");
            CHECK(!first_array.empty() && first_array == reused_array,
                  "2D-array base-slice cache preserves the selected pixels");
            CHECK(unmap(source_va, source_mapping_size, 0, 0, 0, 0) == 0,
                  "persistent BC test unmaps its guest source");
        }
    }
#endif

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
    LiveTargetSnapshot chained_producer_snapshot;
    CHECK(read_live_render_target(producer.color0_base, chained_producer_snapshot) &&
              chained_producer_snapshot.width == producer.color0_width &&
              chained_producer_snapshot.height == producer.color0_height &&
              chained_producer_snapshot.pixels &&
              chained_producer_snapshot.pixels->size() ==
                  static_cast<size_t>(producer.color0_width) * producer.color0_height * 4,
          "compute reader materializes a deferred GPU-only graphics target on demand");

    // A native 32x32 offscreen target receives a scissor that was globally scaled from 32 to 16
    // for the 128->64 presentation reduction above. The pass-local target correction must undo that
    // scale for scissors as well as viewports. Otherwise only the target's top-left 16x16 quadrant is
    // rendered (the exact collapse caught by the Dead Cells gameplay snapshot at render scale 4).
    DrawItem native_scissor = replay.items[0];
    native_scissor.color0_base = 0x4f0000;
    native_scissor.color0_width = 32;
    native_scissor.color0_height = 32;
    native_scissor.ps.has_scissor = true;
    native_scissor.ps.scissor_left = 0;
    native_scissor.ps.scissor_top = 0;
    native_scissor.ps.scissor_right = 16;
    native_scissor.ps.scissor_bottom = 16;
    std::vector<uint8_t> scissored = render_submit_items({native_scissor}, W, H);
    bool native_scissor_reaches_lower_right = scissored.size() == 32u * 32u * 4u;
    if (native_scissor_reaches_lower_right) {
        const uint8_t* lower_right = &scissored[(24u * 32u + 24u) * 4u];
        native_scissor_reaches_lower_right =
            lower_right[0] > 0xc0 && lower_right[1] < 0x40 && lower_right[2] < 0x40;
    }
    CHECK(native_scissor_reaches_lower_right,
          "native offscreen pass undoes the global scale for its guest scissor");

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
    const uint64_t producer_center = producer.color0_base +
        (static_cast<uint64_t>(producer.color0_width) + producer.color0_width / 2u) * 4u;
    std::vector<uint8_t> producer_range;
    CHECK(read_live_render_target_bytes(producer_center, 4, producer_range) ==
              LiveTargetByteReadResult::Success && producer_range.size() == 4,
          "live renderer exposes a bounded interior byte range from its authoritative target");
    std::vector<uint8_t> invalid_range;
    CHECK(read_live_render_target_bytes(
              producer.color0_base + producer.color0_width * producer.color0_height * 4u - 2u,
              4, invalid_range) == LiveTargetByteReadResult::InvalidRange,
          "live renderer rejects a source range that crosses the native target boundary");

    uint8_t copied_target[4] = {};
    GpuState ordered_state;
    GpuState::Draw semantic_producer;
    semantic_producer.command_order = 100;
    ordered_state.draws.push_back(semantic_producer);
    DrawItem ordered_producer = producer;
    ordered_producer.color0_base = 0x100300000ull;
    ordered_producer.draw_index = 0;
    ordered_producer.command_order = 100;
    const uint64_t ordered_center = ordered_producer.color0_base +
        (static_cast<uint64_t>(ordered_producer.color0_width) +
         ordered_producer.color0_width / 2u) * 4u;
    ordered_state.dma_copies.push_back({
        reinterpret_cast<uint64_t>(copied_target), ordered_center, sizeof copied_target,
        0, 200, 0});
    LiveRenderFn ordered_render = [](const std::vector<DrawItem>& items, uint32_t width,
                                     uint32_t height) {
        return render_submit_items(items, width, height);
    };
    execute_ordered_items(plan_submit_operations(ordered_state), {ordered_producer}, {},
                          ordered_state.dma_copies, ordered_render, {}, W, H);
    std::vector<uint8_t> ordered_range;
    const bool ordered_source_read = read_live_render_target_bytes(
        ordered_center, sizeof copied_target, ordered_range) == LiveTargetByteReadResult::Success;
    CHECK(ordered_source_read && ordered_range.size() == sizeof copied_target &&
          std::memcmp(copied_target, ordered_range.data(), sizeof copied_target) == 0,
          "ordered DMA copies current bytes produced by the preceding live-render span");

    DrawItem replay_producer = producer;
    replay_producer.color0_base = 0x100400000ull;
    replay_producer.draw_index = 0;
    replay_producer.command_order = 100;
    const uint64_t replay_center = replay_producer.color0_base +
        (static_cast<uint64_t>(replay_producer.color0_width) +
         replay_producer.color0_width / 2u) * 4u;
    uint8_t stale_captured_source[4] = {0x5a, 0x5a, 0x5a, 0x5a};
    uint8_t replay_copied_target[4] = {};
    ReplayDmaCopy replay_copy;
    replay_copy.dst = reinterpret_cast<uint64_t>(replay_copied_target);
    replay_copy.src = replay_center;
    replay_copy.bytes = sizeof replay_copied_target;
    replay_copy.command_order = 200;
    replay_copy.destination_data = replay_copied_target;
    replay_copy.destination_size = sizeof replay_copied_target;
    replay_copy.source_data = stale_captured_source;
    replay_copy.source_size = sizeof stale_captured_source;
    execute_ordered_items(
        {{SubmitOperationKind::Draw, 0, 100}, {SubmitOperationKind::DmaCopy, 0, 200}},
        {replay_producer}, {}, {replay_copy}, ordered_render, {}, W, H);
    std::vector<uint8_t> replay_range;
    const bool replay_source_read = read_live_render_target_bytes(
        replay_center, sizeof replay_copied_target, replay_range) ==
        LiveTargetByteReadResult::Success;
    CHECK(replay_source_read && replay_range.size() == sizeof replay_copied_target &&
          std::memcmp(replay_copied_target, replay_range.data(), sizeof replay_copied_target) == 0 &&
          std::memcmp(replay_copied_target, stale_captured_source,
                      sizeof replay_copied_target) != 0,
          "offline DMA replay prefers the preceding draw's live target over its captured blob");

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
    for (size_t i = 0; i < ds_seed.depth.size() / sizeof(float); ++i) {
        const float value = static_cast<float>(i + 1) /
                            static_cast<float>(ds_seed.depth.size() / sizeof(float) + 1);
        std::memcpy(ds_seed.depth.data() + i * sizeof(float), &value, sizeof(value));
    }
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

    // #1435: prove the live renderer uploads buffer bytes beyond the old 1 MiB clamp. The fragment
    // shader reads a raw dword through a direct V# at a scalar SOFFSET, then exports those float bits
    // as red. A below-clamp sentinel is the positive control; the 1.5 MiB sentinel turns robust-OOB
    // zero under the old truncated upload, so the framebuffer observes the defect end to end without
    // involving vertex fetch or geometry.
    {
        constexpr uint32_t low_offset = 0x80000u;
        constexpr uint32_t high_offset = 0x180000u;
        std::vector<uint32_t> buffer_words(high_offset / sizeof(uint32_t) + 1u, 0u);
        buffer_words[low_offset / sizeof(uint32_t)] = 0x3e800000u;   // 0.25f
        buffer_words[high_offset / sizeof(uint32_t)] = 0x3f400000u; // 0.75f

        ShaderResourceTable buffer_table;
        ShaderResource buffer{};
        buffer.cls = ResourceClass::ConstantBuffer;
        buffer.format = DataFormat::Float32;
        buffer.num_components = 1;
        buffer.binding = 32;
        buffer.sgpr_base = 0;
        buffer.gpu_addr = 0x10000000u;
        buffer.size = static_cast<uint32_t>(buffer_words.size() * sizeof(uint32_t));
        buffer.host_data = reinterpret_cast<uint8_t*>(buffer_words.data());
        buffer.host_data_size = buffer.size;
        buffer_table.resources.push_back(buffer);

        auto render_sentinel = [&](uint32_t byte_offset, uint64_t target_base) {
            const uint32_t buffer_ps[] = {
                0xbe8403ffu, byte_offset, // s_mov_b32 s4, byte_offset
                0xe0300000u, 0x04000000u, // buffer_load_dword v0, off, s[0:3], s4
                0x7e020280u,              // v_mov_b32 v1, 0
                0x7e040280u,              // v_mov_b32 v2, 0
                0x7e0602f2u,              // v_mov_b32 v3, 1.0
                0xf800000fu, 0x03020100u, // exp mrt0 v0, v1, v2, v3
                0xbf810000u,
            };
            DrawItem draw = replay.items[0];
            draw.fs = recompile_fragment(buffer_ps, std::size(buffer_ps), &buffer_table);
            draw.fs_shared.reset();
            draw.fs_identity = 0;
            draw.prt = std::make_shared<ShaderResourceTable>(buffer_table);
            draw.color0_base = target_base;
            if (draw.fs.empty()) return std::vector<uint8_t>{};
            return render_submit_items({draw}, W, H);
        };

        const std::vector<uint8_t> low_pixels = render_sentinel(low_offset, 0xd00000u);
        bool low_sentinel_visible = low_pixels.size() == static_cast<size_t>(W) * H * 4u;
        if (low_sentinel_visible) {
            const uint8_t* center =
                &low_pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4u];
            low_sentinel_visible = center[0] >= 56 && center[0] <= 72 &&
                                   center[1] < 8 && center[2] < 8;
        }
        CHECK(low_sentinel_visible,
              "fragment buffer read below 1 MiB exposes its distinct pixel sentinel");

        const std::vector<uint8_t> high_pixels = render_sentinel(high_offset, 0xd10000u);
        bool high_sentinel_visible = high_pixels.size() == static_cast<size_t>(W) * H * 4u;
        if (high_sentinel_visible) {
            const uint8_t* center =
                &high_pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4u];
            high_sentinel_visible = center[0] >= 184 && center[0] <= 200 &&
                                    center[1] < 8 && center[2] < 8;
        }
        CHECK(high_sentinel_visible,
              "fragment buffer read at 1.5 MiB survives the former upload clamp (#1435)");
    }

    // VideoOut's registered dimensions are authoritative even when CB_COLOR0_ATTRIB2 describes a
    // larger backing allocation. Treating this 256x256 declaration as the visible extent scales the
    // pass to 128x128 and prevents the final 64x64 scanout lookup from publishing the rendered frame.
    // Keep this last because publishing a registered front buffer intentionally makes it the source
    // selected by later callbacks.
    DrawItem oversized_scanout = replay.items[0];
    oversized_scanout.color0_base = reinterpret_cast<uint64_t>(scanout0.data());
    oversized_scanout.color0_width = PRESENT_W * 2;
    oversized_scanout.color0_height = PRESENT_H * 2;
    std::vector<uint8_t> scanout_pixels = render_submit_items({oversized_scanout}, W, H);
    CHECK(scanout_pixels.size() == static_cast<size_t>(W) * H * 4,
          "registered scanout uses the VideoOut extent instead of its oversized CB allocation");
    if (scanout_pixels.size() == static_cast<size_t>(W) * H * 4) {
        const uint8_t* center = &scanout_pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
        CHECK(center[0] > 0xC0 && center[1] < 0x40 && center[2] < 0x40,
              "oversized registered scanout still publishes its rendered pixels");
    }

    // #1427: a buffer binding must upload its whole declared range. A vertex fetch indexes anywhere
    // inside the descriptor, so a silently clamped upload makes every element past the clamp read
    // ZEROS — those vertices then transform to one clip point and the primitive dies as degenerate,
    // with no reject and no log. The old 1 MiB clamp did exactly that to 44 of 248 Blue Prince
    // entrance-hall draws (the tile floor, the far table, most of the room). Real vertex streams in
    // that scene run 1.4-3.0 MiB, and the draw that straddled the clamp lost precisely the vertices
    // whose offset exceeded it, so these sizes are the measured contract, not round numbers.
    {
        using prosper::frontend::buffer_upload_bytes;
        CHECK(buffer_upload_bytes(454024u) == 454024u,
              "a sub-megabyte vertex stream uploads whole (unchanged behavior)");
        CHECK(buffer_upload_bytes(1662044u) == 1662044u,
              "the 1.66 MiB stream that straddled the old clamp uploads whole (#1427)");
        CHECK(buffer_upload_bytes(2598440u) == 2598440u,
              "the 2.48 MiB stream that collapsed to one clip point uploads whole (#1427)");
        CHECK(buffer_upload_bytes(1u << 20) == (1u << 20),
              "the old clamp value itself is not a boundary any more");
        CHECK((buffer_upload_bytes(4098u) & 3u) == 0u &&
              buffer_upload_bytes(4098u) == 4096u,
              "uploads stay dword-aligned by truncation");
        CHECK(buffer_upload_bytes(0u) == 0u, "a zero declaration uploads nothing");
        // The ceiling still exists as a bound on a pathological descriptor; the caller reports any
        // binding that reaches it ([buffer-truncated]) instead of dropping the range in silence.
        CHECK(buffer_upload_bytes(0xFFFFFFFCu) == (64u << 20),
              "an absurd declaration is still bounded by the 64 MiB ceiling");
    }

    // A cache ceiling is not a preallocation. Scale capable hosts/devices to 4 GiB so one large
    // immutable atlas cannot evict the rest of a frame's hot set, while an 8 GiB machine retains the
    // historical 1 GiB bound. Explicit overrides remain exact for diagnostics and constrained hosts.
    {
        constexpr uint64_t GiB = 1024ull * 1024ull * 1024ull;
        using prosper::frontend::texture_decode_cache_limit_bytes;
        CHECK(texture_decode_cache_limit_bytes(nullptr, 8ull * GiB) == 1ull * GiB,
              "an 8 GiB host keeps the 1 GiB decoded-texture ceiling");
        CHECK(texture_decode_cache_limit_bytes(nullptr, 16ull * GiB) == 2ull * GiB,
              "a capable host admits the 2 GiB decoded-texture ceiling");
        CHECK(texture_decode_cache_limit_bytes(nullptr, 32ull * GiB) == 4ull * GiB,
              "a large host admits the 4 GiB decoded-texture ceiling");
        CHECK(texture_decode_cache_limit_bytes(nullptr, 128ull * GiB) == 4ull * GiB,
              "decoded-texture auto sizing remains capped at 4 GiB");
        CHECK(texture_decode_cache_limit_bytes("512", 128ull * GiB) == 512ull * 1024ull * 1024ull,
              "the decoded-texture MiB override takes precedence over host memory");
        CHECK(prosper::test::persistent_texture_cache_budget_for_heap(8ull * GiB) == 1ull * GiB,
              "an 8 GiB device keeps the 1 GiB sampled-image ceiling");
        CHECK(prosper::test::persistent_texture_cache_budget_for_heap(16ull * GiB) == 2ull * GiB,
              "a capable device admits the 2 GiB sampled-image ceiling");
        CHECK(prosper::test::persistent_texture_cache_budget_for_heap(32ull * GiB) == 4ull * GiB,
              "a large device admits the 4 GiB sampled-image ceiling");
        CHECK(prosper::test::persistent_texture_cache_budget_for_heap(128ull * GiB) == 4ull * GiB,
              "sampled-image auto sizing remains capped at 4 GiB");

        // Live exact-extent RTT consumers borrow an immutable snapshot instead of copying it
        // through frontend scratch. FrameResource copies retain that backing through upload setup.
        auto pixels = std::make_shared<const std::vector<uint8_t>>(16, 0x5a);
        prosper::test::FrameResource source;
        source.tex_rgba_owner = pixels;
        source.tex_rgba = pixels->data();
        prosper::test::FrameResource retained = source;
        source.tex_rgba_owner.reset();
        pixels.reset();
        CHECK(retained.tex_rgba_owner &&
                  retained.tex_rgba == retained.tex_rgba_owner->data() &&
                  retained.tex_rgba_owner->front() == 0x5a,
              "a copied FrameResource retains borrowed RTT pixels");

        using prosper::frontend::texture_decode_cache_candidate;
        CHECK(texture_decode_cache_candidate(false, false, false, 1u, true, true),
              "an ordinary supported guest 2D texture uses the decoded-texture cache");
        CHECK(texture_decode_cache_candidate(false, false, false, 2u, true, true),
              "a supported guest 3D volume uses the decoded-texture cache");
        CHECK(texture_decode_cache_candidate(false, false, false, 5u, true, true),
              "a supported guest 2D-array base slice uses the decoded-texture cache");
        CHECK(!texture_decode_cache_candidate(false, false, false, 3u, true, true),
              "a cube texture stays on its layer-aware decode path");
        CHECK(!texture_decode_cache_candidate(false, true, false, 1u, true, true),
              "a retained sampled-depth image bypasses guest-byte decode-cache work");
        CHECK(!texture_decode_cache_candidate(true, false, false, 1u, true, true),
              "a retained color image bypasses guest-byte decode-cache work");

        using prosper::frontend::texture_decode_source_address;
        CHECK(texture_decode_source_address(0x100000u, 5u, false, 0x24000u) == 0x124000u,
              "a 2D-array base slice decodes from its selected mip offset");
        CHECK(texture_decode_source_address(0x100000u, 5u, true, 0x24000u) == 0x100000u,
              "a packed 2D-array mip tail retains its shared block base");
        CHECK(texture_decode_source_address(0x100000u, 1u, false, 0x24000u) == 0x100000u,
              "an ordinary 2D texture keeps its descriptor base address");

        using prosper::frontend::texture_source_snapshot_can_follow_watch;
        CHECK(texture_source_snapshot_can_follow_watch(false, false, true, 4096u, 4096u),
              "an exact encoded snapshot becomes redundant behind an armed write watch");
        CHECK(!texture_source_snapshot_can_follow_watch(true, false, true, 0u, 4096u),
              "decoded pixels that are also the exact source baseline remain retained");
        CHECK(!texture_source_snapshot_can_follow_watch(false, true, true, 4096u, 4096u),
              "validation audit mode retains the exact source baseline");
        CHECK(!texture_source_snapshot_can_follow_watch(false, false, false, 4096u, 4096u),
              "an unsupported write watch keeps exact source validation available");
        CHECK(!texture_source_snapshot_can_follow_watch(false, false, true, 2048u, 4096u),
              "a partial source snapshot is never discarded as a complete baseline");

    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
