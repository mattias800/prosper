#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/guest_texture_layout.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/tile.hpp"
#include "../src/gpu/videoout_present.hpp"
#include "../src/hle/dispatch.hpp"
#include "../frontends/shared/live_compute.hpp"
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
#include "test_scratch.h"

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

constexpr bool mapped_readback_plan_satisfies_vuid_01389(
        prosper::test::MappedReadbackPlan plan, VkDeviceSize allocation,
        VkDeviceSize atom) {
    if (!plan.map_size || !plan.invalidate_size || !allocation || !atom)
        return false;
    const VkDeviceSize mapping_end =
        plan.map_size == VK_WHOLE_SIZE ? allocation : plan.map_size;
    if (mapping_end > allocation) return false;
    if (plan.invalidate_size == VK_WHOLE_SIZE)
        return mapping_end == allocation || mapping_end % atom == 0;
    return plan.invalidate_size <= mapping_end &&
           (plan.invalidate_size == allocation || plan.invalidate_size % atom == 0);
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

    // Graphics storage IMAGE_LOAD uses the same compact-guest -> raw-uvec4 converter as compute.
    // Prove its tiled source gate with the exact old failure shape: a 1x1 RGBA8 surface has only four
    // useful bytes but a 4 KiB tiled footprint. A 4095-byte capture used to be zero-extended and
    // detiled successfully because it exceeded the tight four-byte size; the detiler is entitled to
    // index the complete 4096 bytes, so only the full source may reach it.
    constexpr uint32_t STORAGE_TILE_MODE = 5u;
    const uint8_t storage_linear_red[4] = {255, 0, 0, 255};
    const size_t storage_tiled_bytes =
        prosper::frontend::storage_image_raw_uvec4_source_bytes(
            DataFormat::Unorm8, 4u, 1u, 1u, 1u, STORAGE_TILE_MODE, false, 0u);
    std::vector<uint8_t> storage_tiled(storage_tiled_bytes, 0);
    if (storage_tiled_bytes)
        tile_surface(storage_tiled.data(), storage_linear_red, 1u, 1u,
                     STORAGE_TILE_MODE, 0u, 4u);
    uint32_t storage_channels[4]{};
    const bool full_storage_materialized =
        prosper::frontend::storage_image_materialize_raw_uvec4(
            storage_tiled.data(), storage_tiled.size(), DataFormat::Unorm8, 4u,
            1u, 1u, 1u, STORAGE_TILE_MODE, false, 0u, 0u, 0u,
            storage_channels, std::size(storage_channels));
    float storage_red = 0.0f;
    std::memcpy(&storage_red, &storage_channels[0], sizeof(storage_red));
    CHECK(storage_tiled_bytes == 4096u && full_storage_materialized && storage_red == 1.0f,
          "raw-uvec4 converter consumes a complete tiled source footprint");
    CHECK(storage_tiled_bytes > 1u &&
              !prosper::frontend::storage_image_materialize_raw_uvec4(
                  storage_tiled.data(), storage_tiled.size() - 1u,
                  DataFormat::Unorm8, 4u, 1u, 1u, 1u, STORAGE_TILE_MODE,
                  false, 0u, 0u, 0u, storage_channels, std::size(storage_channels)),
          "raw-uvec4 converter rejects a short padded tiled backing before detile");

    // Exercise the submit-local identity map through live build_R, not a manually populated
    // FrameResource. Both references use the same compact guest backing and formatless UINT shader.
    // The first call converts 4 B/texel to 16 B/texel; the second must reuse those pixels together
    // with their R32G32B32A32_UINT interpretation. Losing that format on the hit makes the backend's
    // numeric-class gate reject the whole pass (#1713).
    const uint32_t storage_load_ps[] = {
        0x7e000280u, 0x7e020280u, 0xf0000f08u, 0x00020000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    auto make_storage_draw = [&](ShaderResource resource,
                                 const std::vector<uint32_t>& fragment,
                                 uint64_t target) {
        auto table = std::make_shared<ShaderResourceTable>();
        table->resources.push_back(resource);
        DrawItem draw = replay.items[0];
        draw.fs_shared.reset();
        draw.fs = fragment;
        draw.prt = std::move(table);
        draw.color0_base = target;
        draw.color0_width = 64;
        draw.color0_height = 64;
        return draw;
    };
    uint8_t compact_storage_texels[16] = {
        255, 0, 0, 255,   255, 0, 0, 255,
        255, 0, 0, 255,   255, 0, 0, 255,
    };
    ShaderResource raw_storage{};
    raw_storage.cls = ResourceClass::StorageImage;
    raw_storage.format = DataFormat::Unorm8;
    raw_storage.num_components = 4;
    raw_storage.binding = 4;
    raw_storage.img_dim = 1;
    raw_storage.width = raw_storage.height = 2;
    raw_storage.depth = 1;
    raw_storage.sgpr_base = 8;
    raw_storage.gpu_addr = 0x17130000u;
    raw_storage.size = sizeof(compact_storage_texels);
    raw_storage.linear_row_pitch_bytes = 8;
    raw_storage.host_data = compact_storage_texels;
    raw_storage.host_data_size = sizeof(compact_storage_texels);
    ShaderResourceTable raw_storage_compile;
    raw_storage_compile.resources.push_back(raw_storage);
    const std::vector<uint32_t> raw_storage_frag = recompile_fragment(
        storage_load_ps, std::size(storage_load_ps), &raw_storage_compile);
    DrawItem raw_storage_draw = make_storage_draw(
        raw_storage, raw_storage_frag, 0x17131000u);
    prosper::frontend::reset_texture_decode_scope_stats();
    render_submit_items({raw_storage_draw, raw_storage_draw}, W, H);
    const auto raw_storage_reuse = prosper::frontend::texture_decode_scope_stats();
    LiveTargetSnapshot raw_storage_snapshot;
    const bool raw_storage_published = read_live_render_target(
        raw_storage_draw.color0_base, raw_storage_snapshot);
    const uint8_t* raw_storage_center = raw_storage_published &&
        raw_storage_snapshot.pixels &&
        raw_storage_snapshot.pixels->size() == static_cast<size_t>(W) * H * 4u
            ? &(*raw_storage_snapshot.pixels)[
                  (static_cast<size_t>(H / 2) * W + W / 2) * 4u]
            : nullptr;
    CHECK(raw_storage_reuse.decodes == 1u &&
              raw_storage_reuse.same_span_reuses == 1u,
          "two same-span storage references execute one production conversion and one cache hit");
    CHECK(raw_storage_center && raw_storage_center[0] > 0xc0 &&
              raw_storage_center[1] < 0x40 && raw_storage_center[2] < 0x40,
          "submit-local raw-uvec4 reuse restores its UINT Vulkan representation");

    // One guest descriptor may be reflected as either formatless raw-uvec4 or exact R32ui. Patch
    // only OpTypeImage's Image Format in a copy of the recompiled read shader, leaving both runtime
    // T# records and both read-only access modes identical. They must remain two decode identities:
    // the former stores 16 host bytes/texel and the latter four.
    uint32_t shared_storage_word = 0x3f800000u;
    ShaderResource shared_storage = raw_storage;
    shared_storage.format = DataFormat::Uint32;
    shared_storage.num_components = 1;
    shared_storage.width = shared_storage.height = 1;
    shared_storage.size = sizeof(shared_storage_word);
    shared_storage.linear_row_pitch_bytes = sizeof(shared_storage_word);
    shared_storage.host_data = reinterpret_cast<uint8_t*>(&shared_storage_word);
    shared_storage.host_data_size = sizeof(shared_storage_word);
    ShaderResourceTable shared_storage_compile;
    shared_storage_compile.resources.push_back(shared_storage);
    std::vector<uint32_t> formatless_storage_frag = recompile_fragment(
        storage_load_ps, std::size(storage_load_ps), &shared_storage_compile);
    std::vector<uint32_t> r32ui_storage_frag = formatless_storage_frag;
    bool patched_r32ui = false;
    for (size_t i = 5; i < r32ui_storage_frag.size();) {
        const uint32_t word_count = r32ui_storage_frag[i] >> 16;
        const uint32_t opcode = r32ui_storage_frag[i] & 0xffffu;
        if (!word_count || i + word_count > r32ui_storage_frag.size()) break;
        if (opcode == 25u && word_count >= 9u) { // OpTypeImage
            r32ui_storage_frag[i + 8u] = kSpirvImageFormatR32ui;
            patched_r32ui = true;
            break;
        }
        i += word_count;
    }
    const DescriptorValidationReport r32ui_interface =
        validate_spirv_descriptor_interface(
            r32ui_storage_frag, &shared_storage_compile, 1,
            SpirvShaderStage::Fragment, false);
    const SpirvDescriptorBinding* r32ui_binding =
        find_spirv_descriptor_binding(r32ui_interface, 1, shared_storage.binding);
    CHECK(patched_r32ui && r32ui_binding &&
              r32ui_binding->storage_image_format == kSpirvImageFormatR32ui,
          "typed-read fixture changes the reflected storage representation lever");
    DrawItem formatless_draw = make_storage_draw(
        shared_storage, formatless_storage_frag, 0x17132000u);
    DrawItem r32ui_draw = make_storage_draw(
        shared_storage, r32ui_storage_frag, 0x17132000u);
    prosper::frontend::reset_texture_decode_scope_stats();
    render_submit_items({formatless_draw, r32ui_draw}, W, H);
    const auto storage_representation_split =
        prosper::frontend::texture_decode_scope_stats();
    CHECK(storage_representation_split.decodes == 2u &&
              storage_representation_split.same_span_reuses == 0u,
          "formatless raw-uvec4 and exact R32ui reads cannot alias one submit-local decode");

    // Finally route the short 4 KiB tiled capture through build_R. The source contains the useful
    // 1x1 red texel, so the pre-fix zero-padding path rendered red; absence of this unique target plus
    // a recorded decode proves the production converter ran and failed the materialization contract
    // before Vulkan accepted the descriptor.
    ShaderResource short_tiled_storage = raw_storage;
    short_tiled_storage.width = short_tiled_storage.height = 1;
    short_tiled_storage.tile_mode = STORAGE_TILE_MODE;
    short_tiled_storage.size = static_cast<uint32_t>(storage_tiled_bytes);
    short_tiled_storage.linear_row_pitch_bytes = 0;
    short_tiled_storage.host_data = storage_tiled.data();
    short_tiled_storage.host_data_size = storage_tiled.size() - 1u;
    ShaderResourceTable short_storage_compile;
    short_storage_compile.resources.push_back(short_tiled_storage);
    const std::vector<uint32_t> short_storage_frag = recompile_fragment(
        storage_load_ps, std::size(storage_load_ps), &short_storage_compile);
    constexpr uint64_t SHORT_STORAGE_TARGET = 0x17133000u;
    prosper::frontend::reset_texture_decode_scope_stats();
    render_submit_items(
        {make_storage_draw(short_tiled_storage, short_storage_frag, SHORT_STORAGE_TARGET)}, W, H);
    const auto short_storage_stats = prosper::frontend::texture_decode_scope_stats();
    LiveTargetSnapshot short_storage_snapshot;
    CHECK(short_storage_stats.decodes == 1u &&
              !read_live_render_target(SHORT_STORAGE_TARGET, short_storage_snapshot),
          "live storage materialization fails closed on a short padded tiled backing");

    // R-Type Delta's movie shader passes pixel coordinates and sets S# FORCE_UNNORMALIZED. Vulkan's
    // native unnormalized sampler cannot preserve the title's wrap/LOD state, so the recompiler
    // divides only spatial coordinates by their resource extents. At texel-space (1.5,0.5) the 2x2
    // source is green; dropping that transform makes the ordinary normalized sampler clamp to the
    // bottom-right white texel. The named check therefore pins rendered behavior, not metadata.
    {
        const uint32_t unnormalized_ps_rdna[] = {
            0x7e0002ffu, 0x3fc00000u, // v0 = 1.5 texels
            0x7e0202ffu, 0x3f000000u, // v1 = 0.5 texels
            0xf0800f08u, 0x00820000u, // image_sample v[0:3], v[0:1], s8, s16
            0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        DrawItem unnormalized_draw = replay.items[0];
        auto unnormalized_table =
            std::make_shared<ShaderResourceTable>(*unnormalized_draw.prt);
        ShaderResource& unnormalized_texture = unnormalized_table->resources[0];
        unnormalized_texture.unnormalized = 1;
        unnormalized_texture.mag_filter = unnormalized_texture.min_filter = 0;
        unnormalized_texture.mip_filter = 0;
        unnormalized_texture.addr_uvw[0] = unnormalized_texture.addr_uvw[1] = 2;
        unnormalized_texture.min_lod = unnormalized_texture.max_lod =
            unnormalized_texture.lod_bias = 0.0f;
        unnormalized_texture.max_aniso_ratio = 0;
        unnormalized_draw.fs = recompile_fragment(
            unnormalized_ps_rdna, std::size(unnormalized_ps_rdna),
            unnormalized_table.get());
        unnormalized_draw.prt = std::move(unnormalized_table);
        const std::vector<uint8_t> unnormalized_pixels =
            render_submit_items({unnormalized_draw}, W, H);
        const uint8_t* center =
            unnormalized_pixels.size() == static_cast<size_t>(W) * H * 4
                ? &unnormalized_pixels[
                    (static_cast<size_t>(H / 2) * W + W / 2) * 4]
                : nullptr;
        CHECK(center && center[1] > 0xC0 && center[0] < 0x40 && center[2] < 0x40,
              "unnormalized sampler preserves texel-space coordinates through the live renderer");
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

        // Extended AvPlayer frames retain 1920 visible bytes inside a 2048-byte physical row.
        // Tactics Ogre samples the interleaved UV plane with (X,X,X,Y), so V reaches alpha. Prove
        // both the HLE layout-provenance arm and the padded native-RG8 row copy: a contiguous copy
        // would sample the zero padding as the second row, while the legacy coverage broadcast
        // would replace alpha with U.
        uint8_t padded_rg_texels[16] = {
            32, 224, 32, 224, 0, 0, 0, 0,
            32, 224, 32, 224, 0, 0, 0, 0,
        };
        DrawItem registered_padded_draw = rg_draw;
        registered_padded_draw.color0_base = 0x211000;
        auto registered_padded_table =
            std::make_shared<ShaderResourceTable>(*registered_padded_draw.prt);
        registered_padded_table->resources.resize(1);
        ShaderResource& registered_rg = registered_padded_table->resources[0];
        registered_rg.gpu_addr = 0x130000;
        registered_rg.size = 8;
        registered_rg.linear_row_pitch_bytes = 8;
        registered_rg.host_data = padded_rg_texels;
        registered_rg.host_data_size = sizeof(padded_rg_texels);
        registered_rg.swizzle[0] = 4; registered_rg.swizzle[1] = 4;
        registered_rg.swizzle[2] = 4; registered_rg.swizzle[3] = 5;
        registered_padded_draw.prt = std::move(registered_padded_table);
        register_guest_linear_texture_layout(
            registered_rg.gpu_addr, sizeof(padded_rg_texels), 8);
        const std::vector<uint8_t> registered_padded_pixels =
            render_submit_items({registered_padded_draw}, W, H);
        unregister_guest_linear_texture_layout(registered_rg.gpu_addr);
        bool registered_padded_preserved =
            registered_padded_pixels.size() == static_cast<size_t>(W) * H * 4;
        if (registered_padded_preserved) {
            const uint8_t* center = &registered_padded_pixels[
                (static_cast<size_t>(H / 2) * W + W / 2) * 4];
            registered_padded_preserved =
                center[0] >= 24 && center[0] <= 40 &&
                center[1] >= 24 && center[1] <= 40 &&
                center[2] >= 24 && center[2] <= 40 &&
                center[3] >= 216 && center[3] <= 232;
        }
        CHECK(registered_padded_preserved,
              "registered padded NV12 chroma preserves V through an alpha swizzle");

        // Captures cannot retain the process-local HLE registry. Recover only the same narrow
        // contract from a matching full-resolution luma plane immediately before the chroma plane.
        DrawItem captured_padded_draw = registered_padded_draw;
        captured_padded_draw.color0_base = 0x212000;
        auto captured_padded_table =
            std::make_shared<ShaderResourceTable>(*captured_padded_draw.prt);
        uint8_t padded_luma_texels[32] = {};
        ShaderResource padded_luma = captured_padded_table->resources[0];
        padded_luma.binding = 5;
        padded_luma.gpu_addr = registered_rg.gpu_addr - sizeof(padded_luma_texels);
        padded_luma.num_components = 1;
        padded_luma.width = padded_luma.height = 4;
        padded_luma.size = 16;
        padded_luma.linear_row_pitch_bytes = 8;
        padded_luma.host_data = padded_luma_texels;
        padded_luma.host_data_size = sizeof(padded_luma_texels);
        captured_padded_table->resources.push_back(padded_luma);
        captured_padded_draw.prt = std::move(captured_padded_table);
        const std::vector<uint8_t> captured_padded_pixels =
            render_submit_items({captured_padded_draw}, W, H);
        bool captured_padded_preserved =
            captured_padded_pixels.size() == static_cast<size_t>(W) * H * 4;
        if (captured_padded_preserved) {
            const uint8_t* center = &captured_padded_pixels[
                (static_cast<size_t>(H / 2) * W + W / 2) * 4];
            captured_padded_preserved =
                center[0] >= 24 && center[0] <= 40 &&
                center[1] >= 24 && center[1] <= 40 &&
                center[2] >= 24 && center[2] <= 40 &&
                center[3] >= 216 && center[3] <= 232;
        }
        CHECK(captured_padded_preserved,
              "captured padded NV12 adjacency preserves V through an alpha swizzle");
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
        const std::filesystem::path dump_dir = prosper_test::test_scratch_dir() /
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
            auto mapped_reader = [&](uint64_t addr, uint8_t* dst, size_t bytes) -> size_t {
                if (addr < source_va) return 0;
                const uint64_t offset = addr - source_va;
                if (offset > source_mapping_size || bytes > source_mapping_size - offset)
                    return 0;
                std::memcpy(dst, reinterpret_cast<const void*>(addr), bytes);
                return bytes;
            };
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
            array_resource.linear_row_pitch_bytes = 0;
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

            GpuCaptureFile array_capture;
            error.clear();
            const bool array_captured = capture_draw_items(
                {array_draw}, meta, mapped_reader, array_capture, error);
            CHECK(array_captured &&
                      array_capture.draws[0].prt.resources[0]
                              .resource.linear_row_pitch_bytes == array_row_pitch,
                  "capture derives the aligned row pitch from a normal dim-5 descriptor");
            GpuReplayFrame array_replay;
            const bool array_materialized = array_captured &&
                materialize_gpu_replay(array_capture, array_replay, error);
            const std::vector<uint8_t> replayed_array = array_materialized
                ? render_submit_items(array_replay.items, W, H) : std::vector<uint8_t>{};
            CHECK(array_materialized &&
                      array_replay.items[0].prt->resources[0].linear_row_pitch_bytes ==
                          array_row_pitch &&
                      replayed_array == first_array,
                  "captured dim-5 base slice replays the same padded selected rows");

            // BC levels apply the same row alignment to their block grid. Keep the two useful
            // 32-byte rows 256 bytes apart so a tight block copy sees zero padding instead of row 1.
            uint8_t* bc_array_base = reinterpret_cast<uint8_t*>(source_va + 0x4000);
            uint8_t* bc_array_selected = bc_array_base + 0x400;
            uint8_t bc3_green[sizeof(bc3_red)];
            std::memcpy(bc3_green, bc3_red, sizeof(bc3_red));
            bc3_green[8] = 0xe0;
            bc3_green[9] = 0x07;
            for (uint32_t block = 0; block < 2; ++block) {
                std::memcpy(bc_array_selected + block * sizeof(bc3_red),
                            bc3_red, sizeof(bc3_red));
                std::memcpy(bc_array_selected + array_row_pitch +
                                block * sizeof(bc3_green),
                            bc3_green, sizeof(bc3_green));
            }
            DrawItem bc_array_draw = bc_draw;
            bc_array_draw.color0_base = 0xda0000;
            auto bc_array_table =
                std::make_shared<ShaderResourceTable>(*bc_array_draw.prt);
            ShaderResource& bc_array_resource = bc_array_table->resources[0];
            bc_array_resource.gpu_addr = reinterpret_cast<uint64_t>(bc_array_base);
            bc_array_resource.size = 2u * 0x1000u;
            bc_array_resource.width = bc_array_resource.height = 8;
            bc_array_resource.depth = 2;
            bc_array_resource.img_dim = 5;
            bc_array_resource.tile_mode = 0;
            bc_array_resource.layer_stride_bytes = 0x1000;
            bc_array_resource.layer_mip_offset_bytes = 0x400;
            bc_array_resource.linear_row_pitch_bytes = 0;
            bc_array_draw.prt = std::move(bc_array_table);
            const uint32_t ps_rdna_lower[] = {
                0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3f400000u,
                0xf0800f08u, 0x00820000u, 0xf800000fu, 0x03020100u,
                0xbf810000u,
            };
            DrawItem bc_array_lower_draw = bc_array_draw;
            bc_array_lower_draw.color0_base = 0xdb0000;
            bc_array_lower_draw.fs = recompile_fragment(
                ps_rdna_lower, std::size(ps_rdna_lower), bc_array_lower_draw.prt.get());
            const std::vector<uint8_t> first_bc_array =
                render_submit_items({bc_array_draw}, W, H);
            const std::vector<uint8_t> first_bc_array_lower =
                render_submit_items({bc_array_lower_draw}, W, H);
            const std::vector<uint8_t> reused_bc_array =
                render_submit_items({bc_array_draw}, W, H);
            auto pixel_at = [&](const std::vector<uint8_t>& image,
                                uint32_t x, uint32_t y) -> const uint8_t* {
                return image.size() != static_cast<size_t>(W) * H * 4u ? nullptr
                    : &image[(static_cast<size_t>(y) * W + x) * 4u];
            };
            const auto red = [](const uint8_t* pixel) {
                return pixel && pixel[0] > 192 && pixel[1] < 64 && pixel[2] < 64;
            };
            const auto green = [](const uint8_t* pixel) {
                return pixel && pixel[0] < 64 && pixel[1] > 192 && pixel[2] < 64;
            };
            CHECK(!bc_array_lower_draw.fs.empty() &&
                      red(pixel_at(first_bc_array, W / 2, H / 2)) &&
                      green(pixel_at(first_bc_array_lower, W / 2, H / 2)) &&
                      reused_bc_array == first_bc_array,
                  "linear BC array decodes and caches both padded block rows");
            GpuCaptureFile bc_array_capture;
            error.clear();
            const bool bc_array_captured = capture_draw_items(
                {bc_array_lower_draw}, meta, mapped_reader, bc_array_capture, error);
            GpuReplayFrame bc_array_replay;
            const bool bc_array_materialized = bc_array_captured &&
                materialize_gpu_replay(bc_array_capture, bc_array_replay, error);
            const std::vector<uint8_t> replayed_bc_array = bc_array_materialized
                ? render_submit_items(bc_array_replay.items, W, H) : std::vector<uint8_t>{};
            CHECK(bc_array_materialized &&
                      bc_array_capture.draws[0].prt.resources[0]
                              .resource.linear_row_pitch_bytes == array_row_pitch &&
                      replayed_bc_array == first_bc_array_lower,
                  "captured BC array preserves its padded block-row pitch and pixels");

            // A live render target at the selected nonzero mip address is authoritative over the
            // unchanged guest bytes. Re-render it with a new color and prove the array consumer does
            // not reuse the prior frame from the guest-keyed persistent decode cache.
            const uint8_t producer_red[16] = {
                255, 0, 0, 255, 255, 0, 0, 255,
                255, 0, 0, 255, 255, 0, 0, 255,
            };
            const uint8_t producer_green[16] = {
                0, 255, 0, 255, 0, 255, 0, 255,
                0, 255, 0, 255, 0, 255, 0, 255,
            };
            DrawItem array_rtt_producer = replay.items[0];
            array_rtt_producer.color0_base =
                reinterpret_cast<uint64_t>(array_selected);
            array_rtt_producer.color0_width = array_rtt_producer.color0_height = 2;
            auto producer_table =
                std::make_shared<ShaderResourceTable>(*array_rtt_producer.prt);
            ShaderResource& producer_resource = producer_table->resources[0];
            producer_resource.gpu_addr = 0xf10000;
            producer_resource.host_data = const_cast<uint8_t*>(producer_red);
            producer_resource.host_data_size = sizeof(producer_red);
            producer_resource.size = sizeof(producer_red);
            producer_resource.width = producer_resource.height = 2;
            producer_resource.depth = 1;
            producer_resource.img_dim = 1;
            producer_resource.tile_mode = 0;
            producer_resource.linear_row_pitch_bytes = 8;
            producer_resource.format = DataFormat::Unorm8;
            producer_resource.num_components = 4;
            array_rtt_producer.prt = std::move(producer_table);
            render_submit_items({array_rtt_producer}, W, H);
            const std::vector<uint8_t> red_array_rtt =
                render_submit_items({array_draw}, W, H);
            producer_resource.host_data = const_cast<uint8_t*>(producer_green);
            producer_resource.host_data_size = sizeof(producer_green);
            render_submit_items({array_rtt_producer}, W, H);
            const std::vector<uint8_t> green_array_rtt =
                render_submit_items({array_draw}, W, H);
            const uint8_t* red_center = pixel_at(red_array_rtt, W / 2, H / 2);
            const uint8_t* green_center = pixel_at(green_array_rtt, W / 2, H / 2);
            CHECK(red(red_center) && green(green_center) &&
                      red_array_rtt != green_array_rtt,
                  "dim-5 selected-address RTT sampling follows renderer-only pixel changes");
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

    // #1717: this checkpoint is the exact defect shape on RADV: 4x3 D32S8 is 48 depth + 12
    // stencil = 60 logical bytes, VkMemoryRequirements rounds the allocation to 64, and the device's
    // nonCoherentAtomSize is 64. Invalidating VK_WHOLE_SIZE after mapping only 60 bytes violates
    // VUID 01389; mapping the whole allocation makes its end equal the memory-object end.
    {
        using prosper::test::MappedReadbackPlan;
        using prosper::test::mapped_readback_plan;
        constexpr VkDeviceSize logical_bytes = 60;
        constexpr VkDeviceSize allocation_bytes = 64;
        constexpr VkDeviceSize atom_bytes = 64;
        constexpr MappedReadbackPlan old_defect{logical_bytes, VK_WHOLE_SIZE};
        constexpr MappedReadbackPlan fixed = mapped_readback_plan(logical_bytes);
        CHECK(!mapped_readback_plan_satisfies_vuid_01389(
                  old_defect, allocation_bytes, atom_bytes) &&
                  mapped_readback_plan_satisfies_vuid_01389(
                      fixed, allocation_bytes, atom_bytes),
              "60-byte D32S8 readback maps through the 64-byte allocation end before invalidation");
    }

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

        // #1284: the host-buffer staging pool was a flat 256 MiB against a measured 974 MiB working
        // set on Blue Prince, so it evicted one buffer for every one it created. Size it from host
        // RAM instead. The floor is the historical default, so no host is given LESS than before.
        {
            constexpr uint64_t MiB = 1024ull * 1024ull;
            using prosper::test::render_host_buffer_pool_limit_bytes;
            CHECK(render_host_buffer_pool_limit_bytes(nullptr, 0) == 256ull * MiB,
                  "an unknown host size falls back to the historical 256 MiB pool floor");
            CHECK(render_host_buffer_pool_limit_bytes(nullptr, 1ull * GiB) == 256ull * MiB,
                  "a small host is clamped up to the 256 MiB pool floor, never below it");
            CHECK(render_host_buffer_pool_limit_bytes(nullptr, 8ull * GiB) == 1024ull * MiB,
                  "an 8 GiB host admits a 1 GiB pool, which covers the measured 974 MiB set");
            CHECK(render_host_buffer_pool_limit_bytes(nullptr, 16ull * GiB) == 2048ull * MiB,
                  "a 16 GiB host reaches the 2 GiB pool ceiling");
            CHECK(render_host_buffer_pool_limit_bytes(nullptr, 128ull * GiB) == 2048ull * MiB,
                  "pool auto sizing remains capped at 2 GiB");
            CHECK(render_host_buffer_pool_limit_bytes("256", 128ull * GiB) == 256ull * MiB,
                  "the pool MiB override takes precedence over host memory");
            CHECK(render_host_buffer_pool_limit_bytes("64", 128ull * GiB) == 64ull * MiB,
                  "the override may go BELOW the floor — it is the constrained-host escape hatch "
                  "and the A/B lever, so it must be exact");
            CHECK(render_host_buffer_pool_limit_bytes("0", 8ull * GiB) == 0,
                  "an explicit zero disables pool retention rather than being clamped up");
        }

        // #1284: eviction must pick the least-recently-released buffer, not an arbitrary hash
        // bucket. Below any budget smaller than the working set the old policy discarded whichever
        // capacity class the map happened to order first — very often the one about to be reused —
        // so raising the budget alone would leave the same failure waiting for a larger title.
        //
        // Populated so that the oldest entry is deliberately NOT in the first-inserted class and NOT
        // in the smallest or largest class: an arbitrary-bucket policy has no reason to select 8192,
        // so this fails without the fix rather than passing by construction.
        {
            using prosper::test::RenderHostBuffer;
            using prosper::test::RenderHostBufferPool;
            using prosper::test::render_host_buffer_pool_lru_key;

            RenderHostBufferPool pool;
            auto add = [&pool](VkDeviceSize capacity, uint64_t stamp) {
                RenderHostBuffer entry;
                entry.bytes = capacity;
                entry.allocation_bytes = capacity;
                entry.last_use = stamp;
                pool.available[capacity].push_back(entry);
            };
            add(4096, 40);
            add(8192, 10);          // oldest overall, middle capacity class
            add(65536, 25);
            add(1024, 33);

            VkDeviceSize victim = 0;
            CHECK(render_host_buffer_pool_lru_key(pool, victim),
                  "a populated pool yields an eviction victim");
            CHECK(victim == 8192,
                  "eviction selects the least-recently-released capacity class (8192)");

            // Within a class the front is the oldest, so a newer release must not shadow it.
            RenderHostBuffer newer;
            newer.bytes = 8192;
            newer.allocation_bytes = 8192;
            newer.last_use = 99;
            pool.available[8192].push_back(newer);
            CHECK(render_host_buffer_pool_lru_key(pool, victim) && victim == 8192,
                  "a later release into the same class does not hide that class's oldest entry");

            // Draining the oldest class hands the victim to the next-oldest, not back to the front
            // of the map.
            pool.available.erase(8192);
            CHECK(render_host_buffer_pool_lru_key(pool, victim) && victim == 65536,
                  "with the oldest class drained the next-oldest (65536) is selected");

            // Empty classes are skipped rather than selected with a stale zero stamp.
            pool.available[16384] = {};
            CHECK(render_host_buffer_pool_lru_key(pool, victim) && victim == 65536,
                  "an empty capacity class is not a victim despite its zero-valued front");

            RenderHostBufferPool empty;
            CHECK(!render_host_buffer_pool_lru_key(empty, victim),
                  "an empty pool reports no victim instead of dereferencing a front");
        }

        // Live exact-extent RTT consumers borrow an immutable snapshot instead of copying it
        // through frontend scratch. FrameResource copies retain that backing through upload setup.
        constexpr uint32_t retained_width = 2;
        constexpr uint32_t retained_height = 2;
        constexpr VkFormat retained_format = VK_FORMAT_R16G16B16A16_SFLOAT;
        const size_t retained_bytes = static_cast<size_t>(retained_width) * retained_height *
            prosper::test::backend_color_bytes_per_pixel(retained_format);
        auto pixels = std::make_shared<const std::vector<uint8_t>>(retained_bytes, 0x5a);
        prosper::test::FrameResource source;
        source.tex_rgba_owner = pixels;
        source.tex_rgba = pixels->data();
        source.tw = retained_width;
        source.th = retained_height;
        source.texture_format = retained_format;
        prosper::test::FrameResource retained = source;
        source.tex_rgba_owner.reset();
        pixels.reset();
        CHECK(retained.tex_rgba_owner &&
                  retained.tex_rgba == retained.tex_rgba_owner->data() &&
                  retained.tex_rgba_owner->size() == retained_bytes &&
                  retained.tw == retained_width && retained.th == retained_height &&
                  retained.texture_format == retained_format &&
                  retained.tex_rgba_owner->front() == 0x5a,
              "a copied FrameResource retains the borrowed RTT lifetime, extent, and FP16 format");

        using prosper::frontend::texture_decode_cache_candidate;
        CHECK(texture_decode_cache_candidate(false, false, false, 1u, true, true, false),
              "an ordinary supported guest 2D texture uses the decoded-texture cache");
        CHECK(texture_decode_cache_candidate(false, false, false, 2u, true, true, false),
              "a supported guest 3D volume uses the decoded-texture cache");
        CHECK(texture_decode_cache_candidate(false, false, false, 5u, true, true, false),
              "a supported guest 2D-array base slice uses the decoded-texture cache");
        CHECK(texture_decode_cache_candidate(false, false, false, 3u, true, true, true),
              "a block-compressed cube uses the exact decoded-texture cache");
        CHECK(!texture_decode_cache_candidate(false, false, false, 3u, true, true, false),
              "a non-block-compressed cube stays on its layer-aware uncached path");
        CHECK(!texture_decode_cache_candidate(false, true, false, 1u, true, true, false),
              "a retained sampled-depth image bypasses guest-byte decode-cache work");
        CHECK(!texture_decode_cache_candidate(true, false, false, 1u, true, true, false),
              "a retained color image bypasses guest-byte decode-cache work");

        using prosper::frontend::block_compressed_cube_source_size;
        constexpr uint64_t syberia_cube_address = 0x212bd41000ull;
        constexpr uint64_t syberia_cube_footprint = 33570816ull;
        CHECK(block_compressed_cube_source_size(
                  true, syberia_cube_address, syberia_cube_footprint) ==
                  syberia_cube_footprint,
              "Syberia's six-face BC6H footprint becomes the exact persistent source span");
        CHECK(block_compressed_cube_source_size(
                  false, syberia_cube_address, syberia_cube_footprint) == 0 &&
                  block_compressed_cube_source_size(true, syberia_cube_address, 0) == 0 &&
                  block_compressed_cube_source_size(
                      true, UINT64_MAX - syberia_cube_footprint + 1u,
                      syberia_cube_footprint) == 0,
              "non-BC, empty, and overflowing cube source ranges remain cache-ineligible");

        using prosper::frontend::sampled_msaa_fetch_shape_supported;
        ShaderResource exact_msaa{};
        exact_msaa.cls = ResourceClass::Texture;
        exact_msaa.img_dim = 6;
        exact_msaa.sample_count = 4;
        exact_msaa.tile_mode = 24;
        exact_msaa.format = DataFormat::Float32;
        exact_msaa.num_components = 1;
        exact_msaa.depth = 1;
        exact_msaa.declared_mip_levels = 1;
        CHECK(sampled_msaa_fetch_shape_supported(
                  exact_msaa, /*is_storage_image=*/false,
                  /*reflected_msaa_fetch=*/true),
              "the exact reflected 4x R32F 2D_MSAA shape reaches materialization");
        ShaderResource wrong_msaa_format = exact_msaa;
        wrong_msaa_format.format = DataFormat::Uint32;
        ShaderResource wrong_msaa_count = exact_msaa;
        wrong_msaa_count.sample_count = 2;
        ShaderResource wrong_msaa_layout = exact_msaa;
        wrong_msaa_layout.tile_mode = 27;
        CHECK(!sampled_msaa_fetch_shape_supported(
                   wrong_msaa_format, false, true) &&
                  !sampled_msaa_fetch_shape_supported(
                   wrong_msaa_count, false, true) &&
                  !sampled_msaa_fetch_shape_supported(
                   wrong_msaa_layout, false, true) &&
                  !sampled_msaa_fetch_shape_supported(
                   exact_msaa, true, true) &&
                  !sampled_msaa_fetch_shape_supported(
                   exact_msaa, false, false),
              "wrong MSAA format, count, layout, storage class, and reflection remain fail-visible");

        using prosper::frontend::persistent_texture_decode_cache_eligible;
        CHECK(persistent_texture_decode_cache_eligible(
                  /*guest_decode_candidate=*/true, /*compute_image_hit=*/false,
                  /*is_storage_image=*/false, /*cache_disabled=*/false,
                  /*compression_supported=*/true, /*cache_limit=*/1u << 30,
                  /*source_size=*/retained_bytes),
              "an ordinary supported guest texture remains eligible for persistent decode caching");
        CHECK(!persistent_texture_decode_cache_eligible(
                   texture_decode_cache_candidate(
                       /*has_live_color_target=*/true, /*has_live_depth_target=*/false,
                       /*has_captured_host_data=*/false, /*image_dimension=*/1u,
                       /*is_sampled_texture=*/true, /*format_supported=*/true,
                       /*block_compressed=*/false),
                   /*compute_image_hit=*/false, /*is_storage_image=*/false,
                   /*cache_disabled=*/false, /*compression_supported=*/true,
                   /*cache_limit=*/1u << 30,
                   // DCC metadata can make this nonzero even though live color is authoritative.
                   /*source_size=*/retained_bytes),
              "a live FP16 RTT cannot be overwritten by a DCC-backed guest decode-cache entry");

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

    // #1691 — the decoded-texture identity map is SUBMIT-scoped, not span-scoped. A submit is cut
    // into a new graphics span at every interleaved compute/DMA operation, so a span-scoped map
    // re-resolved every identity once per span (Blue Prince: 56 identities, 853 resolves, 15.2x).
    // Widening the lifetime is only sound while an entry that crosses a span boundary carries proof
    // that nothing rewrote the bytes it decoded — the failure this guards is #780's shape, where a
    // compute reset of a backing invalidated one cached view and not another.
    {
        using prosper::frontend::reset_texture_decode_scope_stats;
        using prosper::frontend::submit_local_texture_decode_reusable;
        using prosper::frontend::texture_decode_scope_stats;

        constexpr uint64_t SRC = 0x40000, LEN = 64;
        CHECK(submit_local_texture_decode_reusable(9, 9, SRC, LEN, SRC, LEN,
                                                   GuestGpuWriteQuery::Unknown),
              "reuse inside the span that decoded the pixels keeps the historical guarantee");
        CHECK(submit_local_texture_decode_reusable(9, 10, SRC, LEN, SRC, LEN,
                                                   GuestGpuWriteQuery::Unchanged),
              "a proven-unchanged source range carries one decode across a span boundary");
        CHECK(!submit_local_texture_decode_reusable(9, 10, SRC, LEN, SRC, LEN,
                                                    GuestGpuWriteQuery::Overlap),
              "a guest GPU write over the decoded range forces a fresh cross-span resolve");
        CHECK(!submit_local_texture_decode_reusable(9, 10, SRC, LEN, SRC, LEN,
                                                    GuestGpuWriteQuery::Unknown),
              "an unusable journal never counts as proof that a decode survived a span boundary");
        CHECK(!submit_local_texture_decode_reusable(9, 10, SRC, LEN, SRC, 0,
                                                    GuestGpuWriteQuery::Unchanged),
              "a resource with no validated source range keeps the span-local lifetime");
        // The resolved range is not a pure function of the decode key: a DCC fast clear moves it to
        // the metadata plane, and the HLE pitch registry can move its extent. An entry proved clean
        // over its own range says nothing about a range that has since moved.
        CHECK(!submit_local_texture_decode_reusable(9, 10, SRC, LEN, SRC + 0x1000, LEN,
                                                    GuestGpuWriteQuery::Unchanged),
              "a retained entry is refused once the binding resolves to a different source address");
        CHECK(!submit_local_texture_decode_reusable(9, 10, SRC, LEN, SRC, LEN * 2,
                                                    GuestGpuWriteQuery::Unchanged),
              "a retained entry is refused once the binding resolves to a different source extent");

        // The end-to-end half needs a LIVE texture: captured replay backing sets host_data, which
        // excludes the resource from the persistent cache and therefore from cross-span retention.
        // Map a real committed guest range and decode out of it, exactly as a booted title does.
        auto map_flexible = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapFlexibleMemory"));
        uint64_t guest_base = 0;
        const bool guest_mapped = map_flexible &&
            map_flexible(reinterpret_cast<uint64_t>(&guest_base), 0x10000u, 0x2u, 0u, 0u, 0u) == 0 &&
            guest_base != 0;
        CHECK(guest_mapped,
              "the decode-scope test maps a committed guest range for a live-decoded texture");

        if (guest_mapped) {
            constexpr uint32_t TW = 16, TH = 16;                 // color target extent per span
            auto* guest = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(guest_base));
            const std::vector<uint8_t> red_2x2 = {
                255, 0, 0, 255,   255, 0, 0, 255,
                255, 0, 0, 255,   255, 0, 0, 255,
            };
            const std::vector<uint8_t> green_2x2 = {
                0, 255, 0, 255,   0, 255, 0, 255,
                0, 255, 0, 255,   0, 255, 0, 255,
            };

            // One submit: draw -> dispatch -> draw. The dispatch is what splits the submit into two
            // graphics spans, and it is also the operation that can rewrite a texture backing.
            // Each draw gets its own retained color target so both spans stay separately readable:
            // an intermediate span publishes no frame, so the assertions read the renderer's target
            // history instead of the callback return.
            uint64_t next_target = 0x100700000ull;
            uint64_t span_targets[2] = {0, 0};
            auto texture_draw = [&](uint64_t texture_addr, uint32_t tw, uint32_t th,
                                    uint32_t pitch, uint64_t draw_index, uint64_t order) {
                DrawItem draw = replay.items[0];
                draw.color0_base = next_target;
                if (draw_index < 2) span_targets[draw_index] = next_target;
                next_target += 0x100000ull;
                draw.color0_width = TW;
                draw.color0_height = TH;
                draw.draw_index = draw_index;
                draw.command_order = order;
                auto table = std::make_shared<ShaderResourceTable>(compile_rt);
                table->resources[0].gpu_addr = texture_addr;
                table->resources[0].width = tw;
                table->resources[0].height = th;
                table->resources[0].size = tw * th * 4u;
                table->resources[0].linear_row_pitch_bytes = pitch;
                table->resources[0].host_data = nullptr;      // live guest memory, not capture bytes
                table->resources[0].host_data_size = 0;
                draw.prt = std::move(table);
                return draw;
            };

            // `dispatch_writes` is the range the interleaved operation rewrites and reports through
            // the ordered guest-write journal. Pointing it AT the sampled backing is the invalidation
            // case; pointing it elsewhere is the control that must still reuse.
            uint64_t dispatch_writes = 0;
            const std::vector<uint8_t>* dispatch_bytes = nullptr;
            LiveComputeFn interleaved_write = [&](const std::vector<ComputeItem>&) {
                if (!dispatch_writes || !dispatch_bytes) return true;
                std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dispatch_writes)),
                            dispatch_bytes->data(), dispatch_bytes->size());
                notify_guest_gpu_write(dispatch_writes, dispatch_bytes->size());
                return true;
            };

            size_t rendered_spans = 0;
            LiveRenderFn counting_render = [&](const std::vector<DrawItem>& items, uint32_t width,
                                               uint32_t height) {
                ++rendered_spans;
                return RenderedFrame(render_submit_items(items, width, height));
            };
            // Center texel of a retained 16x16 color target, or nullptr if it is not readable at the
            // declared extent. Reading through the renderer's own target history keeps the assertion
            // on the pixels a later pass would actually sample.
            auto target_center = [&](uint64_t target, std::vector<uint8_t>& keep) -> const uint8_t* {
                LiveTargetSnapshot snapshot;
                if (!target || !read_live_render_target(target, snapshot) || !snapshot.pixels ||
                    snapshot.width != TW || snapshot.height != TH ||
                    snapshot.pixels->size() != static_cast<size_t>(TW) * TH * 4u)
                    return nullptr;
                keep = *snapshot.pixels;
                return &keep[(static_cast<size_t>(TH / 2) * TW + TW / 2) * 4];
            };

            std::vector<ComputeItem> dispatches(1);
            dispatches[0].dispatch_index = 0;
            const std::vector<SubmitOperation> two_spans = {
                {SubmitOperationKind::Draw, 0, 100},
                {SubmitOperationKind::Dispatch, 0, 150},
                {SubmitOperationKind::Draw, 1, 200},
            };

            // (1) CONTROL — the dispatch writes an unrelated range. Both spans sample the same
            // identity, so the submit-scoped map must serve the second one without decoding again.
            const uint64_t control_texture = guest_base + 0x1000;
            uint64_t unrelated = guest_base + 0x8000;
            std::memcpy(guest + 0x1000, red_2x2.data(), red_2x2.size());
            dispatch_writes = unrelated;
            dispatch_bytes = &green_2x2;
            rendered_spans = 0;
            reset_texture_decode_scope_stats();
            execute_ordered_items(
                two_spans,
                {texture_draw(control_texture, 2, 2, 8, 0, 100),
                 texture_draw(control_texture, 2, 2, 8, 1, 200)},
                dispatches, counting_render, interleaved_write, W, H);
            auto control = texture_decode_scope_stats();
            CHECK(rendered_spans == 2,
                  "an interleaved dispatch splits one submit into two graphics spans");
            CHECK(control.decodes == 1 && control.cross_span_reuses == 1 &&
                      control.invalidations == 0,
                  "one identity sampled in two spans of one submit decodes exactly once");
            std::vector<uint8_t> control_a, control_b;
            const uint8_t* control_first = target_center(span_targets[0], control_a);
            const uint8_t* control_second = target_center(span_targets[1], control_b);
            CHECK(control_first && control_second &&
                      control_first[0] > 0xC0 && control_first[1] < 0x40 &&
                      control_second[0] > 0xC0 && control_second[1] < 0x40,
                  "both spans sample the unchanged red guest texture");

            // (2) INVALIDATION — the same shape, except the dispatch rewrites the sampled backing
            // and reports it. Serving the retained entry here would publish pixels the guest has
            // already replaced; the second span must observe the new bytes. This assertion fails if
            // the widened map skips the in-submit journal check.
            const uint64_t written_texture = guest_base + 0x2000;
            std::memcpy(guest + 0x2000, red_2x2.data(), red_2x2.size());
            dispatch_writes = written_texture;
            dispatch_bytes = &green_2x2;
            rendered_spans = 0;
            reset_texture_decode_scope_stats();
            execute_ordered_items(
                two_spans,
                {texture_draw(written_texture, 2, 2, 8, 0, 100),
                 texture_draw(written_texture, 2, 2, 8, 1, 200)},
                dispatches, counting_render, interleaved_write, W, H);
            auto invalidated = texture_decode_scope_stats();
            CHECK(invalidated.decodes == 2 && invalidated.cross_span_reuses == 0 &&
                      invalidated.invalidations == 1,
                  "a guest GPU write to the backing between spans forces a second decode");
            std::vector<uint8_t> written_a, written_b;
            const uint8_t* written_first = target_center(span_targets[0], written_a);
            const uint8_t* written_second = target_center(span_targets[1], written_b);
            CHECK(written_first && written_second &&
                      written_first[0] > 0xC0 && written_first[1] < 0x40 &&
                      written_second[1] > 0xC0 && written_second[0] < 0x40,
                  "the span after the write samples the rewritten green texture, not the red cache");

            // (3) KEY COMPLETENESS — two descriptors at the same address and the same extent whose
            // decoded bytes differ. `linear_row_pitch_bytes` 8 reads 16 contiguous bytes; 16 reads
            // row 0 at +0 and row 1 at +16. The reuse path re-derives extent and format from the
            // descriptor and takes only the PIXELS from the map, so an under-specified key shows up
            // exactly here: dropping the pitch field would serve the first decode to the second draw
            // and collapse this to one decode with a cross-span reuse.
            const uint64_t pitch_texture = guest_base + 0x3000;
            std::memcpy(guest + 0x3000, red_2x2.data(), red_2x2.size());
            std::memcpy(guest + 0x3010, green_2x2.data(), green_2x2.size());
            dispatch_writes = unrelated;
            dispatch_bytes = &green_2x2;
            rendered_spans = 0;
            reset_texture_decode_scope_stats();
            execute_ordered_items(
                two_spans,
                {texture_draw(pitch_texture, 2, 2, 8, 0, 100),
                 texture_draw(pitch_texture, 2, 2, 16, 1, 200)},
                dispatches, counting_render, interleaved_write, W, H);
            auto distinct_keys = texture_decode_scope_stats();
            CHECK(distinct_keys.decodes == 2 && distinct_keys.cross_span_reuses == 0 &&
                      distinct_keys.invalidations == 0,
                  "two identities differing only in source row pitch decode separately");

            // (4) SCRATCH PINNING — the aliasing hazard, which the cases above cannot reach because
            // the persistent cache accepts their pixels and therefore owns them. Re-decoding an
            // identity the persistent cache already holds THIS submit cannot replace that entry (its
            // LRU generation is current), so the fresh pixels stay in the scratch slot and the map
            // entry points straight at it. A later span must not be handed that slot.
            //
            //   span 1: decode -> persistent insert
            //   dispatch A: rewrite the backing red->green and report it
            //   span 2: forced re-decode, persistent entry not replaceable -> pixels live in scratch
            //   dispatch B: write an unrelated range
            //   span 3: draw a DIFFERENT texture first (the slot grab), then the retained identity
            //
            // Unpinned, span 3's first decode takes the slot span 2 is still pointing at and the
            // second draw renders blue. Pinned, the allocator skips it and green survives.
            const uint64_t pinned_texture = guest_base + 0x5000;
            const uint64_t other_texture = guest_base + 0x6000;
            const std::vector<uint8_t> blue_2x2 = {
                0, 0, 255, 255,   0, 0, 255, 255,
                0, 0, 255, 255,   0, 0, 255, 255,
            };
            std::memcpy(guest + 0x5000, red_2x2.data(), red_2x2.size());
            std::memcpy(guest + 0x6000, blue_2x2.data(), blue_2x2.size());
            std::vector<ComputeItem> two_dispatches(2);
            two_dispatches[0].dispatch_index = 0;
            two_dispatches[1].dispatch_index = 1;
            int dispatch_seen = 0;
            LiveComputeFn write_then_idle = [&](const std::vector<ComputeItem>&) {
                if (dispatch_seen++ == 0) {
                    std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(pinned_texture)),
                                green_2x2.data(), green_2x2.size());
                    notify_guest_gpu_write(pinned_texture, green_2x2.size());
                } else {
                    notify_guest_gpu_write(unrelated, 4);
                }
                return true;
            };
            uint64_t pin_targets[4] = {0, 0, 0, 0};
            uint64_t pin_next_target = 0x101000000ull;
            auto pin_draw = [&](uint64_t texture_addr, uint64_t draw_index, uint64_t order) {
                DrawItem draw = texture_draw(texture_addr, 2, 2, 8, 0, order);
                draw.draw_index = draw_index;
                draw.color0_base = pin_next_target;
                pin_targets[draw_index] = pin_next_target;
                pin_next_target += 0x100000ull;
                return draw;
            };
            dispatch_seen = 0;
            reset_texture_decode_scope_stats();
            execute_ordered_items(
                {{SubmitOperationKind::Draw, 0, 100},
                 {SubmitOperationKind::Dispatch, 0, 150},
                 {SubmitOperationKind::Draw, 1, 200},
                 {SubmitOperationKind::Dispatch, 1, 250},
                 {SubmitOperationKind::Draw, 2, 300},
                 {SubmitOperationKind::Draw, 3, 310}},
                {pin_draw(pinned_texture, 0, 100), pin_draw(pinned_texture, 1, 200),
                 pin_draw(other_texture, 2, 300), pin_draw(pinned_texture, 3, 310)},
                two_dispatches, counting_render, write_then_idle, W, H);
            const auto pinned = texture_decode_scope_stats();
            std::vector<uint8_t> pin_second, pin_last;
            const uint8_t* pin_span2 = target_center(pin_targets[1], pin_second);
            const uint8_t* pin_span3 = target_center(pin_targets[3], pin_last);
            // `scratch_pins == 1` is what makes this case self-validating rather than merely true.
            // The pin is only reached because `decode_generation` is per SUBMIT, which is what makes
            // `can_replace` false for span 2's re-decode. Revert that to a per-span bump and the
            // pixels move into the persistent cache instead, no pin is taken, and both green
            // assertions below still pass while covering nothing — so assert the mechanism, not only
            // its consequence. It doubles as the guard on the per-submit generation, which is
            // otherwise untestable without provoking a use-after-free.
            CHECK(pinned.scratch_pins == 1 && pinned.cross_span_reuses == 1 &&
                      pinned.invalidations == 1 && pinned.decodes == 3,
                  "the twice-decoded identity is pinned in scratch and served across the next span");
            CHECK(pin_span2 && pin_span3 &&
                      pin_span2[1] > 0xC0 && pin_span2[0] < 0x40 &&
                      pin_span3[1] > 0xC0 && pin_span3[0] < 0x40,
                  "a scratch-backed retained entry survives a later span decoding another texture");

            // (5) The identity map does not outlive its submit: an entry established in one submit
            // must never serve the next, whose indirect descriptor memory may have moved on. The
            // discriminating counter is `invalidations` — a map that leaked across the boundary
            // would FIND the stale entry and then refuse it on its foreign submit serial, which is
            // safe but is not the contract. Zero invalidations means it was never there to find.
            const uint64_t across_texture = guest_base + 0x4000;
            std::memcpy(guest + 0x4000, red_2x2.data(), red_2x2.size());
            dispatch_writes = 0;
            dispatch_bytes = nullptr;
            reset_texture_decode_scope_stats();
            execute_ordered_items({{SubmitOperationKind::Draw, 0, 100}},
                                  {texture_draw(across_texture, 2, 2, 8, 0, 100)},
                                  {}, counting_render, interleaved_write, W, H);
            const auto first_submit = texture_decode_scope_stats();
            execute_ordered_items({{SubmitOperationKind::Draw, 0, 100}},
                                  {texture_draw(across_texture, 2, 2, 8, 0, 100)},
                                  {}, counting_render, interleaved_write, W, H);
            const auto second_submit = texture_decode_scope_stats();
            CHECK(first_submit.decodes == 1 && first_submit.invalidations == 0 &&
                      second_submit.invalidations == 0 &&
                      second_submit.cross_span_reuses == 0,
                  "a new submit rebuilds the identity map instead of inheriting the previous one");
        }
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
