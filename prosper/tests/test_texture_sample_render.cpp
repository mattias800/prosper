// test_texture_sample_render — a recompiled PIXEL shader that does MIMG image_sample reads a real
// bound texture and the sampled texel reaches the framebuffer.
//
// This is stage 4 of the resource-binding plan (textures). The pixel shader samples a 2x2 RGBA texture
// with four distinct texels; a NEAREST/normalized sampler maps coord 0.25 -> texel column/row 0 and
// coord 0.75 -> column/row 1. We render the fullscreen triangle twice with two coords and assert the
// framebuffer shows the corresponding texel color — proving MIMG decode, OpImageSampleImplicitLod,
// coordinate assembly (VADDR VGPRs -> u,v), the combined-image-sampler binding, and dmask->VDATA all
// work end to end. A broken path would leave the frame blue (clear) or the wrong texel.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_texture_sample_render ==\n");
    const uint32_t W = 64, H = 64;

    // Fullscreen-triangle VS (from gl_VertexIndex; no resource table needed).
    const uint32_t vs[] = {
        0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
        0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
    };
    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]));
    CHECK(!vert.empty() && vert[0] == 0x07230203u, "recompiled fullscreen-triangle VS -> SPIR-V");

    // Pixel shader: u,v = literal coords; image_sample v[0:3], v[0:1], s[8:15], s[16:19] dim:2D dmask:0xf;
    // exp mrt0 v0..v3. The two literal dwords (indices 1 and 3) are the u,v coords — patched per render.
    const uint32_t ps_template[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    // The T# is placed directly in user-data SGPRs (SRSRC base = s8) -> resolved via sgpr_base.
    ShaderResourceTable rt;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1; /*2D*/
      t.width = 2; t.height = 2; t.sgpr_base = 8; rt.resources.push_back(t); }

    // 2x2 RGBA8: (0,0)=red (1,0)=green (0,1)=blue (1,1)=white.
    const uint8_t texels[2*2*4] = {
        255,0,0,255,   0,255,0,255,
        0,0,255,255,   255,255,255,255,
    };
    prosper::test::TexDesc td{ /*binding*/4, /*w*/2, /*h*/2, texels };

    auto sample_center = [&](uint32_t u_bits, uint32_t v_bits, uint8_t out_rgb[3]) -> bool {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = u_bits; ps[3] = v_bits;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        if (frag.empty() || frag[0] != 0x07230203u) return false;
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
        if (px.size() != (size_t)W * H * 4) return false;
        const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
        out_rgb[0] = c[0]; out_rgb[1] = c[1]; out_rgb[2] = c[2];
        return true;
    };

    // coord 0.25 -> texel index floor(0.25*2)=0; coord 0.75 -> floor(0.75*2)=1.
    const uint32_t C025 = 0x3e800000u /*0.25f*/, C075 = 0x3f400000u /*0.75f*/;
    uint8_t rgb[3];

    CHECK(recompile_fragment(ps_template, sizeof(ps_template)/sizeof(ps_template[0]), nullptr).empty(),
          "image_sample PS is rejected without a resource table");

    bool ok0 = sample_center(C025, C025, rgb);
    printf("  (0.25,0.25) center=(%u,%u,%u)\n", ok0?rgb[0]:0, ok0?rgb[1]:0, ok0?rgb[2]:0);
    CHECK(ok0 && rgb[0] > 0x80 && rgb[1] < 0x40 && rgb[2] < 0x40, "sampling texel (0,0) yields RED");

    bool ok1 = sample_center(C075, C025, rgb);
    printf("  (0.75,0.25) center=(%u,%u,%u)\n", ok1?rgb[0]:0, ok1?rgb[1]:0, ok1?rgb[2]:0);
    CHECK(ok1 && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40, "sampling texel (1,0) yields GREEN (proves u routing)");

    // The live renderer commonly submits hundreds of draws that reference the same few decoded
    // textures. The backend must upload identical pixels once per call while preserving the legacy
    // rendered bytes and separate per-descriptor views/samplers.
    {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = texels; resource.tw = 2; resource.th = 2;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;
        prosper::test::FrameResource swizzled_resource = resource;
        swizzled_resource.swizzle[0] = 5;
        swizzled_resource.swizzle[1] = 4;
        prosper::test::BackendDraw swizzled_draw = draw;
        swizzled_draw.R = {swizzled_resource};

        std::vector<uint8_t> same_binding =
            prosper::test::render_draws_rgba({draw, draw}, W, H);
        const auto same_binding_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(!same_binding.empty() &&
                  same_binding_stats.texture_binding_references == 2 &&
                  same_binding_stats.unique_texture_bindings == 1,
              "identical texture view and sampler contracts share one Vulkan binding pair");

        std::vector<uint8_t> shared =
            prosper::test::render_draws_rgba({draw, swizzled_draw}, W, H);
        const auto shared_stats = prosper::test::backend_texture_upload_stats();
        const auto shared_resource_stats = prosper::test::backend_resource_reuse_stats();
        CHECK(shared_stats.references == 2 && shared_stats.unique_uploads == 1,
              "draws with separate views over shared pixels produce one backend texture upload");
        CHECK(shared_resource_stats.texture_binding_references == 2 &&
                  shared_resource_stats.unique_texture_bindings == 2,
              "different component swizzles retain distinct Vulkan image views");

#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1");
#else
        setenv("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1", 1);
#endif
        std::vector<uint8_t> legacy =
            prosper::test::render_draws_rgba({draw, swizzled_draw}, W, H);
        const auto legacy_stats = prosper::test::backend_texture_upload_stats();
#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "");
#else
        unsetenv("PROSPER_NO_BACKEND_TEXTURE_SHARE");
#endif
        CHECK(legacy_stats.references == 2 && legacy_stats.unique_uploads == 2,
              "disable switch restores one backend upload per texture reference");
        CHECK(!shared.empty() && shared == legacy,
              "shared and legacy uploads with distinct view swizzles render byte-identically");
    }

    // A non-zero content ID is an exact-validation proof supplied by the frontend. The backend may
    // retain that uploaded image across render callbacks, but the diagnostic switch must preserve
    // rendered output while forcing the old upload path for controlled A/B runs.
    {
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        prosper::test::FrameResource resource;
        resource.binding = 4; resource.set = 1;
        resource.tex_rgba = texels; resource.tw = 2; resource.th = 2;
        resource.persistent_texture_id = 0x7020000000000001ull;
        prosper::test::BackendDraw draw;
        draw.vs = vert; draw.fs = frag; draw.R = {resource}; draw.vcount = 3;

        std::vector<uint8_t> first = prosper::test::render_draws_rgba({draw}, W, H);
        const auto first_stats = prosper::test::backend_texture_upload_stats();
        std::vector<uint8_t> reused = prosper::test::render_draws_rgba({draw}, W, H);
        const auto reused_stats = prosper::test::backend_texture_upload_stats();
        CHECK(first_stats.persistent_misses == 1 && first_stats.unique_uploads == 1,
              "first exact-validated texture version is uploaded into the persistent cache");
        CHECK(reused_stats.persistent_hits == 1 && reused_stats.unique_uploads == 0 &&
                  reused_stats.upload_bytes == 0,
              "same exact-validated texture version skips its next callback upload");
        CHECK(!first.empty() && first == reused,
              "persistent texture reuse renders byte-identically to its initial upload");

#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "1");
#else
        setenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "1", 1);
#endif
        std::vector<uint8_t> bypassed = prosper::test::render_draws_rgba({draw}, W, H);
        const auto bypassed_stats = prosper::test::backend_texture_upload_stats();
#ifdef _WIN32
        _putenv_s("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES", "");
#else
        unsetenv("PROSPER_NO_BACKEND_PERSISTENT_TEXTURES");
#endif
        CHECK(bypassed_stats.persistent_hits == 0 && bypassed_stats.unique_uploads == 1,
              "persistent texture disable switch restores an upload on every callback");
        CHECK(reused == bypassed,
              "persistent cache and forced-upload paths render byte-identically");

        uint8_t changed_texels[sizeof(texels)];
        std::memcpy(changed_texels, texels, sizeof(texels));
        changed_texels[4] = 255;  // sampled top-right texel: green -> yellow
        resource.tex_rgba = changed_texels;
        resource.persistent_texture_id = 0x7020000000000002ull;
        draw.R = {resource};
        std::vector<uint8_t> changed = prosper::test::render_draws_rgba({draw}, W, H);
        const auto changed_stats = prosper::test::backend_texture_upload_stats();
        CHECK(changed_stats.persistent_misses == 1 && changed_stats.unique_uploads == 1,
              "a new exact-validated content version cannot hit the prior image");
        CHECK(!changed.empty() && changed != reused,
              "a new content version uploads and renders its changed pixels");
    }

    // A guest color target can remain on the backend GPU between render calls and be sampled by
    // exact identity. Compare that path with the established CPU readback+upload route, then prove
    // a LOADing second pass can skip readback without changing the eventual sampled pixels.
    {
        static const uint32_t kRedPs[] = {
            0x7E0002F2u, 0x7E020280u, 0x7E040280u, 0x7E0602F2u,
            0xF800180Fu, 0x03020100u, 0xBF810000u};
        static const uint32_t kGreenPs[] = {
            0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u,
            0xF800180Fu, 0x03020100u, 0xBF810000u};
        std::vector<uint32_t> red = recompile_fragment(
            kRedPs, sizeof(kRedPs) / sizeof(kRedPs[0]), nullptr);
        std::vector<uint32_t> green = recompile_fragment(
            kGreenPs, sizeof(kGreenPs) / sizeof(kGreenPs[0]), nullptr);
        std::vector<uint32_t> sample_ps(
            ps_template, ps_template + sizeof(ps_template) / sizeof(ps_template[0]));
        sample_ps[1] = C025; sample_ps[3] = C025;
        std::vector<uint32_t> sample = recompile_fragment(
            sample_ps.data(), sample_ps.size(), &rt);

        ResolvedPipelineState opaque{};
        opaque.topology = 3; opaque.color_write_mask = 0xF;
        ResolvedPipelineState additive = opaque;
        additive.blend_enable = true;
        additive.src_color_blend_factor = 1; // ONE
        additive.dst_color_blend_factor = 1; // ONE
        additive.color_blend_op = 0;         // ADD

        prosper::test::BackendDraw producer;
        producer.vs = vert; producer.fs = red; producer.ps = &opaque; producer.vcount = 3;
        constexpr uint64_t target_id = 0x7590000000000001ull;
        prosper::test::BackendColorTarget first_target{target_id, false, true};
        std::vector<uint8_t> first = prosper::test::render_draws_rgba(
            {producer}, W, H, nullptr, nullptr, false, &first_target);
        const auto first_target_stats = prosper::test::backend_color_target_stats();
        CHECK(first.size() == (size_t)W * H * 4 && first_target_stats.writes == 1 &&
                  first_target_stats.write_hits == 0 && first_target_stats.readbacks == 1,
              "first persistent color-target write materializes its requested CPU result");

        prosper::test::FrameResource cpu_resource;
        cpu_resource.binding = 4; cpu_resource.set = 1;
        cpu_resource.tex_rgba = first.data(); cpu_resource.tw = W; cpu_resource.th = H;
        prosper::test::BackendDraw cpu_sample;
        cpu_sample.vs = vert; cpu_sample.fs = sample; cpu_sample.R = {cpu_resource};
        cpu_sample.vcount = 3;
        std::vector<uint8_t> cpu_roundtrip = prosper::test::render_draws_rgba(
            {cpu_sample}, W, H);

        prosper::test::FrameResource gpu_resource = cpu_resource;
        gpu_resource.tex_rgba = nullptr;
        gpu_resource.persistent_render_target_id = target_id;
        prosper::test::BackendDraw gpu_sample = cpu_sample;
        gpu_sample.R = {gpu_resource};
        std::vector<uint8_t> gpu_resident = prosper::test::render_draws_rgba(
            {gpu_sample}, W, H);
        const auto sampled_stats = prosper::test::backend_color_target_stats();
        CHECK(!gpu_resident.empty() && gpu_resident == cpu_roundtrip &&
                  sampled_stats.sampled_hits == 1,
              "GPU-resident target sampling matches CPU readback+upload byte-for-byte");

        prosper::test::BackendDraw add_green;
        add_green.vs = vert; add_green.fs = green; add_green.ps = &additive;
        add_green.vcount = 3;
        std::vector<uint8_t> cpu_accumulated = prosper::test::render_draws_rgba(
            {add_green}, W, H, first.data());
        cpu_resource.tex_rgba = cpu_accumulated.data();
        cpu_sample.R = {cpu_resource};
        std::vector<uint8_t> cpu_accumulated_sample = prosper::test::render_draws_rgba(
            {cpu_sample}, W, H);

        prosper::test::BackendColorTarget deferred_target{target_id, true, false};
        std::vector<uint8_t> deferred = prosper::test::render_draws_rgba(
            {add_green}, W, H, nullptr, nullptr, false, &deferred_target);
        const auto deferred_stats = prosper::test::backend_color_target_stats();
        const bool pinned = prosper::test::pin_persistent_color_target(
            target_id, W, H, VK_FORMAT_R8G8B8A8_UNORM);
        // Exceed the 64-entry target-cache cap after the deferred target becomes the oldest live
        // entry. Without the submit-lifetime pin, normal LRU pressure discards its only current pixels
        // before the frontend's final callback can materialize them.
        for (uint64_t pressure = 0; pressure < 70; ++pressure) {
            prosper::test::BackendColorTarget pressure_target{
                target_id + 0x1000 + pressure, false, false};
            prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &pressure_target);
        }
        std::vector<uint8_t> materialized;
        std::string materialize_error;
        const bool materialized_ok = prosper::test::readback_persistent_color_target(
            target_id, W, H, VK_FORMAT_R8G8B8A8_UNORM, materialized, materialize_error);
        const bool unpinned = prosper::test::unpin_persistent_color_target(
            target_id, W, H, VK_FORMAT_R8G8B8A8_UNORM);
        std::vector<uint8_t> deferred_sample = prosper::test::render_draws_rgba(
            {gpu_sample}, W, H);
        CHECK(deferred.empty() && deferred_stats.write_hits == 1 &&
                  deferred_stats.readbacks == 0,
              "persistent LOAD pass can complete without allocating a CPU readback");
        CHECK(pinned && materialized_ok && materialized == cpu_accumulated && unpinned,
              "pinned GPU-only target survives eviction pressure until ordered materialization");
        CHECK(!deferred_sample.empty() && deferred_sample == cpu_accumulated_sample,
              "deferred-readback accumulation matches the CPU reference after sampling");

        prosper::test::invalidate_persistent_color_target(target_id);
        gpu_resource.tex_rgba = cpu_accumulated.data();
        gpu_sample.R = {gpu_resource};
        std::vector<uint8_t> invalidated_fallback = prosper::test::render_draws_rgba(
            {gpu_sample}, W, H);
        const auto fallback_stats = prosper::test::backend_color_target_stats();
        CHECK(fallback_stats.sampled_hits == 0 &&
                  invalidated_fallback == cpu_accumulated_sample,
              "invalidated GPU target uses the supplied CPU fallback instead of stale pixels");
    }

    // Renderer-owned FP16 targets must retain HDR values through both CPU RTT readback/upload and
    // direct GPU-resident sampling. A producer value of 2.0 multiplied by 0.25 in the consumer is
    // 0.5; the historical RGBA8 conversion clamped it to 1.0 first and therefore produced 0.25.
    {
        static const uint32_t kHdrPs[] = {
            0x7E0002FFu, 0x40000000u,  // v0 = 2.0
            0x7E0202FFu, 0x3E800000u,  // v1 = 0.25
            0x7E040280u,               // v2 = 0.0
            0x7E0602F2u,               // v3 = 1.0
            0xF800180Fu, 0x03020100u, 0xBF810000u,
        };
        const uint32_t kHdrSamplePs[] = {
            0x7E0002FFu, C025, 0x7E0202FFu, C025,
            0xF0800F08u, 0x00820000u,  // image_sample v[0:3]
            0x100000FFu, 0x3E800000u,  // v0 *= 0.25
            0xF800000Fu, 0x03020100u, 0xBF810000u,
        };
        std::vector<uint32_t> hdr = recompile_fragment(
            kHdrPs, sizeof(kHdrPs) / sizeof(kHdrPs[0]), nullptr);
        std::vector<uint32_t> hdr_sample = recompile_fragment(
            kHdrSamplePs, sizeof(kHdrSamplePs) / sizeof(kHdrSamplePs[0]), &rt);
        CHECK(!hdr.empty() && !hdr_sample.empty(),
              "recompiled native FP16 producer and sampled consumer shaders");
        if (!hdr.empty() && !hdr_sample.empty()) {
            ResolvedPipelineState fp16_state{};
            fp16_state.topology = 3;
            fp16_state.color_write_mask = 0xF;
            fp16_state.color0_format = VK_FORMAT_R16G16B16A16_SFLOAT;
            prosper::test::BackendDraw producer;
            producer.vs = vert; producer.fs = hdr; producer.ps = &fp16_state;
            producer.vcount = 3;

            constexpr uint64_t fp16_target_id = 0x7590000000000016ull;
            prosper::test::BackendColorTarget fp16_target{
                fp16_target_id, false, true, VK_FORMAT_R16G16B16A16_SFLOAT};
            std::vector<uint8_t> native = prosper::test::render_draws_rgba(
                {producer}, W, H, nullptr, nullptr, false, &fp16_target);
            bool native_ok = native.size() == static_cast<size_t>(W) * H * 8;
            float native_red = 0.0f;
            if (native_ok) {
                uint16_t half = 0;
                std::memcpy(&half, native.data() +
                    ((static_cast<size_t>(H / 2) * W + W / 2) * 8), sizeof(half));
                native_red = half_to_float(half);
                native_ok = native_red > 1.99f && native_red < 2.01f;
            }
            CHECK(native_ok, "native FP16 target readback preserves HDR value 2.0");

            prosper::test::FrameResource fp16_resource;
            fp16_resource.binding = 4; fp16_resource.set = 1;
            fp16_resource.tex_rgba = native.data(); fp16_resource.tw = W; fp16_resource.th = H;
            fp16_resource.texture_format = VK_FORMAT_R16G16B16A16_SFLOAT;
            prosper::test::BackendDraw consumer;
            consumer.vs = vert; consumer.fs = hdr_sample; consumer.R = {fp16_resource};
            consumer.vcount = 3;
            std::vector<uint8_t> cpu_sampled = prosper::test::render_draws_rgba(
                {consumer}, W, H);
            auto center_red = [&](const std::vector<uint8_t>& pixels) -> uint8_t {
                return pixels.size() == static_cast<size_t>(W) * H * 4
                    ? pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4] : 0;
            };
            const uint8_t cpu_red = center_red(cpu_sampled);
            CHECK(cpu_red >= 126 && cpu_red <= 129,
                  "FP16 readback/upload consumer observes 2.0 before scaling to 0.5");

            prosper::test::FrameResource gpu_fp16_resource = fp16_resource;
            gpu_fp16_resource.tex_rgba = nullptr;
            gpu_fp16_resource.persistent_render_target_id = fp16_target_id;
            consumer.R = {gpu_fp16_resource};
            std::vector<uint8_t> gpu_sampled = prosper::test::render_draws_rgba(
                {consumer}, W, H);
            const auto gpu_sample_stats = prosper::test::backend_color_target_stats();
            CHECK(gpu_sample_stats.sampled_hits == 1 && gpu_sampled == cpu_sampled,
                  "GPU-resident FP16 target sampling matches native CPU RTT round-trip");
            prosper::test::invalidate_persistent_color_target(fp16_target_id);
        }
    }

    // The backend upload key includes depth and image dimensionality. Exercise a real 3D image here
    // so depth-1 3D resources cannot accidentally regress to a 2D Vulkan image/view during sharing.
    {
        const uint32_t ps_3d[] = {
            0x7e0002ffu, C025, 0x7e0202ffu, C025, 0x7e0402ffu, C075,
            0xf0800f10u, 0x00820000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
        };
        ShaderResourceTable rt_3d;
        { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 2;
          t.width = 2; t.height = 2; t.depth = 2; t.sgpr_base = 8;
          rt_3d.resources.push_back(t); }
        std::vector<uint32_t> frag_3d =
            recompile_fragment(ps_3d, sizeof(ps_3d) / sizeof(ps_3d[0]), &rt_3d);
        CHECK(!frag_3d.empty(), "recompiled a 3D image_sample fragment shader");
        if (!frag_3d.empty()) {
            const uint8_t volume[2*2*2*4] = {
                255,0,0,255, 255,0,0,255, 255,0,0,255, 255,0,0,255,
                0,255,0,255, 0,255,0,255, 0,255,0,255, 0,255,0,255,
            };
            prosper::test::FrameResource resource_3d;
            resource_3d.binding = 4; resource_3d.set = 1;
            resource_3d.tex_rgba = volume;
            resource_3d.tw = 2; resource_3d.th = 2; resource_3d.td = 2;
            resource_3d.img_dim = 2;
            prosper::test::BackendDraw draw_3d;
            draw_3d.vs = vert; draw_3d.fs = frag_3d; draw_3d.R = {resource_3d};
            draw_3d.vcount = 3;
            std::vector<uint8_t> volume_px =
                prosper::test::render_draws_rgba({draw_3d, draw_3d}, W, H);
            const auto volume_stats = prosper::test::backend_texture_upload_stats();
            bool volume_ok = volume_px.size() == static_cast<size_t>(W) * H * 4;
            if (volume_ok) {
                const uint8_t* c = &volume_px[((size_t)(H/2) * W + W/2) * 4];
                volume_ok = c[1] > 0x80 && c[0] < 0x40 && c[2] < 0x40;
            }
            CHECK(volume_ok, "shared 3D upload samples the selected depth slice");
            CHECK(volume_stats.references == 2 && volume_stats.unique_uploads == 1 &&
                      volume_stats.upload_bytes == sizeof(volume),
                  "3D texture sharing accounts for depth and uploads the volume once");
        }
    }

    bool ok2 = sample_center(C025, C075, rgb);
    printf("  (0.25,0.75) center=(%u,%u,%u)\n", ok2?rgb[0]:0, ok2?rgb[1]:0, ok2?rgb[2]:0);
    CHECK(ok2 && rgb[2] > 0x80 && rgb[0] < 0x40 && rgb[1] < 0x40, "sampling texel (0,1) yields BLUE (proves v routing)");

    // #275: anisotropy applied. A sampler built with max_aniso_ratio > 0 (here 4 -> 16x) must create
    // validly and still sample correctly — the fullscreen quad isn't minified, so aniso changes nothing
    // about the result, but a broken apply (invalid usage / wrong sampler) would blank the frame or move
    // the texel. Same (0.75,0.25) -> texel (1,0) = GREEN check, now through the anisotropic sampler.
    {
        prosper::test::TexDesc td_a{ /*binding*/4, /*w*/2, /*h*/2, texels, /*max_aniso_ratio*/4u };
        std::vector<uint32_t> ps(ps_template, ps_template + sizeof(ps_template)/sizeof(ps_template[0]));
        ps[1] = C075; ps[3] = C025;
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        bool okA = !frag.empty() && frag[0] == 0x07230203u;
        if (okA) {
            std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td_a);
            okA = px.size() == (size_t)W*H*4;
            if (okA) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        }
        printf("  aniso(16x) (0.75,0.25) center=(%u,%u,%u)\n", okA?rgb[0]:0, okA?rgb[1]:0, okA?rgb[2]:0);
        CHECK(okA && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40,
              "anisotropic sampler (ratio 4/16x) still samples texel (1,0) = GREEN (valid apply)");
    }

    // image_load (integer texel fetch, no sampler): x,y = inline-int coords into v0,v1; image_load
    // v[0:3], v[0:1], s[8:15]; exp mrt0. Coord (x,y) directly indexes the texel — no filtering.
    const uint32_t il_template[] = {
        0x7e000280u, 0x7e020280u, 0xf0000f08u, 0x00020000u, 0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    auto fetch_center = [&](uint32_t x, uint32_t y, uint8_t o[3]) -> bool {
        std::vector<uint32_t> ps(il_template, il_template + sizeof(il_template)/sizeof(il_template[0]));
        ps[0] = 0x7e000200u | (128u + x);   // v_mov_b32 v0, x   (inline int)
        ps[1] = 0x7e020200u | (128u + y);   // v_mov_b32 v1, y
        std::vector<uint32_t> frag = recompile_fragment(ps.data(), ps.size(), &rt);
        if (frag.empty() || frag[0] != 0x07230203u) return false;
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, nullptr, nullptr, nullptr, &td);
        if (px.size() != (size_t)W * H * 4) return false;
        const uint8_t* c = &px[((size_t)(H/2) * W + W/2) * 4];
        o[0]=c[0]; o[1]=c[1]; o[2]=c[2]; return true;
    };
    bool okL0 = fetch_center(0, 0, rgb);
    printf("  image_load(0,0) center=(%u,%u,%u)\n", okL0?rgb[0]:0, okL0?rgb[1]:0, okL0?rgb[2]:0);
    CHECK(okL0 && rgb[0] > 0x80 && rgb[1] < 0x40 && rgb[2] < 0x40, "image_load texel (0,0) yields RED");
    bool okL1 = fetch_center(1, 1, rgb);
    printf("  image_load(1,1) center=(%u,%u,%u)\n", okL1?rgb[0]:0, okL1?rgb[1]:0, okL1?rgb[2]:0);
    CHECK(okL1 && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80, "image_load texel (1,1) yields WHITE (proves integer coords)");

    // The same image_load instruction becomes OpImageRead when its resource class is StorageImage.
    // That SPIR-V interface must be backed by a STORAGE_IMAGE descriptor over an image with STORAGE
    // usage in GENERAL layout, not the sampled texture's combined-image-sampler contract (#374).
    {
        ShaderResourceTable storage_rt;
        { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
          image.binding = 4; image.img_dim = 1; image.width = 2; image.height = 2;
          image.sgpr_base = 8; storage_rt.resources.push_back(image); }
        std::vector<uint32_t> storage_frag = recompile_fragment(
            il_template, sizeof(il_template) / sizeof(il_template[0]), &storage_rt);
        CHECK(!storage_frag.empty() && storage_frag[0] == 0x07230203u,
              "recompiled graphics image_load as a storage-image OpImageRead");
        if (!storage_frag.empty()) {
            prosper::test::FrameResource storage_resource;
            storage_resource.binding = 4; storage_resource.set = 1;
            storage_resource.tex_rgba = texels; storage_resource.tw = 2; storage_resource.th = 2;
            storage_resource.is_storage_image = true;
            prosper::test::BackendDraw storage_draw;
            storage_draw.vs = vert; storage_draw.fs = storage_frag;
            storage_draw.R = {storage_resource}; storage_draw.vcount = 3;
            prosper::test::FrameResource sampled_resource = storage_resource;
            sampled_resource.is_storage_image = false;
            prosper::test::BackendDraw sampled_draw = storage_draw;
            sampled_draw.fs = recompile_fragment(
                il_template, sizeof(il_template) / sizeof(il_template[0]), &rt);
            sampled_draw.R = {sampled_resource};
            std::vector<uint8_t> storage_px = prosper::test::render_draws_rgba(
                {storage_draw}, W, H);
            bool storage_ok = storage_px.size() == static_cast<size_t>(W) * H * 4;
            if (storage_ok) {
                const uint8_t* c = &storage_px[((size_t)(H / 2) * W + W / 2) * 4];
                storage_ok = c[0] > 0x80 && c[1] < 0x40 && c[2] < 0x40;
            }
            CHECK(storage_ok,
                  "graphics storage-image descriptor reads texel (0,0) into the framebuffer");
            // Separately exercise the alias case. Both descriptors reference the same decoded bytes
            // in one backend call but need distinct VkImages because their usage and layouts differ.
            prosper::test::render_draws_rgba({sampled_draw, storage_draw}, W, H);
            const auto storage_stats = prosper::test::backend_texture_upload_stats();
            CHECK(storage_stats.references == 2 && storage_stats.unique_uploads == 2,
                  "sampled and storage descriptors never share an incompatible image upload");
        }
    }

    // image_sample_lz (explicit LOD 0): coords (0.75,0.25) -> texel (1,0) = green, same as image_sample.
    const uint32_t lz[] = {
        0x7e0002ffu, 0x3f400000u, 0x7e0202ffu, 0x3e800000u, 0xf09c0f08u, 0x00820000u,
        0xf800080fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> lzf = recompile_fragment(lz, sizeof(lz)/sizeof(lz[0]), &rt);
    bool okLz = !lzf.empty() && lzf[0] == 0x07230203u;
    if (okLz) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, lzf, W, H, nullptr, nullptr, nullptr, &td);
        okLz = px.size() == (size_t)W*H*4;
        if (okLz) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
    }
    printf("  image_sample_lz(0.75,0.25) center=(%u,%u,%u)\n", okLz?rgb[0]:0, okLz?rgb[1]:0, okLz?rgb[2]:0);
    CHECK(okLz && rgb[1] > 0x80 && rgb[0] < 0x40 && rgb[2] < 0x40, "image_sample_lz (explicit LOD 0) samples texel (1,0) = GREEN");

    // image_sample_lz_o (#273 — DOLL FXAA): LOD-0 sample with a packed TEXEL offset in the first
    // vaddr (x=+1 in bits[5:0], y=+1 in bits[13:8]). Sampling (0.25,0.25) with offset (+1,+1) must
    // land on texel (1,1) = WHITE (the offset folds into the normalized coords via the level-0 size).
    const uint32_t lzo[] = {
        0x7e0002ffu, 0x00000101u, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu, 0x3e800000u,
        0xf0dc0f08u, 0x00820000u, 0xf800080fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> lzof = recompile_fragment(lzo, sizeof(lzo)/sizeof(lzo[0]), &rt);
    bool okLzo = !lzof.empty() && lzof[0] == 0x07230203u;
    CHECK(okLzo, "recompiled image_sample_lz_o PS -> SPIR-V");
    if (okLzo) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, lzof, W, H, nullptr, nullptr, nullptr, &td);
        okLzo = px.size() == (size_t)W*H*4;
        if (okLzo) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        printf("  image_sample_lz_o(0.25,0.25,+1,+1) center=(%u,%u,%u)\n", okLzo?rgb[0]:0, okLzo?rgb[1]:0, okLzo?rgb[2]:0);
        CHECK(okLzo && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80,
              "image_sample_lz_o offset (+1,+1) from texel (0,0) samples texel (1,1) = WHITE");
    }

    // image_gather4_lz_o (locks the #296 helper's operand-ID fix): gather the ALPHA channel (dmask
    // 0x8) with a packed (+1,+1) offset — every texel's alpha is 255, so all four gathered values
    // are 1.0 and the export is WHITE. (Before the fix the emitted OpBitFieldSExtract used raw
    // integers 0/6/8 as operand IDs — an invalid module the driver rejects -> nothing renders.)
    const uint32_t g4o[] = {
        0x7e0002ffu, 0x00000101u, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu, 0x3e800000u,
        0xf15c0808u, 0x00820400u, 0xf800000fu, 0x07060504u, 0xbf810000u,
    };
    std::vector<uint32_t> g4of = recompile_fragment(g4o, sizeof(g4o)/sizeof(g4o[0]), &rt);
    bool okG4o = !g4of.empty() && g4of[0] == 0x07230203u;
    CHECK(okG4o, "recompiled image_gather4_lz_o PS -> SPIR-V");
    if (okG4o) {
        std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, g4of, W, H, nullptr, nullptr, nullptr, &td);
        okG4o = px.size() == (size_t)W*H*4;
        if (okG4o) { const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4]; rgb[0]=c[0]; rgb[1]=c[1]; rgb[2]=c[2]; }
        printf("  image_gather4_lz_o alpha center=(%u,%u,%u)\n", okG4o?rgb[0]:0, okG4o?rgb[1]:0, okG4o?rgb[2]:0);
        CHECK(okG4o && rgb[0] > 0x80 && rgb[1] > 0x80 && rgb[2] > 0x80,
              "image_gather4_lz_o gathers alpha=1.0 x4 -> WHITE (valid module, offset decoded)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
