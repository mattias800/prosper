// Mixed cubes need both renderer-generation and guest-face content proofs.
#include "fixtures/render_runner.h"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "shared/live/live_renderer.hpp"
#include "shared/live/depth_cube_source_snapshot.hpp"
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::frontend;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}
int main(int argc, char** argv) {
    const bool control = argc == 2 && std::strcmp(argv[1], "--control") == 0;
    prosper::register_builtin_hle();
    constexpr uint32_t W = 8, H = 8;
    const uint32_t vs[]{0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u,
        0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u, 0x08020D01u, 0x10040B02u,
        0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u};
    auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapNamedFlexibleMemory"));
    auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    uint64_t guest = 0;
    check(map && unmap && map(reinterpret_cast<uint64_t>(&guest), 0x10000, 2, 0,
                             reinterpret_cast<uint64_t>("cube-snapshot"), 0) == 0 && guest,
          "fixture maps real guest backing");
    if (!guest) return 1;
    const size_t face_bytes = tiled_surface_bytes(W, H, 5, 0, 2);
    const size_t stride = face_bytes * 2;
    check(6 * stride <= 0x10000, "fixture includes real stride padding");
    std::memset(reinterpret_cast<void*>(guest), 0x55, 0x10000);
    auto table = std::make_shared<ShaderResourceTable>();
    ShaderResource source{};
    source.cls = ResourceClass::Texture; source.format = DataFormat::Unorm16;
    source.num_components = 1; source.binding = 4; source.sgpr_base = 8;
    source.img_dim = 3; source.depth = 6; source.width = W; source.height = H;
    source.tile_mode = 5; source.layer_stride_bytes = stride;
    source.gpu_addr = guest; source.size = 6 * stride;
    source.mag_filter = source.min_filter = 0;
    table->resources.push_back(source);
    DrawItem draw;
    draw.vs = recompile_vertex(vs, std::size(vs)); draw.vertex_count = 3;
    draw.ps.topology = 3; draw.ps.color_write_mask = 15;
    draw.color0_base = 0x7b000000;
    draw.color0_width = W; draw.color0_height = H; draw.prt = table;
    prosper::frontend::register_live_renderer(".", false);
    auto select_face = [&](unsigned face) {
        const uint32_t ps[]{0x7e0002ffu, 0x3fc00000u, 0x7e0202ffu, 0x3fc00000u,
            0x7e0402ffu, std::bit_cast<uint32_t>(float(face)), 0xf09c0f18u, 0x00820000u,
            0xf800080fu, 0x03020100u, 0xbf810000u};
        draw.fs = recompile_fragment(ps, std::size(ps), table.get());
        check(!draw.fs.empty(), "cube face shader recompiles");
    };
    auto render = [&] { return render_submit_items({draw}, W, H); };
    auto center = [&](const std::vector<uint8_t>& image, unsigned color) {
        if (image.size() != W * H * 4) return false;
        const size_t at = (W * (H / 2) + W / 2) * 4;
        return image[at] == color && image[at+1] == color &&
            image[at+2] == color && image[at+3] == 255;
    };
    auto seed_faces = [&](unsigned count, float value) {
        prosper::test::BackendPersistentResourceGuard guard;
        prosper::test::invalidate_persistent_ds_guest_write(table->resources[0].gpu_addr, 6 * stride);
        std::vector<GpuCaptureDsSeed> seeds;
        for (unsigned face = 0; face < count; ++face) {
            GpuCaptureDsSeed seed;
            seed.depth_read_base = seed.depth_write_base = table->resources[0].gpu_addr;
            seed.width = W; seed.height = H; seed.slice = face;
            seed.format = GpuCaptureDsFormat::D32Float; seed.depth_valid = true;
            seed.depth.resize(W * H * sizeof(float));
            for (size_t at = 0; at < seed.depth.size(); at += sizeof(float))
                std::memcpy(seed.depth.data() + at, &value, sizeof(value));
            seeds.push_back(std::move(seed));
        }
        std::string error;
        const bool restored = restore_gpu_replay_ds_seeds(seeds, error);
        if (!restored) std::fprintf(stderr, "DS restore: %s\n", error.c_str());
        check(restored, "real Vulkan depth faces restore");
        // Replay seeds establish GPU pixels; mark the same depth-write generations that an
        // ordinary producer draw records so the mixed-authority cache is actually eligible.
        for (auto& [key, image] : prosper::test::persistent_ds_cache())
            if (key.dr == table->resources[0].gpu_addr && key.slice < count && image.depth_valid)
                prosper::test::note_persistent_ds_depth_write(image, true, true);
    };
    seed_faces(5, 0.25f);
    select_face(5);
    reset_texture_decode_scope_stats();
    check(center(render(), 85), "missing face comes from guest bytes");
    auto stats = texture_decode_scope_stats();
    check(stats.cube_snapshot_guest_bytes == face_bytes &&
          stats.cube_snapshot_renderer_bytes == 5 * face_bytes &&
          stats.late_snapshot_read_bytes == (control ? 6 : 1) * face_bytes,
          "five renderer-owned faces need no guest snapshot reads, with an eager control");
    reset_texture_decode_scope_stats();
    check(center(render(), 85) && texture_decode_scope_stats().decodes == 0,
          "unchanged mixed authority reuses the decoded cube");
    // Deliberately no GPU notification: the renderer generation must remain identical, so
    // only exact validation of the missing guest face can detect this CPU write.
    std::memset(reinterpret_cast<void*>(guest + 5 * stride), 0xaa, face_bytes);
    reset_texture_decode_scope_stats();
    check(center(render(), 170) && texture_decode_scope_stats().decodes == 1,
          "CPU write to missing face invalidates unchanged renderer generation");
    select_face(0);
    check(center(render(), 64), "retained face overrides stale guest bytes");
    seed_faces(5, 0.75f);
    check(center(render(), 191), "renderer rewrite replaces its face generation");
    seed_faces(6, 0.5f);
    select_face(5);
    reset_texture_decode_scope_stats();
    check(center(render(), 128), "new renderer face replaces former guest authority");
    stats = texture_decode_scope_stats();
    check(stats.cube_snapshot_guest_bytes == 0 && stats.cube_snapshot_renderer_bytes == 6 * face_bytes &&
          stats.late_snapshot_read_bytes == (control ? 6 * face_bytes : 0),
          "fully renderer-owned cube retains no guest snapshot");
    seed_faces(5, 0.25f);
    check(center(render(), 170), "removing renderer face restores current guest authority");

    // A failed readback must not freeze fallback pixels under the unchanged renderer key.
    seed_faces(5, 0.75f);
    select_face(0);
    prosper::test::depth_cube_readback_failure_once() = true;
    reset_texture_decode_scope_stats();
    const auto fallback = render();
    check(!fallback.empty() && texture_decode_scope_stats().cube_snapshot_refusals == 1,
          "injected readback failure declines persistent cube admission");
    reset_texture_decode_scope_stats();
    check(center(render(), 191) && texture_decode_scope_stats().decodes == 1,
          "same renderer generation retries after readback failure");

    // The mapped path keeps successful faces private until every selected face has been
    // acquired. Failing after face 0 must leave the destination on the ordinary fallback and
    // must not admit a cache entry under the unchanged renderer generation.
    if (!std::getenv("PROSPER_NO_MAPPED_DEPTH_CUBE")) {
        seed_faces(5, 0.75f); // Advance the renderer generation beyond the successful retry.
        prosper::test::depth_cube_readback_failure_after_faces() = 1;
        reset_texture_decode_scope_stats();
        const auto late_fallback = render();
        check(!late_fallback.empty() && texture_decode_scope_stats().cube_snapshot_refusals == 1,
              "later-face failure declines partial mapped cube publication");
        reset_texture_decode_scope_stats();
        check(center(render(), 191) && texture_decode_scope_stats().decodes == 1,
              "later-face failure retries the same renderer generation");
    }

    seed_faces(5, 0.5f);
    prosper::test::depth_cube_readback_failure_once() = true;
    reset_texture_decode_scope_stats();
    check(center(render_submit_items({draw, draw}, W, H), 128),
          "second draw in the same submit retries renderer authority after failure");
    stats = texture_decode_scope_stats();
    check(stats.decodes == 2 && stats.cube_snapshot_refusals == 1,
          "failed fallback cannot enter the submit-local identity cache");

    // The decoder's established short-source fallback is white when less than one logical face
    // is readable. The cache still records the exact one-byte prefix, not zero-filled padding.
    table->resources[0].gpu_addr = guest + 0x10000 - 5 * stride - 1;
    seed_faces(5, 0.25f);
    select_face(5);
    reset_texture_decode_scope_stats();
    check(center(render(), 255), "short missing face preserves the established decoder fallback");
    check(texture_decode_scope_stats().cube_snapshot_guest_bytes == 1,
          "short guest face retains only its actual prefix");
    reset_texture_decode_scope_stats();
    check(center(render(), 255) && texture_decode_scope_stats().decodes == 0,
          "short face has a reusable exact snapshot");

    // Equal concatenations alone cannot distinguish a readability boundary moving between faces.
    DepthCubeSourceLayout layout{0x1000, 8, 8};
    DepthCubeSourceSnapshot snapshot;
    std::vector<uint8_t> scratch;
    size_t reads = 0;
    std::array<size_t, 6> prefixes{2, 4, 0, 0, 0, 0};
    auto copy = [&](uint8_t* dst, uint64_t address, size_t) {
        size_t count = prefixes[(address - layout.base) / layout.stride];
        std::memset(dst, 7, count); return count;
    };
    const size_t size = capture_depth_cube_source(layout, 0x3c, false, scratch, snapshot, copy, reads);
    scratch.resize(size);
    auto span = [&](uint64_t address, size_t) { return prefixes[(address-layout.base)/layout.stride]; };
    auto equal = [&](const uint8_t*, uint64_t, size_t bytes, size_t& compared) {
        compared = bytes; return true;
    };
    size_t compared = 0;
    check(equal_depth_cube_source(layout, snapshot, scratch, span, equal, compared),
          "two independently short faces validate initially");
    prefixes[0] = 3; prefixes[1] = 3;
    check(!equal_depth_cube_source(layout, snapshot, scratch, span, equal, compared),
          "same total and identical concatenation cannot hide changed per-face readable lengths");
    check(!DepthCubeSourceLayout{UINT64_MAX - 8, 8, 8}.fits(UINT64_MAX - 8, 8),
          "overflowing face addresses are rejected");
    check(!layout.fits(layout.base, 5 * layout.stride), "full face ranges must fit the mutation-proof span");
    unmap(guest, 0x10000, 0, 0, 0, 0);
    std::printf("depth cube source snapshot: %d failures\n", failures);
    return failures ? 1 : 0;
}
