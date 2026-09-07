// Drive the shipped texture upload, cube sampling and submission ownership path.
#include "fixtures/render_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "shared/live/live_renderer.hpp"
#include <bit>

using namespace prosper::gpu;
using namespace prosper::test;
static int failures = 0;
static void check(bool ok, const char* message) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", message);
    failures += !ok;
}

int main(int argc, char** argv) {
    const bool pressure = argc == 2 && std::strcmp(argv[1], "--staging-pressure") == 0;
    constexpr uint32_t width = 129, height = 65, faces = 6;
    constexpr uint64_t base = 0x200000000ull, offset = 4096;
    const uint64_t face_bytes = tiled_surface_bytes(width, height, 27, 0, 8);
    const uint64_t stride = face_bytes + offset + 4096;
    std::vector<uint8_t> guest(stride * faces, 0xa5);
    const uint16_t colors[6][4]{{0x3c00, 0, 0, 0x3c00}, {0, 0x3c00, 0, 0x3c00},
        {0, 0, 0x3c00, 0x3c00}, {0x3800, 0x3400, 0x3000, 0x3c00},
        {0x7c00, 0xfc00, 0x7fff, 0x3c00}, {0xbc00, 0x3c00, 0x3c00, 0x3c00}};
    std::vector<uint8_t> expected(size_t(width) * height * faces * 4);
    auto seed = [&](unsigned rotation) {
        for (uint32_t face = 0; face < faces; ++face) {
            std::vector<uint16_t> linear(size_t(width) * height * 4);
            for (size_t i = 0; i < linear.size(); ++i) {
                const uint16_t value = colors[(face + rotation) % faces][i % 4];
                linear[i] = value;
                const float f = half_to_float(value);
                expected[size_t(face) * width * height * 4 + i] =
                    !std::isfinite(f) || f <= 0 ? 0 : f >= 1 ? 255 : std::lround(f * 255);
            }
            tile_surface(guest.data() + face * stride + offset,
                         reinterpret_cast<const uint8_t*>(linear.data()), width, height, 27, 0, 8);
        }
    };
    seed(0);
    size_t reads = 0;
    auto copy = [&](uint8_t* dst, uint64_t addr, size_t bytes) -> size_t {
        ++reads;
        if (addr < base || addr - base > guest.size() || bytes > guest.size() - (addr - base))
            return 0;
        std::memcpy(dst, guest.data() + (addr - base), bytes);
        return bytes;
    };
    auto upload = prepare_render_gpu_detile(width, height, faces, base, stride, offset, copy);
    check(upload && reads == 6, "six complete faces snapshot from padded strides and selected mip offset");
    if (!upload) return 1;
    const auto& ctx = render_vk_ctx();
    auto reject = [&](uint32_t w, uint32_t h, uint32_t f, uint64_t addr,
                      uint64_t step, uint64_t off, VkPhysicalDeviceLimits limits) {
        const size_t before = reads;
        const auto declined = prepare_gpu_detile_upload(*upload->program, ctx.phys, limits,
                                                        w, h, f, addr, step, off, copy);
        check(!declined && reads == before, "invalid layout/device limits decline before source reads");
    };
    reject(0, height, faces, base, stride, offset, ctx.detile_limits);
    reject(width, height, 0, base, stride, offset, ctx.detile_limits);
    reject(width, height, faces, UINT64_MAX - 8, stride, offset, ctx.detile_limits);
    reject(width, height, faces, base, face_bytes, offset, ctx.detile_limits);
    auto limited = ctx.detile_limits; limited.maxStorageBufferRange = 128;
    reject(width, height, faces, base, stride, offset, limited);
    limited = ctx.detile_limits; limited.maxComputeWorkGroupCount[0] = 1;
    reject(width, height, faces, base, stride, offset, limited);
    limited = ctx.detile_limits; limited.maxComputeWorkGroupInvocations = 64;
    reject(width, height, faces, base, stride, offset, limited);
    const auto short_source = prepare_render_gpu_detile(width, height, faces, base, stride, offset,
        [&](uint8_t* dst, uint64_t addr, size_t bytes) { return copy(dst, addr, bytes) - 1; });
    check(!short_source, "short source declines GPU materialization instead of publishing partial pixels");
    const size_t before_last_reads = reads;
    const auto before_last_recordings = gpu_detile_recordings().load();
    const auto short_last = prepare_render_gpu_detile(width, height, faces, base, stride, offset,
        [&](uint8_t* dst, uint64_t addr, size_t bytes) {
            const size_t got = copy(dst, addr, bytes);
            return addr == base + 5 * stride + offset ? got - 1 : got;
        });
    check(!short_last && reads == before_last_reads + 6 &&
              gpu_detile_recordings().load() == before_last_recordings,
          "short sixth face releases five successful snapshots without recording a conversion");

    const uint32_t vs[]{0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u,
        0x7e0a02f6u, 0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
        0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u};
    ShaderResourceTable table;
    ShaderResource texture{};
    texture.cls = ResourceClass::Texture; texture.binding = 4; texture.sgpr_base = 8;
    texture.img_dim = 3; texture.width = width; texture.height = height;
    texture.format = DataFormat::Float16; texture.num_components = 4;
    table.resources.push_back(texture);
    FrameResource resource;
    resource.binding = 4; resource.set = 1; resource.img_dim = 3;
    resource.tw = width; resource.th = height * faces;
    resource.gpu_detile = upload;
    BackendDraw draw;
    draw.vs = recompile_vertex(vs, std::size(vs)); draw.vcount = 3;
    auto select_face = [&](uint32_t face) {
        // Cube-processed coordinates: (1.5,1.5,face) -> center of the selected face.
        const uint32_t ps[]{0x7e0002ffu, 0x3fc00000u, 0x7e0202ffu, 0x3fc00000u,
            0x7e0402ffu, std::bit_cast<uint32_t>(float(face)), 0xf09c0f18u, 0x00820000u,
            0xf800080fu, 0x03020100u, 0xbf810000u};
        draw.fs = recompile_fragment(ps, std::size(ps), &table);
        check(!draw.fs.empty(), "cube sampling shader recompiles");
    };
    auto center = [](const std::vector<uint8_t>& pixels) {
        std::array<uint8_t, 4> rgba{};
        if (pixels.size() == 16 * 16 * 4) std::memcpy(rgba.data(), pixels.data() + (8 * 16 + 8) * 4, 4);
        return rgba;
    };
    auto expected_face = [&](unsigned face) {
        std::array<uint8_t, 4> rgba{};
        std::memcpy(rgba.data(), expected.data() + size_t(face) * width * height * 4, 4);
        return rgba;
    };
    const auto before = gpu_detile_recordings().load();
    for (uint32_t face = 0; face < faces; ++face) {
        select_face(face);
        draw.R = {resource};
        const auto actual = render_draws_rgba({draw, draw}, 16, 16);
        check(actual.size() == 16 * 16 * 4 && center(actual) == expected_face(face),
              "GPU detile reaches the selected cube face through real sampling");
        check(backend_texture_upload_stats().unique_uploads == 1 &&
                  backend_texture_upload_stats().gpu_detile_dispatches == 1 &&
                  backend_texture_upload_stats().gpu_detile_source_bytes == upload->source_bytes,
              "same immutable upload is shared within one render call");
    }
    check(gpu_detile_recordings().load() == before + 6,
          "six render calls recorded six GPU conversions, with no CPU pixel payload");
    select_face(3);
    resource.swizzle[0] = 6; resource.swizzle[2] = 4;
    draw.R = {resource};
    auto permuted = expected_face(3); std::swap(permuted[0], permuted[2]);
    check(center(render_draws_rgba({draw}, 16, 16)) == permuted,
          "sampler component swizzle remains independent of converted storage");
    resource.swizzle[0] = 4; resource.swizzle[2] = 6;
    // The established CPU payload remains accepted after failed GPU admission.
    resource.gpu_detile.reset(); resource.tex_rgba = expected.data(); draw.R = {resource};
    check(center(render_draws_rgba({draw}, 16, 16)) == expected_face(3),
          "CPU fallback samples the same canonical pixels");
    resource.tex_rgba = nullptr; resource.gpu_detile = upload;
    draw.R = {resource};
    const VkDevice saved_device = upload->device;
    upload->device = VK_NULL_HANDLE;
    check(render_draws_rgba({draw}, 16, 16).empty(), "foreign device identity is refused before recording");
    upload->device = saved_device;
    auto* mutable_program = const_cast<GpuDetilePipeline*>(upload->program);
    mutable_program->device = VK_NULL_HANDLE;
    check(render_draws_rgba({draw}, 16, 16).empty(), "foreign program device is refused before recording");
    mutable_program->device = saved_device;

    // First publication is captured, then guest backing changes at the same VA
    // before submission. Both outputs must retain their own immutable version.
    select_face(0);
    draw.R = {resource};
    BackendSubmissionBatch batch;
    BackendColorTarget old_target{0x34070001u, false, false};
    const auto deferred = render_draws_rgba({draw}, 16, 16, nullptr, nullptr, false, &old_target,
                                           nullptr, nullptr, nullptr, &batch, false, nullptr, false);
    const VkBuffer pending_input = upload->input;
    const VkDeviceMemory pending_memory = upload->input_memory;
    const auto pending_lease = upload->input_lease;
    const void* pending_mapping = upload->input_mapped;
    std::weak_ptr<GpuDetileUpload> old_owner = upload;
    upload.reset(); resource.gpu_detile.reset(); draw.R.clear();
    check(deferred.empty() && batch.pending() && !old_owner.expired(),
          "pending submission retains detile buffers after frontend ownership ends");
    if (pressure) {
        const auto evictions = mapped_staging_cache().evictions;
        // Fill the idle budget with other shapes while the first cube is still
        // owned by the pending GPU submission. Its input is never an eviction candidate.
        for (VkDeviceSize bytes : {2ull * 1024 * 1024, 2ull * 1024 * 1024 + 16}) {
            auto filler = acquire_mapped_staging(ctx.dev, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                [&](uint32_t bits) { return render_memory_type(ctx.phys, bits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); });
            check(filler.mapped && release_mapped_staging(ctx.dev, filler.buffer, filler.memory,
                      filler.mapped, filler.lease), "pressure allocation is released through the cache");
        }
        check(mapped_staging_cache().evictions > evictions && !old_owner.expired() && batch.pending(),
              "idle eviction occurs while the earlier cube submission remains pending");
    }
    seed(1);
    auto changed = prepare_render_gpu_detile(width, height, faces, base, stride, offset, copy);
    check(changed && changed->input != pending_input && changed->input_mapped != pending_mapping,
          "overlapping pending and new uploads cannot share input storage or mapping");
    if (!changed) return 1;
    resource.gpu_detile = changed; draw.R = {resource};
    const auto current = render_draws_rgba({draw}, 16, 16, nullptr, nullptr, false, nullptr,
                                           nullptr, nullptr, nullptr, &batch, true);
    check(center(current) == expected_face(0) && !batch.pending() && old_owner.expired(),
          "later publication samples new bytes and completed submission releases old buffers");
    std::vector<uint8_t> original;
    std::string error;
    {
        BackendPersistentResourceGuard guard;
        check(readback_persistent_color_target(old_target.persistent_id, 16, 16,
                  VK_FORMAT_R8G8B8A8_UNORM, original, error), "first publication target remains readable");
    }
    check(center(original) == std::array<uint8_t, 4>{255, 0, 0, 255},
          "guest mutation before flush did not alter the earlier publication");

    auto& staging = mapped_staging_cache();
    const auto foreign_releases = staging.cross_device_skips;
    check(release_mapped_staging(VK_NULL_HANDLE, changed->input, changed->input_memory,
                                changed->input_mapped, changed->input_lease, false) &&
              staging.cross_device_skips == foreign_releases + 1 &&
              staging.in_use.count(changed->input_lease),
          "wrong-device release cannot destroy or surrender an owned input");
    if (!PROSPER_ENV_ON("PROSPER_NO_GPU_DETILE_INPUT_REUSE") && mapped_staging_reuse_enabled()) {
        const auto duplicates = staging.double_releases;
        const bool safe_duplicate = release_mapped_staging(changed->device, pending_input, pending_memory,
                                      const_cast<void*>(pending_mapping), pending_lease, false) &&
            staging.double_releases == duplicates + 1 && staging.owned.count(pending_lease);
        check(safe_duplicate, "duplicate release cannot free a retained mapping even when reuse is disabled");
        if (!safe_duplicate) return 1; // Do not dereference a freed mapping in the negative control.
    }

    // Completion returned the old input to the pool. The newer upload remains
    // owned, so reacquisition can reuse only an idle allocation, never that one.
    const auto changed_face0 = expected_face(0);
    seed(2);
    const size_t before_reuse_reads = reads;
    auto recycled = prepare_render_gpu_detile(width, height, faces, base, stride, offset, copy);
    check(recycled && recycled->input_reused && recycled->input == pending_input &&
              recycled->input_mapped == pending_mapping && recycled->input != changed->input,
          "final submission owner release permits reuse of the same live input mapping");
    check(recycled && reads == before_reuse_reads + faces,
          "reused allocation still snapshots all six changed faces");
    if (!recycled) return 1;
    resource.gpu_detile = recycled;
    for (uint32_t face = 0; face < faces; ++face) {
        select_face(face); draw.R = {resource};
        check(center(render_draws_rgba({draw}, 16, 16)) == expected_face(face),
              "reused mapped input samples fresh pixels on every cube face");
    }
    resource.gpu_detile = changed; select_face(0); draw.R = {resource};
    check(center(render_draws_rgba({draw}, 16, 16)) == changed_face0,
          "an overlapping upload retains its earlier snapshot after another input is reused");

    // Exercise actual frontend admission and refusal with captured guest backing.
    // The missing last byte is unused tile padding, so CPU fallback has the same
    // visible pixels; its path must be proved by the conversion counter as well.
    prosper::frontend::register_live_renderer("", false);
    select_face(5);
    DrawItem item;
    item.vs = draw.vs; item.fs = draw.fs; item.vertex_count = 3;
    item.ps.topology = 3; item.ps.color_write_mask = 0xf;
    item.color0_base = 0x34071000;
    item.prt = std::make_shared<ShaderResourceTable>(table);
    auto& source = item.prt->resources[0];
    source.gpu_addr = base; source.depth = 6; source.tile_mode = 27;
    source.layer_stride_bytes = stride; source.layer_mip_offset_bytes = offset;
    source.size = guest.size(); source.host_data = guest.data(); source.host_data_size = guest.size();
    const auto frontend_before = gpu_detile_recordings().load();
    const auto live_gpu = render_submit_items({item}, 16, 16);
    check(live_gpu.size() == 16 * 16 * 4 && center(live_gpu) == expected_face(5) &&
              gpu_detile_recordings().load() == frontend_before + 1,
          "live frontend admits full captured cube backing and samples its GPU conversion");
    source.host_data_size = 5 * stride + offset + face_bytes - 1;
    item.color0_base = 0x34072000;
    const auto live_fallback = render_submit_items({item}, 16, 16);
    check(live_fallback.size() == 16 * 16 * 4 && center(live_fallback) == expected_face(5) &&
              gpu_detile_recordings().load() == frontend_before + 1,
          "live frontend refusal allocates CPU fallback and preserves last-face pixels");
    return failures ? 1 : 0;
}
