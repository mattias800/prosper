// Exact GPU FP16 storage initialization must preserve subsequent graphics writes.
#include "fixtures/render_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "shared/live/live_renderer.hpp"
#include <bit>

using namespace prosper::gpu;
using namespace prosper::test;
static int failures = 0;
static void check(bool ok, const char *what) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", what);
    failures += !ok;
}
int main() {
    prosper::frontend::register_live_renderer("", false);
    const bool unshared = std::getenv("PROSPER_NO_BACKEND_TEXTURE_SHARE") != nullptr;
    const uint32_t vs[]{0x36020081u, 0x2c040081u, 0x7e020d01u, 0x7e040d02u, 0x7e0a02f6u,
                        0x7e0c02f2u, 0x10020b01u, 0x08020d01u, 0x10040b02u, 0x08040d02u,
                        0x7e060280u, 0x7e0802f2u, 0xf80008cfu, 0x04030201u, 0xbf810000u};
    const uint32_t read_ps[]{0x7e000280u, 0x7e020280u, 0xf0000f08u, 0x00020000u,
                             0xf800000fu, 0x03020100u, 0xbf810000u};
    // One fragment writes (1,.25,.125,.5) to texel(0,0); every other texel is untouched.
    const uint32_t write_ps[]{0x7e0002ffu, 0x3f800000u, 0x7e0202ffu, 0x3e800000u, 0x7e0402ffu,
                              0x3e000000u, 0x7e0602ffu, 0x3f000000u, 0x7e080280u, 0x7e0a0280u,
                              0xf0200f08u, 0x00020004u, 0xf800000fu, 0x03020100u, 0xbf810000u};
    constexpr uint32_t w = 129, h = 129;
    const std::array<uint32_t, 4> written{0x3f800000, 0x3e800000, 0x3e000000, 0x3f000000};
    for (uint32_t components : {2u, 4u}) {
        const uint64_t address = 0x3407a00000ull + components * 0x1000000ull;
        const size_t bytes = tiled_surface_bytes(w, h, 27, 0, components * 2);
        std::vector<uint8_t> tiled(bytes, 0xa5), linear(size_t(w) * h * components * 2);
        const uint16_t halves[]{0x3800, 0x3c00, 0x3400, 0x3c00};
        for (size_t i = 0; i < size_t(w) * h; ++i)
            std::memcpy(linear.data() + i * components * 2, halves, components * 2);
        tile_surface(tiled.data(), linear.data(), w, h, 27, 0, components * 2);
        auto copy = [&](uint8_t *dst, uint64_t addr, size_t size) -> size_t {
            if (addr != address || size != bytes)
                return 0;
            std::memcpy(dst, tiled.data(), size);
            return size;
        };
        auto upload = prepare_render_gpu_detile(w, h, 1, address, 0, 0, copy, components,
                                                Float16DetileOutput::RawUvec4, true);
        check(unshared
                  ? !upload
                  : upload && upload->readback.mapped && upload->output_bytes == size_t(w) * h * 16,
              "writable FP16 preparation owns one mapped result or refuses unshared images");
        if (!unshared && !upload)
            return 1;
        ShaderResourceTable table;
        ShaderResource source{};
        source.cls = ResourceClass::StorageImage;
        source.binding = 4;
        source.sgpr_base = 8;
        source.img_dim = 1;
        source.width = w;
        source.height = h;
        source.depth = 1;
        source.format = DataFormat::Float16;
        source.num_components = components;
        source.tile_mode = 27;
        source.gpu_addr = address;
        source.size = bytes;
        source.host_data = tiled.data();
        source.host_data_size = bytes;
        table.resources.push_back(source);
        BackendDraw draw;
        draw.vs = recompile_vertex(vs, std::size(vs));
        draw.vcount = 3;
        const auto read = recompile_fragment(read_ps, std::size(read_ps), &table);
        const auto write = recompile_fragment(write_ps, std::size(write_ps), &table);
        check(!read.empty() && !write.empty(), "portable storage readers and writers compile");
        if (upload) {
            FrameResource plane;
            plane.binding = 4;
            plane.set = 1;
            plane.img_dim = 1;
            plane.tw = w;
            plane.th = h;
            plane.is_storage_image = true;
            plane.storage_image_numeric_class = SpirvImageNumericClass::Uint;
            plane.texture_format = VK_FORMAT_R32G32B32A32_UINT;
            plane.gpu_detile = upload;
            draw.fs = read;
            draw.R = {plane};
            const auto old_pixels = render_draws_rgba({draw}, 1, 1);
            check(old_pixels ==
                      std::vector<uint8_t>({128, 255, uint8_t(components == 2 ? 0 : 64), 255}),
                  "storage shader reads exact GPU-expanded channels through UINT image ABI");
            std::vector<uint32_t> result;
            unsigned callbacks = 0;
            plane.storage_image_writeback = [&](const uint8_t *data, size_t size) {
                ++callbacks;
                result.resize(size / 4);
                std::memcpy(result.data(), data, size);
            };
            // A plain readonly upload cannot stand in for a writable one without its result lease.
            auto no_result = prepare_render_gpu_detile(w, h, 1, address, 0, 0, copy, components,
                                                       Float16DetileOutput::RawUvec4);
            plane.gpu_detile = no_result;
            check(no_result && !backend_texture_plane_span_valid(plane),
                  "writable GPU storage without a result mapping is refused before submission");
            plane.gpu_detile = upload;
            draw.fs = write;
            draw.R = {plane};
            // A previously prepared owner must also be refused by backend admission
            // if two unshared images would otherwise overwrite its single result lease.
#if defined(_WIN32)
            _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1");
#else
            setenv("PROSPER_NO_BACKEND_TEXTURE_SHARE", "1", 1);
#endif
            const auto before_unshared = gpu_detile_recordings().load();
            check(render_draws_rgba({draw, draw}, 1, 1).empty() && callbacks == 0 &&
                      gpu_detile_recordings().load() == before_unshared,
                  "backend refuses duplicate unshared writable images before either can record");
#if defined(_WIN32)
            _putenv_s("PROSPER_NO_BACKEND_TEXTURE_SHARE", "");
#else
            unsetenv("PROSPER_NO_BACKEND_TEXTURE_SHARE");
#endif
            BackendSubmissionBatch batch;
            BackendColorTarget target{address + 0x100000, false, false};
            const auto no_color =
                render_draws_rgba({draw}, 1, 1, nullptr, nullptr, false, &target, nullptr, nullptr,
                                  nullptr, &batch, false, nullptr, false);
            check(
                no_color.empty() && !batch.pending() && callbacks == 1 &&
                    result.size() == size_t(w) * h * 4,
                "storage writeback completes and publishes without requiring framebuffer readback");
            bool exact = result.size() == size_t(w) * h * 4;
            for (size_t texel = 0; exact && texel < size_t(w) * h; ++texel)
                for (uint32_t c = 0; c < 4; ++c) {
                    const uint32_t expected =
                        texel == 0       ? written[c]
                        : c < components ? std::bit_cast<uint32_t>(half_to_float(halves[c]))
                        : c == 3         ? 0x3f800000u
                                         : 0u;
                    exact &= result[texel * 4 + c] == expected;
                }
            check(exact, "completed result preserves both the stored texel and all untouched GPU "
                         "conversions");
            auto overlapping = prepare_render_gpu_detile(w, h, 1, address, 0, 0, copy, components,
                                                          Float16DetileOutput::RawUvec4, true);
            check(overlapping && overlapping->readback.buffer != upload->readback.buffer,
                  "completed backend cleanup cannot recycle a result mapping while its owner is held");
            plane.storage_image_writeback = {};
            draw.fs = read;
            draw.R = {plane};
            check(render_draws_rgba({draw}, 1, 1) == old_pixels,
                  "re-recording an immutable snapshot does not inherit prior storage-image writes");
        }
        DrawItem item;
        item.vs = draw.vs;
        item.fs = write;
        item.vertex_count = 3;
        item.ps.topology = 3;
        item.ps.color_write_mask = 0xf;
        item.color0_base = address + 0x200000;
        item.prt = std::make_shared<ShaderResourceTable>(table);
        const auto before = gpu_detile_recordings().load();
        const uint64_t read_dispatch = PROSPER_ENV_ON("PROSPER_NO_GPU_DETILE_2D") ? 0 : 1;
        const uint64_t write_dispatch = unshared ? 0 : read_dispatch;
        render_submit_items({item}, 1, 1);
        uint16_t actual[4]{};
        std::memcpy(actual, tiled.data(), components * 2);
        const uint16_t expected_halves[]{0x3c00, 0x3400, 0x3000, 0x3800};
        check(
            !std::memcmp(actual, expected_halves, components * 2) &&
                gpu_detile_recordings().load() == before + write_dispatch,
            "live graphics store reaches tiled guest backing with GPU preparation or CPU control");
        item.fs = read;
        const auto fresh = render_submit_items({item}, 1, 1);
        check(fresh == std::vector<uint8_t>({255, 64, uint8_t(components == 2 ? 0 : 32),
                                             uint8_t(components == 2 ? 255 : 128)}) &&
                  gpu_detile_recordings().load() == before + write_dispatch + read_dispatch,
              "later live storage read uses the newly written guest version, not the old snapshot");
        const auto before_short = tiled;
        const auto before_short_dispatches = gpu_detile_recordings().load();
        item.prt->resources[0].host_data_size = bytes - 1;
        item.fs = write;
        render_submit_items({item}, 1, 1);
        check(tiled == before_short && gpu_detile_recordings().load() == before_short_dispatches,
              "short storage backing refuses both GPU preparation and guest writes");
    }
    return failures ? 1 : 0;
}
