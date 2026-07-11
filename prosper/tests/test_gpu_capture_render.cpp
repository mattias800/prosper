#include "../src/gpu/gpu_capture.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../frontends/shared/live_renderer.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

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

    prosper::frontend::register_live_renderer(".", false);
    std::vector<uint8_t> pixels = render_submit_items(replay.items, W, H);
    CHECK(pixels.size() == static_cast<size_t>(W) * H * 4, "replayed draw renders through live backend");
    if (!pixels.empty()) {
        const uint8_t* center = &pixels[(static_cast<size_t>(H / 2) * W + W / 2) * 4];
        CHECK(center[0] > 0xC0 && center[1] < 0x40 && center[2] < 0x40,
              "replay samples captured red texel from owned backing");
    }

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n"); return 0;
}
