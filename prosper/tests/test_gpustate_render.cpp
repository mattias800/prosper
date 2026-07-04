// test_gpustate_render — the full back-half spine, end to end: a GpuState (exactly what the
// CommandProcessor folds a Dcb into) -> recompile the vertex+pixel shaders from their SHADER_PGM
// addresses -> resolve the pipeline state -> render a real Vulkan frame.
//
// prosper maps the guest 1:1, so a guest GPU address IS a host pointer. We place the RDNA2 shader
// blobs at 256-byte-aligned addresses, encode those addresses into SPI_SHADER_PGM_{LO,HI}_{ES,PS}
// (the RDNA2 (lo<<8)|((hi&0xff)<<40) convention), and let the bridge read them straight back out —
// the same path a real submitted command buffer will drive once the boot clears resource residency.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/render_state.hpp"
#include "../src/gpu/pm4_registers.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Fullscreen-triangle vertex shader + solid-green pixel shader (llvm-mc gfx1030), 256-byte aligned
// so their host addresses round-trip through the RDNA2 (lo<<8)|((hi&0xff)<<40) address encoding.
alignas(256) static const uint32_t kVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
    0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
};
alignas(256) static const uint32_t kPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};

// Write a shader's host address into a SHADER_PGM_{LO,HI} register pair (RDNA2 encoding).
static void set_pgm(GpuState& st, uint32_t lo_off, uint32_t hi_off, const void* p) {
    uint64_t a = (uint64_t)(uintptr_t)p;
    st.sh[lo_off] = (uint32_t)((a >> 8) & 0xFFFFFFFFu);
    st.sh[hi_off] = (uint32_t)((a >> 40) & 0xFFu);
}

int main() {
    printf("== test_gpustate_render ==\n");
    const uint32_t W = 64, H = 64;

    // Build the GpuState the CommandProcessor would produce for this draw.
    GpuState st;
    set_pgm(st, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, kVs);   // vertex/export stage
    set_pgm(st, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kPs);   // pixel stage
    st.cx[P::VGT_PRIMITIVE_TYPE] = 4;      // triangle list
    st.cx[P::CB_TARGET_MASK]     = 0xF;    // write RGBA
    st.draws.push_back({3});               // one draw, 3 vertices

    RenderState rs = extract_render_state(st);
    CHECK(rs.es_addr == (uint64_t)(uintptr_t)kVs, "ES shader address round-trips through the register pair");
    CHECK(rs.ps_addr == (uint64_t)(uintptr_t)kPs, "PS shader address round-trips through the register pair");
    if (rs.es_addr != (uint64_t)(uintptr_t)kVs || rs.ps_addr != (uint64_t)(uintptr_t)kPs) {
        printf("== FAIL: address encoding ==\n"); return 1;   // (would need >2^48 or misalignment to fail)
    }

    // The bridge: recompile each stage from its address, resolve fixed-function state, render.
    std::vector<uint32_t> vert = recompile_vertex((const uint32_t*)rs.es_addr, sizeof(kVs)/sizeof(kVs[0]));
    std::vector<uint32_t> frag = recompile_fragment((const uint32_t*)rs.ps_addr, sizeof(kPs)/sizeof(kPs[0]));
    ResolvedPipelineState ps = resolve_pipeline_state(rs);
    CHECK(!vert.empty() && !frag.empty(), "recompiled both stages straight from the GpuState addresses");
    CHECK(ps.topology == 3 && ps.color_write_mask == 0xF, "resolved pipeline state from the GpuState registers");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H, &ps);
    CHECK(px.size() == (size_t)W*H*4, "rendered a frame from the GpuState");
    if (px.size() != (size_t)W*H*4) { printf("== FAIL: render failed ==\n"); return 1; }

    // Fullscreen triangle + green pixel shader -> every sampled pixel green.
    auto isGreen = [&](uint32_t x, uint32_t y){ const uint8_t* p=&px[((size_t)y*W+x)*4]; return p[1]>0x80 && p[0]<0x40 && p[2]<0x40; };
    uint32_t green=0, total=0; const uint32_t xs[]={0,W/2,W-1}, ys[]={0,H/2,H-1};
    for (uint32_t y:ys) for (uint32_t x:xs){ total++; if(isGreen(x,y)) green++; }
    const uint8_t* c=&px[((size_t)(H/2)*W+W/2)*4];
    printf("  center=(%u,%u,%u,%u)  green %u/%u\n", c[0],c[1],c[2],c[3], green, total);
    CHECK(green==total, "GpuState -> recompiled shaders -> resolved pipeline -> GREEN frame (full spine)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
