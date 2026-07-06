// test_gpu_execute — the GPU-executor core spine (Stage A): a GpuState (exactly what SubmitDcb folds a
// Dcb into) -> execute_gpustate() [recompile shaders from their PGM addresses + resolve pipeline] ->
// a caller-supplied Vulkan render -> present_write_frame -> present_readback. Proves the executor entry
// point that agc_driver_submit_dcb will call, and the scanout round-trip, end to end on llvmpipe.
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/videoout_present.hpp"
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

// Fullscreen-triangle VS + solid-green PS (llvm-mc gfx1030), 256-aligned so their host addresses round-trip
// through the RDNA2 (lo<<8)|((hi&0xff)<<40) SHADER_PGM encoding. (Same blobs as test_gpustate_render.)
alignas(256) static const uint32_t kVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
    0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
};
alignas(256) static const uint32_t kPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};
static void set_pgm(GpuState& st, uint32_t lo_off, uint32_t hi_off, const void* p) {
    uint64_t a = (uint64_t)(uintptr_t)p;
    st.sh[lo_off] = (uint32_t)((a >> 8) & 0xFFFFFFFFu);
    st.sh[hi_off] = (uint32_t)((a >> 40) & 0xFFu);
}

int main() {
    printf("== test_gpu_execute ==\n");
    const uint32_t W = 64, H = 64;

    // Build the GpuState the CommandProcessor produces for one green fullscreen-triangle draw.
    GpuState st;
    set_pgm(st, P::SPI_SHADER_PGM_LO_ES, P::SPI_SHADER_PGM_HI_ES, kVs);
    set_pgm(st, P::SPI_SHADER_PGM_LO_PS, P::SPI_SHADER_PGM_HI_PS, kPs);
    st.cx[P::VGT_PRIMITIVE_TYPE] = 4;     // triangle list
    st.cx[P::CB_TARGET_MASK]     = 0xF;   // write RGBA
    st.draws.push_back({3});

    // The executor core, with the offscreen Vulkan renderer supplied as the backend (as the HLE will
    // supply the live-device renderer). execute_gpustate does recompile + resolve + render internally.
    auto backend = [&](const std::vector<uint32_t>& vs, const std::vector<uint32_t>& fs,
                       const ResolvedPipelineState& ps) {
        return prosper::test::render_triangle_rgba(vs, fs, W, H, &ps);
    };
    std::vector<uint8_t> px = execute_gpustate(st, backend);
    CHECK(px.size() == (size_t)W * H * 4, "execute_gpustate rendered a frame from the GpuState");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: executor produced no frame ==\n"); return 1; }

    auto isGreen = [&](uint32_t x, uint32_t y){ const uint8_t* p = &px[((size_t)y*W+x)*4]; return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40; };
    uint32_t green = 0, total = 0;
    for (uint32_t y : {0u, H/2, H-1}) for (uint32_t x : {0u, W/2, W-1}) { total++; if (isGreen(x, y)) green++; }
    CHECK(green == total, "GpuState -> executor -> GREEN frame (full recompile+resolve+render spine)");

    // Present round-trip: hand the frame to the scanout path and read it back (what videoout presents).
    prosper::gpu::present_reset();
    prosper::gpu::present_write_frame(px.data(), W, H);
    CHECK(prosper::gpu::present_has_frame(), "present accepted the executor's frame");
    std::vector<uint8_t> scan((size_t)W * H * 4, 0);
    size_t n = prosper::gpu::present_readback(scan.data(), scan.size());
    CHECK(n == px.size() && scan == px, "present_readback returns the executor's frame byte-for-byte");

    // An empty (state-only) submit renders nothing — mirrors the game's setup Dcb (0 draws).
    GpuState empty; empty.draws.clear();
    CHECK(execute_gpustate(empty, backend).empty(), "a draw-less GpuState renders no frame (state-only submit)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
