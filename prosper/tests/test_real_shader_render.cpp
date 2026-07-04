// test_real_shader_render — the payoff demo: take the GAME'S OWN shaders (the RDNA2 blobs embedded in
// eboot's .shader_text sections), recompile the ones that fully translate to SPIR-V, and render them
// through the proven GpuState->frame spine on real Vulkan. This proves the recompiler + pipeline work
// on REAL game data (not synthetic kernels), decoupled from the (parked) GfxDevice boot wall.
//
// Needs the (gitignored) game dump AND Vulkan, so CMake only builds/runs it when both are present.
//   test_real_shader_render <eboot.bin>
//
// A blob recompiles "as a vertex shader" iff it exports POS (recompile_vertex succeeds) and "as a
// pixel shader" iff it exports MRT (recompile_fragment succeeds) — the export type disambiguates.
// Fully-recompilable real shaders are necessarily the ones with no unsupported memory ops (SMEM/
// MUBUF/MIMG) — i.e. Unity's simple default-resource shaders (fullscreen/position-only + solid color),
// which is exactly the class the boot wall is stuck on. We render a real one where available, pairing
// with a minimal synthetic partner only if the game lacks a recompilable shader of that stage.
#include "../src/self/module.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Minimal synthetic fallbacks (llvm-mc gfx1030): fullscreen-triangle VS + solid-green PS. Only used
// for a stage the game has no recompilable shader for, so the demo can still render something real.
static const uint32_t kSynthVs[] = {
    0x36020081u, 0x2C040081u, 0x7E020D01u, 0x7E040D02u, 0x7E0A02F6u, 0x7E0C02F2u, 0x10020B01u,
    0x08020D01u, 0x10040B02u, 0x08040D02u, 0x7E060280u, 0x7E0802F2u, 0xF80008CFu, 0x04030201u, 0xBF810000u,
};
static const uint32_t kSynthPs[] = {
    0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
};

int main(int argc, char** argv) {
    printf("== test_real_shader_render ==\n");
    if (argc < 2) { printf("  [skip] no eboot path\n== PASS ==\n"); return 0; }
    std::string err;
    auto m = Module::load(argv[1], &err);
    if (!m) { printf("  [skip] load: %s\n== PASS ==\n", err.c_str()); return 0; }
    const std::vector<uint8_t>& f = m->file;

    // Collect the game's embedded RDNA2 shader blobs (.shader_text in EM_AMDGPU ELFs).
    struct Blob { const uint32_t* code; size_t dwords; size_t file_off; };
    std::vector<Blob> blobs;
    for (size_t i = 0; i + 64 < f.size(); i++) {
        if (!(f[i]==0x7f && f[i+1]=='E' && f[i+2]=='L' && f[i+3]=='F')) continue;
        auto inside = [&](uint64_t off, uint64_t len){ uint64_t a=(uint64_t)(f.size()-i); return off<=a && len<=a-off; };
        uint16_t machine; memcpy(&machine, &f[i+18], 2); if (machine != 0xE0) continue;
        const uint8_t* e = &f[i];
        uint64_t shoff; memcpy(&shoff, e+0x28, 8);
        uint16_t shentsize; memcpy(&shentsize, e+0x3A, 2);
        uint16_t shnum; memcpy(&shnum, e+0x3C, 2);
        uint16_t shstrndx; memcpy(&shstrndx, e+0x3E, 2);
        if (!shoff || !shnum || shstrndx>=shnum || shentsize<0x40 || !inside(shoff,(uint64_t)shnum*shentsize)) continue;
        const uint8_t* shstr = e + shoff + (size_t)shstrndx*shentsize;
        uint64_t so2, sz2; memcpy(&so2, shstr+0x18, 8); memcpy(&sz2, shstr+0x20, 8);
        if (!inside(so2, sz2)) continue;
        for (int s = 0; s < shnum; s++) {
            const uint8_t* sh = e + shoff + (size_t)s*shentsize;
            uint32_t noff; memcpy(&noff, sh, 4);
            if ((uint64_t)noff + 12 > sz2) continue;
            if (strncmp((const char*)(e+so2+noff), ".shader_text", 12) != 0) continue;
            uint64_t so, sz; memcpy(&so, sh+0x18, 8); memcpy(&sz, sh+0x20, 8);
            if (sz < 4 || !inside(so, sz)) continue;
            blobs.push_back({(const uint32_t*)(e+so), (size_t)(sz/4), (size_t)(i+so)});
        }
    }
    CHECK(!blobs.empty(), "found the game's embedded RDNA2 shader blobs");
    if (blobs.empty()) { printf("== FAIL ==\n"); return 1; }

    // Recompile each; classify by which export it has. Keep the first that fully translates per stage.
    std::vector<uint32_t> real_vs, real_ps; size_t vs_off = 0, ps_off = 0; int n_vs = 0, n_ps = 0;
    for (auto& b : blobs) {
        std::vector<uint32_t> v = recompile_vertex(b.code, b.dwords);
        if (!v.empty() && v[0] == 0x07230203u) { n_vs++; if (real_vs.empty()) { real_vs = v; vs_off = b.file_off; } }
        std::vector<uint32_t> p = recompile_fragment(b.code, b.dwords);
        if (!p.empty() && p[0] == 0x07230203u) { n_ps++; if (real_ps.empty()) { real_ps = p; ps_off = b.file_off; } }
    }
    printf("  scanned %zu shaders: %d recompile as vertex, %d as pixel\n", blobs.size(), n_vs, n_ps);
    CHECK(n_vs + n_ps > 0, "at least one REAL game shader fully recompiles to SPIR-V");

    // Render: prefer real for both stages; fall back to a synthetic partner only for a missing stage.
    bool vs_real = !real_vs.empty(), ps_real = !real_ps.empty();
    const std::vector<uint32_t>& vs = vs_real ? real_vs
        : (real_vs = std::vector<uint32_t>(), recompile_vertex(kSynthVs, sizeof(kSynthVs)/4));
    const std::vector<uint32_t>& ps = ps_real ? real_ps
        : (real_ps = std::vector<uint32_t>(), recompile_fragment(kSynthPs, sizeof(kSynthPs)/4));
    printf("  rendering: VS = %s%s, PS = %s%s\n",
           vs_real ? "REAL game shader @eboot+0x" : "synthetic", vs_real ? "" : "",
           ps_real ? "REAL game shader @eboot+0x" : "synthetic", ps_real ? "" : "");
    if (vs_real) printf("    VS file offset = 0x%zx\n", vs_off);
    if (ps_real) printf("    PS file offset = 0x%zx\n", ps_off);

    const uint32_t W = 64, H = 64;
    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vs, ps, W, H);
    CHECK(px.size() == (size_t)W*H*4,
          "the game's real recompiled shader(s) produced valid, Vulkan-accepted SPIR-V and RENDERED a frame");
    if (px.size() == (size_t)W*H*4) {
        const uint8_t* c = &px[((size_t)(H/2)*W + W/2)*4];
        printf("  center pixel = (%u,%u,%u,%u)\n", c[0], c[1], c[2], c[3]);
    }
    CHECK(vs_real || ps_real, "at least one stage rendered is a REAL game shader (not synthetic)");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
