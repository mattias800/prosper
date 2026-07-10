// test_recompiled_fragment — render a triangle whose PIXEL SHADER is recompiled from RDNA2.
// The vertex shader is the placeholder (positions), but the fragment shader is real RDNA2 (assembled
// by llvm-mc) recompiled to SPIR-V by recompile_fragment: it exports green via EXP MRT0. We render
// and assert the triangle is GREEN (not the placeholder's red), proving RDNA2->SPIR-V works for an
// actual graphics-stage shader wired into a real pipeline.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "render_runner.h"
#include "spirv_triangle.h"     // kTriVertSpv: placeholder vertex shader (positions)
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_recompiled_fragment ==\n");
    const uint32_t W = 64, H = 64;

    // RDNA2 green pixel shader (llvm-mc gfx1030): v0=0(r) v1=1.0(g) v2=0(b) v3=1.0(a); exp mrt0.
    //   v_mov_b32 v0,0 | v_mov_b32 v1,1.0 | v_mov_b32 v2,0 | v_mov_b32 v3,1.0 | exp mrt0 ... | s_endpgm
    const uint32_t ps[] = { 0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu, 0x03020100u, 0xBF810000u };
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
    CHECK(!frag.empty() && frag[0] == 0x07230203u, "recompiled RDNA2 pixel shader -> SPIR-V module");
    if (frag.empty()) { printf("== FAIL ==\n"); return 1; }

    std::vector<uint32_t> vert(kTriVertSpv, kTriVertSpv + sizeof(kTriVertSpv)/sizeof(kTriVertSpv[0]));
    std::vector<uint8_t> px = prosper::test::render_triangle_rgba(vert, frag, W, H);
    CHECK(px.size() == (size_t)W * H * 4, "rendered with the recompiled fragment shader (pipeline accepted it)");
    if (px.size() != (size_t)W * H * 4) { printf("== FAIL: render failed ==\n"); return 1; }

    auto at = [&](uint32_t x, uint32_t y) { return &px[((size_t)y * W + x) * 4]; };
    const uint8_t* c = at(W/2, H/2);   // triangle covers the center
    const uint8_t* k = at(0, 0);       // corner = clear
    printf("  center=(%u,%u,%u,%u) corner=(%u,%u,%u,%u)\n", c[0],c[1],c[2],c[3], k[0],k[1],k[2],k[3]);
    CHECK(c[1] > 0x80 && c[0] < 0x40 && c[2] < 0x40, "center pixel is GREEN (from the recompiled shader)");
    CHECK(k[2] > 0x80 && k[0] < 0x40 && k[1] < 0x40, "corner pixel is the BLUE clear");

    // VCC kill-mask early-out (#273 — DOLL's alpha-cull PS): `v_cmp; s_andn2_b64 vcc, exec, vcc;
    // s_cbranch_scc0 <null-export>; s_mov_b64 exec, vcc; export`. The kill mask lives in VCC itself
    // (not a saved SGPR pair); mask_test_branches must recognize it so the branch linearizes and the
    // narrowed-EXEC export lowers to a per-invocation discard. Two variants (llvm-mc gfx1010):
    //  survive: v0=0.5 -> cmp(0.5<v0)=false -> survivors=all -> WHITE triangle;
    //  killed:  v0=1.0 -> cmp=true -> survivors=none -> every fragment discarded -> clear stays.
    auto killmask_ps = [](uint32_t v0mov) {
        return std::vector<uint32_t>{ v0mov, 0x7c0200f0u, 0x8aea6a7eu, 0xbf840005u, 0xbefe046au,
                                      0x7e0202f2u, 0xf800180fu, 0x01010101u, 0xbf810000u,
                                      0xbefe0480u, 0xf8001800u, 0x00000000u, 0xbf810000u };
    };
    std::vector<uint32_t> psSurvive = killmask_ps(0x7e0002f0u);   // v_mov_b32 v0, 0.5
    std::vector<uint32_t> psKilled  = killmask_ps(0x7e0002f2u);   // v_mov_b32 v0, 1.0
    std::vector<uint32_t> fragS = recompile_fragment(psSurvive.data(), psSurvive.size());
    std::vector<uint32_t> fragK = recompile_fragment(psKilled.data(),  psKilled.size());
    CHECK(!fragS.empty() && !fragK.empty(), "recompiled VCC kill-mask PS (both variants) -> SPIR-V");
    if (!fragS.empty() && !fragK.empty()) {
        std::vector<uint8_t> pxS = prosper::test::render_triangle_rgba(vert, fragS, W, H);
        std::vector<uint8_t> pxK = prosper::test::render_triangle_rgba(vert, fragK, W, H);
        CHECK(pxS.size() == (size_t)W*H*4 && pxK.size() == (size_t)W*H*4, "rendered both kill-mask variants");
        if (pxS.size() == (size_t)W*H*4 && pxK.size() == (size_t)W*H*4) {
            const uint8_t* cs = &pxS[((size_t)(H/2) * W + W/2) * 4];
            const uint8_t* ck = &pxK[((size_t)(H/2) * W + W/2) * 4];
            printf("  survive center=(%u,%u,%u,%u) killed center=(%u,%u,%u,%u)\n",
                   cs[0],cs[1],cs[2],cs[3], ck[0],ck[1],ck[2],ck[3]);
            CHECK(cs[0] > 0x80 && cs[1] > 0x80 && cs[2] > 0x80, "survive variant: center WHITE (lanes pass the mask)");
            CHECK(ck[2] > 0x80 && ck[0] < 0x40 && ck[1] < 0x40, "killed variant: center stays BLUE clear (OpKill discarded)");
        }
    }

    // DIVERGENT execz region (#273 — DOLL's FXAA PS shape): v_cmpx narrows EXEC, s_cbranch_execz
    // skips a block containing a SCALAR write read after the merge (so it is NOT safe-linearizable
    // and must go through the structured exec-if). Then-arm sets v1 = s5 = 1.0; the export runs
    // under the narrowed EXEC (surviving lanes only).
    //  taken:   v0=1.0 -> 0.5<1.0 -> all lanes survive -> WHITE;
    //  skipped: v0=0.5 -> 0.5<0.5 false -> exec 0 -> every fragment discarded -> clear stays.
    auto execz_ps = [](uint32_t v0mov) {
        return std::vector<uint32_t>{ 0x7e020280u, v0mov, 0x7c2200f0u, 0xbf880002u,
                                      0xbe8503f2u, 0x7e020205u,            // block: s5=1.0; v1=s5
                                      0x7e040205u,                          // post-merge s5 read (v2=s5)
                                      0xf800180fu, 0x01010101u, 0xbf810000u };
    };
    std::vector<uint32_t> psTaken   = execz_ps(0x7e0002f2u);   // v_mov_b32 v0, 1.0
    std::vector<uint32_t> psSkipped = execz_ps(0x7e0002f0u);   // v_mov_b32 v0, 0.5
    std::vector<uint32_t> fragT = recompile_fragment(psTaken.data(),   psTaken.size());
    std::vector<uint32_t> fragX = recompile_fragment(psSkipped.data(), psSkipped.size());
    CHECK(!fragT.empty() && !fragX.empty(), "recompiled divergent-execz PS (both variants) -> SPIR-V");
    if (!fragT.empty() && !fragX.empty()) {
        std::vector<uint8_t> pxT = prosper::test::render_triangle_rgba(vert, fragT, W, H);
        std::vector<uint8_t> pxX = prosper::test::render_triangle_rgba(vert, fragX, W, H);
        CHECK(pxT.size() == (size_t)W*H*4 && pxX.size() == (size_t)W*H*4, "rendered both execz variants");
        if (pxT.size() == (size_t)W*H*4 && pxX.size() == (size_t)W*H*4) {
            const uint8_t* ct = &pxT[((size_t)(H/2) * W + W/2) * 4];
            const uint8_t* cx = &pxX[((size_t)(H/2) * W + W/2) * 4];
            printf("  execz taken center=(%u,%u,%u,%u) skipped center=(%u,%u,%u,%u)\n",
                   ct[0],ct[1],ct[2],ct[3], cx[0],cx[1],cx[2],cx[3]);
            CHECK(ct[0] > 0x80 && ct[1] > 0x80 && ct[2] > 0x80, "execz taken: center WHITE (block ran, s5 -> v1)");
            CHECK(cx[2] > 0x80 && cx[0] < 0x40 && cx[1] < 0x40, "execz skipped: center stays BLUE clear (all lanes inactive)");
        }
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
