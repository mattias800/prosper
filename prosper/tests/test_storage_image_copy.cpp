// test_storage_image_copy — execution-differential test for the STORAGE-IMAGE recompiler path
// (MIMG image_load 0x00 + image_store 0x08, no sampler). Recompiles a real gfx10.1-shaped 1D
// image-copy compute kernel and runs it on real Vulkan: it must read every texel of a source
// storage image and write it to a destination storage image, bit-exact.
//
// This is the class the game's 9 "barefoot" blit/copy compute shaders belong to (image_load v[0:3],
// vN, s[0:7] dim:1D ; image_store v[0:3], vN, s[8:15] dim:1D). Needs Vulkan, so CMake only builds it
// when Vulkan is present.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "image_compute_runner.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_storage_image_copy ==\n");

    // 1D image copy (assembled by llvm-mc gfx1010 — the PS5 shaders' target ISA):
    //   v_mov_b32 v4, v0                                  ; v4 = texel index (v0 = input[gid] = gid; see below)
    //   image_load  v[0:3], v4, s[0:7]  dmask:0xf dim:1D  ; read RGBA texel from src (U# at s0)
    //   s_waitcnt vmcnt(0)
    //   image_store v[0:3], v4, s[8:15] dmask:0xf dim:1D  ; write it to dst (U# at s8)
    //   s_endpgm
    // The coordinate comes from the shell's input buffer (v0 = input[gid]); the harness fills binding-0
    // with [0..W-1], so v0 = gid. (The game shaders derive it from the compute thread-ID registers
    // s16=workgroup.id / v0=local.id via v_lshl_add_u32 — that ID ABI is a separate modeling step; this
    // test isolates and proves the OpImageRead->OpImageWrite storage-image path.)
    const uint32_t code[] = {
        0x7E080300u, 0xF0000F00u, 0x00000004u, 0xBF8C3F70u, 0xF0200F00u, 0x00020004u, 0xBF810000u,
    };

    // Resource table: two 1D storage images. SRSRC s0 -> binding 4 (src), s8 -> binding 5 (dst); the
    // harness (image_compute_runner) binds R32G32B32A32_UINT 1D images at exactly those bindings.
    ShaderResourceTable rt;
    { ShaderResource src{}; src.cls = ResourceClass::StorageImage; src.img_dim = 0 /*1D*/;
      src.binding = 4; src.sgpr_base = 0;  rt.resources.push_back(src); }
    { ShaderResource dst{}; dst.cls = ResourceClass::StorageImage; dst.img_dim = 0 /*1D*/;
      dst.binding = 5; dst.sgpr_base = 8;  rt.resources.push_back(dst); }

    // num_inputs=1: the shell loads v0 = input[gid] (the harness fills binding-0 with the linear index).
    std::vector<uint32_t> spv = recompile_valu(code, sizeof(code)/sizeof(code[0]), 1, 0, &rt);
    CHECK(!spv.empty() && spv[0] == 0x07230203u, "1D image-copy kernel recompiled to a SPIR-V module");
    if (spv.empty()) { printf("== FAIL ==\n"); return 1; }

    // Source texels: 64 texels (a multiple of the 64-wide workgroup, so dispatch covers exactly the
    // image with no out-of-range invocations), each RGBA distinct so a mis-copy is unambiguous.
    const uint32_t W = 64;
    std::vector<uint32_t> src(W * 4);
    for (uint32_t i = 0; i < W; i++) {
        src[i*4+0] = 0xA0000000u + i;
        src[i*4+1] = 0xB0000000u + i*7u + 1u;
        src[i*4+2] = 0xC0000000u + i*13u + 2u;
        src[i*4+3] = 0xD0000000u + i*29u + 3u;
    }

    std::vector<uint32_t> dst = prosper::test::run_image_copy(spv, W, src);
    CHECK(dst.size() == src.size(), "storage-image copy kernel ran on Vulkan and returned dst texels");
    if (dst.size() == src.size()) {
        uint32_t bad = 0;
        for (uint32_t i = 0; i < W * 4; i++) if (dst[i] != src[i]) bad++;
        printf("  mismatched components = %u / %u (texel0 R: src=0x%08x dst=0x%08x)\n",
               bad, W*4, src[0], dst[0]);
        CHECK(bad == 0, "OpImageRead->OpImageWrite copied every RGBA texel bit-exact (src == dst)");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
