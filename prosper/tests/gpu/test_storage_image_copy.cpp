// test_storage_image_copy — execution-differential test for the STORAGE-IMAGE recompiler path
// (MIMG image_load 0x00 + image_store 0x08, no sampler). Recompiles a real gfx10.1-shaped 1D
// image-copy compute kernel and runs it on real Vulkan: it must read every texel of a source
// storage image and write it to a destination storage image, bit-exact.
//
// This is the class the game's 9 "barefoot" blit/copy compute shaders belong to (image_load v[0:3],
// vN, s[0:7] dim:1D ; image_store v[0:3], vN, s[8:15] dim:1D). Needs Vulkan, so CMake only builds it
// when Vulkan is present.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/image_compute_runner.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

#define CHECK(c, m) EXPECT_TRUE(c) << (m)

TEST(StorageImageCopy, Contract) {
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
    if (spv.empty()) { printf("== FAIL ==\n"); FAIL() << "legacy early exit"; }

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

    // NON-multiple width (#131): 70 texels dispatch ceil(70/64)=2 workgroups = 128 invocations, so
    // lanes 70..127 read AND write out of the image's range. With robustImageAccess (now enabled by
    // the runner when the device offers it) OOB reads return zero and OOB writes are discarded —
    // the copy of the real 70 texels must still be bit-exact and nothing may fault.
    const uint32_t W2 = 70;
    std::vector<uint32_t> src2(W2 * 4);
    for (uint32_t i = 0; i < W2; i++) {
        src2[i*4+0] = 0x10000000u + i;        src2[i*4+1] = 0x20000000u + i*3u + 1u;
        src2[i*4+2] = 0x30000000u + i*11u+2u; src2[i*4+3] = 0x40000000u + i*17u + 3u;
    }
    std::vector<uint32_t> dst2 = prosper::test::run_image_copy(spv, W2, src2);
    CHECK(dst2.size() == src2.size(), "non-multiple-width copy ran (OOB tail lanes tolerated)");
    if (dst2.size() == src2.size()) {
        uint32_t bad2 = 0;
        for (uint32_t i = 0; i < W2 * 4; i++) if (dst2[i] != src2[i]) bad2++;
        printf("  non-multiple width: mismatched components = %u / %u\n", bad2, W2*4);
        CHECK(bad2 == 0, "70-texel copy bit-exact with a 128-invocation dispatch (grid-tail OOB safe)");
    }

}

// A storage image's T# DST_SEL routes image_store VDATA too: the stored channel c receives the
// component whose selector names c. Hollow Knight: Silksong's dynamic-font glyph upload stores into
// an Alpha8 atlas with DST_SEL = (0,0,0,X), so the glyph travels in .w; storing VDATA unrouted wrote
// .x (zero) and left every menu string blank (#3549). Same kernel and harness as the copy above,
// with only the destination descriptor's selector changed.
namespace {

const uint32_t kCopyKernel[] = {
    0x7E080300u, 0xF0000F00u, 0x00000004u, 0xBF8C3F70u, 0xF0200F00u, 0x00020004u, 0xBF810000u,
};

std::vector<uint32_t> compile_copy(const uint32_t (&dst_sel)[4], uint32_t dst_components,
                                   const uint32_t (&src_sel)[4] = {4, 5, 6, 7}) {
    ShaderResourceTable rt;
    ShaderResource src{};
    src.cls = ResourceClass::StorageImage; src.img_dim = 0; src.binding = 4; src.sgpr_base = 0;
    for (uint32_t k = 0; k < 4; ++k) src.swizzle[k] = src_sel[k];
    rt.resources.push_back(src);
    ShaderResource dst{};
    dst.cls = ResourceClass::StorageImage; dst.img_dim = 0; dst.binding = 5; dst.sgpr_base = 8;
    dst.num_components = dst_components;
    for (uint32_t k = 0; k < 4; ++k) dst.swizzle[k] = dst_sel[k];
    rt.resources.push_back(dst);
    return recompile_valu(kCopyKernel, sizeof(kCopyKernel) / sizeof(kCopyKernel[0]), 1, 0, &rt);
}

std::vector<uint32_t> distinct_texels(uint32_t width) {
    std::vector<uint32_t> texels(width * 4);
    for (uint32_t i = 0; i < width; ++i) {
        texels[i * 4 + 0] = 0xA0000000u + i;
        texels[i * 4 + 1] = 0xB0000000u + i;
        texels[i * 4 + 2] = 0xC0000000u + i;
        texels[i * 4 + 3] = 0xD0000000u + i;
    }
    return texels;
}

}  // namespace

TEST(StorageImageCopy, AlphaOnlySelectorStoresTheWComponent) {
    const uint32_t sel[4] = {0, 0, 0, 4};   // SQ_SEL_0, 0, 0, X: Unity's Alpha8 descriptor
    const std::vector<uint32_t> spv = compile_copy(sel, 1);
    ASSERT_FALSE(spv.empty()) << "a one-channel destination named by .w must compile";
    const uint32_t W = 64;
    const std::vector<uint32_t> src = distinct_texels(W);
    const std::vector<uint32_t> dst = prosper::test::run_image_copy(spv, W, src);
    ASSERT_EQ(dst.size(), src.size());
    uint32_t bad = 0;
    for (uint32_t i = 0; i < W; ++i) bad += dst[i * 4 + 0] != src[i * 4 + 3];
    EXPECT_EQ(bad, 0u) << "stored channel X must receive VDATA.w; texel0 X=0x" << std::hex
                       << dst[0];
}

TEST(StorageImageCopy, PermutedSelectorStoresTheInverse) {
    const uint32_t sel[4] = {6, 5, 4, 7};   // Z, Y, X, W: BGRA order
    const std::vector<uint32_t> spv = compile_copy(sel, 4);
    ASSERT_FALSE(spv.empty());
    const uint32_t W = 64;
    const std::vector<uint32_t> src = distinct_texels(W);
    const std::vector<uint32_t> dst = prosper::test::run_image_copy(spv, W, src);
    ASSERT_EQ(dst.size(), src.size());
    uint32_t bad = 0;
    for (uint32_t i = 0; i < W; ++i) {
        bad += dst[i * 4 + 0] != src[i * 4 + 2];
        bad += dst[i * 4 + 1] != src[i * 4 + 1];
        bad += dst[i * 4 + 2] != src[i * 4 + 0];
        bad += dst[i * 4 + 3] != src[i * 4 + 3];
    }
    EXPECT_EQ(bad, 0u) << "channel c receives the component whose selector names c";
}

TEST(StorageImageCopy, UnnamedStoredChannelIsRefused) {
    // Every stored channel of a four-channel format must be named by some selector. W here is
    // SQ_SEL_1, which supplies nothing on a store, so the store is refused rather than guessed.
    const uint32_t constant_w[4] = {4, 5, 6, 1};
    EXPECT_TRUE(compile_copy(constant_w, 4).empty());
    // The same selector over a three-channel format names every stored channel.
    EXPECT_FALSE(compile_copy(constant_w, 3).empty());
    // Two selectors naming one channel, and the reserved SQ_SEL 2, are both undecodable.
    const uint32_t duplicate[4] = {4, 4, 6, 7};
    EXPECT_TRUE(compile_copy(duplicate, 4).empty());
    const uint32_t reserved[4] = {4, 5, 2, 7};
    EXPECT_TRUE(compile_copy(reserved, 4).empty());
}

// #4274: a storage image's DST_SEL routes image_load too, forward: returned channel k takes the
// stored component its selector names, or the constant 0 / 1. With the store routing above, the
// same descriptor on both sides of a copy round-trips.
TEST(StorageImageCopy, LoadSelectorRoutesForwardWithConstants) {
    const uint32_t identity[4] = {4, 5, 6, 7};
    const uint32_t src_sel[4] = {1, 0, 5, 4};   // 1, 0, Y, X
    const std::vector<uint32_t> spv = compile_copy(identity, 4, src_sel);
    ASSERT_FALSE(spv.empty());
    const uint32_t W = 64;
    const std::vector<uint32_t> src = distinct_texels(W);
    const std::vector<uint32_t> dst = prosper::test::run_image_copy(spv, W, src);
    ASSERT_EQ(dst.size(), src.size());
    uint32_t bad = 0;
    for (uint32_t i = 0; i < W; ++i) {
        bad += dst[i * 4 + 0] != 0x3f800000u;   // SQ_SEL_1 on a non-integer format: float 1.0
        bad += dst[i * 4 + 1] != 0u;            // SQ_SEL_0
        bad += dst[i * 4 + 2] != src[i * 4 + 1];
        bad += dst[i * 4 + 3] != src[i * 4 + 0];
    }
    EXPECT_EQ(bad, 0u) << "texel0 = " << std::hex << dst[0] << " " << dst[1] << " " << dst[2]
                       << " " << dst[3];
}

TEST(StorageImageCopy, AlphaOnlyDescriptorRoundTrips) {
    const uint32_t alpha8[4] = {0, 0, 0, 4};
    const std::vector<uint32_t> spv = compile_copy(alpha8, 1, alpha8);
    ASSERT_FALSE(spv.empty());
    const uint32_t W = 64;
    const std::vector<uint32_t> src = distinct_texels(W);
    const std::vector<uint32_t> dst = prosper::test::run_image_copy(spv, W, src);
    ASSERT_EQ(dst.size(), src.size());
    uint32_t bad = 0;
    for (uint32_t i = 0; i < W; ++i) bad += dst[i * 4 + 0] != src[i * 4 + 0];
    EXPECT_EQ(bad, 0u) << "stored X must survive an Alpha8 load -> store copy";
}
