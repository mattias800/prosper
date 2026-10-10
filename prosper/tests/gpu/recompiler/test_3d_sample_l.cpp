// test_3d_sample_l -- image_sample_l with dim:3D compiles to an explicit-LOD sample
// carrying the guest LOD (#4814). The dim3d lowering admitted only implicit-LOD, LOD-0
// and fetch forms, so an explicit-LOD sample of a 3D view refused even though the
// OpImageSampleExplicitLod emitter already existed for _lz. Compile-level only: the
// texel path (tail placement + upload) is pinned by BuildShaderResources.VolumeTail*;
// this pins the translator gate and the LOD operand reaching the instruction.
//
// Kernels are llvm-mc gfx1030 encodings (checked back through the decoder in-test):
//   P: image_sample_l v[0:3], [v4,v5,v6,v7], s[0:7], s[12:15] dmask:0xf dim:3D
//   C: image_sample_lz v[0:3], [v4,v5,v6], s[0:7], s[12:15] dmask:0xf dim:3D
//   N: image_sample_b v[0:3], [v4,v5,v6,v7], s[0:7], s[12:15] dmask:0xf dim:3D
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

// image_sample_l v[0:3], [v4,v5,v6,v7], s[0:7], s[12:15] dmask:0xf dim:3D; s_endpgm
const uint32_t kSampleL[] = {0xf0900f12u, 0x00600004u, 0x00070605u, 0xbf810000u};
// v_mov_b32 v7, 1.0: defines the guest LOD VGPR. An undefined v7 reads back as
// constant zero at compile time, which would make the LOD-operand scan vacuous --
// the scan must tell a carried 1.0 from a zeroed LOD, not undefined from zero.
// image_sample_lz v[0:3], [v4,v5,v6], s[0:7], s[12:15] dmask:0xf dim:3D; s_endpgm
const uint32_t kSampleLz[] = {0xf09c0f12u, 0x00600004u, 0x00000605u, 0xbf810000u};
// image_sample_b v[0:3], [v4,v5,v6,v7], s[0:7], s[12:15] dmask:0xf dim:3D; s_endpgm
const uint32_t kSampleB[] = {0xf0940f12u, 0x00600004u, 0x00070605u, 0xbf810000u};

ShaderResourceTable volume_table() {
    ShaderResourceTable table;
    ShaderResource vol{};
    vol.cls = ResourceClass::Texture;
    vol.format = DataFormat::Unorm8;
    vol.num_components = 1;
    vol.binding = 0;
    vol.img_dim = 2;
    vol.width = 32;
    vol.height = 32;
    vol.depth = 32;
    vol.tile_mode = 9;
    vol.sgpr_base = 0;
    vol.declared_mip_levels = 1;
    vol.sample_count = 1;
    vol.min_filter = 1;
    vol.mag_filter = 1;
    table.resources.push_back(vol);
    return table;
}

std::vector<uint32_t> compile(const uint32_t* code, size_t dwords) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    const ShaderResourceTable table = volume_table();
    return recompile_compute(code, dwords, &table, config,
                             {RecompileDiagnosticStage::Compute, 0xa4814001ULL});
}

// SPIR-V opcode numbers (rdna2_to_spirv_internal.hpp): OpImageSampleExplicitLod = 88,
// OpConstant = 43. The ExplicitLod instruction is header + 6 words (type, id, image,
// coord, mask, lod), so its LOD operand is the last word.
bool contains_explicit_lod(const std::vector<uint32_t>& words) {
    if (words.size() < 6) return false;
    for (size_t i = 5; i < words.size();) {
        const uint32_t wc = words[i] >> 16;
        const uint32_t op = words[i] & 0xffffu;
        if (wc == 0 || i + wc > words.size()) return false;
        if (op == 88u) return true;
        i += wc;
    }
    return false;
}

// The LOD operand of the ExplicitLod sample must not be a constant zero: _lz lowers
// with uconst(0) while _l must carry the guest LOD VGPR (bitcast to float). The emitter
// wraps the LOD in OpBitcast either way, so the scan looks through one bitcast to the
// constant set: a guest value resolves to a load, a zeroed LOD to OpConstant 0.
bool lod_is_guest_value(const std::vector<uint32_t>& words) {
    if (words.size() < 6) return false;
    std::vector<uint32_t> zero_const_ids;
    std::vector<std::pair<uint32_t, uint32_t>> bitcasts;
    uint32_t lod_id = 0;
    bool found = false;
    for (size_t i = 5; i < words.size();) {
        const uint32_t wc = words[i] >> 16;
        const uint32_t op = words[i] & 0xffffu;
        if (wc == 0 || i + wc > words.size()) return false;
        if (op == 43u && wc == 4u && words[i + 3] == 0u) zero_const_ids.push_back(words[i + 2]);
        if (op == 124u && wc == 4u) bitcasts.emplace_back(words[i + 2], words[i + 3]);
        if (op == 88u && wc == 7u) {
            lod_id = words[i + 6];
            found = true;
        }
        i += wc;
    }
    if (!found) return false;
    for (unsigned hop = 0; hop < 4; ++hop) {
        bool moved = false;
        for (const auto& [id, input] : bitcasts)
            if (id == lod_id) {
                lod_id = input;
                moved = true;
                break;
            }
        if (!moved) break;
    }
    for (uint32_t id : zero_const_ids)
        if (id == lod_id) return false;
    return true;
}

}  // namespace

TEST(SampleL3D, FixtureEncodesWhatItClaims) {
    const Rdna2Inst l = rdna2_decode_one(kSampleL, 3);
    ASSERT_EQ(l.opcode, 0x24u) << "P kernel is image_sample_l";
    ASSERT_EQ(l.mimg_dim, 2u) << "P kernel samples dim:3D";
    const Rdna2Inst lz = rdna2_decode_one(kSampleLz, 3);
    ASSERT_EQ(lz.opcode, 0x27u) << "C kernel is image_sample_lz";
    const Rdna2Inst b = rdna2_decode_one(kSampleB, 3);
    ASSERT_EQ(b.opcode, 0x25u) << "N kernel is image_sample_b";
}

TEST(SampleL3D, ExplicitLodSampleCompilesWithGuestLod) {
    const uint32_t kernel[] = {0x7e0e02f2u, 0xf0900f12u, 0x00600004u, 0x00070605u, 0xbf810000u};
    const std::vector<uint32_t> mod = compile(kernel, std::size(kernel));
    ASSERT_FALSE(mod.empty()) << "image_sample_l dim:3D compiles instead of refusing";
    EXPECT_TRUE(contains_explicit_lod(mod)) << "lowering emits OpImageSampleExplicitLod";
    EXPECT_TRUE(lod_is_guest_value(mod)) << "LOD operand is the guest value, not const 0";
}

TEST(SampleL3D, LodZeroControlStillCompiles) {
    EXPECT_FALSE(compile(kSampleLz, std::size(kSampleLz)).empty())
        << "image_sample_lz dim:3D keeps compiling (the pre-existing arm)";
}

TEST(SampleL3D, BiasFormStillRefuses) {
    EXPECT_TRUE(compile(kSampleB, std::size(kSampleB)).empty())
        << "image_sample_b dim:3D stays refused (no bias lowering exists)";
}
