// test_split_t8_image_write_extent -- an earlier image_store does not revoke a descriptor proof when
// its footprint cannot reach the descriptor bytes.
//
// THE DEFECT. The split-T# proof refuses a descriptor if ANY instruction that can write memory runs
// before its consumer, because a store could rewrite the descriptor bytes the CPU snapshot read.
// Assassin's Creed Black Flag Resynced's compute program `0x407f8ae300` loads two adjacent T#s with
// one s_load_dwordx16, stores through the first and then image_loads through the second. The store
// is to a 960x540 image megabytes away from the descriptor table, so it cannot change the second
// T#, but the proof had no way to say so and the whole program was skipped (its output never ran).
//
// WHAT EACH TEST KILLS:
//   StoreToDistantImageKeepsProof        the extent is ignored again, or computed too large
//   StoreToImageOverlappingTableRefused  a store whose footprint covers the table is admitted
//   ExtentIsASupersetOfTheSurface        the footprint underestimates a padded, mipped, multi-slice surface
//   CompressedSurfaceHasNoExtent         a surface with metadata is given a bound that ignores the metadata
//   SamplerPredicate                     a store's unused sampler field is read as a use of s0..s3
//   X16ProofAdmitsTwoStoresDespiteUnusedSamplerField  the unused sampler field of image_store revokes the x16 proof
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/split_t8_proof.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/smem_x16_descriptor_proof.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

alignas(64) uint32_t g_table[16];

// Captured shape of a 960x540 2D image T# (base is patched per test) and a 240x136 one.
std::array<uint32_t, 8> store_t8(uint64_t base) {
    return { static_cast<uint32_t>(base >> 8), 0xc0d00000u | static_cast<uint32_t>((base >> 40) & 0xffu), 0x0086c0efu, 0x90900204u, 0, 0x00700000u, 0, 0 };
}
std::array<uint32_t, 8> load_t8() {
    return { 0x421abe00u, 0xc1400000u, 0x0021c03bu, 0x90900004u, 0, 0x00700000u, 0, 0 };
}

// pc0      s_cbranch_execz +0           (makes the program branchy, so the CFG proof owns the load)
// pc1-2    s_load_dwordx16 s[4:19], s[2:3], 0
// pc3      s_waitcnt
// pc4-5    image_store ... s[4:11]      (the earlier writer)
// pc6-7    image_load  ... s[12:19]     (the consumer)
// pc8      s_endpgm
std::vector<uint32_t> program() {
    return {
        0xBF880000u,
        0xF4100101u, 0xFA000000u,
        0xBF8CC07Fu,
        0xF0200108u, 0x00010604u,
        0xF0000208u, 0x00030704u,
        0xBF810000u,
    };
}

std::vector<SrtUse> uses_for(const std::array<uint32_t, 8>& first, const std::array<uint32_t, 8>& second) {
    std::copy(first.begin(), first.end(), g_table);
    std::copy(second.begin(), second.end(), g_table + 8);
    const uint64_t base = reinterpret_cast<uint64_t>(g_table);
    const uint32_t seed[4] = { 0u, 0u, static_cast<uint32_t>(base), static_cast<uint32_t>(base >> 32u) };
    std::vector<SrtUse> result;
    const auto code = program();
    resolve_dynamic_fetch(code.data(), code.size(), seed, 4, 0, &result);
    return result;
}

bool has_use(const std::vector<SrtUse>& uses, uint32_t pc) {
    return std::any_of(uses.begin(), uses.end(),
                       [pc](const SrtUse& u) { return u.kind == 0 && u.use_pc == pc; });
}

}  // namespace

TEST(SplitT8ImageWriteExtent, StoreToDistantImageKeepsProof) {
    EXPECT_TRUE(has_use(uses_for(store_t8(0x421c4d0000ull), load_t8()), 6u))
        << "the store lands megabytes from the table, so the second T# is still what the CPU read";
}

TEST(SplitT8ImageWriteExtent, StoreToImageOverlappingTableRefused) {
    // The stored image's footprint covers the descriptor table itself.
    const uint64_t table = reinterpret_cast<uint64_t>(g_table);
    EXPECT_FALSE(has_use(uses_for(store_t8(table & ~0xffull), load_t8()), 6u))
        << "a store that can reach the descriptor bytes must still revoke the proof";
}

TEST(SplitT8ImageWriteExtent, ExtentIsASupersetOfTheSurface) {
    uint64_t lo = 0, hi = 0;
    ASSERT_TRUE(storage_image_write_extent(store_t8(0x421c4d0000ull), lo, hi));
    EXPECT_EQ(lo, 0x421c4d0000ull);
    // 960x540 padded to 256x256 tiles is 1024x768; at one byte per texel that is the floor.
    EXPECT_GE(hi - lo, 1024ull * 768ull) << "the footprint must cover the padded surface";
    EXPECT_LT(hi - lo, 1ull << 31) << "and stay a usable bound, not the whole address space";
}

TEST(SplitT8ImageWriteExtent, CompressedSurfaceHasNoExtent) {
    auto t8 = store_t8(0x421c4d0000ull);
    t8[6] |= 1u << 21;   // WORD6 COMPRESSION_EN: the surface owns a metadata plane this bound ignores
    uint64_t lo = 0, hi = 0;
    EXPECT_FALSE(storage_image_write_extent(t8, lo, hi));
}

TEST(SplitT8ImageWriteExtent, SamplerPredicate) {
    auto mimg = [](uint32_t opcode) {
        Rdna2Inst in{};
        in.fmt = Rdna2Format::MIMG;
        in.opcode = opcode;
        return in;
    };
    EXPECT_FALSE(rdna2_mimg_reads_sampler(mimg(0x00))) << "image_load has no sampler";
    EXPECT_FALSE(rdna2_mimg_reads_sampler(mimg(0x08))) << "image_store has no sampler";
    EXPECT_FALSE(rdna2_mimg_reads_sampler(mimg(0x0e))) << "image_get_resinfo has no sampler";
    EXPECT_FALSE(rdna2_mimg_reads_sampler(mimg(0x11))) << "image atomics have no sampler";
    EXPECT_TRUE(rdna2_mimg_reads_sampler(mimg(0x20))) << "image_sample reads its sampler";
    EXPECT_TRUE(rdna2_mimg_reads_sampler(mimg(0x7f))) << "an unknown opcode fails closed to sampled";
    Rdna2Inst smem{};
    smem.fmt = Rdna2Format::SMEM;
    EXPECT_FALSE(rdna2_mimg_reads_sampler(smem));
}

// The recompiler's x16 proof, on the real shape: s_load_dwordx16 s[0:15], then image_store through
// s[0:7] and image_store through s[8:15]. Each store's SSAMP field encodes s0, which neither reads.
//   pc0-1 load   pc2 waitcnt   pc3-4 store s[0:7]   pc5-6 store s[8:15]   pc7 endpgm
TEST(SplitT8ImageWriteExtent, X16ProofAdmitsTwoStoresDespiteUnusedSamplerField) {
    const std::vector<uint32_t> code = {
        0xF4100001u, 0xFA000000u,
        0xBF8CC07Fu,
        0xF0200F08u, 0x00000004u,
        0xF0200108u, 0x00020204u,
        0xBF810000u,
    };
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    ShaderResourceTable table;
    for (uint32_t pc : {3u, 5u}) {
        ShaderResource image;
        image.cls = ResourceClass::StorageImage;
        image.fetch_pc = pc;
        table.resources.push_back(image);
    }
    const auto loads = proven_smem_x16_descriptor_loads(ins, &table, 64);
    EXPECT_TRUE(loads.contains(0u))
        << "an image_store has no sampler, so its SSAMP field must not count as reading s0";
}
