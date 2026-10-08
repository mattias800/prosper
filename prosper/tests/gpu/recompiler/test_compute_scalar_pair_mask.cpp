// test_compute_scalar_pair_mask -- a Wave64 COMPUTE program's B64 mask logical over an ordinary
// scalar DATA pair projects the pair onto the invocation's own lane bit, but never when either word
// may be the structured emitter's fabricated zero (#4714, GPU-5/FAIL-1; the fragment half is #4711).
//
// WHAT EACH ARM KILLS:
//   DefinedPairStillProjects    a guard so broad it refuses a pair every word of which is defined
//                               (GTA copies ballots into scalar scratch; Sonic Cyber Space uses {1,1})
//   OnePathPairRefuses          projecting the merge's fabricated zero as a lane mask, directly and
//                               through a copy (the mark must follow `s_mov_b64`)
// The one-path programs are assembled by hand here, outside the recompiler code that sets the mark.
// Compile-level only: lane selection itself is executed by test_fragment_scalar_pair_mask.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLanes = 64;

// v_mov v5,0 | v_readfirstlane s0,v5 (no fold knows s0) | v_cmp_eq_u32 s[2:3],0,0 (every lane) |
// s_cmp_eq_u32 s0,0 | s_cbranch_scc1 +2 | s_mov_b32 s4,-1 | s_mov_b32 s5,-1 |   <- the one path
// s_and_b64 s[6:7], s[4:5], s[2:3] | v_cndmask_b32_e64 v3,0,1.0,s[6:7] |
// buffer_store_dword v3 -> out[x] | s_endpgm.   (llvm-mc -mcpu=gfx1030 -mattr=+wavefrontsize64)
#define TAIL_AND_STORE 0xd5010003u, 0x0019e480u, 0xe0702000u, 0x80020300u, 0xbf810000u
constexpr uint32_t kOnePath[] = {
    0x7e0a0280u, 0x7e000505u, 0xd4c20002u, 0x00010080u, 0xbf068000u,
    0xbf850002u, 0xbe8403c1u, 0xbe8503c1u, 0x87860204u, TAIL_AND_STORE,
};
// The same, with the merged pair copied first (s[8:11] is the output V#, so the copy is s[12:13]).
constexpr uint32_t kOnePathCopied[] = {
    0x7e0a0280u, 0x7e000505u, 0xd4c20002u, 0x00010080u, 0xbf068000u,    0xbf850002u,
    0xbe8403c1u, 0xbe8503c1u, 0xbe8c0404u, 0x8786020cu, TAIL_AND_STORE,
};
// Control: no branch. s4 = -1 (low half), s5 = 0, so lanes 0..31 are selected.
constexpr uint32_t kDefinedLow[] = {
    0xd4c20002u, 0x00010080u, 0xbe8403c1u, 0xbe850380u, 0x87860204u, TAIL_AND_STORE,
};
// Control: high half set, so lanes 32..63 are selected.
constexpr uint32_t kDefinedHigh[] = {
    0xd4c20002u, 0x00010080u, 0xbe840380u, 0xbe8503c1u, 0x87860204u, TAIL_AND_STORE,
};
#undef TAIL_AND_STORE

ShaderResourceTable output_table() {
    ShaderResourceTable table;
    ShaderResource out{};
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    table.resources.push_back(out);
    return table;
}

template <size_t N>
std::vector<uint32_t> compile(const uint32_t (&code)[N]) {
    ComputeShaderConfig config;
    config.local_x = kLanes;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    const ShaderResourceTable table = output_table();
    return recompile_compute(code, N, &table, config);
}

}   // namespace

TEST(ComputeScalarPairMask, DefinedPairStillProjects) {
    EXPECT_FALSE(compile(kDefinedLow).empty())
        << "a pair whose words are both definitely assigned is a per-lane bit, not a refusal";
    EXPECT_FALSE(compile(kDefinedHigh).empty());
}

TEST(ComputeScalarPairMask, OnePathPairRefuses) {
    EXPECT_TRUE(compile(kOnePath).empty())
        << "the skipped edge's fabricated zero must refuse, not stand in for a lane mask";
    EXPECT_TRUE(compile(kOnePathCopied).empty()) << "a copy of the merged pair carries its mark";
}
