// Interpolation (VINTRP) reads VGPR params and M0 only: it never reads an ordinary SGPR, so the
// scalar liveness proof must scan through it the way it scans through DS/EXP. Stopping the walk
// there kept a scratch SGPR written inside a forward EXECZ region live past its only consumer and
// refused the merge (Black Flag ps_407eece900: s4 compared at pc 43, read at pc 49, merged at 51).
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kMovB32S4S0 = 0xbe840300u;    // s_mov_b32 s4, s0
constexpr uint32_t kInterpP1 = 0xc8100100u;       // v_interp_p1_f32 v4, v0, attr0.x
constexpr uint32_t kAddS5S4S0 = 0x80050004u;      // s_add_u32 s5, s4, s0
constexpr uint32_t kEnd = 0xbf810000u;            // s_endpgm

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> result;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), result), code.size());
    return result;
}
}   // namespace

TEST(VintrpScalarLiveness, InterpolationDoesNotReadOrdinarySgprs) {
    const auto interp = decode({kInterpP1, kEnd});
    ASSERT_EQ(interp[0].fmt, Rdna2Format::VINTRP);
    const auto mov = decode({kMovB32S4S0, kEnd});
    ASSERT_EQ(mov[0].fmt, Rdna2Format::SOP1);
    ASSERT_EQ(mov[0].dst.value, 4);
    const auto program = decode({kMovB32S4S0, kInterpP1, kEnd});
    EXPECT_TRUE(sgpr_dead_at_merge(program, 0, 4)) << "s4 dies at its own redefinition";
    EXPECT_TRUE(sgpr_dead_at_merge(program, 1, 4)) << "VINTRP reads no ordinary SGPR";
}

TEST(VintrpScalarLiveness, AnSgprReadAfterInterpolationStillBlocks) {
    const auto program = decode({kMovB32S4S0, kInterpP1, kAddS5S4S0, kEnd});
    EXPECT_FALSE(sgpr_dead_at_merge(program, 1, 4)) << "s_add reads s4";
}

TEST(VintrpScalarLiveness, M0IsNeverDeadAcrossInterpolation) {
    // M0 is never dead anywhere (unmodelled implicit readers); the carve-out holds here too.
    const auto program = decode({kMovB32S4S0, kInterpP1, kEnd});
    EXPECT_FALSE(sgpr_dead_at_merge(program, 1, 124));
}
