// The counter boundary grants only DATA ancestry. Authentic initialization, real exit
// deadness and the absence of cross-lane producers remain independent obligations.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_loop_exec_data.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;
std::vector<uint32_t> counter_code() {
    return {0xbea90380u, 0x7d840100u, 0x7da306f9u, 0x86860029u, 0xbf880002u,
            0x81298129u, 0xbf82fffau, 0xbea90380u, 0xbf810000u};
}
struct Frame {
    SpirvCompute b;
    RegState rs;
    std::vector<Rdna2Inst> ins;
    DivLoop loop;
    explicit Frame(const std::vector<uint32_t>& words = counter_code()) {
        ComputeShaderConfig launch;
        launch.local_x = launch.wave_size = launch.native_subgroup_size = 64;
        launch.user_sgprs.resize(16, 0);
        prosper::test::complete_compute_launch(launch);
        b.native_subgroup_size = 64;
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        prosper::test::bind_checked_compute_launch(b, launch);
        b.saved_mask_original_words = words;
        b.saved_mask_original_compute_launch = b.guest_compute_launch;
        rdna2_walk(words.data(), words.size(), ins);
        rs.exec = b.ucmp(Op_ULessThan, b.localid_comp[0], b.uconst(33));
        rs.exec_narrowed = true;
        rs.vector_wave_size = 64;
        rs.sreg[41] = b.uconst(0);
        for (const auto& in : ins) {
            if (in.fmt != Rdna2Format::SOPP) continue;
            if (in.opcode == kSoppOpcodeCbranchExecz) loop.exit_branch_pc = in.pc;
            if (in.opcode == kSoppOpcodeBranch && branch_target(in) < in.pc) {
                loop.header_pc = branch_target(in);
                loop.backedge_pc = in.pc;
                loop.exit_pc = in.pc + in.len_dwords;
            }
        }
        loop.condition = DivLoop::Condition::Exec;
    }
    bool bound() const { return native_loop_data_exec_bound(b, ins, loop, rs); }
};
} // namespace

TEST(UniformCounterDataBoundary, DataAncestryAddsNeitherARouteNorPhysicalWords) {
    Frame f;
    ASSERT_TRUE(f.bound());
    const auto scalars = f.rs.sreg;
    const auto vectors = f.rs.vreg;
    LoopExecSubset proof(f.b, f.ins, f.loop, f.rs);
    EXPECT_FALSE(proof.admitted());
    EXPECT_TRUE(proof.has_data_exec_bound());
    EXPECT_EQ(f.rs.sreg, scalars);
    EXPECT_EQ(f.rs.vreg, vectors);
    EXPECT_TRUE(f.rs.sreg_bool.empty());
    EXPECT_TRUE(f.rs.vreg_source_marks.empty());
}

TEST(UniformCounterDataBoundary, AScalarLiveOutRetainsItsExistingRefusal) {
    auto words = counter_code();
    words[7] = 0x7e0e0229u; // v7 = s41 before any scalar replacement
    Frame f(words);
    ASSERT_EQ(loop_live_wave_scalar(f.ins, f.loop), 41);
    EXPECT_FALSE(f.bound());
}

TEST(UniformCounterDataBoundary, AVccLiveOutRetainsItsExistingRefusal) {
    auto words = counter_code();
    words[7] = 0x02020701u; // v1 = VCC ? v3 : v1 before any VCC replacement
    Frame f(words);
    ASSERT_EQ(loop_live_wave_scalar(f.ins, f.loop), 106);
    EXPECT_FALSE(f.bound());
}

TEST(UniformCounterDataBoundary, AParsedInitializationCannotReplaceItsOriginalPacket) {
    Frame f;
    f.ins.front().words[0] ^= 1;
    EXPECT_FALSE(f.bound());
}

TEST(UniformCounterDataBoundary, EveryPreheaderPathMustRunTheLiteralInitialization) {
    auto words = counter_code();
    words.insert(words.begin(), 0xbf840001u); // SCC branch skips the literal move
    words.insert(words.begin() + 2, 0xbf800000u); // target lies before the loop header
    Frame f(words);
    ASSERT_EQ(f.loop.header_pc, 3u);
    ASSERT_LT(branch_target(f.ins.front()), f.loop.header_pc);
    EXPECT_FALSE(f.bound());
}

TEST(UniformCounterDataBoundary, AnotherSourceCannotBecomeTheCarriedCounter) {
    auto words = counter_code();
    words[5] = 0x81298128u; // s41 = s40 + 1, not the initialized counter
    Frame f(words);
    ASSERT_EQ(f.ins[4].fmt, Rdna2Format::SOP2);
    ASSERT_EQ(f.ins[4].src[0].kind, OperandKind::SGPR);
    ASSERT_EQ(f.ins[4].src[0].value, 40);
    EXPECT_FALSE(f.bound());
}

TEST(UniformCounterDataBoundary, DeadReadfirstResultsDoNotAuthenticateCrossLaneBodies) {
    auto words = counter_code();
    words.insert(words.begin() + 5, 0x7e560500u); // s43 = readfirstlane(v0)
    words[4] = 0xbf880003u;
    words[7] = 0xbf82fff9u;
    words.insert(words.end() - 1, 0xbeab0380u); // independently kill s43 after the loop
    Frame f(words);
    ASSERT_LT(loop_live_wave_scalar(f.ins, f.loop), 0);
    EXPECT_FALSE(f.bound());
}

TEST(UniformCounterDataBoundary, MissingOrFabricatedCounterStorageStillRefuses) {
    Frame missing;
    missing.rs.sreg.erase(41);
    EXPECT_FALSE(missing.bound());
    Frame fabricated;
    fabricated.rs.sreg_merge_placeholder.insert(41);
    EXPECT_FALSE(fabricated.bound());
}
