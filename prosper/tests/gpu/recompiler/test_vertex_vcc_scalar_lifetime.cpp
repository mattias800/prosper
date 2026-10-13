// #5014: a scalar VCC high-word lifetime needs no predicate when the original CFG
// proves its sibling unobserved as a mask. Reconstructing a retired predicate would
// reject genuine descriptor arithmetic or silently restore departed lane bits.
#include "fragment_predicate_oracle.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

namespace prosper::gpu {
namespace {
constexpr uint32_t kEnd = 0xbf810000u;
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.begin_vertex();
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.scc = b.bfalse();
    }
    bool instruction(uint32_t word) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(&word, 1, ins);
        if (ins.size() != 1) return false;
        const auto before = snapshot_saved_b64_masks(b, rs, ins[0]);
        bool ok = true;
        if (!emit_alu(b, rs, ins[0], ok, true, nullptr, true, nullptr) || !ok) return false;
        record_scalar_write(rs, ins[0], false, before);
        return true;
    }
    bool body(const std::vector<uint32_t>& code) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        return emit_body(b, rs, ins, {}, nullptr, true, false, {}, code.data(), code.size());
    }
};
std::vector<uint32_t> program() {
    // Both CSELECT alternatives genuinely define 16; AND consumes that current
    // high DATA word, followed by an ordinary scalar-to-vector observation.
    return {0x856b9090u, 0x876bff6bu, 0x1f0u, 0x7e00026bu, kEnd};
}
} // namespace

TEST(VertexVccScalarLifetime, DeadSiblingKeepsTheExactCurrentHighDword) {
    Emitter e;
    ASSERT_TRUE(e.body(program()));
    EXPECT_TRUE(e.b.vcc_b32_low_only_pcs.contains(1u));
    EXPECT_EQ(e.rs.vcc, 0u) << "scalar-only admission must not reconstruct a lane predicate";
    EXPECT_FALSE(e.rs.sreg_bool.contains(106));
    ASSERT_TRUE(e.rs.vreg.contains(0));
    fragment_predicate_oracle::Values values(e.b);
    const auto observed = values.evaluate(e.rs.vreg.at(0), 0);
    ASSERT_TRUE(observed.has_value()) << values.error;
    EXPECT_EQ(observed.value_or(UINT32_MAX), 16u)
        << "the original current high word reaches its DATA consumer";
    EXPECT_FALSE(values.evaluate(UINT32_MAX, 0).has_value())
        << "an undefined SSA id must not masquerade as a measured zero";
}

TEST(VertexVccScalarLifetime, AReachableImplicitMaskConsumerStillRequiresTheSibling) {
    Emitter e;
    auto code = program();
    code.insert(code.end() - 1, 0x02000080u); // V_CNDMASK_B32 v0,0,v0,VCC
    EXPECT_FALSE(e.body(code));
    EXPECT_FALSE(e.b.vcc_b32_low_only_pcs.contains(1u));
    EXPECT_EQ(e.rs.vcc, 0u) << "a current high DATA word cannot restore the retired predicate";
}

TEST(VertexVccScalarLifetime, CurrentDataArithmeticDoesNotNeedAReplacementPredicate) {
    Emitter e;
    // MOV ends the old predicate. AND observes the actual current numeric low word; no lane
    // identity or ballot can be inferred from the generic vertex invocation's packing.
    const std::vector<uint32_t> code{0xbeea0387u, 0xbeeb0380u, 0x876a836au, 0x7e00026au, kEnd};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    // Exercise the real instruction emitter without borrowing the separate whole-CFG
    // dead-sibling fact. A subsequent mask consumer retains its own obligation below.
    for (const auto& in : ins) {
        if (in.is_end) break;
        const auto before = snapshot_saved_b64_masks(e.b, e.rs, in);
        bool ok = true;
        ASSERT_TRUE(emit_alu(e.b, e.rs, in, ok, true, nullptr, true, nullptr));
        ASSERT_TRUE(ok) << "pc " << in.pc;
        record_scalar_write(e.rs, in, false, before);
    }
    EXPECT_TRUE(e.b.vcc_b32_low_only_pcs.empty());
    EXPECT_EQ(e.rs.vcc, 0u);
    ASSERT_TRUE(e.rs.vreg.contains(0));
    fragment_predicate_oracle::Values values(e.b);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto observed = values.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(observed.has_value()) << values.error;
        EXPECT_EQ(observed.value_or(UINT32_MAX), 3u) << "lane " << lane;
    }
    EXPECT_FALSE(values.evaluate(UINT32_MAX, 0).has_value());
}

TEST(VertexVccScalarLifetime, DirectDataCannotAuthenticateAnAbsentSelfSource) {
    Emitter e;
    ASSERT_TRUE(e.instruction(0xbeeb0380u)); // genuine zero VCC_HI, retired predicate
    ASSERT_FALSE(e.rs.sreg.contains(106));
    EXPECT_FALSE(e.instruction(0x886a826au)); // OR VCC_LO,VCC_LO,2
    EXPECT_EQ(e.rs.vcc, 0u);
}

TEST(VertexVccScalarLifetime, DirectDataCannotAuthenticateAMarkedSelfSource) {
    for (bool unavailable : {false, true}) {
        Emitter e;
        ASSERT_TRUE(e.instruction(0xbeea0387u));
        ASSERT_TRUE(e.instruction(0xbeeb0380u));
        if (unavailable)
            e.rs.unavailable_saved_mask_words.insert(106);
        else
            e.rs.sreg_merge_placeholder.insert(106);
        EXPECT_FALSE(e.instruction(0x886a826au)) << "unavailable=" << unavailable;
        EXPECT_EQ(e.rs.vcc, 0u);
    }
}

TEST(VertexVccScalarLifetime, DirectDataCannotAuthenticateItsNewSccFromAPoisonedInput) {
    for (bool marked : {false, true}) {
        Emitter e;
        ASSERT_TRUE(e.instruction(0xbeeb0380u)); // the untouched sibling is genuine DATA
        if (marked)
            e.rs.scc_merge_placeholder = true;
        else
            e.rs.scc = 0;
        EXPECT_FALSE(e.instruction(0x886a82fdu)) << "marked=" << marked; // OR VCC_LO,SCC,2
        EXPECT_EQ(e.rs.vcc, 0u);
    }
}

TEST(VertexVccScalarLifetime, ACurrentDataPairCannotBorrowAMissingSibling) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(2);
    const std::vector<uint32_t> code{0x806a8382u, 0x886a6a00u, 0x7e020287u, 0x02000289u, kEnd};
    EXPECT_FALSE(e.body(code));
    EXPECT_EQ(e.rs.vcc, 0u);
}

TEST(VertexVccScalarLifetime, ACurrentDataPairCannotBorrowAMergePlaceholder) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(2);
    e.rs.sreg[107] = e.b.uconst(0);
    e.rs.sreg_merge_placeholder.insert(107);
    const std::vector<uint32_t> code{0x806a8382u, 0x886a6a00u, 0x7e020287u, 0x02000289u, kEnd};
    EXPECT_FALSE(e.body(code));
    EXPECT_EQ(e.rs.vcc, 0u);
}

TEST(VertexVccScalarLifetime, ACurrentDataPairCannotLaunderAMissingScalarSource) {
    Emitter e;
    e.rs.sreg[0] = e.b.uconst(2);
    const std::vector<uint32_t> code{0x806a8114u, 0xbeeb0380u, 0x886a6a00u,
                                     0x7e020287u, 0x02000289u, kEnd};
    EXPECT_FALSE(e.body(code)) << "s20's missing value remains marked through the ADD";
    EXPECT_EQ(e.rs.vcc, 0u);
}
} // namespace prosper::gpu
