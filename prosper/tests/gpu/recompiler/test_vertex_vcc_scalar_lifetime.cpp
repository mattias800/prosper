// #5014: a scalar VCC high-word lifetime needs no predicate when the original CFG
// proves its sibling unobserved as a mask. Reconstructing a retired predicate would
// reject genuine descriptor arithmetic or silently restore departed lane bits.
#include "fragment_predicate_oracle.hpp"
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
} // namespace prosper::gpu
