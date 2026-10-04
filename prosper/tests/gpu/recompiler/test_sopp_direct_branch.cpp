// test_sopp_direct_branch — pins sopp_opcode_is_direct_branch against the ISA SOPP table.
//
// The predicate decides which instructions carry a branch target for control-flow discovery
// (closed-tail admission, CFG building). Missing a branch loses an edge; inventing one invents
// control flow. The set is small but load-bearing: s_branch (0x02) plus the six s_cbranch_*
// conditionals (scc0/scc1/vccz/vccnz/execz/execnz, 0x04-0x09). Everything else — s_nop,
// s_endpgm, barriers, traps, waits — is not a direct branch.
//
// Oracle: AMD RDNA2 ISA 70648 SOPP opcode table (primary contract). Note rdna2_decode.cpp
// carries a local copy of this exact set for tail admission; the two must stay identical (the
// #2120 duplication lesson), and this pin is what fails first if either drifts.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper::gpu;

TEST(SoppDirectBranch, ExactSetMatchesTheIsaTable) {
    for (uint32_t op = 0; op <= 0x7Fu; ++op) {
        const bool direct = op == 0x02u || (op >= 0x04u && op <= 0x09u);
        EXPECT_EQ(sopp_opcode_is_direct_branch(op), direct) << "SOPP op=" << op;
    }
    for (uint32_t op = 0x80u; op <= 0xFFu; ++op) {
        EXPECT_FALSE(sopp_opcode_is_direct_branch(op)) << "SOPP op=" << op;
    }
}

TEST(SoppDirectBranch, NamedConstantsNameTheSameSet) {
    EXPECT_EQ(kSoppOpcodeBranch, 0x02u);
    EXPECT_EQ(kSoppOpcodeCbranchScc0, 0x04u);
    EXPECT_EQ(kSoppOpcodeCbranchExecz, 0x08u);
    EXPECT_EQ(kSoppOpcodeCbranchExecnz, 0x09u);
    EXPECT_TRUE(sopp_opcode_is_direct_branch(kSoppOpcodeBranch));
    EXPECT_TRUE(sopp_opcode_is_direct_branch(kSoppOpcodeCbranchScc0));
    EXPECT_TRUE(sopp_opcode_is_direct_branch(kSoppOpcodeCbranchExecz));
    EXPECT_TRUE(sopp_opcode_is_direct_branch(kSoppOpcodeCbranchExecnz));
}
