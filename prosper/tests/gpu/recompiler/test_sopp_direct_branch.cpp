// test_sopp_direct_branch — pins sopp_opcode_is_direct_branch, the SOPP branch-target predicate.
//
// The predicate decides which instructions carry a branch target for control-flow discovery
// (closed-tail admission, CFG building). Missing a branch loses an edge; inventing one invents
// control flow. It admits s_branch (0x02) and the six s_cbranch_* conditionals
// (scc0/scc1/vccz/vccnz/execz/execnz, 0x04-0x09). Everything else — s_nop, s_endpgm, barriers,
// traps, waits — reports false.
//
// Oracle: AMD RDNA2 ISA 70648 SOPP opcode table, cross-checked with llvm-mc gfx1030, which lists
// ELEVEN SOPP ops with a SIMM16 branch target: the seven above plus the four debugger branches
// s_cbranch_cdbgsys / cdbguser / cdbgsys_or_user / cdbgsys_and_user (0x17-0x1A). The predicate
// excludes those four DELIBERATELY: prosper runs with no GPU debugger attached, so they are never
// taken (the emitter lowers 0x17 as a fall-through and rejects the other three), and the analyses
// that must stay conservative add them explicitly beside this predicate. Do not "fix" the
// exclusion by widening the predicate; that would invent edges for every other caller.
//
// Scope: this pins the header predicate only. rdna2_decode.cpp keeps a private copy of the set
// for closed-tail admission, which this suite does not call.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper::gpu;

TEST(SoppDirectBranch, ExactSetOverTheSevenBitOpcodeSpace) {
    // SOPP's opcode field is 7 bits, so 0x00-0x7F is every value the decoder can produce.
    for (uint32_t op = 0; op <= 0x7Fu; ++op) {
        const bool direct = op == 0x02u || (op >= 0x04u && op <= 0x09u);
        EXPECT_EQ(sopp_opcode_is_direct_branch(op), direct) << "SOPP op=0x" << std::hex << op;
    }
}

TEST(SoppDirectBranch, NamedConstantsNameTheSameSet) {
    EXPECT_EQ(kSoppOpcodeBranch, 0x02u);
    EXPECT_EQ(kSoppOpcodeCbranchScc0, 0x04u);
    EXPECT_EQ(kSoppOpcodeCbranchExecz, 0x08u);
    EXPECT_EQ(kSoppOpcodeCbranchExecnz, 0x09u);
}

TEST(SoppDirectBranch, DebuggerBranchesAreDeliberatelyExcluded) {
    // ISA branches with a target, but never taken without an attached debugger (see header).
    EXPECT_EQ(kSoppOpcodeCbranchCdbgsys, 0x17u);
    EXPECT_EQ(kSoppOpcodeCbranchCdbguser, 0x18u);
    EXPECT_EQ(kSoppOpcodeCbranchCdbgsysOrUser, 0x19u);
    EXPECT_EQ(kSoppOpcodeCbranchCdbgsysAndUser, 0x1Au);
    for (uint32_t op : {kSoppOpcodeCbranchCdbgsys, kSoppOpcodeCbranchCdbguser,
                        kSoppOpcodeCbranchCdbgsysOrUser, kSoppOpcodeCbranchCdbgsysAndUser}) {
        EXPECT_FALSE(sopp_opcode_is_direct_branch(op)) << "SOPP op=0x" << std::hex << op;
    }
}
