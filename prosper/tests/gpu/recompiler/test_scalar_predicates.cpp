// test_scalar_predicates — pins the three small scalar opcode predicates in rdna2_decode.hpp
// against their contracts. Each is one line of logic feeding analyses that trust it blindly,
// and none had any test reference.
//
// Oracle sources (all primary contracts, independent of prosper):
//   sop2_is_b32_logical — LLVM SOPInstructions.td: exactly the SOP2_32-class logic ops, gfx10
//     numbers AND 0x0E, OR 0x10, XOR 0x12, ANDN2 0x14, ORN2 0x16, NAND 0x18, NOR 0x1A, XNOR 0x1C
//     (SOP2_Real_gfx6_gfx7_gfx10 defms; the B64 siblings are the odd neighbours and differ).
//   smem_opcode_is_buffer_load — ISA SMEM opcode table: s_buffer_load_dword[xN] occupy
//     0x08-0x0C, right after the plain s_load range.
//   sop1_opcode_is_emitted_saveexec_b32 — deliberately narrower than the architectural family:
//     only the three B32 forms observed in guest shaders (0x3C AND, 0x40 ORN2, 0x44 ANDN1).
//     Pinned exactly so growing it is a reviewed decision, not drift; the architectural range
//     itself is covered by the saveexec/wrexec oracle in test_rdna2_predicate_tables.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper::gpu;

TEST(ScalarPredicates, B32LogicalSetMatchesLlvmSop232Class) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const bool logical_b32 = op == 0x0Eu || op == 0x10u || op == 0x12u || op == 0x14u ||
                                 op == 0x16u || op == 0x18u || op == 0x1Au || op == 0x1Cu;
        EXPECT_EQ(sop2_is_b32_logical(op), logical_b32) << "SOP2 op=" << op;
    }
    EXPECT_EQ(kSop2OpcodeAndB32, 0x0Eu);
    EXPECT_EQ(kSop2OpcodeXnorB32, 0x1Cu);
}

TEST(ScalarPredicates, SmemBufferLoadRangeMatchesTheIsaTable) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        EXPECT_EQ(smem_opcode_is_buffer_load(op), op >= 0x08u && op <= 0x0Cu)
            << "SMEM op=" << op;
    }
    EXPECT_EQ(kSmemOpcodeBufferLoadDword, 0x08u);
    EXPECT_EQ(kSmemOpcodeBufferLoadDwordX16, 0x0Cu);
}

TEST(ScalarPredicates, EmittedSaveexecStaysTheThreeObservedForms) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const bool emitted = op == 0x3Cu || op == 0x40u || op == 0x44u;
        EXPECT_EQ(sop1_opcode_is_emitted_saveexec_b32(op), emitted) << "SOP1 op=" << op;
    }
}
