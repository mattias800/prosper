// test_sop2_dest_width — checks rdna2_sop2_dest_dwords against LLVM's AMDGPU backend.
//
// Under-counting a destination is silent: control-flow discovery, scalar liveness and the
// register file then reason about a stale SGPR while the instruction overwrote its neighbour.
// Over-counting only wastes a register. So this pins the UNSOUND direction exactly — every
// 64-bit SOP2 opcode must report a pair — over the full opcode domain, while the B32 side is
// pinned one-way (a reported single dword must really be B32; LLVM-B32 ops the function leaves
// at the fail-closed pair width stay conservative and are documented, not forced).
//
// Oracle: llvm-project/llvm/lib/Target/AMDGPU/SOPInstructions.td, independent of prosper.
// Width comes from each pseudo's class — SOP2_64 / SOP2_64_32 / SOP2_64_32_32 (pair) versus
// SOP2_32 (single) — and gfx10 opcode numbers from the SOP2_Real_gfx6_gfx7_gfx10_* defm lines
// (e.g. S_CSELECT_B32 0x0A, S_AND_B32 0x0E, S_BFE_U32 0x27, S_ABSDIFF_I32 0x2C,
// S_LSHL1_ADD_U32 0x2E, S_PACK_LL_B32_B16 0x32, S_MUL_HI_U32 0x35). Beware the older _vi
// numbering in the same file (S_AND_B32_vi is 0x0C there); only the gfx10 Real numbers apply.
// SALU-float SOP2 ops have no gfx10 encoding in LLVM (gfx11+ only), so the domain ends at 0x36.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

using namespace prosper::gpu;

namespace {

// [LLVM] Every gfx10 SOP2 opcode whose pseudo writes a 64-bit SDST: the B64 logic/shift pairs,
// the 64-bit shifts of a 32-bit count, the B64 bit-field ops and the B64 bit-manipulate forms.
constexpr uint32_t kLlvmPair[] = {
    0x0Bu,                                                  // S_CSELECT_B64
    0x0Fu, 0x11u, 0x13u, 0x15u, 0x17u, 0x19u, 0x1Bu, 0x1Du,  // B64 logic family
    0x1Fu, 0x21u, 0x23u,                                    // S_LSHL/LSHR/ASHR_B64/I64
    0x25u,                                                  // S_BFM_B64
    0x29u, 0x2Au,                                           // S_BFE_U64/I64
};

// [LLVM] Every gfx10 SOP2 opcode whose pseudo writes a 32-bit SDST. A superset of what the
// function reports as single: S_ABSDIFF_I32, the S_LSHLn_ADD_U32 row, the S_PACK row and the
// S_MUL_HI pair stay at the fail-closed pair width today (conservative: a missed narrow proof,
// never a stale register), so they are documented here but not forced.
constexpr uint32_t kLlvmSingle[] = {
    0x00u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u, 0x09u, 0x0Au,
    0x0Eu, 0x10u, 0x12u, 0x14u, 0x16u, 0x18u, 0x1Au, 0x1Cu,  // B32 logic family
    0x1Eu, 0x20u, 0x22u,                                    // S_LSHL/LSHR/ASHR_B32/I32
    0x24u,                                                  // S_BFM_B32
    0x26u,                                                  // S_MUL_I32
    0x27u, 0x28u,                                           // S_BFE_U32/I32
    0x2Cu,                                                  // S_ABSDIFF_I32
    0x2Eu, 0x2Fu, 0x30u, 0x31u,                              // S_LSHL1-4_ADD_U32
    0x32u, 0x33u, 0x34u,                                    // S_PACK_LL/LH/HH_B32_B16
    0x35u, 0x36u,                                           // S_MUL_HI_U32/I32
};

bool contains(const uint32_t* set, size_t n, uint32_t op) {
    for (size_t k = 0; k < n; ++k)
        if (set[k] == op) return true;
    return false;
}

// Opcodes with no gfx10 SOP2 assignment: the 0x0C/0x0D hole, the 0x2B/0x2D holes, and everything
// past the last assigned op (SALU-float SOP2 starts at 0x40 on gfx11+ and does not exist here).
bool isSop2Gap(uint32_t op) {
    return op == 0x0Cu || op == 0x0Du || op == 0x2Bu || op == 0x2Du || op >= 0x37u;
}

}  // namespace

TEST(Rdna2Sop2Width, PairWritersAreNeverNarrowed) {
    for (uint32_t op : kLlvmPair) {
        EXPECT_EQ(rdna2_sop2_dest_dwords(op), 2u) << "SOP2 op=" << op;
    }
}

TEST(Rdna2Sop2Width, ReportedSinglesAreGenuinelyB32) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        if (rdna2_sop2_dest_dwords(op) != 1u) continue;
        EXPECT_TRUE(contains(kLlvmSingle, std::size(kLlvmSingle), op)) << "SOP2 op=" << op;
    }
}

TEST(Rdna2Sop2Width, GapsStayFailClosedToAPair) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        if (!isSop2Gap(op)) continue;
        EXPECT_EQ(rdna2_sop2_dest_dwords(op), 2u) << "gap op=" << op;
    }
}
