// test_rdna2_predicate_tables — checks prosper's hand-maintained SOP1/VOP3 opcode-list
// predicates in rdna2_decode.hpp against an independent table: LLVM's AMDGPU backend.
//
// A wrong opcode list here is silent: too wide a list invents dependencies or drops mask writes
// (wrong pixels, no diagnostic — #2120's cmpx windows, the bcnt1 SCC entry, the SDWA integer
// family), too narrow a list only misses a proof. So every TEST below pins the UNSOUND direction
// exactly — no false positives against the external source — over the FULL opcode domain, not
// just the guest-observed members. Spot checks on observed packets cannot catch drift; enumerating
// the domain does, the same method that fixed the three #2120-class bugs (hand-maintained lists
// checked against LLVM encodings, here llvm-project/llvm/lib/Target/AMDGPU/SOPInstructions.td and
// VOP3Instructions.td plus AMD RDNA2 ISA 70648).
//
// Oracle sources (all independent of prosper):
//   [LLVM-SOP1] SOPInstructions.td: gfx10 opcode numbers come from the
//     `SOP1_Real_gfx6_gfx7_gfx10_*<0xNN>` defm lines; SCC behaviour comes from each pseudo
//     `def`'s `Defs` (S_NOT/S_WQM/S_BCNT/S_QUADMASK/S_ABS and every SAVEEXEC/WREXEC carry
//     `Defs = [SCC]`; S_MOV/S_CMOV/S_BREV/S_FF/S_FLBIT/S_SEXT/S_BITSET/S_GETPC/S_BITREPLICATE
//     carry none — S_CMOV only `Uses = [SCC]`); EXEC behaviour comes from the SAVEEXEC/WREXEC
//     pseudos' `Defs = [EXEC, SCC], Uses = [EXEC]`.
//   [LLVM-VOP3] VOP3Instructions.td: the VOP3B carry family uses the `VOP3b_*` profiles (sdst
//     scalar destination); the ISA manual sec 13.3.4 names the same ten opcodes the decoder lists.
//
// What this deliberately does NOT do: force the conservative gaps closed. Several LLVM-clean SOP1
// ops (BREV_B64, FF0 pair, FF1_B32, SEXT pair, BITSET_B64 pair, MOVRELS/D family) are absent from
// `sop1_opcode_leaves_scc_unmodified`, which only costs a missed proof. The TESTs pin that the
// listed set stays within the LLVM-clean set and that unknown opcodes stay conservative, so a
// future extension is a deliberate, reviewed diff rather than drift.
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

using namespace prosper::gpu;

namespace {

// [LLVM-SOP1] Every gfx10 SOP1 opcode whose pseudo carries `Defs = [SCC]` (directly or via the
// SAVEEXEC/WREXEC `Defs = [EXEC, SCC]` group). Anything here must NEVER be reported as leaving
// SCC unmodified: s_bcnt1_i32_b64 slipped into the old shared list this way and let a branch
// prove uniformity from a stale compare (see the header comment on the predicate).
constexpr uint32_t kLlvmSccWriters[] = {
    0x07u, 0x08u,  // S_NOT_B32/B64
    0x09u, 0x0Au,  // S_WQM_B32/B64
    0x0Du, 0x0Eu,  // S_BCNT0_I32_B32/B64
    0x0Fu, 0x10u,  // S_BCNT1_I32_B32/B64
    0x24u, 0x25u, 0x26u, 0x27u, 0x28u, 0x29u, 0x2Au, 0x2Bu,  // SAVEEXEC_B64 x8
    0x2Cu, 0x2Du,  // S_QUADMASK_B32/B64
    0x34u,        // S_ABS_I32
    0x37u, 0x38u,  // S_ANDN1/ORN1_SAVEEXEC_B64 (gfx9+)
    0x39u, 0x3Au,  // S_ANDN1/ANDN2_WREXEC_B64 (gfx9+)
    0x3Cu, 0x3Du, 0x3Eu, 0x3Fu, 0x40u, 0x41u, 0x42u, 0x43u, 0x44u,
    0x45u,  // SAVEEXEC_B32 x10
    0x46u, 0x47u,  // S_ANDN1/ANDN2_WREXEC_B32
};

// [LLVM-SOP1] gfx10 SOP1 opcodes whose pseudo carries NO `Defs = [SCC]`. This is a superset of
// the predicate's list: the unlisted members (BREV_B64, FF0 pair, FF1_B32, SEXT pair,
// BITSET_B64 pair, MOVRELS/D family, SETPC/SWAPPC/RFE control ops, MOVRELSD, BITREPLICATE is
// listed) are conservatively absent from the predicate, which only costs a missed proof.
// 0x32 (S_SET_GPR_IDX_IDX on some generations) is deliberately NOT here: its gfx10 encoding is
// unconfirmed, so it stays in the conservative-gap set below.
constexpr uint32_t kLlvmSccClean[] = {
    0x03u, 0x04u,  // S_MOV_B32/B64
    0x05u, 0x06u,  // S_CMOV_B32/B64 (Uses SCC, never Defs)
    0x0Bu, 0x0Cu,  // S_BREV_B32/B64
    0x11u, 0x12u,  // S_FF0_I32_B32/B64
    0x13u, 0x14u,  // S_FF1_I32_B32/B64
    0x15u, 0x16u,  // S_FLBIT_I32_B32/B64
    0x17u, 0x18u,  // S_FLBIT_I32/I64
    0x19u, 0x1Au,  // S_SEXT_I32_I8/I16
    0x1Bu, 0x1Cu,  // S_BITSET0_B32/B64
    0x1Du, 0x1Eu,  // S_BITSET1_B32/B64
    0x1Fu,        // S_GETPC_B64
    0x20u, 0x21u, 0x22u,  // S_SETPC/SWAPPC/RFE_B64 (control flow, no SCC Defs)
    0x2Eu, 0x2Fu,  // S_MOVRELS_B32/B64 (Uses M0 only)
    0x30u, 0x31u,  // S_MOVRELD_B32/B64 (Uses M0 only)
    0x3Bu,        // S_BITREPLICATE_B64_B32
    0x49u,        // S_MOVRELSD_2_B32 (Uses M0 only)
};

bool contains(const uint32_t* set, size_t n, uint32_t op) {
    for (size_t k = 0; k < n; ++k)
        if (set[k] == op) return true;
    return false;
}

// Opcodes with no confirmed gfx10 SOP1 assignment: below the MOV base, the documented holes,
// and everything past the last assigned op. The predicates must stay conservative here.
bool isSop1Gap(uint32_t op) {
    if (op <= 0x02u) return true;
    switch (op) {
        case 0x23u:
        case 0x32u:
        case 0x33u:
        case 0x35u:
        case 0x36u:
        case 0x48u:
            return true;
        default:
            break;
    }
    return op >= 0x4Au;
}

// VOP3 dword0 builder: prefix 0xD4, OP[25:16], VDST + mid field (abs/opsel/clamp bits).
uint32_t vop3_w0(uint32_t op, uint32_t vdst = 0u, uint32_t mid = 0u) {
    return 0xD4000000u | (op << 16) | mid | vdst;
}
uint32_t vop3_srcs(uint32_t s0, uint32_t s1, uint32_t s2) { return s0 | (s1 << 9) | (s2 << 18); }
Rdna2Inst decode_vop3(uint32_t op, uint32_t s2 = 0u, uint32_t mid = 0u) {
    const uint32_t code[2] = {vop3_w0(op, 4u, mid), vop3_srcs(256u, 257u, s2)};
    return rdna2_decode_one(code, 2);
}

}  // namespace

TEST(Rdna2PredicateTables, Sop1SccWritersAreNeverListedAsUnmodified) {
    for (uint32_t op : kLlvmSccWriters) {
        EXPECT_FALSE(sop1_opcode_leaves_scc_unmodified(op))
            << "SOP1 op=" << op << " writes SCC per [LLVM-SOP1]";
    }
}

TEST(Rdna2PredicateTables, Sop1UnmodifiedListStaysWithinLlvmCleanSet) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        if (!sop1_opcode_leaves_scc_unmodified(op)) continue;
        EXPECT_TRUE(contains(kLlvmSccClean, std::size(kLlvmSccClean), op))
            << "SOP1 op=" << op << " is listed but not LLVM-clean";
    }
}

TEST(Rdna2PredicateTables, Sop1GapsStayConservative) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        if (!isSop1Gap(op)) continue;
        EXPECT_FALSE(sop1_opcode_leaves_scc_unmodified(op)) << "gap op=" << op;
        EXPECT_FALSE(sop1_opcode_writes_exec_b64(op)) << "gap op=" << op;
        EXPECT_FALSE(sop1_opcode_writes_exec_b32(op)) << "gap op=" << op;
    }
}

TEST(Rdna2PredicateTables, Sop1SaveexecWrexecRangesMatchLlvm) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const bool saveexec_b64 =
            (op >= 0x24u && op <= 0x2Bu) || op == 0x37u || op == 0x38u;  // [LLVM-SOP1]
        const bool wrexec_b64 = op == 0x39u || op == 0x3Au;              // [LLVM-SOP1] gfx9+
        const bool saveexec_b32 = op >= 0x3Cu && op <= 0x45u;            // [LLVM-SOP1] gfx10
        const bool wrexec_b32 = op == 0x46u || op == 0x47u;             // [LLVM-SOP1] gfx10
        EXPECT_EQ(sop1_opcode_is_saveexec_b64(op), saveexec_b64) << "op=" << op;
        EXPECT_EQ(sop1_opcode_is_wrexec_b64(op), wrexec_b64) << "op=" << op;
        EXPECT_EQ(sop1_opcode_writes_exec_b64(op), saveexec_b64 || wrexec_b64)
            << "op=" << op;
        EXPECT_EQ(sop1_opcode_is_saveexec_b32(op), saveexec_b32) << "op=" << op;
        EXPECT_EQ(sop1_opcode_is_wrexec_b32(op), wrexec_b32) << "op=" << op;
        EXPECT_EQ(sop1_opcode_writes_exec_b32(op), saveexec_b32 || wrexec_b32)
            << "op=" << op;
    }
}

TEST(Rdna2PredicateTables, MayChangeExecFlagsEveryLlvmExecWriter) {
    // Neutral destination (SGPR0): only the opcode leg may fire. The dst/sdst legs are covered
    // by the EXEC-destination arms below.
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        Rdna2Inst in{};
        in.fmt = Rdna2Format::SOP1;
        in.opcode = op;
        in.dst = {OperandKind::SGPR, 0};
        in.n_src = 1;
        const bool writes_exec = (op >= 0x24u && op <= 0x2Bu) || (op >= 0x37u && op <= 0x3Au) ||
                                 (op >= 0x3Cu && op <= 0x47u);  // [LLVM-SOP1] Defs=[EXEC,SCC]
        EXPECT_EQ(rdna2_instruction_may_change_exec(in), writes_exec)
            << "SOP1 op=" << op;
    }
    // An explicit EXEC destination fires the predicate for any opcode, including a plain move.
    for (uint32_t dst : {126u, 127u}) {
        Rdna2Inst in{};
        in.fmt = Rdna2Format::SOP1;
        in.opcode = 0x03u;  // s_mov_b32: no implicit EXEC effect
        in.dst = {OperandKind::SGPR, static_cast<int32_t>(dst)};
        in.n_src = 1;
        EXPECT_TRUE(rdna2_instruction_may_change_exec(in)) << "dst=" << dst;
    }
}

TEST(Rdna2PredicateTables, Vop3bWhitelistIsExactlyTheTenCarryOps) {
    // [ISA 70648 sec 13.3.4] + [LLVM-VOP3] VOP3b_* profiles: v_add/sub/subrev_co_ci_u32,
    // v_div_scale_f32/f64, v_mad_u64_u32/i64_i32, v_add/sub/subrev_co_u32.
    const uint32_t members[] = {0x128u, 0x129u, 0x12Au, 0x16Du, 0x16Eu,
                                0x176u, 0x177u, 0x30Fu, 0x310u, 0x319u};
    // Nearest neighbours outside each group must NOT decode a scalar destination.
    const uint32_t neighbours[] = {0x127u, 0x12Bu, 0x16Cu, 0x16Fu, 0x175u,
                                   0x178u, 0x30Eu, 0x311u, 0x318u, 0x31Au};
    for (uint32_t op : members) {
        // mid sets all three ABS bits plus CLAMP: members must decode the scalar field as
        // SDST (not abs) while keeping CLAMP.
        const Rdna2Inst in = decode_vop3(op, 258u, (0x7u << 8) | (1u << 15));
        EXPECT_EQ(in.sdst.kind, OperandKind::SGPR) << "VOP3B op=" << op;
        EXPECT_EQ(in.sdst.value, 7) << "op=" << op;
        for (uint32_t k = 0; k < 3; ++k) EXPECT_FALSE(in.src_abs[k]) << "op=" << op;
        EXPECT_TRUE(in.clamp) << "CLAMP stays valid in VOP3B, op=" << op;
    }
    for (uint32_t op : neighbours) {
        const Rdna2Inst in = decode_vop3(op, 258u, (0x7u << 8) | (1u << 15));
        EXPECT_EQ(in.sdst.kind, OperandKind::None) << "non-VOP3B op=" << op;
        EXPECT_TRUE(in.src_abs[0] && in.src_abs[1] && in.src_abs[2])
            << "abs bits survive outside VOP3B, op=" << op;
    }
}

TEST(Rdna2PredicateTables, OpselCaptureSetIsExactlyTheSixteenBitFamily) {
    // The decoder honours OPSEL[2:0]+OPSEL[3] only for scalar 16-bit VOP3 ops. Members below are
    // the 0x303-0x30E integer VALU run (0x306 is not an instruction), the f16 min/max/med triples
    // (llvm-mc gfx1030: 0x351/0x354/0x357 f16, interleaved i16 at 0x352/0x355/0x358 and u16 at
    // 0x353/0x356/0x359), and the admitted singles 0x311/0x314/0x340/0x34B/0x35E.
    const uint32_t members[] = {0x311u, 0x34Bu, 0x351u, 0x354u, 0x357u, 0x303u, 0x304u,
                                0x305u, 0x307u, 0x308u, 0x309u, 0x30Au, 0x30Bu, 0x30Cu,
                                0x30Du, 0x30Eu, 0x314u, 0x340u, 0x35Eu, 0x352u, 0x353u,
                                0x355u, 0x356u, 0x358u, 0x359u};
    for (uint32_t op = 0x300u; op <= 0x36Fu; ++op) {
        const Rdna2Inst in = decode_vop3(op, 258u, 0xBu << 11);
        const bool expected = contains(members, std::size(members), op);
        EXPECT_EQ(in.vop3p_opsel, expected ? 0xBu : 0u) << "VOP3 op=" << op;
    }
    EXPECT_EQ(decode_vop3(0x306u, 258u, 0xFu << 11).vop3p_opsel, 0u) << "0x306 hole";
}

TEST(Rdna2PredicateTables, TwoSourcePredicateMatchesDecodedArity) {
    // The predicate and the decoder's SRC2 clearing must agree on every VOP3 opcode: a listed op
    // decodes two sources (reserved SRC2 reads as s0 and must not escape), an unlisted op keeps
    // all three (fail-closed for unsupported encodings).
    for (uint32_t op = 0x100u; op <= 0x3FFu; ++op) {
        const Rdna2Inst in = decode_vop3(op, 0u);
        const bool listed = emitted_vop3_opcode_has_two_data_sources(op);
        EXPECT_EQ(in.n_src == 2 && in.src[2].kind == OperandKind::None, listed)
            << "VOP3 op=" << op;
    }
    // Negative controls: genuine three-source ops the list must never absorb. 0x128 keeps its
    // carry-in (the live GTA V _co_ci_ consumer), 0x143/0x14B/0x36D/0x371 are ordinary ternaries.
    for (uint32_t op : {0x128u, 0x143u, 0x14Bu, 0x36Du, 0x371u}) {
        EXPECT_FALSE(emitted_vop3_opcode_has_two_data_sources(op))
            << "three-source VOP3 op=" << op;
        EXPECT_EQ(decode_vop3(op, 0u).n_src, 3) << "op=" << op;
    }
    // The 0x306 hole is not an instruction but still decodes fail-closed with three fields.
    EXPECT_FALSE(emitted_vop3_opcode_has_two_data_sources(0x306u));
    EXPECT_EQ(decode_vop3(0x306u, 0u).n_src, 3);
}
