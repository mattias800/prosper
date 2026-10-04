// test_rdna2_decode_sweep -- exhaustive per-family sweep of the RDNA2 instruction walker.
//
// test_rdna2_decode checks a hand-picked stream of guest-observed shapes. This suite instead walks
// EVERY opcode of each encoding family, building words from the published RDNA2 field layout (ISA
// reference 70648: encoding prefix, opcode field, operand fields) rather than from an assembler, and
// asserts the three properties a stream walker depends on: the format class, the length in dwords
// (inline-literal and always-literal rules included) and that the opcode field round-trips. A wrong
// length is the silent failure -- the trailing dword re-decodes as a phantom instruction and every
// later pc desyncs -- so this is the defect class the sweep is aimed at.
//
// Words are built by this file from the layout; no third-party test text is reused.
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr uint32_t kLiteralSrc = 0xFFu;      // SSRC/SRC field value selecting a trailing literal
constexpr uint32_t kS_ENDPGM   = 0xBF810000u;

// Decode with one trailing dword of padding so a literal-bearing form is not truncated.
Rdna2Inst decode(uint32_t w0, uint32_t w1 = 0xDEADBEEFu) {
    const uint32_t code[3] = {w0, w1, 0xCAFEF00Du};
    return rdna2_decode_one(code, 3);
}

}  // namespace

TEST(Rdna2DecodeSweep, Sop2EveryOpcodeIsOneDwordWithoutLiteral) {
    // SOP2: [31:30]=10, OP[29:23] (0x00..0x2F; 0x30+ is SOPK), SDST[22:16], SSRC1[15:8], SSRC0[7:0].
    for (uint32_t op = 0; op <= 0x2F; ++op) {
        const uint32_t w = 0x80000000u | (op << 23) | (4u << 16) | (2u << 8) | 1u;
        const Rdna2Inst in = decode(w);
        EXPECT_EQ(in.fmt, Rdna2Format::SOP2) << "op=" << op;
        EXPECT_EQ(in.len_dwords, 1u) << "op=" << op;
        EXPECT_EQ(in.opcode, op) << "op=" << op;
        EXPECT_FALSE(in.has_literal) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Sop2LiteralOnEitherSourceAddsADword) {
    for (uint32_t op = 0; op <= 0x2F; ++op) {
        const uint32_t base = 0x80000000u | (op << 23) | (4u << 16);
        const Rdna2Inst a = decode(base | (2u << 8) | kLiteralSrc, 0x12345678u);
        EXPECT_EQ(a.len_dwords, 2u) << "ssrc0 literal, op=" << op;
        EXPECT_TRUE(a.has_literal) << "op=" << op;
        EXPECT_EQ(a.literal, 0x12345678u) << "op=" << op;
        const Rdna2Inst b = decode(base | (kLiteralSrc << 8) | 1u, 0x9ABCDEF0u);
        EXPECT_EQ(b.len_dwords, 2u) << "ssrc1 literal, op=" << op;
        EXPECT_EQ(b.literal, 0x9ABCDEF0u) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, SopkLengthIsOneExceptSetregImm32) {
    // SOPK: [31:28]=1011, OP[27:23]. Only op 21 (s_setreg_imm32_b32) carries a mandatory literal.
    // Field values 0x1D..0x1F are not SOPK: they spell the SOP1 (0xBE80), SOPC (0xBF00) and SOPP
    // (0xBF80) prefixes, so the family is 0x00..0x1C.
    for (uint32_t op = 0; op <= 0x1C; ++op) {
        const uint32_t w = 0xB0000000u | (op << 23) | (3u << 16) | 0x1234u;
        const Rdna2Inst in = decode(w, 0x55AA55AAu);
        EXPECT_EQ(in.fmt, Rdna2Format::SOPK) << "op=" << op;
        EXPECT_EQ(in.len_dwords, op == 21u ? 2u : 1u) << "op=" << op;
        EXPECT_EQ(in.has_literal, op == 21u) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Sop1EveryOpcode) {
    // SOP1: prefix 0xBE80, SDST[22:16], OP[15:8], SSRC0[7:0].
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const Rdna2Inst plain = decode(0xBE800000u | (4u << 16) | (op << 8) | 1u);
        EXPECT_EQ(plain.fmt, Rdna2Format::SOP1) << "op=" << op;
        EXPECT_EQ(plain.len_dwords, 1u) << "op=" << op;
        EXPECT_EQ(plain.opcode, op) << "op=" << op;
        const Rdna2Inst lit = decode(0xBE800000u | (4u << 16) | (op << 8) | kLiteralSrc, 0x0BADF00Du);
        EXPECT_EQ(lit.len_dwords, 2u) << "op=" << op;
        EXPECT_EQ(lit.literal, 0x0BADF00Du) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, SopcEveryOpcode) {
    // SOPC: prefix 0xBF00, OP[22:16], SSRC1[15:8], SSRC0[7:0].
    for (uint32_t op = 0; op <= 0x7F; ++op) {
        const Rdna2Inst plain = decode(0xBF000000u | (op << 16) | (2u << 8) | 1u);
        EXPECT_EQ(plain.fmt, Rdna2Format::SOPC) << "op=" << op;
        EXPECT_EQ(plain.len_dwords, 1u) << "op=" << op;
        const Rdna2Inst l0 = decode(0xBF000000u | (op << 16) | (2u << 8) | kLiteralSrc);
        EXPECT_EQ(l0.len_dwords, 2u) << "ssrc0 literal, op=" << op;
        const Rdna2Inst l1 = decode(0xBF000000u | (op << 16) | (kLiteralSrc << 8) | 1u);
        EXPECT_EQ(l1.len_dwords, 2u) << "ssrc1 literal, op=" << op;
    }
}

TEST(Rdna2DecodeSweep, SoppIsAlwaysOneDwordAndOnlyEndpgmTerminates) {
    // SOPP: prefix 0xBF80, OP[22:16], SIMM16[15:0]. Even a SIMM16 of 0xFF is not a literal marker.
    for (uint32_t op = 0; op <= 0x7F; ++op) {
        for (uint32_t simm : {0u, 0xFFu, 0xFFFFu}) {
            const uint32_t w = 0xBF800000u | (op << 16) | simm;
            const Rdna2Inst in = decode(w);
            EXPECT_EQ(in.fmt, Rdna2Format::SOPP) << "op=" << op;
            EXPECT_EQ(in.len_dwords, 1u) << "op=" << op << " simm=" << simm;
            EXPECT_EQ(in.is_end, w == kS_ENDPGM) << "op=" << op << " simm=" << simm;
        }
    }
}

TEST(Rdna2DecodeSweep, Vop1EveryOpcode) {
    // VOP1: prefix 0x7E, VDST[24:17], OP[16:9], SRC0[8:0].
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const Rdna2Inst plain = decode(0x7E000000u | (2u << 17) | (op << 9) | 0x101u);
        EXPECT_EQ(plain.fmt, Rdna2Format::VOP1) << "op=" << op;
        EXPECT_EQ(plain.len_dwords, 1u) << "op=" << op;
        EXPECT_EQ(plain.opcode, op) << "op=" << op;
        const Rdna2Inst lit = decode(0x7E000000u | (2u << 17) | (op << 9) | kLiteralSrc, 0x3F800000u);
        EXPECT_EQ(lit.len_dwords, 2u) << "op=" << op;
        EXPECT_EQ(lit.literal, 0x3F800000u) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, VopcEveryOpcode) {
    // VOPC: prefix 0x7C, OP[24:17], VSRC1[16:9], SRC0[8:0].
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const Rdna2Inst plain = decode(0x7C000000u | (op << 17) | (1u << 9) | 0x100u);
        EXPECT_EQ(plain.fmt, Rdna2Format::VOPC) << "op=" << op;
        EXPECT_EQ(plain.len_dwords, 1u) << "op=" << op;
        EXPECT_EQ(plain.opcode, op) << "op=" << op;
        const Rdna2Inst lit = decode(0x7C000000u | (op << 17) | (1u << 9) | kLiteralSrc, 0x40000000u);
        EXPECT_EQ(lit.len_dwords, 2u) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Vop2LengthHonoursTheSixMandatoryLiteralOpcodes) {
    // VOP2: bit31=0, OP[30:25] (0x00..0x3D; 0x3E/0x3F collide with the VOPC/VOP1 prefixes), VDST[24:17],
    // VSRC1[16:9], SRC0[8:0]. The six K-carrying mul-adds are 2 dwords whatever SRC0 is.
    const auto always_literal = [](uint32_t op) {
        return op == 0x20 || op == 0x21 || op == 0x2C || op == 0x2D || op == 0x37 || op == 0x38;
    };
    for (uint32_t op = 0; op <= 0x3D; ++op) {
        const uint32_t base = (op << 25) | (2u << 17) | (1u << 9);
        const Rdna2Inst vgpr = decode(base | 0x100u);
        EXPECT_EQ(vgpr.fmt, Rdna2Format::VOP2) << "op=" << op;
        EXPECT_EQ(vgpr.opcode, op) << "op=" << op;
        EXPECT_EQ(vgpr.len_dwords, always_literal(op) ? 2u : 1u) << "op=" << op;
        const Rdna2Inst lit = decode(base | kLiteralSrc);
        EXPECT_EQ(lit.len_dwords, 2u) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, VopSrc0ControlWordSelectorsAreTwoDwords) {
    // SDWA (0xF9), DPP16 (0xFA), DPP8 (0xE9) and DPP8-FI (0xEA) each place a control word after the
    // instruction. Missing one decodes that word as a phantom instruction.
    for (uint32_t src0 : {0xF9u, 0xFAu, 0xE9u, 0xEAu}) {
        const Rdna2Inst v1 = decode(0x7E000000u | (2u << 17) | (0x01u << 9) | src0, 0u);
        EXPECT_EQ(v1.fmt, Rdna2Format::VOP1) << "src0=" << src0;
        EXPECT_EQ(v1.len_dwords, 2u) << "src0=" << src0;
        EXPECT_TRUE(v1.has_modifier) << "src0=" << src0;
        const Rdna2Inst v2 = decode((0x03u << 25) | (2u << 17) | (1u << 9) | src0, 0u);
        EXPECT_EQ(v2.fmt, Rdna2Format::VOP2) << "src0=" << src0;
        EXPECT_EQ(v2.len_dwords, 2u) << "src0=" << src0;
    }
}

TEST(Rdna2DecodeSweep, Vop3EveryOpcodeAndLiteralSlot) {
    // VOP3 (prefix 0xD4/0xD5 -> top6 0x35): OP[25:16] in dword0; SRC0[8:0], SRC1[17:9], SRC2[26:18] in
    // dword1. A literal marker in ANY of the three slots adds one dword.
    for (uint32_t op = 0; op < 0x400; ++op) {
        const uint32_t w0 = 0xD4000000u | (op << 16);
        const Rdna2Inst plain = decode(w0, 0x100u | (0x101u << 9) | (0x102u << 18));
        // The e64 compare encoding occupies opcodes 0x000..0x0FF; the decoder classes it as VOPC.
        EXPECT_EQ(plain.fmt, op < 0x100u ? Rdna2Format::VOPC : Rdna2Format::VOP3) << "op=" << op;
        EXPECT_EQ(plain.len_dwords, 2u) << "op=" << op;
        for (uint32_t slot = 0; slot < 3; ++slot) {
            uint32_t d1 = 0x100u | (0x101u << 9) | (0x102u << 18);
            d1 = (d1 & ~(0x1FFu << (slot * 9))) | (kLiteralSrc << (slot * 9));
            const Rdna2Inst lit = decode(w0, d1);
            EXPECT_EQ(lit.len_dwords, 3u) << "op=" << op << " slot=" << slot;
            EXPECT_TRUE(lit.has_literal) << "op=" << op << " slot=" << slot;
        }
    }
}

TEST(Rdna2DecodeSweep, FixedLengthMemoryAndExportFormats) {
    // top6 -> format; all two dwords with no literal rule.
    struct Row { uint32_t top6; Rdna2Format fmt; };
    const Row rows[] = {
        {0x36, Rdna2Format::DS},   {0x37, Rdna2Format::FLAT},  {0x38, Rdna2Format::MUBUF},
        {0x3A, Rdna2Format::MTBUF}, {0x3D, Rdna2Format::SMEM}, {0x3E, Rdna2Format::EXP},
    };
    for (const Row& r : rows) {
        for (uint32_t low : {0x0000000u, 0x3FFFFFFu}) {  // all-clear and all-set payload bits
            const Rdna2Inst in = decode((r.top6 << 26) | low, 0xFFFFFFFFu);
            EXPECT_EQ(in.fmt, r.fmt) << "top6=" << r.top6;
            EXPECT_EQ(in.len_dwords, 2u) << "top6=" << r.top6 << " low=" << low;
            EXPECT_FALSE(in.has_literal) << "top6=" << r.top6;
        }
    }
}

TEST(Rdna2DecodeSweep, VintrpIsOneDword) {
    const Rdna2Inst in = decode(0xC8000000u);
    EXPECT_EQ(in.fmt, Rdna2Format::VINTRP);
    EXPECT_EQ(in.len_dwords, 1u);
}

TEST(Rdna2DecodeSweep, MimgLengthFollowsTheNsaField) {
    // MIMG (top6 0x3C): dword0[2:1] = NSA extra-dword count, so total length is 2 + NSA.
    for (uint32_t nsa = 0; nsa <= 3; ++nsa) {
        const uint32_t code[5] = {0xF0000000u | (nsa << 1), 1, 2, 3, 4};
        const Rdna2Inst in = rdna2_decode_one(code, 5);
        EXPECT_EQ(in.fmt, Rdna2Format::MIMG) << "nsa=" << nsa;
        EXPECT_EQ(in.len_dwords, 2u + nsa) << "nsa=" << nsa;
    }
}

TEST(Rdna2DecodeSweep, Vop3pHonoursTheLiteralRule) {
    const Rdna2Inst plain = decode(0xCC000000u, 0x100u | (0x101u << 9) | (0x102u << 18));
    EXPECT_EQ(plain.fmt, Rdna2Format::VOP3P);
    EXPECT_EQ(plain.len_dwords, 2u);
    const Rdna2Inst lit = decode(0xCC000000u, kLiteralSrc | (0x101u << 9) | (0x102u << 18));
    EXPECT_EQ(lit.len_dwords, 3u);
}

TEST(Rdna2DecodeSweep, ReservedPrefixesDecodeUnknownButStillAdvance) {
    // top6 values with no defined encoding must not be misclassified, and must clamp to one dword so
    // a walker terminates rather than looping.
    for (uint32_t top6 : {0x30u, 0x31u, 0x39u, 0x3Bu, 0x3Fu}) {
        const Rdna2Inst in = decode((top6 << 26) | 0x123456u);
        EXPECT_EQ(in.fmt, Rdna2Format::Unknown) << "top6=" << top6;
        EXPECT_EQ(in.len_dwords, 1u) << "top6=" << top6;
    }
}

TEST(Rdna2DecodeSweep, TruncatedStreamClampsLengthToWhatIsAvailable) {
    const uint32_t two[1] = {0xF4080002u};  // SMEM needs two dwords
    EXPECT_EQ(rdna2_decode_one(two, 1).len_dwords, 1u);
    const uint32_t lit[1] = {0xBE8203FFu};  // SOP1 with literal marker, literal missing
    EXPECT_EQ(rdna2_decode_one(lit, 1).len_dwords, 1u);
    EXPECT_EQ(rdna2_decode_one(lit, 0).len_dwords, 0u);
    EXPECT_EQ(rdna2_decode_one(lit, 0).fmt, Rdna2Format::Unknown);
}

TEST(Rdna2DecodeSweep, WalkerLandsExactlyOnEndpgmAcrossMixedLengths) {
    // One of each length class back to back; a single wrong length anywhere shifts every later pc.
    const uint32_t code[] = {
        0x80000000u | (4u << 16) | (2u << 8) | 1u,            // SOP2 (1)
        0xBE800000u | (4u << 16) | (3u << 8) | kLiteralSrc,   // SOP1 + literal (2)
        0x11111111u,                                          //   literal data
        0x7E000000u | (2u << 17) | (1u << 9) | 0x101u,        // VOP1 (1)
        (0x20u << 25) | (2u << 17) | (1u << 9) | 0x100u,      // v_madmk_f32 (2, mandatory K)
        0x22222222u,                                          //   K
        0xD4000000u,      0x100u | (0x101u << 9) | (kLiteralSrc << 18),  // VOP3 + literal (3)
        0x33333333u,                                          //   literal data
        0xF0000000u | (2u << 1), 0, 0, 0,                     // MIMG NSA=2 (4)
        kS_ENDPGM,
    };
    std::vector<Rdna2Inst> out;
    const size_t n = sizeof(code) / sizeof(code[0]);
    EXPECT_EQ(rdna2_walk(code, n, out), n);
    const uint32_t expect_pc[] = {0, 1, 3, 4, 6, 9, 13};
    ASSERT_EQ(out.size(), 7u);
    for (size_t k = 0; k < out.size(); ++k) EXPECT_EQ(out[k].pc, expect_pc[k]) << "inst " << k;
    EXPECT_TRUE(out.back().is_end);
}

// ---- VOPC cmp/cmpx pairing ------------------------------------------------------------------
// VOPC lays its 256 opcodes out as alternating 16-wide cmp / cmpx blocks. Each block pair covers one
// operand-type family; 0x40..0x7F and the two slots 0xAF / 0xBF are not compare opcodes at all.

TEST(Rdna2DecodeSweep, VopcCmpxIsExactlyTheSecondHalfOfEachValidBlockPair) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const bool invalid = (op >= 0x40 && op <= 0x7F) || op == 0xAF || op == 0xBF;
        const bool second_half_of_pair = (op & 0x10u) != 0;
        EXPECT_EQ(vopc_is_cmpx(op), !invalid && second_half_of_pair) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, EveryCmpxHasAValidCmpCounterpartSixteenBelow) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        if (!vopc_is_cmpx(op)) continue;
        const uint32_t base = op - 0x10u;
        EXPECT_FALSE(vopc_is_cmpx(base)) << "base op=" << base;
        const bool base_invalid = (base >= 0x40 && base <= 0x7F) || base == 0xAF || base == 0xBF;
        EXPECT_FALSE(base_invalid) << "cmpx op=" << op << " pairs with an invalid base";
    }
}

TEST(Rdna2DecodeSweep, OnlyCmpxVopcMayChangeExecAmongPlainCompares) {
    // v_cmp_* writes a VCC/SGPR mask; v_cmpx_* writes EXEC. The decoder-level predicate must agree.
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const Rdna2Inst in = decode(0x7C000000u | (op << 17) | (1u << 9) | 0x100u);
        EXPECT_EQ(rdna2_instruction_may_change_exec(in), vopc_is_cmpx(op)) << "op=" << op;
    }
}

// ---- Source operand field decode ------------------------------------------------------------

TEST(Rdna2DecodeSweep, SourceFieldClassesFollowTheIsaOperandTable) {
    for (uint32_t f = 0; f <= 511; ++f) {
        const Operand o = decode_src_field(f);
        if (f <= 105) {
            EXPECT_EQ(o.kind, OperandKind::SGPR) << f;
            EXPECT_EQ(o.value, static_cast<int32_t>(f)) << f;
        } else if (f == 128) {
            EXPECT_EQ(o.kind, OperandKind::InlineInt) << f;
            EXPECT_EQ(o.value, 0) << f;
        } else if (f >= 129 && f <= 192) {
            EXPECT_EQ(o.kind, OperandKind::InlineInt) << f;
            EXPECT_EQ(o.value, static_cast<int32_t>(f) - 128) << f;   // +1 .. +64
        } else if (f >= 193 && f <= 208) {
            EXPECT_EQ(o.kind, OperandKind::InlineInt) << f;
            EXPECT_EQ(o.value, -static_cast<int32_t>(f - 192)) << f;  // -1 .. -16
        } else if (f >= 240 && f <= 248) {
            EXPECT_EQ(o.kind, OperandKind::InlineFloat) << f;
            EXPECT_EQ(o.value, static_cast<int32_t>(f)) << f;
        } else if (f == 255) {
            EXPECT_EQ(o.kind, OperandKind::Literal) << f;
        } else if (f >= 256) {
            EXPECT_EQ(o.kind, OperandKind::VGPR) << f;
            EXPECT_EQ(o.value, static_cast<int32_t>(f) - 256) << f;
        } else {
            // VCC, TTMP, M0, EXEC, null, DPP/SDWA markers, SCC, LDS_DIRECT and unused codes.
            EXPECT_EQ(o.kind, OperandKind::Special) << f;
            EXPECT_EQ(o.value, static_cast<int32_t>(f)) << f;
        }
    }
}

TEST(Rdna2DecodeSweep, ExecAndVccFieldsAreSpecialNotSgpr) {
    // 106/107 VCC_LO/HI, 124 M0, 126/127 EXEC_LO/HI must never be mistaken for ordinary SGPRs, or
    // mask-write tracking misses them.
    for (uint32_t f : {106u, 107u, 124u, 125u, 126u, 127u, 253u, 254u}) {
        EXPECT_EQ(decode_src_field(f).kind, OperandKind::Special) << f;
        EXPECT_EQ(decode_src_field(f).value, static_cast<int32_t>(f)) << f;
    }
}

TEST(Rdna2DecodeSweep, InlineFloatConstantsMatchTheIsaTable) {
    EXPECT_EQ(inline_float_value(240), 0.5f);
    EXPECT_EQ(inline_float_value(241), -0.5f);
    EXPECT_EQ(inline_float_value(242), 1.0f);
    EXPECT_EQ(inline_float_value(243), -1.0f);
    EXPECT_EQ(inline_float_value(244), 2.0f);
    EXPECT_EQ(inline_float_value(245), -2.0f);
    EXPECT_EQ(inline_float_value(246), 4.0f);
    EXPECT_EQ(inline_float_value(247), -4.0f);
    EXPECT_NEAR(inline_float_value(248), 0.15915494f, 1e-8f);   // 1 / (2 * pi)
    for (uint32_t c : {0u, 128u, 239u, 249u, 255u, 256u}) {
        EXPECT_EQ(inline_float_value(c), 0.0f) << "non-float code " << c;
    }
}

// ---- Operand fields reach the decoded instruction -------------------------------------------

TEST(Rdna2DecodeSweep, Vop1OperandsDecodeDstAndSrc0) {
    // v_mov_b32 (op 1): VDST[24:17], SRC0[8:0].
    const Rdna2Inst v = decode(0x7E000000u | (7u << 17) | (1u << 9) | (256u + 5u));
    EXPECT_EQ(v.dst.kind, OperandKind::VGPR);
    EXPECT_EQ(v.dst.value, 7);
    ASSERT_GE(v.n_src, 1);
    EXPECT_EQ(v.src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(v.src[0].value, 5);

    const Rdna2Inst s = decode(0x7E000000u | (7u << 17) | (1u << 9) | 9u);
    EXPECT_EQ(s.src[0].kind, OperandKind::SGPR);
    EXPECT_EQ(s.src[0].value, 9);

    const Rdna2Inst c = decode(0x7E000000u | (7u << 17) | (1u << 9) | 0xC1u);   // inline -1
    EXPECT_EQ(c.src[0].kind, OperandKind::InlineInt);
    EXPECT_EQ(c.src[0].value, -1);
}

TEST(Rdna2DecodeSweep, Vop2OperandsDecodeAllThreeFields) {
    // v_add_f32 (op 3): VDST[24:17], VSRC1[16:9] (a VGPR), SRC0[8:0].
    const Rdna2Inst in = decode((0x03u << 25) | (4u << 17) | (6u << 9) | (256u + 2u));
    EXPECT_EQ(in.dst.kind, OperandKind::VGPR);
    EXPECT_EQ(in.dst.value, 4);
    ASSERT_GE(in.n_src, 2);
    EXPECT_EQ(in.src[0].value, 2);
    EXPECT_EQ(in.src[1].kind, OperandKind::VGPR);
    EXPECT_EQ(in.src[1].value, 6);
}

TEST(Rdna2DecodeSweep, Sop2OperandsDecodeSdstAndBothSources) {
    // s_add_u32 (op 0): SDST[22:16], SSRC1[15:8], SSRC0[7:0].
    const Rdna2Inst in = decode(0x80000000u | (10u << 16) | (3u << 8) | 2u);
    EXPECT_EQ(in.dst.kind, OperandKind::SGPR);
    EXPECT_EQ(in.dst.value, 10);
    ASSERT_GE(in.n_src, 2);
    EXPECT_EQ(in.src[0].value, 2);
    EXPECT_EQ(in.src[1].value, 3);
}

TEST(Rdna2DecodeSweep, SoppSimm16IsSignExtended) {
    // s_branch (op 2): SIMM16[15:0] is a signed dword offset; backward branches are negative.
    EXPECT_EQ(decode(0xBF820000u | 0x0005u).simm16, 5);
    EXPECT_EQ(decode(0xBF820000u | 0xFFFEu).simm16, -2);
    EXPECT_EQ(decode(0xBF820000u | 0x8000u).simm16, -32768);
}

// ---- VOP3 operand decode ---------------------------------------------------------------------
// VOP3A layout: dword0 = VDST[7:0], ABS[10:8], OPSEL[14:11], CLAMP[15], OP[25:16];
//               dword1 = SRC0[8:0], SRC1[17:9], SRC2[26:18], OMOD[28:27], NEG[31:29].
// 0x14B is v_fma_f32 (three real sources, no 16-bit select, not VOP3B).

namespace {
constexpr uint32_t kVop3Fma = 0x14Bu;
constexpr uint32_t vop3_w0(uint32_t op, uint32_t vdst = 0u, uint32_t mid = 0u) {
    return 0xD4000000u | (op << 16) | mid | vdst;
}
constexpr uint32_t vop3_srcs(uint32_t s0, uint32_t s1, uint32_t s2) {
    return s0 | (s1 << 9) | (s2 << 18);
}
}  // namespace

TEST(Rdna2DecodeSweep, Vop3DecodesDestinationAndThreeSources) {
    const Rdna2Inst in = decode(vop3_w0(kVop3Fma, 9u), vop3_srcs(256u + 1u, 5u, 0xF2u));
    ASSERT_EQ(in.fmt, Rdna2Format::VOP3);
    EXPECT_EQ(in.opcode, kVop3Fma);
    EXPECT_EQ(in.dst.kind, OperandKind::VGPR);
    EXPECT_EQ(in.dst.value, 9);
    ASSERT_EQ(in.n_src, 3);
    EXPECT_EQ(in.src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(in.src[0].value, 1);
    EXPECT_EQ(in.src[1].kind, OperandKind::SGPR);
    EXPECT_EQ(in.src[1].value, 5);
    EXPECT_EQ(in.src[2].kind, OperandKind::InlineFloat);
    EXPECT_EQ(in.src[2].value, 0xF2);   // 1.0
}

TEST(Rdna2DecodeSweep, Vop3AbsAndNegAreIndependentPerSource) {
    // Every one of the 64 abs x neg combinations must land on exactly its own sources.
    for (uint32_t abs = 0; abs < 8; ++abs) {
        for (uint32_t neg = 0; neg < 8; ++neg) {
            const Rdna2Inst in = decode(vop3_w0(kVop3Fma, 0u, abs << 8),
                                        vop3_srcs(256u, 257u, 258u) | (neg << 29));
            for (uint32_t k = 0; k < 3; ++k) {
                EXPECT_EQ(in.src_abs[k], ((abs >> k) & 1u) != 0) << "abs=" << abs << " k=" << k;
                EXPECT_EQ(in.src_neg[k], ((neg >> k) & 1u) != 0) << "neg=" << neg << " k=" << k;
            }
        }
    }
}

TEST(Rdna2DecodeSweep, Vop3ClampAndOmodFields) {
    EXPECT_FALSE(decode(vop3_w0(kVop3Fma), vop3_srcs(256, 257, 258)).clamp);
    EXPECT_TRUE(decode(vop3_w0(kVop3Fma, 0, 1u << 15), vop3_srcs(256, 257, 258)).clamp);
    for (uint32_t omod = 0; omod < 4; ++omod) {
        const Rdna2Inst in = decode(vop3_w0(kVop3Fma), vop3_srcs(256, 257, 258) | (omod << 27));
        EXPECT_EQ(in.omod, omod) << "omod=" << omod;
    }
}

TEST(Rdna2DecodeSweep, Vop3bFamilyReadsSdstAndClearsAbs) {
    // For VOP3B, dword0[14:8] is a scalar carry/flag destination, not three abs bits. CLAMP (bit 15)
    // stays meaningful and NEG stays in dword1.
    for (uint32_t op : {0x128u, 0x129u, 0x12Au, 0x16Du, 0x16Eu, 0x176u, 0x177u, 0x30Fu, 0x310u, 0x319u}) {
        const Rdna2Inst in = decode(vop3_w0(op, 3u, 42u << 8 | (1u << 15)),
                                    vop3_srcs(256, 257, 258) | (1u << 29));
        EXPECT_EQ(in.sdst.kind, OperandKind::SGPR) << "op=" << op;
        EXPECT_EQ(in.sdst.value, 42) << "op=" << op;
        for (uint32_t k = 0; k < 3; ++k) EXPECT_FALSE(in.src_abs[k]) << "op=" << op << " k=" << k;
        EXPECT_TRUE(in.clamp) << "op=" << op;
        EXPECT_TRUE(in.src_neg[0]) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Vop3aOpcodesOutsideTheVop3bListKeepAbsAndHaveNoSdst) {
    const Rdna2Inst in = decode(vop3_w0(kVop3Fma, 0u, 0x7u << 8), vop3_srcs(256, 257, 258));
    EXPECT_EQ(in.sdst.kind, OperandKind::None);
    for (uint32_t k = 0; k < 3; ++k) EXPECT_TRUE(in.src_abs[k]) << k;
}

TEST(Rdna2DecodeSweep, Vop3TwoSourceOpcodesDropTheReservedThirdSource) {
    // v_mul_lo_u32 (0x169) has two data sources. SRC2 is reserved and commonly reads as s0; exposing
    // it would invent a scalar dependency, so the decoder must report two sources.
    const Rdna2Inst two = decode(vop3_w0(kVop3OpcodeMulLoU32), vop3_srcs(256, 257, 0));
    EXPECT_EQ(two.n_src, 2);
    EXPECT_EQ(two.src[2].kind, OperandKind::None);
    const Rdna2Inst three = decode(vop3_w0(kVop3Fma), vop3_srcs(256, 257, 0));
    EXPECT_EQ(three.n_src, 3);
    EXPECT_EQ(three.src[2].kind, OperandKind::SGPR);
}

TEST(Rdna2DecodeSweep, Vop3OpselIsCapturedOnlyForTheSixteenBitFamily) {
    // OPSEL[2:0] pick each source half, OPSEL[3] the destination half. Only the 16-bit integer / f16
    // scalar family honours it; elsewhere the bits must not leak into the selector.
    for (uint32_t op : {0x311u, 0x34Bu, 0x351u, 0x352u, 0x353u, 0x354u, 0x355u, 0x356u, 0x357u,
                        0x358u, 0x359u, 0x303u, 0x30Eu, 0x314u, 0x340u, 0x35Eu}) {
        for (uint32_t sel = 0; sel < 16; ++sel) {
            const Rdna2Inst in = decode(vop3_w0(op, 0u, sel << 11), vop3_srcs(256, 257, 258));
            EXPECT_EQ(in.vop3p_opsel, sel) << "op=" << op << " sel=" << sel;
        }
    }
    // 0x306 sits inside the 0x303..0x30E window but is not an instruction; it is excluded.
    EXPECT_EQ(decode(vop3_w0(0x306u, 0u, 0xFu << 11), vop3_srcs(256, 257, 258)).vop3p_opsel, 0);
    EXPECT_EQ(decode(vop3_w0(kVop3Fma, 0u, 0xFu << 11), vop3_srcs(256, 257, 258)).vop3p_opsel, 0);
}

TEST(Rdna2DecodeSweep, PermlaneOverloadsOpselAsFetchInactiveAndBoundCtrl) {
    for (uint32_t op : {0x377u, 0x378u}) {
        for (uint32_t bits = 0; bits < 4; ++bits) {
            const Rdna2Inst in = decode(vop3_w0(op, 0u, bits << 11), vop3_srcs(256, 257, 258));
            EXPECT_EQ(in.permlane_fetch_inactive, (bits & 1u) != 0) << "op=" << op << " bits=" << bits;
            EXPECT_EQ(in.permlane_bound_ctrl, (bits & 2u) != 0) << "op=" << op << " bits=" << bits;
        }
    }
    const Rdna2Inst other = decode(vop3_w0(kVop3Fma, 0u, 3u << 11), vop3_srcs(256, 257, 258));
    EXPECT_FALSE(other.permlane_fetch_inactive);
    EXPECT_FALSE(other.permlane_bound_ctrl);
}

TEST(Rdna2DecodeSweep, Vop3EncodedCompareUsesAScalarMaskDestination) {
    // The e64 compare encoding (opcodes 0x00..0xFF): dword0[6:0] is an SGPR mask destination, not
    // VDST, and only two data sources exist.
    const Rdna2Inst in = decode(vop3_w0(0xC4u /*v_cmp_*_u32*/, 0x2Au), vop3_srcs(256, 257, 258));
    ASSERT_EQ(in.fmt, Rdna2Format::VOPC);
    EXPECT_EQ(in.dst.kind, OperandKind::SGPR);
    EXPECT_EQ(in.dst.value, 0x2A);
    EXPECT_EQ(in.n_src, 2);
    EXPECT_EQ(in.src[2].kind, OperandKind::None);
}

TEST(Rdna2DecodeSweep, Vop3LiteralOperandReportsItsValue) {
    const uint32_t code[3] = {vop3_w0(kVop3Fma), vop3_srcs(256, kLiteralSrc, 258), 0x40490FDBu};
    const Rdna2Inst in = rdna2_decode_one(code, 3);
    ASSERT_EQ(in.len_dwords, 3u);
    EXPECT_TRUE(in.has_literal);
    EXPECT_EQ(in.literal, 0x40490FDBu);
    EXPECT_EQ(in.src[1].kind, OperandKind::Literal);
}

// ---- Memory / export / interpolation format fields -------------------------------------------
// Each field is exercised across its full range so a shifted or mis-masked field cannot hide behind
// one lucky value. Layouts follow the RDNA2 ISA reference encoding tables.

namespace {
constexpr uint32_t kTop6Ds = 0x36u << 26, kTop6Flat = 0x37u << 26, kTop6Mubuf = 0x38u << 26,
                   kTop6Mtbuf = 0x3Au << 26, kTop6Smem = 0x3Du << 26, kTop6Exp = 0x3Eu << 26,
                   kTop6Vintrp = 0x32u << 26;
}  // namespace

TEST(Rdna2DecodeSweep, ExpDecodesTargetEnableComprAndFourVgprs) {
    for (uint32_t target = 0; target < 64; ++target) {
        const Rdna2Inst in = decode(kTop6Exp | (target << 4) | 0xAu, 0x04030201u);
        ASSERT_EQ(in.fmt, Rdna2Format::EXP);
        EXPECT_EQ(in.exp_target, target);
        EXPECT_EQ(in.exp_en, 0xAu);
        EXPECT_FALSE(in.exp_compr);
    }
    for (uint32_t en = 0; en < 16; ++en) EXPECT_EQ(decode(kTop6Exp | en).exp_en, en);
    EXPECT_TRUE(decode(kTop6Exp | (1u << 10)).exp_compr);
    const Rdna2Inst in = decode(kTop6Exp, 0x04030201u);
    ASSERT_EQ(in.n_src, 4);
    for (int k = 0; k < 4; ++k) {
        EXPECT_EQ(in.src[k].kind, OperandKind::VGPR) << k;
        EXPECT_EQ(in.src[k].value, k + 1) << k;
    }
}

TEST(Rdna2DecodeSweep, SmemDecodesOpcodeBaseDestinationAndSignedOffset) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const Rdna2Inst in = decode(kTop6Smem | (op << 18) | (12u << 6) | 3u, 0u);
        ASSERT_EQ(in.fmt, Rdna2Format::SMEM);
        EXPECT_EQ(in.opcode, op);
        EXPECT_EQ(in.dst.value, 12);
        EXPECT_EQ(in.src[0].value, 6) << "SBASE field is a pair index, so 3 -> s6";
    }
    // OFFSET is a signed 21-bit byte immediate, sign-extended into `literal`.
    EXPECT_EQ(decode(kTop6Smem, 0x000010u).literal, 0x10u);
    EXPECT_EQ(decode(kTop6Smem, 0x0FFFFFu).literal, 0x0FFFFFu);
    EXPECT_EQ(static_cast<int32_t>(decode(kTop6Smem, 0x100000u).literal), -0x100000);
    EXPECT_EQ(static_cast<int32_t>(decode(kTop6Smem, 0x1FFFFFu).literal), -1);
    // SOFFSET lives in d1[31:25]; 125 is NULL (immediate-only).
    EXPECT_EQ(decode(kTop6Smem, 125u << 25).src[1].value, 125);
    EXPECT_EQ(decode(kTop6Smem, 7u << 25).src[1].kind, OperandKind::SGPR);
}

TEST(Rdna2DecodeSweep, MubufDecodesOpcodeFlagsAndAddressFields) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        EXPECT_EQ(decode(kTop6Mubuf | (op << 18), 0u).opcode, op) << "8-bit opcode, op=" << op;
    }
    struct Flag { uint32_t bit; bool Rdna2Inst::*field; const char* name; };
    const Flag flags[] = {{14, &Rdna2Inst::mubuf_glc, "glc"}, {15, &Rdna2Inst::mubuf_dlc, "dlc"},
                          {16, &Rdna2Inst::mubuf_lds, "lds"}};
    for (const Flag& f : flags) {
        EXPECT_TRUE(decode(kTop6Mubuf | (1u << f.bit)).*f.field) << f.name;
        EXPECT_FALSE(decode(kTop6Mubuf).*f.field) << f.name;
        for (const Flag& other : flags) {
            if (&other != &f) {
                EXPECT_FALSE(decode(kTop6Mubuf | (1u << f.bit)).*other.field)
                    << f.name << " leaked into " << other.name;
            }
        }
    }
    EXPECT_TRUE(decode(kTop6Mubuf, 1u << 23).mubuf_tfe);
    EXPECT_FALSE(decode(kTop6Mubuf, 0u).mubuf_tfe);

    const Rdna2Inst in = decode(kTop6Mubuf, 0x03u | (0x05u << 8) | (6u << 16) | (0x80u << 24));
    EXPECT_EQ(in.src[0].value, 3);        // VADDR
    EXPECT_EQ(in.dst.value, 5);           // VDATA
    EXPECT_EQ(in.src[1].value, 24);       // SRSRC is a x4 SGPR base
    EXPECT_EQ(in.src[2].kind, OperandKind::InlineInt);   // SOFFSET 0x80 = inline 0, not s0
    EXPECT_EQ(in.src[2].value, 0);
}

TEST(Rdna2DecodeSweep, MubufOffsetOffenAndIdxenPackIntoLiteral) {
    for (uint32_t off : {0u, 1u, 0x7FFu, 0xFFFu}) {
        EXPECT_EQ(decode(kTop6Mubuf | off).literal, off) << off;
    }
    EXPECT_EQ(decode(kTop6Mubuf | (1u << 12)).literal, 1u << 12);   // OFFEN
    EXPECT_EQ(decode(kTop6Mubuf | (1u << 13)).literal, 1u << 13);   // IDXEN
    EXPECT_EQ(decode(kTop6Mubuf | 0x3FFFu).literal, 0x3FFFu);
    // glc/dlc/lds sit above the packed bits and must not leak into them.
    EXPECT_EQ(decode(kTop6Mubuf | (7u << 14)).literal, 0u);
}

TEST(Rdna2DecodeSweep, MtbufOpcodeSplitsAcrossDword0AndDword1) {
    // gfx10 keeps OP[2:0] in dword0[18:16] and OP[3] in dword1 bit 21; opcodes 8..15 are packed-D16.
    for (uint32_t op = 0; op < 16; ++op) {
        const Rdna2Inst in = decode(kTop6Mtbuf | ((op & 7u) << 16), (op >> 3) << 21);
        ASSERT_EQ(in.fmt, Rdna2Format::MTBUF);
        EXPECT_EQ(in.opcode, op) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, MtbufCombinedFormatIsSevenBits) {
    for (uint32_t fmt = 0; fmt < 128; ++fmt) {
        EXPECT_EQ(decode(kTop6Mtbuf | (fmt << 19)).mtbuf_format, fmt) << "fmt=" << fmt;
    }
    // 32_FLOAT is combined format 22 on gfx10; the older DFMT/NFMT split would read it wrongly.
    EXPECT_EQ(decode(kTop6Mtbuf | (22u << 19)).mtbuf_format, 22u);
}

TEST(Rdna2DecodeSweep, MtbufFlagsTfeAndOperands) {
    EXPECT_TRUE(decode(kTop6Mtbuf | (1u << 14)).mubuf_glc);
    EXPECT_TRUE(decode(kTop6Mtbuf | (1u << 15)).mubuf_dlc);
    EXPECT_TRUE(decode(kTop6Mtbuf, 1u << 23).mtbuf_tfe);
    EXPECT_FALSE(decode(kTop6Mtbuf, 0u).mtbuf_tfe);
    const Rdna2Inst in =
        decode(kTop6Mtbuf | 0x2345u, 0x01u | (0x02u << 8) | (3u << 16) | (0x84u << 24));
    EXPECT_EQ(in.src[0].value, 1);
    EXPECT_EQ(in.dst.value, 2);
    EXPECT_EQ(in.src[1].value, 12);
    EXPECT_EQ(in.src[2].kind, OperandKind::InlineInt);
    EXPECT_EQ(in.src[2].value, 4);
    EXPECT_EQ(in.literal, 0x2345u);
}

TEST(Rdna2DecodeSweep, FlatSegmentFlagsAndSignedOffset) {
    for (uint32_t seg = 0; seg < 4; ++seg) {
        EXPECT_EQ(decode(kTop6Flat | (seg << 14)).flat_segment, seg) << seg;
    }
    EXPECT_TRUE(decode(kTop6Flat | (1u << 16)).flat_glc);
    EXPECT_TRUE(decode(kTop6Flat | (1u << 17)).flat_slc);
    EXPECT_TRUE(decode(kTop6Flat | (1u << 12)).flat_dlc);
    EXPECT_TRUE(decode(kTop6Flat | (1u << 13)).flat_lds);
    const Rdna2Inst none = decode(kTop6Flat);
    EXPECT_FALSE(none.flat_glc);
    EXPECT_FALSE(none.flat_slc);
    EXPECT_FALSE(none.flat_dlc);
    EXPECT_FALSE(none.flat_lds);
    // OFFSET[11:0] is signed (the gfx10 immediate range is -2048..2047).
    EXPECT_EQ(decode(kTop6Flat | 0x7FFu).literal, 0x7FFu);
    EXPECT_EQ(static_cast<int32_t>(decode(kTop6Flat | 0x800u).literal), -2048);
    EXPECT_EQ(static_cast<int32_t>(decode(kTop6Flat | 0xFFFu).literal), -1);
}

TEST(Rdna2DecodeSweep, FlatStoresTakeVdataAndLoadsTakeVdst) {
    // d1: VADDR[7:0], VDATA[15:8], SADDR[22:16], VDST[31:24].
    const uint32_t d1 = 0x01u | (0x02u << 8) | (125u << 16) | (0x03u << 24);
    for (uint32_t op = 0; op < 0x80; ++op) {
        const Rdna2Inst in = decode(kTop6Flat | (op << 18), d1);
        const bool store = op >= 0x18 && op <= 0x1F;
        EXPECT_EQ(in.opcode, op);
        EXPECT_EQ(in.dst.value, store ? 2 : 3) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, FlatScratchOffFormDropsTheVaddrOperand) {
    // Scratch segment (1) with an SGPR SADDR and VADDR 0 is the canonical `off, sN` form.
    const Rdna2Inst off = decode(kTop6Flat | (1u << 14), 0u | (8u << 16));
    EXPECT_EQ(off.src[0].kind, OperandKind::None);
    EXPECT_EQ(off.src[1].kind, OperandKind::SGPR);
    EXPECT_EQ(off.src[1].value, 8);
    // SADDR=NULL (125) is the `vN, off` form and keeps VADDR.
    const Rdna2Inst vaddr = decode(kTop6Flat | (1u << 14), 4u | (125u << 16));
    EXPECT_EQ(vaddr.src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(vaddr.src[0].value, 4);
    // Global (2) always keeps VADDR.
    EXPECT_EQ(decode(kTop6Flat | (2u << 14), 0u | (8u << 16)).src[0].kind, OperandKind::VGPR);
}

TEST(Rdna2DecodeSweep, DsDecodesOpcodeOffsetAndFourVgprFields) {
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        EXPECT_EQ(decode(kTop6Ds | (op << 18)).opcode, op) << op;
    }
    EXPECT_EQ(decode(kTop6Ds | 0x1234u).literal, 0x1234u);
    EXPECT_EQ(decode(kTop6Ds | 0xFFFFu).literal, 0xFFFFu);
    const Rdna2Inst in = decode(kTop6Ds, 0x01u | (0x02u << 8) | (0x03u << 16) | (0x04u << 24));
    EXPECT_EQ(in.src[0].value, 1);   // ADDR
    EXPECT_EQ(in.src[1].value, 2);   // DATA0
    EXPECT_EQ(in.src[2].value, 3);   // DATA1
    EXPECT_EQ(in.dst.value, 4);      // VDST
}

TEST(Rdna2DecodeSweep, DsGdsFlagCapturesBothCandidateBits) {
    // llvm-mc places GDS at bit 17; bit 16 is captured too so an unknown flag is rejected visibly
    // rather than silently running a device-global op against workgroup LDS.
    EXPECT_FALSE(decode(kTop6Ds).ds_gds);
    EXPECT_TRUE(decode(kTop6Ds | (1u << 17)).ds_gds);
    EXPECT_TRUE(decode(kTop6Ds | (1u << 16)).ds_gds);
    EXPECT_TRUE(decode(kTop6Ds | (3u << 16)).ds_gds);
}

TEST(Rdna2DecodeSweep, VintrpDecodesOpcodeDestinationAttributeAndChannel) {
    for (uint32_t op = 0; op < 4; ++op) {
        EXPECT_EQ(decode(kTop6Vintrp | (op << 16)).opcode, op) << op;
    }
    for (uint32_t attr = 0; attr < 64; ++attr) {
        EXPECT_EQ(decode(kTop6Vintrp | (attr << 10)).vintrp_attr, attr) << attr;
    }
    for (uint32_t chan = 0; chan < 4; ++chan) {
        EXPECT_EQ(decode(kTop6Vintrp | (chan << 8)).vintrp_chan, chan) << chan;
    }
    const Rdna2Inst in = decode(kTop6Vintrp | (9u << 18) | 0x7u);
    EXPECT_EQ(in.dst.value, 9);
    EXPECT_EQ(in.src[0].value, 7);
    EXPECT_EQ(in.n_src, 1);
}

// ---- MIMG control fields and mip-operand shape helpers ---------------------------------------
// dword0: OP[0] at bit 0, NSA[2:1], DIM[5:3], DLC[7], DMASK[11:8], UNORM[12], GLC[13], R128[15],
//         TFE[16], LWE[17], OP[7:1] at [24:18], SLC[25]; bits 6 and 14 are reserved.
// dword1: VADDR[7:0], VDATA[15:8], SRSRC[20:16] (x4), SSAMP[25:21] (x4), reserved [29:26],
//         A16[30], D16[31].

namespace {
constexpr uint32_t kMimg = 0xF0000000u;
constexpr uint32_t mimg_op(uint32_t op) { return ((op >> 7) & 1u) | ((op & 0x7Fu) << 18); }
constexpr uint32_t kMimgLoadMip = 0x01u;
constexpr uint32_t kMimgStoreMip = 0x09u;
}  // namespace

TEST(Rdna2DecodeSweep, MimgOpcodeUsesBitZeroAsItsMostSignificantBit) {
    // Dropping dword0 bit 0 would alias IMAGE_MSAA_LOAD (128) onto IMAGE_LOAD (0): every opcode in
    // 0..255 must round-trip, not just the low half.
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const uint32_t w = kMimg | ((op >> 7) & 1u) | (((op & 0x7Fu)) << 18);
        const Rdna2Inst in = decode(w);
        ASSERT_EQ(in.fmt, Rdna2Format::MIMG);
        EXPECT_EQ(in.opcode, op) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, MimgDmaskDimAndNsaFieldsRoundTrip) {
    for (uint32_t dmask = 0; dmask < 16; ++dmask) {
        EXPECT_EQ(decode(kMimg | (dmask << 8)).mimg_dmask, dmask) << dmask;
    }
    for (uint32_t dim = 0; dim < 8; ++dim) {
        EXPECT_EQ(decode(kMimg | (dim << 3)).mimg_dim, dim) << dim;
    }
    for (uint32_t nsa = 0; nsa < 4; ++nsa) {
        EXPECT_EQ(decode(kMimg | (nsa << 1)).mimg_nsa, nsa) << nsa;
    }
}

TEST(Rdna2DecodeSweep, MimgEachControlBitMapsToExactlyItsOwnField) {
    struct Bit { uint32_t dword; uint32_t bit; bool Rdna2Inst::*field; const char* name; };
    const Bit bits[] = {
        {0, 7, &Rdna2Inst::mimg_dlc, "dlc"},    {0, 12, &Rdna2Inst::mimg_unorm, "unorm"},
        {0, 13, &Rdna2Inst::mimg_glc, "glc"},   {0, 15, &Rdna2Inst::mimg_r128, "r128"},
        {0, 16, &Rdna2Inst::mimg_tfe, "tfe"},   {0, 17, &Rdna2Inst::mimg_lwe, "lwe"},
        {0, 25, &Rdna2Inst::mimg_slc, "slc"},   {1, 30, &Rdna2Inst::mimg_a16, "a16"},
        {1, 31, &Rdna2Inst::mimg_d16, "d16"},
    };
    for (const Bit& set : bits) {
        const Rdna2Inst in = set.dword == 0 ? decode(kMimg | (1u << set.bit), 0u)
                                             : decode(kMimg, 1u << set.bit);
        for (const Bit& probe : bits) {
            const bool expected = &probe == &set;
            EXPECT_EQ(static_cast<bool>(in.*probe.field), expected)
                << "setting " << set.name << " read back through " << probe.name;
        }
        EXPECT_FALSE(in.mimg_reserved) << set.name << " is a defined control, not a reserved bit";
    }
    const Rdna2Inst clear = decode(kMimg, 0u);
    for (const Bit& b : bits) EXPECT_FALSE(static_cast<bool>(clear.*b.field)) << b.name;
}

TEST(Rdna2DecodeSweep, MimgReservedHolesAreFlaggedAndOnlyThose) {
    // dword0 bits 6 and 14, dword1 bits 26..29 are reserved: an unsupported raw packet must not be
    // mistaken for the ordinary form.
    EXPECT_FALSE(decode(kMimg, 0u).mimg_reserved);
    EXPECT_TRUE(decode(kMimg | (1u << 6), 0u).mimg_reserved);
    EXPECT_TRUE(decode(kMimg | (1u << 14), 0u).mimg_reserved);
    for (uint32_t bit = 26; bit <= 29; ++bit) {
        EXPECT_TRUE(decode(kMimg, 1u << bit).mimg_reserved) << "d1 bit " << bit;
    }
    // Neighbours of the holes are defined fields, not reserved.
    for (uint32_t bit : {5u, 7u, 13u, 15u}) {
        EXPECT_FALSE(decode(kMimg | (1u << bit), 0u).mimg_reserved) << "d0 bit " << bit;
    }
    for (uint32_t bit : {25u, 30u, 31u}) {
        EXPECT_FALSE(decode(kMimg, 1u << bit).mimg_reserved) << "d1 bit " << bit;
    }
}

TEST(Rdna2DecodeSweep, MimgDataAddressAndDescriptorRegistersScaleCorrectly) {
    // SRSRC and SSAMP are x4 SGPR bases: a T# is 8 SGPRs and an S# is 4, both 4-aligned.
    for (uint32_t srsrc = 0; srsrc < 32; ++srsrc) {
        for (uint32_t ssamp : {0u, 5u, 31u}) {
            const Rdna2Inst in = decode(kMimg, 0x07u | (0x09u << 8) | (srsrc << 16) | (ssamp << 21));
            EXPECT_EQ(in.src[0].kind, OperandKind::VGPR);
            EXPECT_EQ(in.src[0].value, 7);
            EXPECT_EQ(in.dst.value, 9);
            EXPECT_EQ(in.src[1].kind, OperandKind::SGPR);
            EXPECT_EQ(in.src[1].value, static_cast<int32_t>(srsrc * 4)) << srsrc;
            EXPECT_EQ(in.src[2].value, static_cast<int32_t>(ssamp * 4)) << ssamp;
            EXPECT_EQ(in.n_src, 3);
        }
    }
}

TEST(Rdna2DecodeSweep, MimgNsaExtraDwordsAreKeptInOrder) {
    const uint32_t code[5] = {kMimg | (3u << 1), 0x11u, 0xA1A2A3A4u, 0xB1B2B3B4u, 0xC1C2C3C4u};
    const Rdna2Inst in = rdna2_decode_one(code, 5);
    EXPECT_EQ(in.len_dwords, 5u);
    EXPECT_EQ(in.words[2], 0xA1A2A3A4u);
    EXPECT_EQ(in.words[3], 0xB1B2B3B4u);
    EXPECT_EQ(in.words[4], 0xC1C2C3C4u);
    // A stream too short for the declared NSA dwords must not read past its end.
    const Rdna2Inst cut = rdna2_decode_one(code, 3);
    EXPECT_EQ(cut.words[2], 0xA1A2A3A4u);
    EXPECT_EQ(cut.words[3], 0u);
    EXPECT_EQ(cut.words[4], 0u);
}

// ---- mip-operand shape helpers ---------------------------------------------------------------

namespace {
Rdna2Inst mimg_load_mip(uint32_t dim, uint32_t vaddr, uint32_t dmask = 0xFu, uint32_t extra0 = 0u,
                        uint32_t extra1 = 0u) {
    return decode(kMimg | mimg_op(kMimgLoadMip) | (dim << 3) | (dmask << 8) | extra0,
                  vaddr | extra1);
}
}  // namespace

TEST(Rdna2DecodeSweep, DynamicMipShapeConsecutiveFormPutsMipLast) {
    uint32_t reg = 999;
    ASSERT_TRUE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 10), &reg));   // 2D = [x, y, mip]
    EXPECT_EQ(reg, 12u);
    ASSERT_TRUE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(5, 10), &reg));   // 2D_ARRAY adds slice
    EXPECT_EQ(reg, 13u);
    EXPECT_TRUE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 10)));          // out param optional
}

TEST(Rdna2DecodeSweep, DynamicMipShapeRejectsOtherDimsAndOpcodes) {
    for (uint32_t dim = 0; dim < 8; ++dim) {
        const bool admitted = rdna2_mimg_dynamic_mip_shape(mimg_load_mip(dim, 4));
        EXPECT_EQ(admitted, dim == 1 || dim == 5) << "dim=" << dim;
    }
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const Rdna2Inst in = decode(kMimg | ((op >> 7) & 1u) | ((op & 0x7Fu) << 18) | (1u << 3), 4u);
        EXPECT_EQ(rdna2_mimg_dynamic_mip_shape(in), op == kMimgLoadMip) << "op=" << op;
    }
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(decode(0x7E000000u)));   // not MIMG at all
}

TEST(Rdna2DecodeSweep, DynamicMipShapeRejectsLayoutChangingModifiersButAdmitsCacheHints) {
    // R128/TFE/LWE/A16/D16 and reserved bits change the address or data layout.
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 4, 0xF, 1u << 15)));   // r128
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 4, 0xF, 1u << 16)));   // tfe
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 4, 0xF, 1u << 17)));   // lwe
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 4, 0xF, 0u, 1u << 30)));  // a16
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 4, 0xF, 0u, 1u << 31)));  // d16
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 4, 0xF, 1u << 6)));       // reserved
    // GLC/SLC/DLC/UNORM are addressing and cache hints: the mip operand does not move.
    for (uint32_t bit : {7u, 12u, 13u, 25u}) {
        EXPECT_TRUE(rdna2_mimg_dynamic_mip_shape(mimg_load_mip(1, 4, 0xF, 1u << bit))) << bit;
    }
}

TEST(Rdna2DecodeSweep, DynamicMipShapeNsaFormReadsTheMipByte) {
    // Byte-for-byte shape of a live NSA 2D load: image_load_mip v[5:7], [v0, v42, v5], dmask:0x7.
    // Address 0 is VADDR, then word2 byte0 names y and byte1 names the mip VGPR.
    const uint32_t code[3] = {0xF004070Au, 0x00080500u, 0x0000052Au};
    const Rdna2Inst in = rdna2_decode_one(code, 3);
    ASSERT_EQ(in.fmt, Rdna2Format::MIMG);
    ASSERT_EQ(in.len_dwords, 3u);
    uint32_t reg = 0;
    ASSERT_TRUE(rdna2_mimg_dynamic_mip_shape(in, &reg));
    EXPECT_EQ(reg, 5u);

    // 2D_ARRAY adds a slice byte, so the mip is byte 2 and a nonzero byte 3 is not this shape.
    uint32_t arr[3] = {kMimg | mimg_op(kMimgLoadMip) | (1u << 1) | (5u << 3) | (0xFu << 8), 0u,
                       0x00071A2Au};
    ASSERT_TRUE(rdna2_mimg_dynamic_mip_shape(rdna2_decode_one(arr, 3), &reg));
    EXPECT_EQ(reg, 7u);
    arr[2] = 0x01071A2Au;
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(rdna2_decode_one(arr, 3)));
    // NSA=2 on this operand count is not something a compiler emits; it stays fail-closed.
    const uint32_t big[4] = {kMimg | mimg_op(kMimgLoadMip) | (2u << 1) | (1u << 3), 0u, 0u, 0u};
    EXPECT_FALSE(rdna2_mimg_dynamic_mip_shape(rdna2_decode_one(big, 4)));
}

TEST(Rdna2DecodeSweep, ZeroMipShapeAdmitsOnlyTheEvidencedPackets) {
    // Branch 1: IMAGE_LOAD_MIP, UNORM+GLC, dmask 1 or 0xF, 2D/2D_ARRAY, consecutive.
    const uint32_t unorm_glc = (1u << 12) | (1u << 13);
    uint32_t reg = 999;
    ASSERT_TRUE(rdna2_mimg_zero_mip_shape(mimg_load_mip(1, 10, 0xF, unorm_glc), &reg));
    EXPECT_EQ(reg, 12u);
    ASSERT_TRUE(rdna2_mimg_zero_mip_shape(mimg_load_mip(5, 10, 0x1, unorm_glc), &reg));
    EXPECT_EQ(reg, 13u);
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(mimg_load_mip(1, 10, 0xF, 0u)));            // needs UNORM+GLC
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(mimg_load_mip(1, 10, 0x3, unorm_glc)));     // dmask 3
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(mimg_load_mip(2, 10, 0xF, unorm_glc)));     // wrong dim
}

TEST(Rdna2DecodeSweep, ZeroMipShapeRejectsAnyLayoutOrCacheModifier) {
    const uint32_t unorm_glc = (1u << 12) | (1u << 13);
    for (uint32_t bit : {7u, 15u, 16u, 17u, 25u, 6u, 14u}) {   // dlc r128 tfe lwe slc + reserved
        EXPECT_FALSE(rdna2_mimg_zero_mip_shape(mimg_load_mip(1, 10, 0xF, unorm_glc | (1u << bit))))
            << "d0 bit " << bit;
    }
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(mimg_load_mip(1, 10, 0xF, unorm_glc, 1u << 30)));  // a16
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(mimg_load_mip(1, 10, 0xF, unorm_glc, 1u << 31)));  // d16
}

TEST(Rdna2DecodeSweep, ZeroMipShapeStoreMipNsaReadsWordTwoByteOne) {
    const uint32_t unorm_glc = (1u << 12) | (1u << 13);
    const uint32_t w0 = kMimg | mimg_op(kMimgStoreMip) | (1u << 1) | (1u << 3) | (0xFu << 8) | unorm_glc;
    const uint32_t ok[3] = {w0, 0u, 0x00005A33u};
    uint32_t reg = 0;
    ASSERT_TRUE(rdna2_mimg_zero_mip_shape(rdna2_decode_one(ok, 3), &reg));
    EXPECT_EQ(reg, 0x5Au);
    const uint32_t high[3] = {w0, 0u, 0x00015A33u};   // bytes above the two modelled ones must be 0
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(rdna2_decode_one(high, 3)));
}

TEST(Rdna2DecodeSweep, ZeroMipShapeLoadMipNsaRequiresClearUnormAndGlc) {
    const uint32_t w0 = kMimg | mimg_op(kMimgLoadMip) | (1u << 1) | (1u << 3) | (0x7u << 8);
    const uint32_t code[3] = {w0, 0u, 0x0000052Au};
    uint32_t reg = 0;
    ASSERT_TRUE(rdna2_mimg_zero_mip_shape(rdna2_decode_one(code, 3), &reg));
    EXPECT_EQ(reg, 5u);
    const uint32_t with_glc[3] = {w0 | (1u << 13), 0u, 0x0000052Au};
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(rdna2_decode_one(with_glc, 3)));
    EXPECT_FALSE(rdna2_mimg_zero_mip_shape(decode(0x7E000000u)));   // not MIMG
}

// ---- SDWA and DPP16 control words ------------------------------------------------------------
// SDWA control dword: SRC0[7:0], DST_SEL[10:8], DST_UNUSED[12:11], CLAMP[13], OMOD[15:14],
//   SRC0_SEL[18:16], SRC0_SEXT[19], SRC0_NEG[20], SRC0_ABS[21], SRC0_S[23], SRC1_SEL[26:24],
//   SRC1_SEXT[27], SRC1_NEG[28], SRC1_ABS[29], SRC1_S[31]. Select 6 = DWORD, 4/5 = WORD_0/1.
// DPP16 control dword: SRC0[7:0], DPP_CTRL[16:8], FI[18], BC[19], SRC0_NEG/ABS and SRC1_NEG/ABS
//   [23:20], BANK_MASK[27:24], ROW_MASK[31:28].
// Anything the decoder does not model must keep has_modifier set so the recompiler rejects it
// visibly instead of lowering the instruction with the wrong semantics.

namespace {
struct Sdwa {
    uint32_t src0 = 0x100u, dsel = 6, dun = 0, clamp = 0, omod = 0;
    uint32_t s0sel = 6, s0sext = 0, s0neg = 0, s0abs = 0, s0s = 0;
    uint32_t s1sel = 6, s1sext = 0, s1neg = 0, s1abs = 0, s1s = 0;
    uint32_t word() const {
        return (src0 & 0xFFu) | (dsel << 8) | (dun << 11) | (clamp << 13) | (omod << 14) |
               (s0sel << 16) | (s0sext << 19) | (s0neg << 20) | (s0abs << 21) | (s0s << 23) |
               (s1sel << 24) | (s1sext << 27) | (s1neg << 28) | (s1abs << 29) | (s1s << 31);
    }
};
constexpr uint32_t kVop2Add = 0x03u;      // v_add_f32: a float op with no sub-dword SDWA model
constexpr uint32_t kVop2AddNc = 0x25u;    // v_add_nc_u32: an integer op
constexpr uint32_t vop2_sdwa_w0(uint32_t op, uint32_t vdst = 2u, uint32_t vsrc1 = 3u) {
    return (op << 25) | (vdst << 17) | (vsrc1 << 9) | 0xF9u;
}
constexpr uint32_t vop1_sdwa_w0(uint32_t op, uint32_t vdst = 2u) {
    return 0x7E000000u | (vdst << 17) | (op << 9) | 0xF9u;
}
}  // namespace

TEST(Rdna2DecodeSweep, Vop1F16UnaryFamilyIsExactlyTheDocumentedRanges) {
    // 0x59 / 0x5A (frexp mantissa / i16 exponent) sit inside the numeric span and are excluded.
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        const bool expected = (op >= 0x54 && op <= 0x58) || (op >= 0x5B && op <= 0x61);
        EXPECT_EQ(vop1_is_f16_unary(op), expected) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Vop2SdwaTrivialFormIsRecordedAndAdmitted) {
    const Rdna2Inst in = decode(vop2_sdwa_w0(kVop2Add), Sdwa{}.word());
    ASSERT_EQ(in.fmt, Rdna2Format::VOP2);
    EXPECT_EQ(in.len_dwords, 2u);
    EXPECT_TRUE(in.has_sdwa);
    EXPECT_FALSE(in.has_modifier) << "all-DWORD selects with no sext/reserved bits are modelled";
    EXPECT_EQ(in.sdwa_dst_sel, 6);
    EXPECT_EQ(in.sdwa_src0_sel, 6);
    EXPECT_EQ(in.sdwa_src1_sel, 6);
    EXPECT_EQ(in.src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(in.src[0].value, 0);
    EXPECT_EQ(in.src[1].value, 3);
    EXPECT_EQ(in.dst.value, 2);
}

TEST(Rdna2DecodeSweep, Vop2SdwaPlainE32IsNotMarkedSdwa) {
    const Rdna2Inst plain = decode((kVop2Add << 25) | (2u << 17) | (3u << 9) | 0x100u);
    EXPECT_FALSE(plain.has_sdwa);
    EXPECT_FALSE(plain.has_modifier);
}

TEST(Rdna2DecodeSweep, Vop2SdwaSourceNegAbsAreReadPerSourceAndKeepTheFormTrivial) {
    for (uint32_t m = 0; m < 16; ++m) {
        Sdwa s;
        s.s0neg = m & 1u; s.s0abs = (m >> 1) & 1u; s.s1neg = (m >> 2) & 1u; s.s1abs = (m >> 3) & 1u;
        const Rdna2Inst in = decode(vop2_sdwa_w0(kVop2Add), s.word());
        EXPECT_EQ(in.src_neg[0], (m & 1u) != 0) << m;
        EXPECT_EQ(in.src_abs[0], (m & 2u) != 0) << m;
        EXPECT_EQ(in.src_neg[1], (m & 4u) != 0) << m;
        EXPECT_EQ(in.src_abs[1], (m & 8u) != 0) << m;
        EXPECT_FALSE(in.has_modifier) << m;
    }
}

TEST(Rdna2DecodeSweep, Vop2SdwaClampAndOmodAreDecoded) {
    Sdwa s;
    s.clamp = 1;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2Add), s.word()).clamp);
    for (uint32_t omod = 0; omod < 4; ++omod) {
        Sdwa o;
        o.omod = omod;
        const Rdna2Inst in = decode(vop2_sdwa_w0(kVop2Add), o.word());
        EXPECT_EQ(in.omod, omod) << omod;
        EXPECT_FALSE(in.has_modifier) << "float output modifiers are applied by the recompiler";
    }
}

TEST(Rdna2DecodeSweep, Vop2SdwaScalarSourceBitsSwitchOperandKind) {
    Sdwa s;
    s.src0 = 5; s.s0s = 1;      // SRC0 is an SGPR field, not a VGPR
    s.s1s = 1;                  // SRC1 reads dword0[16:9] as an SSRC field
    const Rdna2Inst in = decode(vop2_sdwa_w0(kVop2Add, 2u, 7u), s.word());
    EXPECT_EQ(in.src[0].kind, OperandKind::SGPR);
    EXPECT_EQ(in.src[0].value, 5);
    EXPECT_EQ(in.src[1].kind, OperandKind::SGPR);
    EXPECT_EQ(in.src[1].value, 7);
    EXPECT_EQ(decode(vop2_sdwa_w0(kVop2Add, 2u, 7u), Sdwa{}.word()).src[1].kind, OperandKind::VGPR);
}

TEST(Rdna2DecodeSweep, Vop2SdwaUnmodelledSubDwordFloatFormsStayRejected) {
    // v_add_f32 has no sub-dword SDWA lowering, so any non-DWORD select or SEXT must keep has_modifier.
    for (uint32_t sel = 0; sel <= 5; ++sel) {
        Sdwa a; a.s0sel = sel;
        EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2Add), a.word()).has_modifier) << "src0_sel=" << sel;
        Sdwa b; b.s1sel = sel;
        EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2Add), b.word()).has_modifier) << "src1_sel=" << sel;
        Sdwa d; d.dsel = sel;
        EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2Add), d.word()).has_modifier) << "dst_sel=" << sel;
    }
    Sdwa sext; sext.s0sext = 1;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2Add), sext.word()).has_modifier);
    Sdwa sext1; sext1.s1sext = 1;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2Add), sext1.word()).has_modifier);
}

TEST(Rdna2DecodeSweep, Vop2SdwaIntegerOpAdmitsEveryByteAndWordSelect) {
    for (uint32_t sel = 0; sel <= 6; ++sel) {
        Sdwa a; a.s0sel = sel; a.s1sel = 6;
        const Rdna2Inst in = decode(vop2_sdwa_w0(kVop2AddNc), a.word());
        EXPECT_FALSE(in.has_modifier) << "src0_sel=" << sel;
        EXPECT_EQ(in.sdwa_src0_sel, sel);
        Sdwa b; b.s0sel = 6; b.s1sel = sel;
        const Rdna2Inst jn = decode(vop2_sdwa_w0(kVop2AddNc), b.word());
        EXPECT_FALSE(jn.has_modifier) << "src1_sel=" << sel;
        EXPECT_EQ(jn.sdwa_src1_sel, sel);
        Sdwa d; d.dsel = sel; d.dun = 2; d.s0sel = 4;
        const Rdna2Inst dn = decode(vop2_sdwa_w0(kVop2AddNc), d.word());
        EXPECT_FALSE(dn.has_modifier) << "dst_sel=" << sel;
        EXPECT_EQ(dn.sdwa_dst_sel, sel);
        EXPECT_EQ(dn.sdwa_dst_unused, 2);
    }
    // Selector value 7 is reserved.
    Sdwa r; r.s0sel = 7;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2AddNc), r.word()).has_modifier);
    Sdwa r1; r1.s1sel = 7;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2AddNc), r1.word()).has_modifier);
}

TEST(Rdna2DecodeSweep, Vop2SdwaIntegerOpDstUnusedAndSextRules) {
    // Only UNUSED_PAD (0) and UNUSED_PRESERVE (2) are modelled; UNUSED_SEXT (1) and 3 are not.
    for (uint32_t dun = 0; dun < 4; ++dun) {
        Sdwa s; s.dsel = 4; s.dun = dun;
        EXPECT_EQ(decode(vop2_sdwa_w0(kVop2AddNc), s.word()).has_modifier, dun == 1 || dun == 3)
            << "dun=" << dun;
    }
    // SEXT is admitted only alongside a real sub-dword select, and is then recorded.
    Sdwa ok; ok.s0sel = 4; ok.s0sext = 1;
    const Rdna2Inst in = decode(vop2_sdwa_w0(kVop2AddNc), ok.word());
    EXPECT_FALSE(in.has_modifier);
    EXPECT_TRUE(in.sdwa_src0_sext);
    EXPECT_FALSE(in.sdwa_src1_sext);
    Sdwa ok1; ok1.s1sel = 5; ok1.s1sext = 1;
    const Rdna2Inst jn = decode(vop2_sdwa_w0(kVop2AddNc), ok1.word());
    EXPECT_FALSE(jn.has_modifier);
    EXPECT_TRUE(jn.sdwa_src1_sext);
    Sdwa bad; bad.s0sel = 6; bad.s0sext = 1;   // SEXT of a full DWORD is not modelled
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2AddNc), bad.word()).has_modifier);
    // Integer saturation and float source modifiers are not modelled on this path.
    Sdwa clamp; clamp.s0sel = 4; clamp.clamp = 1;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2AddNc), clamp.word()).has_modifier);
    Sdwa omod; omod.s0sel = 4; omod.omod = 1;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2AddNc), omod.word()).has_modifier);
    Sdwa neg; neg.s0sel = 4; neg.s0neg = 1;
    EXPECT_TRUE(decode(vop2_sdwa_w0(kVop2AddNc), neg.word()).has_modifier);
}

TEST(Rdna2DecodeSweep, Vop2SdwaIntegerFamilyBoundaries) {
    // The integer VOP2 ops with a sub-dword SDWA lowering: 0x0B, 0x11-0x14, 0x16, 0x18, 0x1A-0x1E,
    // 0x25-0x2A. A byte select on a neighbour outside that set stays rejected.
    const auto integer_op = [](uint32_t op) {
        return op == 0x0B || (op >= 0x11 && op <= 0x14) || op == 0x16 || op == 0x18 ||
               (op >= 0x1A && op <= 0x1E) || (op >= 0x25 && op <= 0x2A);
    };
    Sdwa byte0; byte0.s0sel = 0;
    for (uint32_t op = 0; op <= 0x3D; ++op) {
        const Rdna2Inst in = decode(vop2_sdwa_w0(op), byte0.word());
        ASSERT_EQ(in.fmt, Rdna2Format::VOP2);
        // The f16 ops (0x32/0x33/0x35/0x39/0x3A), cndmask (0x01) and 0x1A have their own rules, so
        // only ops outside every SDWA-capable family are asserted here.
        const bool special = op == 0x01 || op == 0x32 || op == 0x33 || op == 0x35 || op == 0x39 ||
                             op == 0x3A;
        if (special) continue;
        EXPECT_EQ(in.has_modifier, !integer_op(op)) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Vop2SdwaWordDestinationF16FamilyAndCndmask) {
    // The f16 half-packing idiom: WORD destination with UNUSED_PRESERVE, WORD/DWORD sources.
    for (uint32_t op : {0x32u, 0x33u, 0x35u, 0x39u, 0x3Au, 0x01u}) {
        Sdwa s; s.dsel = 5; s.dun = 2; s.s0sel = 5; s.s1sel = 4;
        const Rdna2Inst in = decode(vop2_sdwa_w0(op), s.word());
        EXPECT_FALSE(in.has_modifier) << "op=" << op;
        EXPECT_EQ(in.sdwa_dst_sel, 5) << "op=" << op;
        EXPECT_EQ(in.sdwa_dst_unused, 2) << "op=" << op;
        EXPECT_EQ(in.sdwa_src0_sel, 5) << "op=" << op;
        EXPECT_EQ(in.sdwa_src1_sel, 4) << "op=" << op;
        Sdwa byte; byte.dsel = 5; byte.dun = 2; byte.s0sel = 1; byte.s1sel = 4;
        EXPECT_TRUE(decode(vop2_sdwa_w0(op), byte.word()).has_modifier) << "BYTE source, op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Vop2LshlrevSdwaAdmitsOneByteFromSrc1Only) {
    // v_lshlrev_b32_sdwa v, 2, v src1_sel:BYTE_k -- the NGG byte-index idiom: full-dword dst and
    // shift amount, one byte of src1.
    for (uint32_t sel = 0; sel <= 5; ++sel) {
        Sdwa s; s.s1sel = sel;
        EXPECT_FALSE(decode(vop2_sdwa_w0(0x1Au), s.word()).has_modifier) << sel;
    }
    // A byte select on SRC0 (the shift amount) is still admitted by the shared integer path; this
    // test pins only that the NGG byte-from-src1 form decodes without a modifier flag.
    Sdwa s0; s0.s0sel = 0;
    EXPECT_FALSE(decode(vop2_sdwa_w0(0x1Au), s0.word()).has_modifier);
}

TEST(Rdna2DecodeSweep, VopcSdwaDestinationAndSelectRules) {
    // VOPC SDWA has no dst_sel; bit 15 (SD) redirects the mask to the SDST SGPR in bits [14:8].
    const uint32_t vopc_w0 = 0x7C000000u | (0xC4u << 17) | (3u << 9) | 0xF9u;   // v_cmp_*_u32 SDWA
    EXPECT_EQ(decode(vopc_w0, Sdwa{}.word()).dst.kind, OperandKind::Special);
    EXPECT_EQ(decode(vopc_w0, Sdwa{}.word()).dst.value, 106);
    Sdwa sdst; sdst.dsel = 0;   // bits [14:8] are SDST here, not a dst_sel, so start them clear
    const Rdna2Inst sd = decode(vopc_w0, sdst.word() | (1u << 15) | (42u << 8));
    EXPECT_EQ(sd.dst.kind, OperandKind::SGPR);
    EXPECT_EQ(sd.dst.value, 42);
    // Trivial DWORD form is modelled and neg/abs are recorded.
    Sdwa n; n.s0neg = 1; n.s1abs = 1;
    const Rdna2Inst triv = decode(vopc_w0, n.word());
    EXPECT_FALSE(triv.has_modifier);
    EXPECT_TRUE(triv.src_neg[0]);
    EXPECT_TRUE(triv.src_abs[1]);
    // An integer compare may select any byte/word (0xC1-0xC6 are v_cmp_*_u32), zero-extended only.
    for (uint32_t sel = 0; sel <= 6; ++sel) {
        Sdwa s; s.s1sel = sel;
        EXPECT_FALSE(decode(vopc_w0, s.word()).has_modifier) << "src1_sel=" << sel;
        EXPECT_EQ(decode(vopc_w0, s.word()).sdwa_src1_sel, sel) << sel;
    }
    Sdwa sext; sext.s1sel = 0; sext.s1sext = 1;
    EXPECT_TRUE(decode(vopc_w0, sext.word()).has_modifier) << "signed extension is not modelled";
    // A float compare (v_cmp_*_f32, op 0x01) with a byte select is not modelled.
    const uint32_t f32_w0 = 0x7C000000u | (0x01u << 17) | (3u << 9) | 0xF9u;
    Sdwa byte; byte.s0sel = 0;
    EXPECT_TRUE(decode(f32_w0, byte.word()).has_modifier);
}

TEST(Rdna2DecodeSweep, VopcCmpxSdwaIsAdmittedThroughItsCmpCounterpart) {
    // v_cmpx_*_u32 (0xD1..0xD6) must follow the same rule as v_cmp_*_u32 (0xC1..0xC6): admission
    // maps a cmpx opcode back to its base. Missing this once rejected every v_cmpx_*_u16 SDWA packet.
    Sdwa s; s.s0sel = 1; s.s1sel = 4;
    for (uint32_t base = 0xC1; base <= 0xC6; ++base) {
        const uint32_t cmp = 0x7C000000u | (base << 17) | (3u << 9) | 0xF9u;
        const uint32_t cmpx = 0x7C000000u | ((base + 0x10u) << 17) | (3u << 9) | 0xF9u;
        EXPECT_FALSE(decode(cmp, s.word()).has_modifier) << "cmp base=" << base;
        EXPECT_FALSE(decode(cmpx, s.word()).has_modifier) << "cmpx base=" << base;
    }
    // The u16 window 0xA9..0xAE and its cmpx counterpart 0xB9..0xBE.
    for (uint32_t base = 0xA9; base <= 0xAE; ++base) {
        const uint32_t cmpx = 0x7C000000u | ((base + 0x10u) << 17) | (3u << 9) | 0xF9u;
        EXPECT_FALSE(decode(cmpx, s.word()).has_modifier) << "u16 cmpx base=" << base;
    }
}

TEST(Rdna2DecodeSweep, Vop1SdwaPlainMovAndF16UnaryAdmission) {
    // v_mov_b32 (op 1) with DWORD selects is trivial; src0 comes from the control dword.
    const Rdna2Inst mov = decode(vop1_sdwa_w0(0x01u), Sdwa{}.word());
    EXPECT_TRUE(mov.has_sdwa);
    EXPECT_FALSE(mov.has_modifier);
    EXPECT_EQ(mov.n_src, 1);
    // The f16 unary family writes one destination half (UNUSED_PRESERVE) from either source half.
    for (uint32_t op = 0; op <= 0xFF; ++op) {
        if (!vop1_is_f16_unary(op)) continue;
        Sdwa good; good.dsel = 5; good.dun = 2; good.s0sel = 4;
        EXPECT_FALSE(decode(vop1_sdwa_w0(op), good.word()).has_modifier) << "op=" << op;
        Sdwa byte; byte.dsel = 5; byte.dun = 2; byte.s0sel = 0;
        EXPECT_TRUE(decode(vop1_sdwa_w0(op), byte.word()).has_modifier) << "BYTE, op=" << op;
        Sdwa pad; pad.dsel = 5; pad.dun = 0; pad.s0sel = 4;
        EXPECT_TRUE(decode(vop1_sdwa_w0(op), pad.word()).has_modifier) << "UNUSED_PAD, op=" << op;
    }
    // The frexp pair is inside the numeric span but is not that family.
    Sdwa good; good.dsel = 5; good.dun = 2; good.s0sel = 4;
    EXPECT_TRUE(decode(vop1_sdwa_w0(0x59u), good.word()).has_modifier);
    EXPECT_TRUE(decode(vop1_sdwa_w0(0x5Au), good.word()).has_modifier);
}

TEST(Rdna2DecodeSweep, Vop1SdwaIntegerToFloatConvertRules) {
    // v_cvt_f32_i32 (0x05) honours SEXT; v_cvt_f32_u32 (0x06) zero-extends only. Full-dword dst.
    for (uint32_t sel = 0; sel <= 5; ++sel) {
        Sdwa u; u.s0sel = sel;
        EXPECT_FALSE(decode(vop1_sdwa_w0(0x06u), u.word()).has_modifier) << "u32 sel=" << sel;
        Sdwa i; i.s0sel = sel; i.s0sext = 1;
        EXPECT_FALSE(decode(vop1_sdwa_w0(0x05u), i.word()).has_modifier) << "i32 sext sel=" << sel;
        EXPECT_TRUE(decode(vop1_sdwa_w0(0x06u), i.word()).has_modifier) << "u32 sext sel=" << sel;
    }
    Sdwa sub; sub.dsel = 4; sub.s0sel = 0;     // a sub-dword destination is not modelled here
    EXPECT_TRUE(decode(vop1_sdwa_w0(0x06u), sub.word()).has_modifier);
}

// ---- DPP16 ----------------------------------------------------------------------------------

namespace {
struct Dpp {
    uint32_t src0 = 4, ctrl = 0xE4, fi = 0, bc = 0, mods = 0, bank = 0xF, row = 0xF;
    uint32_t word() const {
        return (src0 & 0xFFu) | (ctrl << 8) | (fi << 18) | (bc << 19) | (mods << 20) | (bank << 24) |
               (row << 28);
    }
};
constexpr uint32_t kDpp16Src = 0xFAu;
constexpr uint32_t vop1_dpp_w0(uint32_t op, uint32_t vdst = 2u) {
    return 0x7E000000u | (vdst << 17) | (op << 9) | kDpp16Src;
}
constexpr uint32_t vop2_dpp_w0(uint32_t op, uint32_t vdst = 2u, uint32_t vsrc1 = 3u) {
    return (op << 25) | (vdst << 17) | (vsrc1 << 9) | kDpp16Src;
}
}  // namespace

TEST(Rdna2DecodeSweep, Dpp16QuadPermAndRowShrAreAdmittedWithFullMasks) {
    for (uint32_t ctrl = 0; ctrl < 0x200; ++ctrl) {
        Dpp d; d.ctrl = ctrl; d.bc = 0;
        const Rdna2Inst in = decode(vop1_dpp_w0(0x01u), d.word());
        ASSERT_EQ(in.fmt, Rdna2Format::VOP1);
        EXPECT_EQ(in.len_dwords, 2u);
        const bool modelled = ctrl < 0x100u || (ctrl >= 0x111u && ctrl <= 0x11Fu);
        EXPECT_EQ(in.has_dpp, modelled) << "ctrl=" << ctrl;
        EXPECT_EQ(in.has_modifier, !modelled) << "ctrl=" << ctrl;
        if (modelled) EXPECT_EQ(in.dpp_ctrl, ctrl) << ctrl;
    }
}

TEST(Rdna2DecodeSweep, Dpp16RowXorFamilyNeedsBoundCtrlAndAnEligibleOpcode) {
    // ROW_ROR:8 (0x128) and ROW_XMASK:0..15 (0x160..0x16F) are the same lane-XOR permutation.
    for (uint32_t ctrl = 0x100; ctrl < 0x200; ++ctrl) {
        const bool xor_ctrl = ctrl == 0x128u || (ctrl >= 0x160u && ctrl <= 0x16Fu);
        const bool shr = ctrl >= 0x111u && ctrl <= 0x11Fu;
        Dpp bc1; bc1.ctrl = ctrl; bc1.bc = 1;
        EXPECT_EQ(decode(vop1_dpp_w0(0x01u), bc1.word()).has_dpp, xor_ctrl || shr) << "ctrl=" << ctrl;
        Dpp bc0; bc0.ctrl = ctrl; bc0.bc = 0;
        EXPECT_EQ(decode(vop1_dpp_w0(0x01u), bc0.word()).has_dpp, shr) << "BC=0, ctrl=" << ctrl;
    }
    Dpp x; x.ctrl = 0x164; x.bc = 1;
    EXPECT_TRUE(decode(vop1_dpp_w0(0x01u), x.word()).has_dpp);           // v_mov_b32
    EXPECT_FALSE(decode(vop1_dpp_w0(0x02u), x.word()).has_dpp);          // v_readfirstlane_b32
    EXPECT_FALSE(decode(vop1_dpp_w0(0x05u), x.word()).has_dpp);          // v_cvt_f32_i32
    EXPECT_TRUE(decode(vop2_dpp_w0(0x03u), x.word()).has_dpp);           // v_add_f32
    // The carry-writing trio keeps a second architectural result the lowering does not restore.
    for (uint32_t op : {0x28u, 0x29u, 0x2Au}) {
        EXPECT_FALSE(decode(vop2_dpp_w0(op), x.word()).has_dpp) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Dpp16CapturesSourceControlBoundCtrlAndMasks) {
    Dpp d; d.src0 = 9; d.ctrl = 0x1B; d.bc = 1;
    const Rdna2Inst in = decode(vop1_dpp_w0(0x01u), d.word());
    ASSERT_TRUE(in.has_dpp);
    EXPECT_EQ(in.dpp_ctrl, 0x1B);
    EXPECT_TRUE(in.dpp_bound_ctrl);
    EXPECT_EQ(in.dpp_bank_mask, 0xF);
    EXPECT_EQ(in.dpp_row_mask, 0xF);
    ASSERT_EQ(in.n_src, 1);
    EXPECT_EQ(in.src[0].kind, OperandKind::VGPR) << "the real SRC0 comes from dword1, not the 0xFA marker";
    EXPECT_EQ(in.src[0].value, 9);

    const Rdna2Inst two = decode(vop2_dpp_w0(0x03u, 2u, 6u), d.word());
    ASSERT_TRUE(two.has_dpp);
    ASSERT_EQ(two.n_src, 2);
    EXPECT_EQ(two.src[0].value, 9);
    EXPECT_EQ(two.src[1].value, 6) << "SRC1 stays the dword0 VGPR field";
}

TEST(Rdna2DecodeSweep, Dpp16RejectsFetchInactiveSourceModifiersAndPartialMasks) {
    Dpp ok;
    ASSERT_TRUE(decode(vop1_dpp_w0(0x01u), ok.word()).has_dpp);
    Dpp fi; fi.fi = 1;
    EXPECT_TRUE(decode(vop1_dpp_w0(0x01u), fi.word()).has_modifier) << "fetch-inactive";
    for (uint32_t bit = 0; bit < 4; ++bit) {   // src0/src1 neg/abs at dword1[23:20]
        Dpp m; m.mods = 1u << bit;
        EXPECT_FALSE(decode(vop1_dpp_w0(0x01u), m.word()).has_dpp) << "mods bit " << bit;
    }
    for (uint32_t mask = 0; mask < 0xF; ++mask) {
        Dpp b; b.bank = mask;
        EXPECT_FALSE(decode(vop1_dpp_w0(0x01u), b.word()).has_dpp) << "bank=" << mask;
        Dpp r; r.row = mask;
        EXPECT_FALSE(decode(vop1_dpp_w0(0x01u), r.word()).has_dpp) << "row=" << mask;
    }
}

TEST(Rdna2DecodeSweep, Dpp16ExactPartialRowMaskFormForIntegerAdd) {
    // v_add_nc_u32 with QUAD_PERM identity, ROW_MASK=0xA (rows 1/3 execute, 0/2 keep VDST),
    // BANK_MASK=0xF, BC=0 is the one admitted partial-mask form; the masks are retained.
    Dpp d; d.ctrl = 0xE4; d.row = 0xA; d.bank = 0xF; d.bc = 0;
    const Rdna2Inst in = decode(vop2_dpp_w0(kVop2AddNc), d.word());
    ASSERT_TRUE(in.has_dpp);
    EXPECT_EQ(in.dpp_row_mask, 0xA);
    EXPECT_EQ(in.dpp_bank_mask, 0xF);
    EXPECT_FALSE(in.dpp_bound_ctrl);
    Dpp bc = d; bc.bc = 1;
    EXPECT_FALSE(decode(vop2_dpp_w0(kVop2AddNc), bc.word()).has_dpp);
    Dpp row = d; row.row = 0x5;
    EXPECT_FALSE(decode(vop2_dpp_w0(kVop2AddNc), row.word()).has_dpp);
    Dpp bank = d; bank.bank = 0xE;
    EXPECT_FALSE(decode(vop2_dpp_w0(kVop2AddNc), bank.word()).has_dpp);
    Dpp ctrl = d; ctrl.ctrl = 0xE5;
    EXPECT_FALSE(decode(vop2_dpp_w0(kVop2AddNc), ctrl.word()).has_dpp);
    EXPECT_FALSE(decode(vop2_dpp_w0(kVop2Add), d.word()).has_dpp) << "other opcodes are not admitted";
}

TEST(Rdna2DecodeSweep, Dpp16IsNeverAdmittedOnVopcAndDpp8IsNeverAdmitted) {
    Dpp d;
    const Rdna2Inst cmp = decode(0x7C000000u | (0xC4u << 17) | (3u << 9) | kDpp16Src, d.word());
    EXPECT_EQ(cmp.len_dwords, 2u);
    EXPECT_FALSE(cmp.has_dpp);
    EXPECT_TRUE(cmp.has_modifier);
    // DPP8 (0xE9) and DPP8 with fetch-inactive (0xEA): correct length, never lowered.
    for (uint32_t marker : {0xE9u, 0xEAu}) {
        const Rdna2Inst in = decode(0x7E000000u | (2u << 17) | (0x01u << 9) | marker, 0u);
        EXPECT_EQ(in.len_dwords, 2u) << marker;
        EXPECT_FALSE(in.has_dpp) << marker;
        EXPECT_TRUE(in.has_modifier) << marker;
        EXPECT_FALSE(in.has_sdwa) << marker;
    }
}

// ---- VOP3P (packed / mixed-precision) --------------------------------------------------------
// dword0: VDST[7:0], NEG_HI[10:8], OPSEL[13:11], OPSEL_HI[2][14], CLAMP[15], OP[22:16].
// dword1: SRC0[8:0], SRC1[17:9], SRC2[26:18], OPSEL_HI[1:0][28:27], NEG[31:29].
// Four families have modelled modifiers: packed f16 (0x0E-0x12), packed 16-bit integer (0x00-0x0D,
// where CLAMP is unmodelled integer saturation), the fma_mix trio (0x20-0x22, where NEG_HI is
// ABS) and the integer dot family (0x14-0x19, where only the dot2 source selectors are modelled).
// Every other opcode must keep has_modifier on any modifier bit.

namespace {
struct Vop3p {
    uint32_t op = 0, vdst = 0, neg_hi = 0, opsel = 0, opsel_hi = 0, clamp = 0, neg = 0;
    uint32_t s0 = 0x100u, s1 = 0x101u, s2 = 0x102u;
    uint32_t w0() const {
        return 0xCC000000u | (op << 16) | vdst | (neg_hi << 8) | (opsel << 11) |
               (((opsel_hi >> 2) & 1u) << 14) | (clamp << 15);
    }
    uint32_t w1() const {
        return s0 | (s1 << 9) | (s2 << 18) | ((opsel_hi & 3u) << 27) | (neg << 29);
    }
    Rdna2Inst decoded() const { return decode(w0(), w1()); }
};
constexpr bool vop3p_is_packed_f16(uint32_t op) { return op >= 0x0E && op <= 0x12; }
constexpr bool vop3p_is_packed_int(uint32_t op) { return op <= 0x0D; }
constexpr bool vop3p_is_mix(uint32_t op) { return op >= 0x20 && op <= 0x22; }
constexpr bool vop3p_is_dot(uint32_t op) {
    return op >= 0x14 && op <= 0x19;
}
}  // namespace

TEST(Rdna2DecodeSweep, Vop3pDecodesOpcodeDestinationAndThreeSources) {
    for (uint32_t op = 0; op < 0x80; ++op) {
        Vop3p v; v.op = op; v.vdst = 9; v.s0 = 256u + 1u; v.s1 = 6u; v.s2 = 0xF2u;
        const Rdna2Inst in = v.decoded();
        ASSERT_EQ(in.fmt, Rdna2Format::VOP3P) << "op=" << op;
        EXPECT_EQ(in.opcode, op);
        EXPECT_EQ(in.len_dwords, 2u);
        EXPECT_EQ(in.dst.value, 9);
        ASSERT_EQ(in.n_src, 3);
        EXPECT_EQ(in.src[0].kind, OperandKind::VGPR);
        EXPECT_EQ(in.src[0].value, 1);
        EXPECT_EQ(in.src[1].kind, OperandKind::SGPR);
        EXPECT_EQ(in.src[1].value, 6);
        EXPECT_EQ(in.src[2].kind, OperandKind::InlineFloat);
    }
}

TEST(Rdna2DecodeSweep, Vop3pPackedF16CapturesEveryModifierAndKeepsTheFormModelled) {
    for (uint32_t op = 0x0E; op <= 0x12; ++op) {
        for (uint32_t neg = 0; neg < 8; ++neg) {
            for (uint32_t neg_hi = 0; neg_hi < 8; ++neg_hi) {
                Vop3p v; v.op = op; v.neg = neg; v.neg_hi = neg_hi; v.opsel = (neg ^ neg_hi) & 7u;
                v.opsel_hi = (neg + neg_hi) & 7u; v.clamp = (neg_hi >> 1) & 1u;
                const Rdna2Inst in = v.decoded();
                for (uint32_t k = 0; k < 3; ++k) {
                    EXPECT_EQ(in.src_neg[k], ((neg >> k) & 1u) != 0) << "op=" << op;
                }
                EXPECT_EQ(in.vop3p_neg_hi, neg_hi) << "op=" << op;
                EXPECT_EQ(in.vop3p_opsel, v.opsel) << "op=" << op;
                EXPECT_EQ(in.vop3p_opsel_hi, v.opsel_hi) << "op=" << op;
                EXPECT_EQ(in.clamp, v.clamp != 0) << "op=" << op;
                EXPECT_FALSE(in.has_modifier) << "every f16 packed modifier is modelled, op=" << op;
            }
        }
    }
}

TEST(Rdna2DecodeSweep, Vop3pOpselHiSpansDword1AndDword0Bit14) {
    // OPSEL_HI is 3 bits: [1:0] in dword1[28:27], bit 2 in dword0 bit 14.
    for (uint32_t hi = 0; hi < 8; ++hi) {
        Vop3p v; v.op = 0x0F; v.opsel_hi = hi;
        EXPECT_EQ(v.decoded().vop3p_opsel_hi, hi) << hi;
    }
}

TEST(Rdna2DecodeSweep, Vop3pPackedIntegerSharesFieldsButRejectsSaturation) {
    for (uint32_t op = 0; op <= 0x0D; ++op) {
        Vop3p v; v.op = op; v.neg = 5; v.neg_hi = 3; v.opsel = 6; v.opsel_hi = 5;
        const Rdna2Inst in = v.decoded();
        EXPECT_EQ(in.src_neg[0], true) << op;
        EXPECT_EQ(in.src_neg[1], false) << op;
        EXPECT_EQ(in.src_neg[2], true) << op;
        EXPECT_EQ(in.vop3p_neg_hi, 3) << op;
        EXPECT_EQ(in.vop3p_opsel, 6) << op;
        EXPECT_EQ(in.vop3p_opsel_hi, 5) << op;
        // The plainest encoding sets the default op_sel_hi:[1,1] and must still be accepted.
        EXPECT_FALSE(in.has_modifier) << "op=" << op;
        Vop3p sat = v; sat.clamp = 1;
        EXPECT_TRUE(sat.decoded().has_modifier) << "integer CLAMP is unmodelled saturation, op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Vop3pMixFamilyMapsNegHiToAbs) {
    for (uint32_t op = 0x20; op <= 0x22; ++op) {
        for (uint32_t m = 0; m < 8; ++m) {
            Vop3p v; v.op = op; v.neg = m; v.neg_hi = (~m) & 7u; v.opsel = 5; v.opsel_hi = 6; v.clamp = 1;
            const Rdna2Inst in = v.decoded();
            for (uint32_t k = 0; k < 3; ++k) {
                EXPECT_EQ(in.src_neg[k], ((m >> k) & 1u) != 0) << "op=" << op << " k=" << k;
                EXPECT_EQ(in.src_abs[k], (((~m) >> k) & 1u) != 0) << "op=" << op << " k=" << k;
            }
            EXPECT_EQ(in.vop3p_opsel, 5) << op;
            EXPECT_EQ(in.vop3p_opsel_hi, 6) << op;
            EXPECT_TRUE(in.clamp) << op;
            EXPECT_FALSE(in.has_modifier) << op;
            EXPECT_EQ(in.vop3p_neg_hi, 0) << "NEG_HI is ABS for the mix family, not a packed neg_hi";
        }
    }
}

TEST(Rdna2DecodeSweep, Vop3pIntegerDotModifiers) {
    for (uint32_t op = 0x14; op <= 0x19; ++op) {
        // Plain default encoding: opsel=0, opsel_hi=7 (or 0)
        Vop3p v;
        v.op = op;
        v.opsel = 0;
        v.opsel_hi = 7;
        EXPECT_FALSE(v.decoded().has_modifier) << "plain default, op=" << op;

        Vop3p clamp = v;
        clamp.clamp = 1;
        EXPECT_TRUE(clamp.decoded().has_modifier) << "clamp, op=" << op;

        Vop3p neg = v;
        neg.neg = 1;
        EXPECT_TRUE(neg.decoded().has_modifier) << "neg, op=" << op;

        Vop3p neg_hi = v;
        neg_hi.neg_hi = 1;
        EXPECT_TRUE(neg_hi.decoded().has_modifier) << "neg_hi, op=" << op;

        if (op == 0x14 || op == 0x15) {
            // v_dot2 allows opsel[1:0] to select halfwords
            Vop3p sel = v;
            sel.opsel = 3;
            EXPECT_FALSE(sel.decoded().has_modifier) << "opsel[1:0] valid for v_dot2, op=" << op;

            Vop3p bad_sel = v;
            bad_sel.opsel = 4;
            EXPECT_TRUE(bad_sel.decoded().has_modifier)
                << "opsel bit 2 invalid for v_dot2, op=" << op;
        } else {
            // v_dot4 and v_dot8 reject any opsel
            Vop3p sel = v;
            sel.opsel = 1;
            EXPECT_TRUE(sel.decoded().has_modifier) << "opsel invalid for dot4/dot8, op=" << op;
        }
    }
}

TEST(Rdna2DecodeSweep, Vop3pUnmodelledOpcodesRejectAnyModifierBit) {
    for (uint32_t op = 0; op < 0x80; ++op) {
        if (vop3p_is_packed_f16(op) || vop3p_is_packed_int(op) || vop3p_is_mix(op) ||
            vop3p_is_dot(op))
            continue;
        EXPECT_FALSE(Vop3p{.op = op}.decoded().has_modifier) << "plain, op=" << op;
        const struct { const char* name; Vop3p v; } cases[] = {
            {"neg_hi", Vop3p{.op = op, .neg_hi = 1}},
            {"opsel", Vop3p{.op = op, .opsel = 1}},
            {"opsel_hi bit2", Vop3p{.op = op, .opsel_hi = 4}},
            {"clamp", Vop3p{.op = op, .clamp = 1}},
            {"opsel_hi low", Vop3p{.op = op, .opsel_hi = 1}},
            {"neg", Vop3p{.op = op, .neg = 1}},
        };
        for (const auto& c : cases) {
            EXPECT_TRUE(c.v.decoded().has_modifier) << c.name << ", op=" << op;
        }
    }
}

TEST(Rdna2DecodeSweep, Vop3pFamilyBoundariesAreExact) {
    // The neighbours of each modelled range must not inherit its modifier handling.
    for (uint32_t op : {0x0Du, 0x0Eu, 0x12u, 0x13u, 0x1Fu, 0x20u, 0x22u, 0x23u}) {
        Vop3p v; v.op = op; v.opsel = 1;
        const bool modelled = vop3p_is_packed_f16(op) || vop3p_is_packed_int(op) || vop3p_is_mix(op);
        EXPECT_EQ(v.decoded().has_modifier, !modelled) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, Vop3pLiteralInAnySourceSlotAddsADword) {
    for (uint32_t slot = 0; slot < 3; ++slot) {
        Vop3p v; v.op = 0x20;
        (slot == 0 ? v.s0 : slot == 1 ? v.s1 : v.s2) = kLiteralSrc;
        const uint32_t code[3] = {v.w0(), v.w1(), 0x3F000000u};
        const Rdna2Inst in = rdna2_decode_one(code, 3);
        EXPECT_EQ(in.len_dwords, 3u) << "slot=" << slot;
        EXPECT_TRUE(in.has_literal) << slot;
        EXPECT_EQ(in.literal, 0x3F000000u) << slot;
    }
    // Without room for the literal the length clamps rather than reading past the stream.
    Vop3p v; v.op = 0x20; v.s1 = kLiteralSrc;
    const uint32_t cut[2] = {v.w0(), v.w1()};
    EXPECT_EQ(rdna2_decode_one(cut, 2).len_dwords, 2u);
    EXPECT_FALSE(rdna2_decode_one(cut, 2).has_literal);
}

// ---- Register-footprint helpers --------------------------------------------------------------
// rdna2_vgpr_write_count / rdna2_vgpr_destination_span / rdna2_tfe_status_vgpr /
// rdna2_vgpr_source_span feed control-flow analyses and register-file sizing. Under-reporting a
// write is silent (a proof then reasons about a stale register), so these are checked against the
// architectural result width of each opcode, taken from the gfx10 encoding tables.

namespace {
Rdna2Inst mubuf(uint32_t op, uint32_t vdata = 8u, uint32_t extra1 = 0u) {
    return decode(kTop6Mubuf | (op << 18), (vdata << 8) | extra1);
}
Rdna2Inst flat(uint32_t op, uint32_t vdst = 8u) {
    return decode(kTop6Flat | (op << 18), (vdst << 24) | (vdst << 8) | (125u << 16));
}
Rdna2Inst mtbuf(uint32_t op, uint32_t vdata = 8u, bool tfe = false) {
    return decode(kTop6Mtbuf | ((op & 7u) << 16),
                  (vdata << 8) | (((op >> 3) & 1u) << 21) | (tfe ? (1u << 23) : 0u));
}
Rdna2Inst mimg(uint32_t op, uint32_t dmask, uint32_t d1 = 0u, uint32_t extra0 = 0u) {
    return decode(kMimg | ((op >> 7) & 1u) | ((op & 0x7Fu) << 18) | (dmask << 8) | extra0,
                  (8u << 8) | d1);
}
}  // namespace

TEST(Rdna2DecodeSweep, WriteCountIsZeroWhenTheDestinationIsNotAVgpr) {
    EXPECT_EQ(rdna2_vgpr_write_count(decode(0x80000000u | (4u << 16) | (2u << 8) | 1u)), 0u);  // SOP2
    EXPECT_EQ(rdna2_vgpr_write_count(decode(0xBE800000u | (4u << 16) | (3u << 8) | 1u)), 0u);  // SOP1
    EXPECT_EQ(rdna2_vgpr_write_count(decode(0x7C000000u | (0xC4u << 17) | (1u << 9) | 0x100u)), 0u);
    EXPECT_EQ(rdna2_vgpr_write_count(decode(kTop6Exp, 0x04030201u)), 0u);
    EXPECT_EQ(rdna2_vgpr_write_count(Rdna2Inst{}), 0u);
}

TEST(Rdna2DecodeSweep, WriteCountForValuAndInterpolation) {
    // v_readfirstlane_b32 (VOP1 0x02) writes an SGPR even though its decoded dst looks like a VGPR.
    EXPECT_EQ(rdna2_vgpr_write_count(decode(0x7E000000u | (3u << 17) | (0x02u << 9) | 0x100u)), 0u);
    EXPECT_EQ(rdna2_vgpr_write_count(decode(0x7E000000u | (3u << 17) | (0x01u << 9) | 0x100u)), 1u);
    EXPECT_EQ(rdna2_vgpr_write_count(decode((0x03u << 25) | (3u << 17) | (1u << 9) | 0x100u)), 1u);
    EXPECT_EQ(rdna2_vgpr_write_count(Vop3p{.op = 0x0F}.decoded()), 1u);
    EXPECT_EQ(rdna2_vgpr_write_count(decode(kTop6Vintrp | (5u << 18))), 1u);
}

TEST(Rdna2DecodeSweep, WriteCountForVop3WideAndScalarResults) {
    const auto count = [](uint32_t op) {
        return rdna2_vgpr_write_count(decode(vop3_w0(op, 4u), vop3_srcs(256, 257, 258)));
    };
    EXPECT_EQ(count(kVop3Fma), 1u);
    EXPECT_EQ(count(0x360u), 0u) << "v_readlane_b32 writes an SGPR";
    for (uint32_t op : {0x16Eu, 0x176u, 0x177u, kVop3OpcodeLshlrevB64, kVop3OpcodeLshrrevB64}) {
        EXPECT_EQ(count(op), 2u) << "64-bit result pair, op=" << op;
    }
}

TEST(Rdna2DecodeSweep, WriteCountForMubufMatchesTheArchitecturalResultWidth) {
    // gfx10 numbering: format loads 0..3 (x..xyzw), format stores 4..7, ubyte..sshort 8..11, dword 12,
    // dwordx2 13, dwordx4 14, dwordx3 15 (x3 FOLLOWS x4), raw stores 0x1C..0x1F.
    for (uint32_t op = 0; op <= 3; ++op) EXPECT_EQ(rdna2_vgpr_write_count(mubuf(op)), op + 1) << op;
    for (uint32_t op = 4; op <= 7; ++op) EXPECT_EQ(rdna2_vgpr_write_count(mubuf(op)), 0u) << op;
    for (uint32_t op = 8; op <= 0xC; ++op) EXPECT_EQ(rdna2_vgpr_write_count(mubuf(op)), 1u) << op;
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(0x0D)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(0x0E)), 4u);
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(0x0F)), 3u);
    for (uint32_t op = 0x1C; op <= 0x1F; ++op) EXPECT_EQ(rdna2_vgpr_write_count(mubuf(op)), 0u) << op;
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(kMubufOpcodeAtomicSwapX2)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(kMubufOpcodeAtomicOrX2)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(kMubufOpcodeAtomicAdd)), 1u);
}

TEST(Rdna2DecodeSweep, WriteCountForMubufD16FormatLoadsPacksHalves) {
    // D16 format loads pack two 16-bit components per dword: x and xy fit one, xyz and xyzw need two.
    // Opcodes 0x80..0x83 = buffer_load_format_d16_x / _xy / _xyz / _xyzw.
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(0x80)), 1u);
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(0x81)), 1u);
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(0x82)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(mubuf(0x83)), 2u);
}

TEST(Rdna2DecodeSweep, WriteCountForMubufD16FormatStoresIsZero) {
    // buffer_store_format_d16_x .. _xyzw (0x84..0x87) read VDATA like the 0x04..0x07 format stores.
    for (uint32_t op = 0x84; op <= 0x87; ++op) {
        EXPECT_EQ(rdna2_vgpr_write_count(mubuf(op)), 0u) << op;
    }
    const uint32_t span[] = {1, 1, 2, 2};
    for (uint32_t k = 0; k < 4; ++k) {
        EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(0x84 + k)), span[k]) << "op=" << (0x84 + k);
    }
}

TEST(Rdna2DecodeSweep, WriteCountForMubufByteShortStoresIsZero) {
    // #4243: buffer_store_byte (0x18), its _d16_hi form (0x19), buffer_store_short (0x1A) and
    // its _d16_hi form (0x1B) read VDATA; they were falling through to `return 1`.
    for (uint32_t op = 0x18; op <= 0x1B; ++op) {
        EXPECT_EQ(rdna2_vgpr_write_count(mubuf(op)), 0u) << op;
        EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(op)), 1u) << "span op=" << op;
    }
}

TEST(Rdna2DecodeSweep, WriteCountForMtbufLoadsAndPackedD16Loads) {
    for (uint32_t op = 0; op <= 3; ++op) EXPECT_EQ(rdna2_vgpr_write_count(mtbuf(op)), op + 1) << op;
    for (uint32_t op = 4; op <= 7; ++op) EXPECT_EQ(rdna2_vgpr_write_count(mtbuf(op)), 0u) << op;
    // Packed D16 loads (8..11) hold two components per dword; D16 stores (12..15) read VDATA.
    const uint32_t d16[] = {1, 1, 2, 2};
    for (uint32_t k = 0; k < 4; ++k) {
        EXPECT_EQ(rdna2_vgpr_write_count(mtbuf(8 + k)), d16[k]) << "op=" << (8 + k);
        EXPECT_EQ(rdna2_vgpr_write_count(mtbuf(12 + k)), 0u) << "op=" << (12 + k);
    }
}

TEST(Rdna2DecodeSweep, WriteCountForFlatLoadsAndStores) {
    for (uint32_t op = 8; op <= 0xC; ++op) EXPECT_EQ(rdna2_vgpr_write_count(flat(op)), 1u) << op;
    EXPECT_EQ(rdna2_vgpr_write_count(flat(0x0D)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(flat(0x0E)), 4u);
    EXPECT_EQ(rdna2_vgpr_write_count(flat(0x0F)), 3u);
    for (uint32_t op = 0x18; op <= 0x1F; ++op) EXPECT_EQ(rdna2_vgpr_write_count(flat(op)), 0u) << op;
}

TEST(Rdna2DecodeSweep, WriteCountForFlatD16LoadsIsOne) {
    // #4243: FLAT D16 narrow loads write one VGPR (read-modify-write of the destination half).
    // 0x20 ubyte_d16, 0x21 ubyte_d16_hi, 0x22 sbyte_d16, 0x23 sbyte_d16_hi, 0x24 short_d16,
    // 0x25 short_d16_hi. Previously the FLAT arm returned 0 for every opcode outside 0x08-0x0F.
    for (uint32_t op = 0x20; op <= 0x25; ++op) {
        EXPECT_EQ(rdna2_vgpr_write_count(flat(op)), 1u) << op;
        EXPECT_EQ(rdna2_vgpr_destination_span(flat(op)), 1u) << "span op=" << op;
    }
}

TEST(Rdna2DecodeSweep, WriteCountForDsResultOpcodes) {
    // The DS opcodes whose VDST is a result: (opcode, dwords). Everything else here is a store or a
    // no-return atomic whose VDST field is a source.
    struct Row { uint32_t op; uint32_t dwords; };
    const Row rows[] = {
        {0x20, 1}, {0x2D, 1}, {0x35, 1}, {0x36, 1}, {0x3D, 1}, {0x3E, 1}, {0xB1, 1}, {0xB3, 1},
        {0x37, 2}, {0x38, 2}, {0x76, 2}, {0xFE, 3}, {0x77, 4}, {0xFF, 4},
    };
    for (const Row& r : rows) {
        EXPECT_EQ(rdna2_vgpr_write_count(decode(kTop6Ds | (r.op << 18), 4u << 24)), r.dwords)
            << "op=" << r.op;
    }
    EXPECT_EQ(rdna2_vgpr_write_count(decode(kTop6Ds | (0x0Du << 18), 4u << 24)), 0u)
        << "ds_write_b32 stores";
}

TEST(Rdna2DecodeSweep, WriteCountForMimgFollowsDmaskAndD16) {
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0x8)), 1u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0x5)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0x7)), 3u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0xF)), 4u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0x0)), 4u) << "an empty mask is accounted conservatively";
    // D16 packs two components per dword, rounded up.
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0xF, 1u << 31)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0x7, 1u << 31)), 2u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x00, 0x1, 1u << 31)), 1u);
    // IMAGE_STORE / IMAGE_STORE_MIP read VDATA.
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x08, 0xF)), 0u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x09, 0xF)), 0u);
}

TEST(Rdna2DecodeSweep, WriteCountForMimgGather4IsAlwaysFourTexelsWhateverTheDmask) {
    // IMAGE_GATHER4 selects ONE component through DMASK and returns that component of four texels, so
    // the result is four dwords (v[64:67] under dmask:0x1 in the gfx10 assembler tables). The whole
    // gather4 family occupies 0x40..0x5F: base, _cl, _l, _b, _b_cl, _lz, then each with _c and _o.
    for (uint32_t op = 0x40; op <= 0x5F; ++op) {
        for (uint32_t dmask : {0x1u, 0x2u, 0x4u, 0x8u}) {
            EXPECT_EQ(rdna2_vgpr_write_count(mimg(op, dmask)), 4u) << "op=" << op << " dmask=" << dmask;
        }
    }
    // D16 halves the footprint to two dwords.
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x47, 0x1, 1u << 31)), 2u);
    // Neighbours of the family keep the dmask rule.
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x3F, 0x1)), 1u);
    EXPECT_EQ(rdna2_vgpr_write_count(mimg(0x60, 0x1)), 1u);
}

TEST(Rdna2DecodeSweep, TfeStatusVgprFollowsTheDataRegisters) {
    // With TFE the fault/status dword is appended after the data results.
    EXPECT_EQ(rdna2_tfe_status_vgpr(mubuf(0x0C, 8u, 1u << 23)), 9);    // dword -> status at v9
    EXPECT_EQ(rdna2_tfe_status_vgpr(mubuf(0x0E, 8u, 1u << 23)), 12);   // dwordx4 -> v12
    EXPECT_EQ(rdna2_tfe_status_vgpr(mubuf(0x0C, 8u, 0u)), -1) << "no TFE, no status register";
    EXPECT_EQ(rdna2_tfe_status_vgpr(mubuf(0x1C, 8u, 1u << 23)), -1) << "a store has no data result";
    EXPECT_EQ(rdna2_tfe_status_vgpr(mimg(0x00, 0x7, 0u, 1u << 16)), 11);   // 3 comps from v8
    EXPECT_EQ(rdna2_tfe_status_vgpr(mimg(0x00, 0x7, 0u, 0u)), -1);
    EXPECT_EQ(rdna2_tfe_status_vgpr(mtbuf(1, 8u, true)), 10);          // x,y -> v10
    EXPECT_EQ(rdna2_tfe_status_vgpr(mtbuf(1, 8u, false)), -1);
    EXPECT_EQ(rdna2_tfe_status_vgpr(Rdna2Inst{}), -1);
    EXPECT_EQ(rdna2_tfe_status_vgpr(decode(0x7E000000u | (3u << 17) | (1u << 9) | 0x100u)), -1);
}

TEST(Rdna2DecodeSweep, TfeStatusAfterAGather4CountsAllFourTexels) {
    EXPECT_EQ(rdna2_tfe_status_vgpr(mimg(0x47, 0x1, 0u, 1u << 16)), 12);
    EXPECT_EQ(rdna2_tfe_status_vgpr(mimg(0x48, 0x2, 0u, 1u << 16)), 12);
}

TEST(Rdna2DecodeSweep, DestinationSpanCoversStoreSourcesAndTfeStatus) {
    // Loads: span = data width. Stores read VDATA, so their footprint is the data width too.
    EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(0x0E)), 4u);
    EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(0x1C)), 1u);
    EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(0x1D)), 2u);
    EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(0x1E)), 4u);
    EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(0x1F)), 3u);
    for (uint32_t op = 4; op <= 7; ++op) EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(op)), op - 3u) << op;
    EXPECT_EQ(rdna2_vgpr_destination_span(flat(0x1C)), 1u);
    EXPECT_EQ(rdna2_vgpr_destination_span(flat(0x1D)), 2u);
    EXPECT_EQ(rdna2_vgpr_destination_span(flat(0x1E)), 4u);
    EXPECT_EQ(rdna2_vgpr_destination_span(flat(0x1F)), 3u);
    EXPECT_EQ(rdna2_vgpr_destination_span(mimg(0x08, 0x7)), 3u);
    // TFE extends the span to include the status register.
    EXPECT_EQ(rdna2_vgpr_destination_span(mubuf(0x0E, 8u, 1u << 23)), 5u);
    EXPECT_EQ(rdna2_vgpr_destination_span(mimg(0x00, 0x3, 0u, 1u << 16)), 3u);
    EXPECT_EQ(rdna2_vgpr_destination_span(mtbuf(2, 8u, true)), 4u);
    EXPECT_EQ(rdna2_vgpr_destination_span(Rdna2Inst{}), 0u);
}

TEST(Rdna2DecodeSweep, SourceSpanIsOneDwordExceptTheWideCases) {
    const Rdna2Inst add = decode((0x03u << 25) | (3u << 17) | (4u << 9) | (256u + 2u));
    EXPECT_EQ(rdna2_vgpr_source_span(add, 0), 1u);
    EXPECT_EQ(rdna2_vgpr_source_span(add, 1), 1u);
    EXPECT_EQ(rdna2_vgpr_source_span(add, 2), 0u) << "past n_src";
    EXPECT_EQ(rdna2_vgpr_source_span(add, 99), 0u);
    // A source that is not a VGPR spans nothing.
    const Rdna2Inst scalar = decode((0x03u << 25) | (3u << 17) | (4u << 9) | 5u);
    EXPECT_EQ(rdna2_vgpr_source_span(scalar, 0), 0u);
    EXPECT_EQ(rdna2_vgpr_source_span(scalar, 1), 1u);
    // v_lshrrev_b64: the shift count is one dword, the value operand is a register PAIR.
    const Rdna2Inst shr = decode(vop3_w0(kVop3OpcodeLshrrevB64, 4u), vop3_srcs(256, 258, 0xC0u));
    EXPECT_EQ(rdna2_vgpr_source_span(shr, 0), 1u);
    EXPECT_EQ(rdna2_vgpr_source_span(shr, 1), 2u);
    // DS sources are a conservative four-dword range: read/write2 and wide packets encode a base only.
    const Rdna2Inst ds = decode(kTop6Ds | (0x0Du << 18), 1u | (2u << 8));
    EXPECT_EQ(rdna2_vgpr_source_span(ds, 0), 4u);
    EXPECT_EQ(rdna2_vgpr_source_span(ds, 1), 4u);
}

// ---- rdna2_sload_required_bytes --------------------------------------------------------------
// s_load_dword[xN] with a pointer-backed SBASE pair and an immediate offset (SOFFSET = NULL, 125).

namespace {
constexpr uint32_t smem_w0(uint32_t op, uint32_t sdata, uint32_t sbase_pair) {
    return kTop6Smem | (op << 18) | (sdata << 6) | sbase_pair;
}
constexpr uint32_t smem_w1(uint32_t off, uint32_t soffset = 125u) { return off | (soffset << 25); }
constexpr uint32_t kEnd = 0xBF810000u;
}  // namespace

TEST(Rdna2DecodeSweep, SloadRequiredBytesIsTheHighestImmediateReach) {
    // SBASE pair index 2 = s[4:5]. dword (4 B), x2 (8), x4 (16), x8 (32), x16 (64).
    const uint32_t widths[] = {4, 8, 16, 32, 64};
    for (uint32_t op = 0; op <= 4; ++op) {
        const uint32_t code[] = {smem_w0(op, 0, 2), smem_w1(0x20), kEnd};
        EXPECT_EQ(rdna2_sload_required_bytes(code, 3, 4), 0x20u + widths[op]) << "op=" << op;
    }
}

TEST(Rdna2DecodeSweep, SloadRequiredBytesTakesTheMaximumOverAllLoads) {
    const uint32_t code[] = {
        smem_w0(2, 0, 2), smem_w1(0x40),    // x4 at 0x40 -> reaches 0x50
        smem_w0(0, 4, 2), smem_w1(0x100),   // dword at 0x100 -> reaches 0x104
        smem_w0(1, 8, 2), smem_w1(0x10),    // x2 at 0x10 -> reaches 0x18
        kEnd,
    };
    EXPECT_EQ(rdna2_sload_required_bytes(code, 7, 4), 0x104u);
}

TEST(Rdna2DecodeSweep, SloadRequiredBytesIgnoresLoadsThroughOtherBasesAndUnboundedForms) {
    const uint32_t other_base[] = {smem_w0(0, 0, 3), smem_w1(0x80), kEnd};      // s[6:7]
    EXPECT_EQ(rdna2_sload_required_bytes(other_base, 3, 4), 0u);
    const uint32_t reg_offset[] = {smem_w0(0, 0, 2), smem_w1(0x80, 7u), kEnd};  // SOFFSET = s7
    EXPECT_EQ(rdna2_sload_required_bytes(reg_offset, 3, 4), 0u) << "a register offset has no static reach";
    const uint32_t negative[] = {smem_w0(0, 0, 2), smem_w1(0x1FFFF0u), kEnd};   // -16
    EXPECT_EQ(rdna2_sload_required_bytes(negative, 3, 4), 0u);
    const uint32_t buffer_load[] = {smem_w0(8, 0, 2), smem_w1(0x80), kEnd};     // s_buffer_load_dword
    EXPECT_EQ(rdna2_sload_required_bytes(buffer_load, 3, 4), 0u) << "bounded by its V#, not a pointer";
    EXPECT_EQ(rdna2_sload_required_bytes(other_base, 0, 4), 0u);
}

TEST(Rdna2DecodeSweep, SloadRequiredBytesStopsAtEndpgm) {
    const uint32_t code[] = {smem_w0(0, 0, 2), smem_w1(0x10), kEnd, smem_w0(0, 0, 2), smem_w1(0x1000)};
    EXPECT_EQ(rdna2_sload_required_bytes(code, 5, 4), 0x14u);
}

TEST(Rdna2DecodeSweep, SloadRequiredBytesSaturatesInsteadOfWrapping) {
    // The largest 21-bit immediate plus a 64-byte load fits in 32 bits, so this documents the cap
    // path's normal range rather than a wrap: the result must still equal offset + width exactly.
    const uint32_t code[] = {smem_w0(4, 0, 2), smem_w1(0x0FFFFFu), kEnd};
    EXPECT_EQ(rdna2_sload_required_bytes(code, 3, 4), 0x0FFFFFu + 64u);
}
