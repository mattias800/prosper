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
