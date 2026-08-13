// test_rdna2_decode — validates the RDNA2 instruction-stream walker (src/gpu/rdna2_decode.cpp)
// against instructions assembled by llvm-mc for gfx1030 (authoritative encodings, not hand-rolled).
// The stream mixes every major encoding class + inline literals + S_ENDPGM; the walker must classify
// each instruction's format, compute its length (incl. literals), and terminate at S_ENDPGM.
#include "../src/gpu/rdna2_decode.hpp"
#include <array>
#include <cstdio>
#include <cstdint>
#include <utility>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_rdna2_decode ==\n");
    // Assembled with: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1030 --show-encoding
    //   s_mov_b32 s0,s1 | s_mov_b32 s2,0x12345678 | s_add_u32 s0,s1,s2 | v_mov_b32 v0,v1 |
    //   v_mov_b32 v2,0x12345678 | v_add_f32 v0,v1,v2 | v_fma_f32 v0,v1,v2,v3 |
    //   s_load_dwordx4 s[0:3],s[4:5],0x0 | exp mrt0 v0,v1,v2,v3 | s_endpgm
    const uint32_t code[] = {
        0xBE800301u,                     // SOP1  s_mov_b32 s0,s1
        0xBE8203FFu, 0x12345678u,        // SOP1  s_mov_b32 s2,lit   (+literal)
        0x80000201u,                     // SOP2  s_add_u32
        0x7E000301u,                     // VOP1  v_mov_b32 v0,v1
        0x7E0402FFu, 0x12345678u,        // VOP1  v_mov_b32 v2,lit   (+literal)
        0x06000501u,                     // VOP2  v_add_f32
        0xD54B0000u, 0x040E0501u,        // VOP3  v_fma_f32
        0xF4080002u, 0xFA000000u,        // SMEM  s_load_dwordx4
        0xF800000Fu, 0x03020100u,        // EXP   exp mrt0
        0xBF810000u,                     // SOPP  s_endpgm
    };
    const size_t n = sizeof(code) / sizeof(code[0]);

    std::vector<Rdna2Inst> ins;
    size_t consumed = rdna2_walk(code, n, ins);
    CHECK(consumed == n, "walker consumed the whole stream (15 dwords)");
    CHECK(ins.size() == 10, "decoded 10 instructions");
    if (ins.size() != 10) { printf("== FAIL: got %zu ==\n", ins.size()); return 1; }

    struct Exp { Rdna2Format fmt; uint32_t pc; uint32_t len; bool lit; };
    const Exp exp[] = {
        {Rdna2Format::SOP1, 0,  1, false},
        {Rdna2Format::SOP1, 1,  2, true },
        {Rdna2Format::SOP2, 3,  1, false},
        {Rdna2Format::VOP1, 4,  1, false},
        {Rdna2Format::VOP1, 5,  2, true },
        {Rdna2Format::VOP2, 7,  1, false},
        {Rdna2Format::VOP3, 8,  2, false},
        {Rdna2Format::SMEM, 10, 2, false},
        {Rdna2Format::EXP,  12, 2, false},
        {Rdna2Format::SOPP, 14, 1, false},
    };
    const char* names[] = {"SOP1","SOP1+lit","SOP2","VOP1","VOP1+lit","VOP2","VOP3","SMEM","EXP","SOPP"};
    for (size_t k = 0; k < 10; k++) {
        char msg[64]; snprintf(msg, sizeof msg, "inst %zu = %s (pc=%u len=%u)", k, names[k], exp[k].pc, exp[k].len);
        bool ok = ins[k].fmt == exp[k].fmt && ins[k].pc == exp[k].pc &&
                  ins[k].len_dwords == exp[k].len && ins[k].has_literal == exp[k].lit;
        CHECK(ok, msg);
    }
    CHECK(ins[1].literal == 0x12345678u && ins[4].literal == 0x12345678u, "inline literals captured");
    CHECK(ins[9].is_end, "S_ENDPGM flagged as end");

    auto isS = [](const Operand& o, int n){ return o.kind == OperandKind::SGPR && o.value == n; };
    auto isV = [](const Operand& o, int n){ return o.kind == OperandKind::VGPR && o.value == n; };

    // --- EXP decode (export target + enable + 4 VGPR sources) ---
    // Assembled: exp mrt0 v0,v1,v2,v3 | exp pos0 v4,v5,v6,v7 | exp param0 v8,v9,v10,v11
    const uint32_t mrt0[]  = {0xF800000Fu, 0x03020100u};
    const uint32_t pos0[]  = {0xF80000CFu, 0x07060504u};
    const uint32_t par0[]  = {0xF800020Fu, 0x0B0A0908u};
    Rdna2Inst e0 = rdna2_decode_one(mrt0, 2), e1 = rdna2_decode_one(pos0, 2), e2 = rdna2_decode_one(par0, 2);
    CHECK(e0.fmt == Rdna2Format::EXP && e0.exp_target == 0 && e0.exp_en == 0xF &&
          isV(e0.src[0],0) && isV(e0.src[1],1) && isV(e0.src[2],2) && isV(e0.src[3],3), "EXP mrt0 v0..v3");
    CHECK(e1.fmt == Rdna2Format::EXP && e1.exp_target == 12 &&
          isV(e1.src[0],4) && isV(e1.src[3],7), "EXP pos0 (target 12) v4..v7");
    CHECK(e2.fmt == Rdna2Format::EXP && e2.exp_target == 32 &&
          isV(e2.src[0],8) && isV(e2.src[3],11), "EXP param0 (target 32) v8..v11");

    // --- operand decode (stage 2) ---
    // inst0: s_mov_b32 s0, s1  -> dst SGPR0, src0 SGPR1
    CHECK(isS(ins[0].dst, 0) && ins[0].n_src == 1 && isS(ins[0].src[0], 1), "s_mov_b32 s0,s1 operands");
    // inst2: s_add_u32 s0, s1, s2 -> dst SGPR0, src0 SGPR1, src1 SGPR2
    CHECK(isS(ins[2].dst, 0) && isS(ins[2].src[0], 1) && isS(ins[2].src[1], 2), "s_add_u32 s0,s1,s2 operands");
    // inst3: v_mov_b32 v0, v1 -> dst VGPR0, src0 VGPR1
    CHECK(isV(ins[3].dst, 0) && isV(ins[3].src[0], 1), "v_mov_b32 v0,v1 operands");
    // inst4: v_mov_b32 v2, lit -> dst VGPR2, src0 Literal
    CHECK(isV(ins[4].dst, 2) && ins[4].src[0].kind == OperandKind::Literal, "v_mov_b32 v2,lit operands");
    // inst5: v_add_f32 v0, v1, v2 -> dst VGPR0, src0 VGPR1, src1 VGPR2
    CHECK(isV(ins[5].dst, 0) && isV(ins[5].src[0], 1) && isV(ins[5].src[1], 2), "v_add_f32 v0,v1,v2 operands");
    // inst6: v_fma_f32 v0, v1, v2, v3 (VOP3) -> dst VGPR0, srcs VGPR1/2/3
    CHECK(isV(ins[6].dst, 0) && ins[6].n_src == 3 && isV(ins[6].src[0], 1) &&
          isV(ins[6].src[1], 2) && isV(ins[6].src[2], 3), "v_fma_f32 VOP3 operands");
    // GTA V exec_cs_413d1bf00 pc458, exact llvm-mc gfx1010 packet:
    // `v_ldexp_f32 v0, v13, v1`. It is a two-source VOP3A op; the zeroed reserved SRC2 field must
    // not escape as a phantom s0 dependency. Keep modifier decoding visible so the bounded emitter
    // can reject unproved forms instead of silently dropping them.
    const uint32_t gta_ldexp_words[] = {0xd7620000u, 0x0002030du};
    const Rdna2Inst gta_ldexp = rdna2_decode_one(gta_ldexp_words, 2);
    CHECK(gta_ldexp.fmt == Rdna2Format::VOP3 && gta_ldexp.opcode == 0x362u &&
          gta_ldexp.n_src == 2 && isV(gta_ldexp.dst, 0) &&
          isV(gta_ldexp.src[0], 13) && isV(gta_ldexp.src[1], 1) &&
          gta_ldexp.src[2].kind == OperandKind::None &&
          !gta_ldexp.clamp && gta_ldexp.omod == 0 &&
          !gta_ldexp.src_abs[0] && !gta_ldexp.src_abs[1] &&
          !gta_ldexp.src_neg[0] && !gta_ldexp.src_neg[1],
          "GTA V v_ldexp_f32 decodes two sources and no phantom s0/modifiers");
    const uint32_t gta_ldexp_modifier_words[] = {
        0xd7628100u, 0x2802030du, // src0 ABS, CLAMP, src0 NEG, OMOD x2
    };
    const Rdna2Inst gta_ldexp_modifier = rdna2_decode_one(gta_ldexp_modifier_words, 2);
    CHECK(gta_ldexp_modifier.src_abs[0] && gta_ldexp_modifier.src_neg[0] &&
          gta_ldexp_modifier.clamp && gta_ldexp_modifier.omod == 1,
          "v_ldexp_f32 modifier bits remain visible to the fail-closed emitter");
    // GTA V exec_cs_413d22d00 pc605: `v_bfm_b32 v22, v22, 0`. V_BFM is also a
    // two-source VOP3A instruction; its reserved zero SRC2 field must not become a phantom s0.
    const uint32_t gta_bfm_words[] = {0xd7630016u, 0x00010116u};
    const Rdna2Inst gta_bfm = rdna2_decode_one(gta_bfm_words, 2);
    CHECK(gta_bfm.fmt == Rdna2Format::VOP3 && gta_bfm.opcode == 0x363u &&
          gta_bfm.n_src == 2 && isV(gta_bfm.dst, 22) &&
          isV(gta_bfm.src[0], 22) &&
          gta_bfm.src[1].kind == OperandKind::InlineInt && gta_bfm.src[1].value == 0 &&
          gta_bfm.src[2].kind == OperandKind::None,
          "GTA V v_bfm_b32 decodes two sources and no phantom s0");
    // GTA V exec_cs_205b658800 pc2546: `v_mac_f32_e64 v15, -s31, v2`. V_MAC reads
    // SRC0/SRC1 and its old VDST; the encoded SRC2 field is reserved and must not invent s0.
    const uint32_t gta_vmac_words[] = {0xd51f000fu, 0x2002041fu};
    const Rdna2Inst gta_vmac = rdna2_decode_one(gta_vmac_words, 2);
    CHECK(gta_vmac.fmt == Rdna2Format::VOP3 && gta_vmac.opcode == 0x11fu &&
          gta_vmac.n_src == 2 && isV(gta_vmac.dst, 15) &&
          isS(gta_vmac.src[0], 31) && isV(gta_vmac.src[1], 2) &&
          gta_vmac.src[2].kind == OperandKind::None && gta_vmac.src_neg[0] &&
          !gta_vmac.src_neg[1],
          "GTA V v_mac_f32 decodes two sources and no phantom s0");
    // GTA V exec_cs_205b67ce00 pc576: `v_mul_lo_u32 v1, v2, v1`. The encoded SRC2 field is
    // reserved. Mutating only this packet's opcode to v_mad_u32_u24 makes the exact same field a
    // real third source, pinning the arity decision to the production site.
    const uint32_t gta_mul_lo_words[] = {0xd5690001u, 0x00020302u};
    const Rdna2Inst gta_mul_lo = rdna2_decode_one(gta_mul_lo_words, 2);
    CHECK(gta_mul_lo.fmt == Rdna2Format::VOP3 &&
          gta_mul_lo.opcode == kVop3OpcodeMulLoU32 && gta_mul_lo.n_src == 2 &&
          isV(gta_mul_lo.dst, 1) && isV(gta_mul_lo.src[0], 2) &&
          isV(gta_mul_lo.src[1], 1) && gta_mul_lo.src[2].kind == OperandKind::None,
          "GTA V v_mul_lo_u32 decodes two sources and no phantom s0");
    const uint32_t gta_mul_lo_mutation_words[] = {0xd5430001u, 0x00020302u};
    const Rdna2Inst gta_mul_lo_mutation =
        rdna2_decode_one(gta_mul_lo_mutation_words, 2);
    CHECK(gta_mul_lo_mutation.opcode == kVop3OpcodeMadU32U24 &&
          gta_mul_lo_mutation.n_src == 3 && isS(gta_mul_lo_mutation.src[2], 0),
          "same-site three-source multiply mutation retains its real s0 dependency");
    // GTA V exec_cs_205b658800 pc61: `v_lshlrev_b64 v[24:25], v4, 1`. The reserved SRC2 field
    // must not surface as s0, and the shared writer inventory must retain both result halves.
    const uint32_t gta_lshlrev_b64_words[] = {0xd6ff0018u, 0x00010304u};
    const Rdna2Inst gta_lshlrev_b64 = rdna2_decode_one(gta_lshlrev_b64_words, 2);
    CHECK(gta_lshlrev_b64.fmt == Rdna2Format::VOP3 &&
          gta_lshlrev_b64.opcode == kVop3OpcodeLshlrevB64 && gta_lshlrev_b64.n_src == 2 &&
          isV(gta_lshlrev_b64.dst, 24) && isV(gta_lshlrev_b64.src[0], 4) &&
          gta_lshlrev_b64.src[1].kind == OperandKind::InlineInt &&
          gta_lshlrev_b64.src[1].value == 1 &&
          gta_lshlrev_b64.src[2].kind == OperandKind::None &&
          rdna2_vgpr_write_count(gta_lshlrev_b64) == 2u &&
          rdna2_vgpr_destination_span(gta_lshlrev_b64) == 2u,
          "GTA V v_lshlrev_b64 decodes two sources and a two-VGPR destination");
    // GTA V exec_cs_413e1ac00 pc59: `v_lshrrev_b64 v[1:2], s2, v[5:6]`. Both the
    // destination and SRC1 name pairs, while the reserved SRC2 field must not invent an s0 read.
    const uint32_t gta_lshrrev_b64_words[] = {0xd7000001u, 0x00020a02u};
    const Rdna2Inst gta_lshrrev_b64 = rdna2_decode_one(gta_lshrrev_b64_words, 2);
    CHECK(gta_lshrrev_b64.fmt == Rdna2Format::VOP3 &&
          gta_lshrrev_b64.opcode == kVop3OpcodeLshrrevB64 &&
          gta_lshrrev_b64.n_src == 2 && isV(gta_lshrrev_b64.dst, 1) &&
          isS(gta_lshrrev_b64.src[0], 2) && isV(gta_lshrrev_b64.src[1], 5) &&
          gta_lshrrev_b64.src[2].kind == OperandKind::None &&
          rdna2_vgpr_source_span(gta_lshrrev_b64, 0) == 0u &&
          rdna2_vgpr_source_span(gta_lshrrev_b64, 1) == 2u &&
          rdna2_vgpr_write_count(gta_lshrrev_b64) == 2u &&
          rdna2_vgpr_destination_span(gta_lshrrev_b64) == 2u,
          "GTA V v_lshrrev_b64 decodes exact B32/B64 sources and a two-VGPR destination");
    // GTA V exec_cs_205b54f200 pc21, exact llvm-mc gfx1030 packet:
    // `v_cvt_rpi_i32_f32_e32 v1, v1`. Keep the plain one-dword form distinct from the
    // modifier encodings that the emitter intentionally leaves unsupported.
    const uint32_t gta_cvt_rpi_words[] = {0x7e021901u};
    const Rdna2Inst gta_cvt_rpi = rdna2_decode_one(gta_cvt_rpi_words, 1);
    CHECK(gta_cvt_rpi.fmt == Rdna2Format::VOP1 && gta_cvt_rpi.len_dwords == 1u &&
          gta_cvt_rpi.opcode == 0x0cu && isV(gta_cvt_rpi.dst, 1) &&
          gta_cvt_rpi.n_src == 1u && isV(gta_cvt_rpi.src[0], 1) &&
          !gta_cvt_rpi.has_literal && !gta_cvt_rpi.has_modifier &&
          !gta_cvt_rpi.has_sdwa && !gta_cvt_rpi.has_dpp,
          "GTA V v_cvt_rpi_i32_f32 decodes exact plain VOP1 operands");
    // inst7: s_load_dwordx4 s[0:3], s[4:5], 0x0 (SMEM) -> op 0x2, SDATA s0, SBASE s4 (pair), offset 0
    CHECK(ins[7].fmt == Rdna2Format::SMEM && ins[7].opcode == 0x2u && isS(ins[7].dst, 0) &&
          isS(ins[7].src[0], 4) && ins[7].literal == 0x0u, "s_load_dwordx4 SMEM op/SDATA/SBASE/offset");
    // MUBUF decode: buffer_load_dwordx4 v[4:7], v2, s[8:11], 0 offen -> op 0xe, VDATA v4, VADDR v2,
    // SRSRC s8 (×4), offen bit set in `literal`. SOFFSET field is 0x80 = inline integer 0 (llvm-mc's
    // encoding of literal `0`) — it must NOT decode as SGPR s0 (8-bit field, not 7).
    const uint32_t mubuf[] = { 0xe0381000u, 0x80020402u };
    Rdna2Inst mb = rdna2_decode_one(mubuf, 2);
    CHECK(mb.fmt == Rdna2Format::MUBUF && mb.opcode == 0xeu && isV(mb.dst, 4) && isV(mb.src[0], 2) &&
          isS(mb.src[1], 8) && ((mb.literal >> 12) & 1u), "buffer_load_dwordx4 MUBUF op/VDATA/VADDR/SRSRC/offen");
    CHECK(mb.src[2].kind == OperandKind::InlineInt && mb.src[2].value == 0,
          "MUBUF SOFFSET 0x80 decodes as inline 0, not SGPR s0");
    // gfx1030 llvm-mc: buffer_load_dword v[0:1], v22, s[4:7], 0 offen tfe.
    const uint32_t mubuf_tfe_words[] = {0xe0301000u, 0x80810016u};
    const Rdna2Inst mubuf_tfe = rdna2_decode_one(mubuf_tfe_words, 2);
    CHECK(mubuf_tfe.fmt == Rdna2Format::MUBUF && mubuf_tfe.mubuf_tfe &&
              rdna2_vgpr_write_count(mubuf_tfe) == 1u &&
              rdna2_tfe_status_vgpr(mubuf_tfe) == 1 &&
              rdna2_vgpr_destination_span(mubuf_tfe) == 2u,
          "MUBUF TFE decodes and appends its status VGPR after the load result");
    // Astro Bot world-map PS, exact final packet: buffer_store_dwordx3
    // v[3:5], v10, s[16:19], 0 idxen. Raw x3 stores use opcode 0x1f, after x4.
    const uint32_t astro_store_x3[] = { 0xe07c2000u, 0x8004030au };
    Rdna2Inst store_x3 = rdna2_decode_one(astro_store_x3, 2);
    CHECK(store_x3.fmt == Rdna2Format::MUBUF && store_x3.opcode == 0x1fu &&
              isV(store_x3.dst, 3) && isV(store_x3.src[0], 10) &&
              isS(store_x3.src[1], 16) && ((store_x3.literal >> 13) & 1u),
          "Astro buffer_store_dwordx3 decodes VDATA/VADDR/SRSRC and IDXEN exactly");
    // gfx1030 llvm-mc: tbuffer_load_format_xy v[4:5], v2, s[8:11], s12
    // format:[BUF_FMT_16_16_FLOAT] offen offset:52. MTBUF uses the Gen5 combined format 29,
    // not the old split DFMT=13/NFMT=1 interpretation of those same seven bits.
    const uint32_t mtbuf_load[] = { 0xe8e91034u, 0x0c020402u };
    Rdna2Inst mt_load = rdna2_decode_one(mtbuf_load, 2);
    CHECK(mt_load.fmt == Rdna2Format::MTBUF && mt_load.opcode == 1u &&
              mt_load.mtbuf_format == 29u && isV(mt_load.dst, 4) &&
              isV(mt_load.src[0], 2) && isS(mt_load.src[1], 8) &&
              isS(mt_load.src[2], 12) && (mt_load.literal & 0xfffu) == 52u &&
              ((mt_load.literal >> 12) & 1u) && !((mt_load.literal >> 13) & 1u),
          "MTBUF load decodes opcode, combined format, operands, offset, and OFFEN");
    // gfx1030 llvm-mc: tbuffer_store_format_xyzw v[32:35], v8, s[32:35], 0
    // format:[BUF_FMT_8_8_8_8_UINT] idxen.
    const uint32_t mtbuf_store[] = { 0xe9e72000u, 0x80082008u };
    Rdna2Inst mt_store = rdna2_decode_one(mtbuf_store, 2);
    CHECK(mt_store.fmt == Rdna2Format::MTBUF && mt_store.opcode == 7u &&
              mt_store.mtbuf_format == 60u && isV(mt_store.dst, 32) &&
              isV(mt_store.src[0], 8) && isS(mt_store.src[1], 32) &&
              mt_store.src[2].kind == OperandKind::InlineInt &&
              mt_store.src[2].value == 0 && ((mt_store.literal >> 13) & 1u),
          "MTBUF store decodes opcode, combined format, operands, and IDXEN");
    // gfx1030 llvm-mc: tbuffer_load_format_d16_x v0, v0, s[8:11], 0
    // format:[BUF_FMT_16_FLOAT] idxen. OP[3] lives in dword1 bit 21 on gfx10.
    const uint32_t mtbuf_d16[] = { 0xe8682000u, 0x80220000u };
    Rdna2Inst mt_d16 = rdna2_decode_one(mtbuf_d16, 2);
    CHECK(mt_d16.fmt == Rdna2Format::MTBUF && mt_d16.opcode == 8u &&
              mt_d16.mtbuf_format == 13u,
          "MTBUF decodes the split high opcode bit instead of aliasing D16 onto ordinary loads");
    const uint32_t mtbuf_tfe[] = { 0xe8b02000u, 0x80820100u };
    Rdna2Inst mt_tfe = rdna2_decode_one(mtbuf_tfe, 2);
    CHECK(mt_tfe.fmt == Rdna2Format::MTBUF && mt_tfe.opcode == 0u &&
              mt_tfe.mtbuf_format == 22u && mt_tfe.mtbuf_tfe,
          "MTBUF decodes TFE instead of silently dropping its status destination");
    // gfx1030 llvm-mc exact words. FLAT's dword1 places SADDR and VDST differently from MUBUF;
    // scratch `off, sN` also uses a real absent VADDR, not an accidental read of v0.
    const uint32_t scratch_load_x2[] = { 0xdc344020u, 0x08060000u };
    Rdna2Inst scratch_l2 = rdna2_decode_one(scratch_load_x2, 2);
    CHECK(scratch_l2.fmt == Rdna2Format::FLAT && scratch_l2.flat_segment == 1u &&
              scratch_l2.opcode == 0x0du && scratch_l2.literal == 32u &&
              isV(scratch_l2.dst, 8) && scratch_l2.src[0].kind == OperandKind::None &&
              isS(scratch_l2.src[1], 6),
          "scratch_load_dwordx2 decodes segment, signed offset, VDST, absent VADDR, and SADDR");
    const uint32_t scratch_store_x2[] = { 0xdc744020u, 0x00060a00u };
    Rdna2Inst scratch_s2 = rdna2_decode_one(scratch_store_x2, 2);
    CHECK(scratch_s2.fmt == Rdna2Format::FLAT && scratch_s2.flat_segment == 1u &&
              scratch_s2.opcode == 0x1du && isV(scratch_s2.dst, 10) &&
              scratch_s2.src[0].kind == OperandKind::None && isS(scratch_s2.src[1], 6),
          "scratch_store_dwordx2 decodes VDATA rather than the load-only VDST field");
    const uint32_t scratch_negative[] = { 0xdc304ffcu, 0x00000000u };
    Rdna2Inst scratch_neg = rdna2_decode_one(scratch_negative, 2);
    CHECK(scratch_neg.flat_segment == 1u && static_cast<int32_t>(scratch_neg.literal) == -4,
          "scratch signed 12-bit OFFSET is sign-extended");
    const uint32_t scratch_lds[] = { 0xdc306000u, 0x00000000u };
    Rdna2Inst scratch_to_lds = rdna2_decode_one(scratch_lds, 2);
    CHECK(scratch_to_lds.flat_segment == 1u && scratch_to_lds.flat_lds,
          "scratch LDS-transfer flag is retained for fail-closed recompilation");
    const uint32_t global_load[] = { 0xdc308000u, 0x007d0002u };
    const uint32_t flat_load[]   = { 0xdc300000u, 0x007d0002u };
    Rdna2Inst global_l = rdna2_decode_one(global_load, 2);
    Rdna2Inst flat_l = rdna2_decode_one(flat_load, 2);
    CHECK(global_l.flat_segment == 2u && isV(global_l.src[0], 2) &&
              global_l.src[1].kind == OperandKind::Special && global_l.src[1].value == 125,
          "global_load_dword retains its VGPR address and NULL SADDR for fail-closed diagnostics");
    CHECK(flat_l.flat_segment == 0u && isV(flat_l.src[0], 2),
          "flat_load_dword remains distinct from global and scratch segments");
    // Exact DS_READ2_B32 words from Astro Bot's loading-surface compute producer. The packed
    // offset bytes are dword indices (offset0 in the low byte, offset1 in the high byte).
    const uint32_t ds_read2_adjacent[] = { 0xd8dc0100u, 0x04000002u };
    Rdna2Inst dr2a = rdna2_decode_one(ds_read2_adjacent, 2);
    CHECK(dr2a.fmt == Rdna2Format::DS && dr2a.opcode == 0x37u && dr2a.literal == 0x0100u &&
          isV(dr2a.src[0], 2) && isV(dr2a.dst, 4),
          "Astro DS_READ2_B32 decodes adjacent offsets, ADDR v2, and VDST v4");
    const uint32_t ds_read2_16_17[] = { 0xd8dc1110u, 0x02000002u };
    Rdna2Inst dr2b = rdna2_decode_one(ds_read2_16_17, 2);
    CHECK(dr2b.fmt == Rdna2Format::DS && dr2b.opcode == 0x37u && dr2b.literal == 0x1110u &&
          isV(dr2b.src[0], 2) && isV(dr2b.dst, 2),
          "Astro DS_READ2_B32 decodes non-zero offsets 16/17");
    // VOP SDWA/DPP forms carry a mandatory 2nd (control) dword — the decoder must count it, or the
    // whole downstream stream mis-aligns. Encodings from llvm-mc gfx1030: SDWA src0=0xf9,
    // DPP16 src0=0xfa, DPP8 src0=0xe9.
    // `v_lshlrev_b32_sdwa v0, 2, v4 dst_sel:DWORD dst_unused:UNUSED_PRESERVE src1_sel:BYTE_0`: a
    // DWORD destination has no unused bits, so UNUSED_PRESERVE is a no-op there and the integer
    // SDWA subset now admits it (#2013). It used to reject only because the admission required
    // UNUSED_PAD exactly. The LENGTH assertion is the part that must never change.
    const uint32_t sdwa[] = { 0x340008f9u, 0x00861682u };
    Rdna2Inst sd = rdna2_decode_one(sdwa, 2);
    CHECK(sd.fmt == Rdna2Format::VOP2 && sd.len_dwords == 2 && !sd.has_modifier &&
          sd.sdwa_dst_sel == 6u && sd.sdwa_dst_unused == 2u && sd.sdwa_src1_sel == 0u,
          "VOP2 SDWA form is 2 dwords and a DWORD destination with UNUSED_PRESERVE is admitted");
    // The same instruction with UNUSED_SEXT still rejects: sign-extending the unwritten bits is a
    // distinct semantic the emitter does not model, so it must stay fail-visible.
    const uint32_t sdwa_sext[] = { 0x340008f9u, 0x00860e82u };
    Rdna2Inst sds = rdna2_decode_one(sdwa_sext, 2);
    CHECK(sds.fmt == Rdna2Format::VOP2 && sds.len_dwords == 2 && sds.has_modifier,
          "VOP2 SDWA with UNUSED_SEXT is 2 dwords and still flagged has_modifier");
    const uint32_t ngg_byte_shift[] = { 0x340018f9u, 0x02860682u };
    Rdna2Inst nbs = rdna2_decode_one(ngg_byte_shift, 2);
    CHECK(nbs.fmt == Rdna2Format::VOP2 && nbs.opcode == 0x1Au && !nbs.has_modifier &&
          isV(nbs.dst, 0) && nbs.src[0].kind == OperandKind::InlineInt &&
          nbs.src[0].value == 2 && isV(nbs.src[1], 12) && nbs.sdwa_src1_sel == 2u,
          "Astro NGG byte-select v_lshlrev SDWA packet is admitted exactly");
    const uint32_t ngg_word_shift[] = { 0x34001ef9u, 0x05860682u };
    Rdna2Inst nws = rdna2_decode_one(ngg_word_shift, 2);
    CHECK(nws.fmt == Rdna2Format::VOP2 && nws.opcode == 0x1Au && !nws.has_modifier &&
          isV(nws.src[1], 15) && nws.sdwa_src1_sel == 5u,
          "Astro NGG word-select v_lshlrev SDWA packet is admitted exactly");
    const uint32_t f16cmp[] = { 0x7db900f9u, 0x86050007u };
    Rdna2Inst fc = rdna2_decode_one(f16cmp, 2);
    CHECK(fc.fmt == Rdna2Format::VOPC && fc.opcode == 0xDCu && !fc.has_modifier &&
          fc.sdwa_src0_sel == 5u && fc.sdwa_src1_sel == 6u,
          "VOPC f16 SDWA WORD_1 source select is decoded for recompilation");

    // #2013: SDWA S0_SEXT / S1_SEXT (dword1 bits 19 and 27) — sign-extending the selected sub-dword
    // source is a DIFFERENT operation from zero-extending it, so the bit must reach the recompiler
    // rather than be folded into the select. Asserted at the DECODE level because the execution
    // kernels in test_rdna2_to_spirv observe only the composed result: a decoder that dropped the
    // flag and a lowering that ignored it are indistinguishable there.
    // Sonic Racing: CrossWorlds, `v_mov_b32_sdwa v14, sext(v8) dst_sel:DWORD dst_unused:UNUSED_PAD
    // src0_sel:WORD_0` and the same instruction with the SEXT bit cleared.
    const uint32_t mov_sext[] = { 0x7e1c02f9u, 0x000c0608u };
    Rdna2Inst ms = rdna2_decode_one(mov_sext, 2);
    CHECK(ms.fmt == Rdna2Format::VOP1 && ms.opcode == 0x01u && !ms.has_modifier &&
          ms.sdwa_dst_sel == 6u && ms.sdwa_src0_sel == 4u && ms.sdwa_src0_sext,
          "VOP1 v_mov_b32_sdwa carries S0_SEXT alongside its WORD source select");
    const uint32_t mov_zext[] = { 0x7e1c02f9u, 0x00040608u };
    Rdna2Inst mz = rdna2_decode_one(mov_zext, 2);
    CHECK(mz.fmt == Rdna2Format::VOP1 && mz.opcode == 0x01u && !mz.has_modifier &&
          mz.sdwa_src0_sel == 4u && !mz.sdwa_src0_sext,
          "the same encoding without bit 19 decodes as the zero-extending form");
    // GTA V compute compaction: reverse the selected low word into a full dword. Both retained
    // kernels use the same WORD_0 + zero-extension shape, differing only in their VGPR number.
    const uint32_t bfrev6[] = { 0x7e0c70f9u, 0x00040606u };
    Rdna2Inst br6 = rdna2_decode_one(bfrev6, 2);
    CHECK(br6.fmt == Rdna2Format::VOP1 && br6.opcode == 0x38u && !br6.has_modifier &&
          br6.has_sdwa && isV(br6.dst, 6) && isV(br6.src[0], 6) &&
          br6.sdwa_dst_sel == 6u && br6.sdwa_dst_unused == 0u &&
          br6.sdwa_src0_sel == 4u,
          "GTA V v_bfrev_b32_sdwa v6 WORD_0 packet is admitted exactly");
    const uint32_t bfrev0[] = { 0x7e0070f9u, 0x00040600u };
    Rdna2Inst br0 = rdna2_decode_one(bfrev0, 2);
    CHECK(br0.fmt == Rdna2Format::VOP1 && br0.opcode == 0x38u && !br0.has_modifier &&
          br0.has_sdwa && isV(br0.dst, 0) && isV(br0.src[0], 0) &&
          br0.sdwa_src0_sel == 4u,
          "GTA V v_bfrev_b32_sdwa v0 WORD_0 packet is admitted exactly");
    const uint32_t bfrev_sext[] = { 0x7e0070f9u, 0x000c0600u };
    const uint32_t bfrev_neg[] = { 0x7e0070f9u, 0x00140600u };
    const uint32_t bfrev_partial_dst[] = { 0x7e0070f9u, 0x00040400u };
    const uint32_t bfrev_reserved_high[] = { 0x7e0070f9u, 0x01040600u };
    const uint32_t bfrev_dword[] = { 0x7e0070f9u, 0x00060600u };
    const uint32_t bfrev_dword_neg[] = { 0x7e0070f9u, 0x00160600u };
    const uint32_t bfrev_dword_abs[] = { 0x7e0070f9u, 0x00260600u };
    CHECK(rdna2_decode_one(bfrev_sext, 2).has_modifier &&
          rdna2_decode_one(bfrev_neg, 2).has_modifier &&
          rdna2_decode_one(bfrev_partial_dst, 2).has_modifier &&
          rdna2_decode_one(bfrev_reserved_high, 2).has_modifier &&
          rdna2_decode_one(bfrev_dword, 2).has_modifier &&
          rdna2_decode_one(bfrev_dword_neg, 2).has_modifier &&
          rdna2_decode_one(bfrev_dword_abs, 2).has_modifier,
          "v_bfrev_b32 SDWA SEXT, modifiers, reserved fields, DWORD source, and partial dst reject");
    // `v_add_nc_u32_sdwa v4, 8, sext(v5) src0_sel:DWORD src1_sel:WORD_0`: SEXT on the SECOND source
    // only, which is the field the two per-source flags exist to keep apart.
    const uint32_t add_s1_sext[] = { 0x4a080af9u, 0x0c860688u };
    Rdna2Inst as1 = rdna2_decode_one(add_s1_sext, 2);
    CHECK(as1.fmt == Rdna2Format::VOP2 && as1.opcode == 0x25u && !as1.has_modifier &&
          as1.sdwa_src1_sel == 4u && as1.sdwa_src1_sext && !as1.sdwa_src0_sext,
          "VOP2 SDWA carries S1_SEXT independently of S0_SEXT");
    // SEXT on a full-DWORD source select is a combination no live encoding produces, so it stays
    // fail-visible rather than being admitted as a no-op. Same instruction, src1_sel:DWORD.
    const uint32_t add_sext_dword[] = { 0x4a080af9u, 0x0e860688u };
    Rdna2Inst asd = rdna2_decode_one(add_sext_dword, 2);
    CHECK(asd.fmt == Rdna2Format::VOP2 && asd.len_dwords == 2 && asd.has_modifier,
          "VOP2 SDWA with SEXT on a DWORD source select is 2 dwords and still has_modifier");
    // The f16 unary family's two holes: 0x59/0x5A return a mantissa and an i16 exponent, so they are
    // outside the shared one-in-one-out lowering and their SDWA form must not decode as admitted.
    CHECK(vop1_is_f16_unary(0x54u) && vop1_is_f16_unary(0x58u) && vop1_is_f16_unary(0x5Bu) &&
          vop1_is_f16_unary(0x61u) && !vop1_is_f16_unary(0x59u) && !vop1_is_f16_unary(0x5Au) &&
          !vop1_is_f16_unary(0x53u) && !vop1_is_f16_unary(0x62u),
          "vop1_is_f16_unary spans 0x54-0x58 and 0x5B-0x61 and excludes the FREXP pair");
    const uint32_t frexp_sdwa[] = { 0x7e0cb2f9u, 0x00051406u };
    Rdna2Inst fx = rdna2_decode_one(frexp_sdwa, 2);
    CHECK(fx.fmt == Rdna2Format::VOP1 && fx.opcode == 0x59u && fx.len_dwords == 2 &&
          fx.has_modifier,
          "v_frexp_mant_f16_sdwa is 2 dwords and stays fail-visible");

    // #2120: the cmpx opcode windows, which the decoder's SDWA admission and the recompiler's
    // EXEC/mask bookkeeping used to define separately. The decoder listed three of the six, so
    // `v_cmpx_*_u16` worked in its plain e32 form and REJECTED in SDWA. Boundaries are asserted
    // from BOTH sides of every window so an off-by-one at either end fails here; the map comes
    // from disassembling all 256 VOPC e32 encodings with llvm-mc for gfx1030.
    {
        // Last v_cmp / first v_cmpx / last v_cmpx / first non-cmpx, per window.
        const uint32_t is_cmpx[]  = { 0x10, 0x1F, 0x30, 0x3F, 0x90, 0x9F,
                                      0xB0, 0xBE, 0xD0, 0xDF, 0xF0, 0xFF };
        const uint32_t not_cmpx[] = { 0x0F, 0x20, 0x2F, 0x40, 0x8F, 0xA0,
                                      0xAE, 0xAF, 0xBF, 0xC0, 0xCF, 0xE0, 0xEF };
        bool ok = true;
        for (uint32_t op : is_cmpx)  if (!vopc_is_cmpx(op)) { ok = false; printf("  cmpx miss 0x%02x\n", op); }
        for (uint32_t op : not_cmpx) if (vopc_is_cmpx(op))  { ok = false; printf("  cmpx false-pos 0x%02x\n", op); }
        CHECK(ok, "vopc_is_cmpx covers all six cmpx windows and stops at every boundary");
    }
    // `v_cmpx_le_u16_sdwa v3, v4 src0_sel:WORD_0 src1_sel:WORD_0` (llvm-mc gfx1030). Opcode 0xbb =
    // base 0xab + 0x10, and 0xab is inside the admitted u16 integer window — so this must be
    // admitted with its selects, exactly as the non-cmpx 0xab form below already was. Before the
    // fix the decoder did not recognise 0xbb as cmpx, left eff = 0xbb, matched no window, and kept
    // has_modifier so the whole shader rejected.
    const uint32_t cmpx_u16[] = { 0x7d7608f9u, 0x04040003u };
    Rdna2Inst cxu = rdna2_decode_one(cmpx_u16, 2);
    CHECK(cxu.fmt == Rdna2Format::VOPC && cxu.opcode == 0xBBu && !cxu.has_modifier &&
          isV(cxu.src[0], 3) && isV(cxu.src[1], 4) &&
          cxu.sdwa_src0_sel == 4u && cxu.sdwa_src1_sel == 4u,
          "v_cmpx_le_u16 SDWA is admitted with its selects, like its v_cmp_le_u16 base");
    const uint32_t cmpx_u16_hi[] = { 0x7d7608f9u, 0x04050003u };
    Rdna2Inst cxh = rdna2_decode_one(cmpx_u16_hi, 2);
    CHECK(cxh.fmt == Rdna2Format::VOPC && cxh.opcode == 0xBBu && !cxh.has_modifier &&
          cxh.sdwa_src0_sel == 5u && cxh.sdwa_src1_sel == 4u,
          "v_cmpx_le_u16 SDWA keeps a WORD_1 source select");
    // The base compare, which was admitted before this fix too — the positive control proving the
    // u16 window itself is reached, so the two arms above isolate the cmpx mapping and nothing else.
    const uint32_t cmp_u16[] = { 0x7d5608f9u, 0x04040003u };
    Rdna2Inst cmu = rdna2_decode_one(cmp_u16, 2);
    CHECK(cmu.fmt == Rdna2Format::VOPC && cmu.opcode == 0xABu && !cmu.has_modifier &&
          cmu.sdwa_src0_sel == 4u && cmu.sdwa_src1_sel == 4u,
          "v_cmp_le_u16 SDWA (the cmpx base) remains admitted with its selects");
    // Exact packet at pc1933 in Astro Bot's world-map visibility compute shader.
    const uint32_t class_f32[] = { 0x7d1106f9u, 0x86068801u };
    Rdna2Inst cf = rdna2_decode_one(class_f32, 2);
    CHECK(cf.fmt == Rdna2Format::VOPC && cf.opcode == 0x88u && cf.len_dwords == 2u &&
          !cf.has_modifier && isS(cf.dst, 8) && isV(cf.src[0], 1) &&
          cf.src[1].kind == OperandKind::InlineInt && cf.src[1].value == 3 &&
          cf.sdwa_src0_sel == 6u && cf.sdwa_src1_sel == 6u,
          "Astro v_cmp_class_f32 SDWA packet retains s[8:9], v1, and NaN class mask 3");
    const uint32_t class_neg_f32[] = { 0x7d1108f9u, 0x86168801u };
    Rdna2Inst cnf = rdna2_decode_one(class_neg_f32, 2);
    CHECK(cnf.fmt == Rdna2Format::VOPC && cnf.opcode == 0x88u &&
          cnf.src_neg[0] && !cnf.src_abs[0] && isV(cnf.src[0], 1) &&
          cnf.src[1].kind == OperandKind::InlineInt && cnf.src[1].value == 4,
          "v_cmp_class_f32 SDWA retains the valid source NEG modifier");
    const uint32_t class_abs_f32[] = { 0x7d1108f9u, 0x86268801u };
    Rdna2Inst caf = rdna2_decode_one(class_abs_f32, 2);
    CHECK(caf.fmt == Rdna2Format::VOPC && caf.opcode == 0x88u &&
          !caf.src_neg[0] && caf.src_abs[0] && isV(caf.src[0], 1),
          "v_cmp_class_f32 SDWA retains the valid source ABS modifier");
    const uint32_t cvt_byte[] = { 0x7e0a0cf9u, 0x0000160bu };
    Rdna2Inst cb = rdna2_decode_one(cvt_byte, 2);
    CHECK(cb.fmt == Rdna2Format::VOP1 && cb.opcode == 0x06u && !cb.has_modifier &&
          isV(cb.dst, 5) && isV(cb.src[0], 11) && cb.sdwa_src0_sel == 0u,
          "Astro v_cvt_f32_u32 SDWA BYTE_0 packet is admitted with exact sub-dword select");
    const uint32_t cvt_signed_word[] = { 0x7e240af9u, 0x000d0610u };
    Rdna2Inst csw = rdna2_decode_one(cvt_signed_word, 2);
    CHECK(csw.fmt == Rdna2Format::VOP1 && csw.opcode == 0x05u && !csw.has_modifier &&
          isV(csw.dst, 18) && isV(csw.src[0], 16) && csw.sdwa_src0_sel == 5u,
          "Astro v_cvt_f32_i32 SDWA WORD_1+SEXT packet is admitted");
    const uint32_t dpp16[] = { 0x4a0e0cfau, 0xff011106u };   // v_add_nc_u32_dpp v7, v6, v6 row_shr:1
    Rdna2Inst dp = rdna2_decode_one(dpp16, 2);
    CHECK(dp.fmt == Rdna2Format::VOP2 && dp.len_dwords == 2 && !dp.has_modifier && dp.has_dpp &&
          dp.dpp_ctrl == 0x111u && !dp.dpp_bound_ctrl &&
          dp.dpp_row_mask == 0xfu && dp.dpp_bank_mask == 0xfu,
          "VOP2 DPP16 ROW_SHR form retains its unbounded lane control");
    const uint32_t gta_vmin_row_ror8[] = { 0x1e2024fau, 0xff092812u };
    Rdna2Inst gvr = rdna2_decode_one(gta_vmin_row_ror8, 2);
    CHECK(gvr.fmt == Rdna2Format::VOP2 && gvr.opcode == 0x0fu &&
          gvr.len_dwords == 2u && !gvr.has_modifier && gvr.has_dpp &&
          gvr.dpp_ctrl == 0x128u && gvr.dpp_bound_ctrl &&
          gvr.dpp_row_mask == 0xfu && gvr.dpp_bank_mask == 0xfu &&
          isV(gvr.dst, 16) && isV(gvr.src[0], 18) && isV(gvr.src[1], 18),
          "GTA V V_MIN_F32 ROW_ROR:8 packet retains exact control and operands");
    const uint32_t gta_vmin_row_ror8_v0[] = { 0x1e0000fau, 0xff092800u };
    Rdna2Inst gvr0 = rdna2_decode_one(gta_vmin_row_ror8_v0, 2);
    CHECK(gvr0.fmt == Rdna2Format::VOP2 && gvr0.opcode == 0x0fu &&
          !gvr0.has_modifier && gvr0.has_dpp && gvr0.dpp_ctrl == 0x128u &&
          gvr0.dpp_bound_ctrl && isV(gvr0.dst, 0) &&
          isV(gvr0.src[0], 0) && isV(gvr0.src[1], 0),
          "GTA V in-place V_MIN_F32 ROW_ROR:8 packet decodes its DPP SRC0");
    const uint32_t gta_vmax_row_ror8[] = { 0x200406fau, 0xff092803u };
    Rdna2Inst gvx = rdna2_decode_one(gta_vmax_row_ror8, 2);
    CHECK(gvx.fmt == Rdna2Format::VOP2 && gvx.opcode == 0x10u &&
          !gvx.has_modifier && gvx.has_dpp && gvx.dpp_ctrl == 0x128u &&
          gvx.dpp_bound_ctrl && gvx.dpp_row_mask == 0xfu &&
          gvx.dpp_bank_mask == 0xfu && isV(gvx.dst, 2) &&
          isV(gvx.src[0], 3) && isV(gvx.src[1], 3),
          "GTA V exact V_MAX_F32 ROW_ROR:8 packet decodes its permuted SRC0");
    const uint32_t gta_vmov_row_ror8[] = { 0x7e2402fau, 0xff092811u };
    Rdna2Inst gvm = rdna2_decode_one(gta_vmov_row_ror8, 2);
    CHECK(gvm.fmt == Rdna2Format::VOP1 && gvm.opcode == 0x01u &&
          !gvm.has_modifier && gvm.has_dpp && gvm.dpp_ctrl == 0x128u &&
          gvm.dpp_bound_ctrl && gvm.dpp_row_mask == 0xfu &&
          gvm.dpp_bank_mask == 0xfu && isV(gvm.dst, 18) &&
          isV(gvm.src[0], 17) && gvm.n_src == 1u,
          "GTA V exact V_MOV_B32 ROW_ROR:8 packet decodes its sole permuted source");
    const std::pair<uint32_t, uint32_t> gta_vmin_row_ror8_mutants[] = {
        {0x1c2024fau, 0xff092812u}, // different VOP2 opcode
        {0x1e2024fau, 0xff012812u}, // BOUND_CTRL=0
        {0x1e2024fau, 0xff092912u}, // ROW_ROR:9
        {0x1e2024fau, 0xef092812u}, // partial ROW_MASK
        {0x1e2024fau, 0xfe092812u}, // partial BANK_MASK
        {0x1e2024fau, 0xff192812u}, // SRC0_NEG
        {0x1e2024fau, 0xff0d2812u}, // FI=1
    };
    for (const auto& mutant : gta_vmin_row_ror8_mutants) {
        const uint32_t words[] = {mutant.first, mutant.second};
        Rdna2Inst rejected = rdna2_decode_one(words, 2);
        CHECK(rejected.has_modifier && !rejected.has_dpp,
              "GTA V ROW_ROR:8 admission rejects opcode/control/mask/modifier mutations");
    }
    const std::pair<uint32_t, uint32_t> gta_new_row_ror8_mutants[] = {
        {0x200406fau, 0xff012803u}, // V_MAX_F32 BOUND_CTRL=0
        {0x200406fau, 0xff092903u}, // V_MAX_F32 ROW_ROR:9
        {0x200406fau, 0xef092803u}, // V_MAX_F32 partial ROW_MASK
        {0x200406fau, 0xff192803u}, // V_MAX_F32 SRC0_NEG
        {0x7e2402fau, 0xff012811u}, // V_MOV_B32 BOUND_CTRL=0
        {0x7e2402fau, 0xff092911u}, // V_MOV_B32 ROW_ROR:9
        {0x7e2402fau, 0xfe092811u}, // V_MOV_B32 partial BANK_MASK
        {0x7e2402fau, 0xff192811u}, // V_MOV_B32 SRC0_NEG
        {0x060406fau, 0xff092803u}, // V_ADD_F32 is outside the admitted live family
    };
    for (const auto& mutant : gta_new_row_ror8_mutants) {
        const uint32_t words[] = {mutant.first, mutant.second};
        Rdna2Inst rejected = rdna2_decode_one(words, 2);
        CHECK(rejected.has_modifier && !rejected.has_dpp,
              "GTA V new ROW_ROR:8 sites reject control, mask, modifier, and opcode mutations");
    }
    const uint32_t gta_row_mask_add[] = { 0x4a2826fau, 0xaf00e414u };
    Rdna2Inst grm = rdna2_decode_one(gta_row_mask_add, 2);
    CHECK(grm.fmt == Rdna2Format::VOP2 && grm.opcode == 0x25u &&
          !grm.has_modifier && grm.has_dpp && grm.dpp_ctrl == 0xe4u &&
          !grm.dpp_bound_ctrl && grm.dpp_row_mask == 0xau &&
          grm.dpp_bank_mask == 0xfu && isV(grm.dst, 20) &&
          isV(grm.src[0], 20) && isV(grm.src[1], 19),
          "GTA V identity DPP add retains exact partial row mask and operands");
    for (uint32_t mutant : {0x9f00e414u, 0xae00e414u, 0xaf08e414u, 0xaf00e514u}) {
        const uint32_t words[] = {0x4a2826fau, mutant};
        Rdna2Inst rejected = rdna2_decode_one(words, 2);
        CHECK(rejected.has_modifier && !rejected.has_dpp,
              "GTA V partial DPP admission rejects row/bank/BC/control mutations");
    }
    const uint32_t ngg_row_shift[] = { 0x4a1e1efau, 0xff09110fu };
    Rdna2Inst nrs = rdna2_decode_one(ngg_row_shift, 2);
    CHECK(nrs.fmt == Rdna2Format::VOP2 && nrs.opcode == 0x25u && !nrs.has_modifier &&
          nrs.has_dpp && nrs.dpp_ctrl == 0x111u && isV(nrs.src[0], 15) && isV(nrs.src[1], 15),
          "Astro NGG bounded DPP row-right add is admitted exactly");
    const uint32_t fragment_row_or[] = { 0x381414fau, 0xff01110au };
    Rdna2Inst fro = rdna2_decode_one(fragment_row_or, 2);
    CHECK(fro.fmt == Rdna2Format::VOP2 && fro.opcode == 0x1cu && !fro.has_modifier &&
          fro.has_dpp && fro.dpp_ctrl == 0x111u && isV(fro.src[0], 10) && isV(fro.src[1], 10),
          "Astro fragment DPP row-right OR is admitted exactly");
    const uint32_t fragment_permlanex[] = { 0xd7781009u, 0x0305830au };
    Rdna2Inst fpx = rdna2_decode_one(fragment_permlanex, 2);
    CHECK(fpx.fmt == Rdna2Format::VOP3 && fpx.opcode == 0x378u && isV(fpx.dst, 9) &&
          isV(fpx.src[0], 10) && fpx.src[1].kind == OperandKind::InlineInt &&
          fpx.src[1].value == -1 && fpx.src[2].kind == OperandKind::InlineInt &&
          fpx.src[2].value == -1,
          "Astro fragment PERMLANEX16 adjacent-row broadcast decodes exactly");
    const uint32_t dpp16_bounded[] = { 0x4a1412fau, 0xff091109u };
    Rdna2Inst dpb = rdna2_decode_one(dpp16_bounded, 2);
    CHECK(dpb.fmt == Rdna2Format::VOP2 && dpb.len_dwords == 2 && !dpb.has_modifier && dpb.has_dpp &&
          dpb.dpp_ctrl == 0x111u && dpb.dpp_bound_ctrl && isV(dpb.src[0], 9) && isV(dpb.src[1], 9),
          "Plucky Squire bounded ROW_SHR:1 preserves source and BOUND_CTRL fields");
    const uint32_t dpp8[] = { 0x4a0e0ce9u, 0xfac68806u };    // v_add_nc_u32_dpp v7, v6, v6 dpp8:[...]
    Rdna2Inst d8 = rdna2_decode_one(dpp8, 2);
    CHECK(d8.fmt == Rdna2Format::VOP2 && d8.len_dwords == 2 && d8.has_modifier,
          "VOP2 DPP8 form is 2 dwords and flagged has_modifier");
    const uint32_t fmaak[] = { 0x5a000501u, 0xd4a0e43au };   // v_fmaak_f32 v0, v1, v2, K (mandatory literal)
    Rdna2Inst fk = rdna2_decode_one(fmaak, 2);
    CHECK(fk.fmt == Rdna2Format::VOP2 && fk.len_dwords == 2 && fk.has_literal && !fk.has_modifier &&
          fk.literal == 0xd4a0e43au, "VOP2 v_fmaak_f32 carries its mandatory 32-bit literal");

    // VOP3 length: 2 dwords, plus a trailing 32-bit literal when a src field is 0xFF. Encodings from
    // llvm-mc gfx1030: v_med3_f32 v0,v1,v2,0x40490fdb (src2=literal) vs v_mad_u32_u24 (no literal).
    const uint32_t vop3_lit[] = { 0xd5570000u, 0x03fe0501u, 0x40490fdbu };
    Rdna2Inst v3l = rdna2_decode_one(vop3_lit, 3);
    CHECK(v3l.fmt == Rdna2Format::VOP3 && v3l.len_dwords == 3 && v3l.has_literal && v3l.literal == 0x40490fdbu,
          "VOP3 with a 0xFF src field carries its trailing 32-bit literal (3 dwords)");
    const uint32_t vop3_nolit[] = { 0xd5430000u, 0x040e0501u };
    Rdna2Inst v3n = rdna2_decode_one(vop3_nolit, 2);
    CHECK(v3n.fmt == Rdna2Format::VOP3 && v3n.len_dwords == 2 && !v3n.has_literal,
          "VOP3 without a literal src is 2 dwords");
    // Exact Syberia source-87 low/high pair. llvm-mc gfx1030 prints OPSEL [0,0,1,0] and
    // [1,1,0,1]; the fourth bit selects which packed destination half is replaced.
    const uint32_t max3_f16_lo[] = { 0xd7542009u, 0x045a2d17u };
    const uint32_t max3_f16_hi[] = { 0xd7545809u, 0x045e2d17u };
    const Rdna2Inst max3_lo = rdna2_decode_one(max3_f16_lo, 2);
    const Rdna2Inst max3_hi = rdna2_decode_one(max3_f16_hi, 2);
    CHECK(max3_lo.fmt == Rdna2Format::VOP3 && max3_lo.opcode == 0x354u &&
              max3_lo.vop3p_opsel == 0x4u &&
              max3_hi.fmt == Rdna2Format::VOP3 && max3_hi.opcode == 0x354u &&
              max3_hi.vop3p_opsel == 0xbu,
          "Syberia v_max3_f16 retains all source and destination OPSEL bits");
    // VOPC-as-VOP3 uses the same 64-bit packet prefix but its low dword0 field is an SGPR-mask
    // destination, never a VGPR. Exact Astro world-map encoding, round-tripped with llvm-mc gfx1010.
    const uint32_t vopc_e64_u64[] = { 0xd4e4006au, 0x0001006au };
    Rdna2Inst ce64 = rdna2_decode_one(vopc_e64_u64, 2);
    CHECK(ce64.fmt == Rdna2Format::VOPC && ce64.opcode == 0xe4u && ce64.len_dwords == 2 &&
          isS(ce64.dst, 106) && ce64.n_src == 2 && ce64.src[0].value == 106 &&
          ce64.src[1].kind == OperandKind::InlineInt && ce64.src[1].value == 0,
          "Astro v_cmp_gt_u64_e64 decodes its explicit VCC destination and two sources");

    // MIMG length: non-NSA image op is 2 dwords; NSA form adds dword0[2:1] extra address dwords.
    // Encodings from llvm-mc gfx1030 (image_load 2D non-NSA; image_sample 2D NSA [v0,v1] = 1 extra).
    const uint32_t mimg_reg[] = { 0xf0000f08u, 0x00000000u };
    CHECK(rdna2_decode_one(mimg_reg, 2).fmt == Rdna2Format::MIMG && rdna2_decode_one(mimg_reg, 2).len_dwords == 2,
          "non-NSA MIMG is 2 dwords");
    const uint32_t mimg_nsa[] = { 0xf0800f0au, 0x00400000u, 0x00000001u };
    CHECK(rdna2_decode_one(mimg_nsa, 3).fmt == Rdna2Format::MIMG && rdna2_decode_one(mimg_nsa, 3).len_dwords == 3,
          "NSA MIMG with one extra address dword is 3 dwords");
    // NSA extra-dword CAPTURE + coord layout: image_load v[0:3], [v0,v7,v3], s[0:7] dim:3D (llvm-mc gfx1010).
    // coord0 = words[1][7:0] = v0; coord1 = words[2][7:0] = v7; coord2 = words[2][15:8] = v3. The recompiler
    // reads exactly these bytes to gather the non-sequential coords.
    const uint32_t mimg_nsa3d[] = { 0xf0000f12u, 0x00000000u, 0x00000307u };
    Rdna2Inst n3 = rdna2_decode_one(mimg_nsa3d, 3);
    CHECK(n3.fmt == Rdna2Format::MIMG && n3.opcode == 0x00u && n3.len_dwords == 3 &&
          n3.mimg_nsa == 1u && n3.mimg_dim == 2u &&
          (n3.words[1] & 0xFFu) == 0u && (n3.words[2] & 0xFFu) == 7u && ((n3.words[2] >> 8) & 0xFFu) == 3u,
          "NSA MIMG 3D captures the extra address dword; coords decode to v0,v7,v3");
    // GTA V gameplay compute packets. The helper identifies only the audited zero-specializable
    // IMAGE_*_MIP shapes and returns the dimension-specific mip VGPR; it does not prove its value.
    const uint32_t gta_load_mip_2d_words[] = {0xf0043108u, 0x00050000u};
    const uint32_t gta_load_mip_2d_xyzw_words[] = {0xf0043f08u, 0x00050000u};
    const uint32_t gta_load_mip_2da_words[] = {0xf0043128u, 0x00050000u};
    const uint32_t gta_store_mip_2d_words[] = {
        0xf024310au, 0x00030004u, 0x00000503u,
    };
    const uint32_t gta_store_mip_2d_xyzw_words[] = {
        0xf0243f0au, 0x00030005u, 0x00000604u,
    };
    const Rdna2Inst gta_load_mip_2d = rdna2_decode_one(gta_load_mip_2d_words, 2);
    const Rdna2Inst gta_load_mip_2d_xyzw =
        rdna2_decode_one(gta_load_mip_2d_xyzw_words, 2);
    const Rdna2Inst gta_load_mip_2da = rdna2_decode_one(gta_load_mip_2da_words, 2);
    const Rdna2Inst gta_store_mip_2d = rdna2_decode_one(gta_store_mip_2d_words, 3);
    const Rdna2Inst gta_store_mip_2d_xyzw =
        rdna2_decode_one(gta_store_mip_2d_xyzw_words, 3);
    uint32_t mip_vgpr = UINT32_MAX;
    CHECK(gta_load_mip_2d.opcode == 0x01u && gta_load_mip_2d.mimg_dim == 1u &&
              rdna2_mimg_zero_mip_shape(gta_load_mip_2d, &mip_vgpr) && mip_vgpr == 2u,
          "GTA V IMAGE_LOAD_MIP 2D identifies v2 as its mip operand");
    CHECK(gta_load_mip_2d_xyzw.mimg_dmask == 0xfu &&
              rdna2_mimg_zero_mip_shape(gta_load_mip_2d_xyzw, &mip_vgpr) &&
              mip_vgpr == 2u,
          "GTA V 0x2042f49800 IMAGE_LOAD_MIP accepts its exact xyzw result mask");
    CHECK(gta_load_mip_2da.opcode == 0x01u && gta_load_mip_2da.mimg_dim == 5u &&
              rdna2_mimg_zero_mip_shape(gta_load_mip_2da, &mip_vgpr) && mip_vgpr == 3u,
          "GTA V IMAGE_LOAD_MIP 2D_ARRAY identifies v3 after the preserved slice");
    CHECK(gta_store_mip_2d.opcode == 0x09u && gta_store_mip_2d.mimg_dim == 1u &&
              gta_store_mip_2d.mimg_nsa == 1u &&
              rdna2_mimg_zero_mip_shape(gta_store_mip_2d, &mip_vgpr) && mip_vgpr == 5u,
          "GTA V IMAGE_STORE_MIP NSA 2D identifies its explicit v5 mip operand");
    CHECK(gta_store_mip_2d_xyzw.mimg_dmask == 0xfu &&
              gta_store_mip_2d_xyzw.dst.kind == OperandKind::VGPR &&
              gta_store_mip_2d_xyzw.dst.value == 0 &&
              gta_store_mip_2d_xyzw.src[0].kind == OperandKind::VGPR &&
              gta_store_mip_2d_xyzw.src[0].value == 5 &&
              rdna2_mimg_zero_mip_shape(gta_store_mip_2d_xyzw, &mip_vgpr) &&
              mip_vgpr == 6u,
          "GTA V 0x2042f49800 IMAGE_STORE_MIP keeps v[0:3], (v5,v4), and v6 mip exact");
    const uint32_t gta_pc10_vop2_word[] = {0x4a000804u};
    const uint32_t wide_vop3_words[] = {0xd5761e01u, 0x040a0100u};
    const uint32_t wide_mimg_words[] = {0xf0003f08u, 0x00050000u};
    const uint32_t mimg_tfe_words[] = {0xf0010308u, 0x00050000u};
    const uint32_t mimg_store_tfe_words[] = {0xf0210308u, 0x00050000u};
    const uint32_t mimg_store_mip_tfe_words[] = {0xf0250308u, 0x00050003u};
    const uint32_t mtbuf_store_tfe_words[] = {0xe9e72000u, 0x80882008u};
    const uint32_t wide_mubuf_words[] = {0xe0382020u, 0x80020000u};
    const uint32_t atomic_x2_words[] = {0xe1406000u, 0x80000000u};
    const Rdna2Inst gta_pc10_vop2 = rdna2_decode_one(gta_pc10_vop2_word, 1);
    const Rdna2Inst wide_vop3 = rdna2_decode_one(wide_vop3_words, 2);
    const Rdna2Inst wide_mimg = rdna2_decode_one(wide_mimg_words, 2);
    const Rdna2Inst mimg_tfe = rdna2_decode_one(mimg_tfe_words, 2);
    const Rdna2Inst mimg_store_tfe = rdna2_decode_one(mimg_store_tfe_words, 2);
    const Rdna2Inst mimg_store_mip_tfe = rdna2_decode_one(mimg_store_mip_tfe_words, 2);
    const Rdna2Inst mtbuf_store_tfe = rdna2_decode_one(mtbuf_store_tfe_words, 2);
    const Rdna2Inst wide_mubuf = rdna2_decode_one(wide_mubuf_words, 2);
    const Rdna2Inst atomic_x2 = rdna2_decode_one(atomic_x2_words, 2);
    CHECK(gta_pc10_vop2.fmt == Rdna2Format::VOP2 && gta_pc10_vop2.dst.value == 0 &&
              rdna2_vgpr_write_count(gta_pc10_vop2) == 1u &&
              wide_vop3.fmt == Rdna2Format::VOP3 && wide_vop3.dst.value == 1 &&
              rdna2_vgpr_write_count(wide_vop3) == 2u &&
              rdna2_vgpr_write_count(wide_mimg) == 4u &&
              mimg_tfe.mimg_tfe && mimg_tfe.mimg_dmask == 3u &&
              rdna2_vgpr_write_count(mimg_tfe) == 2u &&
              rdna2_tfe_status_vgpr(mimg_tfe) == 2 &&
              rdna2_vgpr_destination_span(mimg_tfe) == 3u &&
              rdna2_vgpr_write_count(wide_mubuf) == 4u &&
              atomic_x2.opcode == kMubufOpcodeAtomicSwapX2 && atomic_x2.mubuf_glc &&
              rdna2_vgpr_write_count(atomic_x2) == 2u &&
              rdna2_vgpr_destination_span(atomic_x2) == 2u &&
              rdna2_vgpr_write_count(mt_tfe) == 1u &&
              rdna2_tfe_status_vgpr(mt_tfe) == 2 &&
              rdna2_vgpr_destination_span(mt_tfe) == 2u &&
              rdna2_vgpr_write_count(gta_store_mip_2d_xyzw) == 0u &&
              rdna2_vgpr_destination_span(gta_store_mip_2d_xyzw) == 4u,
          "shared VGPR writer inventory distinguishes scalar, pair, wide memory, and store packets");
    CHECK(mimg_store_tfe.fmt == Rdna2Format::MIMG &&
              mimg_store_tfe.opcode == 0x08u && mimg_store_tfe.mimg_tfe &&
              mimg_store_tfe.mimg_dmask == 3u &&
              rdna2_vgpr_write_count(mimg_store_tfe) == 0u &&
              rdna2_tfe_status_vgpr(mimg_store_tfe) == 2 &&
              rdna2_vgpr_destination_span(mimg_store_tfe) == 3u &&
              mimg_store_mip_tfe.opcode == 0x09u && mimg_store_mip_tfe.mimg_tfe &&
              rdna2_vgpr_write_count(mimg_store_mip_tfe) == 0u &&
              rdna2_tfe_status_vgpr(mimg_store_mip_tfe) == 2 &&
              rdna2_vgpr_destination_span(mimg_store_mip_tfe) == 3u &&
              mtbuf_store_tfe.fmt == Rdna2Format::MTBUF &&
              mtbuf_store_tfe.opcode == 7u && mtbuf_store_tfe.mtbuf_tfe &&
              rdna2_vgpr_write_count(mtbuf_store_tfe) == 0u &&
              rdna2_tfe_status_vgpr(mtbuf_store_tfe) == 36 &&
              rdna2_vgpr_destination_span(mtbuf_store_tfe) == 5u,
          "store TFE keeps its VDATA source prefix separate from the trailing status write");
    const uint32_t ordinary_load_words[] = {0xf0003108u, 0x00050000u};
    const uint32_t load_without_glc_words[] = {0xf0041108u, 0x00050000u};
    const uint32_t load_partial_mask_words[] = {0xf0043308u, 0x00050000u};
    const uint32_t store_extra_address_words[] = {
        0xf024310au, 0x00030004u, 0x00010503u,
    };
    const uint32_t store_partial_mask_words[] = {
        0xf024330au, 0x00030004u, 0x00000503u,
    };
    CHECK(!rdna2_mimg_zero_mip_shape(rdna2_decode_one(ordinary_load_words, 2)) &&
              !rdna2_mimg_zero_mip_shape(rdna2_decode_one(load_without_glc_words, 2)) &&
              !rdna2_mimg_zero_mip_shape(rdna2_decode_one(load_partial_mask_words, 2)) &&
              !rdna2_mimg_zero_mip_shape(rdna2_decode_one(store_extra_address_words, 3)) &&
              !rdna2_mimg_zero_mip_shape(rdna2_decode_one(store_partial_mask_words, 3)),
          "zero-mip shape rejects opcode, control, and unused-address mutations at the packet gate");
    // Astro Bot's world-map ray traversal uses the maximum three NSA dwords to name eleven input
    // VGPRs. Retaining dword4 is required for ray_inv_dir.y/z (v71/v72).
    const uint32_t mimg_bvh[] = {
        0xf1989f07u, 0x00040303u, 0x43440d3fu, 0x46424140u, 0x00004847u,
    };
    Rdna2Inst bvh = rdna2_decode_one(mimg_bvh, std::size(mimg_bvh));
    CHECK(bvh.fmt == Rdna2Format::MIMG && bvh.opcode == 0xe6u && bvh.len_dwords == 5 &&
          bvh.mimg_dmask == 0xfu && bvh.mimg_unorm && bvh.mimg_dim == 0u &&
          bvh.dst.value == 3 && bvh.src[0].value == 3 && bvh.src[1].value == 16 &&
          bvh.words[4] == 0x00004847u,
          "Astro IMAGE_BVH_INTERSECT_RAY retains its third NSA address dword");
    // Astro Bot's live image_atomic_swap packet. Bit 13 is GLC: atomically exchange v9 with the
    // R32_UINT texel at (v0,v1), returning the pre-operation value to v9.
    const uint32_t mimg_atomic_swap[] = { 0xf03c2108u, 0x00000900u };
    Rdna2Inst atomic_swap = rdna2_decode_one(mimg_atomic_swap, 2);
    CHECK(atomic_swap.fmt == Rdna2Format::MIMG && atomic_swap.opcode == 0x0fu &&
          atomic_swap.mimg_dim == 1u && atomic_swap.mimg_dmask == 1u && atomic_swap.mimg_glc &&
          atomic_swap.dst.value == 9 && atomic_swap.src[0].value == 0 && atomic_swap.src[1].value == 0,
          "Astro image_atomic_swap decodes 2D/R32 data and the return-pre-op GLC flag");
    // Astro's world-map visibility kernel uses the corresponding GFX10 IMAGE_ATOMIC_ADD encoding.
    const uint32_t mimg_atomic_add[] = { 0xf0442108u, 0x00070104u };
    Rdna2Inst atomic_add = rdna2_decode_one(mimg_atomic_add, 2);
    CHECK(atomic_add.fmt == Rdna2Format::MIMG && atomic_add.opcode == 0x11u &&
          atomic_add.mimg_dim == 1u && atomic_add.mimg_dmask == 1u && atomic_add.mimg_glc &&
          atomic_add.dst.value == 1 && atomic_add.src[0].value == 4 && atomic_add.src[1].value == 28,
          "Astro image_atomic_add decodes the live 2D/R32 visibility packet");

    // Every MIMG Table 100 control is retained independently. These are exact llvm-mc gfx1030
    // IMAGE_GET_LOD encodings except D16, which llvm-mc rejects for this opcode; the raw D16 field
    // still has to remain fail-visible rather than alias the ordinary FP32 result form.
    const Rdna2Inst lod_plain = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf1800108u, 0x01480809u}.data(), 2);
    const Rdna2Inst lod_dlc = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf1800188u, 0x01480809u}.data(), 2);
    const Rdna2Inst lod_slc = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf3800108u, 0x01480809u}.data(), 2);
    const Rdna2Inst lod_r128 = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf1808108u, 0x01480809u}.data(), 2);
    const Rdna2Inst lod_tfe = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf1810108u, 0x01480809u}.data(), 2);
    const Rdna2Inst lod_lwe = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf1820108u, 0x01480809u}.data(), 2);
    const Rdna2Inst lod_a16 = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf1800108u, 0x41480809u}.data(), 2);
    const Rdna2Inst lod_d16_raw = rdna2_decode_one(
        std::array<uint32_t, 2>{0xf1800108u, 0x81480809u}.data(), 2);
    CHECK(lod_plain.mimg_nsa == 0u && !lod_plain.mimg_unorm && !lod_plain.mimg_dlc &&
              !lod_plain.mimg_glc && !lod_plain.mimg_slc && !lod_plain.mimg_r128 &&
              !lod_plain.mimg_tfe && !lod_plain.mimg_lwe && !lod_plain.mimg_a16 &&
              !lod_plain.mimg_d16 && !lod_plain.mimg_reserved,
          "ordinary IMAGE_GET_LOD has no auxiliary MIMG controls");
    CHECK(lod_dlc.mimg_dlc && lod_slc.mimg_slc && lod_r128.mimg_r128 &&
              lod_tfe.mimg_tfe && lod_lwe.mimg_lwe && lod_a16.mimg_a16 &&
              lod_d16_raw.mimg_d16,
          "IMAGE_GET_LOD retains exact DLC/SLC/R128/TFE/LWE/A16 and raw D16 fields");
    bool all_reserved_retained = true;
    for (const auto words : std::array<std::array<uint32_t, 2>, 6>{
             std::array<uint32_t, 2>{0xf1800148u, 0x01480809u}, // dword0 bit 6
             std::array<uint32_t, 2>{0xf1804108u, 0x01480809u}, // dword0 bit 14
             std::array<uint32_t, 2>{0xf1800108u, 0x05480809u}, // dword1 bit 26
             std::array<uint32_t, 2>{0xf1800108u, 0x09480809u}, // dword1 bit 27
             std::array<uint32_t, 2>{0xf1800108u, 0x11480809u}, // dword1 bit 28
             std::array<uint32_t, 2>{0xf1800108u, 0x21480809u}, // dword1 bit 29
         })
        all_reserved_retained &= rdna2_decode_one(words.data(), words.size()).mimg_reserved;
    CHECK(all_reserved_retained,
          "all six reserved MIMG control bits are retained for fail-visible rejection");

    // inline-constant field decode: SGPR106 special, field 129 -> +1, 193 -> -1, 242 -> 1.0f
    CHECK(decode_src_field(0).kind == OperandKind::SGPR && decode_src_field(0).value == 0, "field 0 -> SGPR0");
    CHECK(decode_src_field(257).kind == OperandKind::VGPR && decode_src_field(257).value == 1, "field 257 -> VGPR1");
    CHECK(decode_src_field(129).kind == OperandKind::InlineInt && decode_src_field(129).value == 1, "field 129 -> int +1");
    CHECK(decode_src_field(193).kind == OperandKind::InlineInt && decode_src_field(193).value == -1, "field 193 -> int -1");
    CHECK(decode_src_field(242).kind == OperandKind::InlineFloat && inline_float_value(242) == 1.0f, "field 242 -> float 1.0");

    // SMEM SOFFSET + signed immediate (#149). Encodings from llvm-mc gfx1030.
    //   imm:  s_buffer_load_dword s0, s[4:7], 0x10 -> SOFFSET=NULL(125), offset 0x10
    //   reg:  s_buffer_load_dword s0, s[4:7], s8   -> SOFFSET=SGPR s8, offset 0
    const uint32_t smem_imm[] = { 0xf4200002u, 0xfa000010u };
    Rdna2Inst si = rdna2_decode_one(smem_imm, 2);
    CHECK(si.fmt == Rdna2Format::SMEM && si.n_src == 2 && si.literal == 0x10u &&
          si.src[1].kind == OperandKind::Special && si.src[1].value == 125,
          "SMEM immediate: offset 0x10, SOFFSET decodes to NULL(125)");
    const uint32_t smem_reg[] = { 0xf4200002u, 0x10000000u };
    Rdna2Inst sr = rdna2_decode_one(smem_reg, 2);
    CHECK(sr.fmt == Rdna2Format::SMEM && sr.literal == 0u && isS(sr.src[1], 8),
          "SMEM register offset: SOFFSET decodes to SGPR s8 (was silently dropped)");
    // Signed 21-bit immediate: a raw 0x1FFFF8 sign-extends to -8 (offset = -8 bytes).
    const uint32_t smem_neg[] = { 0xf4000002u, 0xfa1ffff8u };
    Rdna2Inst sn = rdna2_decode_one(smem_neg, 2);
    CHECK((int32_t)sn.literal == -8, "SMEM 21-bit immediate is sign-extended (0x1ffff8 -> -8)");

    // s_load uses a 64-bit address pair, not V# bounds. UE4's root SRT at s[12:13] loads x4 at
    // byte offset 0x250, requiring a 0x260-byte mapped range even when s14 happens to contain 1.
    const uint32_t sload_range[] = { 0xf4080706u, 0xfa000250u, 0xbf810000u };
    CHECK(rdna2_sload_required_bytes(sload_range, 3, 12) == 0x260u,
          "immediate s_load range inference includes offset plus x4 width");
    CHECK(rdna2_sload_required_bytes(sload_range, 3, 8) == 0u,
          "s_load range inference ignores a different SBASE pair");

    // --- 2026-07 ISA-audit decode fixes (#878/#882) ---
    // MUBUF opcode is 8 bits [25:18]: buffer_load_format_d16_x (op 128) must NOT alias onto
    // buffer_load_format_x (op 0). llvm-mc gfx1030: 0xe2000000 0x80000000.
    const uint32_t mubuf_d16[] = { 0xe2000000u, 0x80000000u };
    Rdna2Inst md = rdna2_decode_one(mubuf_d16, 2);
    CHECK(md.fmt == Rdna2Format::MUBUF && md.opcode == 128u,
          "MUBUF 8-bit opcode: buffer_load_format_d16_x decodes as 128, not 0");
    // MUBUF GLC (bit 14): buffer_atomic_umax ... glc = 0xe0e04004 (plain live word is 0xe0e00004).
    const uint32_t mubuf_glc[] = { 0xe0e04004u, 0x80000000u };
    const uint32_t mubuf_noglc[] = { 0xe0e00004u, 0x80000000u };
    CHECK(rdna2_decode_one(mubuf_glc, 2).mubuf_glc && !rdna2_decode_one(mubuf_noglc, 2).mubuf_glc,
          "MUBUF GLC bit 14 decodes (atomics: return pre-op value)");
    // GTA V pc224: buffer_load_dword ... glc dlc. Clearing only bit 15 is the mutation arm: GLC
    // remains set while DLC must disappear, so this checks the production decoder field itself.
    const uint32_t mubuf_glc_dlc[] = { 0xe030e010u, 0x80001e1du };
    const uint32_t mubuf_glc_only[] = { 0xe0306010u, 0x80001e1du };
    const Rdna2Inst mubuf_both = rdna2_decode_one(mubuf_glc_dlc, 2);
    const Rdna2Inst mubuf_without_dlc = rdna2_decode_one(mubuf_glc_only, 2);
    CHECK(mubuf_both.mubuf_glc && mubuf_both.mubuf_dlc &&
              mubuf_without_dlc.mubuf_glc && !mubuf_without_dlc.mubuf_dlc,
          "MUBUF DLC bit 15 decodes independently from GLC on GTA V's exact polling load");
    const uint32_t mtbuf_dlc[] = { 0xe8b0a000u, 0x80020100u };
    const uint32_t mtbuf_nodlc[] = { 0xe8b02000u, 0x80020100u };
    CHECK(rdna2_decode_one(mtbuf_dlc, 2).mubuf_dlc &&
              !rdna2_decode_one(mtbuf_nodlc, 2).mubuf_dlc,
          "MTBUF shares the decoded DLC bit 15 cache-policy field");
    // MUBUF LDS (bit 16): hand-set on the plain dword load (llvm-mc gfx1030 rejects the syntax,
    // but the field is architectural — Table 98).
    const uint32_t mubuf_lds[] = { 0xe0310000u, 0x80000000u };
    CHECK(rdna2_decode_one(mubuf_lds, 2).mubuf_lds && !rdna2_decode_one(mubuf_noglc, 2).mubuf_lds,
          "MUBUF LDS bit 16 decodes (buffer<->LDS transfer flag)");
    // MIMG opcode combines dword0 bit 0 with [24:18]: image_msaa_load (op 128) must NOT alias
    // onto image_load (op 0). llvm-mc gfx1030: 0xf0000f31 0x00000000.
    const uint32_t mimg_msaa[] = { 0xf0000f31u, 0x00000000u };
    Rdna2Inst mm = rdna2_decode_one(mimg_msaa, 2);
    CHECK(mm.fmt == Rdna2Format::MIMG && mm.opcode == 128u,
          "MIMG 8-bit opcode: image_msaa_load decodes as 128, not 0 (image_load)");
    // v_fmamk_f16 (0x37) / v_fmaak_f16 (0x38) carry a mandatory 32-bit literal K: 2 dwords, or the
    // K re-decodes as a phantom instruction and desyncs the walk. llvm-mc gfx1030.
    const uint32_t fmamk16[] = { 0x6e000501u, 0x00001234u };
    const uint32_t fmaak16[] = { 0x70000501u, 0x00001234u };
    CHECK(rdna2_decode_one(fmamk16, 2).len_dwords == 2 && rdna2_decode_one(fmamk16, 2).has_literal &&
          rdna2_decode_one(fmaak16, 2).len_dwords == 2,
          "v_fmamk_f16/v_fmaak_f16 count their mandatory literal K (2 dwords)");
    // SOPK s_setreg_imm32_b32 (op 21) trails a 32-bit literal: 2 dwords. llvm-mc gfx1030.
    const uint32_t setreg32[] = { 0xba80f801u, 0x12345678u };
    Rdna2Inst sk = rdna2_decode_one(setreg32, 2);
    CHECK(sk.fmt == Rdna2Format::SOPK && sk.len_dwords == 2 && sk.has_literal && sk.literal == 0x12345678u,
          "s_setreg_imm32_b32 counts its trailing data literal (2 dwords)");
    const uint32_t movk[] = { 0xb0040005u };
    CHECK(rdna2_decode_one(movk, 1).len_dwords == 1, "plain SOPK (s_movk_i32) stays 1 dword");
    // VOP3B carve-out covers the whole family (sec 13.3.4): v_add_co_u32 e64 (0x30F) decodes its
    // SDST from dword0[14:8] instead of misreading it as ABS bits. (Hand-built word: the VOP3B
    // field layout is the one llvm-mc round-trips for 0x128 v_add_co_ci_u32.)
    const uint32_t addco[] = { 0xd70f0000u | (4u << 8), 0x00020501u };
    Rdna2Inst ac = rdna2_decode_one(addco, 2);
    CHECK(ac.fmt == Rdna2Format::VOP3 && ac.opcode == 0x30Fu &&
          ac.sdst.kind == OperandKind::SGPR && ac.sdst.value == 4 &&
          ac.n_src == 2 && ac.src[2].kind == OperandKind::None &&
          !ac.src_abs[0] && !ac.src_abs[1] && !ac.src_abs[2],
          "VOP3B v_add_co_u32 (0x30F) decodes two sources, SDST s4, and no phantom SRC2");
    // Exact GTA V packets at the reported join and its later carry consumer. The producer's zero
    // reserved field must not decode as s0; the _co_ci_ consumer's identically valued SRC2 is real.
    const uint32_t live_addco[] = { 0xd70f0016u, 0x00021f1bu };
    const uint32_t live_addcoci[] = { 0xd5286a17u, 0x00024080u };
    const Rdna2Inst lac = rdna2_decode_one(live_addco, 2);
    const Rdna2Inst laci = rdna2_decode_one(live_addcoci, 2);
    CHECK(lac.opcode == 0x30Fu && lac.dst.kind == OperandKind::VGPR && lac.dst.value == 22 &&
          lac.sdst.kind == OperandKind::SGPR && lac.sdst.value == 0 && lac.n_src == 2 &&
          lac.src[0].kind == OperandKind::VGPR && lac.src[0].value == 27 &&
          lac.src[1].kind == OperandKind::VGPR && lac.src[1].value == 15 &&
          lac.src[2].kind == OperandKind::None &&
          laci.opcode == 0x128u && laci.sdst.value == 106 && laci.n_src == 3 &&
          laci.src[2].kind == OperandKind::SGPR && laci.src[2].value == 0,
          "exact GTA V VOP3B producer has two sources while its carry consumer has three");
    // The no-carry-in subtract forms have the same two-source VOP3B layout. Keep their decoded
    // source inventory aligned with emission as well; only the 0x128..0x12A _co_ci_ family has an
    // architectural carry-in in SRC2.
    const uint32_t subco[] = { 0xd7100000u | (4u << 8), 0x00020501u };
    const uint32_t subrevco[] = { 0xd7190000u | (4u << 8), 0x00020501u };
    const Rdna2Inst sc = rdna2_decode_one(subco, 2);
    const Rdna2Inst src = rdna2_decode_one(subrevco, 2);
    CHECK(sc.opcode == 0x310u && sc.n_src == 2 && sc.src[2].kind == OperandKind::None &&
          src.opcode == 0x319u && src.n_src == 2 && src.src[2].kind == OperandKind::None,
          "VOP3B v_sub/subrev_co_u32 decode two sources and no phantom SRC2");
    // DS GDS flag is dword0 bit 17 (llvm-mc gfx1030: ds_add_u32 gds = 0xd8020000 vs 0xd8000000;
    // ds_append gds = 0xd8fa0000 vs plain 0xd8f80000).
    const uint32_t ds_plain[] = { 0xd8000000u, 0x00000201u };
    const uint32_t ds_gds[]   = { 0xd8020000u, 0x00000201u };
    CHECK(!rdna2_decode_one(ds_plain, 2).ds_gds && rdna2_decode_one(ds_gds, 2).ds_gds,
          "DS GDS flag (bit 17) decodes: ds_add_u32 vs ds_add_u32 gds");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
