// test_rdna2_decode — validates the RDNA2 instruction-stream walker (src/gpu/rdna2_decode.cpp)
// against instructions assembled by llvm-mc for gfx1030 (authoritative encodings, not hand-rolled).
// The stream mixes every major encoding class + inline literals + S_ENDPGM; the walker must classify
// each instruction's format, compute its length (incl. literals), and terminate at S_ENDPGM.
#include "../src/gpu/rdna2_decode.hpp"
#include <cstdio>
#include <cstdint>

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
    // VOP SDWA/DPP forms carry a mandatory 2nd (control) dword — the decoder must count it (miss it and
    // the whole downstream stream mis-aligns) and flag has_modifier so the recompiler rejects it.
    // Encodings from llvm-mc gfx1030: SDWA src0=0xf9, DPP16 src0=0xfa, DPP8 src0=0xe9.
    const uint32_t sdwa[] = { 0x340008f9u, 0x00861682u };    // v_lshlrev_b32_sdwa v0, 2, v4 ...
    Rdna2Inst sd = rdna2_decode_one(sdwa, 2);
    CHECK(sd.fmt == Rdna2Format::VOP2 && sd.len_dwords == 2 && sd.has_modifier,
          "VOP2 SDWA form is 2 dwords and flagged has_modifier");
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
    CHECK(dp.fmt == Rdna2Format::VOP2 && dp.len_dwords == 2 && dp.has_modifier,
          "VOP2 DPP16 form is 2 dwords and flagged has_modifier");
    const uint32_t ngg_row_shift[] = { 0x4a1e1efau, 0xff09110fu };
    Rdna2Inst nrs = rdna2_decode_one(ngg_row_shift, 2);
    CHECK(nrs.fmt == Rdna2Format::VOP2 && nrs.opcode == 0x25u && !nrs.has_modifier &&
          nrs.has_dpp && nrs.dpp_ctrl == 0x111u && isV(nrs.src[0], 15) && isV(nrs.src[1], 15),
          "Astro NGG bounded DPP row-right add is admitted exactly");
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
    CHECK(n3.fmt == Rdna2Format::MIMG && n3.opcode == 0x00u && n3.len_dwords == 3 && n3.mimg_dim == 2u &&
          (n3.words[1] & 0xFFu) == 0u && (n3.words[2] & 0xFFu) == 7u && ((n3.words[2] >> 8) & 0xFFu) == 3u,
          "NSA MIMG 3D captures the extra address dword; coords decode to v0,v7,v3");
    // Astro Bot's live image_atomic_swap packet. Bit 13 is GLC: atomically exchange v9 with the
    // R32_UINT texel at (v0,v1), returning the pre-operation value to v9.
    const uint32_t mimg_atomic_swap[] = { 0xf03c2108u, 0x00000900u };
    Rdna2Inst atomic_swap = rdna2_decode_one(mimg_atomic_swap, 2);
    CHECK(atomic_swap.fmt == Rdna2Format::MIMG && atomic_swap.opcode == 0x0fu &&
          atomic_swap.mimg_dim == 1u && atomic_swap.mimg_dmask == 1u && atomic_swap.mimg_glc &&
          atomic_swap.dst.value == 9 && atomic_swap.src[0].value == 0 && atomic_swap.src[1].value == 0,
          "Astro image_atomic_swap decodes 2D/R32 data and the return-pre-op GLC flag");

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
          !ac.src_abs[0] && !ac.src_abs[1] && !ac.src_abs[2],
          "VOP3B v_add_co_u32 (0x30F) decodes SDST s4 and clears the mis-read abs bits");
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
