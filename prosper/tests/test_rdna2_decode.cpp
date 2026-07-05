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
    // SRSRC s8 (×4), offen bit set in `literal`.
    const uint32_t mubuf[] = { 0xe0381000u, 0x80020402u };
    Rdna2Inst mb = rdna2_decode_one(mubuf, 2);
    CHECK(mb.fmt == Rdna2Format::MUBUF && mb.opcode == 0xeu && isV(mb.dst, 4) && isV(mb.src[0], 2) &&
          isS(mb.src[1], 8) && ((mb.literal >> 12) & 1u), "buffer_load_dwordx4 MUBUF op/VDATA/VADDR/SRSRC/offen");
    // VOP SDWA/DPP forms carry a mandatory 2nd (control) dword — the decoder must count it (miss it and
    // the whole downstream stream mis-aligns) and flag has_modifier so the recompiler rejects it.
    // Encodings from llvm-mc gfx1030: SDWA src0=0xf9, DPP16 src0=0xfa, DPP8 src0=0xe9.
    const uint32_t sdwa[] = { 0x340008f9u, 0x00861682u };    // v_lshlrev_b32_sdwa v0, 2, v4 ...
    Rdna2Inst sd = rdna2_decode_one(sdwa, 2);
    CHECK(sd.fmt == Rdna2Format::VOP2 && sd.len_dwords == 2 && sd.has_modifier,
          "VOP2 SDWA form is 2 dwords and flagged has_modifier");
    const uint32_t dpp16[] = { 0x4a0e0cfau, 0xff011106u };   // v_add_nc_u32_dpp v7, v6, v6 row_shr:1
    Rdna2Inst dp = rdna2_decode_one(dpp16, 2);
    CHECK(dp.fmt == Rdna2Format::VOP2 && dp.len_dwords == 2 && dp.has_modifier,
          "VOP2 DPP16 form is 2 dwords and flagged has_modifier");
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

    // inline-constant field decode: SGPR106 special, field 129 -> +1, 193 -> -1, 242 -> 1.0f
    CHECK(decode_src_field(0).kind == OperandKind::SGPR && decode_src_field(0).value == 0, "field 0 -> SGPR0");
    CHECK(decode_src_field(257).kind == OperandKind::VGPR && decode_src_field(257).value == 1, "field 257 -> VGPR1");
    CHECK(decode_src_field(129).kind == OperandKind::InlineInt && decode_src_field(129).value == 1, "field 129 -> int +1");
    CHECK(decode_src_field(193).kind == OperandKind::InlineInt && decode_src_field(193).value == -1, "field 193 -> int -1");
    CHECK(decode_src_field(242).kind == OperandKind::InlineFloat && inline_float_value(242) == 1.0f, "field 242 -> float 1.0");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
