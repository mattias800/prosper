// test_dynfetch_fold — unit tests for the wave-uniform scalar const-fold interpreter behind
// bindless-dynamic vertex-fetch resolution (resolve_dynamic_fetch, gpu_executor.cpp). The fold's
// s_bfe_u64 must read a 64-bit src0 by the RDNA2 scalar-operand rules (#155): an SGPR pair; a
// SIGN-extended integer inline constant; a zero-extended 32-bit literal; and an inline FLOAT
// constant reads as a 64-bit double, which the fold does not model — it must stay UNKNOWN, never
// fold as a wrong value. A wrong fold flows into the s_cselect tail and the emitted vertex
// descriptor (a confidently-wrong V#).
//
// Observable output: the V# a buffer_load_format_x resolves to. Each kernel builds s[8:11] from
// tracked values, routing the s_bfe_u64 result into V#.dword3 (desc_v3). All encodings assembled /
// round-trip verified with llvm-mc -mcpu=gfx1030.
#include "../src/gpu/gpu_execute.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include <cstdio>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_dynfetch_fold ==\n");

    uint32_t readable_probe = 0;
    CHECK(!guest_readable(0, sizeof(uint32_t)), "null guest address is not readable");
    CHECK(guest_readable((uint64_t)(uintptr_t)&readable_probe, sizeof(readable_probe)),
          "mapped host/guest address is readable");

    // Kernel 1: negative INLINE INT src0 sign-extends to 64 bits.
    //   s_mov_b32 s8, 0x1000 | s_mov_b32 s9, 0 | s_mov_b32 s10, 64
    //   s_bfe_u64 s[12:13], -1, 0x80028      ; bits [47:40] of 0xFFFF..FF = 0xFF (hi dword all-ones)
    //   s_mov_b32 s11, s12
    //   buffer_load_format_x v1, v0, s[8:11], 0 idxen
    const uint32_t k1[] = {
        0xBE8803FFu, 0x00001000u,   // s_mov_b32 s8, 0x1000
        0xBE890380u,                // s_mov_b32 s9, 0
        0xBE8A03C0u,                // s_mov_b32 s10, 64
        0x948CFFC1u, 0x00080028u,   // s_bfe_u64 s[12:13], -1, 0x80028 (off=40, wid=8)
        0xBE8B030Cu,                // s_mov_b32 s11, s12
        0xE0002000u, 0x80020100u,   // buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,                // s_endpgm
    };
    std::vector<DynFetch> f1 = resolve_dynamic_fetch(k1, sizeof(k1)/sizeof(k1[0]), nullptr, 0, 0);
    CHECK(f1.size() == 1, "kernel 1 resolves its fetch (all V# dwords known)");
    CHECK(f1.size() == 1 && f1[0].srsrc == 8, "kernel 1 SRSRC = s8");
    CHECK(f1.size() == 1 && f1[0].desc_v3 == 0xFFu,
          "s_bfe_u64 of inline -1 sign-extends: bits[47:40] fold to 0xFF (not 0)");

    // Kernel 2: 32-bit LITERAL src0 zero-extends — a field entirely in the high dword folds to 0.
    //   s_mov_b32 s14, 0x80020 | s_bfe_u64 s[12:13], 0x12345678, s14   ; bits [39:32] of zext = 0
    const uint32_t k2[] = {
        0xBE8803FFu, 0x00001000u,   // s_mov_b32 s8, 0x1000
        0xBE890380u,                // s_mov_b32 s9, 0
        0xBE8A03C0u,                // s_mov_b32 s10, 64
        0xBE8E03FFu, 0x00080020u,   // s_mov_b32 s14, 0x80020 (off=32, wid=8)
        0x948C0EFFu, 0x12345678u,   // s_bfe_u64 s[12:13], 0x12345678, s14
        0xBE8B030Cu,                // s_mov_b32 s11, s12
        0xE0002000u, 0x80020100u,   // buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,                // s_endpgm
    };
    std::vector<DynFetch> f2 = resolve_dynamic_fetch(k2, sizeof(k2)/sizeof(k2[0]), nullptr, 0, 0);
    CHECK(f2.size() == 1 && f2[0].desc_v3 == 0u,
          "s_bfe_u64 of a 32-bit literal zero-extends: bits[39:32] fold to 0");

    // Kernel 3: inline FLOAT src0 (1.0) reads as a 64-bit DOUBLE — the fold does not model that
    // encoding, so the destination must stay UNKNOWN and the fetch must NOT resolve (an unknown V#
    // dword is dropped, never fabricated).
    const uint32_t k3[] = {
        0xBE8803FFu, 0x00001000u,   // s_mov_b32 s8, 0x1000
        0xBE890380u,                // s_mov_b32 s9, 0
        0xBE8A03C0u,                // s_mov_b32 s10, 64
        0x948CFFF2u, 0x00080028u,   // s_bfe_u64 s[12:13], 1.0, 0x80028
        0xBE8B030Cu,                // s_mov_b32 s11, s12
        0xE0002000u, 0x80020100u,   // buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,                // s_endpgm
    };
    std::vector<DynFetch> f3 = resolve_dynamic_fetch(k3, sizeof(k3)/sizeof(k3[0]), nullptr, 0, 0);
    CHECK(f3.empty(), "s_bfe_u64 of an inline FLOAT does not fold (V# dword unknown -> fetch unresolved)");

    // Kernel 4: SEED-V# fallback (#294). The V# sits directly in the user-data SGPRs (s[8:11] =
    // user block index 0..3, user_sgpr_base 8) and an ALU patch makes one dword UNKNOWN (s_mov from
    // an untracked SGPR — DOLL's format patch does this via an s_cselect on an NGG system SGPR).
    // Neither 'patched' nor 'descr' resolves; the fetch must fall back to the SEED (load-time /
    // pre-patch) descriptor. No s_load touches s[8:11], so the reloaded guard stays clear.
    const uint32_t k4[] = {
        0xBE8B0332u,                // s_mov_b32 s11, s50 (s50 unknown -> s11 unknown; ALU, not a reload)
        0xE0002000u, 0x80020100u,   // buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,                // s_endpgm
    };
    const uint32_t seed4[4] = { 0x00020000u, 0x00100000u, 64u, 0u };   // base=0x20000 stride=16 recs=64
    std::vector<DynFetch> f4 = resolve_dynamic_fetch(k4, sizeof(k4)/sizeof(k4[0]), seed4, 4, 8);
    CHECK(f4.size() == 1, "kernel 4 seed-V# fallback resolves the fetch");
    CHECK(f4.size() == 1 && f4[0].desc.base == 0x20000ull && f4[0].desc.stride == 16u,
          "kernel 4 fallback V# = the SEED (user-data) descriptor");

    // Kernel 4b: same, but s11 was RELOADED from memory (s_load_dword) — the stale seed must NOT be
    // used (the register no longer holds user data). The load's address is unknown (s[40:41]
    // untracked), so the value is unknown too and the fetch must stay unresolved.
    const uint32_t k4b[] = {
        0xF40002D4u, 0xFA000000u,   // s_load_dword s11, s[40:41], 0x0 (unknown base -> s11 reloaded+unknown)
        0xE0002000u, 0x80020100u,   // buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,                // s_endpgm
    };
    std::vector<DynFetch> f4b = resolve_dynamic_fetch(k4b, sizeof(k4b)/sizeof(k4b[0]), seed4, 4, 8);
    CHECK(f4b.empty(), "kernel 4b reloaded SGPR blocks the seed fallback (no fabricated V#)");

    // #370 review: a fully-known dynamic packed V# used to decode its unsupported DST_SEL as Unknown,
    // then build_stage_table's legacy Unknown->Float32 fallback resurrected it as four raw dwords.
    // resolve_dynamic_fetch must omit deliberate packed rejections while retaining identity descriptors.
    const uint32_t k4packed[] = {
        0xE00C2000u, 0x80020100u,   // buffer_load_format_xyzw v[1:4], v0, s[8:11], 0 idxen
        0xBF810000u,
    };
    const uint32_t packed_permuted[4] = {
        0x00020000u, 0x00040000u, 64u, (50u << 12) | 0x977u, // A/B/G/R
    };
    const uint32_t packed_constant[4] = {
        0x00020000u, 0x00040000u, 64u, 50u << 12,           // constant-zero selectors
    };
    const uint32_t packed_scaled[4] = {
        0x00020000u, 0x00040000u, 64u, (52u << 12) | 0xFACu, // unsupported USCALED
    };
    CHECK(resolve_dynamic_fetch(k4packed, sizeof(k4packed)/sizeof(k4packed[0]),
                                packed_permuted, 4, 8).empty() &&
          resolve_dynamic_fetch(k4packed, sizeof(k4packed)/sizeof(k4packed[0]),
                                packed_constant, 4, 8).empty() &&
          resolve_dynamic_fetch(k4packed, sizeof(k4packed)/sizeof(k4packed[0]),
                                packed_scaled, 4, 8).empty(),
          "dynamic packed permutations/constants/scaled formats cannot reach the Float32 fallback");
    const uint32_t packed_identity[4] = {
        0x00020000u, 0x00040000u, 64u, (50u << 12) | 0xFACu,
    };
    std::vector<DynFetch> f4packed = resolve_dynamic_fetch(
        k4packed, sizeof(k4packed)/sizeof(k4packed[0]), packed_identity, 4, 8);
    CHECK(f4packed.size() == 1 && f4packed[0].desc.format == DataFormat::Unorm2_10_10_10 &&
          !f4packed[0].desc.forbid_unknown_fallback,
          "dynamic identity packed V# reaches the real unpack format instead of Float32");

    // #370 review: entry metadata may legitimately publish an identity packed direct V# while the
    // shader rewrites only dword 3 before the fetch. If that live selector is unsupported, dynamic
    // resolution omits it. Recompilation must not then fall back to the stale entry-time identity V#.
    ShaderResourceTable packed_entry_table;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer;
      vb.format = DataFormat::Unorm2_10_10_10; vb.num_components = 4;
      vb.binding = 3; vb.stride = 4; vb.sgpr_base = 20;
      packed_entry_table.resources.push_back(vb); }
    const uint32_t rejected_v3[] = {
        (50u << 12) | 0x977u, // A/B/G/R permutation
        50u << 12,            // constant-zero selectors
        (52u << 12) | 0xFACu, // unsupported USCALED conversion
    };
    bool rejected_rewrites_stay_closed = true;
    for (uint32_t v3 : rejected_v3) {
        const uint32_t code[] = {
            0xBE9703FFu, v3,           // s_mov_b32 s23, literal (rewrite V#.dword3 only)
            0xE00C2000u, 0x80050100u, // buffer_load_format_xyzw v[1:4], v0, s[20:23], 0 idxen
            0xBF810000u,
        };
        rejected_rewrites_stay_closed &=
            resolve_dynamic_fetch(code, sizeof(code)/sizeof(code[0]), packed_identity, 4, 20).empty();
        rejected_rewrites_stay_closed &=
            recompile_valu(code, sizeof(code)/sizeof(code[0]), 1, 1, &packed_entry_table).empty();
    }
    CHECK(rejected_rewrites_stay_closed,
          "rejected packed SRSRC rewrites cannot compile through the stale direct identity V#");

    // A shader write remains provenance even when its value cannot be represented. This B64 move
    // reads an untracked pair, so the scalar SSA entries for s[20:21] are erased. The later fetch must
    // still reject instead of resurrecting the entry-time direct V# merely because the map is empty.
    // Encoding round-tripped with llvm-mc gfx1030: s_mov_b64 s[20:21], s[50:51].
    const uint32_t erased_srsrc[] = {
        0xBE940432u,
        0xE00C2000u, 0x80050100u, // buffer_load_format_xyzw v[1:4], v0, s[20:23], 0 idxen
        0xBF810000u,
    };
    CHECK(recompile_valu(erased_srsrc, sizeof(erased_srsrc)/sizeof(erased_srsrc[0]),
                         1, 1, &packed_entry_table).empty(),
          "unrepresentable SRSRC writes cannot resurrect an entry-time direct V#");

    // The same invalidation is a MAY-write property at structured joins: the untouched else edge
    // cannot make entry metadata safe on the then edge that erased s[20:21]. Branch offsets and all
    // instruction encodings were round-tripped with llvm-mc gfx1030.
    const uint32_t branch_erased_srsrc[] = {
        0xBF060000u,             // s_cmp_eq_u32 s0, s0
        0xBF840002u,             // s_cbranch_scc0 else
        0xBE940432u,             // then: s_mov_b64 s[20:21], s[50:51]
        0xBF820001u,             // s_branch merge
        0xBE840380u,             // else: s_mov_b32 s4, 0
        0xE00C2000u, 0x80050100u,
        0xBF810000u,
    };
    CHECK(recompile_valu(branch_erased_srsrc,
                         sizeof(branch_erased_srsrc)/sizeof(branch_erased_srsrc[0]),
                         1, 1, &packed_entry_table).empty(),
          "structured joins preserve erased SRSRC write provenance from either arm");

    // Vector instructions can also have explicit scalar-pair destinations. Both encodings overwrite
    // s[20:21], so neither may leave the entry-time packed V# usable by the following format fetch.
    // Round-tripped with llvm-mc gfx1030.
    const uint32_t vopc_srsrc[] = {
        0x7C0400F9u, 0x06069400u, // v_cmp_eq_f32_sdwa s20, v0, v0 (SDST pair)
        0xE00C2000u, 0x80050100u,
        0xBF810000u,
    };
    CHECK(recompile_valu(vopc_srsrc, sizeof(vopc_srsrc)/sizeof(vopc_srsrc[0]),
                         1, 1, &packed_entry_table).empty(),
          "explicit VOPC SGPR-pair writes invalidate a direct packed V#");
    const uint32_t vop3_carry_srsrc[] = {
        0xD5761401u, 0x040A0100u, // v_mad_u64_u32 v[1:2], s20, v0, v0, v[2:3]
        0xE00C2000u, 0x80050100u,
        0xBF810000u,
    };
    CHECK(recompile_valu(vop3_carry_srsrc,
                         sizeof(vop3_carry_srsrc)/sizeof(vop3_carry_srsrc[0]),
                         4, 1, &packed_entry_table).empty(),
          "VOP3 carry-out SGPR-pair writes invalidate a direct packed V#");

    // The same pair writes must invalidate an INDIRECT descriptor loaded from the SRT. The physical
    // SGPR words no longer contain the s_load result, so retaining only its old sreg_srt tag would
    // bind the pre-overwrite resource even though direct-entry fallback is already disabled.
    ShaderResourceTable packed_srt_table;
    { ShaderResource vb{}; vb.cls = ResourceClass::VertexBuffer;
      vb.format = DataFormat::Unorm2_10_10_10; vb.num_components = 4;
      vb.binding = 3; vb.stride = 4; vb.srt_offset = 0x40;
      packed_srt_table.resources.push_back(vb); }
    const uint32_t vopc_srt_srsrc[] = {
        0xF4080504u, 0xFA000040u, // s_load_dwordx4 s[20:23], s[8:9], 0x40
        0x7C0400F9u, 0x06069400u, // v_cmp_eq_f32_sdwa s20, v0, v0
        0xE00C2000u, 0x80050100u,
        0xBF810000u,
    };
    CHECK(recompile_valu(vopc_srt_srsrc, std::size(vopc_srt_srsrc),
                         1, 1, &packed_srt_table).empty(),
          "explicit VOPC SGPR-pair writes invalidate stale SRT descriptor provenance");
    const uint32_t vop3_srt_srsrc[] = {
        0xF4080504u, 0xFA000040u,
        0xD5761401u, 0x040A0100u, // v_mad_u64_u32 carry-out -> s[20:21]
        0xE00C2000u, 0x80050100u,
        0xBF810000u,
    };
    CHECK(recompile_valu(vop3_srt_srsrc, std::size(vop3_srt_srsrc),
                         4, 1, &packed_srt_table).empty(),
          "VOP3 carry-out SGPR-pair writes invalidate stale SRT descriptor provenance");
    const uint32_t vopc_srt_high_srsrc[] = {
        0xF4080504u, 0xFA000040u,
        0x7C0400F9u, 0x06069600u, // v_cmp_eq_f32_sdwa s22, v0, v0
        0xE00C2000u, 0x80050100u,
        0xBF810000u,
    };
    CHECK(recompile_valu(vopc_srt_high_srsrc, std::size(vopc_srt_high_srsrc),
                         1, 1, &packed_srt_table).empty(),
          "an upper-half SRSRC write invalidates the complete SRT descriptor tag");

    // A loop header is compiled once but executes again after every back-edge. If any reachable
    // body instruction overwrites the header fetch's SRSRC, the entry-time direct V# is valid only
    // for iteration one and cannot be baked into the generated header. Reject both supported loop
    // structurizers until a loop-varying descriptor can be represented exactly.
    const uint32_t counted_header_srsrc[] = {
        0xBE800380u, 0xBE820382u,                         // s0=0, s2=2
        0xE00C2000u, 0x80050100u,                         // loop: packed fetch via s[20:23]
        0xBF0A0200u, 0xBF840003u,                         // s0<s2; exit when false
        0xBE940380u, 0x80008100u, 0xBF82FFF9u,            // overwrite s20; ++s0; back-edge
        0xBF810000u,
    };
    CHECK(recompile_valu(counted_header_srsrc, std::size(counted_header_srsrc),
                         1, 1, &packed_entry_table).empty(),
          "counted-loop header cannot reuse a direct V# overwritten on its back-edge");
    const uint32_t divergent_header_srsrc[] = {
        0xBE800380u, 0xBE820382u, 0x7E0A0202u,            // s0=0, s2=2, uniform v5=s2
        0xE00C2000u, 0x80050100u,                         // loop: packed fetch via s[20:23]
        0x7D820A00u, 0xBF860003u,                         // uniform vcc=(s0<v5); vccz exit
        0xBE940380u, 0x80008100u, 0xBF82FFF9u,            // overwrite s20; ++s0; back-edge
        0xBF810000u,
    };
    CHECK(recompile_valu(divergent_header_srsrc, std::size(divergent_header_srsrc),
                         1, 1, &packed_entry_table).empty(),
          "divergent-loop header cannot reuse a direct V# overwritten on its back-edge");
    const uint32_t counted_header_srt_high_srsrc[] = {
        0xF4080504u, 0xFA000040u,                         // SRT V# -> s[20:23]
        0xBE800380u, 0xBE820382u,                         // s0=0, s2=2
        0xE00C2000u, 0x80050100u,                         // loop: packed fetch via s[20:23]
        0xBF0A0200u, 0xBF840003u,                         // s0<s2; exit when false
        0xBE960380u, 0x80008100u, 0xBF82FFF9u,            // overwrite s22; ++s0; back-edge
        0xBF810000u,
    };
    CHECK(recompile_valu(counted_header_srt_high_srsrc,
                         std::size(counted_header_srt_high_srsrc),
                         1, 1, &packed_srt_table).empty(),
          "loop upper-word writes invalidate the complete SRT descriptor tag at the header");

    // Kernel 5: descriptor-TABLE uses (#294). s[8:9] = a pointer to a host-memory table; the shader
    // s_loads an 8-dword T# (imm 0x40) + a 4-dword S#/V# (imm 0x80), consumes them via image_sample
    // (SRSRC/SSAMP) and s_buffer_load (SBASE). Each use must be reported with the load immediate as
    // its key and the dwords as loaded. (Encodings llvm-mc gfx1030 round-trip verified.)
    static uint32_t table[64];
    for (int i = 0; i < 64; i++) table[i] = 0xD0000000u + (uint32_t)i;
    // The V# at +0x80 must be a READABLE buffer for the interpreter's s_buffer_load data read:
    // point it back at `table` itself (num_records=256 bytes, stride 0).
    uint64_t tbase = (uint64_t)(uintptr_t)table;
    table[32] = (uint32_t)tbase; table[33] = (uint32_t)(tbase >> 32) & 0xFFFFu; table[34] = 256u; table[35] = 0u;
    const uint32_t k5[] = {
        0xF40C0304u, 0xFA000040u,   // s_load_dwordx8 s[12:19], s[8:9], 0x40
        0xF4080504u, 0xFA000080u,   // s_load_dwordx4 s[20:23], s[8:9], 0x80
        0xF0800F08u, 0x00A30000u,   // image_sample v[0:3], v[0:1], s[12:19], s[20:23] dmask:0xf 2D
        0xF420060Au, 0xFA000010u,   // s_buffer_load_dword s24, s[20:23], 0x10
        0xBF810000u,                // s_endpgm
    };
    uint32_t seed5[2] = { (uint32_t)tbase, (uint32_t)(tbase >> 32) };
    std::vector<SrtUse> uses;
    resolve_dynamic_fetch(k5, sizeof(k5)/sizeof(k5[0]), seed5, 2, 8, &uses);
    bool have_tex = false, have_cbuf = false, tex_ok = false, samp_ok = false, cbuf_ok = false;
    for (const auto& u : uses) {
        if (u.kind == 0 && u.key == 0x40) { have_tex = true;
            tex_ok  = (u.t8[0] == table[16] && u.t8[7] == table[23]);
            samp_ok = (u.has_samp && u.s4[0] == table[32] && u.s4[3] == table[35]); }
        if (u.kind == 1 && u.key == 0x80) { have_cbuf = true;
            cbuf_ok = (u.v4[0] == table[32] && u.v4[2] == 256u); }
    }
    CHECK(have_tex,  "kernel 5 image_sample reports a Texture table-use keyed by the T# load imm");
    CHECK(tex_ok,    "kernel 5 T# dwords match the table as loaded");
    CHECK(samp_ok,   "kernel 5 paired SSAMP S# resolved alongside the T#");
    CHECK(have_cbuf, "kernel 5 s_buffer_load reports a ConstantBuffer table-use keyed by the V# load imm");
    CHECK(cbuf_ok,   "kernel 5 V# dwords match the table as loaded");

    // Kernel 5n: an s_buffer_load_dwordx8 result is typeless SGPR data. A later scalar buffer load
    // may consume its first four words as a nested V#, as DOLL's post-process shader does. Since the
    // V# came through another buffer descriptor it has no immediate key; preserve it by consumer pc.
    static uint32_t nested_payload[16];
    static uint32_t nested_descriptors[8];
    uint64_t nested_payload_base = (uint64_t)(uintptr_t)nested_payload;
    nested_descriptors[0] = (uint32_t)nested_payload_base;
    nested_descriptors[1] = (uint32_t)(nested_payload_base >> 32) & 0xFFFFu;
    nested_descriptors[2] = sizeof(nested_payload);
    nested_descriptors[3] = 0u;
    uint64_t nested_table_base = (uint64_t)(uintptr_t)nested_descriptors;
    const uint32_t k5n[] = {
        0xF42C0404u, 0xFA000000u,   // s_buffer_load_dwordx8 s[16:23], s[8:11], 0
        0xF4280608u, 0xFA000000u,   // s_buffer_load_dwordx4 s[24:27], s[16:19], 0
        0xBF810000u,
    };
    const uint32_t seed5n[4] = {
        (uint32_t)nested_table_base, (uint32_t)(nested_table_base >> 32) & 0xFFFFu,
        sizeof(nested_descriptors), 0u,
    };
    std::vector<SrtUse> nested_uses;
    resolve_dynamic_fetch(k5n, sizeof(k5n)/sizeof(k5n[0]), seed5n, 4, 8, &nested_uses);
    bool nested_ok = false;
    for (const auto& u : nested_uses) {
        if (u.kind == 1 && u.key == 0xFFFFFFFFu && u.use_pc == 2 && u.required_size == 16 &&
            u.v4[0] == nested_descriptors[0] && u.v4[1] == nested_descriptors[1] &&
            u.v4[2] == nested_descriptors[2] && u.v4[3] == nested_descriptors[3])
            nested_ok = true;
    }
    CHECK(nested_ok, "kernel 5n nested s_buffer V# resolves by the consuming instruction pc");

    // Kernel 5s: UE's large post shaders spill descriptor words into fixed lanes of one VGPR and
    // later restore them with v_readlane before a scalar buffer load. The restored V# has no static
    // SRT tag in the recompiler, so the fold must preserve its words and key it by the consumer pc.
    const uint32_t k5s[] = {
        0xD7610024u, 0x00010848u,   // v_writelane_b32 v36, s72, 4
        0xD7610024u, 0x00010A49u,   // v_writelane_b32 v36, s73, 5
        0xD7610024u, 0x00010C4Au,   // v_writelane_b32 v36, s74, 6
        0xD7610024u, 0x00010E4Bu,   // v_writelane_b32 v36, s75, 7
        0xD7600028u, 0x00010924u,   // v_readlane_b32 s40, v36, 4
        0xD7600029u, 0x00010B24u,   // v_readlane_b32 s41, v36, 5
        0xD760002Au, 0x00010D24u,   // v_readlane_b32 s42, v36, 6
        0xD760002Bu, 0x00010F24u,   // v_readlane_b32 s43, v36, 7
        0xF4281014u, 0xFA000000u,   // s_buffer_load_dwordx4 s[64:67], s[40:43], 0
        0xBF810000u,
    };
    std::vector<SrtUse> spill_uses;
    resolve_dynamic_fetch(k5s, sizeof(k5s)/sizeof(k5s[0]), seed5n, 4, 72, &spill_uses);
    bool spill_ok = false;
    for (const auto& u : spill_uses) {
        if (u.kind == 1 && u.key == 0xFFFFFFFFu && u.use_pc == 16 && u.required_size == 16 &&
            u.v4[0] == seed5n[0] && u.v4[1] == seed5n[1] &&
            u.v4[2] == seed5n[2] && u.v4[3] == seed5n[3])
            spill_ok = true;
    }
    CHECK(spill_ok, "kernel 5s scalar-spilled V# resolves by the consuming instruction pc");

    // Kernel 5sr: an ordinary write to the spill VGPR replaces every lane. Restoring after that
    // write must not resurrect descriptor words saved by earlier v_writelane instructions.
    const uint32_t k5sr[] = {
        0xD7610024u, 0x00010848u,   // v_writelane_b32 v36, s72, 4
        0xD7610024u, 0x00010A49u,   // v_writelane_b32 v36, s73, 5
        0xD7610024u, 0x00010C4Au,   // v_writelane_b32 v36, s74, 6
        0xD7610024u, 0x00010E4Bu,   // v_writelane_b32 v36, s75, 7
        0x7E4802F2u,                // v_mov_b32 v36, 1.0
        0xD7600028u, 0x00010924u,   // v_readlane_b32 s40, v36, 4
        0xD7600029u, 0x00010B24u,   // v_readlane_b32 s41, v36, 5
        0xD760002Au, 0x00010D24u,   // v_readlane_b32 s42, v36, 6
        0xD760002Bu, 0x00010F24u,   // v_readlane_b32 s43, v36, 7
        0xF4281014u, 0xFA000000u,   // s_buffer_load_dwordx4 s[64:67], s[40:43], 0
        0xBF810000u,
    };
    std::vector<SrtUse> recycled_spill_uses;
    resolve_dynamic_fetch(k5sr, sizeof(k5sr)/sizeof(k5sr[0]), seed5n, 4, 72,
                          &recycled_spill_uses);
    CHECK(recycled_spill_uses.empty(),
          "kernel 5sr ordinary VGPR write invalidates scalar spill lanes");

    // Kernel 5d: direct T#/S# sharps already occupy the initial PS user SGPRs, so no s_load creates
    // descr8/descr snapshots. The MIMG use must still receive a pc-keyed texture and paired sampler.
    const uint32_t k5d[] = {
        0xF0800F08u, 0x00400000u,   // image_sample v[0:3], v[0:1], s[0:7], s[8:11] dmask:0xf 2D
        0xBF810000u,                // s_endpgm
    };
    const uint32_t seed5d[12] = {
        0x20753500u, 0xC2400000u, 0x021BC3BFu, 0x91B003ACu, // T#: 0x2075350000, 3840x2160, 2D
        0x00000000u, 0x00700000u, 0x006B0000u, 0x00205072u,
        0x00000092u, 0x00FFF000u, 0x06500000u, 0x00000000u, // S#
    };
    std::vector<SrtUse> direct_uses;
    resolve_dynamic_fetch(k5d, sizeof(k5d)/sizeof(k5d[0]), seed5d, 12, 0, &direct_uses);
    CHECK(direct_uses.size() == 1 && direct_uses[0].kind == 0 &&
          direct_uses[0].key == 0xFFFFFFFFu && direct_uses[0].use_pc == 0,
          "direct user-SGPR T# resolves the image_sample by exact pc");
    CHECK(direct_uses.size() == 1 && direct_uses[0].t8[0] == seed5d[0] &&
          direct_uses[0].t8[7] == seed5d[7] && direct_uses[0].has_samp &&
          direct_uses[0].s4[0] == seed5d[8] && direct_uses[0].s4[3] == seed5d[11],
          "direct user-SGPR T#/S# dwords are preserved");

    // Evergate's title PS uses a bounded PC-relative dispatch. An omitted alternative reloads the
    // same T# SGPRs that the selected arm consumes directly; a linear fold used to walk that reload
    // and attach the alternative texture to the selected image_sample's pc. Specialize the fold to
    // target pc19, exactly as the fragment recompiler specializes the executable instruction stream.
    static uint32_t dispatch_selector[5] = {0, 0, 0, 0, 1}; // add -1 -> table index 0 -> pc19
    static uint32_t alternative_t8[8];
    for (uint32_t i = 0; i < 8; ++i) alternative_t8[i] = seed5d[i];
    alternative_t8[0] ^= 0x10000000u; // distinguish the omitted arm's T# from the direct selected T#

    uint32_t dispatch_sgprs[32]{};
    for (uint32_t i = 0; i < 12; ++i) dispatch_sgprs[i] = seed5d[i];
    const uint64_t selector_base = (uint64_t)(uintptr_t)dispatch_selector;
    dispatch_sgprs[24] = (uint32_t)selector_base;
    dispatch_sgprs[25] = (uint32_t)(selector_base >> 32) & 0xFFFFu;
    dispatch_sgprs[26] = 5u;
    dispatch_sgprs[27] = 4u << 12;
    const uint64_t alternative_base = (uint64_t)(uintptr_t)alternative_t8;
    dispatch_sgprs[28] = (uint32_t)alternative_base;
    dispatch_sgprs[29] = (uint32_t)(alternative_base >> 32) & 0xFFFFu;
    dispatch_sgprs[30] = 8u;
    dispatch_sgprs[31] = 4u << 12;

    const uint32_t pcrel_texture_dispatch[] = {
        0xF4201A8Cu, 0xFA000010u, // pc0:  s_buffer_load_dword s106, s[24:27], 0x10
        0x816AC16Au,              // pc2:  s_add_i32 s106, s106, -1
        0x83EA826Au,              // pc3:  s_min_u32 s106, s106, 2
        0x8F6A836Au,              // pc4:  s_lshl_b32 s106, s106, 3
        0xBEA01F00u,              // pc5:  s_getpc_b64 s[32:33]
        0x802020FFu, 80u,         // pc6:  table starts at aligned pc26
        0x82212180u,              // pc8:  s_addc_u32 s33, 0, s33
        0xF4040890u, 0xD4000000u, // pc9:  s_load_dwordx2 s[34:35], s[32:33], s106
        0xBEA81F00u,              // pc11: s_getpc_b64 s[40:41]
        0x80282228u,              // pc12: s_add_u32 s40, s40, s34
        0x82292329u,              // pc13: s_addc_u32 s41, s41, s35
        0xBE802028u,              // pc14: s_setpc_b64 s[40:41]
        0xF42C000Eu, 0xFA000000u, // pc15: omitted s_buffer_load_dwordx8 s[0:7], s[28:31], 0
        0xBF8CC07Fu,              // pc17: s_waitcnt
        0xBF820003u,              // pc18: s_branch merge at pc22
        0xF0800F08u, 0x00400000u, // pc19: selected image_sample ..., s[0:7], s[8:11]
        0xBF820000u,              // pc21: s_branch merge at pc22
        0xF800180Fu, 0x00000000u, // pc22: common exp mrt0
        0xBF810000u,              // pc24: s_endpgm
        0u,                       // pc25: qword alignment
        28u, 0u,                  // pc26: target pc19, relative to pc12
        12u, 0u,                  // pc28: target pc15
        40u, 0u,                  // pc30: merge pc22
    };
    const PcrelDispatchInfo texture_dispatch = rdna2_pcrel_dispatch_info(
        pcrel_texture_dispatch,
        sizeof(pcrel_texture_dispatch) / sizeof(pcrel_texture_dispatch[0]));
    CHECK(texture_dispatch.valid && texture_dispatch.target_pcs.size() == 3 &&
          texture_dispatch.target_pcs[0] == 19,
          "texture dispatch fixture proves selected target pc19");

    std::vector<SrtUse> unspecialized_dispatch_uses;
    resolve_dynamic_fetch(pcrel_texture_dispatch,
                          sizeof(pcrel_texture_dispatch) / sizeof(pcrel_texture_dispatch[0]),
                          dispatch_sgprs, 32, 0, &unspecialized_dispatch_uses);
    bool unspecialized_contaminated = false;
    for (const auto& use : unspecialized_dispatch_uses)
        if (use.kind == 0 && use.use_pc == 19 && use.t8[0] == alternative_t8[0])
            unspecialized_contaminated = true;
    CHECK(unspecialized_contaminated,
          "linear fixture demonstrates omitted-arm texture contamination");

    std::vector<SrtUse> specialized_dispatch_uses;
    resolve_dynamic_fetch(pcrel_texture_dispatch,
                          sizeof(pcrel_texture_dispatch) / sizeof(pcrel_texture_dispatch[0]),
                          dispatch_sgprs, 32, 0, &specialized_dispatch_uses, 19);
    bool selected_uses_direct_texture = false;
    bool selected_uses_alternative_texture = false;
    for (const auto& use : specialized_dispatch_uses) {
        if (use.kind != 0 || use.use_pc != 19) continue;
        selected_uses_direct_texture |= use.t8[0] == seed5d[0] && use.t8[7] == seed5d[7];
        selected_uses_alternative_texture |= use.t8[0] == alternative_t8[0];
    }
    CHECK(selected_uses_direct_texture && !selected_uses_alternative_texture,
          "selected dispatch arm keeps direct texture provenance at image_sample pc19");

    // DOLL scene VS: an untyped buffer_load_dwordx3 uses a V# placed directly in user-data
    // s[8:11]. There is no preceding s_load and it is not a format/vertex fetch, so it needs its
    // own pc-keyed buffer use. The same eight seed SGPRs may also happen to decode as a plausible
    // T#; the consuming MUBUF instruction is definitive that these first four words are a V#.
    const uint32_t k5v[] = {
        0xE03C2000u, 0x80020003u,   // buffer_load_dwordx3 v[0:2], v3, s[8:11], 0 idxen
        0xBF810000u,
    };
    const uint32_t seed5v[4] = {
        0x00020000u, 0x00040000u, 1480u, 75u << 12, // base=0x20000 stride=4 size=5920
    };
    std::vector<SrtUse> direct_v_uses;
    resolve_dynamic_fetch(k5v, sizeof(k5v)/sizeof(k5v[0]), seed5v, 4, 8, &direct_v_uses);
    CHECK(direct_v_uses.size() == 1 && direct_v_uses[0].kind == 1 &&
          direct_v_uses[0].key == 0xFFFFFFFFu && direct_v_uses[0].use_pc == 0,
          "direct user-SGPR V# resolves raw MUBUF by exact pc");
    CHECK(direct_v_uses.size() == 1 && direct_v_uses[0].v4[0] == seed5v[0] &&
          direct_v_uses[0].v4[1] == seed5v[1] && direct_v_uses[0].v4[2] == seed5v[2],
          "direct raw-MUBUF V# preserves base/stride/record dwords");

    // #636: Dead Cells' format-copy compute places its declared source V# at s0 and its otherwise
    // undeclared destination V# at s4. Resource discovery must follow the two actual MUBUF uses:
    // the load arrives through DynFetch and the store through a pc-keyed SrtUse. No all-SGPR scan is
    // needed, and rewriting any destination dword before the store invalidates the seed descriptor.
    const uint32_t direct_copy[] = {
        0xE00C2000u, 0x80000100u,   // buffer_load_format_xyzw v[1:4], v0, s[0:3], 0 idxen
        0xE01C2000u, 0x80010101u,   // buffer_store_format_xyzw v[1:4], v1, s[4:7], 0 idxen
        0xBF810000u,
    };
    const uint32_t direct_copy_seed[8] = {
        0x00020000u, 0x00040000u, 16u, (2u << 12) | 0xFACu,
        0x00030000u, 0x00040000u, 16u, (2u << 12) | 0xFACu,
    };
    std::vector<SrtUse> direct_copy_uses;
    const std::vector<DynFetch> direct_copy_fetches = resolve_dynamic_fetch(
        direct_copy, sizeof(direct_copy) / sizeof(direct_copy[0]),
        direct_copy_seed, 8, 0, &direct_copy_uses);
    CHECK(direct_copy_fetches.size() == 1 && direct_copy_fetches[0].fetch_pc == 0 &&
          direct_copy_fetches[0].srsrc == 0 && direct_copy_fetches[0].desc.base == 0x20000,
          "#636: direct format-copy source resolves only at its load instruction");
    CHECK(direct_copy_uses.size() == 1 && direct_copy_uses[0].kind == 1 &&
          direct_copy_uses[0].key == 0xFFFFFFFFu && direct_copy_uses[0].use_pc == 2 &&
          direct_copy_uses[0].v4[0] == direct_copy_seed[4],
          "#636: Dead Cells s4 destination resolves only at its format-store instruction");

    ShaderResourceTable direct_copy_table;
    add_compute_buffer_resources(direct_copy_table, direct_copy,
                                 sizeof(direct_copy) / sizeof(direct_copy[0]),
                                 direct_copy_seed, 8);
    assign_convention_bindings(direct_copy_table, 2);
    CHECK(direct_copy_table.resources.size() == 2 &&
          direct_copy_table.by_fetch_pc(0) && direct_copy_table.by_fetch_pc(2) &&
          direct_copy_table.by_fetch_pc(0)->binding == 2 &&
          direct_copy_table.by_fetch_pc(2)->binding == 3,
          "#636: production compute discovery emits exactly the used source and destination");
    ComputeShaderConfig direct_copy_config;
    direct_copy_config.user_sgprs.assign(direct_copy_seed, direct_copy_seed + 8);
    direct_copy_config.local_x = direct_copy_config.local_y = direct_copy_config.local_z = 1;
    CHECK(!recompile_compute(direct_copy, sizeof(direct_copy) / sizeof(direct_copy[0]),
                             &direct_copy_table, direct_copy_config).empty(),
          "#636: instruction-provenance table keeps the Dead Cells format-copy dispatch realizable");

    // MTBUF's instruction format must also survive the production compute discovery path. Both V#s
    // deliberately carry a different but valid 32_FLOAT format; the encoded gfx1030 BUF_FMT 56
    // supplies Unorm8x4 for the load and store independently of descriptor dword3.
    const uint32_t direct_mtbuf_copy[] = {
        0xE9C32000u, 0x80000100u,   // tbuffer_load_format_xyzw ..., s[0:3], fmt 56, idxen
        0xE9C72000u, 0x80010101u,   // tbuffer_store_format_xyzw ..., s[4:7], fmt 56, idxen
        0xBF810000u,
    };
    const uint32_t direct_mtbuf_seed[8] = {
        0x00020000u, 0x00040000u, 16u, 22u << 12,
        0x00030000u, 0x00040000u, 16u, 22u << 12,
    };
    std::vector<SrtUse> direct_mtbuf_uses;
    const std::vector<DynFetch> direct_mtbuf_fetches = resolve_dynamic_fetch(
        direct_mtbuf_copy, std::size(direct_mtbuf_copy), direct_mtbuf_seed, 8, 0,
        &direct_mtbuf_uses);
    CHECK(direct_mtbuf_fetches.size() == 1 &&
              direct_mtbuf_fetches[0].instruction_format == 56u &&
              direct_mtbuf_fetches[0].desc.format == DataFormat::Float32,
          "MTBUF load discovery retains the instruction format over a different valid V# format");
    CHECK(direct_mtbuf_uses.size() == 1 &&
              direct_mtbuf_uses[0].instruction_format == 56u &&
              direct_mtbuf_uses[0].use_pc == 2,
          "MTBUF store discovery retains instruction format and exact consumer pc");
    ShaderResourceTable direct_mtbuf_table;
    add_compute_buffer_resources(direct_mtbuf_table, direct_mtbuf_copy,
                                 std::size(direct_mtbuf_copy), direct_mtbuf_seed, 8);
    assign_convention_bindings(direct_mtbuf_table, 2);
    bool mtbuf_formats_ok = direct_mtbuf_table.resources.size() == 2;
    for (const auto& resource : direct_mtbuf_table.resources)
        mtbuf_formats_ok &= resource.format == DataFormat::Unorm8 &&
                            resource.num_components == 4;
    ComputeShaderConfig direct_mtbuf_config;
    direct_mtbuf_config.user_sgprs.assign(direct_mtbuf_seed, direct_mtbuf_seed + 8);
    direct_mtbuf_config.local_x = direct_mtbuf_config.local_y =
        direct_mtbuf_config.local_z = 1;
    CHECK(mtbuf_formats_ok &&
              !recompile_compute(direct_mtbuf_copy, std::size(direct_mtbuf_copy),
                                 &direct_mtbuf_table, direct_mtbuf_config).empty(),
          "production compute discovery emits instruction-typed MTBUF source and destination resources");
    const uint32_t invalid_mtbuf_seed[8] = {
        0x00020000u, 0x00040000u, 16u, 0u,
        0x00030000u, 0x00040000u, 16u, 0u,
    };
    std::vector<SrtUse> invalid_mtbuf_uses;
    const std::vector<DynFetch> invalid_mtbuf_fetches = resolve_dynamic_fetch(
        direct_mtbuf_copy, std::size(direct_mtbuf_copy), invalid_mtbuf_seed, 8, 0,
        &invalid_mtbuf_uses);
    ShaderResourceTable invalid_mtbuf_table;
    add_compute_buffer_resources(invalid_mtbuf_table, direct_mtbuf_copy,
                                 std::size(direct_mtbuf_copy), invalid_mtbuf_seed, 8);
    CHECK(invalid_mtbuf_fetches.empty() && invalid_mtbuf_uses.empty() &&
              invalid_mtbuf_table.resources.empty(),
          "MTBUF leaves FORMAT=INVALID descriptors unbound instead of replacing their format");

    // DynFetch shifts graphics vertex descriptors by a constant instruction offset because their
    // special address path drops the original OFFSET/SOFFSET. Compute resources use ConstantBuffer's
    // faithful MUBUF path instead, so the production helper must retain base B and let that path add
    // offset 16 once (not bind B+16 and then index another 16 bytes).
    const uint32_t offset_format_load[] = {
        0xE00C2010u, 0x80000100u,   // buffer_load_format_xyzw ..., s[0:3], offset:16 idxen
        0xBF810000u,
    };
    const std::vector<DynFetch> offset_fetches = resolve_dynamic_fetch(
        offset_format_load, std::size(offset_format_load), direct_copy_seed, 4, 0);
    CHECK(offset_fetches.size() == 1 && offset_fetches[0].desc.base == 0x20010u &&
          offset_fetches[0].unshifted_desc.base == 0x20000u,
          "#636: DynFetch retains both shifted graphics and original compute descriptor bases");
    ShaderResourceTable offset_compute_table;
    add_compute_buffer_resources(offset_compute_table, offset_format_load,
                                 std::size(offset_format_load), direct_copy_seed, 4);
    assign_convention_bindings(offset_compute_table, 2);
    CHECK(offset_compute_table.resources.size() == 1 &&
          offset_compute_table.resources[0].gpu_addr == 0x20000u &&
          offset_compute_table.resources[0].fetch_pc == 0 &&
          !recompile_compute(offset_format_load, std::size(offset_format_load),
                             &offset_compute_table, direct_copy_config).empty(),
          "#636: compute format-load keeps base B and applies nonzero instruction offset once");

    const uint32_t rewritten_copy_dest[] = {
        0xBE840380u,                // s_mov_b32 s4, 0 (seed destination is no longer live)
        0xE01C2000u, 0x80010101u,   // buffer_store_format_xyzw v[1:4], v1, s[4:7], 0 idxen
        0xBF810000u,
    };
    std::vector<SrtUse> rewritten_copy_uses;
    resolve_dynamic_fetch(rewritten_copy_dest,
                          sizeof(rewritten_copy_dest) / sizeof(rewritten_copy_dest[0]),
                          direct_copy_seed, 8, 0, &rewritten_copy_uses);
    CHECK(rewritten_copy_uses.empty(),
          "#636: rewritten direct destination cannot resurrect its entry-time V#");

    // Astro Bot's compact compute dispatcher s_loads output V#s from a table, then consumes them
    // with buffer_store_dword/dwordx2/dwordx3. Stores need the same descriptor provenance as raw
    // loads or the compute resource front-half never creates their writable storage-buffer binding.
    static uint32_t store_output[16];
    static uint32_t store_table[8];
    const uint64_t store_output_base = (uint64_t)(uintptr_t)store_output;
    store_table[4] = (uint32_t)store_output_base;
    store_table[5] = (uint32_t)(store_output_base >> 32) & 0xFFFFu;
    store_table[6] = 16u;
    store_table[7] = 4u << 12;
    const uint64_t store_table_base = (uint64_t)(uintptr_t)store_table;
    const uint32_t k5vs[] = {
        0xF4080504u, 0xFA000010u,   // s_load_dwordx4 s[20:23], s[8:9], 0x10
        0xE0702000u, 0x80051100u,   // buffer_store_dword v17, v0, s[20:23], 0 idxen
        0xBF810000u,
    };
    const uint32_t seed5vs[2] = {
        (uint32_t)store_table_base, (uint32_t)(store_table_base >> 32),
    };
    std::vector<SrtUse> store_uses;
    resolve_dynamic_fetch(k5vs, sizeof(k5vs) / sizeof(k5vs[0]), seed5vs, 2, 8,
                          &store_uses);
    CHECK(store_uses.size() == 1 && store_uses[0].kind == 1 &&
          store_uses[0].key == 0x10 && store_uses[0].use_pc == 2,
          "table-loaded V# resolves raw buffer_store_dword by SRT key");
    CHECK(store_uses.size() == 1 && store_uses[0].v4[0] == store_table[4] &&
          store_uses[0].v4[1] == store_table[5] && store_uses[0].v4[2] == 16u,
          "raw buffer-store V# preserves the table-loaded descriptor dwords");

    // A scalar patch after the table load changes the descriptor that the store actually consumes.
    // The table key is no longer live as a complete four-dword provenance tag, so publish the current
    // V# by exact store pc rather than retaining the pre-patch output range.
    static uint32_t rewritten_store_output[16];
    static uint32_t rewritten_store_table[4];
    const uint64_t rewritten_output_base = (uint64_t)(uintptr_t)rewritten_store_output;
    rewritten_store_table[0] = store_table[4];
    rewritten_store_table[1] = store_table[5] | (4u << 16);
    rewritten_store_table[2] = store_table[6];
    rewritten_store_table[3] = (2u << 12) | 0xFACu;
    const uint64_t rewritten_table_base = (uint64_t)(uintptr_t)rewritten_store_table;
    const uint32_t rewritten_table_store[] = {
        0xF4080104u, 0xFA000000u,                       // s_load_dwordx4 s[4:7], s[8:9], 0
        0xBE8403FFu, (uint32_t)rewritten_output_base,   // s_mov_b32 s4, literal (live base patch)
        0xE01C2000u, 0x80010101u,                       // buffer_store_format_xyzw ..., s[4:7]
        0xBF810000u,
    };
    const uint32_t rewritten_table_seed[2] = {
        (uint32_t)rewritten_table_base, (uint32_t)(rewritten_table_base >> 32),
    };
    uint32_t rewritten_compute_seed[10] = {};
    rewritten_compute_seed[8] = rewritten_table_seed[0];
    rewritten_compute_seed[9] = rewritten_table_seed[1];
    std::vector<SrtUse> rewritten_table_uses;
    resolve_dynamic_fetch(rewritten_table_store, std::size(rewritten_table_store),
                          rewritten_table_seed, 2, 8, &rewritten_table_uses);
    CHECK(rewritten_table_uses.size() == 1 && rewritten_table_uses[0].use_pc == 4 &&
          rewritten_table_uses[0].key == 0xFFFFFFFFu &&
          rewritten_table_uses[0].v4[0] == (uint32_t)rewritten_output_base,
          "#636: table-loaded format store uses its live patched V# by exact pc");

    // The analogous format LOAD must materialize only DynFetch's live pc-keyed descriptor. Before
    // #636 it also produced an SrtUse from the old table snapshot, duplicating the binding and capture
    // range. Exercise the same helper called by realize_compute_dispatches, including binding assignment.
    const uint32_t rewritten_table_load[] = {
        0xF4080104u, 0xFA000000u,                       // s_load_dwordx4 s[4:7], s[8:9], 0
        0xBE8403FFu, (uint32_t)rewritten_output_base,   // s_mov_b32 s4, literal
        0xE00C2000u, 0x80010100u,                       // buffer_load_format_xyzw ..., s[4:7]
        0xBF810000u,
    };
    ShaderResourceTable rewritten_load_table;
    const std::vector<SrtUse> rewritten_load_uses = add_compute_buffer_resources(
        rewritten_load_table, rewritten_table_load, std::size(rewritten_table_load),
        rewritten_compute_seed, 10);
    assign_convention_bindings(rewritten_load_table, 2);
    CHECK(rewritten_load_uses.empty() && rewritten_load_table.resources.size() == 1,
          "#636: table-loaded format load has one production-path resource identity");
    CHECK(rewritten_load_table.resources.size() == 1 &&
          rewritten_load_table.resources[0].gpu_addr == rewritten_output_base &&
          rewritten_load_table.resources[0].fetch_pc == 4 &&
          rewritten_load_table.resources[0].srt_offset == 0xFFFFFFFFu &&
          rewritten_load_table.resources[0].binding == 2,
          "#636: format-load resource uses the live range, pc provenance, and assigned binding");

    // Raw byte-addressed descriptors use stride zero, and the recompiler's supported atomic_umax is
    // an external-buffer consumer too. Both were accepted before the catch-all scan was removed and
    // must remain discoverable through instruction provenance.
    static uint32_t atomic_output[16];
    const uint64_t atomic_output_base = (uint64_t)(uintptr_t)atomic_output;
    const uint32_t atomic_seed[4] = {
        (uint32_t)atomic_output_base,
        (uint32_t)(atomic_output_base >> 32) & 0xFFFFu,
        sizeof(atomic_output), 0u,
    };
    const uint32_t stride_zero_raw[] = {
        0xE0300000u, 0x80000000u, // buffer_load_dword v0, off, s[0:3], 0
        0xBF810000u,
    };
    std::vector<SrtUse> stride_zero_raw_uses;
    resolve_dynamic_fetch(stride_zero_raw, std::size(stride_zero_raw), atomic_seed, 4, 0,
                          &stride_zero_raw_uses);
    ShaderResourceTable stride_zero_raw_table;
    add_compute_buffer_resources(stride_zero_raw_table, stride_zero_raw,
                                 std::size(stride_zero_raw), atomic_seed, 4);
    assign_convention_bindings(stride_zero_raw_table, 2);
    ComputeShaderConfig raw_config;
    raw_config.user_sgprs.assign(atomic_seed, atomic_seed + 4);
    raw_config.local_x = raw_config.local_y = raw_config.local_z = 1;
    CHECK(stride_zero_raw_uses.size() == 1 && stride_zero_raw_table.resources.size() == 1 &&
          stride_zero_raw_table.resources[0].stride == 0 &&
          !recompile_compute(stride_zero_raw, std::size(stride_zero_raw),
                             &stride_zero_raw_table, raw_config).empty(),
          "#636: direct stride-zero raw buffer remains discovered and realizable");
    const uint32_t atomic_umax[] = {
        0xE0E00004u, 0x80000000u, // buffer_atomic_umax v0, v0, s[0:3], offset:4
        0xBF810000u,
    };
    std::vector<SrtUse> atomic_uses;
    resolve_dynamic_fetch(atomic_umax, std::size(atomic_umax), atomic_seed, 4, 0,
                          &atomic_uses);
    CHECK(atomic_uses.size() == 1 && atomic_uses[0].use_pc == 0 &&
          atomic_uses[0].v4[2] == sizeof(atomic_output),
          "#636: direct stride-zero buffer_atomic_umax retains its exact resource use");
    ShaderResourceTable atomic_table;
    add_compute_buffer_resources(atomic_table, atomic_umax, std::size(atomic_umax), atomic_seed, 4);
    assign_convention_bindings(atomic_table, 2);
    ComputeShaderConfig atomic_config;
    atomic_config.user_sgprs.assign(atomic_seed, atomic_seed + 4);
    atomic_config.local_x = atomic_config.local_y = atomic_config.local_z = 1;
    CHECK(atomic_table.resources.size() == 1 && atomic_table.resources[0].stride == 0 &&
          !recompile_compute(atomic_umax, std::size(atomic_umax),
                             &atomic_table, atomic_config).empty(),
          "#636: production resource table keeps stride-zero supported atomic realizable");

    // Kernel 5dr: once any T# SGPR is reloaded, the initial sharp is stale. An unresolved reload must
    // not silently resurrect it and fabricate a texture mapping for the later image operation.
    const uint32_t k5dr[] = {
        0xF4000014u, 0xFA000000u,   // s_load_dword s0, s[40:41], 0x0 (unknown base -> reloaded+unknown)
        0xF0800F08u, 0x00400000u,   // image_sample v[0:3], v[0:1], s[0:7], s[8:11]
        0xBF810000u,                // s_endpgm
    };
    std::vector<SrtUse> direct_reloaded_uses;
    resolve_dynamic_fetch(k5dr, sizeof(k5dr)/sizeof(k5dr[0]), seed5d, 12, 0, &direct_reloaded_uses);
    CHECK(direct_reloaded_uses.empty(),
          "reloaded direct T# SGPR blocks the stale seed fallback");

    // Kernel 5t: AGC user data may carry non-address aperture/tag bits above Base48. A scalar load
    // must canonicalize an otherwise-unreadable tagged pair before the host memory read; the exact
    // same descriptor table should still resolve. (The untagged address is the real in-process table,
    // so the full-range readability check also guards the fallback.)
    uint32_t tagged_seed5[2] = { seed5[0], seed5[1] | 0xABCD0000u };
    std::vector<SrtUse> tagged_uses;
    resolve_dynamic_fetch(k5, sizeof(k5)/sizeof(k5[0]), tagged_seed5, 2, 8, &tagged_uses);
    bool tagged_tex = false, tagged_cbuf = false;
    for (const auto& u : tagged_uses) {
        if (u.kind == 0 && u.key == 0x40 && u.t8[0] == table[16]) tagged_tex = true;
        if (u.kind == 1 && u.key == 0x80 && u.v4[0] == table[32]) tagged_cbuf = true;
    }
    CHECK(tagged_tex && tagged_cbuf,
          "tagged scalar Base48 pointer canonicalizes to the mapped descriptor table");

    // Kernel 5m (#398): same as k5 but the T# s_load uses m0 as SOFFSET (field 124), a special register
    // the fold does not track. It must NOT be folded to offset 0 and snapshotted — that would decode a T#
    // from base+0x40+0 (the wrong address in general) and report it as valid. The fix marks s[12:19]
    // UNKNOWN (soff_ok=false), so the image_sample resolves no Texture use. (SOFFSET field = words[1]>>25:
    // 0xF8000040 -> 124 = m0, vs k5's 0xFA000040 -> 125 = SGPR_NULL.)
    const uint32_t k5m[] = {
        0xF40C0304u, 0xF8000040u,   // s_load_dwordx8 s[12:19], s[8:9], m0  (SOFFSET 124 = m0, untracked)
        0xF4080504u, 0xFA000080u,   // s_load_dwordx4 s[20:23], s[8:9], 0x80 (SSAMP, NULL soffset)
        0xF0800F08u, 0x00A30000u,   // image_sample v[0:3], v[0:1], s[12:19], s[20:23]
        0xBF810000u,                // s_endpgm
    };
    std::vector<SrtUse> uses_m0;
    resolve_dynamic_fetch(k5m, sizeof(k5m)/sizeof(k5m[0]), seed5, 2, 8, &uses_m0);
    bool have_tex_m0 = false;
    for (const auto& u : uses_m0) if (u.kind == 0 && u.key == 0x40) have_tex_m0 = true;
    CHECK(!have_tex_m0, "#398: m0-SOFFSET s_load marks the T# UNKNOWN (no descriptor fabricated from soffset 0)");

    // The instruction cache must reuse an unchanged shader but invalidate when code at the same guest
    // address changes. The second case is important for games that recycle or patch shader allocations:
    // stale decoded literals would resolve a confidently wrong descriptor.
    clear_shader_decode_cache();
    std::vector<uint32_t> mutable_code(k1, k1 + sizeof(k1) / sizeof(k1[0]));
    auto cached_first = resolve_dynamic_fetch(mutable_code.data(), mutable_code.size(), nullptr, 0, 0);
    auto cached_second = resolve_dynamic_fetch(mutable_code.data(), mutable_code.size(), nullptr, 0, 0);
    ShaderDecodeCacheStats cache_stats = shader_decode_cache_stats();
    CHECK(cache_stats.misses == 1 && cache_stats.hits == 1,
          "unchanged shader bytes reuse the decoded-instruction cache");
    mutable_code[1] = 0x00002000u;  // same address, change s_mov_b32 literal: V# base 0x1000 -> 0x2000
    auto patched = resolve_dynamic_fetch(mutable_code.data(), mutable_code.size(), nullptr, 0, 0);
    cache_stats = shader_decode_cache_stats();
    CHECK(cache_stats.invalidations == 1 && cache_stats.misses == 2,
          "same-address shader mutation invalidates the decoded instructions");
    CHECK(cached_first.size() == 1 && cached_second.size() == 1 && patched.size() == 1 &&
              cached_first[0].desc.base == 0x1000u && patched[0].desc.base == 0x2000u,
          "cache invalidation preserves the real fold result after a shader literal changes");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
