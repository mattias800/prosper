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
