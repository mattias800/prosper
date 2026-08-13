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
#include "../src/gpu/rdna2_decode.hpp"
#include "../src/gpu/rdna2_gta5_compute_contracts.hpp"
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "gta5_nullable_output_fixture.hpp"
#include "gta5_zero_record_execz_fixture.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static void set_test_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}

int main() {
    printf("== test_dynfetch_fold ==\n");

    set_test_env("PROSPER_DYNTRACE_FAIL", nullptr);
    set_test_env("PROSPER_DYNTRACE_FAIL_ADDR", nullptr);
    CHECK(!dyntrace_failed_shader_enabled(0x500571000ull),
          "failed-shader dyntrace remains disabled by default");
    set_test_env("PROSPER_DYNTRACE_FAIL", "1");
    CHECK(dyntrace_failed_shader_enabled(0x500571000ull),
          "unfiltered failed-shader dyntrace preserves all-program behavior");
    set_test_env("PROSPER_DYNTRACE_FAIL_ADDR", "0x500571000");
    CHECK(dyntrace_failed_shader_enabled(0x500571000ull) &&
              !dyntrace_failed_shader_enabled(0x500571100ull),
          "failed-shader dyntrace address filter selects one exact program");
    set_test_env("PROSPER_DYNTRACE_FAIL_ADDR", "not-an-address");
    CHECK(!dyntrace_failed_shader_enabled(0x500571000ull),
          "malformed failed-shader dyntrace address filter fails closed");
    set_test_env("PROSPER_DYNTRACE_FAIL", nullptr);
    set_test_env("PROSPER_DYNTRACE_FAIL_ADDR", nullptr);

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
    CHECK(f1.size() == 1 && f1[0].index_mode == VertexFetchIndexMode::Automatic,
          "legacy non-NGG fetches retain the established automatic index heuristic");

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

    // The merged GS/ES ABI supplies s3 outside the user-data block. The live Plucky scene
    // fetch prologue reads it before constructing its V# descriptors, while recompile_vertex
    // already models one active ES vertex as s3=1. Prove the resource fold uses the same ABI:
    // shift s3 into a valid buffer base, then consume the descriptor through s[8:11].
    const uint32_t ngg_s3_fetch[] = {
        0x8f089103u,               // s_lshl_b32 s8, s3, 17 -> 0x20000
        0xe0002000u, 0x80020100u,  // buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xbf810000u,
    };
    const uint32_t ngg_s3_seed[4] = {0u, 0x00100000u, 64u, 0u};
    const auto ngg_s3_resolved = resolve_dynamic_fetch(
        ngg_s3_fetch, std::size(ngg_s3_fetch), ngg_s3_seed, std::size(ngg_s3_seed), 8);
    CHECK(ngg_s3_resolved.size() == 1 && ngg_s3_resolved[0].desc.base == 0x20000ull,
          "NGG dynamic fold seeds merged-wave s3 consistently with vertex recompilation");

    // DQ's NGG fetch prologue selects between the merged-stage ABI inputs with a wave-uniform
    // s_cselect_b64 + v_cndmask_b32_e64 pair. The same shader selects instance_id for a transform
    // lookup and vertex_id for positions. The descriptor fold must publish that distinction; treating
    // both as gl_VertexIndex reads past the 252-record transform table on a 1718-vertex draw.
    const uint32_t ngg_select_fetch[] = {
        0xBF066A80u,                // s_cmp_eq_u32 0, s106
        0x85A0807Eu,                // s_cselect_b64 s[32:33], exec, 0
        0xD5010005u, 0x00820B08u,   // v_cndmask_b32_e64 v5, v8, v5, s[32:33]
        0xE0082000u, 0x6B100E05u,   // buffer_load_format_xyz v[14:16], v5, s[64:67], 0 idxen
        0xBF810000u,
    };
    std::vector<uint32_t> ngg_seed(100);
    ngg_seed[64 - 8] = 0x00020000u;
    ngg_seed[65 - 8] = 12u << 16;
    ngg_seed[66 - 8] = 1718u;
    ngg_seed[67 - 8] = (74u << 12) | 0x3ACu; // Float32 xyz
    ngg_seed[106 - 8] = 0u;                  // SCC=true -> select v5 (vertex_id)
    auto vertex_selected = resolve_dynamic_fetch(
        ngg_select_fetch, std::size(ngg_select_fetch), ngg_seed.data(), ngg_seed.size(), 8);
    CHECK(vertex_selected.size() == 1 &&
              vertex_selected[0].index_mode == VertexFetchIndexMode::Vertex,
          "NGG cselect/cndmask fold identifies a vertex_id buffer fetch");
    ngg_seed[106 - 8] = 1u;                  // SCC=false -> select v8 (instance_id)
    auto instance_selected = resolve_dynamic_fetch(
        ngg_select_fetch, std::size(ngg_select_fetch), ngg_seed.data(), ngg_seed.size(), 8);
    CHECK(instance_selected.size() == 1 &&
              instance_selected[0].index_mode == VertexFetchIndexMode::Instance,
          "NGG cselect/cndmask fold identifies an instance_id buffer fetch");

    // R-Type Delta's AGC fetch prologue saves EXEC once (`s_mov_b64 s[16:17], exec`) and then routes
    // EVERY per-attribute vertex/instance selection through that saved copy rather than naming EXEC
    // again. `s_mov_b64 sDST, exec` is not a scalar-data move, so the fold's fail-closed erase used to
    // drop the wave mask with the value; s_cselect_b64 then folded to Unknown and the attribute was
    // published as shader-computed. The recompiler's faithful-address path then read record 0 for
    // every vertex AND re-added the instruction's OFFSET/SOFFSET to an already-resolved base, so the
    // whole post-movie composite sampled one texel (#2006).
    //
    // The cndmask writes v3, not v5, exactly as the live prologue does: v5 and v8 are pre-seeded
    // Vertex/Instance at entry, so a kernel that selects back into one of them could report the right
    // answer without the selection being modelled at all. Writing an unseeded VGPR makes the check
    // depend on the fold actually resolving the saved mask.
    //
    // The instance arm below does NOT discriminate this fix — `s_cselect_b64`'s false arm is the
    // literal zero mask, which was always resolvable, so it passes with or without the mask save. It
    // is kept as the opposite-polarity control: it fails any "fix" that hands back all-lanes for a
    // saved mask instead of tracking it.
    const uint32_t ngg_saved_exec_fetch[] = {
        0xBE90047Eu,                // s_mov_b64 s[16:17], exec
        0xBF066A80u,                // s_cmp_eq_u32 0, s106
        0x85A08010u,                // s_cselect_b64 s[32:33], s[16:17], 0
        0xD5010003u, 0x00820B08u,   // v_cndmask_b32_e64 v3, v8, v5, s[32:33]
        0xE0082000u, 0x6B100E03u,   // buffer_load_format_xyz v[14:16], v3, s[64:67], 0 idxen
        0xBF810000u,
    };
    ngg_seed[106 - 8] = 0u;                  // SCC=true -> select v5 (vertex_id)
    auto saved_exec_vertex = resolve_dynamic_fetch(
        ngg_saved_exec_fetch, std::size(ngg_saved_exec_fetch), ngg_seed.data(), ngg_seed.size(), 8);
    CHECK(saved_exec_vertex.size() == 1 &&
              saved_exec_vertex[0].index_mode == VertexFetchIndexMode::Vertex,
          "a saved-EXEC copy still identifies a vertex_id fetch through s_cselect_b64");
    ngg_seed[106 - 8] = 1u;                  // SCC=false -> select v8 (instance_id)
    auto saved_exec_instance = resolve_dynamic_fetch(
        ngg_saved_exec_fetch, std::size(ngg_saved_exec_fetch), ngg_seed.data(), ngg_seed.size(), 8);
    CHECK(saved_exec_instance.size() == 1 &&
              saved_exec_instance[0].index_mode == VertexFetchIndexMode::Instance,
          "a saved-EXEC copy still identifies an instance_id fetch through s_cselect_b64");

    // A saved mask copied on through a second `s_mov_b64 sDST, sSRC` keeps its lifetime; the pair's
    // scalar VALUE stays unknown either way, so only the mask domain can carry it.
    const uint32_t ngg_saved_exec_chain_fetch[] = {
        0xBE90047Eu,                // s_mov_b64 s[16:17], exec
        0xBE940410u,                // s_mov_b64 s[20:21], s[16:17]
        0xBF066A80u,                // s_cmp_eq_u32 0, s106
        0x85A08014u,                // s_cselect_b64 s[32:33], s[20:21], 0
        0xD5010003u, 0x00820B08u,   // v_cndmask_b32_e64 v3, v8, v5, s[32:33]
        0xE0082000u, 0x6B100E03u,   // buffer_load_format_xyz v[14:16], v3, s[64:67], 0 idxen
        0xBF810000u,
    };
    ngg_seed[106 - 8] = 0u;
    auto saved_exec_chain = resolve_dynamic_fetch(
        ngg_saved_exec_chain_fetch, std::size(ngg_saved_exec_chain_fetch),
        ngg_seed.data(), ngg_seed.size(), 8);
    CHECK(saved_exec_chain.size() == 1 &&
              saved_exec_chain[0].index_mode == VertexFetchIndexMode::Vertex,
          "a saved wave mask survives a second s_mov_b64 SGPR-pair copy");

    // The buffer load reads VADDR before writing its destination. A prologue may reuse the same VGPR
    // for both, so destination invalidation must not erase the index mode of the current fetch.
    const uint32_t ngg_inplace_fetch[] = {
        0xBF066A80u,                // s_cmp_eq_u32 0, s106
        0x85A0807Eu,                // s_cselect_b64 s[32:33], exec, 0
        0xD5010000u, 0x00820B08u,   // v_cndmask_b32_e64 v0, v8, v5, s[32:33]
        0xE0082000u, 0x6B100000u,   // buffer_load_format_xyz v[0:2], v0, s[64:67], 0 idxen
        0xBF810000u,
    };
    ngg_seed[106 - 8] = 0u;
    auto inplace_selected = resolve_dynamic_fetch(
        ngg_inplace_fetch, std::size(ngg_inplace_fetch), ngg_seed.data(), ngg_seed.size(), 8);
    CHECK(inplace_selected.size() == 1 &&
              inplace_selected[0].index_mode == VertexFetchIndexMode::Vertex,
          "an in-place buffer load retains its read-before-write NGG index provenance");

    // Evergate's Unity fetch prologue uses the compact e32 cndmask form, whose condition is the
    // implicit VCC pair, before later attributes. The fold used to retain only e64 cndmask and
    // therefore downgraded the second fetch to a shader-computed address even though VCC selected
    // the same vertex-id ABI input again.
    const uint32_t ngg_vop2_reselect_fetch[] = {
        0xBE800380u,                // s_mov_b32 s0, 0
        0xBF060000u,                // s_cmp_eq_u32 s0, s0
        0x85EA807Eu,                // s_cselect_b64 vcc, exec, 0
        0x02000B08u,                // v_cndmask_b32_e32 v0, v8, v5
        0xE0082000u, 0x0C020900u,   // buffer_load_format_xyz v[9:11], v0, s[8:11], s12 idxen
        0xBF060000u,
        0x85EA807Eu,
        0x02000B08u,
        0xE0082000u, 0x0C020900u,
        0xBF810000u,
    };
    std::vector<uint32_t> vop2_seed(100);
    vop2_seed[0] = 0x00020000u;
    vop2_seed[1] = 12u << 16;
    vop2_seed[2] = 1718u;
    vop2_seed[3] = (74u << 12) | 0x3ACu;
    auto vop2_reselected = resolve_dynamic_fetch(
        ngg_vop2_reselect_fetch, std::size(ngg_vop2_reselect_fetch),
        vop2_seed.data(), vop2_seed.size(), 8);
    CHECK(vop2_reselected.size() == 2 &&
              vop2_reselected[0].index_mode == VertexFetchIndexMode::Vertex &&
              vop2_reselected[1].index_mode == VertexFetchIndexMode::Vertex,
          "NGG e32 cndmask reselects vertex_id for every Unity attribute fetch");

    // A later packed-attribute address can reuse the ABI VGPR after arithmetic. Its provenance is no
    // longer vertex_id even though an earlier cndmask selected vertex_id into the same register.
    // DQ does this for v5 = 3*vertex_id+1 immediately after its position fetch.
    const uint32_t ngg_computed_fetch[] = {
        0xBF066A80u,                // s_cmp_eq_u32 0, s106
        0x85A0807Eu,                // s_cselect_b64 s[32:33], exec, 0
        0xD5010005u, 0x00820B08u,   // v_cndmask_b32_e64 v5, v8, v5, s[32:33]
        0x4A0A0881u,                // v_add_nc_u32 v5, 1, v4 (computed packed-record address)
        0xE0082000u, 0x6B100E05u,   // buffer_load_format_xyz v[14:16], v5, s[64:67], 0 idxen
        0xBF810000u,
    };
    auto computed_selected = resolve_dynamic_fetch(
        ngg_computed_fetch, std::size(ngg_computed_fetch),
        ngg_seed.data(), ngg_seed.size(), 8);
    CHECK(computed_selected.size() == 1 &&
              computed_selected[0].index_mode == VertexFetchIndexMode::Shader,
          "a computed VADDR kills inherited NGG ABI index provenance");

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

    // A fused NGG back shader receives a driver stage-data pointer in system s[0:1], before user
    // data begins at s8. Its prologue loads the live V# from that table into s[8:11]. Preserve those
    // system seeds so the exact consuming MUBUF can be resolved like any other loaded descriptor.
    static uint32_t ngg_stage_data[4];
    static uint32_t ngg_vertices[16];
    const uint64_t ngg_vertex_base = reinterpret_cast<uint64_t>(ngg_vertices);
    ngg_stage_data[0] = static_cast<uint32_t>(ngg_vertex_base);
    ngg_stage_data[1] = static_cast<uint32_t>(ngg_vertex_base >> 32);
    ngg_stage_data[2] = 16;
    ngg_stage_data[3] = 0;
    const uint32_t ngg_system_sgprs[2] = {
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(ngg_stage_data)),
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(ngg_stage_data) >> 32),
    };
    const uint32_t ngg_loaded_vsharp[] = {
        0xF4080200u, 0xFA000000u,   // s_load_dwordx4 s[8:11], s[0:1], 0
        0xE0002000u, 0x80020100u,   // buffer_load_format_x v1, v0, s[8:11], 0 idxen
        0xBF810000u,
    };
    const auto ngg_fetches = resolve_dynamic_fetch(
        ngg_loaded_vsharp, std::size(ngg_loaded_vsharp), nullptr, 0, 8, nullptr,
        UINT32_MAX, nullptr, ngg_system_sgprs, std::size(ngg_system_sgprs));
    CHECK(ngg_fetches.size() == 1 && ngg_fetches[0].fetch_pc == 2 &&
              ngg_fetches[0].desc.base == ngg_vertex_base,
          "fused NGG system stage-data pointer resolves its loaded vertex descriptor");

    // Astro Bot loads four consecutive V# descriptors from the same stage-data pointer in one
    // s_load_dwordx16, then consumes the first with an untyped MUBUF operation. The wide load is
    // still descriptor provenance, but must resolve by consuming pc because its one immediate
    // offset cannot identify which of the four V#s was selected.
    alignas(16) static uint32_t ngg_wide_stage_data[16]{};
    std::copy(std::begin(ngg_stage_data), std::end(ngg_stage_data),
              std::begin(ngg_wide_stage_data));
    const uint32_t ngg_wide_system_sgprs[2] = {
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(ngg_wide_stage_data)),
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(ngg_wide_stage_data) >> 32),
    };
    const uint32_t ngg_loaded_wide_vsharp[] = {
        0xF4100200u, 0xFA000000u,   // s_load_dwordx16 s[8:23], s[0:1], 0
        0xE0382020u, 0x80020509u,   // buffer_load_dwordx4 v[5:8], v9, s[8:11], s2 offen
        0xBF810000u,
    };
    std::vector<SrtUse> ngg_wide_uses;
    resolve_dynamic_fetch(
        ngg_loaded_wide_vsharp, std::size(ngg_loaded_wide_vsharp), nullptr, 0, 8,
        &ngg_wide_uses, UINT32_MAX, nullptr, ngg_wide_system_sgprs,
        std::size(ngg_wide_system_sgprs));
    CHECK(ngg_wide_uses.size() == 1 && ngg_wide_uses[0].use_pc == 2 &&
              ngg_wide_uses[0].key == 0xFFFFFFFFu &&
              decode_buffer_descriptor(ngg_wide_uses[0].v4.data()).base == ngg_vertex_base,
          "fused NGG x16 stage-data load publishes its raw-buffer V# by consuming pc");

    // Astro Bot's world-map NGG block begins with an 8-dword T# and also carries V#s later in the
    // same x16 load. SGPR memory is typeless: preserving only the four V# interpretations makes the
    // image_sample's s[8:15] permanently unknown and drops the main world-map draw at pc205.
    alignas(16) static uint32_t ngg_mixed_stage_data[16] = {
        0x01234000u, 0x00000005u, 0x00000000u, 0x00100000u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x00000008u,
        0x89abcdefu, 0x01234567u, 0x76543210u, 0xfedcba98u,
        0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u,
    };
    const uint32_t ngg_mixed_system_sgprs[2] = {
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(ngg_mixed_stage_data)),
        static_cast<uint32_t>(reinterpret_cast<uint64_t>(ngg_mixed_stage_data) >> 32),
    };
    const uint32_t ngg_loaded_wide_texture[] = {
        0xF4100200u, 0xFA000000u,   // s_load_dwordx16 s[8:23], s[0:1], 0
        0xF0800F08u, 0x00A20000u,   // image_sample v[0:3], v[0:1], s[8:15], s[20:23]
        0xBF810000u,
    };
    std::vector<SrtUse> ngg_wide_texture_uses;
    resolve_dynamic_fetch(
        ngg_loaded_wide_texture, std::size(ngg_loaded_wide_texture), nullptr, 0, 8,
        &ngg_wide_texture_uses, UINT32_MAX, nullptr, ngg_mixed_system_sgprs,
        std::size(ngg_mixed_system_sgprs));
    CHECK(ngg_wide_texture_uses.size() == 1 &&
              ngg_wide_texture_uses[0].kind == 0 &&
              ngg_wide_texture_uses[0].use_pc == 2 &&
              ngg_wide_texture_uses[0].key == 0xFFFFFFFFu &&
              std::equal(ngg_wide_texture_uses[0].t8.begin(),
                         ngg_wide_texture_uses[0].t8.end(),
                         std::begin(ngg_mixed_stage_data)) &&
              ngg_wide_texture_uses[0].has_samp &&
              ngg_wide_texture_uses[0].s4[0] == ngg_mixed_stage_data[12],
          "fused NGG x16 mixed stage-data load publishes its T# and paired S# by consuming pc");

    // The x16 snapshot is only provenance. A modeled scalar patch changes the physical T# word
    // consumed by MIMG, so the emitted use must carry the live descriptor rather than stale load-time
    // bytes. (An unmodeled write makes that word unknown and suppresses the use instead.)
    const uint32_t ngg_patched_wide_texture[] = {
        0xF4100200u, 0xFA000000u,   // s_load_dwordx16 s[8:23], s[0:1], 0
        0xBE8A0381u,                // s_mov_b32 s10, 1 (patch T#.word2)
        0xF0800F08u, 0x00A20000u,   // image_sample v[0:3], v[0:1], s[8:15], s[20:23]
        0xBF810000u,
    };
    std::vector<SrtUse> ngg_patched_texture_uses;
    resolve_dynamic_fetch(
        ngg_patched_wide_texture, std::size(ngg_patched_wide_texture), nullptr, 0, 8,
        &ngg_patched_texture_uses, UINT32_MAX, nullptr, ngg_mixed_system_sgprs,
        std::size(ngg_mixed_system_sgprs));
    CHECK(ngg_patched_texture_uses.size() == 1 &&
              ngg_patched_texture_uses[0].kind == 0 &&
              ngg_patched_texture_uses[0].use_pc == 3 &&
              ngg_patched_texture_uses[0].key == 0xFFFFFFFFu &&
              ngg_patched_texture_uses[0].t8[0] == ngg_mixed_stage_data[0] &&
              ngg_patched_texture_uses[0].t8[2] == 1u &&
              ngg_patched_texture_uses[0].t8[3] == ngg_mixed_stage_data[3],
          "patched x16 T# publishes its live eight words at the MIMG consumer");

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

    // GTA V also supplies a direct T# as eight exact zero entry SGPRs. MIMG proves the register
    // class, and the sampled-image materializer gives that architectural null state zero-sample
    // semantics. The normal seed plausibility gate must not discard it before materialization.
    uint32_t direct_null_seed[12]{};
    std::copy(seed5d + 8, seed5d + 12, direct_null_seed + 8);
    std::vector<SrtUse> direct_null_uses;
    resolve_dynamic_fetch(k5d, std::size(k5d), direct_null_seed,
                          std::size(direct_null_seed), 0, &direct_null_uses);
    CHECK(direct_null_uses.size() == 1 && direct_null_uses[0].kind == 0 &&
              direct_null_uses[0].key == 0xffffffffu &&
              direct_null_uses[0].use_pc == 0 &&
              std::all_of(direct_null_uses[0].t8.begin(), direct_null_uses[0].t8.end(),
                          [](uint32_t word) { return word == 0; }),
          "exact-null direct user-SGPR T# reaches sampled-image materialization");

    // Same instruction and same base-zero descriptor, but one non-base word is nonzero. This must
    // fail the exact-null exception rather than broadening it to malformed base-zero T# values.
    direct_null_seed[2] = 1;
    std::vector<SrtUse> mutated_direct_null_uses;
    resolve_dynamic_fetch(k5d, std::size(k5d), direct_null_seed,
                          std::size(direct_null_seed), 0, &mutated_direct_null_uses);
    CHECK(mutated_direct_null_uses.empty(),
          "nonzero word at the same direct T# site fails the exact-null exception");

    // Astro Bot's world-map kernel uses IMAGE_BVH_INTERSECT_RAY with a compact four-dword BVH
    // descriptor in s[16:19]. It must not be mistaken for an eight-dword texture descriptor.
    alignas(256) static uint32_t astro_bvh_backing[1024]{};
    const uint64_t astro_bvh_base = reinterpret_cast<uint64_t>(astro_bvh_backing);
    const uint32_t astro_bvh_seed[4] = {
        static_cast<uint32_t>(astro_bvh_base >> 8),
        static_cast<uint32_t>((astro_bvh_base >> 40) & 0xFFu) | (6u << 23),
        63u,                         // (63 + 1) * 64 = 4096 bytes
        0x81000000u,                // TYPE=8, TRIANGLE_RETURN_MODE=1
    };
    const uint32_t astro_bvh_intersect[] = {
        0xf1989f07u, 0x00040303u, 0x43440d3fu, 0x46424140u, 0x00004847u,
        0xbf810000u,
    };
    const DecodedBvhDescriptor decoded_bvh = decode_bvh_descriptor(astro_bvh_seed);
    CHECK(decoded_bvh.base == astro_bvh_base && decoded_bvh.size_bytes == sizeof(astro_bvh_backing) &&
              decoded_bvh.type == 8u && decoded_bvh.triangle_return_mode &&
              decoded_bvh.box_grow == 6u && !decoded_bvh.box_node_64b &&
              !decoded_bvh.sort_enabled,
          "GFX10 BVH descriptor decodes its 256-byte base, 64-byte count and flags");
    uint32_t zero_grow_bvh_seed[4];
    std::copy(std::begin(astro_bvh_seed), std::end(astro_bvh_seed), zero_grow_bvh_seed);
    zero_grow_bvh_seed[1] &= ~(0xFFu << 23);
    CHECK(decode_bvh_descriptor(zero_grow_bvh_seed).box_grow == 0u,
          "GFX10 BVH descriptor preserves a zero box-grow value");
    std::vector<SrtUse> astro_bvh_uses;
    resolve_dynamic_fetch(astro_bvh_intersect, std::size(astro_bvh_intersect),
                          astro_bvh_seed, std::size(astro_bvh_seed), 16, &astro_bvh_uses);
    CHECK(astro_bvh_uses.size() == 1 && astro_bvh_uses[0].kind == 2 &&
              astro_bvh_uses[0].use_pc == 0 && astro_bvh_uses[0].bvh4[0] == astro_bvh_seed[0] &&
              astro_bvh_uses[0].bvh4[3] == astro_bvh_seed[3],
          "Astro IMAGE_BVH_INTERSECT_RAY publishes its live four-word BVH descriptor by pc");
    std::array<uint32_t, 20> astro_compute_seed{};
    std::copy(std::begin(astro_bvh_seed), std::end(astro_bvh_seed),
              astro_compute_seed.begin() + 16);
    ShaderResourceTable astro_bvh_table;
    add_compute_buffer_resources(astro_bvh_table, astro_bvh_intersect,
                                 std::size(astro_bvh_intersect), astro_compute_seed.data(),
                                 astro_compute_seed.size());
    CHECK(astro_bvh_table.resources.size() == 1 &&
              astro_bvh_table.resources[0].cls == ResourceClass::ConstantBuffer &&
              astro_bvh_table.resources[0].gpu_addr == astro_bvh_base &&
              astro_bvh_table.resources[0].size == sizeof(astro_bvh_backing) &&
              astro_bvh_table.resources[0].bvh_box_grow == 6u &&
              astro_bvh_table.resources[0].fetch_pc == 0,
          "Astro BVH bytes materialize as the instruction-scoped read-only compute buffer");

    // Concrete descriptor words alone are diagnostic evidence, not binding provenance. Keep this
    // case unresolved while allowing dyntrace to print the exact live words that led to rejection.
    const uint32_t unproven_live_bvh[] = {
        0xBE9003FFu, 0x00123456u,            // pc0: s_mov_b32 s16, literal
        0xBE9103FFu, 0x03000000u,            // pc2: s_mov_b32 s17, literal
        0xBE9203FFu, 0x0000003Fu,            // pc4: s_mov_b32 s18, literal
        0xBE9303FFu, 0x81000000u,            // pc6: s_mov_b32 s19, literal
        0xF1989F07u, 0x00040303u, 0x43440D3Fu, 0x46424140u, 0x00004847u,
        0xBF810000u,
    };
    std::vector<SrtUse> unproven_live_bvh_uses;
    resolve_dynamic_fetch(unproven_live_bvh, std::size(unproven_live_bvh), nullptr, 0, 0,
                          &unproven_live_bvh_uses);
    CHECK(unproven_live_bvh_uses.empty(),
          "live-known BVH words without descriptor provenance remain fail-visible");

    // Astro's first world-map dispatch explicitly writes a null qword to the root pointer consumed
    // by this scalar chain. Preserve that fact only through the exact dependent descriptor ALU, and
    // only when an EXEC-changing instruction plus forward EXECZ branch dominates the ray use. This
    // is deliberately stricter than the old generic unresolved-BVH fallback rejected in review.
    alignas(8) uint64_t null_bvh_root = 0;
    const uint64_t null_bvh_root_addr = reinterpret_cast<uint64_t>(&null_bvh_root);
    std::array<uint32_t, 8> null_bvh_seed{};
    null_bvh_seed[0] = static_cast<uint32_t>(null_bvh_root_addr);
    null_bvh_seed[1] = static_cast<uint32_t>(null_bvh_root_addr >> 32);
    std::array<uint32_t, 24> guarded_null_bvh = {
        0xBEAB446Au,                         // pc0:  Astro s_andn1_saveexec_b32 (narrows EXEC)
        0xBF880014u,                         // pc1:  s_cbranch_execz pc22
        0xF4040080u, 0xFA000000u,            // pc2:  s_load_dwordx2 s[2:3], s[0:1], 0 -> null
        0xBF8CC07Fu,                         // pc4:  s_waitcnt
        0xF4040181u, 0xFA000058u,            // pc5:  s_load_dwordx2 s[6:7], s[2:3], 0x58
        0xBF8CC07Fu,                         // pc7:  s_waitcnt
        0x8006C106u,                         // pc8:  s_add_u32 s6,s6,-1
        0x8207C107u,                         // pc9:  s_addc_u32 s7,s7,-1
        0x876A07FFu, 0x000003FFu,            // pc10: s_and_b32 s106, 0x3ff, s7
        0x9484FF02u, 0x00280008u,            // pc12: s_bfe_u64 s[4:5], s[2:3], 8:40
        0x8807FF6Au, 0x81000000u,            // pc14: s_or_b32 s7, s106, TYPE/mode
        0xBE851D9Fu,                         // pc16: s_bitset1_b32 s5,31 (SORT field)
        0xF1989F07u, 0x00010303u, 0x094F4E3Fu, 0x11100F0Eu, 0x00001312u,
        0xBF800000u,                         // pc22: merge
        0xBF810000u,                         // pc23: s_endpgm
    };
    std::vector<SrtUse> guarded_null_uses;
    resolve_dynamic_fetch(guarded_null_bvh.data(), guarded_null_bvh.size(),
                          null_bvh_seed.data(), null_bvh_seed.size(), 0,
                          &guarded_null_uses);
    CHECK(guarded_null_uses.size() == 1 && guarded_null_uses[0].kind == 3 &&
              guarded_null_uses[0].use_pc == 17,
          "mapped null BVH root plus dominating EXEC guard publishes an exact-pc null use");
    ShaderResourceTable guarded_null_table;
    add_compute_buffer_resources(guarded_null_table, guarded_null_bvh.data(),
                                 guarded_null_bvh.size(), null_bvh_seed.data(),
                                 null_bvh_seed.size());
    CHECK(guarded_null_table.resources.size() == 1 &&
              is_proven_null_bvh(guarded_null_table.resources[0]) &&
              guarded_null_table.resources[0].fetch_pc == 17,
          "proven null BVH materializes as a bounded capture-stable marker");

    guarded_null_bvh[1] = 0xBF800000u;       // remove the EXECZ region proof
    std::vector<SrtUse> unguarded_null_uses;
    resolve_dynamic_fetch(guarded_null_bvh.data(), guarded_null_bvh.size(),
                          null_bvh_seed.data(), null_bvh_seed.size(), 0,
                          &unguarded_null_uses);
    CHECK(unguarded_null_uses.empty(),
          "mapped null BVH root without a dominating EXEC guard remains fail-visible");

    // A straight-line compiler sequence can compute a branch condition exclusively from shader
    // constants: 37 & 7 = 5; 5 + (-4) = 1; 1 <= 2. Resource discovery and translation must consume
    // the same pruned instruction stream. The proof must not broaden to an entry/user-SGPR comparison.
    const uint32_t shader_constant_dead_fetch[] = {
        0xBE8B03A5u,             // pc0: s_mov_b32 s11, 37
        0x876A870Bu,             // pc1: s_and_b32 s106, s11, 7
        0x816AC46Au,             // pc2: s_add_i32 s106, s106, -4
        0xBF0B826Au,             // pc3: s_cmp_le_u32 s106, 2
        0xBF850002u,             // pc4: s_cbranch_scc1 pc7
        0xE0002000u, 0x80030100u,// pc5: dead buffer_load_format_x ..., s[12:15]
        0xBF810000u,             // pc7: s_endpgm
    };
    std::array<uint32_t, 16> constant_branch_seed{};
    constant_branch_seed[12] = 0x10000u;
    constant_branch_seed[14] = 64u;
    std::vector<Rdna2Inst> pruned_constant_branch;
    rdna2_walk(shader_constant_dead_fetch, std::size(shader_constant_dead_fetch),
               pruned_constant_branch);
    CHECK(rdna2_specialize_shader_constant_branches(pruned_constant_branch) == 1 &&
              std::none_of(pruned_constant_branch.begin(), pruned_constant_branch.end(),
                           [](const Rdna2Inst& in) { return in.pc == 5; }),
          "shader-constant SCC branch prunes its dead resource block");
    CHECK(resolve_dynamic_fetch(shader_constant_dead_fetch,
                                std::size(shader_constant_dead_fetch),
                                constant_branch_seed.data(), constant_branch_seed.size(), 0).empty(),
          "dynamic resource discovery ignores the provably dead buffer fetch");

    // Scalar ALU opcodes the constant folder did not model. Every word below was produced by
    // `llvm-mc -arch=amdgcn -mcpu=gfx1030 -show-encoding`, not by reading this repo's own tables --
    // prosper's decoder is upstream of those tables, so it cannot check them, and #2481 records a
    // mnemonic error that survived three internally consistent internal anchors and inverted a
    // frontier conclusion.
    //
    // These arms discriminate: without the corresponding fold entry, `shader_constant_operand`
    // reports the shifted value unknown, `shader_constant_compare` declines, and the specializer
    // returns 0 with the dead block retained. Verified by removing case 0x1e and re-running --
    // the first arm reports 0 rather than 1.
    const uint32_t shifted_dead_fetch[] = {
        0xBE800381u,             // pc0: s_mov_b32 s0, 1
        0x8F018300u,             // pc1: s_lshl_b32 s1, s0, 3   -> 8
        0xBF078801u,             // pc2: s_cmp_lg_u32 s1, 8     -> SCC = (8 != 8) = false
        0xBF840002u,             // pc3: s_cbranch_scc0 +2      -> taken, skips pc4..5
        0xE0002000u, 0x80030100u,// pc4: dead buffer_load_format_x
        0xBF810000u,             // pc6: s_endpgm
    };
    std::vector<Rdna2Inst> shifted_branch;
    rdna2_walk(shifted_dead_fetch, std::size(shifted_dead_fetch), shifted_branch);
    CHECK(rdna2_specialize_shader_constant_branches(shifted_branch) == 1 &&
              std::none_of(shifted_branch.begin(), shifted_branch.end(),
                           [](const Rdna2Inst& in) { return in.pc == 4; }),
          "s_lshl_b32 folds, so its SCC branch prunes the dead resource block");

    // The two right shifts, each pinned against the OTHER one. The shifted value has bit 31 set,
    // which is the whole point: an operand with bit 31 clear cannot separate them, because
    // `8 >> 3` is 1 under both a logical and an arithmetic shift. A first version of this arm used
    // 8 and would have passed with 0x20 and 0x22 swapped -- and since both entries arrived in one
    // commit, nothing else pinned them apart. Same failure as comparing against 8 instead of 1 at
    // the left-vs-right level, one level deeper.
    //
    // 0x80000000 >> 28 is 0x8 logically and 0xFFFFFFF8 arithmetically, so each arm's compare
    // constant is the OTHER opcode's wrong answer's complement: the logical arm compares against 8
    // and the arithmetic arm against -8. Mutation-verified in both directions.
    const uint32_t logical_shift_fetch[] = {
        0xBE8003FFu, 0x80000000u,// pc0: s_mov_b32 s0, 0x80000000
        0x90019C00u,             // pc2: s_lshr_b32 s1, s0, 28  -> 0x00000008
        0xBF078801u,             // pc3: s_cmp_lg_u32 s1, 8     -> SCC = (8 != 8) = false
        0xBF840002u,             // pc4: s_cbranch_scc0 +2      -> taken, skips pc5..6
        0xE0002000u, 0x80030100u,// pc5: dead buffer_load_format_x
        0xBF810000u,             // pc7: s_endpgm
    };
    std::vector<Rdna2Inst> logical_shift;
    rdna2_walk(logical_shift_fetch, std::size(logical_shift_fetch), logical_shift);
    CHECK(rdna2_specialize_shader_constant_branches(logical_shift) == 1 &&
              std::none_of(logical_shift.begin(), logical_shift.end(),
                           [](const Rdna2Inst& in) { return in.pc == 5; }),
          "s_lshr_b32 folds LOGICALLY -- 0x80000000 >> 28 is 8, not -8");

    const uint32_t arithmetic_shift_fetch[] = {
        0xBE8003FFu, 0x80000000u,// pc0: s_mov_b32 s0, 0x80000000
        0x91019C00u,             // pc2: s_ashr_i32 s1, s0, 28  -> 0xFFFFFFF8
        0xBF07C801u,             // pc3: s_cmp_lg_u32 s1, -8    -> SCC = (-8 != -8) = false
        0xBF840002u,             // pc4: s_cbranch_scc0 +2      -> taken, skips pc5..6
        0xE0002000u, 0x80030100u,// pc5: dead buffer_load_format_x
        0xBF810000u,             // pc7: s_endpgm
    };
    std::vector<Rdna2Inst> arithmetic_shift;
    rdna2_walk(arithmetic_shift_fetch, std::size(arithmetic_shift_fetch), arithmetic_shift);
    CHECK(rdna2_specialize_shader_constant_branches(arithmetic_shift) == 1 &&
              std::none_of(arithmetic_shift.begin(), arithmetic_shift.end(),
                           [](const Rdna2Inst& in) { return in.pc == 5; }),
          "s_ashr_i32 folds ARITHMETICALLY -- 0x80000000 >> 28 is -8, not 8");

    // Exact instruction words and PCs from the live traversal loop. The loop-selected BVH sites at
    // PC968/1007 can write work into v59/v60 and return through PC1671/1674, so shader-constant
    // folding alone must retain them. The first entry, however, initializes s45=0 and s11=37, visits
    // the dispatch-proven null root at PC1346, and reaches the PC1665 empty-stack exit without a
    // write to s45. That closed cycle has no first work item and can be pruned only as one proof.
    std::vector<Rdna2Inst> loop_variant_branch;
    auto append_exact_instruction = [&](uint32_t pc, std::initializer_list<uint32_t> words) {
        Rdna2Inst instruction = rdna2_decode_one(words.begin(), words.size());
        instruction.pc = pc;
        loop_variant_branch.push_back(instruction);
    };
    append_exact_instruction(847,  {0xBEAD0380u}); // s_mov_b32 s45, 0 (empty stack)
    append_exact_instruction(848,  {0xBE8B03A5u}); // s_mov_b32 s11, 37 (root selector)
    append_exact_instruction(854,  {0xBE88037Eu}); // loop header
    append_exact_instruction(855,  {0x876A870Bu}); // loop: s_and_b32 s106, s11, 7
    append_exact_instruction(856,  {0x876B087Eu});
    append_exact_instruction(857,  {0x816AC46Au}); // s_add_i32 s106, s106, -4
    append_exact_instruction(858,  {0xBF0B826Au}); // s_cmp_le_u32 s106, 2
    append_exact_instruction(859,  {0xBF8501DEu}); // s_cbranch_scc1 1338
    append_exact_instruction(968,  {0xF1989F07u, 0x00040303u, 0x43440D3Fu,
                                    0x46424140u, 0x00004847u}); // unresolved loop BVH
    append_exact_instruction(1007, {0xF1989F07u, 0x00040303u, 0x43440D3Fu,
                                    0x46424140u, 0x00004847u}); // unresolved loop BVH
    append_exact_instruction(1338, {0x7E0602C1u}); // v3..v6 = invalid
    append_exact_instruction(1339, {0x7E0802C1u});
    append_exact_instruction(1340, {0x7E0A02C1u});
    append_exact_instruction(1341, {0x7E0C02C1u});
    append_exact_instruction(1342, {0xBEEA3C6Bu});
    append_exact_instruction(1343, {0xBF88000Au}); // empty EXEC skips to the no-hit compare
    append_exact_instruction(1344, {0x7E06020Bu}); // root index from s11
    append_exact_instruction(1345, {0xBF800000u});
    append_exact_instruction(1346, {0xF1989F07u, 0x00010303u, 0x094F4E3Fu,
                                    0x11100F0Eu, 0x00001312u}); // guarded null BVH
    append_exact_instruction(1351, {0xBF8C3F70u});
    append_exact_instruction(1352, {0x7D8A06F9u, 0x068688C1u}); // v_cmp_ne_u32 s8,-1,v3
    append_exact_instruction(1354, {0xBF070880u});              // s_cmp_lg_u32 0,s8
    append_exact_instruction(1355, {0xBEFE036Au});              // s_mov_b32 exec_lo,vcc_lo
    append_exact_instruction(1356, {0xBF840133u});              // s_cbranch_scc0 1664
    append_exact_instruction(1610, {0xBE8B03C1u});              // s_mov_b32 s11, -1
    append_exact_instruction(1663, {0xBF82FCD7u});              // s_branch 855 (back-edge)
    append_exact_instruction(1664, {0xBF072D80u});              // s_cmp_lg_u32 0,s45
    append_exact_instruction(1665, {0xBF840009u});              // empty stack -> 1675
    append_exact_instruction(1666, {0x812DC12Du});              // pop stack
    append_exact_instruction(1667, {0xBF09A02Du});
    append_exact_instruction(1668, {0xBF840003u});
    append_exact_instruction(1669, {0xD760000Bu, 0x00005B3Cu}); // s11 = readlane(v60,s45)
    append_exact_instruction(1671, {0xBF82FCCEu});              // s_branch 854
    append_exact_instruction(1672, {0xD760000Bu, 0x00005B3Bu}); // s11 = readlane(v59,s45)
    append_exact_instruction(1674, {0xBF82FCCBu});              // s_branch 854
    append_exact_instruction(1675, {0x06067EFFu, 0x3A83126Fu}); // exact empty-stack target
    append_exact_instruction(3276, {0xBF810000u});              // s_endpgm
    CHECK(rdna2_specialize_shader_constant_branches(loop_variant_branch) == 0 &&
              std::any_of(loop_variant_branch.begin(), loop_variant_branch.end(),
                          [](const Rdna2Inst& in) {
                              return in.pc == 859 && in.opcode == 0x05;
                          }),
          "loop-carried scalar mutation prevents one-shot SCC branch specialization");

    alignas(256) std::array<uint8_t, 256> loop_null_bvh{};
    ShaderResourceTable loop_null_table;
    { ShaderResource marker{};
      marker.cls = ResourceClass::ConstantBuffer;
      marker.format = DataFormat::Uint32;
      marker.num_components = 1;
      marker.size = loop_null_bvh.size();
      marker.fetch_pc = 1346;
      marker.host_data = loop_null_bvh.data();
      marker.host_data_size = loop_null_bvh.size();
      loop_null_table.resources.push_back(marker); }
    for (uint32_t fetch_pc : {968u, 1007u}) {
        ShaderResource dead{};
        dead.cls = ResourceClass::ConstantBuffer;
        dead.format = DataFormat::Uint32;
        dead.num_components = 1;
        dead.size = 4;
        dead.fetch_pc = fetch_pc;
        loop_null_table.resources.push_back(dead);
    }
    std::vector<Rdna2Inst> null_pruned_loop = loop_variant_branch;
    const ComputeResourcePathSpecializationReport null_path_report =
        specialize_compute_resource_paths(null_pruned_loop, loop_null_table, 32);
    CHECK(null_path_report.proven_null_exits == 1 &&
              null_path_report.shader_constant_branches == 0 &&
              null_path_report.removed_resources == 2 &&
              std::find(null_path_report.removed_pcs.begin(),
                        null_path_report.removed_pcs.end(), 968) !=
                  null_path_report.removed_pcs.end() &&
              std::find(null_path_report.removed_pcs.begin(),
                        null_path_report.removed_pcs.end(), 1007) !=
                  null_path_report.removed_pcs.end() &&
              std::none_of(null_pruned_loop.begin(), null_pruned_loop.end(),
                           [](const Rdna2Inst& in) {
                               return in.pc == 968 || in.pc == 1007 || in.pc == 1610 ||
                                      in.pc == 1663 || in.pc == 1671 || in.pc == 1674;
                           }) &&
              loop_null_table.resources.size() == 1 &&
              is_proven_null_bvh(loop_null_table.resources.front()) &&
              loop_null_table.resources.front().fetch_pc == 1346,
          "executor resource-path specialization reports the exact removed traversal PCs, "
          "prunes their resources, and retains the null proof for raw translation");

    // The same compiler template may allocate its scalar traversal-stack depth to a different
    // ordinary SGPR. Astro's larger sibling kernel uses s41 rather than s45; the proof must follow
    // the register linked by the zero initializer and empty-stack comparison, not a title-specific
    // physical number.
    std::vector<Rdna2Inst> s41_loop = loop_variant_branch;
    auto s41_at = [&](uint32_t pc) {
        return std::find_if(s41_loop.begin(), s41_loop.end(),
                            [&](const Rdna2Inst& in) { return in.pc == pc; });
    };
    const std::array<uint32_t, 1> s41_init_words{0xBEA90380u};
    Rdna2Inst s41_init = rdna2_decode_one(s41_init_words.data(), s41_init_words.size());
    s41_init.pc = 847;
    *s41_at(847) = s41_init;
    const std::array<uint32_t, 1> s41_compare_words{0xBF072980u};
    Rdna2Inst s41_compare =
        rdna2_decode_one(s41_compare_words.data(), s41_compare_words.size());
    s41_compare.pc = 1664;
    *s41_at(1664) = s41_compare;
    ShaderResourceTable s41_loop_table = loop_null_table;
    for (uint32_t fetch_pc : {968u, 1007u}) {
        ShaderResource dead{};
        dead.cls = ResourceClass::ConstantBuffer;
        dead.format = DataFormat::Uint32;
        dead.num_components = 1;
        dead.size = 4;
        dead.fetch_pc = fetch_pc;
        s41_loop_table.resources.push_back(dead);
    }
    const ComputeResourcePathSpecializationReport s41_path_report =
        specialize_compute_resource_paths(s41_loop, s41_loop_table, 32);
    CHECK(s41_path_report.proven_null_exits == 1 &&
              s41_path_report.removed_resources == 2 &&
              std::none_of(s41_loop.begin(), s41_loop.end(),
                           [](const Rdna2Inst& in) {
                               return in.pc == 968 || in.pc == 1007;
                           }),
          "empty-stack BVH cycle proof derives the compiler-allocated stack SGPR");

    std::vector<Rdna2Inst> mismatched_stack_loop = loop_variant_branch;
    auto mismatched_compare = std::find_if(
        mismatched_stack_loop.begin(), mismatched_stack_loop.end(),
        [](const Rdna2Inst& in) { return in.pc == 1664; });
    mismatched_compare->src[1].value = 41;
    CHECK(rdna2_specialize_proven_null_bvh_paths(
              mismatched_stack_loop, &loop_null_table, 32) == 1 &&
              std::any_of(mismatched_stack_loop.begin(), mismatched_stack_loop.end(),
                          [](const Rdna2Inst& in) {
                              return in.pc == 968 || in.pc == 1007;
                          }),
          "empty-stack BVH cycle rejects a counter without a matching zero initializer");

    ShaderResourceTable nonnull_loop_table = loop_null_table;
    nonnull_loop_table.resources[0].gpu_addr = 0x10000u;
    std::vector<Rdna2Inst> nonnull_loop = loop_variant_branch;
    CHECK(rdna2_specialize_proven_null_bvh_paths(
              nonnull_loop, &nonnull_loop_table, 32) == 0 &&
              rdna2_specialize_shader_constant_branches(nonnull_loop) == 0 &&
              std::any_of(nonnull_loop.begin(), nonnull_loop.end(),
                          [](const Rdna2Inst& in) { return in.pc == 968 || in.pc == 1007; }),
          "non-null resource preserves loop-reachable BVH sites");

    for (int deviation = 0; deviation < 4; ++deviation) {
        std::vector<Rdna2Inst> changed = loop_variant_branch;
        auto wait = std::find_if(changed.begin(), changed.end(),
                                 [](const Rdna2Inst& in) { return in.pc == 1351; });
        auto compare = std::find_if(changed.begin(), changed.end(),
                                    [](const Rdna2Inst& in) { return in.pc == 1352; });
        auto exit = std::find_if(changed.begin(), changed.end(),
                                 [](const Rdna2Inst& in) { return in.pc == 1356; });
        if (deviation == 0) compare->opcode = 0xc4u;
        if (deviation == 1) compare->src[1].value = 4;
        if (deviation == 2) exit->opcode = 0x05u;
        if (deviation == 3) wait->words[0] = 0xbf8c3f71u;
        CHECK(rdna2_specialize_proven_null_bvh_paths(
                  changed, &loop_null_table, 32) == 0,
              "null-BVH exit proof rejects opcode/register/guard deviations");
    }

    for (int deviation = 0; deviation < 5; ++deviation) {
        std::vector<Rdna2Inst> changed = loop_variant_branch;
        auto at = [&](uint32_t pc) {
            return std::find_if(changed.begin(), changed.end(),
                                [&](const Rdna2Inst& in) { return in.pc == pc; });
        };
        if (deviation == 0) at(847)->src[0].value = 1;
        if (deviation == 1) at(1339)->dst.value = 5;
        if (deviation == 2) at(1342)->opcode = 0x03u;
        if (deviation == 3) at(1664)->src[1].value = 44;
        if (deviation == 4) at(1665)->opcode = 0x05u;
        CHECK(rdna2_specialize_proven_null_bvh_paths(
                  changed, &loop_null_table, 32) == 1 &&
                  std::any_of(changed.begin(), changed.end(),
                              [](const Rdna2Inst& in) {
                                  return in.pc == 968 || in.pc == 1007;
                              }),
              "empty-stack cycle proof rejects initializer/prefix/tail deviations");
    }

    std::vector<Rdna2Inst> externally_entered_loop = loop_variant_branch;
    const std::array<uint32_t, 1> external_entry_words{0xBF820007u}; // pc846 -> pc854
    Rdna2Inst external_entry = rdna2_decode_one(external_entry_words.data(),
                                                external_entry_words.size());
    external_entry.pc = 846;
    externally_entered_loop.insert(externally_entered_loop.begin(), external_entry);
    CHECK(rdna2_specialize_proven_null_bvh_paths(
              externally_entered_loop, &loop_null_table, 32) == 1 &&
              std::any_of(externally_entered_loop.begin(), externally_entered_loop.end(),
                          [](const Rdna2Inst& in) { return in.pc == 968 || in.pc == 1007; }),
          "external entry that bypasses the zero initializer keeps the cycle fail-visible");

    // Exact reduced Wave64 sibling from Astro's larger traversal shader. PC2148/2187 are reachable
    // only through loop back-edges, while the first selector (s20=37) enters the guarded root at
    // PC2434. A null result creates a complete zero mask pair, takes the u64 SCC0 exit, then observes
    // the s46=0 stack. This proves that no first work item can reach either unresolved ray site.
    std::vector<Rdna2Inst> wave64_loop;
    auto append_wave64_instruction = [&](uint32_t pc,
                                         std::initializer_list<uint32_t> words) {
        Rdna2Inst instruction = rdna2_decode_one(words.begin(), words.size());
        instruction.pc = pc;
        wave64_loop.push_back(instruction);
    };
    append_wave64_instruction(2029, {0xBEAE0380u}); // s_mov_b32 s46, 0 (empty stack)
    append_wave64_instruction(2032, {0x802F080Au}); // independent address setup
    append_wave64_instruction(2033, {0xBE9403A5u}); // s_mov_b32 s20, 37 (root selector)
    append_wave64_instruction(2039, {0xBE88047Eu}); // loop header: s_mov_b64 s[8:9], exec
    append_wave64_instruction(2040, {0x876A8714u});
    append_wave64_instruction(2041, {0x816AC46Au});
    append_wave64_instruction(2042, {0xBF0B826Au});
    append_wave64_instruction(2043, {0xBF85017Eu}); // s_cbranch_scc1 2426
    append_wave64_instruction(2148, {0xF1989F07u, 0x00040505u, 0x4442413Du,
                                     0x4543403Eu, 0x00004746u}); // unresolved loop BVH
    append_wave64_instruction(2187, {0xF1989F07u, 0x00040505u, 0x4442413Du,
                                     0x4543403Eu, 0x00004746u}); // unresolved loop BVH
    append_wave64_instruction(2426, {0x7E0A02C1u}); // v5..v8 = invalid
    append_wave64_instruction(2427, {0x7E0C02C1u});
    append_wave64_instruction(2428, {0x7E0E02C1u});
    append_wave64_instruction(2429, {0x7E1002C1u});
    append_wave64_instruction(2430, {0xBEEA2408u}); // s_and_saveexec_b64 vcc, s[8:9]
    append_wave64_instruction(2431, {0xBF88000Au}); // empty EXEC -> no-hit compare
    append_wave64_instruction(2432, {0x7E0A0214u}); // root index from s20
    append_wave64_instruction(2433, {0xBF800000u});
    append_wave64_instruction(2434, {0xF1989F07u, 0x00010505u, 0x39370A3Du,
                                     0x33323130u, 0x00003534u}); // guarded null BVH
    append_wave64_instruction(2439, {0xBF8C3F70u});
    append_wave64_instruction(2440, {0x7D8A0AF9u, 0x068688C1u}); // v_cmp_ne_u32 s[8:9],-1,v5
    append_wave64_instruction(2442, {0xBF138008u});              // s_cmp_lg_u64 s[8:9],0
    append_wave64_instruction(2443, {0xBEFE046Au});              // s_mov_b64 exec,vcc
    append_wave64_instruction(2444, {0xBF8400D8u});              // s_cbranch_scc0 2661
    append_wave64_instruction(2624, {0xBE9403C1u});              // loop-carried s20 = -1
    append_wave64_instruction(2660, {0xBF82FD93u});              // s_branch 2040
    append_wave64_instruction(2661, {0xBF072E80u});              // s_cmp_lg_u32 0,s46
    append_wave64_instruction(2662, {0xBF840004u});              // empty stack -> 2667
    append_wave64_instruction(2663, {0x812EC12Eu});              // pop stack
    append_wave64_instruction(2664, {0xD7600014u, 0x00005D3Bu});
    append_wave64_instruction(2666, {0xBF82FD8Cu});              // s_branch 2039
    append_wave64_instruction(2667, {0x060A7AFFu, 0x3A83126Fu});
    append_wave64_instruction(3271, {0xBF810000u});

    alignas(256) std::array<uint8_t, 256> wave64_null_bvh{};
    auto wave64_resources = [&]() {
        ShaderResourceTable table;
        ShaderResource marker{};
        marker.cls = ResourceClass::ConstantBuffer;
        marker.format = DataFormat::Uint32;
        marker.num_components = 1;
        marker.binding = 4;
        marker.size = wave64_null_bvh.size();
        marker.fetch_pc = 2434;
        marker.host_data = wave64_null_bvh.data();
        marker.host_data_size = wave64_null_bvh.size();
        table.resources.push_back(marker);
        for (uint32_t fetch_pc : {2148u, 2187u}) {
            ShaderResource unresolved{};
            unresolved.cls = ResourceClass::ConstantBuffer;
            unresolved.format = DataFormat::Uint32;
            unresolved.num_components = 1;
            unresolved.binding = static_cast<uint32_t>(table.resources.size() + 4);
            unresolved.size = 4;
            unresolved.fetch_pc = fetch_pc;
            table.resources.push_back(unresolved);
        }
        return table;
    };
    std::vector<Rdna2Inst> wave64_constant_only = wave64_loop;
    CHECK(rdna2_specialize_shader_constant_branches(wave64_constant_only) == 0 &&
              std::any_of(wave64_constant_only.begin(), wave64_constant_only.end(),
                          [](const Rdna2Inst& in) {
                              return in.pc == 2148 || in.pc == 2187;
                          }),
          "Wave64 loop-carried selector keeps unresolved BVHs before the null-cycle proof");

    ShaderResourceTable wave64_pruned_resources = wave64_resources();
    std::vector<Rdna2Inst> wave64_pruned_loop = wave64_loop;
    const ComputeResourcePathSpecializationReport wave64_path_report =
        specialize_compute_resource_paths(wave64_pruned_loop, wave64_pruned_resources, 64);
    CHECK(wave64_path_report.proven_null_exits == 1 &&
              wave64_path_report.shader_constant_branches == 0 &&
              wave64_path_report.removed_resources == 2 &&
              std::find(wave64_path_report.removed_pcs.begin(),
                        wave64_path_report.removed_pcs.end(), 2148) !=
                  wave64_path_report.removed_pcs.end() &&
              std::find(wave64_path_report.removed_pcs.begin(),
                        wave64_path_report.removed_pcs.end(), 2187) !=
                  wave64_path_report.removed_pcs.end() &&
              std::none_of(wave64_pruned_loop.begin(), wave64_pruned_loop.end(),
                           [](const Rdna2Inst& in) {
                               return in.pc == 2148 || in.pc == 2187 || in.pc == 2624 ||
                                      in.pc == 2660 || in.pc == 2663 || in.pc == 2666;
                           }) &&
              wave64_pruned_resources.resources.size() == 1 &&
              is_proven_null_bvh(wave64_pruned_resources.resources.front()) &&
              wave64_pruned_resources.resources.front().fetch_pc == 2434,
          "exact Wave64 null cycle prunes loop BVHs, back-edges, and their resources");

    // Materialize the same exact-PC reduced stream with NOPs in omitted spans so the production raw
    // translator repeats the proof. Retain an independent runtime loop at entry so this follows the
    // target shader's arbitrary-CFG compute path (which already supports its exact Wave64 mask tail),
    // rather than accidentally exercising the ordinary structured emitter after reducing the target.
    // Every exit from this loop still reaches the stack initializer, so it cannot bypass the proof.
    // The synthetic terminal at PC2667 is the proven empty-stack target.
    std::vector<uint32_t> wave64_null_exit_shader(2668, 0xBF800000u);
    wave64_null_exit_shader[0] = 0xBF850002u; // s_cbranch_scc1 pc3
    wave64_null_exit_shader[1] = 0xBF840001u; // s_cbranch_scc0 pc3
    wave64_null_exit_shader[2] = 0xBF82FFFDu; // s_branch pc0 (multi-exit back-edge)
    for (const Rdna2Inst& instruction : wave64_loop) {
        if (instruction.pc >= 2667) continue;
        for (uint32_t word = 0; word < instruction.len_dwords; ++word)
            wave64_null_exit_shader[instruction.pc + word] = instruction.words[word];
    }
    wave64_null_exit_shader[2667] = 0xBF810000u;
    ShaderResourceTable wave64_translation_resources = wave64_resources();
    ComputeShaderConfig wave64_null_exit_config;
    wave64_null_exit_config.local_x = 1;
    wave64_null_exit_config.wave_size = 64;
    CHECK(!recompile_compute(wave64_null_exit_shader.data(),
                             wave64_null_exit_shader.size(),
                             &wave64_translation_resources,
                             wave64_null_exit_config).empty(),
          "exact-PC Wave64 null cycle reaches production compute translation");

    auto wave64_has_unresolved_sites = [](const std::vector<Rdna2Inst>& instructions) {
        return std::any_of(instructions.begin(), instructions.end(),
                           [](const Rdna2Inst& in) {
                               return in.pc == 2148 || in.pc == 2187;
                           });
    };
    for (int deviation = 0; deviation < 7; ++deviation) {
        std::vector<Rdna2Inst> changed = wave64_loop;
        auto at = [&](uint32_t pc) {
            return std::find_if(changed.begin(), changed.end(),
                                [&](const Rdna2Inst& in) { return in.pc == pc; });
        };
        if (deviation == 0) at(2442)->opcode = 0x07u;       // low-word compare only
        if (deviation == 1) at(2442)->src[0].value = 10;    // different mask pair
        if (deviation == 2) at(2443)->opcode = 0x03u;       // low-word EXEC copy only
        if (deviation == 3) at(2440)->dst.value = 106;      // not an ordinary complete pair
        if (deviation == 4) at(2440)->src[0].value = 0;     // compare is not against no-hit -1
        if (deviation == 5) at(2440)->src[1].value = 6;     // compare is not the ray result
        if (deviation == 6) at(2440)->dst.value = 105;      // pair overlaps architectural VCC
        ShaderResourceTable table = wave64_resources();
        CHECK(rdna2_specialize_proven_null_bvh_paths(changed, &table, 64) == 0 &&
                  wave64_has_unresolved_sites(changed),
              "Wave64 no-hit proof rejects width/pair/EXEC/result shape deviations");
    }

    ShaderResourceTable wave64_nonnull_resources = wave64_resources();
    wave64_nonnull_resources.resources[0].gpu_addr = 0x10000u;
    std::vector<Rdna2Inst> wave64_nonnull_loop = wave64_loop;
    CHECK(rdna2_specialize_proven_null_bvh_paths(
              wave64_nonnull_loop, &wave64_nonnull_resources, 64) == 0 &&
              wave64_has_unresolved_sites(wave64_nonnull_loop),
          "Wave64 cycle keeps unresolved sites when the root resource is not proven null");

    for (int deviation = 0; deviation < 2; ++deviation) {
        std::vector<Rdna2Inst> changed = wave64_loop;
        auto at = [&](uint32_t pc) {
            return std::find_if(changed.begin(), changed.end(),
                                [&](const Rdna2Inst& in) { return in.pc == pc; });
        };
        if (deviation == 0) at(2029)->src[0].value = 1; // non-empty initial stack
        if (deviation == 1) at(2033)->src[0].value = 36;// different root selector
        ShaderResourceTable table = wave64_resources();
        CHECK(rdna2_specialize_proven_null_bvh_paths(changed, &table, 64) == 1 &&
                  wave64_has_unresolved_sites(changed),
              "Wave64 empty-cycle proof rejects stack/selector deviations");
    }

    std::vector<Rdna2Inst> wave64_stack_pair_clobber = wave64_loop;
    const std::array<uint32_t, 1> stack_pair_clobber_words{
        0xBEAD047Eu}; // s_mov_b64 s[45:46], exec: high word overlaps stack s46
    Rdna2Inst stack_pair_clobber = rdna2_decode_one(
        stack_pair_clobber_words.data(), stack_pair_clobber_words.size());
    stack_pair_clobber.pc = 2032;
    *std::find_if(wave64_stack_pair_clobber.begin(), wave64_stack_pair_clobber.end(),
                  [](const Rdna2Inst& in) { return in.pc == 2032; }) = stack_pair_clobber;
    ShaderResourceTable wave64_stack_pair_resources = wave64_resources();
    CHECK(rdna2_specialize_proven_null_bvh_paths(
              wave64_stack_pair_clobber, &wave64_stack_pair_resources, 64) == 1 &&
              wave64_has_unresolved_sites(wave64_stack_pair_clobber),
          "Wave64 empty-cycle proof sees a B64 high-word stack clobber");

    std::vector<Rdna2Inst> wave64_external_entry = wave64_loop;
    const std::array<uint32_t, 1> wave64_external_words{0xBF82000Au}; // pc2028 -> pc2039
    Rdna2Inst wave64_external = rdna2_decode_one(
        wave64_external_words.data(), wave64_external_words.size());
    wave64_external.pc = 2028;
    wave64_external_entry.insert(wave64_external_entry.begin(), wave64_external);
    ShaderResourceTable wave64_external_resources = wave64_resources();
    CHECK(rdna2_specialize_proven_null_bvh_paths(
              wave64_external_entry, &wave64_external_resources, 64) == 1 &&
              wave64_has_unresolved_sites(wave64_external_entry),
          "Wave64 external entry bypassing stack initialization keeps the cycle fail-visible");

    const uint32_t runtime_dead_fetch[] = {
        0xBF0B8200u,             // pc0: s_cmp_le_u32 s0, 2 (entry/user SGPR)
        0xBF850002u,             // pc1: s_cbranch_scc1 pc4
        0xE0002000u, 0x80030100u,// pc2: runtime-reachable buffer_load_format_x ..., s[12:15]
        0xBF810000u,             // pc4: s_endpgm
    };
    std::vector<DynFetch> runtime_branch_fetch = resolve_dynamic_fetch(
        runtime_dead_fetch, std::size(runtime_dead_fetch), constant_branch_seed.data(),
        constant_branch_seed.size(), 0);
    CHECK(runtime_branch_fetch.size() == 1 && runtime_branch_fetch[0].fetch_pc == 2,
          "entry-dependent scalar branch keeps both resource paths fail-visible");

    const uint32_t vcc_clobbered_dead_fetch[] = {
        0xBEEA0381u,             // pc0: s_mov_b32 s106, 1 (ordinary scalar-data lifetime)
        0x7D846A80u,             // pc1: v_cmp_* -> implicit architectural VCC overwrite
        0xBF0B826Au,             // pc2: s_cmp_le_u32 s106, 2 (reads the new VCC bits)
        0xBF850002u,             // pc3: s_cbranch_scc1 pc6
        0xE0002000u, 0x80030100u,// pc4: potentially reachable buffer fetch
        0xBF810000u,             // pc6: s_endpgm
    };
    std::vector<Rdna2Inst> vcc_clobbered_instructions;
    rdna2_walk(vcc_clobbered_dead_fetch, std::size(vcc_clobbered_dead_fetch),
               vcc_clobbered_instructions);
    CHECK(rdna2_specialize_shader_constant_branches(vcc_clobbered_instructions) == 0 &&
              resolve_dynamic_fetch(vcc_clobbered_dead_fetch,
                                    std::size(vcc_clobbered_dead_fetch),
                                    constant_branch_seed.data(),
                                    constant_branch_seed.size(), 0).size() == 1,
          "implicit VCC writer invalidates a prior scalar-data constant in s106:s107");

    const uint32_t shader_constant_dead_bvh[] = {
        0xBE8B03A5u,                         // pc0:  s_mov_b32 s11, 37
        0x876A870Bu,                         // pc1:  s_and_b32 s106, s11, 7
        0x816AC46Au,                         // pc2:  s_add_i32 s106, s106, -4
        0xBF0B826Au,                         // pc3:  s_cmp_le_u32 s106, 2
        0xBF850005u,                         // pc4:  s_cbranch_scc1 pc10
        0xF1989F07u, 0x00040303u, 0x43440D3Fu, 0x46424140u, 0x00004847u,
        0xF1989F07u, 0x00010303u, 0x094F4E3Fu, 0x11100F0Eu, 0x00001312u,
        0xBF810000u,                         // pc15: s_endpgm
    };
    alignas(256) std::array<uint8_t, 256> shader_constant_null_bvh{};
    ShaderResourceTable shader_constant_bvh_table;
    { ShaderResource marker{};
      marker.cls = ResourceClass::ConstantBuffer;
      marker.format = DataFormat::Uint32;
      marker.num_components = 1;
      marker.binding = 4;
      marker.size = shader_constant_null_bvh.size();
      marker.fetch_pc = 10;
      marker.host_data = shader_constant_null_bvh.data();
      marker.host_data_size = shader_constant_null_bvh.size();
      shader_constant_bvh_table.resources.push_back(marker); }
    ComputeShaderConfig shader_constant_bvh_config;
    shader_constant_bvh_config.local_x = 1;
    CHECK(!recompile_compute(shader_constant_dead_bvh,
                             std::size(shader_constant_dead_bvh),
                             &shader_constant_bvh_table,
                             shader_constant_bvh_config).empty(),
          "compute translation omits a dead BVH site proven by shader constants");

    // Contiguous form of the exact null-result loop relationship above. PC5 is unresolved but can
    // execute only through a loop back-edge. A proven null marker at PC10 makes the PC20 SCC0 exit
    // unconditional; after that prune, the shader-constant PC4 entry branch removes PC5. This tests
    // the complete translation path rather than only the decoded-instruction helper.
    const uint32_t null_exit_loop_shader[] = {
        0xBE8B03A5u,                         // pc0:  s_mov_b32 s11, 37
        0x876A870Bu,                         // pc1:  loop condition from s11
        0x816AC46Au,                         // pc2
        0xBF0B826Au,                         // pc3
        0xBF850005u,                         // pc4:  initial entry -> pc10
        0xF1989F07u, 0x00040303u, 0x43440D3Fu, 0x46424140u, 0x00004847u,
        0xF1989F07u, 0x00010303u, 0x094F4E3Fu, 0x11100F0Eu, 0x00001312u,
        0xBF8C3F70u,                         // pc15: wait for null ray
        0x7D8A06F9u, 0x068688C1u,            // pc16: v_cmp_ne_u32 s8,-1,v3
        0xBF070880u,                         // pc18: s_cmp_lg_u32 0,s8
        0xBEFE036Au,                         // pc19: s_mov_b32 exec_lo,vcc_lo
        0xBF840002u,                         // pc20: s_cbranch_scc0 pc23
        0xBE8B03C1u,                         // pc21: loop-carried s11 = -1
        0xBF82FFEAu,                         // pc22: s_branch pc1
        0xBF810000u,                         // pc23: s_endpgm
    };
    ShaderResourceTable null_exit_loop_table;
    { ShaderResource marker{};
      marker.cls = ResourceClass::ConstantBuffer;
      marker.format = DataFormat::Uint32;
      marker.num_components = 1;
      marker.binding = 4;
      marker.size = shader_constant_null_bvh.size();
      marker.fetch_pc = 10;
      marker.host_data = shader_constant_null_bvh.data();
      marker.host_data_size = shader_constant_null_bvh.size();
      null_exit_loop_table.resources.push_back(marker); }
    ComputeShaderConfig null_exit_loop_config;
    null_exit_loop_config.local_x = 1;
    null_exit_loop_config.wave_size = 32;
    std::vector<Rdna2Inst> contiguous_null_exit;
    rdna2_walk(null_exit_loop_shader, std::size(null_exit_loop_shader), contiguous_null_exit);
    CHECK(rdna2_specialize_proven_null_bvh_paths(
              contiguous_null_exit, &null_exit_loop_table, 32) == 1 &&
              rdna2_specialize_shader_constant_branches(contiguous_null_exit) == 1 &&
              std::none_of(contiguous_null_exit.begin(), contiguous_null_exit.end(),
                           [](const Rdna2Inst& in) { return in.pc == 5 || in.pc == 21 || in.pc == 22; }),
          "contiguous null-result loop shares the discovery/translation specialization decision");
    CHECK(!recompile_compute(null_exit_loop_shader,
                             std::size(null_exit_loop_shader),
                             &null_exit_loop_table,
                             null_exit_loop_config).empty(),
          "proven null result prunes loop-reachable unresolved BVH before translation");

    ShaderResourceTable nonnull_exit_loop_table = null_exit_loop_table;
    nonnull_exit_loop_table.resources[0].gpu_addr = 0x20000u;
    CHECK(recompile_compute(null_exit_loop_shader,
                            std::size(null_exit_loop_shader),
                            &nonnull_exit_loop_table,
                            null_exit_loop_config).empty(),
          "different/non-null resource preserves unresolved loop BVH fail-visible");
    std::vector<uint32_t> changed_null_exit(std::begin(null_exit_loop_shader),
                                           std::end(null_exit_loop_shader));
    changed_null_exit[20] = 0xBF850002u;      // SCC1 is not the proven no-hit exit
    CHECK(recompile_compute(changed_null_exit.data(), changed_null_exit.size(),
                            &null_exit_loop_table,
                            null_exit_loop_config).empty(),
          "null loop guard opcode deviation remains fail-visible in translation");

    const uint32_t runtime_dead_bvh[] = {
        0xBF0B8200u,                         // pc0: s_cmp_le_u32 s0, 2
        0xBF850005u,                         // pc1: s_cbranch_scc1 pc7
        0xF1989F07u, 0x00040303u, 0x43440D3Fu, 0x46424140u, 0x00004847u,
        0xF1989F07u, 0x00010303u, 0x094F4E3Fu, 0x11100F0Eu, 0x00001312u,
        0xBF810000u,                         // pc12: s_endpgm
    };
    ShaderResourceTable runtime_bvh_table = shader_constant_bvh_table;
    runtime_bvh_table.resources[0].fetch_pc = 7;
    ComputeShaderConfig runtime_bvh_config = shader_constant_bvh_config;
    runtime_bvh_config.user_sgprs = {1u};
    CHECK(recompile_compute(runtime_dead_bvh, std::size(runtime_dead_bvh),
                            &runtime_bvh_table, runtime_bvh_config).empty(),
          "compute translation rejects an unresolved BVH on an entry-dependent branch");

    // Recreate the live descriptor builder rather than seeding its final words: the header pointer
    // is stored in 8-byte units, the allocation count is loaded from byte offset 0x58, and a
    // carry-propagating subtract produces the 64-byte count-minus-one before TYPE/mode are ORed in.
    // Dropping either S_LSHL_B64's high half or S_ADDC_U32's carry leaves this use unresolved.
    astro_bvh_backing[22] = 64u;
    astro_bvh_backing[23] = 0u;
    const uint32_t astro_bvh_builder[] = {
        0x8F88831Cu,                         // pc0:  s_lshl_b64 s[8:9], s[28:29], 3
        0xF4100404u, 0xFA000000u,            // pc1:  s_load_dwordx16 s[16:31], s[8:9], 0
        0xF4080904u, 0xFA000058u,            // pc3:  s_load_dwordx4 s[36:39], s[8:9], 0x58
        0x8012C124u,                         // pc5:  s_add_u32 s18, s36, -1
        0x8213C125u,                         // pc6:  s_addc_u32 s19, s37, -1
        0x876A13FFu, 0x000003FFu,            // pc7:  s_and_b32 s106, 0x3ff, s19
        0x9490FF08u, 0x00280008u,            // pc9:  s_bfe_u64 s[16:17], s[8:9], 8:40
        0x8813FF6Au, 0x81000000u,            // pc11: s_or_b32 s19, s106, TYPE/mode
        0xF1989F07u, 0x00040303u, 0x43440D3Fu, 0x46424140u, 0x00004847u,
        0xBF810000u,
    };
    std::array<uint32_t, 40> astro_builder_seed{};
    const uint64_t astro_bvh_base8 = astro_bvh_base >> 3;
    astro_builder_seed[28] = static_cast<uint32_t>(astro_bvh_base8);
    astro_builder_seed[29] = static_cast<uint32_t>(astro_bvh_base8 >> 32);
    std::vector<SrtUse> astro_builder_uses;
    resolve_dynamic_fetch(astro_bvh_builder, std::size(astro_bvh_builder),
                          astro_builder_seed.data(), astro_builder_seed.size(), 0,
                          &astro_builder_uses);
    const DecodedBvhDescriptor built_bvh = astro_builder_uses.size() == 1
        ? decode_bvh_descriptor(astro_builder_uses[0].bvh4.data())
        : DecodedBvhDescriptor{};
    CHECK(astro_builder_uses.size() == 1 && astro_builder_uses[0].kind == 2 &&
              built_bvh.base == astro_bvh_base &&
              built_bvh.size_bytes == sizeof(astro_bvh_backing) &&
              built_bvh.type == 8u && built_bvh.triangle_return_mode,
          "Astro's carry/shift descriptor builder resolves the exact BVH binding");

    // GTA V builds the same RTIP 1.1 descriptor from a pointer inside a mapped 16-dword object
    // header. Both live programs enable BOX_SORT_EN. The second relocates the final descriptor from
    // the loaded s8:s23 range into s44:s47, so a load-time snapshot alone cannot prove it.
    astro_bvh_backing[22] = 63u; // descriptor stores count-minus-one: 64 * 64 = 4096 bytes
    astro_bvh_backing[23] = 0u;
    alignas(8) std::array<uint32_t, 16> gta_bvh_header{};
    gta_bvh_header[12] = static_cast<uint32_t>(astro_bvh_base8);
    gta_bvh_header[13] = static_cast<uint32_t>(astro_bvh_base8 >> 32);
    const uint64_t gta_bvh_header_addr = reinterpret_cast<uint64_t>(gta_bvh_header.data());
    auto put_words = [](std::vector<uint32_t>& code, uint32_t pc,
                        std::initializer_list<uint32_t> words) {
        std::copy(words.begin(), words.end(), code.begin() + pc);
    };

    std::vector<uint32_t> gta_bvh_pc1180(1186u, 0xbf800000u); // s_nop 0
    put_words(gta_bvh_pc1180, 1107u, {0xf4100226u, 0xfa000000u});
    put_words(gta_bvh_pc1180, 1158u, {
        0x8715ff15u, 0x003fffffu,             // s_and_b32 s21,s21,0x003fffff
        0x8f948314u,                          // s_lshl_b64 s[20:21],s[20:21],3
        0xf408000au, 0xfa000058u,             // s_load_dwordx4 s[0:3],s[20:21],0x58
        0x9490ff14u, 0x00280008u,             // s_bfe_u64 s[16:17],s[20:21],8:40
        0xbf8cc07fu,                          // s_waitcnt
        0x816a8100u,                          // s_add_i32 vcc_lo,s0,1
        0x8801ff15u, 0x00080000u,             // unrelated header patch
        0x80126ac1u,                          // s_add_u32 s18,-1,vcc_lo
        0xbe800314u,                          // unrelated move
        0x821380c1u,                          // s_addc_u32 s19,-1,0
        0xbe8303ffu, 0x00016204u,             // unrelated move
        0x876a13ffu, 0x000003ffu,             // s_and_b32 vcc_lo,0x3ff,s19
        0xbe911d9fu,                          // s_bitset1_b32 s17,31
        0x8813ff6au, 0x81000000u,             // s_or_b32 s19,vcc_lo,TYPE/mode
        0xbf8c3f70u,
        0xf1989f07u, 0x00040006u, 0x2726251cu, 0x2f2a2928u, 0x00002d2eu,
        0xbf810000u,
    });
    std::array<uint32_t, 78> gta_pc1180_seed{};
    gta_pc1180_seed[76] = static_cast<uint32_t>(gta_bvh_header_addr);
    gta_pc1180_seed[77] = static_cast<uint32_t>(gta_bvh_header_addr >> 32);
    std::vector<SrtUse> gta_pc1180_uses;
    resolve_dynamic_fetch(gta_bvh_pc1180.data(), gta_bvh_pc1180.size(),
                          gta_pc1180_seed.data(), gta_pc1180_seed.size(), 0,
                          &gta_pc1180_uses);
    CHECK(gta_pc1180_uses.size() == 1 && gta_pc1180_uses[0].kind == 2 &&
              gta_pc1180_uses[0].use_pc == 1180u &&
              decode_bvh_descriptor(gta_pc1180_uses[0].bvh4.data()).sort_enabled,
          "GTA pc1180 publishes its exact sorted BVH descriptor");
    ShaderResourceTable gta_pc1180_table;
    add_compute_buffer_resources(gta_pc1180_table, gta_bvh_pc1180.data(),
                                 gta_bvh_pc1180.size(), gta_pc1180_seed.data(),
                                 gta_pc1180_seed.size());
    CHECK(gta_pc1180_table.resources.size() == 1 &&
              gta_pc1180_table.resources[0].fetch_pc == 1180u &&
              gta_pc1180_table.resources[0].gpu_addr == astro_bvh_base &&
              gta_pc1180_table.resources[0].size == sizeof(astro_bvh_backing) &&
              gta_pc1180_table.resources[0].bvh_sort_enabled,
          "GTA pc1180 materializes BOX_SORT_EN in its instruction-scoped BVH binding");

    std::vector<uint32_t> gta_unproven_pc1180 = gta_bvh_pc1180;
    gta_unproven_pc1180[1160] = 0x8f948318u; // same lshl site/value, but s24:s25 has no x16 origin
    std::array<uint32_t, 78> gta_unproven_pc1180_seed = gta_pc1180_seed;
    gta_unproven_pc1180_seed[24] = static_cast<uint32_t>(astro_bvh_base8);
    gta_unproven_pc1180_seed[25] = static_cast<uint32_t>(astro_bvh_base8 >> 32);
    std::vector<SrtUse> gta_unproven_pc1180_uses;
    resolve_dynamic_fetch(gta_unproven_pc1180.data(), gta_unproven_pc1180.size(),
                          gta_unproven_pc1180_seed.data(), gta_unproven_pc1180_seed.size(), 0,
                          &gta_unproven_pc1180_uses);
    CHECK(gta_unproven_pc1180_uses.empty(),
          "GTA pc1180 rejects byte-identical descriptor values from an unproven pointer origin");
    ShaderResourceTable gta_unproven_pc1180_table;
    add_compute_buffer_resources(gta_unproven_pc1180_table, gta_unproven_pc1180.data(),
                                 gta_unproven_pc1180.size(),
                                 gta_unproven_pc1180_seed.data(),
                                 gta_unproven_pc1180_seed.size());
    CHECK(gta_unproven_pc1180_table.resources.empty(),
          "GTA pc1180 cannot materialize an ALU-modified historical x16 snapshot");

    // The live pc1180 dispatch carries a null root in header qword s20:s21. Unlike the non-null
    // positive above, its count dereference at pc1161 must fail; the exact scalar chain still proves
    // that all four descriptor words derive from that one null qword. Keep the real EXEC writer and
    // EXECZ branch, shortened only by replacing their distant merge body with NOPs.
    alignas(8) std::array<uint32_t, 17> gta_null_bvh_header{};
    const uint64_t gta_null_bvh_header_addr =
        reinterpret_cast<uint64_t>(gta_null_bvh_header.data());
    std::vector<uint32_t> gta_null_pc1180 = gta_bvh_pc1180;
    gta_null_pc1180.resize(1489u, 0xbf800000u);
    gta_null_pc1180[1099] = 0xbeca246au; // s_and_saveexec_b64 s[74:75],vcc
    gta_null_pc1180[1100] = 0xbf880182u; // s_cbranch_execz pc1487
    gta_null_pc1180[1185] = 0xbf800000u; // do not terminate before the guard merge
    gta_null_pc1180[1487] = 0xbf800000u;
    gta_null_pc1180[1488] = 0xbf810000u;
    std::array<uint32_t, 78> gta_null_pc1180_seed{};
    gta_null_pc1180_seed[76] = static_cast<uint32_t>(gta_null_bvh_header_addr);
    gta_null_pc1180_seed[77] = static_cast<uint32_t>(gta_null_bvh_header_addr >> 32);
    std::vector<SrtUse> gta_null_pc1180_uses;
    resolve_dynamic_fetch(gta_null_pc1180.data(), gta_null_pc1180.size(),
                          gta_null_pc1180_seed.data(), gta_null_pc1180_seed.size(), 0,
                          &gta_null_pc1180_uses);
    CHECK(gta_null_pc1180_uses.size() == 1 && gta_null_pc1180_uses[0].kind == 3 &&
              gta_null_pc1180_uses[0].use_pc == 1180u,
          "GTA pc1180 publishes its guarded x16-header null BVH use");
    ShaderResourceTable gta_null_pc1180_table;
    add_compute_buffer_resources(gta_null_pc1180_table, gta_null_pc1180.data(),
                                 gta_null_pc1180.size(), gta_null_pc1180_seed.data(),
                                 gta_null_pc1180_seed.size());
    CHECK(gta_null_pc1180_table.resources.size() == 1 &&
              gta_null_pc1180_table.resources[0].fetch_pc == 1180u &&
              is_proven_null_bvh(gta_null_pc1180_table.resources[0]),
          "GTA pc1180 materializes its guarded null root as a no-hit BVH marker");

    std::vector<uint32_t> gta_null_offset_pc1180 = gta_null_pc1180;
    gta_null_offset_pc1180[1108] = 0xfa000004u; // same x16 load, not the header-base site
    std::vector<SrtUse> gta_null_offset_pc1180_uses;
    resolve_dynamic_fetch(gta_null_offset_pc1180.data(), gta_null_offset_pc1180.size(),
                          gta_null_pc1180_seed.data(), gta_null_pc1180_seed.size(), 0,
                          &gta_null_offset_pc1180_uses);
    CHECK(gta_null_offset_pc1180_uses.empty(),
          "GTA pc1180 does not seed null provenance from a shifted x16 header load");

    std::vector<uint32_t> gta_null_unproven_pc1180 = gta_null_pc1180;
    gta_null_unproven_pc1180[1160] = 0x8f948318u; // s24:s25 is zero but not header-derived
    std::vector<SrtUse> gta_null_unproven_pc1180_uses;
    resolve_dynamic_fetch(gta_null_unproven_pc1180.data(), gta_null_unproven_pc1180.size(),
                          gta_null_pc1180_seed.data(), gta_null_pc1180_seed.size(), 0,
                          &gta_null_unproven_pc1180_uses);
    CHECK(gta_null_unproven_pc1180_uses.empty(),
          "GTA pc1180 rejects byte-identical nulls from an unproven qword");

    // Each aligned zero qword receives a distinct origin. Copy one dword from each of two adjacent
    // qwords into a nominal 64-bit operand: the shift must not collapse that splice onto one origin.
    std::vector<uint32_t> gta_null_spliced_pc1180 = gta_null_pc1180;
    gta_null_spliced_pc1180[1156] = 0xbe980314u; // s_mov_b32 s24,s20 (qword A low)
    gta_null_spliced_pc1180[1157] = 0xbe990316u; // s_mov_b32 s25,s22 (qword B low)
    gta_null_spliced_pc1180[1160] = 0x8f948318u; // consume spliced s24:s25 pair
    std::vector<SrtUse> gta_null_spliced_pc1180_uses;
    resolve_dynamic_fetch(gta_null_spliced_pc1180.data(), gta_null_spliced_pc1180.size(),
                          gta_null_pc1180_seed.data(), gta_null_pc1180_seed.size(), 0,
                          &gta_null_spliced_pc1180_uses);
    CHECK(gta_null_spliced_pc1180_uses.empty(),
          "GTA pc1180 rejects a 64-bit pair spliced from distinct header qwords");

    // A conditional edge may not bypass the mapped x16 load and then enter its dependent builder.
    // Seed the skipped path with a non-null root so accepting the linear walk as null would change
    // guest-visible ray results, not merely attach an imprecise provenance label.
    std::vector<uint32_t> gta_null_skipped_load_pc1180 = gta_null_pc1180;
    gta_null_skipped_load_pc1180[1105] = 0xbf850003u; // s_cbranch_scc1 pc1109
    std::array<uint32_t, 78> gta_null_skipped_load_seed = gta_null_pc1180_seed;
    gta_null_skipped_load_seed[20] = static_cast<uint32_t>(astro_bvh_base8);
    gta_null_skipped_load_seed[21] = static_cast<uint32_t>(astro_bvh_base8 >> 32);
    std::vector<SrtUse> gta_null_skipped_load_pc1180_uses;
    resolve_dynamic_fetch(gta_null_skipped_load_pc1180.data(),
                          gta_null_skipped_load_pc1180.size(),
                          gta_null_skipped_load_seed.data(),
                          gta_null_skipped_load_seed.size(), 0,
                          &gta_null_skipped_load_pc1180_uses);
    CHECK(gta_null_skipped_load_pc1180_uses.empty(),
          "GTA pc1180 rejects null provenance whose x16 seed does not dominate the use");

    std::vector<uint32_t> gta_null_indirect_pc1180 = gta_null_pc1180;
    gta_null_indirect_pc1180[1101] = 0xbe802000u; // s_setpc_b64 s[0:1]
    std::vector<SrtUse> gta_null_indirect_pc1180_uses;
    resolve_dynamic_fetch(gta_null_indirect_pc1180.data(), gta_null_indirect_pc1180.size(),
                          gta_null_pc1180_seed.data(), gta_null_pc1180_seed.size(), 0,
                          &gta_null_indirect_pc1180_uses);
    CHECK(gta_null_indirect_pc1180_uses.empty(),
          "GTA pc1180 rejects a null proof in a program with indirect control flow");

    std::vector<uint32_t> gta_null_unguarded_pc1180 = gta_null_pc1180;
    gta_null_unguarded_pc1180[1100] = 0xbf800000u; // remove exact EXECZ region proof
    std::vector<SrtUse> gta_null_unguarded_pc1180_uses;
    resolve_dynamic_fetch(gta_null_unguarded_pc1180.data(),
                          gta_null_unguarded_pc1180.size(),
                          gta_null_pc1180_seed.data(), gta_null_pc1180_seed.size(), 0,
                          &gta_null_unguarded_pc1180_uses);
    CHECK(gta_null_unguarded_pc1180_uses.empty(),
          "GTA pc1180 keeps an unguarded x16-header null BVH fail-visible");

    std::vector<uint32_t> gta_bvh_pc313(319u, 0xbf800000u); // s_nop 0
    put_words(gta_bvh_pc313, 244u, {0xf4100203u, 0xfa000000u});
    put_words(gta_bvh_pc313, 294u, {
        0x8715ff15u, 0x003fffffu,             // s_and_b32 s21,s21,0x003fffff
        0x8f888314u,                          // s_lshl_b64 s[8:9],s[20:21],3
        0xf4080304u, 0xfa000058u,             // s_load_dwordx4 s[12:15],s[8:9],0x58
        0xbf8cc07fu,
        0x816a810cu,                          // s_add_i32 vcc_lo,s12,1
        0x94acff08u, 0x00280008u,             // s_bfe_u64 s[44:45],s[8:9],8:40
        0x802e6ac1u,                          // s_add_u32 s46,-1,vcc_lo
        0xbe8a030eu,                          // unrelated move
        0x822f80c1u,                          // s_addc_u32 s47,-1,0
        0xbe891d93u,                          // unrelated bitset
        0x876a2fffu, 0x000003ffu,             // s_and_b32 vcc_lo,0x3ff,s47
        0xbead1d9fu,                          // s_bitset1_b32 s45,31
        0x882fff6au, 0x81000000u,             // s_or_b32 s47,vcc_lo,TYPE/mode
        0xbf8c3f70u,
        0xf1989f07u, 0x000b0004u, 0x2322211bu, 0x2b262524u, 0x0000292au,
        0xbf810000u,
    });
    std::array<uint32_t, 26> gta_pc313_seed{};
    gta_pc313_seed[6] = static_cast<uint32_t>(gta_bvh_header_addr);
    gta_pc313_seed[7] = static_cast<uint32_t>(gta_bvh_header_addr >> 32);
    gta_pc313_seed[24] = static_cast<uint32_t>(astro_bvh_base8);
    gta_pc313_seed[25] = static_cast<uint32_t>(astro_bvh_base8 >> 32);
    std::vector<SrtUse> gta_pc313_uses;
    resolve_dynamic_fetch(gta_bvh_pc313.data(), gta_bvh_pc313.size(),
                          gta_pc313_seed.data(), gta_pc313_seed.size(), 0,
                          &gta_pc313_uses);
    CHECK(gta_pc313_uses.size() == 1 && gta_pc313_uses[0].kind == 2 &&
              gta_pc313_uses[0].use_pc == 313u &&
              decode_bvh_descriptor(gta_pc313_uses[0].bvh4.data()).sort_enabled,
          "GTA pc313 proves the relocated sorted BVH descriptor builder");

    std::vector<uint32_t> gta_unproven_pc313 = gta_bvh_pc313;
    gta_unproven_pc313[296] = 0x8f888318u; // same lshl site, identical seed value, no x16 origin
    std::vector<SrtUse> gta_unproven_pc313_uses;
    resolve_dynamic_fetch(gta_unproven_pc313.data(), gta_unproven_pc313.size(),
                          gta_pc313_seed.data(), gta_pc313_seed.size(), 0,
                          &gta_unproven_pc313_uses);
    CHECK(gta_unproven_pc313_uses.empty(),
          "GTA pc313 rejects byte-identical descriptor values from an unproven pointer origin");

    // Astro Bot's live visibility packet consumes a direct R32_UINT T# in s[0:7]. Opcode 0x0f is
    // IMAGE_ATOMIC_SWAP, so its instruction-scoped resource must be materialized as a storage image
    // even though the packet's unused SSAMP field aliases s0.
    const uint32_t image_atomic_swap[] = {
        0xf03c2108u, 0x00000900u,
        0xbf810000u,
    };
    const uint32_t atomic_image_seed[8] = {
        0x055c0100u, 0xc1400000u, 0x001fc01fu, 0x91b00204u,
        0x00000000u, 0x00700000u, 0x00000000u, 0x00000000u,
    };
    std::vector<SrtUse> atomic_image_uses;
    resolve_dynamic_fetch(image_atomic_swap, std::size(image_atomic_swap),
                          atomic_image_seed, std::size(atomic_image_seed), 0,
                          &atomic_image_uses);
    const std::array<uint32_t, 8> expected_atomic_t8 = {
        0x055c0100u, 0xc1400000u, 0x001fc01fu, 0x91b00204u,
        0x00000000u, 0x00700000u, 0x00000000u, 0x00000000u,
    };
    CHECK(atomic_image_uses.size() == 1 && atomic_image_uses[0].kind == 0 &&
              atomic_image_uses[0].use_pc == 0 &&
              atomic_image_uses[0].is_storage_image &&
              atomic_image_uses[0].t8 == expected_atomic_t8,
          "image_atomic_swap publishes the live direct T# as a storage-image use");
    const uint32_t image_atomic_add[] = {
        0xf0442108u, 0x00000900u,
        0xbf810000u,
    };
    std::vector<SrtUse> atomic_add_uses;
    resolve_dynamic_fetch(image_atomic_add, std::size(image_atomic_add),
                          atomic_image_seed, std::size(atomic_image_seed), 0,
                          &atomic_add_uses);
    CHECK(atomic_add_uses.size() == 1 && atomic_add_uses[0].kind == 0 &&
              atomic_add_uses[0].use_pc == 0 &&
              atomic_add_uses[0].is_storage_image &&
              atomic_add_uses[0].t8 == expected_atomic_t8,
          "image_atomic_add publishes the live direct T# as a storage-image use");

    // GTA V's gameplay compute programs supply IMAGE_LOAD_MIP's final address VGPR and
    // IMAGE_STORE_MIP's NSA mip VGPR with a plain move from a zero wave-uniform SGPR. The proof is
    // instruction-scoped: changing that exact reaching definition, its scalar value, or its basic
    // block must clear the marker while retaining the ordinary image use for fail-visible lowering.
    uint32_t zero_mip_seed[28]{};
    std::copy(std::begin(atomic_image_seed), std::end(atomic_image_seed), zero_mip_seed + 20);
    const uint32_t gta_load_mip_2d[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7 (known zero)
        0xf0043108u, 0x00050000u,            // IMAGE_LOAD_MIP 2D [v0,v1,v2], s[20:27]
        0xbf810000u,
    };
    std::vector<SrtUse> zero_mip_load_uses;
    resolve_dynamic_fetch(gta_load_mip_2d, std::size(gta_load_mip_2d),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &zero_mip_load_uses);
    CHECK(zero_mip_load_uses.size() == 1 && !zero_mip_load_uses[0].is_storage_image &&
              zero_mip_load_uses[0].use_pc == 1 && zero_mip_load_uses[0].proven_zero_mip,
          "GTA V IMAGE_LOAD_MIP 2D carries its same-block v2=zero proof");

    // Exact 0x2042f49200 gameplay program. The pc10 VOP2 writes only v0: it must not erase the
    // independent v2=zero definition at pc5 before IMAGE_LOAD_MIP at pc14. The later pc12 v5=zero
    // definition similarly reaches the exact IMAGE_STORE_MIP packet at pc17.
    alignas(16) uint32_t live_zero_mip_table[16]{};
    std::copy(std::begin(atomic_image_seed), std::end(atomic_image_seed),
              std::begin(live_zero_mip_table));
    uint32_t live_zero_mip_seed[28]{};
    const uint64_t live_zero_mip_table_address =
        reinterpret_cast<uint64_t>(live_zero_mip_table);
    live_zero_mip_seed[0] = static_cast<uint32_t>(live_zero_mip_table_address);
    live_zero_mip_seed[1] = static_cast<uint32_t>(live_zero_mip_table_address >> 32);
    std::copy(std::begin(atomic_image_seed), std::end(atomic_image_seed),
              live_zero_mip_seed + 20);
    const uint32_t gta_live_load_store_mip[] = {
        0xbfa00001u,
        0xd7460004u, 0x04010a08u,
        0x4a020af9u, 0x86860609u,
        0x7e040207u,                         // pc5: v_mov_b32 v2, s7 (known zero)
        0xf4100300u, 0xfa000000u,
        0x4a0606f9u, 0x86860609u,
        0x4a000804u,                         // pc10: v_add_nc_u32 v0, s4, v4
        0x4a080802u,
        0x7e0a0206u,                         // pc12: v_mov_b32 v5, s6 (known zero)
        0xbf8cc07fu,
        0xf0043108u, 0x00050000u,            // pc14: IMAGE_LOAD_MIP, mip=v2
        0xbf8c3f70u,
        0xf024310au, 0x00030004u, 0x00000503u, // pc17: IMAGE_STORE_MIP, mip=v5
        0xbf810000u,
    };
    std::vector<SrtUse> gta_live_zero_mip_uses;
    resolve_dynamic_fetch(gta_live_load_store_mip, std::size(gta_live_load_store_mip),
                          live_zero_mip_seed, std::size(live_zero_mip_seed), 0,
                          &gta_live_zero_mip_uses);
    const auto live_load_mip = std::find_if(
        gta_live_zero_mip_uses.begin(), gta_live_zero_mip_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 14u; });
    const auto live_store_mip = std::find_if(
        gta_live_zero_mip_uses.begin(), gta_live_zero_mip_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 17u; });
    CHECK(live_load_mip != gta_live_zero_mip_uses.end() &&
              !live_load_mip->is_storage_image && live_load_mip->proven_zero_mip &&
              live_store_mip != gta_live_zero_mip_uses.end() &&
              live_store_mip->is_storage_image && live_store_mip->proven_zero_mip,
          "GTA V's exact pc5/pc10/pc14 load and pc12/pc17 store retain both zero-mip proofs");

    const uint32_t gta_load_mip_2da[] = {
        0x7e060206u,                         // v_mov_b32 v3, s6 (known zero; v2 is slice)
        0xf0043128u, 0x00050000u,            // IMAGE_LOAD_MIP 2D_ARRAY [v0,v1,v2,v3]
        0xbf810000u,
    };
    std::vector<SrtUse> zero_mip_array_uses;
    resolve_dynamic_fetch(gta_load_mip_2da, std::size(gta_load_mip_2da),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &zero_mip_array_uses);
    CHECK(zero_mip_array_uses.size() == 1 && zero_mip_array_uses[0].proven_zero_mip,
          "GTA V IMAGE_LOAD_MIP 2D_ARRAY proves v3 without consuming the v2 slice");

    uint32_t nonzero_mip_seed[28]{};
    std::copy(std::begin(zero_mip_seed), std::end(zero_mip_seed), nonzero_mip_seed);
    nonzero_mip_seed[7] = 1u;
    std::vector<SrtUse> nonzero_mip_uses;
    resolve_dynamic_fetch(gta_load_mip_2d, std::size(gta_load_mip_2d),
                          nonzero_mip_seed, std::size(nonzero_mip_seed), 0,
                          &nonzero_mip_uses);
    CHECK(nonzero_mip_uses.size() == 1 && !nonzero_mip_uses[0].proven_zero_mip,
          "nonzero wave-uniform mip remains an image use but receives no specialization marker");

    std::array<uint32_t, std::size(gta_load_mip_2d)> changed_mip_writer{};
    std::copy(std::begin(gta_load_mip_2d), std::end(gta_load_mip_2d),
              changed_mip_writer.begin());
    changed_mip_writer[0] = 0x7e060207u;     // same site now writes v3, not the consumed v2
    clear_shader_decode_cache();
    std::vector<SrtUse> changed_mip_writer_uses;
    resolve_dynamic_fetch(changed_mip_writer.data(), changed_mip_writer.size(),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &changed_mip_writer_uses);
    CHECK(changed_mip_writer_uses.size() == 1 &&
              !changed_mip_writer_uses[0].proven_zero_mip,
          "same-site v_mov destination mutation removes the IMAGE_LOAD_MIP proof");

    const uint32_t branched_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xbf820000u,                         // s_branch to the immediately following new block
        0xf0043108u, 0x00050000u,
        0xbf810000u,
    };
    std::vector<SrtUse> branched_zero_mip_uses;
    resolve_dynamic_fetch(branched_zero_mip, std::size(branched_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &branched_zero_mip_uses);
    CHECK(branched_zero_mip_uses.size() == 1 &&
              !branched_zero_mip_uses[0].proven_zero_mip,
          "a control-flow boundary prevents a stale zero-mip reaching-definition proof");

    const uint32_t dropped_block_start_zero_mip[] = {
        0x7e040207u,                         // pc0: v_mov_b32 v2, s7
        0xbf820000u,                         // pc1: s_branch pc2
        0xf80000cfu, 0x00000000u,            // pc2: exp (omitted from the compact fold stream)
        0xf0043108u, 0x00050000u,            // pc4: IMAGE_LOAD_MIP
        0xbf810000u,
    };
    std::vector<SrtUse> dropped_block_start_uses;
    resolve_dynamic_fetch(dropped_block_start_zero_mip,
                          std::size(dropped_block_start_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &dropped_block_start_uses);
    CHECK(dropped_block_start_uses.size() == 1 &&
              !dropped_block_start_uses[0].proven_zero_mip,
          "a compacted-out first instruction still starts a new zero-mip proof block");

    const uint32_t exec_changed_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0x7da40100u,                         // v_cmpx_eq_u32 v0,v0 (implicit EXEC write)
        0xf0043108u, 0x00050000u,
        0xbf810000u,
    };
    std::vector<SrtUse> exec_changed_zero_mip_uses;
    resolve_dynamic_fetch(exec_changed_zero_mip, std::size(exec_changed_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &exec_changed_zero_mip_uses);
    CHECK(exec_changed_zero_mip_uses.size() == 1 &&
              !exec_changed_zero_mip_uses[0].proven_zero_mip,
          "an otherwise-compacted v_cmpx invalidates the predicated all-lanes zero proof");

    const uint32_t predicated_zero_move[] = {
        0x7da40100u,                         // v_cmpx_eq_u32 v0,v0 (EXEC is now untracked)
        0x7e040207u,                         // predicated v_mov_b32 v2, s7
        0xf0043108u, 0x00050000u,
        0xbf810000u,
    };
    std::vector<SrtUse> predicated_zero_move_uses;
    resolve_dynamic_fetch(predicated_zero_move, std::size(predicated_zero_move),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &predicated_zero_move_uses);
    CHECK(predicated_zero_move_uses.size() == 1 &&
              !predicated_zero_move_uses[0].proven_zero_mip,
          "a v_mov after an untracked EXEC mutation cannot manufacture an all-lanes zero proof");

    const uint32_t valu_overwrites_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0x4a040300u,                         // v_add_nc_u32 v2, v0, v1 (compacted without retention)
        0xf0043108u, 0x00050000u,
        0xbf810000u,
    };
    std::vector<SrtUse> valu_overwrites_zero_mip_uses;
    resolve_dynamic_fetch(valu_overwrites_zero_mip, std::size(valu_overwrites_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &valu_overwrites_zero_mip_uses);
    CHECK(valu_overwrites_zero_mip_uses.size() == 1 &&
              !valu_overwrites_zero_mip_uses[0].proven_zero_mip,
          "an otherwise-compacted overlapping VALU writer kills the retained zero proof");

    const uint32_t mimg_overwrites_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xf0003108u, 0x00050200u,            // IMAGE_LOAD v2, [v0,v1], s[20:27]
        0xf0043108u, 0x00050000u,            // IMAGE_LOAD_MIP consumes the overwritten v2
        0xbf810000u,
    };
    std::vector<SrtUse> mimg_overwrites_zero_mip_uses;
    resolve_dynamic_fetch(mimg_overwrites_zero_mip, std::size(mimg_overwrites_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &mimg_overwrites_zero_mip_uses);
    const auto overwritten_mip_use = std::find_if(
        mimg_overwrites_zero_mip_uses.begin(), mimg_overwrites_zero_mip_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 3u; });
    CHECK(overwritten_mip_use != mimg_overwrites_zero_mip_uses.end() &&
              !overwritten_mip_use->proven_zero_mip,
          "an intervening MIMG VDATA write kills the zero proof at the later mip use");

    const uint32_t wide_mimg_overwrites_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xf0003f08u, 0x00050000u,            // IMAGE_LOAD dmask:xyzw writes v[0:3]
        0xf0043108u, 0x00050000u,            // IMAGE_LOAD_MIP consumes overwritten v2
        0xbf810000u,
    };
    std::vector<SrtUse> wide_mimg_overwrite_uses;
    resolve_dynamic_fetch(wide_mimg_overwrites_zero_mip,
                          std::size(wide_mimg_overwrites_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &wide_mimg_overwrite_uses);
    const auto wide_mimg_mip_use = std::find_if(
        wide_mimg_overwrite_uses.begin(), wide_mimg_overwrite_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 3u; });
    CHECK(wide_mimg_mip_use != wide_mimg_overwrite_uses.end() &&
              !wide_mimg_mip_use->proven_zero_mip,
          "an overlapping four-register MIMG result kills every covered zero-mip proof");

    const uint32_t wide_valu_overwrites_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xd5761e01u, 0x040a0100u,            // v_mad_u64_u32 writes v[1:2]
        0xf0043108u, 0x00050000u,            // IMAGE_LOAD_MIP consumes overwritten v2
        0xbf810000u,
    };
    std::vector<SrtUse> wide_valu_overwrite_uses;
    resolve_dynamic_fetch(wide_valu_overwrites_zero_mip,
                          std::size(wide_valu_overwrites_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &wide_valu_overwrite_uses);
    const auto wide_valu_mip_use = std::find_if(
        wide_valu_overwrite_uses.begin(), wide_valu_overwrite_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 3u; });
    CHECK(wide_valu_mip_use != wide_valu_overwrite_uses.end() &&
              !wide_valu_mip_use->proven_zero_mip,
          "an overlapping two-register VOP3 result kills the covered zero-mip proof");

    const uint32_t movreld_overwrites_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xbefc0382u,                         // s_mov_b32 m0, 2
        0x7e008500u,                         // v_movreld_b32 v0, v0 writes dynamic v2
        0xf0043108u, 0x00050000u,            // IMAGE_LOAD_MIP consumes overwritten v2
        0xbf810000u,
    };
    std::vector<SrtUse> movreld_overwrite_uses;
    resolve_dynamic_fetch(movreld_overwrites_zero_mip,
                          std::size(movreld_overwrites_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &movreld_overwrite_uses);
    const auto movreld_mip_use = std::find_if(
        movreld_overwrite_uses.begin(), movreld_overwrite_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 3u; });
    CHECK(movreld_mip_use != movreld_overwrite_uses.end() &&
              !movreld_mip_use->proven_zero_mip,
          "an M0-relative dynamic VGPR destination kills every zero-mip proof");

    const uint32_t mimg_tfe_overwrites_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xf0010308u, 0x00050000u,            // IMAGE_LOAD TFE dmask:xy writes data v[0:1], status v2
        0xf0043108u, 0x00050000u,            // IMAGE_LOAD_MIP consumes overwritten v2
        0xbf810000u,
    };
    std::vector<SrtUse> mimg_tfe_overwrite_uses;
    resolve_dynamic_fetch(mimg_tfe_overwrites_zero_mip,
                          std::size(mimg_tfe_overwrites_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &mimg_tfe_overwrite_uses);
    const auto mimg_tfe_mip_use = std::find_if(
        mimg_tfe_overwrite_uses.begin(), mimg_tfe_overwrite_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 3u; });
    CHECK(mimg_tfe_mip_use != mimg_tfe_overwrite_uses.end() &&
              !mimg_tfe_mip_use->proven_zero_mip,
          "a MIMG TFE status destination kills the adjacent zero-mip proof");

    const uint32_t mimg_store_tfe_overwrites_zero_mip[] = {
        0x7e040207u,                         // v_mov_b32 v2, s7
        0xf0210308u, 0x00050000u,            // IMAGE_STORE TFE dmask:xy reads v[0:1], writes status v2
        0xf0043108u, 0x00050000u,            // IMAGE_LOAD_MIP consumes overwritten v2
        0xbf810000u,
    };
    std::vector<SrtUse> mimg_store_tfe_overwrite_uses;
    resolve_dynamic_fetch(mimg_store_tfe_overwrites_zero_mip,
                          std::size(mimg_store_tfe_overwrites_zero_mip),
                          zero_mip_seed, std::size(zero_mip_seed), 0,
                          &mimg_store_tfe_overwrite_uses);
    const auto mimg_store_tfe_mip_use = std::find_if(
        mimg_store_tfe_overwrite_uses.begin(), mimg_store_tfe_overwrite_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 3u; });
    CHECK(mimg_store_tfe_mip_use != mimg_store_tfe_overwrite_uses.end() &&
              !mimg_store_tfe_mip_use->proven_zero_mip,
          "a MIMG store TFE status destination kills the adjacent zero-mip proof");

    uint32_t zero_store_mip_seed[20]{};
    std::copy(std::begin(atomic_image_seed), std::end(atomic_image_seed),
              zero_store_mip_seed + 12);
    const uint32_t gta_store_mip_2d[] = {
        0x7e0a0206u,                         // v_mov_b32 v5, s6 (known zero)
        0xf024310au, 0x00030004u, 0x00000503u, // IMAGE_STORE_MIP v4,[v0,v3,v5],s[12:19]
        0xbf810000u,
    };
    std::vector<SrtUse> zero_store_mip_uses;
    resolve_dynamic_fetch(gta_store_mip_2d, std::size(gta_store_mip_2d),
                          zero_store_mip_seed, std::size(zero_store_mip_seed), 0,
                          &zero_store_mip_uses);
    CHECK(zero_store_mip_uses.size() == 1 && zero_store_mip_uses[0].is_storage_image &&
              zero_store_mip_uses[0].use_pc == 1 &&
              zero_store_mip_uses[0].proven_zero_mip,
          "GTA V IMAGE_STORE_MIP is classified as storage and carries its v5=zero proof");

    const uint32_t gta_store_mip_xyzw_2d[] = {
        0x7e0c0206u,                         // v_mov_b32 v6, s6 (known zero)
        0xf0243f0au, 0x00030005u, 0x00000604u,
        0xbf810000u,
    };
    std::vector<SrtUse> zero_store_mip_xyzw_uses;
    resolve_dynamic_fetch(gta_store_mip_xyzw_2d, std::size(gta_store_mip_xyzw_2d),
                          zero_store_mip_seed, std::size(zero_store_mip_seed), 0,
                          &zero_store_mip_xyzw_uses);
    CHECK(zero_store_mip_xyzw_uses.size() == 1 &&
              zero_store_mip_xyzw_uses[0].is_storage_image &&
              zero_store_mip_xyzw_uses[0].proven_zero_mip,
          "GTA V 0x2042f49800 dmask-xyzw IMAGE_STORE_MIP carries its exact v6 proof");

    Gen5ImageFormatInfo zero_mip_format{};
    const DecodedImageDescriptor zero_mip_descriptor =
        decode_image_descriptor(atomic_image_seed);
    CHECK(gen5_image_format(zero_mip_descriptor.format, &zero_mip_format),
          "zero-mip materialization fixture uses a mapped image format");
    const DecodedImageView zero_mip_view =
        image_base_level_view(zero_mip_descriptor, zero_mip_format);
    const SrtUse materialized_zero_mip_use = zero_mip_load_uses.empty()
        ? SrtUse{} : zero_mip_load_uses[0];
    CHECK(shader_resource_allows_zero_mip_specialization(
              materialized_zero_mip_use, zero_mip_descriptor, zero_mip_view),
          "single-level uncompressed T# admits the proven IMAGE_LOAD_MIP use at materialization");
    // Audit the second live 64x64 mip0:0 descriptor rather than assuming that a tail-sized surface
    // is a packed-tail view. MAX_MIP=0 means there are no sibling levels sharing the first 4 KiB
    // block, so the layout correctly treats level zero as a standalone tiled surface. This does not
    // weaken the real packed-tail gate: the synthetic packed-view arm below remains rejected.
    const uint32_t live_tail_zero_mip_t8[8] = {
        0x1095c880u, 0xc3c00000u, 0x000fc00fu, 0x90500f2eu,
        0x00000000u, 0x00700000u, 0x00000000u, 0x00000000u,
    };
    const DecodedImageDescriptor live_tail_zero_mip_descriptor =
        decode_image_descriptor(live_tail_zero_mip_t8);
    Gen5ImageFormatInfo live_tail_zero_mip_format{};
    CHECK(gen5_image_format(live_tail_zero_mip_descriptor.format,
                            &live_tail_zero_mip_format),
          "live 64x64 zero-mip tail descriptor uses a mapped image format");
    const DecodedImageView live_tail_zero_mip_view = image_base_level_view(
        live_tail_zero_mip_descriptor, live_tail_zero_mip_format);
    CHECK(live_tail_zero_mip_descriptor.width == 64u &&
              live_tail_zero_mip_descriptor.height == 64u &&
              live_tail_zero_mip_descriptor.base_level == 0u &&
              live_tail_zero_mip_descriptor.last_level == 0u &&
              live_tail_zero_mip_descriptor.max_mip == 0u,
          "live 64x64 zero-mip descriptor declares exactly one base level");
    CHECK(!live_tail_zero_mip_view.in_mip_tail &&
              shader_resource_allows_zero_mip_specialization(
                  materialized_zero_mip_use, live_tail_zero_mip_descriptor,
                  live_tail_zero_mip_view),
          "live 64x64 single-level surface is standalone, not a packed-tail relaxation");
    DecodedImageView packed_tail_zero_mip_view = live_tail_zero_mip_view;
    packed_tail_zero_mip_view.in_mip_tail = true;
    CHECK(!shader_resource_allows_zero_mip_specialization(
                  materialized_zero_mip_use, live_tail_zero_mip_descriptor,
                  packed_tail_zero_mip_view),
          "a true packed-tail zero-mip view remains fail-visible without a tail proof");
    uint32_t dcc_zero_mip_t8[8]{};
    std::copy(std::begin(atomic_image_seed), std::end(atomic_image_seed), dcc_zero_mip_t8);
    dcc_zero_mip_t8[6] = 0x00280000u;       // live 0x2042f49a DCC/pipe-aligned control shape
    dcc_zero_mip_t8[7] = 0x20553100u;       // nonzero metadata address payload
    const DecodedImageDescriptor dcc_zero_mip_descriptor =
        decode_image_descriptor(dcc_zero_mip_t8);
    CHECK(dcc_zero_mip_descriptor.compression_enabled &&
              !shader_resource_allows_zero_mip_specialization(
                  materialized_zero_mip_use, dcc_zero_mip_descriptor, zero_mip_view),
          "GTA V's DCC-backed IMAGE_LOAD_MIP remains fail-visible at materialization");

    // The same live Astro visibility kernel assembles a sampled T# in s[44:51] from four
    // consecutive entry-user-data pairs before image_load. Preserve both lanes of s_mov_b64 and
    // accept the reassembled descriptor only while all eight entry origins remain consecutive.
    uint32_t moved_image_seed[14] = {};
    std::copy(std::begin(atomic_image_seed), std::end(atomic_image_seed),
              moved_image_seed + 2);
    const uint32_t moved_direct_image[] = {
        0xbeac0402u, 0xbeae0404u, 0xbeb00406u, 0xbeb20408u,
        0xf0000108u, 0x000b0707u, // image_load v7, [v7,v8], s[44:51] dmask:x 2D
        0xbf810000u,
    };
    std::vector<SrtUse> moved_image_uses;
    resolve_dynamic_fetch(moved_direct_image, std::size(moved_direct_image),
                          moved_image_seed, std::size(moved_image_seed), 0,
                          &moved_image_uses);
    CHECK(moved_image_uses.size() == 1 && moved_image_uses[0].kind == 0 &&
              moved_image_uses[0].use_pc == 4 &&
              moved_image_uses[0].t8 == expected_atomic_t8,
          "Astro moved consecutive entry pairs publish the image_load T# by exact pc");

    // Keep the assembled bytes identical and plausible, but source the high half from a disjoint
    // second copy. Shape checks alone would accept this; non-consecutive origins must not fabricate
    // moved-descriptor provenance.
    std::copy(std::begin(atomic_image_seed) + 4, std::end(atomic_image_seed),
              moved_image_seed + 10);
    const uint32_t moved_gapped_image[] = {
        0xbeac0402u, 0xbeae0404u, 0xbeb0040au, 0xbeb2040cu,
        0xf0000108u, 0x000b0707u,
        0xbf810000u,
    };
    std::vector<SrtUse> moved_gapped_image_uses;
    resolve_dynamic_fetch(moved_gapped_image, std::size(moved_gapped_image),
                          moved_image_seed, std::size(moved_image_seed), 0,
                          &moved_gapped_image_uses);
    CHECK(moved_gapped_image_uses.empty(),
          "gapped entry pairs do not fabricate moved image-descriptor provenance");

    // Astro Bot's world-map compaction kernel keeps an x8 descriptor-table pointer in s[14:15],
    // while s_abs_i32 writes the adjacent s13 immediately before the load. S_ABS has a 32-bit
    // destination: the scalar fold must not erase s14 as though this were an unknown pair write,
    // or the second image_store's exact descriptor disappears.
    alignas(16) uint32_t abs_adjacent_image[8];
    std::copy(std::begin(atomic_image_seed), std::end(atomic_image_seed),
              std::begin(abs_adjacent_image));
    const uint64_t abs_adjacent_image_addr =
        reinterpret_cast<uint64_t>(abs_adjacent_image);
    uint32_t abs_adjacent_seed[25]{};
    abs_adjacent_seed[14] = static_cast<uint32_t>(abs_adjacent_image_addr);
    abs_adjacent_seed[15] = static_cast<uint32_t>(abs_adjacent_image_addr >> 32);
    abs_adjacent_seed[24] = 17u;
    const uint32_t abs_adjacent_load[] = {
        0xbe8d3418u,                // s_abs_i32 s13, s24 (must leave s14 untouched)
        0xf40c0907u, 0xfa000000u,   // s_load_dwordx8 s[36:43], s[14:15], 0
        0xf0200108u, 0x00090600u,   // image_store v6, v0, s[36:43] dmask:x 2D
        0xbf810000u,
    };
    std::vector<SrtUse> abs_adjacent_uses;
    resolve_dynamic_fetch(abs_adjacent_load, std::size(abs_adjacent_load),
                          abs_adjacent_seed, std::size(abs_adjacent_seed), 0,
                          &abs_adjacent_uses);
    CHECK(abs_adjacent_uses.size() == 1 && abs_adjacent_uses[0].kind == 0 &&
              abs_adjacent_uses[0].use_pc == 3 &&
              abs_adjacent_uses[0].is_storage_image &&
              abs_adjacent_uses[0].t8 == expected_atomic_t8 &&
              abs_adjacent_uses[0].descriptor_source_addr == abs_adjacent_image_addr,
          "Astro s_abs_i32 preserves the adjacent x8 image and its exact table source");

    // Mutate the exact provenance path after the same x8 load: duplicate source lane zero into
    // lane one. The live T# remains fully known and all words descend from the same load window,
    // but it no longer equals the contiguous 32 bytes at that address. A window-only origin would
    // make the post-submit probe report a false change even when memory stayed untouched.
    const uint32_t reordered_image_load[] = {
        0xf40c0907u, 0xfa000000u,   // s_load_dwordx8 s[36:43], s[14:15], 0
        0xbea50324u,                // s_mov_b32 s37, s36 (duplicate/reorder one source lane)
        0xf0200108u, 0x00090600u,   // image_store v6, v0, s[36:43] dmask:x 2D
        0xbf810000u,
    };
    std::vector<SrtUse> reordered_image_uses;
    resolve_dynamic_fetch(reordered_image_load, std::size(reordered_image_load),
                          abs_adjacent_seed, std::size(abs_adjacent_seed), 0,
                          &reordered_image_uses);
    CHECK(reordered_image_uses.size() == 1 && reordered_image_uses[0].kind == 0 &&
              reordered_image_uses[0].use_pc == 3 &&
              reordered_image_uses[0].t8[1] == reordered_image_uses[0].t8[0] &&
              reordered_image_uses[0].descriptor_source_addr == 0,
          "same-window lane reorder drops contiguous image-source provenance");

    // Astro's title PS consumes a V# placed directly in s[24:27] with a scalar offset computed in
    // VCC_LO. No s_load gives that descriptor an SRT key, and AGC metadata need not publish a sharp
    // for it. The fold nevertheless knows all four entry dwords and the exact offset, so retain the
    // resource by the consuming SMEM pc. The production resource path must then give the fragment
    // recompiler a real binding instead of its unbound binding-2 fallback.
    static uint32_t direct_sbuffer_payload[64]{};
    const uint64_t direct_sbuffer_base =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(direct_sbuffer_payload));
    uint32_t direct_sbuffer_seed[28]{};
    direct_sbuffer_seed[24] = static_cast<uint32_t>(direct_sbuffer_base);
    direct_sbuffer_seed[25] =
        (static_cast<uint32_t>(direct_sbuffer_base >> 32) & 0xffffu) | (4u << 16);
    direct_sbuffer_seed[26] = std::size(direct_sbuffer_payload);
    direct_sbuffer_seed[27] = 4u << 12;
    const uint32_t direct_sbuffer[] = {
        0xbeea03ffu, 0x00000010u,   // s_mov_b32 vcc_lo, 16
        0xf424000cu, 0xd4000004u,   // s_buffer_load_dwordx2 s[0:1], s[24:27], vcc_lo offset:4
        0xbf810000u,
    };
    std::vector<SrtUse> direct_sbuffer_uses;
    resolve_dynamic_fetch(direct_sbuffer, std::size(direct_sbuffer),
                          direct_sbuffer_seed, std::size(direct_sbuffer_seed), 0,
                          &direct_sbuffer_uses);
    CHECK(direct_sbuffer_uses.size() == 1 && direct_sbuffer_uses[0].kind == 1 &&
              direct_sbuffer_uses[0].key == 0xffffffffu &&
              direct_sbuffer_uses[0].use_pc == 2 &&
              direct_sbuffer_uses[0].required_size == 28 &&
              direct_sbuffer_uses[0].v4[0] == direct_sbuffer_seed[24] &&
              direct_sbuffer_uses[0].v4[1] == direct_sbuffer_seed[25],
          "#1029: direct user-SGPR V# resolves scalar buffer load by exact pc");
    ShaderResourceTable direct_sbuffer_table;
    add_compute_buffer_resources(direct_sbuffer_table, direct_sbuffer,
                                 std::size(direct_sbuffer), direct_sbuffer_seed,
                                 std::size(direct_sbuffer_seed));
    assign_convention_bindings(direct_sbuffer_table, 2);
    ComputeShaderConfig direct_sbuffer_config;
    direct_sbuffer_config.user_sgprs.assign(
        direct_sbuffer_seed, direct_sbuffer_seed + std::size(direct_sbuffer_seed));
    direct_sbuffer_config.local_x = direct_sbuffer_config.local_y =
        direct_sbuffer_config.local_z = 1;
    CHECK(direct_sbuffer_table.resources.size() == 1 &&
              direct_sbuffer_table.by_fetch_pc(2) &&
              direct_sbuffer_table.by_fetch_pc(2)->binding == 2 &&
              !recompile_compute(direct_sbuffer, std::size(direct_sbuffer),
                                 &direct_sbuffer_table, direct_sbuffer_config).empty(),
          "#1029: pc-keyed direct scalar buffer is bound and recompiles");

    // GTA V selects one of two adjacent descriptor fragments with the low bit of a raw table
    // pointer. The high-half selector is `s_andn2_b32 1, s0`: without folding AND-NOT, VCC_HI and
    // the register-SOFFSET x2 load become unknown, so the exact buffer_store at pc10 cannot publish
    // its V#. Exercise both pointer parities because reversing AND-NOT's operands produces a very
    // different (and unsound) table offset while still looking superficially like a bit clear.
    alignas(16) uint32_t andn2_fragment_payload[16]{};
    const uint64_t andn2_payload_base =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(andn2_fragment_payload));
    alignas(16) uint32_t andn2_fragment_table[10] = {
        0u, 0u, 0u, 0u, 0u, 0u,
        static_cast<uint32_t>(andn2_payload_base),
        (static_cast<uint32_t>(andn2_payload_base >> 32) & 0xffffu) | (4u << 16),
        static_cast<uint32_t>(andn2_payload_base),
        (static_cast<uint32_t>(andn2_payload_base >> 32) & 0xffffu) | (4u << 16),
    };
    const uint32_t gta_andn2_fragment[] = {
        0x8a6b0081u,              // pc0: s_andn2_b32 s107, 1, s0
        0x8f6b836bu,              // pc1: s_lshl_b32 s107, s107, 3
        0x871600c2u,              // pc2: s_and_b32 s22, -2, s0
        0xbe970301u,              // pc3: s_mov_b32 s23, s1
        0xf404020bu, 0xd6000018u, // pc4: s_load_dwordx2 s[8:9], s[22:23], s107
        0xbe8a03ffu, 16u,         // pc6: s_mov_b32 s10, 16 records
        0xbe8b03ffu, 4u << 12,    // pc8: s_mov_b32 s11, raw R32 descriptor control
        0xe0702000u, 0x80020103u, // pc10: buffer_store_dword v1, v3, s[8:11]
        0xbf810000u,
    };
    for (uint32_t parity = 0; parity < 2; ++parity) {
        const uint64_t tagged_table =
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(andn2_fragment_table)) | parity;
        uint32_t andn2_seed[2] = {
            static_cast<uint32_t>(tagged_table),
            static_cast<uint32_t>(tagged_table >> 32),
        };
        ShaderResourceTable andn2_table;
        const std::vector<SrtUse> andn2_uses = add_compute_buffer_resources(
            andn2_table, gta_andn2_fragment, std::size(gta_andn2_fragment),
            andn2_seed, std::size(andn2_seed));
        assign_convention_bindings(andn2_table, 2);
        ComputeShaderConfig andn2_config;
        andn2_config.user_sgprs.assign(andn2_seed, andn2_seed + std::size(andn2_seed));
        andn2_config.local_x = andn2_config.local_y = andn2_config.local_z = 1;
        CHECK(andn2_uses.size() == 1 && andn2_uses[0].use_pc == 10 &&
                  andn2_uses[0].v4[0] == static_cast<uint32_t>(andn2_payload_base) &&
                  andn2_uses[0].v4[1] == andn2_fragment_table[7] &&
                  andn2_table.resources.size() == 1 && andn2_table.by_fetch_pc(10) &&
                  !recompile_compute(gta_andn2_fragment, std::size(gta_andn2_fragment),
                                     &andn2_table, andn2_config).empty(),
              "GTA V s_andn2 table selector resolves the exact pc10 descriptor fragment");

        std::vector<uint32_t> reversed_andn2(
            std::begin(gta_andn2_fragment), std::end(gta_andn2_fragment));
        reversed_andn2[0] = 0x8a6b8100u; // pc0: s_andn2_b32 s107, s0, 1
        ShaderResourceTable reversed_andn2_table;
        const std::vector<SrtUse> reversed_andn2_uses = add_compute_buffer_resources(
            reversed_andn2_table, reversed_andn2.data(), reversed_andn2.size(),
            andn2_seed, std::size(andn2_seed));
        CHECK(reversed_andn2_uses.empty() && reversed_andn2_table.resources.empty(),
              "same-site reversed s_andn2 operands do not inherit GTA V's fragment proof");
    }

    // Astro's world-map PS derives VCC_LO from v_readfirstlane. That value is uniform at runtime
    // and the SPIR-V translator tracks it, but the CPU descriptor fold intentionally cannot assign
    // a concrete value to a VGPR read. Retain the exact bounded V# with required_size=0: production
    // uploads the descriptor's complete declared range and the translated SMEM performs the dynamic
    // index. Never substituting offset zero is the important distinction from an unsafe const-fold.
    const uint32_t wave_offset_sbuffer[] = {
        0x7ed40500u,              // v_readfirstlane_b32 vcc_lo, v0
        0xf420000cu, 0xd4000000u, // s_buffer_load_dword s0, s[24:27], vcc_lo
        0xbf810000u,
    };
    std::vector<SrtUse> wave_offset_sbuffer_uses;
    resolve_dynamic_fetch(wave_offset_sbuffer, std::size(wave_offset_sbuffer),
                          direct_sbuffer_seed, std::size(direct_sbuffer_seed), 0,
                          &wave_offset_sbuffer_uses);
    CHECK(wave_offset_sbuffer_uses.size() == 1 &&
              wave_offset_sbuffer_uses[0].kind == 1 &&
              wave_offset_sbuffer_uses[0].key == 0xffffffffu &&
              wave_offset_sbuffer_uses[0].use_pc == 1 &&
              wave_offset_sbuffer_uses[0].required_size == 0 &&
              wave_offset_sbuffer_uses[0].v4[0] == direct_sbuffer_seed[24] &&
              wave_offset_sbuffer_uses[0].v4[1] == direct_sbuffer_seed[25],
          "Astro wave-derived SOFFSET retains the full bounded direct V# by exact pc");
    ShaderResourceTable wave_offset_sbuffer_table;
    add_compute_buffer_resources(wave_offset_sbuffer_table, wave_offset_sbuffer,
                                 std::size(wave_offset_sbuffer), direct_sbuffer_seed,
                                 std::size(direct_sbuffer_seed));
    assign_convention_bindings(wave_offset_sbuffer_table, 2);
    CHECK(wave_offset_sbuffer_table.resources.size() == 1 &&
              wave_offset_sbuffer_table.by_fetch_pc(1) &&
              wave_offset_sbuffer_table.by_fetch_pc(1)->size ==
                  sizeof(direct_sbuffer_payload) &&
              !recompile_compute(wave_offset_sbuffer, std::size(wave_offset_sbuffer),
                                 &wave_offset_sbuffer_table, direct_sbuffer_config).empty(),
          "Astro wave-derived SOFFSET binds its descriptor range and recompiles dynamically");

    // GTA V's 0x413ce6000/0x413ce6d00 programs load a four-dword V# through an S_BUFFER_LOAD whose
    // VCC-derived SOFFSET is intentionally unknown to this CPU fold. The outer V# is fully known and
    // has NUM_RECORDS=0, while the positive immediate begins beyond its effective scalar bound: publish
    // a marker for the scalar load itself, propagate four exact zero dwords, and let the exact raw-MUBUF
    // consumer publish its own marker.
    // These are the live scalar-load and first consumer packets (relocated to pc1/pc3 in the fixture).
    const uint32_t zero_record_sbuffer_chain[] = {
        0x7ed40500u,              // v_readfirstlane_b32 vcc_lo, v0 (runtime-uniform, fold-unknown)
        0xf4280202u, 0xd4000008u, // s_buffer_load_dwordx4 s[8:11], s[4:7], vcc_lo offset:8
        0xe0382000u, 0x80020006u, // buffer_load_dwordx4 v[0:3], v6, s[8:11], 0
        0xbf810000u,
    };
    uint32_t zero_record_sbuffer_seed[8]{};
    std::vector<SrtUse> zero_record_sbuffer_uses;
    resolve_dynamic_fetch(zero_record_sbuffer_chain, std::size(zero_record_sbuffer_chain),
                          zero_record_sbuffer_seed, std::size(zero_record_sbuffer_seed), 0,
                          &zero_record_sbuffer_uses);
    CHECK(zero_record_sbuffer_uses.size() == 2 &&
              zero_record_sbuffer_uses[0].use_pc == 1 &&
              zero_record_sbuffer_uses[0].zero_record_raw &&
              zero_record_sbuffer_uses[1].use_pc == 3 &&
              zero_record_sbuffer_uses[1].zero_record_raw &&
              std::all_of(zero_record_sbuffer_uses[1].v4.begin(),
                          zero_record_sbuffer_uses[1].v4.end(),
                          [](uint32_t word) { return word == 0u; }),
          "zero-record dynamic-SOFFSET S_BUFFER_LOAD produces an exact zero V# consumer");
    ShaderResourceTable zero_record_sbuffer_table;
    add_compute_buffer_resources(zero_record_sbuffer_table, zero_record_sbuffer_chain,
                                 std::size(zero_record_sbuffer_chain), zero_record_sbuffer_seed,
                                 std::size(zero_record_sbuffer_seed));
    assign_convention_bindings(zero_record_sbuffer_table, 2);
    ComputeShaderConfig zero_record_sbuffer_config;
    zero_record_sbuffer_config.user_sgprs.assign(
        zero_record_sbuffer_seed,
        zero_record_sbuffer_seed + std::size(zero_record_sbuffer_seed));
    zero_record_sbuffer_config.local_x = zero_record_sbuffer_config.local_y =
        zero_record_sbuffer_config.local_z = 1;
    const std::vector<uint32_t> zero_record_sbuffer_spirv = recompile_compute(
        zero_record_sbuffer_chain, std::size(zero_record_sbuffer_chain),
        &zero_record_sbuffer_table, zero_record_sbuffer_config);
    const DescriptorValidationReport zero_record_sbuffer_report =
        validate_spirv_descriptor_interface(zero_record_sbuffer_spirv,
                                            &zero_record_sbuffer_table, 0,
                                            SpirvShaderStage::Compute, false);
    CHECK(zero_record_sbuffer_table.resources.size() == 2 &&
              is_zero_record_raw_buffer(zero_record_sbuffer_table.resources[0]) &&
              is_zero_record_raw_buffer(zero_record_sbuffer_table.resources[1]) &&
              zero_record_sbuffer_table.by_fetch_pc(1) &&
              zero_record_sbuffer_table.by_fetch_pc(3) &&
              !zero_record_sbuffer_spirv.empty() && zero_record_sbuffer_report.ok() &&
              zero_record_sbuffer_report.descriptors.empty(),
          "production zero-record scalar-load chain recompiles without backing buffers");

    // The sibling 0x413ce6d00 site uses the same shape at pc148 with different registers. Preserve
    // its exact scalar packet and first raw consumer too, so register-specific decode drift cannot
    // leave one program behind while the other fixture stays green.
    const uint32_t zero_record_sbuffer_chain_6d[] = {
        0x7ed40500u,
        0xf4280508u, 0xd4000008u, // s_buffer_load_dwordx4 s[20:23], s[16:19], vcc_lo offset:8
        0xe0382000u, 0x80050006u, // buffer_load_dwordx4 v[0:3], v6, s[20:23], 0
        0xbf810000u,
    };
    uint32_t zero_record_sbuffer_seed_6d[20]{};
    std::vector<SrtUse> zero_record_sbuffer_uses_6d;
    resolve_dynamic_fetch(zero_record_sbuffer_chain_6d,
                          std::size(zero_record_sbuffer_chain_6d),
                          zero_record_sbuffer_seed_6d,
                          std::size(zero_record_sbuffer_seed_6d), 0,
                          &zero_record_sbuffer_uses_6d);
    CHECK(zero_record_sbuffer_uses_6d.size() == 2 &&
              zero_record_sbuffer_uses_6d[0].use_pc == 1 &&
              zero_record_sbuffer_uses_6d[0].zero_record_raw &&
              zero_record_sbuffer_uses_6d[1].use_pc == 3 &&
              zero_record_sbuffer_uses_6d[1].zero_record_raw,
          "second GTA V zero-record scalar packet reaches its exact raw consumer");

    // 0x413d59600 consumes the same empty V# through x4, x1 and x16 scalar loads. These are its
    // exact pc16/pc20/pc22 packets, relocated together after one fold-unknown VCC producer.
    const uint32_t zero_record_sbuffer_widths[] = {
        0x7ed40500u,
        0xf4280b06u, 0xd4000004u, // x4  s[44:47], s[12:15], vcc_lo offset:4
        0xf42000c6u, 0xd4000014u, // x1  s3,       s[12:15], vcc_lo offset:20
        0xf4300706u, 0xd4000038u, // x16 s[28:43], s[12:15], vcc_lo offset:56
        0xbf810000u,
    };
    uint32_t zero_record_sbuffer_width_seed[16]{};
    ShaderResourceTable zero_record_sbuffer_width_table;
    const std::vector<SrtUse> zero_record_sbuffer_width_uses = add_compute_buffer_resources(
        zero_record_sbuffer_width_table, zero_record_sbuffer_widths,
        std::size(zero_record_sbuffer_widths), zero_record_sbuffer_width_seed,
        std::size(zero_record_sbuffer_width_seed));
    assign_convention_bindings(zero_record_sbuffer_width_table, 2);
    ComputeShaderConfig zero_record_sbuffer_width_config;
    zero_record_sbuffer_width_config.user_sgprs.assign(
        zero_record_sbuffer_width_seed,
        zero_record_sbuffer_width_seed + std::size(zero_record_sbuffer_width_seed));
    zero_record_sbuffer_width_config.local_x = zero_record_sbuffer_width_config.local_y =
        zero_record_sbuffer_width_config.local_z = 1;
    const std::vector<uint32_t> zero_record_sbuffer_width_spirv = recompile_compute(
        zero_record_sbuffer_widths, std::size(zero_record_sbuffer_widths),
        &zero_record_sbuffer_width_table, zero_record_sbuffer_width_config);
    const DescriptorValidationReport zero_record_sbuffer_width_report =
        validate_spirv_descriptor_interface(zero_record_sbuffer_width_spirv,
                                            &zero_record_sbuffer_width_table, 0,
                                            SpirvShaderStage::Compute, false);
    CHECK(zero_record_sbuffer_width_uses.size() == 3 &&
              zero_record_sbuffer_width_table.resources.size() == 3 &&
              std::all_of(zero_record_sbuffer_width_uses.begin(),
                          zero_record_sbuffer_width_uses.end(),
                          [](const SrtUse& use) { return use.zero_record_raw; }) &&
              !zero_record_sbuffer_width_spirv.empty() &&
              zero_record_sbuffer_width_report.ok() &&
              zero_record_sbuffer_width_report.descriptors.empty(),
          "GTA V zero-record x1/x4/x16 scalar loads compile to exact zeros without bindings");

    // NUM_RECORDS is the production-site boundary. A null-base descriptor with one record does not
    // make a runtime SOFFSET knowable and must not acquire either zero marker.
    uint32_t one_record_sbuffer_seed[8]{};
    one_record_sbuffer_seed[6] = 1u;
    ShaderResourceTable one_record_sbuffer_table;
    const std::vector<SrtUse> one_record_sbuffer_uses = add_compute_buffer_resources(
        one_record_sbuffer_table, zero_record_sbuffer_chain,
        std::size(zero_record_sbuffer_chain), one_record_sbuffer_seed,
        std::size(one_record_sbuffer_seed));
    ComputeShaderConfig one_record_sbuffer_config = zero_record_sbuffer_config;
    one_record_sbuffer_config.user_sgprs.assign(
        one_record_sbuffer_seed, one_record_sbuffer_seed + std::size(one_record_sbuffer_seed));
    CHECK(one_record_sbuffer_table.resources.empty() &&
              std::none_of(one_record_sbuffer_uses.begin(), one_record_sbuffer_uses.end(),
                           [](const SrtUse& use) { return use.zero_record_raw; }) &&
              recompile_compute(zero_record_sbuffer_chain,
                                std::size(zero_record_sbuffer_chain),
                                &one_record_sbuffer_table,
                                one_record_sbuffer_config).empty(),
          "base-zero one-record scalar-load chain stays unresolved");

    // NUM_RECORDS=0 does not by itself make a stride-zero scalar descriptor empty: scalar SMEM uses
    // an effective one-dword M_SIZE. With immediate zero the first dword remains in range, so the
    // unknown SOFFSET prevents folding and neither the scalar load nor its consumer gets a marker.
    const uint32_t zero_record_sbuffer_at_zero[] = {
        0x7ed40500u,
        0xf4280202u, 0xd4000000u,
        0xe0382000u, 0x80020006u,
        0xbf810000u,
    };
    ShaderResourceTable zero_record_sbuffer_at_zero_table;
    const std::vector<SrtUse> zero_record_sbuffer_at_zero_uses = add_compute_buffer_resources(
        zero_record_sbuffer_at_zero_table, zero_record_sbuffer_at_zero,
        std::size(zero_record_sbuffer_at_zero), zero_record_sbuffer_seed,
        std::size(zero_record_sbuffer_seed));
    CHECK(zero_record_sbuffer_at_zero_table.resources.empty() &&
              std::none_of(zero_record_sbuffer_at_zero_uses.begin(),
                           zero_record_sbuffer_at_zero_uses.end(),
                           [](const SrtUse& use) { return use.zero_record_raw; }) &&
              recompile_compute(zero_record_sbuffer_at_zero,
                                std::size(zero_record_sbuffer_at_zero),
                                &zero_record_sbuffer_at_zero_table,
                                zero_record_sbuffer_config).empty(),
          "stride-zero zero-record scalar load at immediate zero stays unresolved");

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

    // GTA V publishes fully-known RAW V#s whose NUM_RECORDS is zero at gameplay entry. These are
    // architectural zero-read/drop-write resources, not scalar raw pointers: retain one explicit
    // exact-PC marker for each consumer so the recompiler can lower both sides without a backing
    // buffer. The captured descriptor shape retains stride/format metadata despite its empty range.
    const uint32_t zero_record_raw[] = {
        0xE0300000u, 0x80000000u, // buffer_load_dword v0, off, s[0:3]
        0xE0700000u, 0x80000100u, // buffer_store_dword v1, off, s[0:3]
        0xBF810000u,
    };
    const uint32_t zero_record_seed[4] = {
        0x00000000u, 0x00100000u, 0x00000000u, 0x00016204u,
    };
    std::vector<SrtUse> zero_record_uses;
    resolve_dynamic_fetch(zero_record_raw, std::size(zero_record_raw), zero_record_seed,
                          std::size(zero_record_seed), 0, &zero_record_uses);
    CHECK(zero_record_uses.size() == 2 && zero_record_uses[0].zero_record_raw &&
              zero_record_uses[1].zero_record_raw && zero_record_uses[0].use_pc == 0 &&
              zero_record_uses[1].use_pc == 2 &&
              decode_buffer_descriptor(zero_record_uses[0].v4.data()).num_records == 0,
          "fully-known zero-record RAW loads/stores retain exact-PC zero semantics");
    ShaderResourceTable zero_record_table;
    const std::vector<SrtUse> materialized_zero_uses = add_compute_buffer_resources(
        zero_record_table, zero_record_raw, std::size(zero_record_raw), zero_record_seed,
        std::size(zero_record_seed));
    assign_convention_bindings(zero_record_table, 2);
    CHECK(materialized_zero_uses.size() == 2 && zero_record_table.resources.size() == 2 &&
              is_zero_record_raw_buffer(zero_record_table.resources[0]) &&
              is_zero_record_raw_buffer(zero_record_table.resources[1]) &&
              zero_record_table.by_fetch_pc(0) && zero_record_table.by_fetch_pc(2) &&
              zero_record_table.resources[0].binding == 2 &&
              zero_record_table.resources[1].binding == 3,
          "production compute materialization creates distinct zero-record RAW markers");
    ComputeShaderConfig zero_record_config;
    zero_record_config.user_sgprs.assign(zero_record_seed,
                                         zero_record_seed + std::size(zero_record_seed));
    zero_record_config.local_x = zero_record_config.local_y = zero_record_config.local_z = 1;
    CHECK(!recompile_compute(zero_record_raw, std::size(zero_record_raw), &zero_record_table,
                             zero_record_config).empty(),
          "production zero-record resource table reaches compute translation");

    // MUBUF TFE is a different architectural packet even when its data result is one dword: it
    // appends a status VGPR. Exercise the production emitter gate itself, not only the decoder and
    // zero-record proof, so deleting that gate makes this same-packet mutation arm fail.
    const std::array<uint32_t, 3> ordinary_raw_load = {
        0xe0300000u, 0x80000000u, // pc0: buffer_load_dword v0, off, s[0:3], 0
        0xbf810000u,
    };
    std::array<uint32_t, 3> ordinary_raw_load_tfe = ordinary_raw_load;
    ordinary_raw_load_tfe[1] |= 0x00800000u;
    ShaderResourceTable ordinary_raw_load_table;
    ShaderResource ordinary_raw_load_resource{};
    ordinary_raw_load_resource.cls = ResourceClass::ConstantBuffer;
    ordinary_raw_load_resource.format = DataFormat::Uint32;
    ordinary_raw_load_resource.num_components = 1;
    ordinary_raw_load_resource.gpu_addr = 0x10000u;
    ordinary_raw_load_resource.size = 4u;
    ordinary_raw_load_resource.fetch_pc = 0u;
    ordinary_raw_load_table.resources.push_back(ordinary_raw_load_resource);
    assign_convention_bindings(ordinary_raw_load_table, 2);
    ComputeShaderConfig ordinary_raw_load_config;
    ordinary_raw_load_config.local_x = ordinary_raw_load_config.local_y =
        ordinary_raw_load_config.local_z = 1;
    CHECK(!recompile_compute(ordinary_raw_load.data(), ordinary_raw_load.size(),
                             &ordinary_raw_load_table, ordinary_raw_load_config).empty() &&
              recompile_compute(ordinary_raw_load_tfe.data(), ordinary_raw_load_tfe.size(),
                                &ordinary_raw_load_table,
                                ordinary_raw_load_config).empty(),
          "same-packet MUBUF TFE mutation reaches and rejects at production emission");

    const uint32_t all_zero_raw_seed[4] = {};
    std::vector<SrtUse> all_zero_raw_uses;
    resolve_dynamic_fetch(zero_record_raw, std::size(zero_record_raw), all_zero_raw_seed,
                          std::size(all_zero_raw_seed), 0, &all_zero_raw_uses);
    CHECK(all_zero_raw_uses.size() == 2 && all_zero_raw_uses[0].zero_record_raw &&
              all_zero_raw_uses[1].zero_record_raw,
          "all-zero unbound RAW V# uses the same exact zero-record contract");

    // The admitted boundary is NUM_RECORDS=0, not merely Base=0. A descriptor with the same zero
    // address but one 16-byte record remains malformed and must not inherit zero semantics.
    uint32_t addr0_nonzero_record_seed[4];
    std::copy(std::begin(zero_record_seed), std::end(zero_record_seed),
              addr0_nonzero_record_seed);
    addr0_nonzero_record_seed[2] = 1u;
    std::vector<SrtUse> addr0_nonzero_record_uses;
    resolve_dynamic_fetch(zero_record_raw, std::size(zero_record_raw),
                          addr0_nonzero_record_seed, std::size(addr0_nonzero_record_seed), 0,
                          &addr0_nonzero_record_uses);
    ShaderResourceTable addr0_nonzero_record_table;
    add_compute_buffer_resources(addr0_nonzero_record_table, zero_record_raw,
                                 std::size(zero_record_raw), addr0_nonzero_record_seed,
                                 std::size(addr0_nonzero_record_seed));
    CHECK(addr0_nonzero_record_uses.empty() && addr0_nonzero_record_table.resources.empty(),
          "addr0 plus nonzero NUM_RECORDS stays rejected at the zero-record boundary");

    // GTA V's 0x413cf6100 pc10 walks a dispatch-sized optional buffer table. The mapped pointer
    // entry at user-data root +0x58 is zero before a later WRITE_DATA initializes it, while scalar
    // code independently builds a stride-4 V# with a nonzero record count. Preserve the exact
    // producer/use packets: only this mapped provenance plus the linear launch proof may return zero.
    const std::array<uint32_t, 13> optional_null_raw = {
        0xbfa00003u,                         // pc0:  s_clause 4
        0xd7460002u, 0x04010c0fu,            // pc1:  v_lshl_add_u32 v2,s15,6,v0
        0xf4040400u, 0xfa000058u,            // pc3:  s_load_dwordx2 s[16:17],s[0:1],0x58
        0x8112c102u,                         // pc5:  s_add_i32 s18,s2,-1
        0xbe9303ffu, kGtaOptionalBufferConfigWord, // pc6: s_mov_b32 s19,config
        0xbf8cc07fu,                         // pc8:  s_waitcnt
        0xbe911d92u,                         // pc9:  s_bitset1_b32 s17,18 (stride 4)
        0xe0302000u, 0x80040002u,            // pc10: buffer_load_dword v0,v2,s[16:19],0 idxen
        0xbf810000u,                         // pc12: s_endpgm
    };
    constexpr uint32_t optional_records = 127u;
    alignas(8) std::array<uint32_t, kGtaOptionalBufferTableBytes / sizeof(uint32_t)>
        optional_table{};
    std::array<uint32_t, 15> optional_user_sgprs{};
    const uint64_t optional_root = reinterpret_cast<uint64_t>(optional_table.data());
    optional_user_sgprs[0] = static_cast<uint32_t>(optional_root);
    optional_user_sgprs[1] = static_cast<uint32_t>(optional_root >> 32);
    optional_user_sgprs[2] = optional_records + 1u;

    ComputeShaderConfig optional_config;
    optional_config.user_sgprs.assign(optional_user_sgprs.begin(), optional_user_sgprs.end());
    optional_config.local_x = kGtaOptionalBufferLocalSize;
    optional_config.local_y = optional_config.local_z = 1u;
    ShaderResourceTable optional_table_resources;
    const std::vector<SrtUse> optional_uses = add_compute_buffer_resources(
        optional_table_resources, optional_null_raw.data(), optional_null_raw.size(),
        optional_user_sgprs.data(), optional_user_sgprs.size(),
        kGtaOptionalBufferLocalSize, optional_records, kGtaOptionalBufferTgidSgpr);
    assign_convention_bindings(optional_table_resources, 2);
    const std::vector<uint32_t> optional_spirv = recompile_compute(
        optional_null_raw.data(), optional_null_raw.size(), &optional_table_resources,
        optional_config);
    const DescriptorValidationReport optional_report = validate_spirv_descriptor_interface(
        optional_spirv, &optional_table_resources, 0, SpirvShaderStage::Compute, false);
    CHECK(optional_uses.size() == 1 && optional_uses[0].use_pc == 10u &&
              optional_uses[0].optional_null_raw_load && !optional_uses[0].zero_record_raw &&
              optional_table_resources.resources.size() == 1 &&
              is_optional_null_raw_load_buffer(optional_table_resources.resources[0]) &&
              !is_zero_record_raw_buffer(optional_table_resources.resources[0]) &&
              optional_table_resources.by_fetch_pc(10u) && !optional_spirv.empty() &&
              optional_report.ok() && optional_report.descriptors.empty(),
          "mapped optional +0x58 entry lowers the exact linear RAW dword load to zero");

    // Direct-CFG counter-arm for the same production chain. The conditional branch reaches the
    // existing wait/stride-patch/consumer tail while skipping count/config construction. A linear
    // fold still visits those skipped writes, so this must remain rejected until the producer roles
    // have a path-sensitive dominance proof.
    const std::array<uint32_t, 14> optional_null_raw_branch = {
        optional_null_raw[0], optional_null_raw[1], optional_null_raw[2],
        optional_null_raw[3], optional_null_raw[4],
        0xbf840003u,                         // pc5: s_cbranch_scc0 +3 -> pc9
        optional_null_raw[5],                // pc6: count (skipped on taken arm)
        optional_null_raw[6], optional_null_raw[7], // pc7: config (skipped)
        optional_null_raw[8], optional_null_raw[9],
        optional_null_raw[10], optional_null_raw[11], optional_null_raw[12],
    };
    ShaderResourceTable optional_branch_resources;
    const std::vector<SrtUse> optional_branch_uses = add_compute_buffer_resources(
        optional_branch_resources, optional_null_raw_branch.data(),
        optional_null_raw_branch.size(), optional_user_sgprs.data(),
        optional_user_sgprs.size(), kGtaOptionalBufferLocalSize, optional_records,
        kGtaOptionalBufferTgidSgpr);
    const bool optional_branch_has_marker = std::any_of(
        optional_branch_resources.resources.begin(),
        optional_branch_resources.resources.end(),
        [](const ShaderResource& resource) {
            return is_optional_null_raw_load_buffer(resource);
        });
    CHECK(std::none_of(optional_branch_uses.begin(), optional_branch_uses.end(),
                       [](const SrtUse& use) { return use.optional_null_raw_load; }) &&
              optional_branch_resources.resources.empty() && !optional_branch_has_marker &&
              recompile_compute(optional_null_raw_branch.data(),
                                optional_null_raw_branch.size(),
                                &optional_branch_resources, optional_config).empty(),
          "conditional branch across count/config keeps the optional chain fail-visible");

    // The sibling 0x413cf5400 terminal reaches the same table convention through an in-place
    // s_or_b32 stride patch rather than s_bitset1_b32. Keep its exact prefix through the
    // post-consumer pc46 branch: later control flow is irrelevant unless it enters the straight-line
    // producer/consumer interval, and a fixture ending at pc40 failed to exercise that distinction.
    const std::array<uint32_t, 48> optional_null_raw_or = {
        0xbfa00003u, 0xd7460016u, 0x04010c0fu, 0x8f6a9e0eu,
        0x8010ff00u, 0x000000e8u, 0xbe920382u, 0x82118001u,
        0xbe9303ffu, 0x00016204u, 0xbe88047eu, 0x7d2d00f9u,
        0x8686006au, 0xbe911d92u, 0x7da42c80u, 0xbf88000fu,
        0x7e000280u, 0xbeea070eu, 0x7e020281u, 0x7e040281u,
        0x3606d4f9u, 0x86860681u, 0x7e080280u, 0xbe860381u,
        0xbe8703ffu, 0x00016204u, 0xbe851d95u, 0xe07c0000u,
        0x80010000u, 0xe0702000u, 0x80040403u,
        0xf4040700u, 0xfa000058u,             // pc31: x2 optional pointer -> s[28:29]
        0xbefe0408u, 0xbf8cc07fu,
        0x881dff1du, kGtaOptionalBufferStrideWord, // pc35: s_or_b32 s29,s29,stride
        0x811ec102u,                         // pc37: s_add_i32 s30,s2,-1
        0xbe9f03ffu, kGtaOptionalBufferConfigWord,
        0xe0302000u, 0x80070016u,            // pc40: exact optional RAW load
        0xbe9f03ffu, kGtaOptionalBufferConfigWord, // pc42: production tail
        0xbf8c3f70u, 0x7daa0087u,
        0xbf88010fu,                         // pc46: production branch -> pc318
        0xbf810000u,
    };
    ShaderResourceTable optional_or_resources;
    const std::vector<SrtUse> optional_or_uses = add_compute_buffer_resources(
        optional_or_resources, optional_null_raw_or.data(), optional_null_raw_or.size(),
        optional_user_sgprs.data(), optional_user_sgprs.size(),
        kGtaOptionalBufferLocalSize, optional_records, kGtaOptionalBufferTgidSgpr);
    const bool optional_or_use = std::any_of(optional_or_uses.begin(), optional_or_uses.end(),
        [](const SrtUse& use) { return use.use_pc == 40u && use.optional_null_raw_load; });
    CHECK(optional_or_use && optional_or_resources.by_fetch_pc(40u) &&
              is_optional_null_raw_load_buffer(*optional_or_resources.by_fetch_pc(40u)),
          "pc40 in-place OR chain materializes the same distinct optional-null marker");

    std::array<uint32_t, 48> optional_or_wrong_stride = optional_null_raw_or;
    optional_or_wrong_stride[36] = 0x00080000u;
    ShaderResourceTable optional_or_wrong_stride_resources;
    const std::vector<SrtUse> optional_or_wrong_stride_uses = add_compute_buffer_resources(
        optional_or_wrong_stride_resources, optional_or_wrong_stride.data(),
        optional_or_wrong_stride.size(), optional_user_sgprs.data(),
        optional_user_sgprs.size(), kGtaOptionalBufferLocalSize, optional_records,
        kGtaOptionalBufferTgidSgpr);
    CHECK(std::none_of(optional_or_wrong_stride_uses.begin(),
                       optional_or_wrong_stride_uses.end(),
                       [](const SrtUse& use) {
                           return use.use_pc == 40u && use.optional_null_raw_load;
                       }) &&
              (!optional_or_wrong_stride_resources.by_fetch_pc(40u) ||
               !is_optional_null_raw_load_buffer(
                   *optional_or_wrong_stride_resources.by_fetch_pc(40u))),
          "pc35 OR-literal mutation removes the second terminal's optional-null proof");

    std::array<uint32_t, 48> optional_or_backedge_into_tail = optional_null_raw_or;
    optional_or_backedge_into_tail[46] = 0xbf88fff7u; // pc46 -> pc38, after the producer
    ShaderResourceTable optional_or_backedge_resources;
    const std::vector<SrtUse> optional_or_backedge_uses = add_compute_buffer_resources(
        optional_or_backedge_resources, optional_or_backedge_into_tail.data(),
        optional_or_backedge_into_tail.size(), optional_user_sgprs.data(),
        optional_user_sgprs.size(), kGtaOptionalBufferLocalSize, optional_records,
        kGtaOptionalBufferTgidSgpr);
    CHECK(std::none_of(optional_or_backedge_uses.begin(), optional_or_backedge_uses.end(),
                       [](const SrtUse& use) {
                           return use.use_pc == 40u && use.optional_null_raw_load;
                       }) &&
              (!optional_or_backedge_resources.by_fetch_pc(40u) ||
               !is_optional_null_raw_load_buffer(
                   *optional_or_backedge_resources.by_fetch_pc(40u))),
          "pc46 backedge into the descriptor tail keeps the optional chain fail-visible");

    // Full routed 0x413cf5400 shape. The exact pc56/pc200 resources are architectural
    // NUM_RECORDS=0 markers. Their dword loads therefore write zero to every active lane. GTA then
    // elects one of those lanes, restores that one-hot mask, ANDs the loaded zero with 7, and uses
    // CMPX_EQ 5 + EXECZ to skip the unresolved optional-buffer stores at pc194/pc314. This is a
    // reachability proof, not optional-null store support: the store packets remain unmodified and
    // must become live again when the compare at the same production site changes from EQ to NE.
    auto zero_record_marker = [](uint32_t fetch_pc) {
        ShaderResource resource{};
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Unknown;
        resource.num_components = 0;
        resource.fetch_pc = fetch_pc;
        return resource;
    };
    auto unresolved_store = [](uint32_t fetch_pc) {
        ShaderResource resource{};
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.size = 4;
        resource.fetch_pc = fetch_pc;
        return resource;
    };
    auto gta_zero_record_execz_resources = [&]() {
        ShaderResourceTable resources;
        resources.resources = {
            zero_record_marker(56), zero_record_marker(200),
            unresolved_store(194), unresolved_store(314),
        };
        return resources;
    };
    auto instruction_at = [](std::vector<Rdna2Inst>& instructions, uint32_t pc) {
        return std::find_if(instructions.begin(), instructions.end(),
                            [=](const Rdna2Inst& in) { return in.pc == pc; });
    };

    std::vector<Rdna2Inst> gta_zero_record_execz;
    rdna2_walk(prosper::test::kGta5ZeroRecordExeczProgram.data(),
               prosper::test::kGta5ZeroRecordExeczProgram.size(), gta_zero_record_execz);
    ShaderResourceTable gta_zero_record_execz_table = gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_execz_report =
        specialize_compute_resource_paths(
            gta_zero_record_execz, gta_zero_record_execz_table, 64);
    const auto pc92 = instruction_at(gta_zero_record_execz, 92);
    const auto pc212 = instruction_at(gta_zero_record_execz, 212);
    CHECK(gta_zero_record_execz_report.proven_null_exits == 0 &&
              gta_zero_record_execz_report.zero_record_execz_exits == 2 &&
              gta_zero_record_execz_report.removed_resources == 2 &&
              gta_zero_record_execz_table.by_fetch_pc(56) &&
              gta_zero_record_execz_table.by_fetch_pc(200) &&
              !gta_zero_record_execz_table.by_fetch_pc(194) &&
              !gta_zero_record_execz_table.by_fetch_pc(314) &&
              pc92 != gta_zero_record_execz.end() &&
              pc92->opcode != kSoppOpcodeCbranchExecz &&
              pc212 != gta_zero_record_execz.end() &&
              pc212->opcode != kSoppOpcodeCbranchExecz &&
              instruction_at(gta_zero_record_execz, 194) == gta_zero_record_execz.end() &&
              instruction_at(gta_zero_record_execz, 314) == gta_zero_record_execz.end(),
          "GTA zero-record elected-lane proof prunes both unreachable optional stores");

    std::array<uint32_t, 319> gta_zero_record_compare_ne =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_compare_ne[91] = 0x7daa0285u; // exact pc91 CMPX_EQ -> CMPX_NE
    std::vector<Rdna2Inst> gta_zero_record_compare_ne_ins;
    rdna2_walk(gta_zero_record_compare_ne.data(), gta_zero_record_compare_ne.size(),
               gta_zero_record_compare_ne_ins);
    ShaderResourceTable gta_zero_record_compare_ne_table = gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_compare_ne_report =
        specialize_compute_resource_paths(
            gta_zero_record_compare_ne_ins, gta_zero_record_compare_ne_table, 64);
    const auto live_pc194 = instruction_at(gta_zero_record_compare_ne_ins, 194);
    CHECK(gta_zero_record_compare_ne_report.zero_record_execz_exits == 1 &&
              gta_zero_record_compare_ne_table.by_fetch_pc(194) &&
              !gta_zero_record_compare_ne_table.by_fetch_pc(314) &&
              live_pc194 != gta_zero_record_compare_ne_ins.end() &&
              live_pc194->words[0] == 0xe0702000u &&
              live_pc194->words[1] == 0x80070201u,
          "same-site pc91 CMPX_NE mutation keeps the exact pc194 store live and fail-visible");

    std::array<uint32_t, 319> gta_zero_record_second_compare_ne =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_second_compare_ne[211] = 0x7daa0285u; // exact pc211 CMPX_EQ -> CMPX_NE
    std::vector<Rdna2Inst> gta_zero_record_second_compare_ne_ins;
    rdna2_walk(gta_zero_record_second_compare_ne.data(),
               gta_zero_record_second_compare_ne.size(),
               gta_zero_record_second_compare_ne_ins);
    ShaderResourceTable gta_zero_record_second_compare_ne_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_second_compare_ne_report =
        specialize_compute_resource_paths(
            gta_zero_record_second_compare_ne_ins,
            gta_zero_record_second_compare_ne_table, 64);
    const auto live_pc314 = instruction_at(gta_zero_record_second_compare_ne_ins, 314);
    CHECK(gta_zero_record_second_compare_ne_report.zero_record_execz_exits == 1 &&
              !gta_zero_record_second_compare_ne_table.by_fetch_pc(194) &&
              gta_zero_record_second_compare_ne_table.by_fetch_pc(314) &&
              live_pc314 != gta_zero_record_second_compare_ne_ins.end() &&
              live_pc314->words[0] == 0xe0702000u &&
              live_pc314->words[1] == 0x80070201u,
          "same-site pc211 CMPX_NE mutation keeps the exact pc314 store live and fail-visible");

    std::array<uint32_t, 319> gta_zero_record_external_entry =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_external_entry[46] = 0xbf88002cu; // pc46 EXECZ -> pc91, bypassing pc56
    std::vector<Rdna2Inst> gta_zero_record_external_entry_ins;
    rdna2_walk(gta_zero_record_external_entry.data(), gta_zero_record_external_entry.size(),
               gta_zero_record_external_entry_ins);
    ShaderResourceTable gta_zero_record_external_entry_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_external_entry_report =
        specialize_compute_resource_paths(
            gta_zero_record_external_entry_ins, gta_zero_record_external_entry_table, 64);
    CHECK(gta_zero_record_external_entry_report.zero_record_execz_exits == 1 &&
              gta_zero_record_external_entry_table.by_fetch_pc(194) &&
              !gta_zero_record_external_entry_table.by_fetch_pc(314),
          "external entry that bypasses pc56 prevents the first zero-reaching proof");

    std::array<uint32_t, 319> gta_zero_record_expanding_restore =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_expanding_restore[67] = 0xbefe04c1u; // pc67: s_mov_b64 exec, -1
    std::vector<Rdna2Inst> gta_zero_record_expanding_restore_ins;
    rdna2_walk(gta_zero_record_expanding_restore.data(),
               gta_zero_record_expanding_restore.size(),
               gta_zero_record_expanding_restore_ins);
    ShaderResourceTable gta_zero_record_expanding_restore_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_expanding_restore_report =
        specialize_compute_resource_paths(
            gta_zero_record_expanding_restore_ins,
            gta_zero_record_expanding_restore_table, 64);
    CHECK(gta_zero_record_expanding_restore_report.zero_record_execz_exits == 1 &&
              gta_zero_record_expanding_restore_table.by_fetch_pc(194) &&
              !gta_zero_record_expanding_restore_table.by_fetch_pc(314),
          "pc67 expanding EXEC restore cannot inherit the pc56 load's active-lane proof");

    std::array<uint32_t, 319> gta_zero_record_v0_clobber =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_v0_clobber[64] = 0x7e000285u; // pc64: v_mov_b32 v0, 5
    std::vector<Rdna2Inst> gta_zero_record_v0_clobber_ins;
    rdna2_walk(gta_zero_record_v0_clobber.data(), gta_zero_record_v0_clobber.size(),
               gta_zero_record_v0_clobber_ins);
    ShaderResourceTable gta_zero_record_v0_clobber_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_v0_clobber_report =
        specialize_compute_resource_paths(
            gta_zero_record_v0_clobber_ins, gta_zero_record_v0_clobber_table, 64);
    CHECK(gta_zero_record_v0_clobber_report.zero_record_execz_exits == 1 &&
              gta_zero_record_v0_clobber_table.by_fetch_pc(194) &&
              !gta_zero_record_v0_clobber_table.by_fetch_pc(314),
          "pc64 v0 clobber prevents the first zero-reaching proof");

    ShaderResourceTable gta_zero_record_missing_marker_table;
    gta_zero_record_missing_marker_table.resources = {
        zero_record_marker(200), unresolved_store(194), unresolved_store(314),
    };
    std::vector<Rdna2Inst> gta_zero_record_missing_marker_ins;
    rdna2_walk(prosper::test::kGta5ZeroRecordExeczProgram.data(),
               prosper::test::kGta5ZeroRecordExeczProgram.size(),
               gta_zero_record_missing_marker_ins);
    const ComputeResourcePathSpecializationReport gta_zero_record_missing_marker_report =
        specialize_compute_resource_paths(
            gta_zero_record_missing_marker_ins, gta_zero_record_missing_marker_table, 64);
    CHECK(gta_zero_record_missing_marker_report.zero_record_execz_exits == 1 &&
              gta_zero_record_missing_marker_table.by_fetch_pc(194) &&
              !gta_zero_record_missing_marker_table.by_fetch_pc(314),
          "missing pc56 zero-record marker keeps the first store live and fail-visible");

    std::array<uint32_t, 319> gta_zero_record_tfe_load =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_tfe_load[57] |= 0x00800000u; // pc56: append a TFE status destination
    std::vector<Rdna2Inst> gta_zero_record_tfe_load_ins;
    rdna2_walk(gta_zero_record_tfe_load.data(), gta_zero_record_tfe_load.size(),
               gta_zero_record_tfe_load_ins);
    ShaderResourceTable gta_zero_record_tfe_load_table = gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_tfe_load_report =
        specialize_compute_resource_paths(
            gta_zero_record_tfe_load_ins, gta_zero_record_tfe_load_table, 64);
    CHECK(gta_zero_record_tfe_load_report.zero_record_execz_exits == 1 &&
              gta_zero_record_tfe_load_table.by_fetch_pc(194) &&
              !gta_zero_record_tfe_load_table.by_fetch_pc(314),
          "same-site pc56 TFE mutation keeps the first store live and fail-visible");

    std::array<uint32_t, 319> gta_zero_record_invalid_srsrc =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_invalid_srsrc[57] = 0x801f0016u; // pc56: invalid V# root s[124:127]
    std::vector<Rdna2Inst> gta_zero_record_invalid_srsrc_ins;
    rdna2_walk(gta_zero_record_invalid_srsrc.data(),
               gta_zero_record_invalid_srsrc.size(),
               gta_zero_record_invalid_srsrc_ins);
    ShaderResourceTable gta_zero_record_invalid_srsrc_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_invalid_srsrc_report =
        specialize_compute_resource_paths(
            gta_zero_record_invalid_srsrc_ins,
            gta_zero_record_invalid_srsrc_table, 64);
    CHECK(gta_zero_record_invalid_srsrc_report.zero_record_execz_exits == 1 &&
              gta_zero_record_invalid_srsrc_table.by_fetch_pc(194) &&
              !gta_zero_record_invalid_srsrc_table.by_fetch_pc(314),
          "same-site pc56 invalid SRSRC quad cannot authorize the first exit");

    std::array<uint32_t, 319> gta_zero_record_exec_hi_restore =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_exec_hi_restore[67] = 0xbeff04c1u; // pc67: invalid B64 dst EXEC_HI, -1
    std::vector<Rdna2Inst> gta_zero_record_exec_hi_restore_ins;
    rdna2_walk(gta_zero_record_exec_hi_restore.data(),
               gta_zero_record_exec_hi_restore.size(),
               gta_zero_record_exec_hi_restore_ins);
    ShaderResourceTable gta_zero_record_exec_hi_restore_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_exec_hi_restore_report =
        specialize_compute_resource_paths(
            gta_zero_record_exec_hi_restore_ins,
            gta_zero_record_exec_hi_restore_table, 64);
    CHECK(gta_zero_record_exec_hi_restore_report.zero_record_execz_exits == 1 &&
              gta_zero_record_exec_hi_restore_table.by_fetch_pc(194) &&
              !gta_zero_record_exec_hi_restore_table.by_fetch_pc(314),
          "pc67 EXEC_HI-rooted B64 restore cannot carry a zero-reaching proof");

    std::array<uint32_t, 319> gta_zero_record_odd_saved_mask =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_odd_saved_mask[62] = 0xbe81246au; // pc62: AND_SAVEEXEC dst s[1:2]
    gta_zero_record_odd_saved_mask[67] = 0xbefe0401u; // pc67: restore from s[1:2]
    std::vector<Rdna2Inst> gta_zero_record_odd_saved_mask_ins;
    rdna2_walk(gta_zero_record_odd_saved_mask.data(),
               gta_zero_record_odd_saved_mask.size(),
               gta_zero_record_odd_saved_mask_ins);
    ShaderResourceTable gta_zero_record_odd_saved_mask_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_odd_saved_mask_report =
        specialize_compute_resource_paths(
            gta_zero_record_odd_saved_mask_ins,
            gta_zero_record_odd_saved_mask_table, 64);
    CHECK(gta_zero_record_odd_saved_mask_report.zero_record_execz_exits == 1 &&
              gta_zero_record_odd_saved_mask_table.by_fetch_pc(194) &&
              !gta_zero_record_odd_saved_mask_table.by_fetch_pc(314),
          "odd-aligned B64 saved-mask packets cannot splice EXEC subset lineage");

    std::array<uint32_t, 319> gta_zero_record_m0_null_saved_mask =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_m0_null_saved_mask[62] = 0xbefc246au; // pc62: invalid B64 dst M0:NULL
    gta_zero_record_m0_null_saved_mask[67] = 0xbefe047cu; // pc67: restore from M0:NULL
    std::vector<Rdna2Inst> gta_zero_record_m0_null_saved_mask_ins;
    rdna2_walk(gta_zero_record_m0_null_saved_mask.data(),
               gta_zero_record_m0_null_saved_mask.size(),
               gta_zero_record_m0_null_saved_mask_ins);
    ShaderResourceTable gta_zero_record_m0_null_saved_mask_table =
        gta_zero_record_execz_resources();
    const ComputeResourcePathSpecializationReport gta_zero_record_m0_null_saved_mask_report =
        specialize_compute_resource_paths(
            gta_zero_record_m0_null_saved_mask_ins,
            gta_zero_record_m0_null_saved_mask_table, 64);
    CHECK(gta_zero_record_m0_null_saved_mask_report.zero_record_execz_exits == 1 &&
              gta_zero_record_m0_null_saved_mask_table.by_fetch_pc(194) &&
              !gta_zero_record_m0_null_saved_mask_table.by_fetch_pc(314),
          "M0:NULL B64 pseudo-pair cannot carry EXEC subset lineage");

    ShaderResourceTable gta_zero_record_shadowed_marker_table;
    gta_zero_record_shadowed_marker_table.resources = {
        unresolved_store(56), zero_record_marker(56), zero_record_marker(200),
        unresolved_store(194), unresolved_store(314),
    };
    std::vector<Rdna2Inst> gta_zero_record_shadowed_marker_ins;
    rdna2_walk(prosper::test::kGta5ZeroRecordExeczProgram.data(),
               prosper::test::kGta5ZeroRecordExeczProgram.size(),
               gta_zero_record_shadowed_marker_ins);
    const ComputeResourcePathSpecializationReport gta_zero_record_shadowed_marker_report =
        specialize_compute_resource_paths(
            gta_zero_record_shadowed_marker_ins,
            gta_zero_record_shadowed_marker_table, 64);
    CHECK(gta_zero_record_shadowed_marker_report.zero_record_execz_exits == 1 &&
              gta_zero_record_shadowed_marker_table.by_fetch_pc(194) &&
              !gta_zero_record_shadowed_marker_table.by_fetch_pc(314),
          "a shadowed pc56 zero marker cannot override the emitter's first exact-PC resource");

    auto short_execz_mutation_keeps_store = [&](const auto& code, uint32_t store_pc) {
        std::vector<Rdna2Inst> instructions;
        rdna2_walk(code.data(), code.size(), instructions);
        ShaderResourceTable resources;
        resources.resources = {zero_record_marker(0), unresolved_store(store_pc)};
        const ComputeResourcePathSpecializationReport report =
            specialize_compute_resource_paths(instructions, resources, 64);
        return report.zero_record_execz_exits == 0 && resources.by_fetch_pc(store_pc);
    };
    const std::array<uint32_t, 9> zero_record_sdwa_and = {
        0xe0302000u, 0x80010016u,             // pc0: zero-record dword -> v0
        0x360200f9u, 0x06861587u,             // pc2: SDWA AND, WORD_1/PRESERVE
        0x7da40285u,                          // pc4: CMPX_EQ 5, v1
        0xbf880002u,                          // pc5: EXECZ -> pc8
        0xe0702000u, 0x80070201u,             // pc6: unresolved store
        0xbf810000u,                          // pc8: END
    };
    CHECK(short_execz_mutation_keeps_store(zero_record_sdwa_and, 6),
          "same-site SDWA AND mutation cannot masquerade as a full-dword zero definition");
    const std::array<uint32_t, 9> zero_record_sdwa_compare = {
        0xe0302000u, 0x80010016u,             // pc0: zero-record dword -> v0
        0x36020087u,                          // pc2: plain AND v1, 7, v0
        0x7da402f9u, 0x06810085u,             // pc3: SDWA CMPX_EQ, src0 BYTE_1
        0xbf880002u,                          // pc5: EXECZ -> pc8
        0xe0702000u, 0x80070201u,             // pc6: unresolved store
        0xbf810000u,                          // pc8: END
    };
    CHECK(short_execz_mutation_keeps_store(zero_record_sdwa_compare, 6),
          "same-site SDWA CMPX mutation cannot masquerade as the plain dword comparison");

    const std::array<uint32_t, 10> zero_record_qword_clobber = {
        0xe0302000u, 0x80010100u,             // pc0: zero-record dword -> v1
        0xe1406000u, 0x80020000u,             // pc2: GLC atomic_swap_x2 -> v[0:1]
        0x36040287u,                          // pc4: AND v2, 7, v1
        0x7da40485u,                          // pc5: CMPX_EQ 5, v2
        0xbf880002u,                          // pc6: EXECZ -> pc9
        0xe0702000u, 0x80070201u,             // pc7: unresolved store
        0xbf810000u,                          // pc9: END
    };
    CHECK(short_execz_mutation_keeps_store(zero_record_qword_clobber, 7),
          "qword atomic overlapping the zero VGPR keeps the dependent store live");

    // Exercise the raw-byte translation boundary too. After the two proven exits, only the
    // prefix's two ordinary stores, optional load, scalar data load, and the two zero-record loads
    // remain. Deliberately omit resources for pc194/pc314: either same-site compare mutation must
    // make the corresponding store visible to the production recompiler and therefore reject.
    auto ordinary_pc_buffer = [](uint32_t fetch_pc, uint32_t size, uint32_t stride) {
        ShaderResource resource{};
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.gpu_addr = 0x10000u + static_cast<uint64_t>(fetch_pc) * 0x100u;
        resource.size = size;
        resource.stride = stride;
        resource.fetch_pc = fetch_pc;
        return resource;
    };
    auto optional_null_marker = [](uint32_t fetch_pc) {
        ShaderResource resource{};
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.stride = kGtaOptionalBufferStride;
        resource.sampler_sgpr_base = kOptionalNullRawLoadMarkerSamplerBase;
        resource.fetch_pc = fetch_pc;
        return resource;
    };
    auto gta_zero_record_compile_resources = [&]() {
        ShaderResourceTable resources;
        resources.resources = {
            ordinary_pc_buffer(27, 32, 32), ordinary_pc_buffer(29, 8, 4),
            optional_null_marker(40), ordinary_pc_buffer(53, 120, 0),
            zero_record_marker(56), zero_record_marker(200),
        };
        assign_convention_bindings(resources, 2);
        return resources;
    };
    ComputeShaderConfig gta_zero_record_compile_config;
    gta_zero_record_compile_config.user_sgprs.resize(15);
    gta_zero_record_compile_config.local_x = 64;
    gta_zero_record_compile_config.local_y = gta_zero_record_compile_config.local_z = 1;
    gta_zero_record_compile_config.exact_thread_extent = true;
    gta_zero_record_compile_config.threads_x = 2063;
    gta_zero_record_compile_config.threads_y = gta_zero_record_compile_config.threads_z = 1;
    gta_zero_record_compile_config.wave_size = 64;
    gta_zero_record_compile_config.native_subgroup_size = 64;
    gta_zero_record_compile_config.tidig_comp_cnt = 0;
    gta_zero_record_compile_config.tgid_x_en = true;
    gta_zero_record_compile_config.lds_bytes = 65536;
    ShaderResourceTable gta_zero_record_compile_table =
        gta_zero_record_compile_resources();
    const std::vector<uint32_t> gta_zero_record_spirv = recompile_compute(
        prosper::test::kGta5ZeroRecordExeczProgram.data(),
        prosper::test::kGta5ZeroRecordExeczProgram.size(),
        &gta_zero_record_compile_table, gta_zero_record_compile_config);
    const DescriptorValidationReport gta_zero_record_spirv_report =
        validate_spirv_descriptor_interface(
            gta_zero_record_spirv, &gta_zero_record_compile_table, 0,
            SpirvShaderStage::Compute, false);
    const SpirvDescriptorBinding* gta_zero_record_pc27_binding =
        find_spirv_descriptor_binding(gta_zero_record_spirv_report, 0, 2);
    CHECK(!gta_zero_record_spirv.empty() && gta_zero_record_spirv_report.ok() &&
              gta_zero_record_pc27_binding && gta_zero_record_pc27_binding->writable &&
              !gta_zero_record_pc27_binding->readable &&
              gta_zero_record_pc27_binding->dynamic_access,
          "GTA pc47 descriptor fragment elides without enlarging pc27's exact 32-byte output");

    std::array<uint32_t, 319> gta_zero_record_pc47_wrong_dst =
        prosper::test::kGta5ZeroRecordExeczProgram;
    gta_zero_record_pc47_wrong_dst[47] = 0xf4040200u; // pc47 SDATA s[4:5] -> s[8:9]
    ShaderResourceTable gta_zero_record_pc47_wrong_dst_table =
        gta_zero_record_compile_resources();
    const std::vector<uint32_t> gta_zero_record_pc47_wrong_dst_spirv =
        recompile_compute(gta_zero_record_pc47_wrong_dst.data(),
                          gta_zero_record_pc47_wrong_dst.size(),
                          &gta_zero_record_pc47_wrong_dst_table,
                          gta_zero_record_compile_config);
    const DescriptorValidationReport gta_zero_record_pc47_wrong_dst_report =
        validate_spirv_descriptor_interface(
            gta_zero_record_pc47_wrong_dst_spirv,
            &gta_zero_record_pc47_wrong_dst_table, 0,
            SpirvShaderStage::Compute, false);
    const size_t gta_zero_record_pc47_wrong_dst_issues = std::count_if(
        gta_zero_record_pc47_wrong_dst_report.issues.begin(),
        gta_zero_record_pc47_wrong_dst_report.issues.end(),
        [](const DescriptorValidationIssue& issue) {
            return issue.code == DescriptorIssueCode::UndersizedBuffer &&
                issue.binding == 2u && issue.required_bytes == 88u &&
                issue.available_bytes == 32u;
        });
    CHECK(!gta_zero_record_pc47_wrong_dst_spirv.empty() &&
              !gta_zero_record_pc47_wrong_dst_report.ok() &&
              gta_zero_record_pc47_wrong_dst_report.issues.size() == 1u &&
              gta_zero_record_pc47_wrong_dst_issues == 1u,
          "same-site pc47 SDATA mutation restores the exact accidental binding-2 read");

    ShaderResourceTable gta_zero_record_compile_ne_table =
        gta_zero_record_compile_resources();
    CHECK(recompile_compute(gta_zero_record_compare_ne.data(),
                            gta_zero_record_compare_ne.size(),
                            &gta_zero_record_compile_ne_table,
                            gta_zero_record_compile_config).empty(),
          "production translation rejects the same-site pc91 CMPX_NE mutation at live pc194");

    ShaderResourceTable gta_zero_record_compile_second_ne_table =
        gta_zero_record_compile_resources();
    CHECK(recompile_compute(gta_zero_record_second_compare_ne.data(),
                            gta_zero_record_second_compare_ne.size(),
                            &gta_zero_record_compile_second_ne_table,
                            gta_zero_record_compile_config).empty(),
          "production translation rejects the same-site pc211 CMPX_NE mutation at live pc314");

    auto optional_mutation_stays_rejected = [&](std::array<uint32_t, 13> code) {
        ShaderResourceTable resources;
        const std::vector<SrtUse> uses = add_compute_buffer_resources(
            resources, code.data(), code.size(), optional_user_sgprs.data(),
            optional_user_sgprs.size(), kGtaOptionalBufferLocalSize, optional_records,
            kGtaOptionalBufferTgidSgpr);
        const bool no_optional_use = std::none_of(uses.begin(), uses.end(),
            [](const SrtUse& use) { return use.optional_null_raw_load; });
        return no_optional_use && resources.resources.empty() &&
            recompile_compute(code.data(), code.size(), &resources, optional_config).empty();
    };

    std::array<uint32_t, 13> optional_wrong_srsrc = optional_null_raw;
    optional_wrong_srsrc[11] = 0x80050002u; // pc10 consumes s[20:23], not the produced V#
    CHECK(optional_mutation_stays_rejected(optional_wrong_srsrc),
          "pc10 SRSRC mutation disconnects the optional producer and stays fail-visible");

    std::array<uint32_t, 13> optional_wrong_offset = optional_null_raw;
    optional_wrong_offset[4] = 0xfa000050u; // same zero table, different producer field
    CHECK(optional_mutation_stays_rejected(optional_wrong_offset),
          "x2 producer offset mutation does not infer optional semantics from another zero qword");

    std::array<uint32_t, 13> optional_no_idxen = optional_null_raw;
    optional_no_idxen[10] = 0xe0300000u;
    CHECK(optional_mutation_stays_rejected(optional_no_idxen),
          "pc10 idxen mutation stays outside the linear optional-null contract");

    std::array<uint32_t, 13> optional_store = optional_null_raw;
    optional_store[10] = 0xe0702000u;
    CHECK(optional_mutation_stays_rejected(optional_store),
          "pc10 store mutation cannot inherit load-only optional-null semantics");
    CHECK(recompile_compute(optional_store.data(), optional_store.size(),
                            &optional_table_resources, optional_config).empty(),
          "even a forged pc10 optional marker cannot turn the load-only convention into a store");

    // Independent domain negative: the exact V# and launch geometry are insufficient without the
    // mapped +0x58 producer. This retains the broad addr0+nonzero rejection even when every static
    // descriptor dword happens to match the admitted live shape.
    const uint32_t direct_optional_shape[] = {
        0xe0302000u, 0x80000000u,
        0xbf810000u,
    };
    const uint32_t direct_optional_seed[4] = {
        0u, kGtaOptionalBufferStrideWord, optional_records,
        kGtaOptionalBufferConfigWord,
    };
    ShaderResourceTable direct_optional_table;
    const std::vector<SrtUse> direct_optional_uses = add_compute_buffer_resources(
        direct_optional_table, direct_optional_shape, std::size(direct_optional_shape),
        direct_optional_seed, std::size(direct_optional_seed),
        kGtaOptionalBufferLocalSize, optional_records, kGtaOptionalBufferTgidSgpr);
    CHECK(direct_optional_uses.empty() && direct_optional_table.resources.empty(),
          "direct addr0 nonzero V# stays rejected without mapped optional-entry provenance");

    // Positive initialized state from the same guest field: after WRITE_DATA places a mapped pointer
    // at +0x58, the exact program must materialize the ordinary resource rather than retain a zero
    // marker. This tests the discriminator that changes over the title's early/later dispatches.
    alignas(8) std::array<uint32_t, optional_records> optional_payload{};
    const uint64_t optional_payload_addr =
        reinterpret_cast<uint64_t>(optional_payload.data());
    optional_table[kGtaOptionalBufferPointerOffset / 4u] =
        static_cast<uint32_t>(optional_payload_addr);
    optional_table[kGtaOptionalBufferPointerOffset / 4u + 1u] =
        static_cast<uint32_t>(optional_payload_addr >> 32);
    ShaderResourceTable initialized_optional_table;
    const std::vector<SrtUse> initialized_optional_uses = add_compute_buffer_resources(
        initialized_optional_table, optional_null_raw.data(), optional_null_raw.size(),
        optional_user_sgprs.data(), optional_user_sgprs.size(),
        kGtaOptionalBufferLocalSize, optional_records, kGtaOptionalBufferTgidSgpr);
    assign_convention_bindings(initialized_optional_table, 2);
    const std::vector<uint32_t> initialized_optional_spirv = recompile_compute(
        optional_null_raw.data(), optional_null_raw.size(), &initialized_optional_table,
        optional_config);
    CHECK(initialized_optional_uses.size() == 1 &&
              !initialized_optional_uses[0].optional_null_raw_load &&
              initialized_optional_table.resources.size() == 1 &&
              initialized_optional_table.resources[0].gpu_addr == optional_payload_addr &&
              initialized_optional_table.resources[0].size == optional_payload.size() * 4u &&
              !is_optional_null_raw_load_buffer(initialized_optional_table.resources[0]) &&
              !initialized_optional_spirv.empty(),
          "initialized +0x58 entry materializes and recompiles its ordinary mapped buffer");
    // GTA V 0x413cf9a00's null-output dispatch rebuilds a one-record V# at s[0:3], then places its
    // three stores behind `~(S | M) & S` and an EXECZ exit. Keep the complete production 81-dword
    // shader so the proof is tied to the observed PCs, packets, descriptor builder, and CFG rather
    // than a helper-shaped toy. Give its unrelated input/table accesses real mapped backing so this
    // test reaches complete compute translation instead of stopping at an earlier resource use.
    const std::vector<uint32_t> gta_413cf9a00 = {
        0xbfa00003u, 0xd7460000u, 0x04010c07u, 0xf4040500u,
        0xfa000090u, 0xbe960304u, 0xbe9703ffu, 0x00016204u,
        0xbf8cc07fu, 0xbe951d92u, 0x816ac104u, 0xe0302000u,
        0x80050100u, 0xf4080300u, 0xfa0000c0u, 0xbf8cc07fu,
        0xf4200406u, 0xfa000000u, 0xbf8cc07fu, 0x7e040210u,
        0xbe9703ffu, 0x00016204u, 0xbe92047eu, 0x7da8006au,
        0xbf880003u, 0x4a040081u, 0xe0302000u, 0x80050202u,
        0xbefe0412u, 0xbf8c3f70u, 0x7da80302u, 0xbf880009u,
        0xf4040200u, 0xfa000098u, 0xbe8a0304u, 0xbe8b03ffu,
        0x00016204u, 0xbf8cc07fu, 0xbe891d92u, 0xe0702000u,
        0x80020001u, 0xbefe0412u, 0xbf128002u, 0x7d8a00f9u,
        0x06868080u, 0x85ea8012u, 0x8dea006au, 0x87fe126au,
        0xbf88001fu, 0x7e0e0210u, 0x936b6a10u, 0x936a056au,
        0x87048106u, 0x816b6a6bu, 0x9aea0510u, 0x93000510u,
        0x81016b6au, 0xbf078004u, 0x4a0620f9u, 0x86860681u,
        0x85ea1000u, 0x7e000210u, 0x7e10026au, 0x7e020210u,
        0x7e040280u, 0x7e080210u, 0x7e0a0281u, 0x7e0c0281u,
        0x8801ff03u, 0x00380000u, 0xbe800302u, 0xbe820381u,
        0xbe8303ffu, 0x00016204u, 0xe0740030u, 0x80000700u,
        0xe0780020u, 0x80000000u, 0xe07c0000u, 0x80000400u,
        0xbf810000u,
    };
    alignas(256) std::array<uint32_t, 64> gta_null_store_srt{};
    alignas(256) std::array<uint32_t, 256> gta_null_store_input{};
    alignas(256) std::array<uint32_t, 256> gta_null_store_output{};
    alignas(256) std::array<uint32_t, 16> gta_null_store_scalar{};
    auto put_address = [](uint32_t* words, uint64_t address) {
        words[0] = static_cast<uint32_t>(address);
        words[1] = static_cast<uint32_t>(address >> 32);
    };
    put_address(gta_null_store_srt.data() + 0x90u / 4u,
                reinterpret_cast<uint64_t>(gta_null_store_input.data()));
    put_address(gta_null_store_srt.data() + 0x98u / 4u,
                reinterpret_cast<uint64_t>(gta_null_store_output.data()));
    put_address(gta_null_store_srt.data() + 0xc0u / 4u,
                reinterpret_cast<uint64_t>(gta_null_store_scalar.data()));
    gta_null_store_srt[0xc0u / 4u + 1u] |= 4u << 16;
    gta_null_store_srt[0xc0u / 4u + 2u] = gta_null_store_scalar.size();
    gta_null_store_srt[0xc0u / 4u + 3u] = 0x00016204u;
    const uint64_t gta_null_store_srt_address =
        reinterpret_cast<uint64_t>(gta_null_store_srt.data());
    const uint32_t gta_413cf9a00_null_seed[7] = {
        static_cast<uint32_t>(gta_null_store_srt_address),
        static_cast<uint32_t>(gta_null_store_srt_address >> 32),
        0u, 0u, // nullable output pointer: the exact live null dispatch
        151u, 4u, 0u,
    };
    ShaderResourceTable gta_null_store_table;
    ShaderResource gta_null_store_srt_resource;
    gta_null_store_srt_resource.cls = ResourceClass::ConstantBuffer;
    gta_null_store_srt_resource.format = DataFormat::Uint32;
    gta_null_store_srt_resource.num_components = 1;
    gta_null_store_srt_resource.gpu_addr = gta_null_store_srt_address;
    gta_null_store_srt_resource.size = sizeof(gta_null_store_srt);
    gta_null_store_table.resources.push_back(gta_null_store_srt_resource);
    const std::vector<SrtUse> gta_null_store_uses = add_compute_buffer_resources(
        gta_null_store_table, gta_413cf9a00.data(), gta_413cf9a00.size(),
        gta_413cf9a00_null_seed, std::size(gta_413cf9a00_null_seed));
    assign_convention_bindings(gta_null_store_table, 2);
    ComputeShaderConfig gta_null_store_config;
    gta_null_store_config.user_sgprs.assign(
        std::begin(gta_413cf9a00_null_seed), std::end(gta_413cf9a00_null_seed));
    gta_null_store_config.local_x = 64u;
    gta_null_store_config.local_y = gta_null_store_config.local_z = 1u;
    gta_null_store_config.wave_size = 64u;
    gta_null_store_config.native_subgroup_size = 64u;
    gta_null_store_config.tidig_comp_cnt = 0u;
    const std::vector<uint32_t> gta_null_store_spirv = recompile_compute(
        gta_413cf9a00.data(), gta_413cf9a00.size(), &gta_null_store_table,
        gta_null_store_config);
    const DescriptorValidationReport gta_null_store_report =
        validate_spirv_descriptor_interface(gta_null_store_spirv, &gta_null_store_table, 0,
                                            SpirvShaderStage::Compute, false);
    const auto gta_null_store_markers = std::count_if(
        gta_null_store_table.resources.begin(), gta_null_store_table.resources.end(),
        is_proven_null_guarded_raw_store);
    CHECK(gta_null_store_markers == 3u &&
              gta_null_store_table.by_fetch_pc(74u) &&
              gta_null_store_table.by_fetch_pc(76u) &&
              gta_null_store_table.by_fetch_pc(78u) &&
              std::count_if(gta_null_store_uses.begin(), gta_null_store_uses.end(),
                          [](const SrtUse& use) {
                              return use.proven_null_guarded_raw_store && !use.zero_record_raw;
                          }) == 3u &&
              !gta_null_store_spirv.empty() && gta_null_store_report.ok(),
          "GTA V 0x413cf9a00 null dispatch materializes three exact guarded-store markers");

    // Production-site mutation arm: change only pc47 from S_AND_B64 EXEC,VCC,s[18:19] to S_OR_B64.
    // The identity no longer proves empty EXEC, so the unchanged one-record base-zero stores must
    // remain unresolved and recompilation must fail. This exercises the proof site itself.
    std::vector<uint32_t> gta_null_store_or_mutation = gta_413cf9a00;
    gta_null_store_or_mutation[47] = 0x88fe126au;
    ShaderResourceTable gta_null_store_or_table;
    gta_null_store_or_table.resources.push_back(gta_null_store_srt_resource);
    const std::vector<SrtUse> gta_null_store_or_uses = add_compute_buffer_resources(
        gta_null_store_or_table, gta_null_store_or_mutation.data(),
        gta_null_store_or_mutation.size(), gta_413cf9a00_null_seed,
        std::size(gta_413cf9a00_null_seed));
    assign_convention_bindings(gta_null_store_or_table, 2);
    CHECK(std::none_of(gta_null_store_or_uses.begin(), gta_null_store_or_uses.end(),
                       [](const SrtUse& use) {
                           return use.proven_null_guarded_raw_store;
                       }) &&
              std::none_of(gta_null_store_or_table.resources.begin(),
                           gta_null_store_or_table.resources.end(),
                           is_proven_null_guarded_raw_store) &&
              recompile_compute(gta_null_store_or_mutation.data(),
                                gta_null_store_or_mutation.size(),
                                &gta_null_store_or_table,
                                gta_null_store_config).empty(),
          "pc47 AND_B64-to-OR_B64 mutation removes the guarded-store proof and rejects pc74");

    // CFG mutation arm: redirect the existing pc24 branch to pc48, entering after the mask identity
    // and EXEC write. The store packets and null seed are unchanged, but the guard no longer
    // dominates them, so resource discovery must not manufacture any guarded-store markers.
    std::vector<uint32_t> gta_null_store_entry_mutation = gta_413cf9a00;
    gta_null_store_entry_mutation[24] = 0xbf820017u; // s_branch pc48
    ShaderResourceTable gta_null_store_entry_table;
    gta_null_store_entry_table.resources.push_back(gta_null_store_srt_resource);
    const std::vector<SrtUse> gta_null_store_entry_uses = add_compute_buffer_resources(
        gta_null_store_entry_table, gta_null_store_entry_mutation.data(),
        gta_null_store_entry_mutation.size(), gta_413cf9a00_null_seed,
        std::size(gta_413cf9a00_null_seed));
    assign_convention_bindings(gta_null_store_entry_table, 2);
    CHECK(std::none_of(gta_null_store_entry_uses.begin(),
                       gta_null_store_entry_uses.end(),
                       [](const SrtUse& use) {
                           return use.proven_null_guarded_raw_store;
                       }) &&
              std::none_of(gta_null_store_entry_table.resources.begin(),
                           gta_null_store_entry_table.resources.end(),
                           is_proven_null_guarded_raw_store) &&
              !gta_null_store_entry_table.by_fetch_pc(74u) &&
              !gta_null_store_entry_table.by_fetch_pc(76u) &&
              !gta_null_store_entry_table.by_fetch_pc(78u) &&
              recompile_compute(gta_null_store_entry_mutation.data(),
                                gta_null_store_entry_mutation.size(),
                                &gta_null_store_entry_table,
                                gta_null_store_config).empty(),
          "an alternate pc24-to-pc48 entry cannot acquire guarded-store resources");

    // Entering at the compare is also unsafe: it executes the visible mask sequence, but bypasses
    // the scalar dataflow that established the compared pointer. Keep this boundary distinct from
    // the pc48 arm so both the proof start and its interior remain fail-closed.
    std::vector<uint32_t> gta_null_store_compare_entry_mutation = gta_413cf9a00;
    gta_null_store_compare_entry_mutation[24] = 0xbf820011u; // s_branch pc42
    ShaderResourceTable gta_null_store_compare_entry_table;
    gta_null_store_compare_entry_table.resources.push_back(gta_null_store_srt_resource);
    const std::vector<SrtUse> gta_null_store_compare_entry_uses =
        add_compute_buffer_resources(
            gta_null_store_compare_entry_table,
            gta_null_store_compare_entry_mutation.data(),
            gta_null_store_compare_entry_mutation.size(), gta_413cf9a00_null_seed,
            std::size(gta_413cf9a00_null_seed));
    assign_convention_bindings(gta_null_store_compare_entry_table, 2);
    CHECK(std::none_of(gta_null_store_compare_entry_uses.begin(),
                       gta_null_store_compare_entry_uses.end(),
                       [](const SrtUse& use) {
                           return use.proven_null_guarded_raw_store;
                       }) &&
              std::none_of(gta_null_store_compare_entry_table.resources.begin(),
                           gta_null_store_compare_entry_table.resources.end(),
                           is_proven_null_guarded_raw_store) &&
              !gta_null_store_compare_entry_table.by_fetch_pc(74u) &&
              !gta_null_store_compare_entry_table.by_fetch_pc(76u) &&
              !gta_null_store_compare_entry_table.by_fetch_pc(78u) &&
              recompile_compute(gta_null_store_compare_entry_mutation.data(),
                                gta_null_store_compare_entry_mutation.size(),
                                &gta_null_store_compare_entry_table,
                                gta_null_store_config).empty(),
          "an alternate pc24-to-pc42 entry cannot acquire guarded-store resources");

    // GFX10.3's debug-condition SOPP branches also carry direct simm16 targets. Mutate the same
    // pc24 entry edge to S_CBRANCH_CDBGSYS pc42; recognizing only the common branch family would
    // let this alternate entry bypass the pointer dataflow while preserving the visible guard.
    std::vector<uint32_t> gta_null_store_debug_entry_mutation = gta_413cf9a00;
    gta_null_store_debug_entry_mutation[24] = 0xbf970011u; // s_cbranch_cdbgsys pc42
    ShaderResourceTable gta_null_store_debug_entry_table;
    gta_null_store_debug_entry_table.resources.push_back(gta_null_store_srt_resource);
    const std::vector<SrtUse> gta_null_store_debug_entry_uses =
        add_compute_buffer_resources(
            gta_null_store_debug_entry_table,
            gta_null_store_debug_entry_mutation.data(),
            gta_null_store_debug_entry_mutation.size(), gta_413cf9a00_null_seed,
            std::size(gta_413cf9a00_null_seed));
    assign_convention_bindings(gta_null_store_debug_entry_table, 2);
    CHECK(std::none_of(gta_null_store_debug_entry_uses.begin(),
                       gta_null_store_debug_entry_uses.end(),
                       [](const SrtUse& use) {
                           return use.proven_null_guarded_raw_store;
                       }) &&
              std::none_of(gta_null_store_debug_entry_table.resources.begin(),
                           gta_null_store_debug_entry_table.resources.end(),
                           is_proven_null_guarded_raw_store) &&
              !gta_null_store_debug_entry_table.by_fetch_pc(74u) &&
              !gta_null_store_debug_entry_table.by_fetch_pc(76u) &&
              !gta_null_store_debug_entry_table.by_fetch_pc(78u) &&
              recompile_compute(gta_null_store_debug_entry_mutation.data(),
                                gta_null_store_debug_entry_mutation.size(),
                                &gta_null_store_debug_entry_table,
                                gta_null_store_config).empty(),
          "an S_CBRANCH_CDBGSYS pc24-to-pc42 entry cannot acquire guarded-store resources");

    auto rejects_guarded_store_control_flow =
        [&](const std::vector<uint32_t>& mutated_shader) {
            ShaderResourceTable table;
            table.resources.push_back(gta_null_store_srt_resource);
            const std::vector<SrtUse> uses = add_compute_buffer_resources(
                table, mutated_shader.data(), mutated_shader.size(),
                gta_413cf9a00_null_seed, std::size(gta_413cf9a00_null_seed));
            assign_convention_bindings(table, 2);
            return std::none_of(uses.begin(), uses.end(),
                                [](const SrtUse& use) {
                                    return use.proven_null_guarded_raw_store;
                                }) &&
                   std::none_of(table.resources.begin(), table.resources.end(),
                                is_proven_null_guarded_raw_store) &&
                   !table.by_fetch_pc(74u) && !table.by_fetch_pc(76u) &&
                   !table.by_fetch_pc(78u) &&
                   recompile_compute(mutated_shader.data(), mutated_shader.size(),
                                     &table, gta_null_store_config).empty();
        };

    // The SOPK subvector-loop pair also carries direct SIMM16 transfers. Exercise both named
    // opcodes at the same pc24 edge; LOOP_END pc48 is the reviewer-observed alternate entry that
    // skips the guard, while LOOP_BEGIN proves the complete family remains fail-closed.
    std::vector<uint32_t> gta_null_store_loop_begin_mutation = gta_413cf9a00;
    gta_null_store_loop_begin_mutation[24] = 0xbd800017u; // s_subvector_loop_begin pc48
    CHECK(rejects_guarded_store_control_flow(gta_null_store_loop_begin_mutation),
          "an S_SUBVECTOR_LOOP_BEGIN pc24-to-pc48 entry cannot acquire guarded-store resources");
    std::vector<uint32_t> gta_null_store_loop_end_mutation = gta_413cf9a00;
    gta_null_store_loop_end_mutation[24] = 0xbe000017u; // s_subvector_loop_end pc48
    CHECK(rejects_guarded_store_control_flow(gta_null_store_loop_end_mutation),
          "an S_SUBVECTOR_LOOP_END pc24-to-pc48 entry cannot acquire guarded-store resources");

    // S_TRAP leaves the shader for an externally configured handler. Even if the packet is placed
    // before the visible guard, its unobservable return path invalidates a closed dominance proof.
    std::vector<uint32_t> gta_null_store_trap_mutation = gta_413cf9a00;
    gta_null_store_trap_mutation[24] = 0xbf920000u; // s_trap 0
    CHECK(rejects_guarded_store_control_flow(gta_null_store_trap_mutation),
          "an S_TRAP transfer prevents guarded-store closed-CFG provenance");

    // Relative scalar moves add runtime M0 to the encoded destination. Keep the encoded base at s0
    // so the ordinary fixed-destination overlap check cannot catch these pc38 mutations: each must
    // be rejected specifically because it can dynamically overwrite nullable s2:s3 before pc42.
    auto rejects_guarded_store_dynamic_destination =
        [&](const std::vector<uint32_t>& mutated_shader) {
            ShaderResourceTable table;
            table.resources.push_back(gta_null_store_srt_resource);
            add_compute_buffer_resources(
                table, mutated_shader.data(), mutated_shader.size(),
                gta_413cf9a00_null_seed, std::size(gta_413cf9a00_null_seed));
            assign_convention_bindings(table, 2);
            return !rdna2_gta5_null_guarded_raw_store_dispatch(
                       mutated_shader.data(), mutated_shader.size(),
                       gta_413cf9a00_null_seed,
                       std::size(gta_413cf9a00_null_seed)) &&
                   std::none_of(table.resources.begin(), table.resources.end(),
                                is_proven_null_guarded_raw_store) &&
                   !table.by_fetch_pc(74u) && !table.by_fetch_pc(76u) &&
                   !table.by_fetch_pc(78u) &&
                   recompile_compute(mutated_shader.data(), mutated_shader.size(),
                                     &table, gta_null_store_config).empty();
        };
    for (const auto& [word, name] : std::array<std::pair<uint32_t, const char*>, 3>{
             std::pair{0xbe803001u, "S_MOVRELD_B32"},
             std::pair{0xbe803102u, "S_MOVRELD_B64"},
             std::pair{0xbe804901u, "S_MOVRELSD_2_B32"},
        }) {
        std::vector<uint32_t> gta_null_store_relative_dst_mutation = gta_413cf9a00;
        gta_null_store_relative_dst_mutation[38] = word;
        CHECK(rejects_guarded_store_dynamic_destination(
                  gta_null_store_relative_dst_mutation), name);
    }

    // A table can be stale or hand-built independently of its shader. The exact pc74 marker and
    // store packet are insufficient without the complete pc42..80 byte/CFG proof.
    std::vector<uint32_t> stale_guarded_store_shader(77u, 0x7e000000u);
    stale_guarded_store_shader[74] = 0xe0740030u;
    stale_guarded_store_shader[75] = 0x80000700u;
    stale_guarded_store_shader[76] = 0xbf810000u;
    ShaderResourceTable stale_guarded_store_table;
    ShaderResource stale_guarded_store_marker;
    stale_guarded_store_marker.cls = ResourceClass::ConstantBuffer;
    stale_guarded_store_marker.format = DataFormat::Unknown;
    stale_guarded_store_marker.num_components = 0u;
    stale_guarded_store_marker.gpu_addr = 0u;
    stale_guarded_store_marker.size = 0u;
    stale_guarded_store_marker.stride = kProvenNullGuardedRawStoreStride;
    stale_guarded_store_marker.srt_offset = 0xFFFFFFFFu;
    stale_guarded_store_marker.sgpr_base = 0xFFFFFFFFu;
    stale_guarded_store_marker.fetch_pc = 74u;
    stale_guarded_store_table.resources.push_back(stale_guarded_store_marker);
    assign_convention_bindings(stale_guarded_store_table, 2);
    CHECK(is_proven_null_guarded_raw_store(stale_guarded_store_marker) &&
              recompile_compute(stale_guarded_store_shader.data(),
                                stale_guarded_store_shader.size(),
                                &stale_guarded_store_table,
                                gta_null_store_config).empty(),
          "a hand-built pc74 marker cannot bypass the complete guarded-shader proof");

    std::vector<uint32_t> gta_null_store_consumer_mutation = gta_413cf9a00;
    gta_null_store_consumer_mutation[74] = 0xe0340030u; // same packet fields, load_dwordx2
    ShaderResourceTable gta_null_store_consumer_table;
    gta_null_store_consumer_table.resources.push_back(gta_null_store_srt_resource);
    const std::vector<SrtUse> gta_null_store_consumer_uses = add_compute_buffer_resources(
        gta_null_store_consumer_table, gta_null_store_consumer_mutation.data(),
        gta_null_store_consumer_mutation.size(), gta_413cf9a00_null_seed,
        std::size(gta_413cf9a00_null_seed));
    assign_convention_bindings(gta_null_store_consumer_table, 2);
    CHECK(std::count_if(gta_null_store_consumer_uses.begin(),
                        gta_null_store_consumer_uses.end(),
                        [](const SrtUse& use) {
                            return use.proven_null_guarded_raw_store;
                        }) == 2u &&
              !gta_null_store_consumer_table.by_fetch_pc(74u) &&
              gta_null_store_consumer_table.by_fetch_pc(76u) &&
              gta_null_store_consumer_table.by_fetch_pc(78u) &&
              recompile_compute(gta_null_store_consumer_mutation.data(),
                                gta_null_store_consumer_mutation.size(),
                                &gta_null_store_consumer_table,
                                gta_null_store_config).empty(),
          "a load at exact pc74 cannot inherit guarded-store no-op semantics");

    uint32_t gta_413cf9a00_nonnull_seed[7];
    std::copy(std::begin(gta_413cf9a00_null_seed), std::end(gta_413cf9a00_null_seed),
              gta_413cf9a00_nonnull_seed);
    gta_413cf9a00_nonnull_seed[2] = 1u;
    ShaderResourceTable gta_nonnull_unmapped_table;
    gta_nonnull_unmapped_table.resources.push_back(gta_null_store_srt_resource);
    const std::vector<SrtUse> gta_nonnull_unmapped_uses = add_compute_buffer_resources(
        gta_nonnull_unmapped_table, gta_413cf9a00.data(), gta_413cf9a00.size(),
        gta_413cf9a00_nonnull_seed, std::size(gta_413cf9a00_nonnull_seed));
    assign_convention_bindings(gta_nonnull_unmapped_table, 2);
    ComputeShaderConfig gta_nonnull_unmapped_config = gta_null_store_config;
    gta_nonnull_unmapped_config.user_sgprs.assign(
        std::begin(gta_413cf9a00_nonnull_seed), std::end(gta_413cf9a00_nonnull_seed));
    CHECK(std::none_of(gta_nonnull_unmapped_uses.begin(), gta_nonnull_unmapped_uses.end(),
                       [](const SrtUse& use) {
                           return use.proven_null_guarded_raw_store;
                       }) &&
              std::none_of(gta_nonnull_unmapped_table.resources.begin(),
                           gta_nonnull_unmapped_table.resources.end(),
                           is_proven_null_guarded_raw_store) &&
              recompile_compute(gta_413cf9a00.data(), gta_413cf9a00.size(),
                                &gta_nonnull_unmapped_table,
                                gta_nonnull_unmapped_config).empty(),
          "a non-null unmapped output pointer does not acquire null-guarded store markers");
    CHECK(recompile_compute(gta_413cf9a00.data(), gta_413cf9a00.size(),
                            &gta_null_store_table,
                            gta_nonnull_unmapped_config).empty(),
          "a stale null-dispatch marker cannot erase stores for non-null user s2:s3");

    // GTA V's two workgroup-list kernels load an optional pointer from the mapped dispatch table at
    // +0x20, then build [base=0,stride=1024,records=s7,config=0x16204]. Keep their complete production
    // arrays: the three stores may disappear and pc166 may return zero only under exact byte, launch,
    // provenance, and retained-witness identity.
    alignas(16) std::array<uint32_t, 16> nullable_output_witness{};
    std::vector<uint32_t> nullable_input_a(64u * 4u);
    std::vector<uint32_t> nullable_input_b(64u * 4u);
    std::vector<uint32_t> nullable_output(
        kGtaNullableOutputFixtureRecordCount * kGtaNullableOutputStride /
        sizeof(uint32_t));
    auto split_address = [](uint64_t address, uint32_t& lo, uint32_t& hi) {
        lo = static_cast<uint32_t>(address);
        hi = static_cast<uint32_t>(address >> 32u);
    };
    auto nullable_seed = [&](uint64_t pointer) {
        nullable_output_witness[kGtaNullableOutputPointerOffset / 4u] =
            static_cast<uint32_t>(pointer);
        nullable_output_witness[kGtaNullableOutputPointerOffset / 4u + 1u] =
            static_cast<uint32_t>(pointer >> 32u);
        std::array<uint32_t, 9> seed{};
        split_address(reinterpret_cast<uint64_t>(nullable_output_witness.data()),
                      seed[0], seed[1]);
        split_address(reinterpret_cast<uint64_t>(nullable_input_a.data()),
                      seed[2], seed[3]);
        split_address(reinterpret_cast<uint64_t>(nullable_input_b.data()),
                      seed[4], seed[5]);
        seed[6] = 64u;
        seed[7] = kGtaNullableOutputFixtureRecordCount;
        seed[8] = kGtaNullableOutputUserSgpr8;
        return seed;
    };
    auto nullable_context = [](bool tgid_x) {
        ComputeResourceDispatchContext context;
        context.local_x = kGtaNullableOutputLocalSize;
        context.local_y = context.local_z = 1u;
        context.threads_x = kGtaNullableOutputFixtureThreads;
        context.threads_y = context.threads_z = 1u;
        context.wave_size = 64u;
        context.tgid_x_en = tgid_x;
        context.tidig_comp_cnt = 0u;
        return context;
    };
    auto nullable_config = [](const std::array<uint32_t, 9>& seed,
                              const ComputeResourceDispatchContext& context) {
        ComputeShaderConfig config;
        config.user_sgprs.assign(seed.begin(), seed.end());
        config.local_x = context.local_x;
        config.local_y = context.local_y;
        config.local_z = context.local_z;
        config.threads_x = context.threads_x;
        config.threads_y = context.threads_y;
        config.threads_z = context.threads_z;
        config.exact_thread_extent = context.exact_thread_extent;
        config.wave_size = context.wave_size;
        config.tgid_x_en = context.tgid_x_en;
        config.tgid_y_en = context.tgid_y_en;
        config.tgid_z_en = context.tgid_z_en;
        config.tidig_comp_cnt = context.tidig_comp_cnt;
        return config;
    };
    auto materialize_nullable = [&](const auto& program, bool tgid_x,
                                    const std::array<uint32_t, 9>& seed,
                                    const ComputeResourceDispatchContext& context,
                                    ShaderResourceTable& table) {
        const std::vector<SrtUse> uses = add_compute_buffer_resources(
            table, program.data(), program.size(), seed.data(), seed.size(),
            context.local_x, context.threads_x,
            tgid_x ? static_cast<uint32_t>(seed.size()) : UINT32_MAX, &context);
        assign_convention_bindings(table, 2u);
        return uses;
    };

    const auto null_nullable_seed = nullable_seed(0u);
    const ComputeResourceDispatchContext short_nullable_context = nullable_context(true);
    ShaderResourceTable short_nullable_table;
    const std::vector<SrtUse> short_nullable_uses = materialize_nullable(
        prosper::test::kGta5WorkgroupStoreProgram, true, null_nullable_seed,
        short_nullable_context, short_nullable_table);
    const ComputeShaderConfig short_nullable_config = nullable_config(
        null_nullable_seed, short_nullable_context);
    const std::vector<uint32_t> short_nullable_spirv = recompile_compute(
        prosper::test::kGta5WorkgroupStoreProgram.data(),
        prosper::test::kGta5WorkgroupStoreProgram.size(), &short_nullable_table,
        short_nullable_config);
    CHECK(std::count_if(short_nullable_uses.begin(), short_nullable_uses.end(),
                        [](const SrtUse& use) {
                            return use.proven_null_nullable_raw_buffer;
                        }) == 1u &&
              std::count_if(short_nullable_table.resources.begin(),
                            short_nullable_table.resources.end(),
                            is_proven_null_nullable_raw_buffer) == 1u &&
              short_nullable_table.by_fetch_pc(38u) &&
              is_proven_null_nullable_raw_buffer(*short_nullable_table.by_fetch_pc(38u)) &&
              rdna2_gta5_nullable_output_dispatch(
                  prosper::test::kGta5WorkgroupStoreProgram.data(),
                  prosper::test::kGta5WorkgroupStoreProgram.size(),
                  short_nullable_config, short_nullable_table) &&
              !short_nullable_spirv.empty(),
          "GTA 0x413e192 exact null dispatch emits its pc38 store as a witnessed no-op");

    // The first retained fixture used 57 groups, while a later live Story Mode route used 63 with
    // identical program bytes and descriptor construction. Count is an exact dispatch equality,
    // not a fixed shader identity: s7, descriptor word2, groups_x, and threads_x/256 must agree.
    constexpr uint32_t kLaterLiveRecordCount = 63u;
    auto variable_nullable_seed = null_nullable_seed;
    variable_nullable_seed[7] = kLaterLiveRecordCount;
    ComputeResourceDispatchContext variable_nullable_context = short_nullable_context;
    variable_nullable_context.threads_x =
        kLaterLiveRecordCount * kGtaNullableOutputLocalSize;
    ShaderResourceTable variable_nullable_table;
    const std::vector<SrtUse> variable_nullable_uses = materialize_nullable(
        prosper::test::kGta5WorkgroupStoreProgram, true, variable_nullable_seed,
        variable_nullable_context, variable_nullable_table);
    const ComputeShaderConfig variable_nullable_config = nullable_config(
        variable_nullable_seed, variable_nullable_context);
    CHECK(std::count_if(variable_nullable_uses.begin(), variable_nullable_uses.end(),
                        [](const SrtUse& use) {
                            return use.proven_null_nullable_raw_buffer;
                        }) == 1u &&
              std::count_if(variable_nullable_table.resources.begin(),
                            variable_nullable_table.resources.end(),
                            is_proven_null_nullable_raw_buffer) == 1u &&
              !recompile_compute(
                  prosper::test::kGta5WorkgroupStoreProgram.data(),
                  prosper::test::kGta5WorkgroupStoreProgram.size(),
                  &variable_nullable_table, variable_nullable_config).empty(),
          "GTA 0x413e192 admits the exact later-live 63-group nullable dispatch");

    const ComputeResourceDispatchContext long_nullable_context = nullable_context(false);
    ShaderResourceTable long_nullable_table;
    const std::vector<SrtUse> long_nullable_uses = materialize_nullable(
        prosper::test::kGta5WorkgroupProcessProgram, false, null_nullable_seed,
        long_nullable_context, long_nullable_table);
    const ComputeShaderConfig long_nullable_config = nullable_config(
        null_nullable_seed, long_nullable_context);
    const std::vector<uint32_t> long_nullable_spirv = recompile_compute(
        prosper::test::kGta5WorkgroupProcessProgram.data(),
        prosper::test::kGta5WorkgroupProcessProgram.size(), &long_nullable_table,
        long_nullable_config);
    CHECK(std::count_if(long_nullable_uses.begin(), long_nullable_uses.end(),
                        [](const SrtUse& use) {
                            return use.proven_null_nullable_raw_buffer;
                        }) == 3u &&
              std::count_if(long_nullable_table.resources.begin(),
                            long_nullable_table.resources.end(),
                            is_proven_null_nullable_raw_buffer) == 3u &&
              long_nullable_table.by_fetch_pc(152u) &&
              long_nullable_table.by_fetch_pc(166u) &&
              long_nullable_table.by_fetch_pc(193u) &&
              rdna2_gta5_nullable_output_dispatch(
                  prosper::test::kGta5WorkgroupProcessProgram.data(),
                  prosper::test::kGta5WorkgroupProcessProgram.size(),
                  long_nullable_config, long_nullable_table) &&
              !long_nullable_spirv.empty(),
          "GTA 0x413e1ac exact null dispatch drops two stores and zero-loads pc166");

    // The exact long kernel extracts entry-s8 bits 30:28 at pc2 as ordinary work-selection data.
    // Routed gameplay exercised all eight values while the independently recovered nullable V#
    // remained exact. Validate the complete selector domain, including the alternate exact-thread
    // representation accepted for a dispatch with no partial workgroup.
    auto live_long_nullable_seed = null_nullable_seed;
    for (uint32_t selector = 0u; selector != 8u; ++selector) {
        live_long_nullable_seed[8] = kGtaNullableOutputUserSgpr8 | (selector << 28u);
        ComputeResourceDispatchContext live_long_nullable_context = long_nullable_context;
        live_long_nullable_context.exact_thread_extent = selector == 7u;
        ShaderResourceTable live_long_nullable_table;
        const std::vector<SrtUse> live_long_nullable_uses = materialize_nullable(
            prosper::test::kGta5WorkgroupProcessProgram, false, live_long_nullable_seed,
            live_long_nullable_context, live_long_nullable_table);
        const ComputeShaderConfig live_long_nullable_config = nullable_config(
            live_long_nullable_seed, live_long_nullable_context);
        CHECK(std::count_if(live_long_nullable_uses.begin(), live_long_nullable_uses.end(),
                            [](const SrtUse& use) {
                                return use.proven_null_nullable_raw_buffer;
                            }) == 3u &&
                  std::count_if(live_long_nullable_table.resources.begin(),
                                live_long_nullable_table.resources.end(),
                                is_proven_null_nullable_raw_buffer) == 3u &&
                  !recompile_compute(
                      prosper::test::kGta5WorkgroupProcessProgram.data(),
                      prosper::test::kGta5WorkgroupProcessProgram.size(),
                      &live_long_nullable_table, live_long_nullable_config).empty(),
              "GTA 0x413e1ac admits each routed s8 work selector");
    }

    auto unknown_long_nullable_seed = null_nullable_seed;
    unknown_long_nullable_seed[8] |= 0x80000000u;
    ShaderResourceTable unknown_long_nullable_table;
    (void)materialize_nullable(
        prosper::test::kGta5WorkgroupProcessProgram, false, unknown_long_nullable_seed,
        long_nullable_context, unknown_long_nullable_table);
    const ComputeShaderConfig unknown_long_nullable_config = nullable_config(
        unknown_long_nullable_seed, long_nullable_context);
    CHECK(std::none_of(unknown_long_nullable_table.resources.begin(),
                       unknown_long_nullable_table.resources.end(),
                       is_nullable_raw_buffer_marker_candidate) &&
              recompile_compute(
                  prosper::test::kGta5WorkgroupProcessProgram.data(),
                  prosper::test::kGta5WorkgroupProcessProgram.size(),
                  &unknown_long_nullable_table, unknown_long_nullable_config).empty(),
          "an unobserved long-kernel s8 flag cannot authorize nullable semantics");
    CHECK(!rdna2_gta5_nullable_output_launch(
              prosper::test::kGta5WorkgroupProcessProgram.data(),
              prosper::test::kGta5WorkgroupProcessProgram.size(),
              live_long_nullable_seed.data(), live_long_nullable_seed.size() - 1u,
              long_nullable_context.local_x, long_nullable_context.local_y,
              long_nullable_context.local_z, long_nullable_context.threads_x,
              long_nullable_context.threads_y, long_nullable_context.threads_z,
              long_nullable_context.exact_thread_extent, long_nullable_context.wave_size,
              long_nullable_context.tgid_x_en, long_nullable_context.tgid_y_en,
              long_nullable_context.tgid_z_en, long_nullable_context.tidig_comp_cnt),
          "a short user-SGPR block rejects before reading the exact s8 ABI word");

    ShaderResourceTable short_nullable_witness = long_nullable_table;
    for (ShaderResource& resource : short_nullable_witness.resources) {
        if (!is_nullable_raw_buffer_marker_candidate(resource)) continue;
        resource.host_data = reinterpret_cast<uint8_t*>(nullable_output_witness.data());
        resource.host_data_size = kGtaNullableOutputWitnessBytes - 1u;
    }
    CHECK(recompile_compute(
              prosper::test::kGta5WorkgroupProcessProgram.data(),
              prosper::test::kGta5WorkgroupProcessProgram.size(),
              &short_nullable_witness, long_nullable_config).empty(),
          "a short retained +0x20 witness cannot authorize nullable load/store semantics");

    ShaderResourceTable unreadable_nullable_witness = long_nullable_table;
    ComputeShaderConfig unreadable_nullable_config = long_nullable_config;
    constexpr uint64_t kUnmappedWitnessAddress = 0xffffffffffff0000ull;
    unreadable_nullable_config.user_sgprs[0] =
        static_cast<uint32_t>(kUnmappedWitnessAddress);
    unreadable_nullable_config.user_sgprs[1] =
        static_cast<uint32_t>(kUnmappedWitnessAddress >> 32u);
    for (ShaderResource& resource : unreadable_nullable_witness.resources) {
        if (!is_nullable_raw_buffer_marker_candidate(resource)) continue;
        resource.gpu_addr = kUnmappedWitnessAddress;
    }
    CHECK(recompile_compute(
              prosper::test::kGta5WorkgroupProcessProgram.data(),
              prosper::test::kGta5WorkgroupProcessProgram.size(),
              &unreadable_nullable_witness, unreadable_nullable_config).empty(),
          "an unreadable retained +0x20 witness cannot authorize nullable load/store semantics");

    // The same guest table after initialization is not a null marker. It must materialize ordinary
    // address-backed resources for all three consumers and retain real load/store behavior.
    const auto nonnull_nullable_seed = nullable_seed(
        reinterpret_cast<uint64_t>(nullable_output.data()));
    ShaderResourceTable nonnull_nullable_table;
    const std::vector<SrtUse> nonnull_nullable_uses = materialize_nullable(
        prosper::test::kGta5WorkgroupProcessProgram, false, nonnull_nullable_seed,
        long_nullable_context, nonnull_nullable_table);
    const ComputeShaderConfig nonnull_nullable_config = nullable_config(
        nonnull_nullable_seed, long_nullable_context);
    CHECK(std::none_of(nonnull_nullable_uses.begin(), nonnull_nullable_uses.end(),
                       [](const SrtUse& use) {
                           return use.proven_null_nullable_raw_buffer;
                       }) &&
              std::none_of(nonnull_nullable_table.resources.begin(),
                           nonnull_nullable_table.resources.end(),
                           is_proven_null_nullable_raw_buffer) &&
              nonnull_nullable_table.by_fetch_pc(152u) &&
              nonnull_nullable_table.by_fetch_pc(166u) &&
              nonnull_nullable_table.by_fetch_pc(193u) &&
              !recompile_compute(
                   prosper::test::kGta5WorkgroupProcessProgram.data(),
                   prosper::test::kGta5WorkgroupProcessProgram.size(),
                   &nonnull_nullable_table, nonnull_nullable_config).empty(),
          "initialized +0x20 output keeps ordinary GTA load/store resources");
    nullable_seed(0u); // restore the zero witness for all mutation arms below

    auto nullable_mutation_rejects = [&](const auto& program, bool tgid_x,
                                         const ComputeResourceDispatchContext& context,
                                         const char* message,
                                         bool require_no_candidate = true) {
        ShaderResourceTable table;
        const std::vector<SrtUse> uses = materialize_nullable(
            program, tgid_x, null_nullable_seed, context, table);
        const ComputeShaderConfig config = nullable_config(null_nullable_seed, context);
        const bool rejected =
            (!require_no_candidate ||
             std::none_of(uses.begin(), uses.end(), [](const SrtUse& use) {
                 return use.proven_null_nullable_raw_buffer;
             })) &&
            std::none_of(table.resources.begin(), table.resources.end(),
                         is_proven_null_nullable_raw_buffer) &&
            recompile_compute(program.data(), program.size(), &table, config).empty();
        CHECK(rejected, message);
    };

    // Same-site mutation arms: perturb each admitted MUBUF packet, not a helper or assertion path.
    auto short_store_mutation = prosper::test::kGta5WorkgroupStoreProgram;
    short_store_mutation[38] ^= 0x00040000u; // store_dword -> store_dwordx2
    nullable_mutation_rejects(short_store_mutation, true, short_nullable_context,
        "pc38 store-width mutation removes the nullable-output proof");
    for (const auto [pc, name] : std::array<std::pair<uint32_t, const char*>, 3>{
             std::pair{152u, "pc152 store-width mutation removes the nullable-output proof"},
             std::pair{166u, "pc166 load-width mutation removes the nullable-output proof"},
             std::pair{193u, "pc193 store-width mutation removes the nullable-output proof"}}) {
        auto mutation = prosper::test::kGta5WorkgroupProcessProgram;
        mutation[pc] ^= 0x00040000u;
        nullable_mutation_rejects(mutation, false, long_nullable_context, name);
    }

    auto nullable_producer_mutation = prosper::test::kGta5WorkgroupProcessProgram;
    nullable_producer_mutation[103] = 0xfa000024u; // exact +0x20 S_LOAD producer -> +0x24
    nullable_mutation_rejects(nullable_producer_mutation, false, long_nullable_context,
        "+0x20 producer-site mutation cannot manufacture nullable markers");
    auto nullable_stride_mutation = prosper::test::kGta5WorkgroupProcessProgram;
    nullable_stride_mutation[106] = 0x02000000u;
    nullable_mutation_rejects(nullable_stride_mutation, false, long_nullable_context,
        "descriptor stride-builder mutation removes the nullable-output proof");
    auto nullable_count_mutation = prosper::test::kGta5WorkgroupProcessProgram;
    nullable_count_mutation[107] = 0xbe8a0306u; // s10=s6 instead of exact dispatch count s7
    nullable_mutation_rejects(nullable_count_mutation, false, long_nullable_context,
        "descriptor count-builder mutation removes the nullable-output proof");
    auto nullable_config_mutation = prosper::test::kGta5WorkgroupProcessProgram;
    nullable_config_mutation[110] ^= 1u;
    nullable_mutation_rejects(nullable_config_mutation, false, long_nullable_context,
        "descriptor config-word mutation removes the nullable-output proof");

    ComputeResourceDispatchContext wrong_nullable_local = long_nullable_context;
    wrong_nullable_local.local_x = 128u;
    nullable_mutation_rejects(prosper::test::kGta5WorkgroupProcessProgram, false,
        wrong_nullable_local, "local-size mutation cannot retain nullable-output markers", false);
    ComputeResourceDispatchContext wrong_nullable_threads = long_nullable_context;
    --wrong_nullable_threads.threads_x;
    nullable_mutation_rejects(prosper::test::kGta5WorkgroupProcessProgram, false,
        wrong_nullable_threads, "group-count mutation cannot retain nullable-output markers", false);

    // GTA V 0x413d59600 pc67 uses this exact FORMAT-load packet through the all-zero descriptor
    // produced by its earlier zero-record scalar loads. Keep the exact pc because the marker is
    // deliberately per-consumer; ordinary nonempty format loads remain on DynFetch's typed path.
    std::vector<uint32_t> zero_record_format_load(70, 0x7e000000u); // v_nop padding
    zero_record_format_load[67] = 0xe0082000u;
    zero_record_format_load[68] = 0x8000110cu;
    zero_record_format_load[69] = 0xbf810000u;
    const uint32_t zero_record_format_seed[4]{};
    ShaderResourceTable zero_record_format_table;
    const std::vector<SrtUse> zero_record_format_uses = add_compute_buffer_resources(
        zero_record_format_table, zero_record_format_load.data(), zero_record_format_load.size(),
        zero_record_format_seed, std::size(zero_record_format_seed));
    assign_convention_bindings(zero_record_format_table, 2);
    ComputeShaderConfig zero_record_format_config;
    zero_record_format_config.user_sgprs.assign(
        zero_record_format_seed, zero_record_format_seed + std::size(zero_record_format_seed));
    zero_record_format_config.local_x = zero_record_format_config.local_y =
        zero_record_format_config.local_z = 1;
    CHECK(zero_record_format_uses.size() == 1 &&
              zero_record_format_uses[0].use_pc == 67 &&
              zero_record_format_uses[0].zero_record_raw &&
              zero_record_format_table.resources.size() == 1 &&
              is_zero_record_raw_buffer(zero_record_format_table.resources[0]) &&
              std::all_of(std::begin(zero_record_format_table.resources[0].swizzle),
                          std::end(zero_record_format_table.resources[0].swizzle),
                          [](uint32_t selector) { return selector == 0u; }) &&
              zero_record_format_table.by_fetch_pc(67) &&
              !recompile_compute(zero_record_format_load.data(), zero_record_format_load.size(),
                                 &zero_record_format_table,
                                 zero_record_format_config).empty(),
          "exact GTA V pc67 zero-record FORMAT load lowers without a backing buffer");

    uint32_t one_record_format_seed[4]{};
    one_record_format_seed[2] = 1u;
    ShaderResourceTable one_record_format_table;
    const std::vector<SrtUse> one_record_format_uses = add_compute_buffer_resources(
        one_record_format_table, zero_record_format_load.data(), zero_record_format_load.size(),
        one_record_format_seed, std::size(one_record_format_seed));
    CHECK(one_record_format_uses.empty() && one_record_format_table.resources.empty(),
          "exact GTA V pc67 FORMAT load keeps base-zero one-record V# unresolved");

    // SQ_SEL_1 is the other zero-record boundary: an OOB FORMAT component selected as constant one
    // does not have an all-zero result. It must not acquire the no-backing zero marker.
    uint32_t one_selector_format_seed[4]{};
    one_selector_format_seed[3] = 1u; // DST_SEL_X = SQ_SEL_1
    ShaderResourceTable one_selector_format_table;
    const std::vector<SrtUse> one_selector_format_uses = add_compute_buffer_resources(
        one_selector_format_table, zero_record_format_load.data(), zero_record_format_load.size(),
        one_selector_format_seed, std::size(one_selector_format_seed));
    CHECK(one_selector_format_table.resources.empty() &&
              std::none_of(one_selector_format_uses.begin(), one_selector_format_uses.end(),
                           [](const SrtUse& use) { return use.zero_record_raw; }),
          "zero-record FORMAT load with SQ_SEL_1 stays unresolved");

    // Astro Bot's 7f5f world-map NGG shader patches only NUM_RECORDS in its direct s[16:19] V#
    // (`s_mov_b32 s18, 1`) before a raw buffer_load_dwordx4 at pc2761. The consumer is itself the
    // architectural descriptor proof; requiring the four words to retain pristine entry-seed
    // provenance dropped this fully-known live descriptor and therefore the whole draw.
    alignas(16) static uint32_t patched_raw_payload[4]{};
    const uint64_t patched_raw_base = reinterpret_cast<uint64_t>(patched_raw_payload);
    uint32_t patched_raw_seed[12]{};
    patched_raw_seed[8]  = static_cast<uint32_t>(patched_raw_base); // shader s16, user block +8
    patched_raw_seed[9]  = (static_cast<uint32_t>(patched_raw_base >> 32) & 0xffffu) |
                           (16u << 16);
    patched_raw_seed[10] = 64u;
    patched_raw_seed[11] = 0u;
    const uint32_t patched_direct_raw[] = {
        0xBE920381u,                // s_mov_b32 s18, 1 (live NUM_RECORDS patch)
        0xE03C2020u, 0x8004074Cu,   // exact Astro raw buffer_load_dwordx4 ..., s[16:19]
        0xBF810000u,
    };
    std::vector<SrtUse> patched_raw_uses;
    resolve_dynamic_fetch(patched_direct_raw, std::size(patched_direct_raw),
                          patched_raw_seed, std::size(patched_raw_seed), 8,
                          &patched_raw_uses);
    CHECK(patched_raw_uses.size() == 1 && patched_raw_uses[0].kind == 1 &&
              patched_raw_uses[0].use_pc == 1 &&
              patched_raw_uses[0].key == 0xFFFFFFFFu &&
              decode_buffer_descriptor(patched_raw_uses[0].v4.data()).base == patched_raw_base &&
              decode_buffer_descriptor(patched_raw_uses[0].v4.data()).num_records == 1,
          "Astro patched direct raw-MUBUF V# resolves by its exact consuming pc");

    // Astro Bot world-map PS consumes its direct s[16:19] V# with the raw x3 store at pc109.
    // Discovery must publish the binding before the recompiler can emit the otherwise-supported store.
    const uint32_t astro_store_x3[] = {
        0xE07C2000u, 0x8004030Au,   // buffer_store_dwordx3 v[3:5], v10, s[16:19], 0 idxen
        0xBF810000u,
    };
    std::vector<SrtUse> astro_store_x3_uses;
    resolve_dynamic_fetch(astro_store_x3, std::size(astro_store_x3),
                          patched_raw_seed, std::size(patched_raw_seed), 8,
                          &astro_store_x3_uses);
    CHECK(astro_store_x3_uses.size() == 1 && astro_store_x3_uses[0].kind == 1 &&
              astro_store_x3_uses[0].use_pc == 0 &&
              astro_store_x3_uses[0].key == 0xFFFFFFFFu &&
              astro_store_x3_uses[0].v4[0] == patched_raw_seed[8] &&
              astro_store_x3_uses[0].v4[1] == patched_raw_seed[9],
          "Astro buffer_store_dwordx3 publishes its direct V# by exact consumer pc");

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

    // A Wave32 B32 SAVEEXEC writes only its explicit one-word SDST. GTA V places that saved mask in
    // s3 immediately before a still-live V# in s[4:7]; pair-erasing the unmodeled SOP1 falsely
    // invalidated the descriptor before its final stores.
    const std::array<uint32_t, 4> adjacent_b32_saveexec = {
        0xBE833C6Bu,                // s_and_saveexec_b32 s3, vcc_hi
        0xE01C2000u, 0x80010101u,   // buffer_store_format_xyzw v[1:4], v1, s[4:7]
        0xBF810000u,
    };
    std::vector<SrtUse> adjacent_b32_saveexec_uses;
    resolve_dynamic_fetch(adjacent_b32_saveexec.data(), adjacent_b32_saveexec.size(),
                          direct_copy_seed, std::size(direct_copy_seed), 0,
                          &adjacent_b32_saveexec_uses);
    CHECK(adjacent_b32_saveexec_uses.size() == 1 &&
              adjacent_b32_saveexec_uses[0].use_pc == 1 &&
              adjacent_b32_saveexec_uses[0].v4[0] == direct_copy_seed[4],
          "B32 SAVEEXEC preserves the adjacent descriptor's first word");

    auto overlapping_b32_saveexec = adjacent_b32_saveexec;
    overlapping_b32_saveexec[0] = 0xBE843C6Bu; // same SAVEEXEC site, now SDST=s4
    std::vector<SrtUse> overlapping_b32_saveexec_uses;
    resolve_dynamic_fetch(overlapping_b32_saveexec.data(), overlapping_b32_saveexec.size(),
                          direct_copy_seed, std::size(direct_copy_seed), 0,
                          &overlapping_b32_saveexec_uses);
    CHECK(overlapping_b32_saveexec_uses.empty(),
          "same-site SAVEEXEC destination mutation invalidates the overlapped descriptor");

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
    // Metadata discovery may already have published the same invalid direct V# for raw SMEM use.
    // MTBUF must not resurrect that older SGPR-keyed resource when its exact validated pc entry is
    // absent.
    ShaderResourceTable invalid_mtbuf_metadata;
    { ShaderResource r{}; r.cls = ResourceClass::ConstantBuffer;
      r.format = DataFormat::Unknown; r.num_components = 0; r.gpu_addr = 0x20000u;
      r.size = 64u; r.stride = 4u; r.sgpr_base = 0u;
      invalid_mtbuf_metadata.resources.push_back(r); }
    assign_convention_bindings(invalid_mtbuf_metadata, 2);
    ComputeShaderConfig invalid_mtbuf_config;
    invalid_mtbuf_config.user_sgprs.assign(invalid_mtbuf_seed, invalid_mtbuf_seed + 8);
    invalid_mtbuf_config.local_x = invalid_mtbuf_config.local_y =
        invalid_mtbuf_config.local_z = 1;
    CHECK(recompile_compute(direct_mtbuf_copy, std::size(direct_mtbuf_copy),
                            &invalid_mtbuf_metadata, invalid_mtbuf_config).empty(),
          "MTBUF cannot fall back to a metadata resource for an INVALID live V#");

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

    // #590: a stride-1 Uint8 buffer_load_format_x at a runtime (non-dword-aligned) byte address.
    // DOLL's post-process LUT compute kernels index a Uint8 table this way; the descriptor-defined
    // aligned packed extraction rejected it (stride 1 is not 4-aligned) until the integer sub-dword
    // dynamic-extract path. It must resolve the V# AND recompile to non-empty SPIR-V.
    static uint32_t u8_table[4] = { 0x03020100u };            // 4 bytes {0,1,2,3}, one dword
    const uint64_t u8_base = (uint64_t)(uintptr_t)u8_table;
    uint32_t u8_seed[12] = {};
    u8_seed[8]  = (uint32_t)u8_base;                          // V# at s[8:11]: base low
    u8_seed[9]  = ((uint32_t)(u8_base >> 32) & 0xFFFFu) | (1u << 16);  // base hi | stride=1
    u8_seed[10] = 4u;                                         // num_records = 4
    u8_seed[11] = (5u << 12) | 4u;                            // format field 5 (8_UINT), dst_sel_x = X
    const uint32_t u8_format_load[] = {
        0xE0002000u, 0x80020000u,   // buffer_load_format_x v0, v0, s[8:11], 0 idxen
        0xBF810000u,
    };
    ShaderResourceTable u8_table_rt;
    add_compute_buffer_resources(u8_table_rt, u8_format_load, std::size(u8_format_load),
                                 u8_seed, std::size(u8_seed));
    assign_convention_bindings(u8_table_rt, 2);
    ComputeShaderConfig u8_config;
    u8_config.user_sgprs.assign(u8_seed, u8_seed + std::size(u8_seed));
    u8_config.local_x = u8_config.local_y = u8_config.local_z = 1;
    CHECK(u8_table_rt.by_fetch_pc(0) &&
          u8_table_rt.by_fetch_pc(0)->format == DataFormat::Uint8 &&
          !recompile_compute(u8_format_load, std::size(u8_format_load),
                             &u8_table_rt, u8_config).empty(),
          "#590: stride-1 Uint8 buffer_load_format_x recompiles via the integer sub-dword dynamic path");

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
    uint32_t store_compute_seed[10] = {};
    store_compute_seed[8] = seed5vs[0];
    store_compute_seed[9] = seed5vs[1];
    ShaderResourceTable store_compute_table;
    add_compute_buffer_resources(store_compute_table, k5vs, std::size(k5vs),
                                 store_compute_seed, std::size(store_compute_seed));
    CHECK(store_compute_table.resources.size() == 1 &&
          store_compute_table.by_srt_offset(0x10) &&
          store_compute_table.by_fetch_pc(2) == store_compute_table.by_srt_offset(0x10),
          "table-loaded raw store retains both SRT and exact consumer provenance");

    const uint32_t repeated_store[] = {
        0xF4080504u, 0xFA000010u,   // s_load_dwordx4 s[20:23], s[8:9], 0x10
        0xE0702000u, 0x80051100u,   // buffer_store_dword v17, v0, s[20:23]
        0xE0702004u, 0x80051100u,   // same V#, second consumer at a different pc
        0xBF810000u,
    };
    ShaderResourceTable repeated_store_table;
    add_compute_buffer_resources(repeated_store_table, repeated_store,
                                 std::size(repeated_store), store_compute_seed,
                                 std::size(store_compute_seed));
    CHECK(repeated_store_table.resources.size() == 2 &&
          repeated_store_table.by_fetch_pc(2) && repeated_store_table.by_fetch_pc(4),
          "each raw store consumer keeps an exact alias across compute CFG block boundaries");

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

    // UE4 emits a tiny one-dimensional clear kernel whose V# publishes the whole allocation's
    // record count. That can exceed the normal 256 MiB upload cap even though this dispatch touches
    // only a small prefix. Prove the exact GlobalInvocationId.x address shape before narrowing the
    // resource to the submitted thread extent; without the launch proof it must remain fail-closed.
    constexpr uint32_t linear_clear_bytes = 2047u * 1136u + 4u;
    alignas(16) static uint8_t linear_clear_output[linear_clear_bytes];
    const uint64_t linear_clear_base = (uint64_t)(uintptr_t)linear_clear_output;
    const uint32_t linear_clear_seed[4] = {
        (uint32_t)linear_clear_base,
        ((uint32_t)(linear_clear_base >> 32) & 0xFFFFu) | (1136u << 16),
        0x0010C01Du,
        0xA1B00FACu,
    };
    const uint32_t linear_clear[] = {
        0xD7460000u, 0x04010C04u, // v_lshl_add_u32 v0, s4, 6, v0
        0x7E020280u,              // v_mov_b32 v1, 0
        0xE0102000u, 0x80000100u, // buffer_store_dword v1, v0, s[0:3], 0 idxen
        0xBF810000u,
    };
    CHECK(resolve_dynamic_fetch(linear_clear, std::size(linear_clear), linear_clear_seed,
                                std::size(linear_clear_seed), 0).empty(),
          "generic descriptor fold keeps the oversized formatless store rejected");
    ShaderResourceTable unproven_linear_clear_table;
    add_compute_buffer_resources(unproven_linear_clear_table, linear_clear,
                                 std::size(linear_clear), linear_clear_seed,
                                 std::size(linear_clear_seed));
    CHECK(unproven_linear_clear_table.resources.empty(),
          "oversized formatless raw store stays rejected without dispatch proof");
    uint32_t short_linear_clear_seed[4];
    std::copy(std::begin(linear_clear_seed), std::end(linear_clear_seed),
              short_linear_clear_seed);
    short_linear_clear_seed[2] = 1; // one 1136-byte record cannot cover the dispatch footprint
    ShaderResourceTable short_linear_clear_table;
    add_compute_buffer_resources(short_linear_clear_table, linear_clear,
                                 std::size(linear_clear), short_linear_clear_seed,
                                 std::size(short_linear_clear_seed), 64, 2048, 4);
    CHECK(short_linear_clear_table.resources.empty(),
          "linear clear proof cannot expand past the guest descriptor's declared range");
    uint32_t bounded_linear_clear_seed[4];
    std::copy(std::begin(linear_clear_seed), std::end(linear_clear_seed),
              bounded_linear_clear_seed);
    bounded_linear_clear_seed[2] = 3000; // valid but below the generic 256 MiB ceiling
    ShaderResourceTable bounded_linear_clear_table;
    add_compute_buffer_resources(bounded_linear_clear_table, linear_clear,
                                 std::size(linear_clear), bounded_linear_clear_seed,
                                 std::size(bounded_linear_clear_seed), 64, 2048, 4);
    CHECK(bounded_linear_clear_table.resources.size() == 1 &&
              bounded_linear_clear_table.resources[0].size == linear_clear_bytes,
          "proven formatless clear suppresses its redundant generic SRT resource");
    uint32_t typed_linear_clear_seed[4];
    std::copy(std::begin(linear_clear_seed), std::end(linear_clear_seed),
              typed_linear_clear_seed);
    typed_linear_clear_seed[2] = 2048;
    typed_linear_clear_seed[3] =
        (typed_linear_clear_seed[3] & ~(0x7Fu << 12)) | (56u << 12); // RGBA8 UNORM
    ShaderResourceTable typed_linear_clear_table;
    add_compute_buffer_resources(typed_linear_clear_table, linear_clear,
                                 std::size(linear_clear), typed_linear_clear_seed,
                                 std::size(typed_linear_clear_seed), 64, 2048, 4);
    CHECK(typed_linear_clear_table.resources.size() == 1 &&
              typed_linear_clear_table.resources[0].format == DataFormat::Unorm8 &&
              typed_linear_clear_table.resources[0].size == 2048u * 1136u,
          "typed clear keeps its width-aware generic resource instead of a Uint32 alias");
    ShaderResourceTable linear_clear_table;
    add_compute_buffer_resources(linear_clear_table, linear_clear, std::size(linear_clear),
                                 linear_clear_seed, std::size(linear_clear_seed),
                                 64, 2048, 4);
    assign_convention_bindings(linear_clear_table, 2);
    ComputeShaderConfig linear_clear_config;
    linear_clear_config.user_sgprs.assign(linear_clear_seed, linear_clear_seed + 4);
    linear_clear_config.local_x = 64;
    linear_clear_config.local_y = linear_clear_config.local_z = 1;
    linear_clear_config.tgid_x_en = true;
    CHECK(linear_clear_table.resources.size() == 1,
          "proven linear raw store creates exactly one resource");
    CHECK(linear_clear_table.resources.size() == 1 &&
              linear_clear_table.resources[0].size == linear_clear_bytes &&
              linear_clear_table.resources[0].stride == 1136 &&
              linear_clear_table.resources[0].format == DataFormat::Uint32 &&
              linear_clear_table.resources[0].num_components == 1 &&
              linear_clear_table.resources[0].fetch_pc == 3,
          "proven linear raw store is bounded to the submitted X invocation extent");
    CHECK(!recompile_compute(linear_clear, std::size(linear_clear),
                             &linear_clear_table, linear_clear_config).empty(),
          "proven linear raw store recompiles through its exact resource");

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

    // GTA V reaches the emitter with buffer_atomic_add/or packets whose descriptors are fully
    // recoverable, but resource discovery historically recognized only atomic_umax. Keep the
    // producer's accepted set in lock-step with the emitter's exact supported 32-bit RMW set. The
    // two full packets below are the game's terminal encodings; using their real SRSRC positions
    // also proves that the result is exact-PC provenance rather than the direct-SGPR fallback.
    uint32_t exact_add_seed[8] = {};
    std::copy(std::begin(atomic_seed), std::end(atomic_seed), exact_add_seed + 4);
    const uint32_t gta_atomic_add[] = {
        0xE0C84000u, 0x80010200u, // buffer_atomic_add v2, v0, s[4:7], 0 glc
        0xBF810000u,
    };
    std::vector<SrtUse> gta_add_uses;
    resolve_dynamic_fetch(gta_atomic_add, std::size(gta_atomic_add), exact_add_seed,
                          std::size(exact_add_seed), 0, &gta_add_uses);
    ShaderResourceTable gta_add_table;
    add_compute_buffer_resources(gta_add_table, gta_atomic_add, std::size(gta_atomic_add),
                                 exact_add_seed, std::size(exact_add_seed));
    assign_convention_bindings(gta_add_table, 2);
    ComputeShaderConfig gta_add_config;
    gta_add_config.user_sgprs.assign(std::begin(exact_add_seed), std::end(exact_add_seed));
    gta_add_config.local_x = gta_add_config.local_y = gta_add_config.local_z = 1;
    CHECK(gta_add_uses.size() == 1 && gta_add_uses[0].use_pc == 0 &&
              gta_add_table.resources.size() == 1 &&
              gta_add_table.resources[0].fetch_pc == 0 &&
              !recompile_compute(gta_atomic_add, std::size(gta_atomic_add),
                                 &gta_add_table, gta_add_config).empty(),
          "GTA V exact buffer_atomic_add publishes and consumes its exact-PC resource");

    uint32_t exact_or_seed[36] = {};
    std::copy(std::begin(atomic_seed), std::end(atomic_seed), exact_or_seed + 32);
    const uint32_t gta_atomic_or[] = {
        0xE0E82000u, 0x80080200u, // buffer_atomic_or v2, v0, s[32:35], 0
        0xBF810000u,
    };
    std::vector<SrtUse> gta_or_uses;
    resolve_dynamic_fetch(gta_atomic_or, std::size(gta_atomic_or), exact_or_seed,
                          std::size(exact_or_seed), 0, &gta_or_uses);
    ShaderResourceTable gta_or_table;
    add_compute_buffer_resources(gta_or_table, gta_atomic_or, std::size(gta_atomic_or),
                                 exact_or_seed, std::size(exact_or_seed));
    assign_convention_bindings(gta_or_table, 2);
    ComputeShaderConfig gta_or_config;
    gta_or_config.user_sgprs.assign(std::begin(exact_or_seed), std::end(exact_or_seed));
    gta_or_config.local_x = gta_or_config.local_y = gta_or_config.local_z = 1;
    CHECK(gta_or_uses.size() == 1 && gta_or_uses[0].use_pc == 0 &&
              gta_or_table.resources.size() == 1 &&
              gta_or_table.resources[0].fetch_pc == 0 &&
              !recompile_compute(gta_atomic_or, std::size(gta_atomic_or),
                                 &gta_or_table, gta_or_config).empty(),
          "GTA V exact buffer_atomic_or publishes and consumes its exact-PC resource");

    const uint32_t supported_atomic_dw0[] = {
        0xE0C00000u, // swap
        0xE0C80000u, // add
        0xE0CC0000u, // sub
        0xE0D40000u, // smin
        0xE0D80000u, // umin
        0xE0DC0000u, // smax
        0xE0E00000u, // umax
        0xE0E40000u, // and
        0xE0E80000u, // or
        0xE0EC0000u, // xor
        0xE0FC0000u, // fmin
        0xE1000000u, // fmax
    };
    uint32_t supported_atomic_uses = 0;
    for (uint32_t dw0 : supported_atomic_dw0) {
        const uint32_t code[] = { dw0, 0x80000000u, 0xBF810000u };
        std::vector<SrtUse> uses;
        resolve_dynamic_fetch(code, std::size(code), atomic_seed, std::size(atomic_seed), 0,
                              &uses);
        supported_atomic_uses += uses.size() == 1 && uses[0].use_pc == 0;
    }
    CHECK(supported_atomic_uses == std::size(supported_atomic_dw0),
          "resource discovery admits exactly every emitter-supported 32-bit MUBUF RMW opcode");

    // GTA V's six-packet reduction tail uses one direct stride-24 V# in s[4:7]. Opcode emission and
    // descriptor discovery are separate passes; accepting FMIN/FMAX in only the former leaves the
    // exact live shader terminal at its first packet with pc_res=null. Exercise the whole production
    // boundary, including one exact-PC alias per consumer and the live FP32 mode.
    static uint32_t gta_float_atomic_output[6] = {};
    const uint64_t gta_float_atomic_base =
        reinterpret_cast<uint64_t>(gta_float_atomic_output);
    uint32_t gta_float_atomic_seed[10] = {};
    gta_float_atomic_seed[4] = static_cast<uint32_t>(gta_float_atomic_base);
    gta_float_atomic_seed[5] = static_cast<uint32_t>(gta_float_atomic_base >> 32) |
                               (24u << 16);
    gta_float_atomic_seed[6] = 1u;
    gta_float_atomic_seed[7] = 0x00005204u;
    const uint32_t gta_float_atomic_tail[] = {
        0xE0FC0000u, 0x80010000u, // buffer_atomic_fmin v0, off, s[4:7], 0
        0xE0FC0004u, 0x80010100u, // buffer_atomic_fmin v1, off, s[4:7], 4
        0xE0FC0008u, 0x80010200u, // buffer_atomic_fmin v2, off, s[4:7], 8
        0xE100000Cu, 0x80010300u, // buffer_atomic_fmax v3, off, s[4:7], 12
        0xBF8CC07Fu,              // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xE1000010u, 0x80010400u, // buffer_atomic_fmax v4, off, s[4:7], 16
        0xE1000014u, 0x80010500u, // buffer_atomic_fmax v5, off, s[4:7], 20
        0xBF810000u,
    };
    std::vector<SrtUse> gta_float_atomic_uses;
    resolve_dynamic_fetch(gta_float_atomic_tail, std::size(gta_float_atomic_tail),
                          gta_float_atomic_seed, std::size(gta_float_atomic_seed), 0,
                          &gta_float_atomic_uses);
    ShaderResourceTable gta_float_atomic_table;
    add_compute_buffer_resources(gta_float_atomic_table, gta_float_atomic_tail,
                                 std::size(gta_float_atomic_tail), gta_float_atomic_seed,
                                 std::size(gta_float_atomic_seed));
    assign_convention_bindings(gta_float_atomic_table, 2);
    ComputeShaderConfig gta_float_atomic_config;
    gta_float_atomic_config.user_sgprs.assign(
        std::begin(gta_float_atomic_seed), std::end(gta_float_atomic_seed));
    gta_float_atomic_config.local_x = gta_float_atomic_config.local_y =
        gta_float_atomic_config.local_z = 1;
    gta_float_atomic_config.compute_pgm_rsrc1 = 0x402c00c3u;
    const uint32_t gta_float_atomic_pcs[] = {0u, 2u, 4u, 6u, 9u, 11u};
    const bool gta_float_atomic_exact_pcs =
        gta_float_atomic_uses.size() == std::size(gta_float_atomic_pcs) &&
        gta_float_atomic_table.resources.size() == std::size(gta_float_atomic_pcs) &&
        std::equal(gta_float_atomic_uses.begin(), gta_float_atomic_uses.end(),
                   std::begin(gta_float_atomic_pcs),
                   [](const SrtUse& use, uint32_t pc) { return use.use_pc == pc; }) &&
        std::equal(gta_float_atomic_table.resources.begin(),
                   gta_float_atomic_table.resources.end(),
                   std::begin(gta_float_atomic_pcs),
                   [](const ShaderResource& resource, uint32_t pc) {
                       return resource.fetch_pc == pc && resource.stride == 24u &&
                              resource.size == 24u;
                   });
    CHECK(gta_float_atomic_exact_pcs &&
              !recompile_compute(gta_float_atomic_tail, std::size(gta_float_atomic_tail),
                                 &gta_float_atomic_table, gta_float_atomic_config).empty(),
          "GTA V float-atomic tail publishes six exact-PC resources and recompiles");

    const uint32_t unknown_atomic_descriptor[] = {
        0xBE801F00u,              // s_getpc_b64 s[0:1]: not a concrete entry V# anymore
        0xE0C80000u, 0x80000000u, // buffer_atomic_add v0, v0, s[0:3]
        0xBF810000u,
    };
    std::vector<SrtUse> unknown_atomic_uses;
    resolve_dynamic_fetch(unknown_atomic_descriptor, std::size(unknown_atomic_descriptor),
                          atomic_seed, std::size(atomic_seed), 0, &unknown_atomic_uses);
    const uint32_t direct_atomic_add[] = {
        0xE0C80000u, 0x80000000u, // buffer_atomic_add v0, v0, s[0:3]
        0xBF810000u,
    };
    uint32_t oversized_atomic_seed[4];
    std::copy(std::begin(atomic_seed), std::end(atomic_seed), oversized_atomic_seed);
    oversized_atomic_seed[2] = 0x10000001u; // stride zero: NUM_RECORDS is the byte size
    std::vector<SrtUse> oversized_atomic_uses;
    resolve_dynamic_fetch(direct_atomic_add, std::size(direct_atomic_add),
                          oversized_atomic_seed, std::size(oversized_atomic_seed), 0,
                          &oversized_atomic_uses);
    uint32_t empty_atomic_seed[4];
    std::copy(std::begin(atomic_seed), std::end(atomic_seed), empty_atomic_seed);
    empty_atomic_seed[2] = 0;
    std::vector<SrtUse> empty_atomic_uses;
    resolve_dynamic_fetch(direct_atomic_add, std::size(direct_atomic_add), empty_atomic_seed,
                          std::size(empty_atomic_seed), 0, &empty_atomic_uses);
    CHECK(unknown_atomic_uses.empty() && oversized_atomic_uses.empty() &&
              empty_atomic_uses.size() == 1 && empty_atomic_uses[0].zero_record_raw &&
              empty_atomic_uses[0].use_pc == 0,
          "supported atomics require a fully-known bounded or exact zero-record live V#");

    // GTA V 0x413e1ac pc12 is the directly-proven empty site. Its sibling dispatch loads the same
    // table base and publishes an all-zero V# at offset 0x10. Retain the exact packet, load key, and
    // consuming PC: a direct-SGPR fallback or a generic address-zero shortcut cannot satisfy these
    // assertions. The marker has no descriptor in the generated module.
    alignas(16) uint32_t gta_413e1ac_table_words[8] = {};
    const uint64_t gta_413e1ac_table_base =
        reinterpret_cast<uint64_t>(gta_413e1ac_table_words);
    const uint32_t gta_413e1ac_seed[2] = {
        static_cast<uint32_t>(gta_413e1ac_table_base),
        static_cast<uint32_t>(gta_413e1ac_table_base >> 32),
    };
    std::vector<uint32_t> gta_413e1ac_site(12, 0xBF800000u); // s_nop padding to exact pc
    gta_413e1ac_site[8] = 0xF4080300u;  // s_load_dwordx4 s[12:15], s[0:1], 0x10
    gta_413e1ac_site[9] = 0xFA000010u;
    gta_413e1ac_site[10] = 0xBF8CC07Fu;
    gta_413e1ac_site.push_back(0xE0C86000u); // exact pc12 buffer_atomic_add, idxen+glc
    gta_413e1ac_site.push_back(0x80030201u);
    gta_413e1ac_site.push_back(0xBF810000u);
    ShaderResourceTable gta_413e1ac_table;
    const std::vector<SrtUse> gta_413e1ac_uses = add_compute_buffer_resources(
        gta_413e1ac_table, gta_413e1ac_site.data(), gta_413e1ac_site.size(),
        gta_413e1ac_seed, std::size(gta_413e1ac_seed));
    assign_convention_bindings(gta_413e1ac_table, 2);
    ComputeShaderConfig gta_413e1ac_config;
    gta_413e1ac_config.user_sgprs.assign(std::begin(gta_413e1ac_seed),
                                         std::end(gta_413e1ac_seed));
    gta_413e1ac_config.local_x = gta_413e1ac_config.local_y =
        gta_413e1ac_config.local_z = 1;
    const std::vector<uint32_t> gta_413e1ac_spirv = recompile_compute(
        gta_413e1ac_site.data(), gta_413e1ac_site.size(), &gta_413e1ac_table,
        gta_413e1ac_config);
    const DescriptorValidationReport gta_413e1ac_report =
        validate_spirv_descriptor_interface(gta_413e1ac_spirv, &gta_413e1ac_table, 0,
                                            SpirvShaderStage::Compute, false);
    CHECK(gta_413e1ac_uses.size() == 1 && gta_413e1ac_uses[0].zero_record_raw &&
              gta_413e1ac_uses[0].key == 0x10u && gta_413e1ac_uses[0].use_pc == 12u &&
              gta_413e1ac_table.resources.size() == 1 &&
              is_zero_record_raw_buffer(gta_413e1ac_table.resources[0]) &&
              gta_413e1ac_table.resources[0].fetch_pc == 12u &&
              !gta_413e1ac_spirv.empty() && gta_413e1ac_report.ok() &&
              gta_413e1ac_report.descriptors.empty(),
          "GTA V 0x413e1ac pc12 exact empty atomic becomes a binding-free PC marker");

    // 0x413d22d pc33 loads its V# from table offset 0x50. Adjacent live slots are proven empty and
    // the fully-known target descriptor can only be zero-sized or rejected as oversized; unlike the
    // site above, no sibling capture directly published this slot. This synthetic zero descriptor
    // therefore verifies the exact packet/key behavior, not an additional live descriptor claim.
    alignas(16) uint32_t gta_413d22d_table_words[24] = {};
    const uint64_t gta_413d22d_table_base =
        reinterpret_cast<uint64_t>(gta_413d22d_table_words);
    uint32_t gta_413d22d_seed[4] = {};
    gta_413d22d_seed[2] = static_cast<uint32_t>(gta_413d22d_table_base);
    gta_413d22d_seed[3] = static_cast<uint32_t>(gta_413d22d_table_base >> 32);
    std::vector<uint32_t> gta_413d22d_site(33, 0xBF800000u);
    gta_413d22d_site[30] = 0xF4080101u; // s_load_dwordx4 s[4:7], s[2:3], 0x50
    gta_413d22d_site[31] = 0xFA000050u;
    gta_413d22d_site[32] = 0xBF8CC07Fu;
    gta_413d22d_site.push_back(0xE0C84000u); // exact pc33 buffer_atomic_add, glc
    gta_413d22d_site.push_back(0x80010200u);
    gta_413d22d_site.push_back(0xBF810000u);
    ShaderResourceTable gta_413d22d_table;
    const std::vector<SrtUse> gta_413d22d_uses = add_compute_buffer_resources(
        gta_413d22d_table, gta_413d22d_site.data(), gta_413d22d_site.size(),
        gta_413d22d_seed, std::size(gta_413d22d_seed));
    ComputeShaderConfig gta_413d22d_config;
    gta_413d22d_config.user_sgprs.assign(std::begin(gta_413d22d_seed),
                                         std::end(gta_413d22d_seed));
    gta_413d22d_config.local_x = gta_413d22d_config.local_y =
        gta_413d22d_config.local_z = 1;
    const std::vector<uint32_t> gta_413d22d_spirv = recompile_compute(
        gta_413d22d_site.data(), gta_413d22d_site.size(), &gta_413d22d_table,
        gta_413d22d_config);
    const DescriptorValidationReport gta_413d22d_report =
        validate_spirv_descriptor_interface(gta_413d22d_spirv, &gta_413d22d_table, 0,
                                            SpirvShaderStage::Compute, false);
    CHECK(gta_413d22d_uses.size() == 1 && gta_413d22d_uses[0].zero_record_raw &&
              gta_413d22d_uses[0].key == 0x50u && gta_413d22d_uses[0].use_pc == 33u &&
              gta_413d22d_table.resources.size() == 1 &&
              is_zero_record_raw_buffer(gta_413d22d_table.resources[0]) &&
              gta_413d22d_table.resources[0].fetch_pc == 33u &&
              !gta_413d22d_spirv.empty() && gta_413d22d_report.ok() &&
              gta_413d22d_report.descriptors.empty(),
          "GTA V 0x413d22d pc33 exact synthetic empty packet recompiles binding-free");

    // 0x413d884 pc163 has no surviving SRT tag. The live fold knows base/stride/type and dword2,
    // but the unpublished value is inferred to be zero because every bounded nonzero value would
    // already materialize (an oversized value would remain rejected). Use the observed descriptor
    // fields plus a synthetic zero count to prove only the exact-PC path and exact OR packet.
    uint32_t gta_413d884_seed[24] = {};
    gta_413d884_seed[20] = 0xF8480000u;
    gta_413d884_seed[21] = 0x00040020u; // base 0x20f8480000, stride 4
    gta_413d884_seed[22] = 0u;          // inferred empty count, not direct live proof
    gta_413d884_seed[23] = 0x00016204u;
    std::vector<uint32_t> gta_413d884_site(163, 0xBF800000u);
    gta_413d884_site.push_back(0xE0E82000u); // exact pc163 buffer_atomic_or, idxen, glc=0
    gta_413d884_site.push_back(0x80051700u);
    gta_413d884_site.push_back(0xBF810000u);
    ShaderResourceTable gta_413d884_table;
    const std::vector<SrtUse> gta_413d884_uses = add_compute_buffer_resources(
        gta_413d884_table, gta_413d884_site.data(), gta_413d884_site.size(),
        gta_413d884_seed, std::size(gta_413d884_seed));
    ComputeShaderConfig gta_413d884_config;
    gta_413d884_config.user_sgprs.assign(std::begin(gta_413d884_seed),
                                         std::end(gta_413d884_seed));
    gta_413d884_config.local_x = gta_413d884_config.local_y =
        gta_413d884_config.local_z = 1;
    const std::vector<uint32_t> gta_413d884_spirv = recompile_compute(
        gta_413d884_site.data(), gta_413d884_site.size(), &gta_413d884_table,
        gta_413d884_config);
    const DescriptorValidationReport gta_413d884_report =
        validate_spirv_descriptor_interface(gta_413d884_spirv, &gta_413d884_table, 0,
                                            SpirvShaderStage::Compute, false);
    CHECK(gta_413d884_uses.size() == 1 && gta_413d884_uses[0].zero_record_raw &&
              gta_413d884_uses[0].key == 0xFFFFFFFFu &&
              gta_413d884_uses[0].use_pc == 163u &&
              gta_413d884_table.resources.size() == 1 &&
              is_zero_record_raw_buffer(gta_413d884_table.resources[0]) &&
              gta_413d884_table.resources[0].fetch_pc == 163u &&
              !gta_413d884_spirv.empty() && gta_413d884_report.ok() &&
              gta_413d884_report.descriptors.empty(),
          "GTA V 0x413d884 pc163 exact synthetic empty OR recompiles binding-free by PC");

    // GTA V 0x413e154 pc154 rebuilds the exact BUFFER_ATOMIC_SWAP_X2 V# at pcs142-146.
    // s13 is dispatch-dependent: the routed scene exercises 25, 2, 3, and 1 qword records.
    alignas(8) uint32_t gta_swap_x2_backing[50] = {0x11223344u, 0x55667788u};
    const uint64_t gta_swap_x2_base = reinterpret_cast<uint64_t>(gta_swap_x2_backing);
    uint32_t gta_swap_x2_seed[15] = {};
    gta_swap_x2_seed[8] = static_cast<uint32_t>(gta_swap_x2_base);
    gta_swap_x2_seed[9] = static_cast<uint32_t>(gta_swap_x2_base >> 32);
    gta_swap_x2_seed[13] = 25u;
    std::vector<uint32_t> gta_swap_x2_site(154, 0xBF800000u);
    gta_swap_x2_site[142] = 0x8801FF09u; // s_or_b32 s1, s9, 0x00080000
    gta_swap_x2_site[143] = 0x00080000u;
    gta_swap_x2_site[144] = 0xBE800308u; // s_mov_b32 s0, s8
    gta_swap_x2_site[145] = 0xBE82030Du; // s_mov_b32 s2, s13
    gta_swap_x2_site[146] = 0xBE8303FFu; // s_mov_b32 s3, 0x00016204
    gta_swap_x2_site[147] = 0x00016204u;
    gta_swap_x2_site.push_back(0xE1402000u);
    gta_swap_x2_site.push_back(0x80000913u);
    gta_swap_x2_site.push_back(0xBF810000u);
    ShaderResourceTable gta_swap_x2_table;
    const std::vector<SrtUse> gta_swap_x2_uses = add_compute_buffer_resources(
        gta_swap_x2_table, gta_swap_x2_site.data(), gta_swap_x2_site.size(),
        gta_swap_x2_seed, std::size(gta_swap_x2_seed));
    assign_convention_bindings(gta_swap_x2_table, 2);
    ComputeShaderConfig gta_swap_x2_config;
    gta_swap_x2_config.user_sgprs.assign(std::begin(gta_swap_x2_seed),
                                          std::end(gta_swap_x2_seed));
    gta_swap_x2_config.local_x = gta_swap_x2_config.local_y =
        gta_swap_x2_config.local_z = 1;
    gta_swap_x2_config.storage_buffer_int64_atomics = true;
    const std::vector<uint32_t> gta_swap_x2_spirv = recompile_compute(
        gta_swap_x2_site.data(), gta_swap_x2_site.size(), &gta_swap_x2_table,
        gta_swap_x2_config);
    const DescriptorValidationReport gta_swap_x2_report =
        validate_spirv_descriptor_interface(gta_swap_x2_spirv, &gta_swap_x2_table, 0,
                                            SpirvShaderStage::Compute, false);
    CHECK(gta_swap_x2_uses.size() == 1 &&
              gta_swap_x2_uses[0].use_pc == 154u &&
              gta_swap_x2_uses[0].required_size == 0u &&
              gta_swap_x2_uses[0].atomic_x2_record_count == 25u &&
              gta_swap_x2_table.resources.size() == 1 &&
              gta_swap_x2_table.resources[0].size == 200u &&
              gta_swap_x2_table.resources[0].stride == 8u &&
              gta_swap_x2_table.resources[0].fetch_pc == 154u &&
              gta_swap_x2_table.resources[0].atomic_x2_record_count == 25u &&
              !gta_swap_x2_spirv.empty() && gta_swap_x2_report.ok() &&
              gta_swap_x2_report.descriptors.size() == 1 &&
              gta_swap_x2_report.descriptors[0].required_bytes == 8u &&
              gta_swap_x2_report.descriptors[0].readable &&
              gta_swap_x2_report.descriptors[0].writable &&
              gta_swap_x2_report.descriptors[0].atomic_access,
          "GTA V 0x413e154 pc154 publishes one exact 25-record qword-atomic resource");
    uint32_t live_sized_atomic_x2_arms = 0;
    for (uint32_t record_count : {1u, 2u, 3u}) {
        uint32_t live_seed[15] = {};
        std::copy(std::begin(gta_swap_x2_seed), std::end(gta_swap_x2_seed), live_seed);
        live_seed[13] = record_count;
        ShaderResourceTable live_table;
        const std::vector<SrtUse> live_uses = add_compute_buffer_resources(
            live_table, gta_swap_x2_site.data(), gta_swap_x2_site.size(),
            live_seed, std::size(live_seed));
        live_sized_atomic_x2_arms += live_uses.size() == 1u &&
            live_uses[0].atomic_x2_record_count == record_count &&
            live_table.resources.size() == 1u &&
            live_table.resources[0].atomic_x2_record_count == record_count &&
            live_table.resources[0].size == record_count * 8u;
    }
    CHECK(live_sized_atomic_x2_arms == 3u,
          "pc145 carries GTA V's live 1/2/3-record dispatch bounds into qword atomics");
    ComputeShaderConfig gta_swap_x2_unsupported_config = gta_swap_x2_config;
    gta_swap_x2_unsupported_config.storage_buffer_int64_atomics = false;
    CHECK(recompile_compute(gta_swap_x2_site.data(), gta_swap_x2_site.size(),
                            &gta_swap_x2_table, gta_swap_x2_unsupported_config).empty(),
          "GTA V swap_x2 remains fail-visible without an enabled Vulkan int64-atomic contract");

    // The same rebuilt V# reaches pc172's BUFFER_ATOMIC_OR_X2. It is the next chronological live
    // rejection after pc154 and must publish the same semantic proof at its own consumer PC.
    std::vector<uint32_t> gta_or_x2_site(172, 0xBF800000u);
    std::copy_n(gta_swap_x2_site.begin(), 154, gta_or_x2_site.begin());
    gta_or_x2_site[154] = 0xBF800000u;
    gta_or_x2_site[155] = 0xBF800000u;
    gta_or_x2_site.push_back(0xE1686000u);
    gta_or_x2_site.push_back(0x80000913u);
    gta_or_x2_site.push_back(0xBF810000u);
    ShaderResourceTable gta_or_x2_table;
    const std::vector<SrtUse> gta_or_x2_uses = add_compute_buffer_resources(
        gta_or_x2_table, gta_or_x2_site.data(), gta_or_x2_site.size(),
        gta_swap_x2_seed, std::size(gta_swap_x2_seed));
    CHECK(gta_or_x2_uses.size() == 1 && gta_or_x2_uses[0].use_pc == 172u &&
              gta_or_x2_uses[0].atomic_x2_record_count == 25u &&
              gta_or_x2_table.resources.size() == 1 &&
              gta_or_x2_table.resources[0].size == 200u &&
              gta_or_x2_table.resources[0].stride == 8u &&
              gta_or_x2_table.resources[0].fetch_pc == 172u &&
              gta_or_x2_table.resources[0].atomic_x2_record_count == 25u,
          "GTA V 0x413e154 pc172 publishes the same exact proof for atomic_or_x2");

    uint32_t rejected_atomic_x2_shapes = 0;
    const auto atomic_x2_rejected = [&](const uint32_t user_sgprs[15], uint32_t word0,
                                        uint32_t word1) {
        std::vector<uint32_t> code = gta_swap_x2_site;
        code[154] = word0;
        code[155] = word1;
        std::vector<SrtUse> uses;
        resolve_dynamic_fetch(code.data(), code.size(), user_sgprs, 15, 0, &uses);
        return uses.empty();
    };
    uint32_t mutated_swap_x2_seed[15];
    std::copy(std::begin(gta_swap_x2_seed), std::end(gta_swap_x2_seed),
              mutated_swap_x2_seed);
    mutated_swap_x2_seed[9] |= 4u << 16; // pc142 rebuilds stride 12 instead of 8
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        mutated_swap_x2_seed, 0xE1402000u, 0x80000913u);
    std::copy(std::begin(gta_swap_x2_seed), std::end(gta_swap_x2_seed),
              mutated_swap_x2_seed);
    mutated_swap_x2_seed[13] = 0u;
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        mutated_swap_x2_seed, 0xE1402000u, 0x80000913u);
    std::copy(std::begin(gta_swap_x2_seed), std::end(gta_swap_x2_seed),
              mutated_swap_x2_seed);
    gta_swap_x2_site[147] |= 1u << 28; // OOB_SELECT=1 at the exact descriptor-build site
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1402000u, 0x80000913u);
    gta_swap_x2_site[147] &= ~(1u << 28);
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1402004u, 0x80000913u); // instruction offset 4
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1403000u, 0x80000913u); // OFFEN as well as IDXEN
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1402000u, 0x00000913u); // register SOFFSET
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1402000u, 0x80400913u); // SLC
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE140A000u, 0x80000913u); // DLC
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1412000u, 0x80000913u); // LDS transfer
    std::copy(std::begin(gta_swap_x2_seed), std::end(gta_swap_x2_seed),
              mutated_swap_x2_seed);
    mutated_swap_x2_seed[9] |= 0x80000000u; // SWIZZLE_ENABLE
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        mutated_swap_x2_seed, 0xE1402000u, 0x80000913u);
    gta_swap_x2_site[147] |= 1u << 23; // ADD_TID_ENABLE
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1402000u, 0x80000913u);
    gta_swap_x2_site[147] &= ~(1u << 23);
    gta_swap_x2_site[147] |= 1u << 21; // INDEX_STRIDE
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1402000u, 0x80000913u);
    gta_swap_x2_site[147] &= ~(1u << 21);
    gta_swap_x2_site[147] |= 1u << 24; // RESOURCE_LEVEL
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1402000u, 0x80000913u);
    gta_swap_x2_site[147] &= ~(1u << 24);
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1422000u, 0x80000913u); // reserved dword0 bit 17
    // A separately dumped OR_X2 uses offset 24 without IDXEN; exact live proof must not spread to it.
    rejected_atomic_x2_shapes += atomic_x2_rejected(
        gta_swap_x2_seed, 0xE1680018u, 0x80000000u);
    CHECK(rejected_atomic_x2_shapes == 15u,
          "qword-atomic discovery rejects every unproved stride/count/OOB/address sibling shape");

    // These neighboring encodings still need distinct operands or wrap/conditional semantics. They
    // must not acquire a resource merely because their descriptor is concrete.
    const uint32_t unsupported_atomic_dw0[] = {
        0xE0C40000u, // cmp-swap
        0xE0D00000u, // csub
        0xE0F00000u, // inc
        0xE0F40000u, // dec
    };
    uint32_t unsupported_atomic_uses = 0;
    uint32_t unsupported_atomic_resources = 0;
    uint32_t unsupported_atomic_recompiles = 0;
    uint32_t unsupported_empty_atomic_uses = 0;
    uint32_t unsupported_empty_atomic_resources = 0;
    for (uint32_t dw0 : unsupported_atomic_dw0) {
        const uint32_t code[] = { dw0, 0x80000000u, 0xBF810000u };
        std::vector<SrtUse> uses;
        resolve_dynamic_fetch(code, std::size(code), atomic_seed, std::size(atomic_seed), 0,
                              &uses);
        unsupported_atomic_uses += !uses.empty();
        ShaderResourceTable table;
        add_compute_buffer_resources(table, code, std::size(code), atomic_seed,
                                     std::size(atomic_seed));
        unsupported_atomic_resources += !table.resources.empty();
        unsupported_atomic_recompiles +=
            !recompile_compute(code, std::size(code), &table, atomic_config).empty();
        std::vector<SrtUse> empty_uses;
        resolve_dynamic_fetch(code, std::size(code), empty_atomic_seed,
                              std::size(empty_atomic_seed), 0, &empty_uses);
        unsupported_empty_atomic_uses += !empty_uses.empty();
        ShaderResourceTable empty_table;
        add_compute_buffer_resources(empty_table, code, std::size(code), empty_atomic_seed,
                                     std::size(empty_atomic_seed));
        unsupported_empty_atomic_resources += !empty_table.resources.empty();
    }
    CHECK(unsupported_atomic_uses == 0 && unsupported_atomic_resources == 0 &&
              unsupported_atomic_recompiles == 0 && unsupported_empty_atomic_uses == 0 &&
              unsupported_empty_atomic_resources == 0,
          "unsupported cmp-swap/csub/inc/dec atomics remain fail-closed even when empty");

    // Terminator 2D supplies descriptor dwords separately, then reassembles its destination V# in
    // s[0:3] with four scalar moves immediately before format stores. This has no SRT-load tag and
    // the destination range no longer equals its entry user data, but the four consecutive seed
    // origins are exact provenance. Arithmetic or a shuffled word must still fail closed.
    static uint32_t moved_store_output[16];
    const uint64_t moved_store_base = (uint64_t)(uintptr_t)moved_store_output;
    uint32_t moved_store_seed[9] = {};
    moved_store_seed[5] = (uint32_t)moved_store_base;
    moved_store_seed[6] = (uint32_t)(moved_store_base >> 32) | (4u << 16);
    moved_store_seed[7] = 16;
    moved_store_seed[8] = 0x10005004u;
    const uint32_t moved_direct_store[] = {
        0xBE800305u, 0xBE810306u, 0xBE820307u, 0xBE830308u,
        0xE0142000u, 0x80000000u, // buffer_store_format_x v0, v0, s[0:3]
        0xBF810000u,
    };
    ShaderResourceTable moved_store_table;
    const std::vector<SrtUse> moved_store_uses = add_compute_buffer_resources(
        moved_store_table, moved_direct_store, std::size(moved_direct_store),
        moved_store_seed, std::size(moved_store_seed));
    assign_convention_bindings(moved_store_table, 2);
    CHECK(moved_store_uses.size() == 1 && moved_store_uses[0].use_pc == 4 &&
              moved_store_table.resources.size() == 1 &&
              moved_store_table.resources[0].gpu_addr == moved_store_base &&
              moved_store_table.resources[0].fetch_pc == 4,
          "moved consecutive entry dwords publish the exact format-store V#");
    const uint32_t moved_shuffled_store[] = {
        0xBE800305u, 0xBE810306u, 0xBE820308u, 0xBE830307u,
        0xE0142000u, 0x80000000u,
        0xBF810000u,
    };
    ShaderResourceTable moved_shuffled_table;
    add_compute_buffer_resources(moved_shuffled_table, moved_shuffled_store,
                                 std::size(moved_shuffled_store), moved_store_seed,
                                 std::size(moved_store_seed));
    CHECK(moved_shuffled_table.resources.empty(),
          "shuffled entry dwords do not fabricate moved-descriptor provenance");

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

    // Some dispatch tables first normalize the high address dword with s_pack_ll_b32_b16. Keep the
    // low 16 address bits from s9 and clear the aperture bits in its upper half before interpreting
    // s[8:9] as a scalar-load base. This exercises the production instruction walk and mapped-memory
    // read, not a standalone ALU helper.
    const uint32_t packed_tagged_k5[] = {
        0x99098009u,                // s_pack_ll_b32_b16 s9, s9, 0
        0xF40C0304u, 0xFA000040u,   // s_load_dwordx8 s[12:19], s[8:9], 0x40
        0xF4080504u, 0xFA000080u,   // s_load_dwordx4 s[20:23], s[8:9], 0x80
        0xF0800F08u, 0x00A30000u,   // image_sample v[0:3], v[0:1], s[12:19], s[20:23]
        0xBF810000u,
    };
    std::vector<SrtUse> packed_tagged_uses;
    resolve_dynamic_fetch(packed_tagged_k5, std::size(packed_tagged_k5), tagged_seed5, 2, 8,
                          &packed_tagged_uses);
    bool packed_tagged_tex = false;
    for (const auto& u : packed_tagged_uses)
        if (u.kind == 0 && u.key == 0x40 && u.t8[0] == table[16]) packed_tagged_tex = true;
    CHECK(packed_tagged_tex,
          "s_pack_ll-normalized scalar pointer resolves its mapped descriptor table");

    std::array<uint32_t, std::size(packed_tagged_k5)> packed_zero_high{};
    std::copy(std::begin(packed_tagged_k5), std::end(packed_tagged_k5),
              packed_zero_high.begin());
    packed_zero_high[0] = 0x9909807Cu; // same pack site, but untracked m0 supplies its low half
    std::vector<SrtUse> packed_zero_high_uses;
    resolve_dynamic_fetch(packed_zero_high.data(), packed_zero_high.size(), tagged_seed5, 2, 8,
                          &packed_zero_high_uses);
    CHECK(packed_zero_high_uses.empty(),
          "an untracked low-half source at the pack site does not fabricate a descriptor mapping");

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

    // Kernel 6 (#2132): BRANCH-EXCLUSIVE WRITES. The fold walks straight-line, so a shader that
    // guards its descriptor setup with a branch had the not-taken side's register writes carried
    // into a block the hardware only ever enters FROM the branch. CrossWorlds' NGG vertex stage is
    // this shape: `s_cbranch_vccz T`, then a `s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60` that
    // overwrites the stage's two seeded direct pointers with float uniforms, then an unconditional
    // `s_branch` OVER T — so T has no fall-through predecessor and exactly one branch reaches it.
    // At T the shader dereferences s[16:17]; the walk had replaced it with 1.0f/0, which read as a
    // "null bindless-table pointer" that no writer ever wrote.
    //
    // Layout mirrors the guest exactly: a 160-byte stride-16 constant buffer whose dwords at +0x60
    // are 1.0f,0,0,0, and a descriptor table reached only through the seeded pointer at dw8.
    alignas(16) uint32_t k6_vbuf[32] = {};                  // 8 records x 16 bytes
    alignas(16) uint32_t k6_cbuf[40] = {};                  // 10 records x 16 bytes = 160
    k6_cbuf[24] = 0x3F800000u;                              // +0x60 = 1.0f  (lands in s16)
    k6_cbuf[25] = 0u; k6_cbuf[26] = 0u; k6_cbuf[27] = 0u;   // +0x64..+0x6c  (s17, s18, s19)
    const uint64_t k6_vbuf_addr = (uint64_t)(uintptr_t)k6_vbuf;
    alignas(16) uint32_t k6_table[4] = {
        (uint32_t)k6_vbuf_addr,
        (uint32_t)((k6_vbuf_addr >> 32) & 0xFFFFu) | (16u << 16),   // stride 16
        8u, 0u,
    };
    const uint64_t k6_cbuf_addr = (uint64_t)(uintptr_t)k6_cbuf;
    const uint64_t k6_table_addr = (uint64_t)(uintptr_t)k6_table;
    uint32_t seed6[12] = {
        (uint32_t)k6_cbuf_addr,                                          // dw0..3 -> s8..s11: cbuf V#
        (uint32_t)((k6_cbuf_addr >> 32) & 0xFFFFu) | (16u << 16),        //   stride 16
        10u, 0u,                                                         //   num_records 10 (160 B)
        0u, 0u, 0u, 0u,                                                  // dw4..7 -> s12..s15
        (uint32_t)k6_table_addr,                                         // dw8..9 -> s16:s17 table ptr
        (uint32_t)(k6_table_addr >> 32),
        0u, 0u,                                                          // dw10..11 -> s18:s19
    };
    const uint32_t k6[] = {
        0xBF860003u,                // pc=0  s_cbranch_vccz 3      -> T (pc=4)
        0xF4300404u, 0xFA000060u,   // pc=1  s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60
        0xBF820004u,                // pc=3  s_branch 4            -> pc=8 (so T has no fall-through)
        0xF4080108u, 0xFA000000u,   // pc=4  T: s_load_dwordx4 s[4:7], s[16:17], 0x0
        0xE0002000u, 0x80010100u,   // pc=6  buffer_load_format_x v1, v0, s[4:7], 0 idxen
        0xBF810000u,                // pc=8  s_endpgm
    };
    auto k6_fetch = resolve_dynamic_fetch(k6, sizeof(k6)/sizeof(k6[0]), seed6, 12, 8);
    CHECK(k6_fetch.size() == 1 &&
              k6_fetch[0].desc.base == ((uint64_t)(uintptr_t)k6_vbuf & 0xFFFFFFFFFFFFull),
          "#2132: a branch-exclusive write does not clobber the seeded pointer its target reads");

    // Counter-arm — the same kernel with ONE dword changed: the unconditional s_branch at pc=3
    // becomes s_nop, so T now HAS a fall-through predecessor and the rule must not fire. Without
    // this arm the assertion above would also pass if the fold simply re-seeded every register at
    // every block, which is a different (and wrong) behaviour.
    uint32_t k6_ft[sizeof(k6)/sizeof(k6[0])];
    std::copy(std::begin(k6), std::end(k6), std::begin(k6_ft));
    k6_ft[3] = 0xBF800000u;         // s_nop 0
    clear_shader_decode_cache();
    auto k6_ft_fetch = resolve_dynamic_fetch(k6_ft, sizeof(k6_ft)/sizeof(k6_ft[0]), seed6, 12, 8);
    CHECK(k6_ft_fetch.empty() ||
              k6_ft_fetch[0].desc.base != ((uint64_t)(uintptr_t)k6_vbuf & 0xFFFFFFFFFFFFull),
          "#2132 counter-arm: a target with a fall-through predecessor keeps the walked value");

    // Kernel 7 (#2202 B1): the fold walks a COMPACTED stream (`retain_fold_instructions`) that drops
    // EXP among other formats while preserving PCs, so "the previous element of `ins`" is not "the
    // instruction before it". Here an `exp` sits physically between the `s_branch` at pc=3 and the
    // target at pc=6, giving the target a real fall-through predecessor — but the previous RETAINED
    // instruction is still that `s_branch`. Without the physical-adjacency conjunct the rule fires
    // and *installs a known wrong value*, which is worse than the bug it fixes. It must decline.
    const uint32_t k7[] = {
        0xBF860005u,                // pc=0  s_cbranch_vccz 5     -> pc=6
        0xF4300404u, 0xFA000060u,   // pc=1  s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60
        0xBF820006u,                // pc=3  s_branch 6           -> pc=10
        0xF80000CFu, 0x00000000u,   // pc=4  exp pos0 v0,v0,v0,v0  (dropped from the fold stream)
        0xF4080108u, 0xFA000000u,   // pc=6  target: s_load_dwordx4 s[4:7], s[16:17], 0x0
        0xE0002000u, 0x80010100u,   // pc=8  buffer_load_format_x v1, v0, s[4:7], 0 idxen
        0xBF810000u,                // pc=10 s_endpgm
    };
    clear_shader_decode_cache();
    auto k7_fetch = resolve_dynamic_fetch(k7, sizeof(k7)/sizeof(k7[0]), seed6, 12, 8);
    CHECK(k7_fetch.empty() ||
              k7_fetch[0].desc.base != ((uint64_t)(uintptr_t)k6_vbuf & 0xFFFFFFFFFFFFull),
          "#2202 B1: a dropped instruction physically between the branch and its target blocks the rule");

    // Kernel 8 (#2202 B2): a predecessor tally that counts only FORWARD branches is not a tally. Two
    // edges reach pc=4 — the forward `s_cbranch_vccz` at pc=0 and the backward `s_branch` at pc=8 —
    // so the target has two predecessors and the rule must decline. The forward branch and the
    // physical adjacency both still hold, so this isolates the direction filter and nothing else.
    const uint32_t k8[] = {
        0xBF860003u,                // pc=0  s_cbranch_vccz 3     -> pc=4
        0xF4300404u, 0xFA000060u,   // pc=1  s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60
        0xBF820004u,                // pc=3  s_branch 4           -> pc=8
        0xF4080108u, 0xFA000000u,   // pc=4  target: s_load_dwordx4 s[4:7], s[16:17], 0x0
        0xE0002000u, 0x80010100u,   // pc=6  buffer_load_format_x v1, v0, s[4:7], 0 idxen
        0xBF82FFFBu,                // pc=8  s_branch -5          -> pc=4  (second predecessor)
        0xBF810000u,                // pc=9  s_endpgm
    };
    clear_shader_decode_cache();
    auto k8_fetch = resolve_dynamic_fetch(k8, sizeof(k8)/sizeof(k8[0]), seed6, 12, 8);
    CHECK(k8_fetch.empty() ||
              k8_fetch[0].desc.base != ((uint64_t)(uintptr_t)k6_vbuf & 0xFFFFFFFFFFFFull),
          "#2202 B2: a backward edge into the target counts as a predecessor and blocks the rule");

    // Kernel 9: the positive control for both of the above — identical to k7 with the `exp` removed,
    // so the `s_branch` at pc=3 IS physically adjacent to the target at pc=4 and nothing else reaches
    // it. Without this arm, k7 and k8 would both pass on a rule that never fires at all.
    const uint32_t k9[] = {
        0xBF860003u,                // pc=0  s_cbranch_vccz 3     -> pc=4
        0xF4300404u, 0xFA000060u,   // pc=1  s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60
        0xBF820004u,                // pc=3  s_branch 4           -> pc=8
        0xF4080108u, 0xFA000000u,   // pc=4  target: s_load_dwordx4 s[4:7], s[16:17], 0x0
        0xE0002000u, 0x80010100u,   // pc=6  buffer_load_format_x v1, v0, s[4:7], 0 idxen
        0xBF810000u,                // pc=8  s_endpgm
    };
    clear_shader_decode_cache();
    auto k9_fetch = resolve_dynamic_fetch(k9, sizeof(k9)/sizeof(k9[0]), seed6, 12, 8);
    CHECK(k9_fetch.size() == 1 &&
              k9_fetch[0].desc.base == ((uint64_t)(uintptr_t)k6_vbuf & 0xFFFFFFFFFFFFull),
          "#2202: with the region's only entry a physically adjacent branch, the rule still fires");

    // Kernel 10 (#2202 B2, second half): indirect control flow makes the CFG unrepresentable by any
    // scan over SOPP displacements, so the rule declines rather than firing on a partial edge set.
    // Same shape as k9 plus one `s_swappc_b64`.
    const uint32_t k10[] = {
        0xBE862104u,                // pc=0  s_swappc_b64 s[6:7], s[4:5]  -- the only difference
        0xBF860003u,                // pc=1  s_cbranch_vccz 3     -> pc=5
        0xF4300404u, 0xFA000060u,   // pc=2  s_buffer_load_dwordx16 s[16:31], s[8:11], 0x60
        0xBF820004u,                // pc=4  s_branch 4           -> pc=9
        0xF4080108u, 0xFA000000u,   // pc=5  target: s_load_dwordx4 s[4:7], s[16:17], 0x0
        0xE0002000u, 0x80010100u,   // pc=7  buffer_load_format_x v1, v0, s[4:7], 0 idxen
        0xBF810000u,                // pc=9  s_endpgm
    };
    clear_shader_decode_cache();
    auto k10_fetch = resolve_dynamic_fetch(k10, sizeof(k10)/sizeof(k10[0]), seed6, 12, 8);
    CHECK(k10_fetch.empty() ||
              k10_fetch[0].desc.base != ((uint64_t)(uintptr_t)k6_vbuf & 0xFFFFFFFFFFFFull),
          "#2202: an indirect control transfer disables the rule for the whole program");

    // Kernel 11 (#2202 B3): TWO qualifying targets, where the second's sole-predecessor branch IS
    // the first target. `pc=4` is both — it is entered only from `pc=0` (so it is a target), and it
    // is the only branch reaching `pc=8` (so it is a predecessor). If the walk saves the state at
    // `pc=4` BEFORE restoring it, the snapshot handed to `pc=8` is the pre-restore chimera and the
    // clobbered s[16:17] is reinstated one block later — the #2132 defect reproduced by the fix's own
    // bookkeeping. Neither region is on the path that reaches `pc=8`, so the seeded table pointer
    // must survive both.
    //
    // None of k6-k10 has two qualifying targets, which is exactly why this shape got through the
    // first round: an absent case cannot fail.
    const uint32_t k11[] = {
        0xBF860003u,                // pc=0  s_cbranch_vccz 3     -> pc=4
        0xF4300404u, 0xFA000060u,   // pc=1  region 1: clobbers s16..s31 with cbuf floats
        0xBF820008u,                // pc=3  s_branch 8           -> pc=12 (so pc=4 has no fall-through)
        0xBF840003u,                // pc=4  TARGET 1, and the only branch into pc=8: s_cbranch_scc0 3
        0xF4300404u, 0xFA000060u,   // pc=5  region 2: clobbers s16..s31 again
        0xBF820004u,                // pc=7  s_branch 4           -> pc=12 (so pc=8 has no fall-through)
        0xF4080108u, 0xFA000000u,   // pc=8  TARGET 2: s_load_dwordx4 s[4:7], s[16:17], 0x0
        0xE0002000u, 0x80010100u,   // pc=10 buffer_load_format_x v1, v0, s[4:7], 0 idxen
        0xBF810000u,                // pc=12 s_endpgm
    };
    clear_shader_decode_cache();
    auto k11_fetch = resolve_dynamic_fetch(k11, sizeof(k11)/sizeof(k11[0]), seed6, 12, 8);
    CHECK(k11_fetch.size() == 1 &&
              k11_fetch[0].desc.base == ((uint64_t)(uintptr_t)k6_vbuf & 0xFFFFFFFFFFFFull),
          "#2202 B3: a target that is itself the next target's predecessor saves the RESTORED state");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
