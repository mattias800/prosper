// test_rdna2_spirv_struct -- structural checks for RDNA2->SPIR-V output that do not require Vulkan.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include <cstdint>
#include <cstdio>
#include <unordered_map>
#include <vector>

using namespace prosper::gpu;

namespace {

enum : uint32_t {
    OpTypeVoid = 19,
    OpTypeBool = 20,
    OpTypeInt = 21,
    OpTypeFloat = 22,
    OpTypeVector = 23,
    OpTypeArray = 28,
    OpTypeRuntimeArray = 29,
    OpTypeStruct = 30,
    OpTypePointer = 32,
    OpTypeFunction = 33,
};

bool is_type_decl(uint32_t op) {
    switch (op) {
        case OpTypeVoid:
        case OpTypeBool:
        case OpTypeInt:
        case OpTypeFloat:
        case OpTypeVector:
        case OpTypeArray:
        case OpTypeRuntimeArray:
        case OpTypeStruct:
        case OpTypePointer:
        case OpTypeFunction:
            return true;
        default:
            return false;
    }
}

bool type_result_ids_are_nonzero(const std::vector<uint32_t>& spv, uint32_t* bad_op) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu;
        uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (is_type_decl(op) && (wc < 2 || spv[i + 1] == 0)) {
            if (bad_op) *bad_op = op;
            return false;
        }
        i += wc;
    }
    return true;
}

bool has_signed_i32_type(const std::vector<uint32_t>& spv) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu;
        uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpTypeInt && wc == 4 && spv[i + 1] != 0 && spv[i + 2] == 32 && spv[i + 3] == 1)
            return true;
        i += wc;
    }
    return false;
}

// Whether the module contains an instruction with the given opcode.
bool has_opcode(const std::vector<uint32_t>& spv, uint32_t opcode) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu;
        uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == opcode) return true;
        i += wc;
    }
    return false;
}

bool binary_id_operands_are_nonzero(const std::vector<uint32_t>& spv, uint32_t opcode) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t word = spv[i];
        const uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        // Integer binary operations are {result type, result id, operand 1 id, operand 2 id}.
        if (op == opcode && (wc != 5 || !spv[i + 1] || !spv[i + 2] ||
                             !spv[i + 3] || !spv[i + 4]))
            return false;
        i += wc;
    }
    return true;
}

bool phi_ids_are_nonzero(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpPhi = 245;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t word = spv[i];
        const uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpPhi) {
            if (wc < 5 || ((wc - 3) & 1u) != 0) return false;
            for (uint32_t operand = 1; operand < wc; ++operand)
                if (spv[i + operand] == 0) return false;
        }
        i += wc;
    }
    return true;
}

// Whether the module contains OpDecorate (71) with the given decoration (word[i+2]).
bool has_decoration(const std::vector<uint32_t>& spv, uint32_t decoration) {
    enum : uint32_t { OpDecorateL = 71 };
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpDecorateL && wc >= 3 && spv[i + 2] == decoration) return true;
        i += wc;
    }
    return false;
}

bool has_builtin(const std::vector<uint32_t>& spv, uint32_t builtin) {
    enum : uint32_t { OpDecorateL = 71, DecBuiltIn = 11 };
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == OpDecorateL && wc >= 4 && spv[i + 2] == DecBuiltIn && spv[i + 3] == builtin)
            return true;
        i += wc;
    }
    return false;
}

// The largest OpTypeArray length (resolved through its OpConstant) in the module — for LDS sizing
// (#130), the Workgroup LDS array is the biggest array the compute shell declares.
uint32_t max_array_length(const std::vector<uint32_t>& spv) {
    enum : uint32_t { OpTypeArrayL = 28, OpConstantL = 43 };
    if (spv.size() < 5) return 0;
    std::unordered_map<uint32_t, uint32_t> const_val;   // result id -> u32 constant value
    uint32_t best = 0;
    for (size_t i = 5; i < spv.size();) {
        uint32_t word = spv[i];
        uint32_t op = word & 0xffffu, wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) break;
        if (op == OpConstantL && wc >= 4) const_val[spv[i + 2]] = spv[i + 3];   // {type,result,value}
        if (op == OpTypeArrayL && wc >= 4) {                                    // {result,elem,length-id}
            auto it = const_val.find(spv[i + 3]);
            if (it != const_val.end() && it->second > best) best = it->second;
        }
        i += wc;
    }
    return best;
}

} // namespace

int main() {
    printf("== test_rdna2_spirv_struct ==\n");

    // Uses signed convert/min/max/ashr, which requires a valid signed i32 type declaration.
    const uint32_t signed_kernel[] = {
        0x7E001100u, 0x7E021101u, 0x7E041102u, 0x22060300u, 0x24080300u,
        0x4C060704u, 0x30060702u, 0x7E000B03u, 0xBF810000u,
    };
    std::vector<uint32_t> spv = recompile_valu(signed_kernel, sizeof(signed_kernel) / sizeof(signed_kernel[0]), 3, 0);
    if (spv.empty()) {
        printf("  [FAIL] signed kernel did not recompile\n");
        return 1;
    }

    uint32_t bad_op = 0;
    if (!type_result_ids_are_nonzero(spv, &bad_op)) {
        printf("  [FAIL] SPIR-V type declaration has invalid result id (op=%u)\n", bad_op);
        return 1;
    }
    printf("  [ok]   SPIR-V type declaration result ids are nonzero\n");

    if (!has_signed_i32_type(spv)) {
        printf("  [FAIL] signed kernel SPIR-V lacks a nonzero signed i32 type\n");
        return 1;
    }
    printf("  [ok]   signed kernel SPIR-V declares signed i32 with a nonzero id\n");

    // Fixed-offset private spill/fill must also produce structurally valid graphics-stage modules.
    // This exercises Function-variable placement in the fragment and vertex shells, independently
    // of the compute execution tests.
    const uint32_t scratch_fragment[] = {
        0xdc704010u, 0x00000000u, // scratch_store_dword off, v0, s0 offset:16
        0x7e000280u,              // v_mov_b32 v0, 0
        0xdc304010u, 0x00000000u, // scratch_load_dword v0, off, s0 offset:16
        0xf800000fu, 0x00000000u, // exp mrt0 v0, v0, v0, v0
        0xBF810000u,
    };
    const auto scratch_fragment_spv = recompile_fragment(
        scratch_fragment, sizeof(scratch_fragment) / sizeof(scratch_fragment[0]));
    uint32_t scratch_fragment_bad_op = 0;
    if (scratch_fragment_spv.empty() || !has_opcode(scratch_fragment_spv, 28) ||
        !type_result_ids_are_nonzero(scratch_fragment_spv, &scratch_fragment_bad_op) ||
        !phi_ids_are_nonzero(scratch_fragment_spv)) {
        printf("  [FAIL] fragment private spill/fill emitted invalid SPIR-V (op=%u)\n",
               scratch_fragment_bad_op);
        return 1;
    }
    printf("  [ok]   fragment private spill/fill emits structurally valid Function storage\n");

    const uint32_t scratch_vertex[] = {
        0xdc704010u, 0x00000000u, // scratch_store_dword off, v0, s0 offset:16
        0x7e000280u,              // v_mov_b32 v0, 0
        0xdc304010u, 0x00000000u, // scratch_load_dword v0, off, s0 offset:16
        0xf80008cfu, 0x00000000u, // exp pos0 v0, v0, v0, v0
        0xBF810000u,
    };
    const auto scratch_vertex_spv = recompile_vertex(
        scratch_vertex, sizeof(scratch_vertex) / sizeof(scratch_vertex[0]));
    uint32_t scratch_vertex_bad_op = 0;
    if (scratch_vertex_spv.empty() || !has_opcode(scratch_vertex_spv, 28) ||
        !type_result_ids_are_nonzero(scratch_vertex_spv, &scratch_vertex_bad_op) ||
        !phi_ids_are_nonzero(scratch_vertex_spv)) {
        printf("  [FAIL] vertex private spill/fill emitted invalid SPIR-V (op=%u)\n",
               scratch_vertex_bad_op);
        return 1;
    }
    printf("  [ok]   vertex private spill/fill emits structurally valid Function storage\n");

    // NGG wave packing writes EXEC through s_lshr_b64 before later structured control flow. The
    // instruction is mask-domain only; inserting rs.sreg[EXEC] left an SSA id 0 that a later merge
    // emitted as an OpPhi input. NVIDIA's Windows driver faults in vkCreateGraphicsPipelines on that
    // invalid module, so guard the exact wave-pack -> forward-if shape independently of a Vulkan driver.
    const uint32_t ngg_exec_if[] = {
        0x93EAFF03u, 0x00080008u, 0x876BFF03u, 0x000000FFu, 0x8F6A8C6Au,
        0x887C6A6Bu, 0xBF900009u, 0x906A8803u, 0x81EA6A80u, 0x90FE6AC1u,
        0xF8000941u, 0x00000000u, 0x81EA0380u, 0x90FE6AC1u,
        0xBE80246Au,                         // s_and_saveexec_b64 s[0:1], vcc
        0xBF880001u,                         // s_cbranch_execz +1
        0x8AFE7E00u,                         // s_andn2_b64 exec, s[0:1], exec
        0xBEFE0400u,                         // s_mov_b64 exec, s[0:1]
        0x34040A81u, 0x36060AC2u, 0x7E000280u, 0x7E0202F2u, 0x36040482u,
        0x4A0606C1u, 0x4A0404C1u, 0x7E060B03u, 0x7E040B02u,
        0xF80008CFu, 0x01000302u, 0xBF810000u,
    };
    const auto ngg_exec_if_spv = recompile_vertex(
        ngg_exec_if, sizeof(ngg_exec_if) / sizeof(ngg_exec_if[0]));
    if (ngg_exec_if_spv.empty() || !phi_ids_are_nonzero(ngg_exec_if_spv)) {
        printf("  [FAIL] NGG EXEC wave-pack control flow emitted a zero-id OpPhi\n");
        return 1;
    }
    printf("  [ok]   NGG EXEC wave-pack control flow emits only valid nonzero OpPhi ids\n");

    // In a one-lane NGG vertex invocation, s_bcnt1_i32_b64 of a wave mask is the integer value of
    // that lane's Boolean bit (0/1). Astro Bot uses this in its final primitive packing arithmetic.
    const uint32_t ngg_mask_count[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0xBEEA04C1u,                         // s_mov_b64 vcc, -1
        0xBE80106Au,                         // s_bcnt1_i32_b64 s0, vcc
        0x7E000C00u,                         // v_cvt_f32_u32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_mask_count_spv = recompile_vertex(
        ngg_mask_count, sizeof(ngg_mask_count) / sizeof(ngg_mask_count[0]));
    if (ngg_mask_count_spv.empty() || !phi_ids_are_nonzero(ngg_mask_count_spv)) {
        printf("  [FAIL] NGG one-lane mask population count did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   NGG one-lane mask population count produces valid SPIR-V\n");

    // The terminal NGG output gate is a compacted-vertex ownership test. Unlike a general vertex
    // CMPX/export (tested below), this exact output-only branch may retain the narrowed EXEC while
    // exporting from the one-lane Vulkan representation.
    const uint32_t ngg_output_gate[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x7DA80300u,                         // v_cmpx_* (narrows EXEC)
        0xBF880002u,                         // s_cbranch_execz -> s_endpgm
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_output_gate_spv = recompile_vertex(
        ngg_output_gate, sizeof(ngg_output_gate) / sizeof(ngg_output_gate[0]));
    if (ngg_output_gate_spv.empty() || !phi_ids_are_nonzero(ngg_output_gate_spv)) {
        printf("  [FAIL] terminal NGG compacted-vertex output gate did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   terminal NGG compacted-vertex output gate produces valid SPIR-V\n");

    // NGG uses small inline B64 masks (1/3/15) while packing its final partial wave. Vertex SPIR-V
    // has no LocalInvocationIndex; the one-lane model must read bit zero directly rather than emit an
    // integer operation with the compute-only built-in's absent (zero) ID.
    const uint32_t ngg_inline_mask[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x92EA8081u,                         // s_bfm_b64 vcc, 1, 0 (lane-zero mask)
        0xBEFE0481u,                         // s_mov_b64 exec, 1
        0xBEFE04C1u,                         // s_mov_b64 exec, -1 (restore before export)
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_inline_mask_spv = recompile_vertex(
        ngg_inline_mask, sizeof(ngg_inline_mask) / sizeof(ngg_inline_mask[0]));
    constexpr uint32_t OpBitwiseAnd = 199;
    if (ngg_inline_mask_spv.empty() ||
        !binary_id_operands_are_nonzero(ngg_inline_mask_spv, OpBitwiseAnd)) {
        printf("  [FAIL] NGG inline mask referenced an absent vertex local-invocation id\n");
        return 1;
    }
    printf("  [ok]   NGG inline mask uses the modeled lane-zero bit without invalid ids\n");

    // The same wave shortcut is not valid for an ordinary vertex shader: without the exact NGG
    // allocation message there is no proof that a Vulkan invocation represents guest lane zero.
    const uint32_t ordinary_vertex_mask_count[] = {
        0xBEEA04C1u,                         // s_mov_b64 vcc, -1
        0xBE80106Au,                         // s_bcnt1_i32_b64 s0, vcc
        0x7E000C00u,                         // v_cvt_f32_u32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,
        0xBF810000u,
    };
    if (!recompile_vertex(ordinary_vertex_mask_count,
                          std::size(ordinary_vertex_mask_count)).empty()) {
        printf("  [FAIL] ordinary vertex shader accepted NGG lane-zero mask semantics\n");
        return 1;
    }
    printf("  [ok]   ordinary vertex shader rejects NGG-only lane-zero mask semantics\n");

    // Even an exact GS_ALLOC_REQ marker does not make arbitrary LDS lane-local. A peer-addressed
    // write/barrier/read shape must stay fail-closed unless the complete observed Astro wrapper is
    // selected by its byte-exact fingerprint.
    const uint32_t unproven_ngg_peer_lds[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ
        0x7E000280u, 0x7E020281u,            // v0=0 byte address, v1=1 data
        0xD8340000u, 0x00000100u,            // ds_write_b32 v0, v1
        0xBF8A0000u,                         // s_barrier
        0xD8D80000u, 0x02000000u,            // ds_read_b32 v2, v0
        0x7E060280u, 0x7E0802F2u,
        0xF80008CFu, 0x04030302u,
        0xBF810000u,
    };
    if (!recompile_vertex(unproven_ngg_peer_lds, std::size(unproven_ngg_peer_lds)).empty()) {
        printf("  [FAIL] unproven NGG peer-LDS shader accepted private one-lane LDS semantics\n");
        return 1;
    }
    printf("  [ok]   unproven NGG peer-LDS shader remains fail-closed\n");

    // v_cmpx_* narrows EXEC. A FRAGMENT export under a narrowed EXEC is a discard (alpha test / kill): it
    // now lowers to a per-invocation OpKill of the inactive lanes followed by an export from the survivors,
    // so fragment recompilation ACCEPTS it and emits valid SPIR-V. (A VERTEX shader cannot discard — OpKill
    // is fragment-only — so the vertex cmpx-export case below still rejects.)
    const uint32_t cmpx_fragment[] = {
        0x7DA80300u, 0xF800180Fu, 0x03020100u, 0xBF810000u,
    };
    auto frag_spv = recompile_fragment(cmpx_fragment, sizeof(cmpx_fragment) / sizeof(cmpx_fragment[0]));
    if (frag_spv.empty()) {
        printf("  [FAIL] fragment cmpx discard shader was rejected (should lower to OpKill + export)\n");
        return 1;
    }
    { uint32_t bad_op = 0;
      if (!type_result_ids_are_nonzero(frag_spv, &bad_op)) {
          printf("  [FAIL] fragment discard SPIR-V has an invalid result id (op=%u)\n", bad_op);
          return 1;
      } }
    printf("  [ok]   fragment cmpx export lowers to a discard (OpKill + export), valid SPIR-V\n");

    // Alpha-test discard via the SCALAR-BRANCH form (not v_cmpx): compare a sampled/interpolated value,
    // ANDN2 the survivor mask into a saved EXEC copy (SCC = "any lane survives"), and s_cbranch_scc0 skips
    // the shading if NO lane survives; the block then narrows EXEC (s_wqm exec, survivors) and shades. This
    // is Unity's clip()/cutout text+sprite shape (The Messenger's cutscene text, #102). The recompiler must
    // lower it — drop the wave early-out, run the block, OpKill the failed lanes at export — instead of
    // rejecting the s_cbranch_scc0, which dropped every alpha-tested text/sprite draw. Bytes assembled with
    // llvm-mc gfx1010 (round-trip verified).
    const uint32_t altest_kill_branch[] = {
        0xbe82047eu,               // s_mov_b64  s[2:3], exec
        0x7c020300u,               // v_cmp_lt_f32_e32 vcc_lo, v0, v1      (alpha < ref -> vcc)
        0x8a826a02u,               // s_andn2_b64 s[2:3], s[2:3], vcc      (survivors; SCC = any-survivor)
        0xbf840003u,               // s_cbranch_scc0 +3                    (skip shading if none survive)
        0xbefe0a02u,               // s_wqm_b64  exec, s[2:3]              (narrow EXEC to survivors)
        0x7e040300u,               // v_mov_b32  v2, v0
        0x7e060301u,               // v_mov_b32  v3, v1
        0x7e0802f2u,               // v_mov_b32  v4, 1.0
        0x7e0a02f2u,               // v_mov_b32  v5, 1.0
        0xf800180fu, 0x05040302u,  // exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,               // s_endpgm
    };
    auto altest_spv = recompile_fragment(altest_kill_branch, sizeof(altest_kill_branch) / sizeof(altest_kill_branch[0]));
    if (altest_spv.empty()) {
        printf("  [FAIL] alpha-test scalar-branch discard (s_cbranch_scc0 on kill mask) was rejected\n");
        return 1;
    }
    { uint32_t bad_op = 0;
      if (!type_result_ids_are_nonzero(altest_spv, &bad_op)) {
          printf("  [FAIL] alpha-test discard SPIR-V has an invalid result id (op=%u)\n", bad_op);
          return 1;
      } }
    { bool has_kill = false;                       // the discard must actually emit OpKill (op 252)
      for (size_t i = 5; i < altest_spv.size(); ) { uint32_t wc = altest_spv[i] >> 16, op = altest_spv[i] & 0xffff;
          if (op == 252u) { has_kill = true; break; } i += wc ? wc : 1; }
      if (!has_kill) { printf("  [FAIL] alpha-test discard SPIR-V lacks an OpKill\n"); return 1; } }
    printf("  [ok]   alpha-test scalar-branch (s_andn2+s_cbranch_scc0) lowers to a discard (OpKill), valid SPIR-V\n");

    const uint32_t cmpx_vertex[] = {
        0x7DA80300u, 0xF80008CFu, 0x03020100u, 0xBF810000u,
    };
    if (!recompile_vertex(cmpx_vertex, sizeof(cmpx_vertex) / sizeof(cmpx_vertex[0])).empty()) {
        printf("  [FAIL] vertex cmpx shader was accepted without EXEC-masked export support\n");
        return 1;
    }
    printf("  [ok]   vertex cmpx shader is rejected until EXEC-masked export is modeled\n");

    // Graphics-path resource binding: a vertex shader that fetches its position from a vertex buffer
    //   v_mov v3, 0 ; v_mov v4, 1.0 ; buffer_load_format_xy v[1:2], v0, s[8:11] idxen ; exp pos0 v1..v4
    // The format-load needs a V# descriptor's data format to translate, which lives in the resource
    // table — so recompilation must FAIL without a table and SUCCEED (to valid SPIR-V) with one.
    const uint32_t vs_fetch[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
    };
    const size_t vs_fetch_n = sizeof(vs_fetch) / sizeof(vs_fetch[0]);
    if (!recompile_vertex(vs_fetch, vs_fetch_n, nullptr).empty()) {
        printf("  [FAIL] vertex fetch was accepted without a resource table (format unknown)\n");
        return 1;
    }
    printf("  [ok]   vertex fetch is rejected without a resource table\n");

    ShaderResourceTable rt;
    ShaderResource vb{};
    vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32; vb.num_components = 2;
    vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;   // V# placed directly in user-data s[8:11]
    rt.resources.push_back(vb);
    std::vector<uint32_t> vspv = recompile_vertex(vs_fetch, vs_fetch_n, &rt);
    if (vspv.empty() || vspv[0] != 0x07230203u) {
        printf("  [FAIL] vertex fetch did not recompile to valid SPIR-V with a resource table\n");
        return 1;
    }
    printf("  [ok]   vertex fetch recompiles to valid SPIR-V with a resource table (binding 3)\n");

    // image_sample LOD mode per execution model (#151): OpImageSampleImplicitLod is only legal in
    // the Fragment execution model — the compute and vertex shells have no derivatives, so an
    // image_sample there must lower to OpImageSampleExplicitLod (LOD 0) or spirv-val rejects the
    // module and pipeline creation fails.
    enum : uint32_t { OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod = 88,
                      OpImageQuerySizeLod = 103, OpImageQueryLevels = 106 };
    ShaderResourceTable rt_tex;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 1; /*2D*/
      t.width = 2; t.height = 2; t.sgpr_base = 8; rt_tex.resources.push_back(t); }

    // Compute shell: v0,v1 = uv inputs; image_sample v[0:3], v[0:1], s[8:15], s[16:19] dmask:0xf dim:2D.
    const uint32_t cs_sample[] = { 0xf0800f08u, 0x00820000u, 0xbf810000u };
    std::vector<uint32_t> cspv = recompile_valu(cs_sample, sizeof(cs_sample)/sizeof(cs_sample[0]), 2, 0, &rt_tex);
    if (cspv.empty() || cspv[0] != 0x07230203u) {
        printf("  [FAIL] compute-shell image_sample did not recompile\n");
        return 1;
    }
    if (has_opcode(cspv, OpImageSampleImplicitLod) || !has_opcode(cspv, OpImageSampleExplicitLod)) {
        printf("  [FAIL] compute-shell image_sample emitted ImplicitLod (fragment-only) instead of ExplicitLod\n");
        return 1;
    }
    printf("  [ok]   compute-shell image_sample lowers to OpImageSampleExplicitLod (LOD 0)\n");

    // Vertex shell: image_sample then export the result as the position.
    const uint32_t vs_sample[] = { 0xf0800f08u, 0x00820000u, 0xf80008cfu, 0x03020100u, 0xbf810000u };
    std::vector<uint32_t> vsspv = recompile_vertex(vs_sample, sizeof(vs_sample)/sizeof(vs_sample[0]), &rt_tex);
    if (vsspv.empty() || vsspv[0] != 0x07230203u) {
        printf("  [FAIL] vertex-shell image_sample did not recompile\n");
        return 1;
    }
    if (has_opcode(vsspv, OpImageSampleImplicitLod) || !has_opcode(vsspv, OpImageSampleExplicitLod)) {
        printf("  [FAIL] vertex-shell image_sample emitted ImplicitLod (fragment-only) instead of ExplicitLod\n");
        return 1;
    }
    printf("  [ok]   vertex-shell image_sample lowers to OpImageSampleExplicitLod (LOD 0)\n");

    // Fragment shell must KEEP implicit LOD (derivative-based mip selection is the hardware behavior).
    const uint32_t ps_sample[] = {
        0x7e0002ffu, 0x3e800000u, 0x7e0202ffu, 0x3e800000u, 0xf0800f08u, 0x00820000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    std::vector<uint32_t> pspv = recompile_fragment(ps_sample, sizeof(ps_sample)/sizeof(ps_sample[0]), &rt_tex);
    if (pspv.empty() || !has_opcode(pspv, OpImageSampleImplicitLod)) {
        printf("  [FAIL] fragment image_sample no longer uses OpImageSampleImplicitLod\n");
        return 1;
    }
    printf("  [ok]   fragment image_sample still uses OpImageSampleImplicitLod\n");

    // UE4/DOLL volume initialization starts by querying a 3D T# with image_get_resinfo, then uses the
    // xyz result for its dispatch bounds. This was the sole rejected opcode in that captured kernel.
    ShaderResourceTable rt_3d;
    { ShaderResource t{}; t.cls = ResourceClass::Texture; t.binding = 4; t.img_dim = 2;
      t.width = 8; t.height = 8; t.sgpr_base = 0; rt_3d.resources.push_back(t); }
    const uint32_t cs_resinfo[] = {
        0x7e060280u,                         // v_mov_b32 v3, 0 (LOD)
        0xf0380710u, 0x00000003u,           // image_get_resinfo v[0:2], v3, s[0:7] dmask:xyz dim:3D
        0xbf810000u,
    };
    std::vector<uint32_t> resinfo_spv = recompile_valu(
        cs_resinfo, sizeof(cs_resinfo)/sizeof(cs_resinfo[0]), 0, 0, &rt_3d);
    if (resinfo_spv.empty() || !has_opcode(resinfo_spv, OpImageQuerySizeLod) ||
        !has_opcode(resinfo_spv, OpImageQueryLevels)) {
        printf("  [FAIL] image_get_resinfo 3D did not lower to SPIR-V image queries\n");
        return 1;
    }
    printf("  [ok]   image_get_resinfo 3D lowers to size/level image queries\n");

    // LDS array is sized from the shader's real allocation (#130), not a hardcoded 16 KB. A compute
    // kernel that uses ds_write/ds_read declares a Workgroup array; its length must be 4096 dwords
    // (16 KB) by default and rise to the requested size (clamped to the RDNA2 64 KB / 16384-dword max)
    // when lds_bytes is plumbed. code32 = lane i writes lds[i], barrier, reads lds[63-i].
    const uint32_t code_lds[] = {
        0x7e020f00u, 0x34040282u, 0x34060281u, 0x4a060681u, 0xd8340000u, 0x00000302u, 0xbf8a0000u,
        0x4c0a02bfu, 0x340c0a82u, 0xd8d80000u, 0x07000006u, 0x7e000d07u, 0xbf810000u,
    };
    const size_t n_lds = sizeof(code_lds)/sizeof(code_lds[0]);
    std::vector<uint32_t> lds_def = recompile_valu(code_lds, n_lds, 1, 0, nullptr);
    std::vector<uint32_t> lds_32k = recompile_valu(code_lds, n_lds, 1, 0, nullptr, 32 * 1024);
    std::vector<uint32_t> lds_big = recompile_valu(code_lds, n_lds, 1, 0, nullptr, 128 * 1024);   // > 64 KB
    if (lds_def.empty() || lds_32k.empty() || lds_big.empty()) {
        printf("  [FAIL] LDS kernel did not recompile\n");
        return 1;
    }
    if (max_array_length(lds_def) != 4096u) {
        printf("  [FAIL] default LDS array length = %u dwords, want 4096 (16 KB)\n", max_array_length(lds_def));
        return 1;
    }
    printf("  [ok]   default LDS array is 4096 dwords (16 KB)\n");
    if (max_array_length(lds_32k) != 8192u) {
        printf("  [FAIL] lds_bytes=32K -> array length = %u dwords, want 8192\n", max_array_length(lds_32k));
        return 1;
    }
    printf("  [ok]   lds_bytes=32 KB -> 8192-dword LDS array\n");
    if (max_array_length(lds_big) != 16384u) {
        printf("  [FAIL] lds_bytes=128K -> array length = %u dwords, want 16384 (clamped to 64 KB)\n",
               max_array_length(lds_big));
        return 1;
    }
    printf("  [ok]   lds_bytes>64 KB clamps to the RDNA2 max (16384 dwords)\n");

    // v_interp_mov explicit-parameter reads (#152/#897). P0-only retains the cheap Flat varying.
    // Mixed smooth+P0 and P10/P20 use the portable generated geometry stage, which publishes the
    // coefficient values through separate Flat locations. Encodings llvm-mc gfx1030 verified.
    enum : uint32_t { OpDecorate = 71, DecFlat = 14 };
    // Flat-only: v_interp_mov v3, p0, attr0.x ; exp mrt0 v3,v3,v3,v3 ; s_endpgm
    const uint32_t ps_flat[] = { 0xc80e0002u, 0xf800000fu, 0x03030303u, 0xbf810000u };
    std::vector<uint32_t> flat_spv = recompile_fragment(ps_flat, sizeof(ps_flat)/sizeof(ps_flat[0]));
    if (flat_spv.empty() || !has_decoration(flat_spv, DecFlat)) {
        printf("  [FAIL] v_interp_mov attribute is not decorated Flat (would smooth-interpolate a flat read)\n");
        return 1;
    }
    printf("  [ok]   v_interp_mov attribute varying is decorated Flat\n");
    // Smooth-only: v_interp_p1 + v_interp_p2 on attr0 ; exp mrt0 v4 -> NOT Flat.
    const uint32_t ps_smooth[] = { 0xc8080000u, 0xc8110002u, 0xf800000fu, 0x04040404u, 0xbf810000u };
    std::vector<uint32_t> smooth_spv = recompile_fragment(ps_smooth, sizeof(ps_smooth)/sizeof(ps_smooth[0]));
    if (smooth_spv.empty() || has_decoration(smooth_spv, DecFlat)) {
        printf("  [FAIL] a smooth-interpolated (v_interp_p2) attribute must NOT be decorated Flat\n");
        return 1;
    }
    printf("  [ok]   v_interp_p2-only attribute stays smooth (no Flat decoration)\n");
    // Mixed: attr0 read via BOTH P0 and smooth interpolation needs separate interface locations.
    const uint32_t ps_mixed[] = { 0xc80e0002u, 0xc8110002u, 0xf800000fu, 0x03030303u, 0xbf810000u };
    FragmentInterpolationLayout mixed_layout = fragment_interpolation_layout(
        ps_mixed, sizeof(ps_mixed)/sizeof(ps_mixed[0]));
    std::vector<uint32_t> mixed_spv = recompile_fragment(
        ps_mixed, sizeof(ps_mixed)/sizeof(ps_mixed[0]), nullptr, nullptr, UINT32_MAX, &mixed_layout);
    std::vector<uint32_t> mixed_gs = recompile_interpolation_geometry(mixed_layout);
    if (!mixed_layout.valid || !mixed_layout.requires_geometry || mixed_spv.empty() ||
        mixed_gs.empty() || !has_opcode(mixed_gs, 218) || !has_opcode(mixed_gs, 219)) {
        printf("  [FAIL] mixed P0+smooth interpolation did not generate a geometry fallback\n");
        return 1;
    }
    printf("  [ok]   mixed P0+smooth interpolation generates coefficient pass-through SPIR-V\n");

    // Explicit P10/P20/P0 values are the three AMD parameters used by Astro Bot's rejected title
    // composite PS. All three fragment inputs are Flat outputs of the generated geometry stage.
    const uint32_t ps_parameters[] = {
        0xc80e0000u, 0xc8120001u, 0xc8160002u,
        0xf800000fu, 0x05050403u, 0xbf810000u,
    };
    FragmentInterpolationLayout parameter_layout = fragment_interpolation_layout(
        ps_parameters, sizeof(ps_parameters)/sizeof(ps_parameters[0]));
    std::vector<uint32_t> parameter_spv = recompile_fragment(
        ps_parameters, sizeof(ps_parameters)/sizeof(ps_parameters[0]), nullptr, nullptr,
        UINT32_MAX, &parameter_layout);
    std::vector<uint32_t> parameter_gs = recompile_interpolation_geometry(parameter_layout);
    if (!parameter_layout.valid || !parameter_layout.requires_geometry || parameter_spv.empty() ||
        parameter_gs.empty() || !has_decoration(parameter_spv, DecFlat) ||
        !has_decoration(parameter_gs, DecFlat)) {
        printf("  [FAIL] P10/P20/P0 explicit parameters did not lower through Flat geometry outputs\n");
        return 1;
    }
    printf("  [ok]   P10/P20/P0 lower through the portable Flat geometry interface\n");

    // Pixel-system VGPR initialization: with perspective sample/center enabled, disabled
    // centroid/pull-model slots reserved by ADDR, and position X/Y enabled, the packed destinations
    // are v12/v13. The fragment shell must source those values from gl_FragCoord rather than the
    // old undefined-register zero. Encodings: exp mrt0 v12,v13,0,1; s_endpgm. BuiltIn FragCoord=15.
    const uint32_t ps_position[] = {
        0x7e1c0280u, 0x7e1e02f2u, // v_mov v14,0; v_mov v15,1
        0xf800000fu, 0x0f0e0d0cu, 0xbf810000u,
    };
    PixelSystemInputMapping sys{};
    sys.ena = (1u << 0) | (1u << 1) | (1u << 8) | (1u << 9);
    sys.addr = sys.ena | (1u << 2) | (1u << 3) | (1u << 6) | (1u << 7);
    std::vector<uint32_t> position_spv = recompile_fragment(
        ps_position, sizeof(ps_position)/sizeof(ps_position[0]), nullptr, &sys);
    if (position_spv.empty() || !has_builtin(position_spv, 15)) {
        printf("  [FAIL] enabled POS_X/Y_FLOAT system VGPRs do not materialize gl_FragCoord\n");
        return 1;
    }
    printf("  [ok]   packed POS_X/Y_FLOAT system VGPRs source gl_FragCoord\n");

    printf("== PASS ==\n");
    return 0;
}
