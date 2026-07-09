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
    enum : uint32_t { OpImageSampleImplicitLod = 87, OpImageSampleExplicitLod = 88 };
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

    printf("== PASS ==\n");
    return 0;
}
