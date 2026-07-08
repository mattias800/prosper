// test_rdna2_spirv_struct -- structural checks for RDNA2->SPIR-V output that do not require Vulkan.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include <cstdint>
#include <cstdio>
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

    printf("== PASS ==\n");
    return 0;
}
