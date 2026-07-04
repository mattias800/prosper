// test_rdna2_spirv_struct -- structural checks for RDNA2->SPIR-V output that do not require Vulkan.
#include "../src/gpu/rdna2_to_spirv.hpp"
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

    printf("== PASS ==\n");
    return 0;
}
