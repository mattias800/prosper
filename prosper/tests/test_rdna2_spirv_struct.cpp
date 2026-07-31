// test_rdna2_spirv_struct -- structural checks for RDNA2->SPIR-V output that do not require Vulkan.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
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

// Whether an instruction's zero-based operand has the requested literal value.
bool has_instruction_operand(const std::vector<uint32_t>& spv, uint32_t opcode,
                             uint32_t operand_index, uint32_t value) {
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t word = spv[i];
        const uint32_t op = word & 0xffffu;
        const uint32_t wc = word >> 16u;
        if (wc == 0 || i + wc > spv.size()) return false;
        if (op == opcode && wc > operand_index + 1u &&
            spv[i + 1u + operand_index] == value)
            return true;
        i += wc;
    }
    return false;
}

bool has_select_with_false_constant(const std::vector<uint32_t>& spv, uint32_t literal) {
    constexpr uint32_t OpConstant = 43, OpSelect = 169;
    std::unordered_set<uint32_t> matching_constants;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4 && spv[i + 3] == literal)
            matching_constants.insert(spv[i + 2]);
        i += wc;
    }
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        // OpSelect {result-type, result, condition, true-object, false-object}.
        if (op == OpSelect && wc == 6 && matching_constants.contains(spv[i + 5]))
            return true;
        i += wc;
    }
    return false;
}

bool has_explicit_lod_constant(const std::vector<uint32_t>& spv, uint32_t expected) {
    constexpr uint32_t OpConstant = 43, OpImageSampleExplicitLod = 88, OpBitcast = 124;
    constexpr uint32_t ImageOperandsLod = 0x2;
    std::unordered_map<uint32_t, uint32_t> constants;
    std::unordered_map<uint32_t, uint32_t> bitcasts;
    if (spv.size() < 5) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstant && wc == 4) constants[spv[i + 2]] = spv[i + 3];
        if (op == OpBitcast && wc == 4) bitcasts[spv[i + 2]] = spv[i + 3];
        i += wc;
    }
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        // {result type, result, sampled image, coordinate, image-operands mask, lod}
        if (op == OpImageSampleExplicitLod && wc >= 7 &&
            (spv[i + 5] & ImageOperandsLod) != 0) {
            uint32_t value_id = spv[i + 6];
            if (const auto bitcast = bitcasts.find(value_id); bitcast != bitcasts.end())
                value_id = bitcast->second;
            const auto value = constants.find(value_id);
            if (value != constants.end() && value->second == expected) return true;
        }
        i += wc;
    }
    return false;
}

struct OutputStoreStats {
    uint32_t stores = 0;
    uint32_t stores_with_one_repeated_source = 0;
};

// Count stores to Output variables and inspect vec4 construction through the final bitcasts. The
// graphics CFG regression deliberately exports four independently-written VGPRs: if its callback
// accidentally reads the entry RegState, every missing VGPR instead resolves to the same zero ID.
OutputStoreStats output_store_stats(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpVariable = 59, OpStore = 62, OpCompositeConstruct = 80, OpBitcast = 124;
    constexpr uint32_t StorageClassOutput = 3;
    std::unordered_set<uint32_t> outputs;
    std::unordered_map<uint32_t, size_t> definitions;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return {};
        if (op == OpVariable && wc >= 4 && spv[i + 3] == StorageClassOutput)
            outputs.insert(spv[i + 2]);
        if ((op == OpCompositeConstruct || op == OpBitcast) && wc >= 3)
            definitions[spv[i + 2]] = i;
        i += wc;
    }
    OutputStoreStats stats;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return {};
        if (op == OpStore && wc == 3 && outputs.contains(spv[i + 1])) {
            ++stats.stores;
            const auto construct = definitions.find(spv[i + 2]);
            if (construct != definitions.end()) {
                const size_t ci = construct->second;
                const uint32_t cop = spv[ci] & 0xffffu, cwc = spv[ci] >> 16u;
                if (cop == OpCompositeConstruct && cwc == 7) {
                    uint32_t source[4]{};
                    bool traced = true;
                    for (uint32_t component = 0; component < 4; ++component) {
                        const auto bitcast = definitions.find(spv[ci + 3 + component]);
                        if (bitcast == definitions.end()) { traced = false; break; }
                        const size_t bi = bitcast->second;
                        if ((spv[bi] & 0xffffu) != OpBitcast || (spv[bi] >> 16u) != 4) {
                            traced = false;
                            break;
                        }
                        source[component] = spv[bi + 3];
                    }
                    if (traced && source[0] == source[1] && source[1] == source[2] &&
                        source[2] == source[3])
                        ++stats.stores_with_one_repeated_source;
                }
            }
        }
        i += wc;
    }
    return stats;
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

// ANDN1_SAVEEXEC must compute old_EXEC & ~source.  With source=false, the emitted boolean graph
// therefore contains LogicalNot(false) as an operand of LogicalAnd.  The formerly reversed
// lowering (~old_EXEC & source) instead fed false directly to the AND and silently killed all lanes.
bool logical_not_of_false_feeds_and(const std::vector<uint32_t>& spv) {
    constexpr uint32_t OpConstantFalse = 42, OpLogicalAnd = 167, OpLogicalNot = 168;
    uint32_t false_id = 0;
    std::unordered_map<uint32_t, uint32_t> logical_not_sources;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpConstantFalse && wc == 3) false_id = spv[i + 2];
        if (op == OpLogicalNot && wc == 4)
            logical_not_sources[spv[i + 2]] = spv[i + 3];
        i += wc;
    }
    if (!false_id) return false;
    for (size_t i = 5; i < spv.size();) {
        const uint32_t op = spv[i] & 0xffffu, wc = spv[i] >> 16u;
        if (!wc || i + wc > spv.size()) return false;
        if (op == OpLogicalAnd && wc == 5) {
            const auto left = logical_not_sources.find(spv[i + 3]);
            const auto right = logical_not_sources.find(spv[i + 4]);
            if ((left != logical_not_sources.end() && left->second == false_id) ||
                (right != logical_not_sources.end() && right->second == false_id))
                return true;
        }
        i += wc;
    }
    return false;
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

    // Astro Bot's Wave32 graphics shaders save/copy/restore active-lane masks through the low
    // halves of EXEC and VCC.  These are the B32 equivalents of the B64 mask moves already modeled
    // below, not scalar-data transfers: treating EXEC_LO as ordinary data rejected the entire draw.
    const uint32_t wave32_fragment_masks[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xbeea037eu,                         // s_mov_b32 vcc_lo, exec_lo
        0xbefe0300u,                         // s_mov_b32 exec_lo, s0
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,            // exp mrt0 v0,v1,v2,v3
        0xbf810000u,
    };
    if (!recompile_fragment(wave32_fragment_masks,
                            std::size(wave32_fragment_masks)).empty()) {
        printf("  [FAIL] Wave64/default graphics shader accepted Wave32 EXEC_LO/VCC_LO masks\n");
        return 1;
    }
    const auto wave32_fragment_spv = recompile_fragment(
        wave32_fragment_masks, std::size(wave32_fragment_masks), nullptr, nullptr,
        UINT32_MAX, nullptr, true);
    uint32_t wave32_fragment_bad_op = 0;
    if (wave32_fragment_spv.empty() ||
        !type_result_ids_are_nonzero(wave32_fragment_spv, &wave32_fragment_bad_op) ||
        !phi_ids_are_nonzero(wave32_fragment_spv)) {
        printf("  [FAIL] Wave32 fragment EXEC_LO/VCC_LO mask moves emitted invalid SPIR-V (op=%u)\n",
               wave32_fragment_bad_op);
        return 1;
    }
    printf("  [ok]   registered Wave32 fragment EXEC_LO/VCC_LO mask moves emit valid SPIR-V\n");

    if (fragment_effective_wave_size_for_test(
            64, 3142, 0x616dd4c0b241fbb1ull) != 32 ||
        fragment_effective_wave_size_for_test(
            64, 3142, 0x616dd4c0b241fbb0ull) != 64) {
        printf("  [FAIL] legacy Astro Wave32 capture did not select one coherent subgroup contract\n");
        return 1;
    }
    printf("  [ok]   legacy Astro Wave32 capture selects a 32-lane subgroup and mask contract\n");

    const uint32_t wave32_compute_masks[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xbeea037eu,                         // s_mov_b32 vcc_lo, exec_lo
        0xbefe0300u,                         // s_mov_b32 exec_lo, s0
        0xbf810000u,
    };
    ComputeShaderConfig wave32_compute_config;
    wave32_compute_config.wave_size = 32;
    if (recompile_compute(wave32_compute_masks, std::size(wave32_compute_masks), nullptr,
                          wave32_compute_config).empty()) {
        printf("  [FAIL] proven Wave32 compute mask moves did not recompile\n");
        return 1;
    }
    // A Wave32 VOPC creates a VCC mask lifetime; one arm then recycles VCC_LO as ordinary scalar
    // data. The mask view is unknown after the merge, so consuming it with cndmask must remain
    // fail-visible instead of silently treating the poisoned edge as architectural false.
    const uint32_t wave32_compute_recycled_vcc[] = {
        0x7d840000u,                         // v_cmp_eq_u32 vcc, v0, v0
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0xbf840001u,                         // s_cbranch_scc0 +1
        0xbeea0385u,                         // s_mov_b32 vcc_lo, 5 (scalar-data lifetime)
        0x02020100u,                         // v_cndmask_b32 v1, v0, v0, vcc
        0xbf810000u,
    };
    wave32_compute_config.native_subgroup_size = 32;
    wave32_compute_config.local_x = 32;
    if (!recompile_compute(wave32_compute_recycled_vcc,
                           std::size(wave32_compute_recycled_vcc), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] poisoned Wave32 VCC was consumed after a CFG merge\n");
        return 1;
    }
    printf("  [ok]   poisoned Wave32 VCC consumption after a CFG merge is rejected\n");
    const uint32_t wave32_compute_recycled_vcc_half[] = {
        0x7d840000u,                         // v_cmp_eq_u32 vcc, v0, v0
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0xbf840001u,                         // s_cbranch_scc0 +1
        0xbeea0385u,                         // one arm recycles vcc_lo as scalar data
        0x876b0100u,                         // s_and_b32 vcc_hi, s0, s1
        0x02020100u,                         // implicit VCC_LO read remains ambiguous
        0xbf810000u,
    };
    wave32_compute_config.user_sgprs = {1u, 1u};
    if (!recompile_compute(wave32_compute_recycled_vcc_half,
                           std::size(wave32_compute_recycled_vcc_half), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] partial VCC write accepted an unknown prior mask\n");
        return 1;
    }
    printf("  [ok]   partial VCC_HI write cannot validate an ambiguous VCC_LO read\n");

    // A complete scalar-data write to VCC_LO defines every architectural predicate bit in Wave32.
    // Consume it immediately through the implicit e32 cndmask form so this cannot pass merely by
    // treating the new lifetime as dead scalar scratch.
    const uint32_t wave32_compute_scalar_vcc_lo_consumer[] = {
        0x876a8181u,                         // s_and_b32 vcc_lo, 1, 1
        0x02020100u,                         // v_cndmask_b32 v1, v0, v0, vcc
        0xbf810000u,
    };
    if (recompile_compute(wave32_compute_scalar_vcc_lo_consumer,
                          std::size(wave32_compute_scalar_vcc_lo_consumer), nullptr,
                          wave32_compute_config).empty()) {
        printf("  [FAIL] full Wave32 scalar VCC_LO write did not feed implicit cndmask\n");
        return 1;
    }
    printf("  [ok]   full Wave32 scalar VCC_LO write feeds its implicit mask consumer\n");

    // Astro's exact PC458 packet explicitly selects physical VCC_HI in Wave32. A typed B32 mask in
    // that word must drive the select independently of VCC_LO; absent or path-dependent HI mask
    // lifetimes must remain fail-visible instead of falling back to the implicit VCC predicate.
    const uint32_t wave32_compute_explicit_vcc_hi_cndmask[] = {
        0xbf060000u,                         // s_cmp_eq_u32 s0,s0
        0x856b807eu,                         // s_cselect_b32 vcc_hi, exec_lo, 0
        0xd5010000u, 0x01ad0280u,            // v_cndmask_b32_e64 v0, 0, 1, vcc_hi
        0xbf810000u,
    };
    const auto explicit_vcc_hi_cndmask_spv = recompile_compute(
        wave32_compute_explicit_vcc_hi_cndmask,
        std::size(wave32_compute_explicit_vcc_hi_cndmask), nullptr,
        wave32_compute_config);
    if (explicit_vcc_hi_cndmask_spv.empty() ||
        !type_result_ids_are_nonzero(explicit_vcc_hi_cndmask_spv, nullptr) ||
        !phi_ids_are_nonzero(explicit_vcc_hi_cndmask_spv)) {
        printf("  [FAIL] explicit Wave32 VCC_HI cndmask source was not preserved\n");
        return 1;
    }
    const uint32_t wave32_compute_absent_vcc_hi_cndmask[] = {
        0xbeeb0380u,                         // scalar-data vcc_hi=0, not a mask
        0xd5010000u, 0x01ad0280u,            // v_cndmask_b32_e64 v0, 0, 1, vcc_hi
        0xbf810000u,
    };
    if (!recompile_compute(wave32_compute_absent_vcc_hi_cndmask,
                           std::size(wave32_compute_absent_vcc_hi_cndmask), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] explicit VCC_HI cndmask accepted an absent mask lifetime\n");
        return 1;
    }
    const uint32_t wave32_compute_ambiguous_vcc_hi_cndmask[] = {
        0xbf060000u,                         // pc0: scalar branch condition
        0xbf840001u,                         // pc1: one edge skips the HI definition
        0x856b807eu,                         // pc2: other edge defines a VCC_HI mask
        0xd5010000u, 0x01ad0280u,            // pc3: joined VCC_HI read
        0xbf810000u,
    };
    if (!recompile_compute(wave32_compute_ambiguous_vcc_hi_cndmask,
                           std::size(wave32_compute_ambiguous_vcc_hi_cndmask), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] explicit VCC_HI cndmask accepted a path-dependent mask lifetime\n");
        return 1;
    }
    printf("  [ok]   explicit Wave32 VCC_HI cndmask requires its own unambiguous mask\n");

    // Exact control/data lifetime from Astro Bot's world-map phase, reduced only to its register-
    // independent instructions: a scalar-data VCC_LO lifetime feeds CMPX (which changes EXEC but
    // preserves VCC), one crossing arm defines and immediately consumes a fresh VCC mask, and the
    // other arm skips that definition. The rejoined VCC lifetime is invalid but dead. Crossing
    // EXECZ regions force the same arbitrary-CFG dispatcher as the 216..306 production phase.
    const uint32_t wave32_compute_dead_vcc_join[] = {
        0xbeea0385u,                         // pc0: scalar-data vcc_lo=5
        0x7da40100u,                         // pc1: v_cmpx_eq_u32 v0,v0 (EXEC only)
        0xbf880003u,                         // pc2: execz -> pc6, skipping the VCC definition
        0x7d840100u,                         // pc3: v_cmp_eq_u32 vcc,v0,v0 (fresh mask)
        0x02020100u,                         // pc4: v_cndmask_b32 v1,v0,v0,vcc
        0xbf880002u,                         // pc5: execz -> pc8 (crosses the pc2 region)
        0xbf060000u,                         // pc6: rejoin; scalar compare, no VCC read
        0xbf850001u,                         // pc7: third branch -> pc9 forces complex CFG
        0x7e040280u,                         // pc8: no mask read before termination
        0xbf810000u,
    };
    const auto dead_vcc_join_spv = recompile_compute(
        wave32_compute_dead_vcc_join, std::size(wave32_compute_dead_vcc_join), nullptr,
        wave32_compute_config);
    if (dead_vcc_join_spv.empty() || !has_opcode(dead_vcc_join_spv, 251) ||
        !type_result_ids_are_nonzero(dead_vcc_join_spv, nullptr) ||
        !phi_ids_are_nonzero(dead_vcc_join_spv)) {
        printf("  [FAIL] Wave32 CFG rejected Astro's dead VCC lifetime at a crossing join\n");
        return 1;
    }
    printf("  [ok]   Wave32 CFG carries Astro's invalid-but-dead VCC lifetime across a join\n");

    std::vector<uint32_t> dead_vcc_read_at_join(
        std::begin(wave32_compute_dead_vcc_join),
        std::end(wave32_compute_dead_vcc_join));
    dead_vcc_read_at_join[6] = 0x02040500u; // v_cndmask_b32 v2,v0,v2,vcc
    if (!recompile_compute(dead_vcc_read_at_join.data(), dead_vcc_read_at_join.size(),
                           nullptr, wave32_compute_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted a VCC read after the ambiguous join\n");
        return 1;
    }
    std::vector<uint32_t> dead_vcc_missing_definition(
        std::begin(wave32_compute_dead_vcc_join),
        std::end(wave32_compute_dead_vcc_join));
    dead_vcc_missing_definition[3] = 0x7da40100u; // CMPX does not define VCC
    if (!recompile_compute(dead_vcc_missing_definition.data(),
                           dead_vcc_missing_definition.size(), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted a cndmask after replacing its VCC definition\n");
        return 1;
    }
    std::vector<uint32_t> dead_vcc_entered_consumer(
        std::begin(wave32_compute_dead_vcc_join),
        std::end(wave32_compute_dead_vcc_join));
    dead_vcc_entered_consumer[2] = 0xbf880001u; // topology now enters pc4, past the definition
    if (!recompile_compute(dead_vcc_entered_consumer.data(),
                           dead_vcc_entered_consumer.size(), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] Wave32 CFG accepted a branch edge entering past the VCC definition\n");
        return 1;
    }
    printf("  [ok]   Wave32 CFG rejects VCC-read, definition, and branch-topology deviations\n");

    // Reduced lifetime from Astro Bot's world-map compute PC396/972/1062/1063/2311/2312. An
    // earlier s_mov_b64 makes s[32:33] a saved mask, a dominating multiword SMEM data load ends that
    // lifetime, and s35 then remains numeric on both entries to the CMPX loop header. The crossing
    // loops force the arbitrary-CFG dispatcher used by the production shader.
    const uint32_t wave32_compute_recycled_b64_before_loop[] = {
        0xbea0047eu,                         // pc0:  s_mov_b64 s[32:33], exec
        0xf4280800u, 0xfa000000u,            // pc1:  s_buffer_load_dwordx4 s[32:35],s[0:3],0
        0xbea30320u,                         // pc3:  s_mov_b32 s35,s32
        0x7d820023u,                         // pc4:  CMPX consumes numeric s35 (reduced v0 source)
        0xbf880006u,                         // pc5:  A exit -> pc12
        0x7d820023u,                         // pc6:  crossing B header consumes numeric s35
        0xbf880004u,                         // pc7:  B exit -> pc12
        0x81238123u,                         // pc8:  s_sub_u32 s35,s35,1
        0xbf82fffau,                         // pc9:  backedge -> pc4
        0x81238123u,                         // pc10: unreachable second decrement
        0xbf82fffau,                         // pc11: syntactic crossing backedge -> pc6
        0xbf810000u,                         // pc12: s_endpgm
    };
    ShaderResourceTable recycled_b64_resources;
    ShaderResource recycled_b64_cbuf;
    recycled_b64_cbuf.cls = ResourceClass::ConstantBuffer;
    recycled_b64_cbuf.format = DataFormat::Uint32;
    recycled_b64_cbuf.num_components = 1;
    recycled_b64_cbuf.binding = 2;
    recycled_b64_cbuf.sgpr_base = 0;
    recycled_b64_cbuf.size = 64;
    recycled_b64_resources.resources.push_back(recycled_b64_cbuf);
    ComputeShaderConfig recycled_b64_config = wave32_compute_config;
    recycled_b64_config.user_sgprs.resize(4);
    const auto recycled_b64_loop_spv = recompile_compute(
        wave32_compute_recycled_b64_before_loop,
        std::size(wave32_compute_recycled_b64_before_loop), &recycled_b64_resources,
        recycled_b64_config);
    if (recycled_b64_loop_spv.empty() || !has_opcode(recycled_b64_loop_spv, 251) ||
        !type_result_ids_are_nonzero(recycled_b64_loop_spv, nullptr) ||
        !phi_ids_are_nonzero(recycled_b64_loop_spv)) {
        printf("  [FAIL] dominating scalar overwrite did not end a stale B64 mask lifetime\n");
        return 1;
    }

    std::vector<uint32_t> recycled_b64_missing_overwrite(
        std::begin(wave32_compute_recycled_b64_before_loop),
        std::end(wave32_compute_recycled_b64_before_loop));
    recycled_b64_missing_overwrite[1] = 0xbf800000u; // remove both dominating data writes
    recycled_b64_missing_overwrite[2] = 0xbf800000u;
    if (!recompile_compute(recycled_b64_missing_overwrite.data(),
                           recycled_b64_missing_overwrite.size(), &recycled_b64_resources,
                           recycled_b64_config).empty()) {
        printf("  [FAIL] B64-mask first entry joined numeric loop backedge without rejection\n");
        return 1;
    }

    const uint32_t wave32_compute_b64_overwrite_bypass[] = {
        0xbea0047eu,                         // pc0:  saved B64 mask in s[32:33]
        0xbf060000u,                         // pc1:  scalar branch condition
        0xbf840002u,                         // pc2:  one edge bypasses both data writes -> pc5
        0xf4280800u, 0xfa000000u,            // pc3:  s_buffer_load_dwordx4 s[32:35],s[0:3],0
        0xbea30320u,                         // pc5:  ambiguous s32 -> s35 copy
        0x7d820023u,                         // pc6:  numeric CMPX consumption (reduced v0 source)
        0xbf880006u,                         // pc7:  A exit -> pc14
        0x7d820023u,                         // pc8:  crossing B header
        0xbf880004u,                         // pc9:  B exit -> pc14
        0x81238123u,                         // pc10: numeric decrement
        0xbf82fffau,                         // pc11: backedge -> pc6
        0x81238123u,                         // pc12: unreachable second decrement
        0xbf82fffau,                         // pc13: syntactic crossing backedge -> pc8
        0xbf810000u,                         // pc14: s_endpgm
    };
    if (!recompile_compute(wave32_compute_b64_overwrite_bypass,
                           std::size(wave32_compute_b64_overwrite_bypass),
                           &recycled_b64_resources, recycled_b64_config).empty()) {
        printf("  [FAIL] branch bypass of the B64-to-data overwrite was accepted\n");
        return 1;
    }

    std::vector<uint32_t> recycled_b64_mask_backedge(
        std::begin(wave32_compute_recycled_b64_before_loop),
        std::end(wave32_compute_recycled_b64_before_loop));
    recycled_b64_mask_backedge[8] = 0xbea3037eu; // reachable backedge keeps a real mask in s35
    if (!recompile_compute(recycled_b64_mask_backedge.data(),
                           recycled_b64_mask_backedge.size(), &recycled_b64_resources,
                           recycled_b64_config).empty()) {
        printf("  [FAIL] numeric first entry joined a real-mask backedge without rejection\n");
        return 1;
    }
    printf("  [ok]   Wave32 B64 mask lifetimes end only on dominating scalar-data overwrites\n");

    // The production shader wraps barrier-separated work in a workgroup-uniform early-out. Its
    // first barrier-free phase has the same crossing Wave32 CFG as the fixture above and no guest
    // S_ENDPGM of its own. The phase splitter must supply an emitter-only terminator so the
    // dispatcher can become inactive, rejoin the uniform outer shell, and reach the guest barrier.
    const uint32_t wave32_compute_guarded_cfg_phase[] = {
        0xbf060000u,                         // pc0: uniform s_cmp_eq_u32 s0,s0
        0xbf84000cu,                         // pc1: early-out -> terminal pc14
        0xbeea0385u,                         // pc2: scalar-data vcc_lo=5
        0x7da40100u,                         // pc3: v_cmpx_eq_u32 v0,v0 (EXEC only)
        0xbf880003u,                         // pc4: execz -> pc8
        0x7d840100u,                         // pc5: fresh VCC definition
        0x02020100u,                         // pc6: consume the fresh VCC
        0xbf880002u,                         // pc7: execz -> pc10
        0xbf060000u,                         // pc8: rejoin without reading VCC
        0xbf850001u,                         // pc9: branch -> pc11, skipping pc10
        0x7e040280u,                         // pc10: first arm
        0x7e060280u,                         // pc11: phase-local join
        0xbf8a0000u,                         // pc12: uniform guest barrier
        0x7e080280u,                         // pc13: tail phase
        0xbf810000u,                         // pc14: terminal s_endpgm
    };
    const auto guarded_cfg_phase_spv = recompile_compute(
        wave32_compute_guarded_cfg_phase, std::size(wave32_compute_guarded_cfg_phase), nullptr,
        wave32_compute_config);
    if (guarded_cfg_phase_spv.empty() || !has_opcode(guarded_cfg_phase_spv, 251) ||
        !type_result_ids_are_nonzero(guarded_cfg_phase_spv, nullptr) ||
        !phi_ids_are_nonzero(guarded_cfg_phase_spv)) {
        printf("  [FAIL] barrier phase rejected Astro's crossing Wave32 CFG\n");
        return 1;
    }
    printf("  [ok]   barrier phase terminates and rejoins Astro's crossing Wave32 CFG\n");

    wave32_compute_config.wave_size = 64;
    if (!recompile_compute(wave32_compute_masks, std::size(wave32_compute_masks), nullptr,
                           wave32_compute_config).empty()) {
        printf("  [FAIL] Wave64 compute accepted Wave32 low-half mask semantics\n");
        return 1;
    }
    printf("  [ok]   compute low-half mask semantics require proven Wave32 launch state\n");

    const uint32_t wave32_fragment_wqm[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xbefe0900u,                         // s_wqm_b32 exec_lo, s0
        0xbefe097eu,                         // s_wqm_b32 exec_lo, exec_lo
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,            // exp mrt0 v0,v1,v2,v3
        0xbf810000u,
    };
    const auto wave32_fragment_wqm_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_wqm, std::size(wave32_fragment_wqm));
    if (wave32_fragment_wqm_spv.empty() ||
        !type_result_ids_are_nonzero(wave32_fragment_wqm_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_wqm_spv)) {
        printf("  [FAIL] Wave32 fragment s_wqm_b32 mask path did not recompile cleanly\n");
        return 1;
    }
    printf("  [ok]   Wave32 fragment s_wqm_b32 mask path emits valid SPIR-V\n");

    // Exact Astro world-map PC1060..1064 shape. LLVM gfx1030 disassembly identifies the compare as
    // `v_cmp_eq_f32_sdwa vcc_hi, 0, v7`; in Wave32 that explicit one-word destination is an
    // independent saved mask consumed by s_andn2_b32 before WQM restores EXEC_LO.
    const uint32_t wave32_fragment_b32_logic[] = {
        0xbec0037eu,                         // s_mov_b32 s64, exec_lo
        0x7e0e0280u,                         // v_mov_b32 v7, 0
        0x7c040ef9u, 0x0686eb80u,            // v_cmp_eq_f32_sdwa vcc_hi, 0, v7
        0x8a406b40u,                         // s_andn2_b32 s64, s64, vcc_hi
        0xbf840008u,                         // s_cbranch_scc0 -> terminal null-export tail
        0xbefe0940u,                         // s_wqm_b32 exec_lo, s64
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
        0xbefe0380u,                         // tail: s_mov_b64 exec, 0
        0xf8001c00u, 0x00000000u,            // exp null off, off, off, off done vm
        0xbf810000u,
    };
    const auto wave32_fragment_b32_logic_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_b32_logic, std::size(wave32_fragment_b32_logic));
    if (wave32_fragment_b32_logic_spv.empty() ||
        !has_opcode(wave32_fragment_b32_logic_spv, 335) ||
        fragment_spirv_required_subgroup_size(wave32_fragment_b32_logic_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_b32_logic_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_b32_logic_spv)) {
        printf("  [FAIL] Wave32 VCC_HI compare/B32 mask logic did not recompile cleanly "
               "(words=%zu vote=%d marker=%d subgroup=%u)\n",
               wave32_fragment_b32_logic_spv.size(),
               has_opcode(wave32_fragment_b32_logic_spv, 335),
               has_opcode(wave32_fragment_b32_logic_spv, 330),
               fragment_spirv_required_subgroup_size(wave32_fragment_b32_logic_spv));
        return 1;
    }
    printf("  [ok]   Wave32 B32 alpha-test vote linearizes its terminal null-export branch\n");

    const auto wave32_mask_branches = mask_test_branches_for_test(
        wave32_fragment_b32_logic, std::size(wave32_fragment_b32_logic), true);
    bool found_mask_branch = false;
    for (const uint32_t pc : wave32_mask_branches)
        if (pc == 5) found_mask_branch = true;
    if (!found_mask_branch) {
        printf("  [FAIL] proven Wave32 B32 mask vote was not recognized at PC5\n");
        return 1;
    }
    const uint32_t wave32_scalar_scc_branch[] = {
        0x87008104u,                         // s_and_b32 s0, s4, 1 (ordinary scalar data)
        0xbf840002u,                         // s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_scalar_scc_branch,
                                     std::size(wave32_scalar_scc_branch), true).empty()) {
        printf("  [FAIL] ordinary Wave32 scalar SCC branch was mistaken for a mask vote\n");
        return 1;
    }
    printf("  [ok]   Wave32 branch linearization requires proven mask provenance\n");

    // A wide scalar write kills every physical SGPR word it covers. Keep the compare in s1
    // deliberately adjacent to s0: if s_mov_b64 only invalidates its low word, the stale s1 mask
    // provenance infects the ordinary s_and_b32 and makes its real SCC branch look disposable.
    const uint32_t wave32_mask_overwritten_by_b64[] = {
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0xbe800480u,                         // s_mov_b64 s[0:1], 0
        0x87008101u,                         // s_and_b32 s0, s1, 1 (ordinary scalar data)
        0xbf840002u,                         // s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_mask_overwritten_by_b64,
                                     std::size(wave32_mask_overwritten_by_b64), true).empty()) {
        printf("  [FAIL] B64 scalar overwrite retained stale high-word mask provenance\n");
        return 1;
    }
    printf("  [ok]   B64 scalar writes invalidate every covered Wave32 mask word\n");

    const uint32_t wave32_mask_overwritten_by_saveexec[] = {
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0xbe802680u,                         // s_xor_saveexec_b64 s[0:1], 0
        0x87008101u,                         // s_and_b32 s0, s1, 1 (ordinary scalar data)
        0xbf840002u,                         // s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_mask_overwritten_by_saveexec,
                                     std::size(wave32_mask_overwritten_by_saveexec), true).empty()) {
        printf("  [FAIL] B64 SAVEEXEC overwrite retained stale high-word mask provenance\n");
        return 1;
    }
    printf("  [ok]   every supported B64 SAVEEXEC form has a two-word write lifetime\n");

    // The compare executes only on the fall-through predecessor. At the join, s1 is not proven to
    // be a mask on every path, so the following scalar SCC branch must remain real control flow.
    const uint32_t wave32_path_dependent_branch_mask[] = {
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0xbf840002u,                         // s_cbranch_scc0 -> join at PC4
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0x87008101u,                         // join: s_and_b32 s0, s1, 1
        0xbf840002u,                         // real s_cbranch_scc0
        0xbf810000u,
    };
    if (!mask_test_branches_for_test(wave32_path_dependent_branch_mask,
                                     std::size(wave32_path_dependent_branch_mask), true).empty()) {
        printf("  [FAIL] path-dependent B32 mask provenance escaped its predecessor block\n");
        return 1;
    }
    printf("  [ok]   Wave32 mask-branch proof does not cross control-flow joins\n");

    // Reduced Astro world-map PS PC9..33 preamble: s64 is a saved mask before a real EXECZ
    // conditional. The fall-through arm refines it with an explicit VOPC mask and immediately votes
    // through SCC. Provenance on that arm is valid even though the EXECZ target bypasses the block.
    const uint32_t wave32_mask_through_execz_fallthrough[] = {
        0xbec0037eu,                         // pc0: s_mov_b32 s64, exec_lo
        0x7c220b31u,                         // pc1: v_cmpx_lt_f32 vcc, v49, v5
        0xbf880004u,                         // pc2: s_cbranch_execz -> merge at PC7
        0x7c0862f9u, 0x0686eb25u,            // pc3: v_cmp_lt_f32_sdwa vcc_hi, s37, v49
        0x8a406b40u,                         // pc5: s_andn2_b32 s64, s64, vcc_hi
        0xbf840001u,                         // pc6: s_cbranch_scc0 -> terminal tail
        0xbf810000u,                         // pc7: merge
    };
    bool found_fallthrough_vote = false;
    for (uint32_t pc : mask_test_branches_for_test(
             wave32_mask_through_execz_fallthrough,
             std::size(wave32_mask_through_execz_fallthrough), true))
        if (pc == 6) found_fallthrough_vote = true;
    if (!found_fallthrough_vote) {
        printf("  [FAIL] valid fall-through Wave32 mask provenance was lost at EXECZ\n");
        return 1;
    }
    printf("  [ok]   Wave32 mask proof preserves valid conditional fall-through provenance\n");

    // The same live shader selects EXEC_LO or an empty mask into VCC_HI from an ordinary scalar
    // comparison. This is s_cselect_b32's mask-domain form, not a scalar integer selection.
    const uint32_t wave32_fragment_b32_cselect[] = {
        0xbec0037eu,                         // s_mov_b32 s64, exec_lo
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0x856b807eu,                         // s_cselect_b32 vcc_hi, exec_lo, 0
        0x8a406b40u,                         // s_andn2_b32 s64, s64, vcc_hi
        0xbefe0940u,                         // s_wqm_b32 exec_lo, s64
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_b32_cselect_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_b32_cselect, std::size(wave32_fragment_b32_cselect));
    if (wave32_fragment_b32_cselect_spv.empty() ||
        fragment_spirv_required_subgroup_size(wave32_fragment_b32_cselect_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_b32_cselect_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_b32_cselect_spv)) {
        printf("  [FAIL] Wave32 s_cselect_b32 did not preserve its VCC_HI mask result\n");
        return 1;
    }
    printf("  [ok]   Wave32 s_cselect_b32 preserves its selected VCC_HI mask\n");

    // Explicit Wave32 VOPC destinations are one-word masks even when they name ordinary SGPRs.
    // The live barycentric block compares into s0 and VCC_HI, writes an independent carry mask to
    // VCC_LO, then ANDs s0 and the still-live VCC_HI mask back into VCC_LO.
    const uint32_t wave32_fragment_explicit_vopc_masks[] = {
        0x7d8654f9u, 0x06868004u,            // v_cmp_le_u32_sdwa s0, s4, v42
        0x7d865cf9u, 0x0686eb04u,            // v_cmp_le_u32_sdwa vcc_hi, s4, v46
        0xd5286a29u, 0x00025880u,            // v_add_co_ci_u32 v41, vcc_lo, 0, v44, s0
        0x876a6b00u,                         // s_and_b32 vcc_lo, s0, vcc_hi
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_explicit_vopc_masks_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_explicit_vopc_masks,
        std::size(wave32_fragment_explicit_vopc_masks));
    if (wave32_fragment_explicit_vopc_masks_spv.empty() ||
        !has_opcode(wave32_fragment_explicit_vopc_masks_spv, 335) ||
        fragment_spirv_required_subgroup_size(wave32_fragment_explicit_vopc_masks_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_explicit_vopc_masks_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_explicit_vopc_masks_spv)) {
        printf("  [FAIL] explicit Wave32 VOPC SGPR destinations lost their B32 mask domain\n");
        return 1;
    }
    printf("  [ok]   explicit Wave32 VOPC SGPR destinations feed B32 mask logic\n");

    // A VOPC write to s0 must not erase the adjacent, independently-live s1 mask. In Wave32 each
    // explicit compare destination occupies exactly one scalar word; the old generic inventory
    // incorrectly treated both compares as two-word writes.
    const uint32_t wave32_fragment_adjacent_vopc_masks[] = {
        0x7d865cf9u, 0x06868104u,            // v_cmp_le_u32_sdwa s1, s4, v46
        0x7d8654f9u, 0x06868004u,            // v_cmp_le_u32_sdwa s0, s4, v42
        0x876a0100u,                         // s_and_b32 vcc_lo, s0, s1
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_fragment_adjacent_vopc_masks,
            std::size(wave32_fragment_adjacent_vopc_masks)).empty()) {
        printf("  [FAIL] adjacent explicit Wave32 VOPC masks clobbered each other\n");
        return 1;
    }
    printf("  [ok]   explicit Wave32 VOPC destinations preserve adjacent SGPR masks\n");

    // VCC_LO/HI remain physical scalar scratch registers too. A B32 ALU destination alone does not
    // prove mask-domain use: the live shader builds M0 from ordinary integer data through VCC_LO.
    const uint32_t wave32_fragment_vcc_scratch[] = {
        0x876a8744u,                         // s_and_b32 vcc_lo, s68, 7
        0x936a856au,                         // s_lshl_b32 vcc_lo, vcc_lo, 5
        0xbefc036au,                         // s_mov_b32 m0, vcc_lo
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_fragment_vcc_scratch,
            std::size(wave32_fragment_vcc_scratch)).empty()) {
        printf("  [FAIL] Wave32 VCC_LO scalar-scratch chain was mistaken for a wave mask\n");
        return 1;
    }
    printf("  [ok]   Wave32 VCC_LO remains available for ordinary scalar scratch data\n");

    // Inline constants 0/1 use numeric operand values that also happen to name s0/s1. Even while
    // s0 is a saved Wave32 mask, an all-inline cselect into VCC_HI is ordinary scalar data.
    const uint32_t wave32_fragment_inline_cselect_data[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo (mask-domain s0)
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0x856b8081u,                         // s_cselect_b32 vcc_hi, 1, 0 (data-domain)
        0xbf060000u,                         // s_cmp_eq_u32 s0, s0
        0x8801fd6bu,                         // s_or_b32 s1, vcc_hi, scc
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_fragment_inline_cselect_data,
            std::size(wave32_fragment_inline_cselect_data)).empty()) {
        printf("  [FAIL] Wave32 inline cselect constants aliased saved-mask register numbers\n");
        return 1;
    }
    printf("  [ok]   Wave32 inline cselect constants remain ordinary scalar data\n");

    // Exact PC790..803 control idiom: save EXEC_LO into VCC_HI while narrowing through VCC_LO,
    // then restore that independently tracked saved mask through a B32 move.
    const uint32_t wave32_fragment_and_saveexec[] = {
        0xbeea037eu,                         // s_mov_b32 vcc_lo, exec_lo
        0xbeeb3c6au,                         // s_and_saveexec_b32 vcc_hi, vcc_lo
        0xbefe036bu,                         // s_mov_b32 exec_lo, vcc_hi
        0xbeea446au,                         // s_andn1_saveexec_b32 vcc_lo, vcc_lo
        0xbefe036au,                         // s_mov_b32 exec_lo, vcc_lo
        0xbea0037eu,                         // s_mov_b32 s32, exec_lo
        0xbeea4020u,                         // s_orn2_saveexec_b32 vcc_lo, s32
        0xbefe036au,                         // s_mov_b32 exec_lo, vcc_lo
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_and_saveexec_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_and_saveexec, std::size(wave32_fragment_and_saveexec));
    if (wave32_fragment_and_saveexec_spv.empty() ||
        !has_opcode(wave32_fragment_and_saveexec_spv, 335) ||
        fragment_spirv_required_subgroup_size(wave32_fragment_and_saveexec_spv) != 32 ||
        !type_result_ids_are_nonzero(wave32_fragment_and_saveexec_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_and_saveexec_spv)) {
        printf("  [FAIL] Wave32 B32 saveexec family did not preserve VCC/EXEC masks\n");
        return 1;
    }
    printf("  [ok]   Wave32 AND/ORN2/ANDN1 saveexec family preserves saved EXEC\n");

    // Use a distinct dynamic old EXEC and a false source so the operand order is observable in the
    // emitted SPIR-V, rather than merely checking that the saveexec family can be translated.
    const uint32_t wave32_fragment_andn1_saveexec[] = {
        0x7c220300u,                         // v_cmpx_lt_f32 v0, v1
        0xbe804480u,                         // s_andn1_saveexec_b32 s0, 0
        0xbefe0300u,                         // s_mov_b32 exec_lo, s0
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_andn1_saveexec_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_andn1_saveexec, std::size(wave32_fragment_andn1_saveexec));
    if (wave32_fragment_andn1_saveexec_spv.empty() ||
        !logical_not_of_false_feeds_and(wave32_fragment_andn1_saveexec_spv)) {
        printf("  [FAIL] s_andn1_saveexec_b32 did not compute old_EXEC & ~source\n");
        return 1;
    }
    printf("  [ok]   s_andn1_saveexec_b32 negates its source operand\n");

    const uint32_t wave32_fragment_readlane31[] = {
        0x7e140280u,                         // v_mov_b32 v10, 0
        0xd7600000u, 0x00013f0au,            // v_readlane_b32 s0, v10, 31
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto wave32_fragment_readlane31_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_readlane31, std::size(wave32_fragment_readlane31));
    if (wave32_fragment_readlane31_spv.empty() ||
        fragment_spirv_required_subgroup_size(wave32_fragment_readlane31_spv) != 32) {
        printf("  [FAIL] Wave32 fragment v_readlane requested the wrong subgroup size\n");
        return 1;
    }
    const uint32_t wave32_fragment_readlane32[] = {
        0x7e140280u,                         // v_mov_b32 v10, 0
        0xd7600000u, 0x0001410au,            // v_readlane_b32 s0, v10, 32 (out of range)
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_fragment_readlane32,
            std::size(wave32_fragment_readlane32)).empty()) {
        printf("  [FAIL] Wave32 fragment v_readlane accepted lane 32\n");
        return 1;
    }
    printf("  [ok]   Wave32 fragment v_readlane retains a 32-lane subgroup contract\n");

    const uint32_t fragment_cvt_i32_word_sdwa[] = {
        0x7e1a10f9u, 0x0006140du,            // v_cvt_i32_f32_sdwa v13,v13 WORD_0/PRESERVE
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    const auto fragment_cvt_i32_word_sdwa_spv = recompile_fragment(
        fragment_cvt_i32_word_sdwa, std::size(fragment_cvt_i32_word_sdwa));
    if (fragment_cvt_i32_word_sdwa_spv.empty() ||
        !type_result_ids_are_nonzero(fragment_cvt_i32_word_sdwa_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cvt_i32_word_sdwa_spv)) {
        printf("  [FAIL] WORD-preserving v_cvt_i32_f32_sdwa did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   WORD-preserving v_cvt_i32_f32_sdwa emits valid SPIR-V\n");

    const uint32_t wave32_vertex_exec[] = {
        0xbefe03c1u,                         // s_mov_b32 exec_lo, -1
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf80008cfu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xbf810000u,
    };
    const auto wave32_vertex_spv = recompile_vertex(
        wave32_vertex_exec, std::size(wave32_vertex_exec));
    if (!wave32_vertex_spv.empty()) {
        printf("  [FAIL] ungated vertex shader accepted Wave32 EXEC_LO restore\n");
        return 1;
    }
    printf("  [ok]   unproven graphics Wave32 mask operations remain fail-visible\n");

    // A one-word Wave32 saved-mask alias ends when that physical SGPR is reused as scalar data.
    // v_cndmask must not prefer the old bool after either an SOPK or SOP2 writer; without a numeric
    // B64 mask representation these deliberately reject instead of silently selecting the stale arm.
    const uint32_t wave32_mask_then_sopk[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0xb0000000u,                         // s_movk_i32 s0, 0
        0xb0010000u,                         // s_movk_i32 s1, 0
        0xd5010003u, 0x0001e8f2u,            // v_cndmask_b32_e64 v3, 1.0, 2.0, s[0:1]
        0x7e000280u, 0x7e020280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_mask_then_sopk, std::size(wave32_mask_then_sopk)).empty()) {
        printf("  [FAIL] SOPK scalar reuse retained a stale Wave32 mask alias\n");
        return 1;
    }
    const uint32_t wave32_mask_then_sop2[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0x87008080u,                         // s_and_b32 s0, 0, 0
        0xb0010000u,                         // s_movk_i32 s1, 0
        0xd5010003u, 0x0001e8f2u,            // v_cndmask_b32_e64 v3, 1.0, 2.0, s[0:1]
        0x7e000280u, 0x7e020280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_mask_then_sop2, std::size(wave32_mask_then_sop2)).empty()) {
        printf("  [FAIL] SOP2 scalar reuse retained a stale Wave32 mask alias\n");
        return 1;
    }
    const uint32_t wave32_sop2_reuse_control[] = {
        0xbe80037eu,                         // s_mov_b32 s0, exec_lo
        0x87008080u,                         // s_and_b32 s0, 0, 0
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (recompile_fragment_wave32_for_test(
            wave32_sop2_reuse_control, std::size(wave32_sop2_reuse_control)).empty()) {
        printf("  [FAIL] SOP2 scalar-reuse control shader did not recompile\n");
        return 1;
    }
    printf("  [ok]   scalar reuse invalidates saved Wave32 mask aliases\n");

    // A folded s_getpc_b64 has no runtime scalar SSA result, but it still overwrites both physical
    // SGPR words. Keep an independent valid embedded-table chain in the shader so the first getpc is
    // accepted, then prove that it kills the old B32 mask rather than letting v_cndmask see it.
    const uint32_t wave32_mask_then_folded_getpc[] = {
        0xbe80037eu,                         // pc0: s_mov_b32 s0, exec_lo
        0xbe801f00u,                         // pc1: s_getpc_b64 s[0:1] (overwrites saved mask)
        0xb0060010u,                         // pc2: s_movk_i32 s6, 16-byte table
        0xbe8703ffu, 0x10005004u,            // pc3: s_mov_b32 s7, V# config
        0xbe841f00u,                         // pc5: s_getpc_b64 s[4:5] (next byte 24)
        0x800404b4u,                         // pc6: s_add_u32 s4, 52, s4 (table byte 76)
        0x82050580u,                         // pc7: s_addc_u32 s5, 0, s5
        0x7e020280u,                         // pc8: v_mov_b32 v1, 0 (table byte offset)
        0xe0301000u, 0x80010101u,            // pc9: buffer_load_dword v1,v1,s[4:7]
        0xbf8c3f70u,                         // pc11: s_waitcnt vmcnt(0)
        0xd5010003u, 0x0001e8f2u,            // pc12: v_cndmask v3,1.0,2.0,s[0:1]
        0x7e000280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
        7u, 11u, 13u, 17u,
    };
    std::vector<uint32_t> wave32_folded_getpc_control(
        std::begin(wave32_mask_then_folded_getpc),
        std::end(wave32_mask_then_folded_getpc));
    wave32_folded_getpc_control[13] = 0x01a9e8f2u; // consume live VCC, not overwritten s[0:1]
    if (recompile_fragment_wave32_for_test(
            wave32_folded_getpc_control.data(),
            wave32_folded_getpc_control.size()).empty()) {
        printf("  [FAIL] folded s_getpc_b64 control shader did not recompile\n");
        return 1;
    }
    if (!recompile_fragment_wave32_for_test(
            wave32_mask_then_folded_getpc,
            std::size(wave32_mask_then_folded_getpc)).empty()) {
        printf("  [FAIL] folded s_getpc_b64 retained a stale Wave32 mask alias\n");
        return 1;
    }
    printf("  [ok]   folded s_getpc_b64 invalidates saved Wave32 mask aliases\n");

    // A no-else forward arm has a skipped predecessor at its merge. If only the taken arm creates a
    // B32 saved mask, the physical-word validity differs between predecessors and cannot be modeled
    // by a bool-value phi alone. Keep the post-merge mask consumer fail-visible.
    const uint32_t wave32_conditional_mask_save[] = {
        0xbe800380u,                         // pc0: s_mov_b32 s0, 0
        0xbf060000u,                         // pc1: s_cmp_eq_u32 s0, s0
        0xbf840001u,                         // pc2: s_cbranch_scc0 -> pc4
        0xbe82037eu,                         // pc3: s_mov_b32 s2, exec_lo (taken arm only)
        0xd5010003u, 0x0009e8f2u,            // pc4: v_cndmask v3,1.0,2.0,s[2:3]
        0x7e000280u, 0x7e020280u, 0x7e040280u,
        0xf800000fu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_fragment_wave32_for_test(
            wave32_conditional_mask_save,
            std::size(wave32_conditional_mask_save)).empty()) {
        printf("  [FAIL] forward-if leaked a path-dependent Wave32 mask lifetime\n");
        return 1;
    }
    printf("  [ok]   forward-if rejects path-dependent Wave32 mask lifetimes\n");

    const uint32_t scalar_abs_compute[] = {
        0xb000ffffu,                         // s_movk_i32 s0, -1
        0xbe813400u,                         // s_abs_i32 s1, s0
        0xbf850001u,                         // s_cbranch_scc1 +1
        0xbf800000u,                         // s_nop 0
        0xbf810000u,
    };
    ComputeShaderConfig scalar_abs_config;
    const auto scalar_abs_spv = recompile_compute(
        scalar_abs_compute, std::size(scalar_abs_compute), nullptr, scalar_abs_config);
    if (scalar_abs_spv.empty() || !type_result_ids_are_nonzero(scalar_abs_spv, nullptr) ||
        !phi_ids_are_nonzero(scalar_abs_spv)) {
        printf("  [FAIL] scalar s_abs_i32/SCC path did not recompile cleanly\n");
        return 1;
    }
    printf("  [ok]   scalar s_abs_i32 writes its result and SCC in valid SPIR-V\n");

    // Prosper does not expose an attached GPU system debugger to guest shaders, so COND_DBG_SYS is
    // permanently clear and s_cbranch_cdbgsys falls through. Astro's world-map NGG wrapper uses this
    // around its position export; rejecting it drops the complete draw despite ordinary hardware also
    // taking the fallthrough path outside a shader-debugging session.
    const uint32_t no_system_debugger_vertex[] = {
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xbf970001u,                         // s_cbranch_cdbgsys +1 (not taken)
        0xf80008cfu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xbf810000u,
    };
    const auto no_system_debugger_spv = recompile_vertex(
        no_system_debugger_vertex, std::size(no_system_debugger_vertex));
    if (no_system_debugger_spv.empty() ||
        !type_result_ids_are_nonzero(no_system_debugger_spv, nullptr) ||
        !phi_ids_are_nonzero(no_system_debugger_spv)) {
        printf("  [FAIL] s_cbranch_cdbgsys did not take the no-debugger fallthrough path\n");
        return 1;
    }
    printf("  [ok]   s_cbranch_cdbgsys falls through when no GPU debugger is exposed\n");

    const uint32_t unsupported_exec_hi[] = {
        0xbeff03c1u,                         // s_mov_b32 exec_hi, -1
        0x7e000280u, 0x7e020280u, 0x7e040280u, 0x7e0602f2u,
        0xf80008cfu, 0x03020100u,
        0xbf810000u,
    };
    if (!recompile_vertex(unsupported_exec_hi, std::size(unsupported_exec_hi)).empty()) {
        printf("  [FAIL] unsupported EXEC_HI B32 write was accepted\n");
        return 1;
    }
    printf("  [ok]   EXEC_HI B32 writes remain fail-closed\n");

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

    // A GS_ALLOC_REQ marker alone does not prove the one-lane model. An unrelated NGG shader that
    // reaches a wave population count must remain fail-closed rather than silently counting lane 0.
    const uint32_t ngg_mask_count[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0xBEEA04C1u,                         // s_mov_b64 vcc, -1
        0xBE80106Au,                         // s_bcnt1_i32_b64 s0, vcc
        0x7E000C00u,                         // v_cvt_f32_u32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    if (!recompile_vertex(ngg_mask_count,
                          sizeof(ngg_mask_count) / sizeof(ngg_mask_count[0])).empty()) {
        printf("  [FAIL] unproven NGG accepted one-lane mask population count\n");
        return 1;
    }
    printf("  [ok]   unproven NGG rejects one-lane mask population count\n");

    // In Astro's byte-exact one-lane NGG projection, find-first-one on a wave mask is exact:
    // the represented lane is bit zero, so an active mask returns 0 and an empty mask returns -1.
    // The test-only entry point exercises that lowering without broadening production acceptance.
    const uint32_t ngg_mask_ff1[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0xBEEA04C1u,                         // s_mov_b64 vcc, -1
        0xBE80146Au,                         // s_ff1_i32_b64 s0, vcc
        0x7E000C00u,                         // v_cvt_f32_u32 v0, s0
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_mask_ff1_spv = recompile_vertex_ngg_one_lane_for_test(
        ngg_mask_ff1, std::size(ngg_mask_ff1));
    if (ngg_mask_ff1_spv.empty() || !type_result_ids_are_nonzero(ngg_mask_ff1_spv, nullptr)) {
        printf("  [FAIL] one-lane NGG s_ff1_i32_b64 did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   one-lane NGG s_ff1_i32_b64 emits valid SPIR-V\n");

    // Astro's 7f5f world-map wrapper constructs a B64 wave mask in ordinary scalar DATA registers
    // and consumes it through v_cndmask_b32_e64. The one-lane projection represents lane zero, so
    // the condition is exactly bit zero of the pair's low dword. Other vertex shaders still reject
    // this wave-dependent form because they have no proven lane identity.
    const uint32_t ngg_scalar_data_mask[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0xBE8403FFu, 0xAAAAAAAAu,            // s_mov_b32 s4, 0xaaaaaaaa
        0xBE850304u,                         // s_mov_b32 s5, s4
        0xD5010005u, 0x00120AFFu, 0x00100800u, // exact v_cndmask_b32_e64 v5, lit, v5, s[4:5]
        0x7E000305u,                         // v_mov_b32 v0, v5
        0x7E020280u, 0x7E040280u, 0x7E0602F2u,
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_scalar_data_mask_spv = recompile_vertex_ngg_one_lane_for_test(
        ngg_scalar_data_mask, std::size(ngg_scalar_data_mask));
    if (ngg_scalar_data_mask_spv.empty() ||
        !type_result_ids_are_nonzero(ngg_scalar_data_mask_spv, nullptr)) {
        printf("  [FAIL] one-lane NGG scalar-data mask cndmask did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   one-lane NGG scalar-data mask cndmask emits valid SPIR-V\n");

    // Exercise the output-selection emitter through its explicit test hook. The production entry
    // point restricts every terminal NGG gate to the byte-exact Astro wrapper, as checked below.
    const uint32_t ngg_output_gate[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x7C3E0300u,                         // v_cmpx_tru_f32 (uniformly narrows no lanes)
        0xBF880002u,                         // s_cbranch_execz -> s_endpgm
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_output_gate_spv = recompile_vertex_terminal_ngg_gate_for_test(
        ngg_output_gate, sizeof(ngg_output_gate) / sizeof(ngg_output_gate[0]));
    if (ngg_output_gate_spv.empty() || !phi_ids_are_nonzero(ngg_output_gate_spv)) {
        printf("  [FAIL] terminal NGG compacted-vertex output gate did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   terminal NGG compacted-vertex output gate produces valid SPIR-V\n");

    // The real Astro wrapper does not export directly after its terminal CMPX gate: it computes an
    // LDS address and reloads the compacted vertex first. Those predicated register-only operations
    // are part of the same bounded gate and must not make its position export look unsafe.
    const uint32_t ngg_output_rebuild_gate[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x7C3E0300u,                         // v_cmpx_tru_f32
        0xBF880005u,                         // s_cbranch_execz -> s_endpgm
        0x7E000280u,                         // v_mov_b32 v0, 0 (address/value setup)
        0xD8D80000u, 0x00000000u,            // ds_read_b32 v0, v0
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_output_rebuild_spv = recompile_vertex_terminal_ngg_gate_for_test(
        ngg_output_rebuild_gate, std::size(ngg_output_rebuild_gate));
    if (ngg_output_rebuild_spv.empty() || !phi_ids_are_nonzero(ngg_output_rebuild_spv)) {
        printf("  [FAIL] terminal NGG LDS output-rebuild gate did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   terminal NGG LDS output-rebuild gate produces valid SPIR-V\n");

    // Astro Bot's first world-map wrapper rebuilds its surviving compacted output through a
    // shader-embedded constant table rather than LDS. The terminal suffix is still side-effect
    // free: scalar ALU constructs the PC-relative V#, MUBUF only reads the proven bounded table,
    // and the results feed POS0/PARAM0 before S_ENDPGM. This is the exact captured 54-dword program
    // plus the table tail required by those two loads.
    const uint32_t astro_worldmap_pcrel_output_gate[] = {
        0xbfa00003u, 0x93ebff03u, 0x00040018u, 0xbefe03c1u,
        0x9380ff02u, 0x00090016u, 0x9381ff02u, 0x0009000cu,
        0xbf8a0000u, 0xbf076b80u, 0xbf850004u, 0x8f6a8c00u,
        0x887c6a01u, 0xbf800000u, 0xbf900009u, 0x8f6a856bu,
        0xd7650001u, 0x0000d4c1u, 0x7da80200u, 0xbf880002u,
        0xf8000941u, 0x00000000u, 0xbf8cff0fu, 0xbefe03c1u,
        0x7da80201u, 0xbf88001bu, 0xd56a0000u, 0x00020affu,
        0xaaaaaaabu, 0xbe8303ffu, 0x10005004u, 0xb0020048u,
        0xbe801f00u, 0x800000ffu, 0x000000acu, 0x82010180u,
        0x2c000081u, 0xd7460000u, 0x04010300u, 0x4c000105u,
        0x34000083u, 0xd7460004u, 0x04010300u, 0xe0381000u,
        0x80000004u, 0xe0341010u, 0x80000404u, 0xbf8c3f71u,
        0xf80008cfu, 0x03020100u, 0xbf8c3f70u, 0xf8000203u,
        0x00000504u, 0xbf810000u,
        // Padding to byte offset 304, then the 72-byte constant table.
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u, 0xbf9f0000u,
        0xbf9f0000u,
        0x00000000u, 0xbf800000u, 0xbf800000u, 0x3f800000u,
        0x3f800000u, 0x00000000u, 0x3f800000u, 0x40400000u,
        0xbf800000u, 0x3f800000u, 0x3f800000u, 0x40000000u,
        0x3f800000u, 0xbf800000u, 0x40400000u, 0x3f800000u,
        0x3f800000u, 0x00000000u, 0xbf800000u,
    };
    const auto astro_worldmap_pcrel_output_spv = recompile_vertex(
        astro_worldmap_pcrel_output_gate, std::size(astro_worldmap_pcrel_output_gate));
    if (astro_worldmap_pcrel_output_spv.empty() ||
        !type_result_ids_are_nonzero(astro_worldmap_pcrel_output_spv, nullptr) ||
        !phi_ids_are_nonzero(astro_worldmap_pcrel_output_spv)) {
        printf("  [FAIL] captured Astro PC-relative NGG output gate did not emit valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   captured Astro PC-relative NGG output gate emits valid SPIR-V\n");

    // Astro's second world-map wrapper exports POS for a surviving compacted vertex, then a regular
    // VCC compare conditionally skips only the trailing PARAM exports. Supplying those otherwise-
    // undefined varyings in the one-lane projection is safe and must not drop the complete draw.
    const uint32_t ngg_output_vcc_tail[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x7C3E0300u,                         // v_cmpx_tru_f32
        0xBF880006u,                         // s_cbranch_execz -> s_endpgm
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0x7D8A5880u,                         // v_cmp_ne_u32 vcc, 0, v44
        0xBF860002u,                         // s_cbranch_vccnz -> s_endpgm
        0xF800020Fu, 0x03020100u,            // exp param0 v0,v1,v2,v3
        0xBF810000u,
    };
    const auto ngg_output_vcc_tail_spv = recompile_vertex_terminal_ngg_gate_for_test(
        ngg_output_vcc_tail, std::size(ngg_output_vcc_tail));
    if (ngg_output_vcc_tail_spv.empty() || !phi_ids_are_nonzero(ngg_output_vcc_tail_spv)) {
        printf("  [FAIL] terminal NGG VCC-gated PARAM tail did not produce valid SPIR-V\n");
        return 1;
    }
    printf("  [ok]   terminal NGG VCC-gated PARAM tail produces valid SPIR-V\n");

    if (!recompile_vertex(ngg_output_gate, std::size(ngg_output_gate)).empty()) {
        printf("  [FAIL] production accepted an unproven constant NGG output gate\n");
        return 1;
    }
    printf("  [ok]   production rejects unproven constant NGG output gates (including points)\n");

    // A per-vertex CMPX under the same superficial terminal shape can create mixed active/inactive
    // primitives. Without the byte-exact Astro wrapper/topology proof it must remain rejected.
    const uint32_t unproven_ngg_output_gate[] = {
        0xBF900009u,
        0x7DA80300u,                         // data-dependent v_cmpx_*
        0xBF880002u,
        0xF80008CFu, 0x03020100u,
        0xBF810000u,
    };
    if (!recompile_vertex(unproven_ngg_output_gate,
                          std::size(unproven_ngg_output_gate)).empty()) {
        printf("  [FAIL] unproven NGG accepted a mixed terminal output gate\n");
        return 1;
    }
    printf("  [ok]   unproven NGG mixed terminal output gate remains fail-closed\n");

    // Small inline B64 masks are also lane-sensitive. Merely resembling an NGG wrapper cannot opt a
    // shader into the captured Astro program's lane-zero projection.
    const uint32_t ngg_inline_mask[] = {
        0xBF900009u,                         // s_sendmsg GS_ALLOC_REQ (marks NGG)
        0x92EA8081u,                         // s_bfm_b64 vcc, 1, 0 (lane-zero mask)
        0xBEFE0481u,                         // s_mov_b64 exec, 1
        0xBEFE04C1u,                         // s_mov_b64 exec, -1 (restore before export)
        0xF80008CFu, 0x03020100u,            // exp pos0 v0,v1,v2,v3
        0xBF810000u,
    };
    if (!recompile_vertex(ngg_inline_mask,
                          sizeof(ngg_inline_mask) / sizeof(ngg_inline_mask[0])).empty()) {
        printf("  [FAIL] unproven NGG accepted lane-zero inline-mask semantics\n");
        return 1;
    }
    printf("  [ok]   unproven NGG rejects lane-zero inline-mask semantics\n");

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

    // Astro Bot's title materials contain large reducible pixel shaders whose forward branch
    // regions overlap rather than forming a lexical if-tree. The compact SSA structurizer must
    // remain conservative, but the per-invocation graphics CFG dispatcher can execute the exact
    // basic-block graph. This small crossing-region shape forces that fallback.
    const uint32_t fragment_cfg_dispatch[] = {
        0x7e040280u,                         // pc0:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc1:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc2:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc3:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc4:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc5:  s_cbranch_vccz -> pc9
        0x7c020300u,                         // pc6:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc7:  s_cbranch_vccz -> pc10 (crosses pc5 region)
        0x7e040281u,                         // pc8:  v_mov_b32 v2, 1
        0x7e060281u,                         // pc9:  v_mov_b32 v3, 1
        0x7c020300u,                         // pc10: v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc11: s_cbranch_vccz -> alternate export at pc15
        0xf800180fu, 0x05040302u,            // pc12: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820003u,                         // pc14: s_branch -> verified tail exit at pc18
        0xf800180fu, 0x05040302u,            // pc15: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc17: s_endpgm
        0xbf810000u,                         // pc18: branch-target s_endpgm
    };
    const auto fragment_cfg_spv = recompile_fragment(
        fragment_cfg_dispatch, std::size(fragment_cfg_dispatch));
    if (fragment_cfg_spv.empty() || !has_opcode(fragment_cfg_spv, 251) ||
        !type_result_ids_are_nonzero(fragment_cfg_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_spv)) {
        printf("  [FAIL] complex fragment CFG did not lower through a valid OpSwitch dispatcher\n");
        return 1;
    }
    const OutputStoreStats cfg_outputs = output_store_stats(fragment_cfg_spv);
    if (cfg_outputs.stores != 2 || cfg_outputs.stores_with_one_repeated_source != 0) {
        printf("  [FAIL] complex fragment CFG exports stale entry state or suppresses an alternate "
               "site (stores=%u repeated-source=%u)\n",
               cfg_outputs.stores, cfg_outputs.stores_with_one_repeated_source);
        return 1;
    }
    printf("  [ok]   complex fragment CFG exports active state from both alternate sites\n");

    // Astro Bot's second world-map material is Wave32 and carries saved one-word masks through the
    // same non-lexical branch graph. The explicit VOPC writes below intentionally target adjacent
    // s1 then s0: both independent masks must survive every dispatcher case and feed the later EXEC
    // restore. Treating either compare as a two-word write erases s1 or persists it as scalar zero.
    const uint32_t wave32_fragment_cfg_masks[] = {
        0x7d865cf9u, 0x06868104u,            // pc0:  v_cmp_le_u32_sdwa s1, s4, v46
        0x7d8654f9u, 0x06868004u,            // pc2:  v_cmp_le_u32_sdwa s0, s4, v42
        0xbefe097eu,                         // pc4:  s_wqm_b32 exec_lo, exec_lo
        0x7e040280u,                         // pc5:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc6:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc7:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc8:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc9:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc10: s_cbranch_vccz -> pc14
        0x7c020300u,                         // pc11: v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc12: s_cbranch_vccz -> pc15 (crosses pc10 region)
        0x7e040281u,                         // pc13: v_mov_b32 v2, 1
        0x7e060281u,                         // pc14: v_mov_b32 v3, 1
        0x7c020300u,                         // pc15: v_cmp_lt_f32 vcc, v0, v1
        0xbf860004u,                         // pc16: s_cbranch_vccz -> alternate restore at pc21
        0x877e0100u,                         // pc17: s_and_b32 exec_lo, s0, s1
        0xf800180fu, 0x05040302u,            // pc18: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820004u,                         // pc20: s_branch -> verified tail exit at pc25
        0x877e0100u,                         // pc21: alternate s_and_b32 exec_lo, s0, s1
        0xf800180fu, 0x05040302u,            // pc22: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc24: s_endpgm
        0xbf810000u,                         // pc25: branch-target s_endpgm
    };
    const auto wave32_fragment_cfg_spv = recompile_fragment_wave32_for_test(
        wave32_fragment_cfg_masks, std::size(wave32_fragment_cfg_masks));
    if (wave32_fragment_cfg_spv.empty() || !has_opcode(wave32_fragment_cfg_spv, 251) ||
        !type_result_ids_are_nonzero(wave32_fragment_cfg_spv, nullptr) ||
        !phi_ids_are_nonzero(wave32_fragment_cfg_spv)) {
        printf("  [FAIL] complex Wave32 fragment CFG lost saved-mask state across cases\n");
        return 1;
    }
    printf("  [ok]   complex Wave32 fragment CFG persists unambiguous saved-mask lifetimes\n");

    // The live world-map PS writes an explicit SGPR mask with VOPC, then compares that whole B64
    // wave mask against zero inside the same dispatcher. SCC is a wave vote, not this invocation's
    // mask bit; fragment pipelines enforce wave64 and can lower it to subgroup-any exactly.
    const uint32_t fragment_mask_compare_prelude[] = {
        0x7e120280u,                         // v_mov_b32 v9, 0
        0x7c0212f9u, 0x06868480u,            // v_cmp_lt_f32_sdwa s[4:5], 0, v9
        0xbf138004u,                         // s_cmp_lg_u64 s[4:5], 0
    };
    std::vector<uint32_t> fragment_cfg_mask_compare(
        std::begin(fragment_mask_compare_prelude), std::end(fragment_mask_compare_prelude));
    fragment_cfg_mask_compare.insert(fragment_cfg_mask_compare.end(),
        std::begin(fragment_cfg_dispatch), std::end(fragment_cfg_dispatch));
    const auto fragment_cfg_mask_compare_spv = recompile_fragment(
        fragment_cfg_mask_compare.data(), fragment_cfg_mask_compare.size());
    if (fragment_cfg_mask_compare_spv.empty() ||
        !has_opcode(fragment_cfg_mask_compare_spv, 251) ||
        !has_opcode(fragment_cfg_mask_compare_spv, 335) ||
        fragment_spirv_required_subgroup_size(fragment_cfg_mask_compare_spv) != 64 ||
        !type_result_ids_are_nonzero(fragment_cfg_mask_compare_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_mask_compare_spv)) {
        printf("  [FAIL] complex fragment CFG rejected a wave-mask zero comparison\n");
        return 1;
    }
    printf("  [ok]   complex fragment CFG lowers wave-mask comparisons to exact wave64 votes\n");

    // Astro Bot's world-map material PS reaches the same graphics CFG dispatcher with an
    // s_orn2_saveexec_b64 whose destination is VCC. Keep a saved EXEC source live across dispatcher
    // cases, update both the explicit VCC SGPR pair and the implicit VCC condition, then restore EXEC.
    // The crossing branch regions below force the fallback which rejected the live shader at this op.
    const uint32_t fragment_cfg_orn2_saveexec[] = {
        0xbe82047eu,                         // pc0:  s_mov_b64 s[2:3], exec
        0xbeea2802u,                         // pc1:  s_orn2_saveexec_b64 vcc, s[2:3]
        0xbefe046au,                         // pc2:  s_mov_b64 exec, vcc (restore saved EXEC)
        0x7e040280u,                         // pc3:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc4:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc5:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc6:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc7:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc8:  s_cbranch_vccz -> pc12
        0x7c020300u,                         // pc9:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc10: s_cbranch_vccz -> pc13 (crossing region)
        0x7e040281u,                         // pc11: v_mov_b32 v2, 1
        0x7e060281u,                         // pc12: v_mov_b32 v3, 1
        0x7c020300u,                         // pc13: v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc14: s_cbranch_vccz -> alternate export at pc18
        0xf800180fu, 0x05040302u,            // pc15: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820003u,                         // pc17: s_branch -> verified tail exit at pc21
        0xf800180fu, 0x05040302u,            // pc18: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc20: s_endpgm
        0xbf810000u,                         // pc21: branch-target s_endpgm
    };
    const auto fragment_cfg_orn2_spv = recompile_fragment(
        fragment_cfg_orn2_saveexec, std::size(fragment_cfg_orn2_saveexec));
    if (fragment_cfg_orn2_spv.empty() || !has_opcode(fragment_cfg_orn2_spv, 251) ||
        !type_result_ids_are_nonzero(fragment_cfg_orn2_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_orn2_spv)) {
        printf("  [FAIL] complex fragment CFG rejected s_orn2_saveexec_b64 VCC form\n");
        return 1;
    }
    printf("  [ok]   complex fragment CFG preserves Astro ORN2-saveexec VCC state\n");

    // Astro's world-map material PS folds a lane-local mask across every 16-lane hardware row.
    // These are the four exact live packets following ORN2-saveexec. They require subgroup shuffle
    // (not the derivative-based FLOAT quad-perm approximation) and a 64-lane fragment subgroup.
    const uint32_t fragment_dpp_row_or[] = {
        0x7e1402c1u,                         // v_mov_b32 v10, -1
        0x381414fau, 0xff01110au,            // v_or_b32_dpp v10,v10,v10 row_shr:1
        0x381414fau, 0xff01120au,            // row_shr:2
        0x381414fau, 0xff01140au,            // row_shr:4
        0x381414fau, 0xff01180au,            // row_shr:8
        0xd7781009u, 0x0305830au,            // v_permlanex16 v9,v10,-1,-1 BC=1
        0x3814130au,                         // v_or_b32 v10,v10,v9
        0xd7600000u, 0x00013f0au,            // v_readlane_b32 s0,v10,31
        0xd7600001u, 0x00017f0au,            // v_readlane_b32 s1,v10,63
        0x883f0100u,                         // s_or_b32 s63,s0,s1
        0xf800000fu, 0x0a0a0a0au,            // exp mrt0 v10,v10,v10,v10
        0xbf810000u,
    };
    const auto fragment_dpp_row_or_spv = recompile_fragment(
        fragment_dpp_row_or, std::size(fragment_dpp_row_or));
    if (fragment_dpp_row_or_spv.empty() ||
        !has_opcode(fragment_dpp_row_or_spv, 345) ||
        !has_builtin(fragment_dpp_row_or_spv, 41) ||
        fragment_spirv_required_subgroup_size(fragment_dpp_row_or_spv) != 64 ||
        !type_result_ids_are_nonzero(fragment_dpp_row_or_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_dpp_row_or_spv)) {
        printf("  [FAIL] Astro fragment DPP row-right OR reduction did not emit valid subgroup SPIR-V\n");
        return 1;
    }
    printf("  [ok]   Astro fragment DPP/PERMLANEX/readlane reduction uses exact wave64 shuffles\n");

    const uint32_t fragment_dpp_distinct_row_or[] = {
        0x7e000281u,                         // v_mov_b32 v0, 1
        0x7e020282u,                         // v_mov_b32 v1, 2
        0x7e0402ffu, 0x12345678u,            // v_mov_b32 v2, unique old destination
        0x380402fau, 0xff011100u,            // v_or_b32_dpp v2,v0,v1 row_shr:1 BC:0
        0xf800000fu, 0x02020202u,
        0xbf810000u,
    };
    const auto fragment_dpp_distinct_spv = recompile_fragment(
        fragment_dpp_distinct_row_or, std::size(fragment_dpp_distinct_row_or));
    const uint32_t fragment_dpp_features =
        fragment_spirv_required_subgroup_features(fragment_dpp_distinct_spv);
    if (fragment_dpp_distinct_spv.empty() ||
        !has_select_with_false_constant(fragment_dpp_distinct_spv, 0x12345678u) ||
        !(fragment_dpp_features & kFragmentSubgroupShuffle) ||
        fragment_subgroup_features_supported(
            fragment_dpp_features,
            kFragmentSubgroupVote | kFragmentSubgroupArithmetic)) {
        printf("  [FAIL] unbounded fragment DPP did not preserve VDST/gate subgroup shuffle\n");
        return 1;
    }
    printf("  [ok]   unbounded fragment DPP preserves VDST and requires host shuffle support\n");
    // #1474: partially-overlapping LOOPS in the fragment shell. Two back-edges whose ranges cross
    // without nesting (B's header lies inside A's body, B's back-edge outside it) are what the narrow
    // pattern structurizer calls unstructured and rejects. Since the graphics CFG dispatcher above
    // exists, that rejection is no longer the end of the road: the per-invocation dispatcher executes
    // the exact block graph, so the region lowers instead of dropping the draw.
    //
    // Two properties of this stream are worth stating, because both are easy to misread:
    //   * The crossing pair is SYNTACTIC only — pc7 is an unconditional s_branch, so pc8/pc9 (B's
    //     back-edge) are unreachable. detect_divergent_loops collects back-edges without a
    //     reachability filter, so the pair check still sees the overlap and rejects. If that pass
    //     ever gains reachability pruning, the accept assertion below fails for a GOOD reason.
    //   * The rejection is OVER-DETERMINED: pc3's execz targets 10, past A's exit_pc of 8 (the dword
    //     after A's back-edge, not A's exit target), so pass 2's "conditional jump past the loop" rule
    //     would reject this stream even with the overlap check deleted. Neither this assertion nor the
    //     pre-#1474 one isolates overlap as the sole cause of rejection.
    //
    // Both directions are pinned here, in the DEVICE-FREE test, rather than only in the
    // Vulkan-execution tests: those are gated on find_package(Vulkan) succeeding, and every CI job
    // that runs ctest either disables Vulkan discovery (Linux, Windows MinGW, macOS) or runs a
    // three-test seam subset (Windows App), so a guard living only there never runs in CI at all.
    const uint32_t overlapping_loops[] = {
        0xbe800380u,               //  0: s_mov_b32 s0, 0
        0x7e020284u,               //  1: v_mov_b32 v1, 4
        0x7da20200u,               //  2: A_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880006u,               //  3: s_cbranch_execz +6 -> 10 (export)
        0x7da20200u,               //  4: B_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880006u,               //  5: s_cbranch_execz +6 -> 12 (endpgm)
        0x81008100u,               //  6: s0++
        0xbf82fffau,               //  7: s_branch -6 -> 2  (A back-edge; A = [2,7])
        0x81008100u,               //  8: s0++
        0xbf82fffau,               //  9: s_branch -6 -> 4  (B back-edge; B = [4,9] crosses A)
        0xf800180fu, 0x05020302u,  // 10: exp mrt0
        0xbf810000u,               // 12: s_endpgm
    };
    const auto overlapping_spv =
        recompile_fragment(overlapping_loops, std::size(overlapping_loops));
    if (overlapping_spv.empty() || !has_opcode(overlapping_spv, 251) ||
        !type_result_ids_are_nonzero(overlapping_spv, nullptr) ||
        !phi_ids_are_nonzero(overlapping_spv)) {
        printf("  [FAIL] #1474: partially-overlapping fragment loops did not lower through a valid "
               "OpSwitch dispatcher\n");
        return 1;
    }
    // Lowering is not enough: a dispatcher that emitted the block graph but dropped the export would
    // satisfy every check above, and the device-side test only asserts the readback's size — that is
    // exactly the "silent skip drops real rendered content" failure the charter warns about. This
    // stream has one EXP site, so it must produce exactly one output store. The count is exact rather
    // than >= 1 so a DOUBLED export (which would write MRT0 twice) fails too; the neighbouring
    // two-site test asserting stores == 2 is the cross-check that this counts sites, not components.
    const OutputStoreStats overlapping_outputs = output_store_stats(overlapping_spv);
    if (overlapping_outputs.stores != 1) {
        printf("  [FAIL] #1474: dispatcher-lowered overlapping loops did not export exactly once "
               "(stores=%u)\n", overlapping_outputs.stores);
        return 1;
    }
    printf("  [ok]   #1474: partially-overlapping fragment loops lower through the CFG dispatcher, "
           "exporting exactly once\n");

    // The fail-visible backstop still has to work. A cross-lane MBCNT inside the same region
    // disqualifies the per-invocation dispatcher — inside a dispatcher case the lanes of one subgroup
    // sit at different guest blocks, so MBCNT's subgroup exclusive scan would be answering for a wave
    // that is not there. See the `reason=mbcnt-cross-lane` reject in rdna2_to_spirv.cpp; if graphics
    // ever gains a synchronized common phase that closes the gap, this assertion fails LOUDLY rather
    // than silently losing the reject coverage. With the narrow structurizer already rejecting, a
    // loud reject is the only remaining outcome. The compute-side #590 case keeps an s_barrier in its
    // region for the same purpose (test_rdna2_to_spirv.cpp).
    //
    // Branch offsets are re-based for the MBCNT's two dwords, and the MBCNT sits on the REACHABLE
    // path (pc5's fallthrough), not in the dead region above. Dropping it makes this stream lower,
    // which is what makes the guard real rather than decorative.
    const uint32_t overlapping_loops_cross_lane[] = {
        0xbe800380u,               //  0: s_mov_b32 s0, 0
        0x7e020284u,               //  1: v_mov_b32 v1, 4
        0x7da20200u,               //  2: A_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880008u,               //  3: s_cbranch_execz +8 -> 12 (export)
        0x7da20200u,               //  4: B_HDR: v_cmpx_lt_u32 s0, v1
        0xbf880008u,               //  5: s_cbranch_execz +8 -> 14 (endpgm)
        0xd7650004u, 0x000100c1u,  //  6: v_mbcnt_lo_u32_b32 v4, -1, 0  (cross-lane)
        0x81008100u,               //  8: s0++
        0xbf82fff8u,               //  9: s_branch -8 -> 2  (A back-edge; A = [2,9])
        0x81008100u,               // 10: s0++
        0xbf82fff8u,               // 11: s_branch -8 -> 4  (B back-edge; B = [4,11] crosses A)
        0xf800180fu, 0x05020302u,  // 12: exp mrt0
        0xbf810000u,               // 14: s_endpgm
    };
    if (!recompile_fragment(overlapping_loops_cross_lane,
                            std::size(overlapping_loops_cross_lane)).empty()) {
        printf("  [FAIL] #1474: cross-lane MBCNT inside an unstructured fragment region must "
               "REJECT, not lower through the per-invocation dispatcher\n");
        return 1;
    }
    printf("  [ok]   #1474: cross-lane op in an unstructured fragment region still rejects loudly\n");

    // The graphics CFG dispatcher must retain the fragment shell's already-proven alpha-test
    // linearization.  This reduced Astro Bot shape prefixes the crossing-region CFG above with a
    // survivor-mask SCC early-out.  The SCC from s_andn2_b64 is a whole-wave reduction, not an SSA
    // scalar boolean; the per-invocation translation drops that optimization, narrows EXEC, and
    // OpKills failed lanes at either export.  Treating the safe branch as a dispatcher terminator
    // instead poisoned SCC and rejected the otherwise-supported material shader.
    const uint32_t fragment_cfg_kill_dispatch[] = {
        0xbe82047eu,                         // pc0:  s_mov_b64 s[2:3], exec
        0x7c020300u,                         // pc1:  v_cmp_lt_f32 vcc, v0, v1
        0x8a826a02u,                         // pc2:  s_andn2_b64 s[2:3], s[2:3], vcc
        0xbf840012u,                         // pc3:  s_cbranch_scc0 -> pc22 (wave early-out)
        0xbefe0a02u,                         // pc4:  s_wqm_b64 exec, s[2:3]
        0x7e040280u,                         // pc5:  v_mov_b32 v2, 0
        0x7e060280u,                         // pc6:  v_mov_b32 v3, 0
        0x7e080280u,                         // pc7:  v_mov_b32 v4, 0
        0x7e0a0280u,                         // pc8:  v_mov_b32 v5, 0
        0x7c020300u,                         // pc9:  v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc10: s_cbranch_vccz -> pc14
        0x7c020300u,                         // pc11: v_cmp_lt_f32 vcc, v0, v1
        0xbf860002u,                         // pc12: s_cbranch_vccz -> pc15 (crossing region)
        0x7e040281u,                         // pc13: v_mov_b32 v2, 1
        0x7e060281u,                         // pc14: v_mov_b32 v3, 1
        0x7c020300u,                         // pc15: v_cmp_lt_f32 vcc, v0, v1
        0xbf860003u,                         // pc16: s_cbranch_vccz -> alternate export at pc20
        0xf800180fu, 0x05040302u,            // pc17: exp mrt0 v2, v3, v4, v5 done vm
        0xbf820003u,                         // pc19: s_branch -> verified tail exit at pc23
        0xf800180fu, 0x05040302u,            // pc20: alternate exp mrt0 v2, v3, v4, v5 done vm
        0xbf810000u,                         // pc22: s_endpgm
        0xbf810000u,                         // pc23: branch-target s_endpgm
    };
    const auto fragment_cfg_kill_spv = recompile_fragment(
        fragment_cfg_kill_dispatch, std::size(fragment_cfg_kill_dispatch));
    if (fragment_cfg_kill_spv.empty() || !has_opcode(fragment_cfg_kill_spv, 251) ||
        !has_opcode(fragment_cfg_kill_spv, 252) ||
        !type_result_ids_are_nonzero(fragment_cfg_kill_spv, nullptr) ||
        !phi_ids_are_nonzero(fragment_cfg_kill_spv)) {
        printf("  [FAIL] complex fragment CFG lost its proven alpha-test branch linearization\n");
        return 1;
    }
    printf("  [ok]   complex fragment CFG retains alpha-test discard linearization\n");

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

    // Astro's NGG vertex shader fetches Float16x4 records with an SGPR SOFFSET.  Even when the
    // descriptor stride is dword-aligned, that runtime offset can select either half of a dword;
    // all four components must be extracted relative to the complete byte address.  This exact
    // MUBUF shape previously hit [mubuf-unaligned] and dropped the draw.
    const uint32_t vs_fetch_half4_soffset[] = {
        0xb0050002u,                         // s_movk_i32 s5, 2
        0xe00c2000u, 0x05000100u,           // buffer_load_format_xyzw v[1:4], v0, s[0:3], s5 idxen
        0xf80008cfu, 0x04030201u,            // exp pos0 v1,v2,v3,v4
        0xbf810000u,
    };
    ShaderResourceTable rt_half4;
    ShaderResource vb_half4{};
    vb_half4.cls = ResourceClass::VertexBuffer;
    vb_half4.format = DataFormat::Float16;
    vb_half4.num_components = 4;
    vb_half4.binding = 5;
    vb_half4.stride = 36;
    vb_half4.fetch_pc = 1;
    vb_half4.fetch_index_mode = VertexFetchIndexMode::Shader;
    rt_half4.resources.push_back(vb_half4);
    const auto half4_spv = recompile_vertex(
        vs_fetch_half4_soffset, std::size(vs_fetch_half4_soffset), &rt_half4);
    if (half4_spv.empty() || half4_spv[0] != 0x07230203u) {
        printf("  [FAIL] Float16x4 vertex fetch with runtime SOFFSET did not recompile\n");
        return 1;
    }
    printf("  [ok]   Float16x4 vertex fetch handles a runtime SOFFSET\n");

    // Astro's world-map VS uses the same arbitrary byte-address shape for a three-component SNORM16
    // attribute. Its stride is dword-aligned, but the shader-computed SOFFSET can select either half
    // of a dword, and a component beginning at byte three must join the following dword.
    const uint32_t vs_fetch_snorm16x3_soffset[] = {
        0xb0040002u,                         // s_movk_i32 s4, 2
        0xe0082000u, 0x04000405u,            // buffer_load_format_xyz v[4:6], v5, s[0:3], s4 idxen
        0x7e0e02f2u,                         // v_mov_b32 v7, 1.0
        0xf80008cfu, 0x07060504u,            // exp pos0 v4,v5,v6,v7
        0xbf810000u,
    };
    ShaderResourceTable rt_snorm16x3;
    ShaderResource vb_snorm16x3{};
    vb_snorm16x3.cls = ResourceClass::VertexBuffer;
    vb_snorm16x3.format = DataFormat::Snorm16;
    vb_snorm16x3.num_components = 3;
    vb_snorm16x3.binding = 6;
    vb_snorm16x3.stride = 28;
    vb_snorm16x3.fetch_pc = 1;
    vb_snorm16x3.fetch_index_mode = VertexFetchIndexMode::Shader;
    rt_snorm16x3.resources.push_back(vb_snorm16x3);
    const auto snorm16x3_spv = recompile_vertex(
        vs_fetch_snorm16x3_soffset, std::size(vs_fetch_snorm16x3_soffset), &rt_snorm16x3);
    if (snorm16x3_spv.empty() || snorm16x3_spv[0] != 0x07230203u) {
        printf("  [FAIL] SNORM16x3 vertex fetch with runtime SOFFSET did not recompile\n");
        return 1;
    }
    printf("  [ok]   SNORM16x3 vertex fetch handles a runtime SOFFSET\n");

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

    // Astro Bot's exact visibility-image packet: exchange v9 with R32_UINT texel (v0,v1), GLC=1.
    // Both SPIR-V operations are essential: accepting the MIMG without a real texel pointer/atomic
    // would merely hide the rejection while dropping the image side effect.
    const uint32_t ps_image_atomic[] = {
        0x7e000280u, 0x7e020280u, 0x7e120280u,
        0xf03c2108u, 0x00000900u,
        0x7e000280u, 0x7e020309u, 0x7e040280u, 0x7e0602f2u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    ShaderResourceTable rt_atomic_image;
    uint32_t atomic_image_pixel = 0;
    { ShaderResource image{}; image.cls = ResourceClass::StorageImage;
      image.format = DataFormat::Uint32; image.num_components = 1;
      image.binding = 4; image.img_dim = 1; image.width = 1; image.height = 1;
      image.depth = 1; image.sgpr_base = 0; image.size = sizeof(atomic_image_pixel);
      image.host_data = reinterpret_cast<uint8_t*>(&atomic_image_pixel);
      image.host_data_size = sizeof(atomic_image_pixel); rt_atomic_image.resources.push_back(image); }
    const std::vector<uint32_t> atomic_image_spv = recompile_fragment(
        ps_image_atomic, std::size(ps_image_atomic), &rt_atomic_image);
    if (atomic_image_spv.empty() || !has_opcode(atomic_image_spv, 60u) ||
        !has_opcode(atomic_image_spv, 229u)) {
        printf("  [FAIL] image_atomic_swap did not lower through OpImageTexelPointer/OpAtomicExchange\n");
        return 1;
    }
    printf("  [ok]   image_atomic_swap lowers to a typed R32_UINT SPIR-V image atomic\n");

    // Astro Bot's world-map visibility kernel uses the adjacent GFX10 IMAGE_ATOMIC_ADD opcode.
    const uint32_t cs_image_atomic_add[] = {
        0x7e000280u, 0x7e020280u, 0x7e120281u,
        0xf0442108u, 0x00000900u,
        0xbf810000u,
    };
    ComputeShaderConfig atomic_add_config;
    atomic_add_config.local_x = 1;
    const std::vector<uint32_t> atomic_add_spv = recompile_compute(
        cs_image_atomic_add, std::size(cs_image_atomic_add),
        &rt_atomic_image, atomic_add_config);
    const DescriptorValidationReport atomic_add_report = validate_spirv_descriptor_interface(
        atomic_add_spv, &rt_atomic_image, 0, SpirvShaderStage::Compute, false);
    if (atomic_add_spv.empty() || has_opcode(atomic_add_spv, 60u) ||
        !has_opcode(atomic_add_spv, 65u) || !has_opcode(atomic_add_spv, 234u) ||
        !has_opcode(atomic_add_spv, 176u) || !has_opcode(atomic_add_spv, 167u) ||
        !has_opcode(atomic_add_spv, 250u) || !has_opcode(atomic_add_spv, 245u) ||
        !atomic_add_report.ok() || atomic_add_report.descriptors.size() != 1 ||
        atomic_add_report.descriptors[0].kind != SpirvDescriptorKind::StorageBuffer ||
        !atomic_add_report.descriptors[0].atomic_access) {
        printf("  [FAIL] compute image_atomic_add lacks its guarded atomic-buffer lowering\n");
        return 1;
    }
    printf("  [ok]   compute image_atomic_add uses a bounds-checked atomic buffer view\n");
    ShaderResourceTable undersized_atomic_image = rt_atomic_image;
    undersized_atomic_image.resources[0].size = sizeof(atomic_image_pixel) - 1;
    const DescriptorValidationReport undersized_atomic_report =
        validate_spirv_descriptor_interface(
            atomic_add_spv, &undersized_atomic_image, 0,
            SpirvShaderStage::Compute, false);
    if (undersized_atomic_report.ok()) {
        printf("  [FAIL] compute atomic-buffer view accepted an undersized image backing\n");
        return 1;
    }
    printf("  [ok]   compute atomic-buffer view rejects an undersized image backing\n");
    ShaderResourceTable compressed_atomic_image = rt_atomic_image;
    compressed_atomic_image.resources[0].compression_enabled = true;
    const DescriptorValidationReport compressed_atomic_report =
        validate_spirv_descriptor_interface(
            atomic_add_spv, &compressed_atomic_image, 0,
            SpirvShaderStage::Compute, false);
    if (compressed_atomic_report.ok() ||
        !recompile_compute(cs_image_atomic_add, std::size(cs_image_atomic_add),
                           &compressed_atomic_image, atomic_add_config).empty()) {
        printf("  [FAIL] compute atomic-buffer view accepted a DCC-compressed image\n");
        return 1;
    }
    ShaderResourceTable tail_atomic_image = rt_atomic_image;
    tail_atomic_image.resources[0].in_mip_tail = true;
    const DescriptorValidationReport tail_atomic_report =
        validate_spirv_descriptor_interface(
            atomic_add_spv, &tail_atomic_image, 0,
            SpirvShaderStage::Compute, false);
    if (tail_atomic_report.ok() ||
        !recompile_compute(cs_image_atomic_add, std::size(cs_image_atomic_add),
                           &tail_atomic_image, atomic_add_config).empty()) {
        printf("  [FAIL] compute atomic-buffer view accepted a mip-tail image\n");
        return 1;
    }
    printf("  [ok]   compute atomic-buffer view rejects compressed and mip-tail images\n");

    // Astro Bot's world-map traversal kernel uses the RTIP 1.1 BVH instruction with eleven NSA
    // address operands. It is lowered to ordinary SSBO loads and scalar ALU, so this remains usable
    // on Vulkan devices without a ray-query feature. Keep the gate exact: accepting a nearby MIMG
    // flag combination would silently assign the wrong hardware intersection contract.
    const uint32_t cs_bvh_intersect[] = {
        0xf1989f07u, 0x00040303u, 0x43440d3fu, 0x46424140u, 0x00004847u,
        0xbf810000u,
    };
    uint32_t bvh_node_words[32]{};
    ShaderResourceTable rt_bvh;
    { ShaderResource bvh{}; bvh.cls = ResourceClass::ConstantBuffer;
      bvh.format = DataFormat::Uint32; bvh.num_components = 1;
      bvh.binding = 4; bvh.size = sizeof(bvh_node_words); bvh.fetch_pc = 0;
      bvh.host_data = reinterpret_cast<uint8_t*>(bvh_node_words);
      bvh.host_data_size = sizeof(bvh_node_words); rt_bvh.resources.push_back(bvh); }
    ComputeShaderConfig bvh_config;
    bvh_config.local_x = 1;
    const std::vector<uint32_t> bvh_spv = recompile_compute(
        cs_bvh_intersect, std::size(cs_bvh_intersect), &rt_bvh, bvh_config);
    const DescriptorValidationReport bvh_report = validate_spirv_descriptor_interface(
        bvh_spv, &rt_bvh, 0, SpirvShaderStage::Compute, false);
    if (bvh_spv.empty() || !bvh_report.ok() || bvh_report.descriptors.size() != 1 ||
        bvh_report.descriptors[0].kind != SpirvDescriptorKind::StorageBuffer ||
        !has_opcode(bvh_spv, 61u) || !has_opcode(bvh_spv, 129u) ||
        !has_opcode(bvh_spv, 133u) || !has_opcode(bvh_spv, 169u)) {
        printf("  [FAIL] IMAGE_BVH_INTERSECT_RAY lacks its SSBO/ALU lowering\n");
        return 1;
    }
    if (!recompile_compute(cs_bvh_intersect, std::size(cs_bvh_intersect),
                           nullptr, bvh_config).empty()) {
        printf("  [FAIL] IMAGE_BVH_INTERSECT_RAY was accepted without its BVH bytes\n");
        return 1;
    }
    uint32_t unsupported_bvh[std::size(cs_bvh_intersect)];
    std::copy(std::begin(cs_bvh_intersect), std::end(cs_bvh_intersect), unsupported_bvh);
    unsupported_bvh[0] &= ~(1u << 15); // R128=0 has a different destination contract.
    if (!recompile_compute(unsupported_bvh, std::size(unsupported_bvh),
                           &rt_bvh, bvh_config).empty()) {
        printf("  [FAIL] unverified IMAGE_BVH_INTERSECT_RAY flags were accepted\n");
        return 1;
    }
    printf("  [ok]   Astro IMAGE_BVH_INTERSECT_RAY lowers through a bounded BVH SSBO\n");

    // Astro Bot's visibility kernel sanitizes a generated coordinate with an explicit-SDST
    // v_cmp_class_f32 SDWA (mask 3 = sNaN|qNaN), followed by v_cndmask reading s[8:9]. Rejecting
    // the compare used to discard the entire 1,954-dword world-map compute shader.
    const uint32_t cs_class_nan[] = {
        0x7e0202ffu, 0x7fc00000u,           // v_mov_b32 v1, qNaN
        0x7d1106f9u, 0x86068801u,           // v_cmp_class_f32_sdwa s8, v1, 3
        0xd5010000u, 0x00210101u,           // v_cndmask_b32_e64 v0, v1, 0, s[8:9]
        0xbf810000u,
    };
    const std::vector<uint32_t> class_nan_spv = recompile_valu(
        cs_class_nan, std::size(cs_class_nan), 1, 0, nullptr);
    if (class_nan_spv.empty() || !has_opcode(class_nan_spv, 199u) ||
        !has_opcode(class_nan_spv, 171u) || !has_opcode(class_nan_spv, 169u) ||
        !type_result_ids_are_nonzero(class_nan_spv, nullptr)) {
        printf("  [FAIL] v_cmp_class_f32 did not lower to raw IEEE class selection/compare\n");
        return 1;
    }
    printf("  [ok]   v_cmp_class_f32 lowers raw IEEE classes and feeds its explicit SGPR mask\n");
    const uint32_t cs_class_modifiers[] = {
        0x7e0202ffu, 0x7f800000u,           // v_mov_b32 v1, +Inf
        0x7d1108f9u, 0x86168801u,           // v_cmp_class_f32_sdwa s8, -v1, 4 (-Inf)
        0x7d1108f9u, 0x86268801u,           // v_cmp_class_f32_sdwa s8, |v1|, 4
        0xbf810000u,
    };
    const std::vector<uint32_t> class_modifier_spv = recompile_valu(
        cs_class_modifiers, std::size(cs_class_modifiers), 1, 0, nullptr);
    bool has_abs_sign_mask = false;
    for (uint32_t word : class_modifier_spv) has_abs_sign_mask |= word == 0x7fffffffu;
    if (class_modifier_spv.empty() || !has_opcode(class_modifier_spv, 198u) ||
        !has_abs_sign_mask || !type_result_ids_are_nonzero(class_modifier_spv, nullptr)) {
        printf("  [FAIL] v_cmp_class_f32 dropped raw ABS/NEG sign-bit modifiers\n");
        return 1;
    }
    printf("  [ok]   v_cmp_class_f32 applies ABS/NEG without canonicalizing raw NaN bits\n");

    // Astro Bot's world-map shader samples a 192-layer BC6H 2D array with explicit LOD. The layer
    // coordinate must survive in OpTypeImage so the live backend creates a matching 2D-array view.
    ShaderResourceTable rt_array;
    { ShaderResource texture{}; texture.cls = ResourceClass::Texture;
      texture.format = DataFormat::Bc6; texture.num_components = 3;
      texture.binding = 4; texture.sgpr_base = 0; texture.img_dim = 5;
      texture.width = 4; texture.height = 4; texture.depth = 2;
      texture.gpu_addr = 0x100000; texture.size = 32;
      rt_array.resources.push_back(texture); }
    const uint32_t cs_sample_array_l[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u, 0x7e060280u,
        0xf0900f28u, 0x00400000u,         // image_sample_l dim:2D_ARRAY [u,v,slice,lod]
        0xbf810000u,
    };
    const std::vector<uint32_t> array_l_spv = recompile_valu(
        cs_sample_array_l, std::size(cs_sample_array_l), 4, 0, &rt_array);
    const DescriptorValidationReport array_l_report = validate_spirv_descriptor_interface(
        array_l_spv, &rt_array, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_l_descriptor = nullptr;
    for (const auto& descriptor : array_l_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            array_l_descriptor = &descriptor;
    if (array_l_spv.empty() || !array_l_descriptor ||
        array_l_descriptor->image_dim != 1u || !array_l_descriptor->image_arrayed ||
        !array_l_descriptor->normalized_sampling) {
        printf("  [FAIL] 2D-array image_sample_l dropped its reflected layer contract\n");
        return 1;
    }
    printf("  [ok]   2D-array image_sample_l retains its layer coordinate and reflected view shape\n");

    // The same map kernel reads its two-layer RGBA atlas with the NSA SAMPLE_LZ form. Its third
    // coordinate is still a layer, despite the fixed level zero, and must select an array view too.
    const uint32_t cs_sample_array_lz[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u,
        0xf09c0f2au, 0x00400000u, 0x00000201u,
        0xbf810000u,
    };
    const std::vector<uint32_t> array_lz_spv = recompile_valu(
        cs_sample_array_lz, std::size(cs_sample_array_lz), 4, 0, &rt_array);
    const DescriptorValidationReport array_lz_report = validate_spirv_descriptor_interface(
        array_lz_spv, &rt_array, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_lz_descriptor = nullptr;
    for (const auto& descriptor : array_lz_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            array_lz_descriptor = &descriptor;
    if (array_lz_spv.empty() || !array_lz_descriptor ||
        array_lz_descriptor->image_dim != 1u || !array_lz_descriptor->image_arrayed ||
        !array_lz_descriptor->normalized_sampling) {
        printf("  [FAIL] 2D-array image_sample_lz dropped its reflected layer contract\n");
        return 1;
    }
    printf("  [ok]   2D-array image_sample_lz retains its layer coordinate and reflected view shape\n");

    // A single binding can be sampled with both ordinary SAMPLE and explicit SAMPLE_L. Compute has
    // no derivatives, so both use explicit LOD while preserving the common array coordinate/type.
    const uint32_t cs_sample_array_mixed[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u, 0x7e060280u,
        0xf0800f2au, 0x00400000u, 0x00000201u,
        0xf0900f2au, 0x00400000u, 0x00030201u,
        0xbf810000u,
    };
    const std::vector<uint32_t> array_mixed_spv = recompile_valu(
        cs_sample_array_mixed, std::size(cs_sample_array_mixed), 4, 0, &rt_array);
    const DescriptorValidationReport array_mixed_report = validate_spirv_descriptor_interface(
        array_mixed_spv, &rt_array, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_mixed_descriptor = nullptr;
    size_t array_mixed_texture_count = 0;
    for (const auto& descriptor : array_mixed_report.descriptors) {
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler) {
            array_mixed_descriptor = &descriptor;
            array_mixed_texture_count++;
        }
    }
    if (array_mixed_spv.empty() || array_mixed_texture_count != 1u ||
        !array_mixed_descriptor || !array_mixed_descriptor->image_arrayed) {
        printf("  [FAIL] mixed compute SAMPLE/SAMPLE_L produced incompatible image types\n");
        return 1;
    }
    printf("  [ok]   mixed compute SAMPLE/SAMPLE_L shares one 2D-array image contract\n");

    // The graphics renderer still exposes DIM=5 textures through its established base-slice 2D
    // view. Keep that descriptor non-arrayed, but consume SAMPLE_L's fourth address as the LOD
    // rather than mistaking the discarded third (slice) address for the LOD.
    const uint32_t ps_sample_array_l[] = {
        0x7e0002ffu, 0x3f000000u, 0x7e0202ffu, 0x3f000000u,
        0x7e0402ffu, 0x3f800000u, 0x7e060280u,
        0xf0900f28u, 0x00400000u,
        0xf800000fu, 0x03020100u, 0xbf810000u,
    };
    const std::vector<uint32_t> graphics_array_l_spv = recompile_fragment(
        ps_sample_array_l, std::size(ps_sample_array_l), &rt_array);
    const DescriptorValidationReport graphics_array_l_report =
        validate_spirv_descriptor_interface(
            graphics_array_l_spv, &rt_array, 0, SpirvShaderStage::Fragment);
    const SpirvDescriptorBinding* graphics_array_l_descriptor = nullptr;
    for (const auto& descriptor : graphics_array_l_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            graphics_array_l_descriptor = &descriptor;
    if (graphics_array_l_spv.empty() || !graphics_array_l_descriptor ||
        graphics_array_l_descriptor->image_arrayed ||
        !has_explicit_lod_constant(graphics_array_l_spv, 0u)) {
        printf("  [FAIL] graphics DIM=5 image_sample_l violated its base-slice 2D contract\n");
        return 1;
    }
    printf("  [ok]   graphics DIM=5 image_sample_l keeps a 2D view and its fourth-address LOD\n");

    // The visibility half of the same kernel comparison-samples a sixteen-layer shadow array.
    // Compute keeps its slice and performs the compare manually over a color-sampled array image.
    ShaderResourceTable rt_array_dref = rt_array;
    rt_array_dref.resources[0].depth_compare = true;
    rt_array_dref.resources[0].depth_compare_func = 4;
    const uint32_t cs_sample_array_c_lz[] = {
        0x7e1402f0u, 0x7e1602f0u, 0x7e1802f0u,
        0x7e1a02ffu, 0x3f800000u,
        0xf0bc012au, 0x0040050au, 0x000d0c0bu,
        0xbf810000u,
    };
    const std::vector<uint32_t> array_c_lz_spv = recompile_valu(
        cs_sample_array_c_lz, std::size(cs_sample_array_c_lz), 4, 0, &rt_array_dref);
    const DescriptorValidationReport array_c_lz_report = validate_spirv_descriptor_interface(
        array_c_lz_spv, &rt_array_dref, 0, SpirvShaderStage::Compute);
    const SpirvDescriptorBinding* array_c_lz_descriptor = nullptr;
    for (const auto& descriptor : array_c_lz_report.descriptors)
        if (descriptor.binding == 4u &&
            descriptor.kind == SpirvDescriptorKind::CombinedImageSampler)
            array_c_lz_descriptor = &descriptor;
    if (array_c_lz_spv.empty() || !array_c_lz_descriptor ||
        array_c_lz_descriptor->image_dim != 1u || !array_c_lz_descriptor->image_arrayed ||
        array_c_lz_descriptor->image_depth ||
        has_opcode(array_c_lz_spv, 90u)) {
        printf("  [FAIL] compute 2D-array image_sample_c_lz lost its manual array contract\n");
        return 1;
    }
    printf("  [ok]   compute 2D-array image_sample_c_lz preserves slice without a Dref sampler\n");

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

    // Transform feedback must decorate the final pre-rasterization stage. When interpolation needs
    // the generated GS, decorating only the VS produces real pixels but writes zero probe vertices.
    const std::vector<uint32_t> mixed_xfb_gs =
        recompile_interpolation_geometry(mixed_layout, true);
    constexpr uint32_t OpCapability = 17, OpExecutionMode = 16, OpMemberDecorate = 72;
    constexpr uint32_t CapTransformFeedback = 53, ExecutionModeXfb = 11;
    constexpr uint32_t DecOffset = 35, DecXfbBuffer = 36, DecXfbStride = 37;
    if (mixed_xfb_gs.empty() ||
        !has_instruction_operand(mixed_xfb_gs, OpCapability, 0, CapTransformFeedback) ||
        !has_instruction_operand(mixed_xfb_gs, OpExecutionMode, 1, ExecutionModeXfb) ||
        !has_instruction_operand(mixed_xfb_gs, OpMemberDecorate, 2, DecXfbBuffer) ||
        !has_instruction_operand(mixed_xfb_gs, OpMemberDecorate, 2, DecXfbStride) ||
        !has_instruction_operand(mixed_xfb_gs, OpMemberDecorate, 2, DecOffset)) {
        printf("  [FAIL] generated interpolation geometry does not own transform-feedback output\n");
        return 1;
    }
    printf("  [ok]   generated interpolation geometry can capture its final positions\n");

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
