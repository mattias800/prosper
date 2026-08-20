// test_rdna2_bvh.cpp -- the IMAGE_BVH_INTERSECT_RAY assertions, moved VERBATIM out of
// test_rdna2_to_spirv.cpp's main(). The block was self-contained: it referenced no declaration from
// main's scope, which is what made it movable without a rewrite.
//
// The invariant that makes the move checkable is the executed-assertion count. main() records the
// count before calling run_bvh_checks() and requires the delta to be exactly 8, so a block that
// failed to arrive fails the test instead of quietly shrinking the suite. Verified by mutation: an
// empty run_bvh_checks() exits 1 with "contributed 0 checks, expected 8".
//
// The declarations below this comment are the original file's preamble, replicated so this
// translation unit compiles. Most of it is unused here and is kept identical rather than pruned,
// so the two files' shared prologue stays comparable.

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/gta5_compute_cfg_fixture.hpp"
#include <algorithm>
#include <bit>
#include <set>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <map>
#include <tuple>
#include <vector>

using namespace prosper::gpu;

extern int fails;
// `checks` counts assertions ACTUALLY EXECUTED, which `fails` cannot. Without it, deleting a block
// of this file -- or failing to carry one across when someone eventually breaks these 12,787 lines
// into several translation units -- leaves the suite green with fewer assertions running, and
// nothing anywhere reports the difference. A silently smaller test is indistinguishable from a
// passing one, which is the failure mode this project's charter names most often.
extern int checks;
#define CHECK(c, m) do { ++checks; \
                         if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// IEEE float32 -> float16 for NORMAL, in-range, exactly-representable inputs (chosen in the tests so
// round-toward-zero (pkrtz) and round-to-nearest (packHalf2x16) agree). Not a general converter.
static uint16_t f32_to_f16_exact(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  e = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = (x & 0x7fffffu) >> 13;
    return (uint16_t)(sign | ((uint32_t)e << 10) | m);
}
static uint32_t bits_of(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
static uint32_t bitrev32(uint32_t v) { uint32_t r = 0; for (int i = 0; i < 32; i++) { r = (r << 1) | (v & 1u); v >>= 1; } return r; }
static size_t count_spirv_opcode(const std::vector<uint32_t>& spv, uint16_t opcode) {
    size_t matches = 0;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return 0;
        if ((spv[word] & 0xFFFFu) == opcode) ++matches;
        word += count;
    }
    return matches;
}
static size_t count_spirv_composite_extract_index(const std::vector<uint32_t>& spv,
                                                  uint32_t index) {
    constexpr uint16_t OpCompositeExtract = 81;
    size_t matches = 0;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return 0;
        if ((spv[word] & 0xffffu) == OpCompositeExtract && count == 5 &&
            spv[word + 4] == index)
            ++matches;
        word += count;
    }
    return matches;
}
static bool has_spirv_builtin(const std::vector<uint32_t>& spv, uint32_t builtin) {
    constexpr uint16_t OpDecorate = 71;
    constexpr uint32_t DecorationBuiltIn = 11;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return false;
        if ((spv[word] & 0xFFFFu) == OpDecorate && count >= 4 &&
            spv[word + 2] == DecorationBuiltIn && spv[word + 3] == builtin)
            return true;
        word += count;
    }
    return false;
}
static bool has_spirv_local_size(const std::vector<uint32_t>& spv,
                                 uint32_t x, uint32_t y, uint32_t z) {
    constexpr uint16_t OpExecutionMode = 16;
    constexpr uint32_t ExecutionModeLocalSize = 17;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return false;
        if ((spv[word] & 0xFFFFu) == OpExecutionMode && count == 6 &&
            spv[word + 2] == ExecutionModeLocalSize && spv[word + 3] == x &&
            spv[word + 4] == y && spv[word + 5] == z)
            return true;
        word += count;
    }
    return false;
}

static bool spirv_has_array_length(const std::vector<uint32_t>& spv,
                                   uint32_t expected_length) {
    constexpr uint16_t OpTypeArray = 28, OpConstant = 43;
    std::map<uint32_t, uint32_t> constants;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[word]);
        if (!count || word + count > spv.size()) return false;
        if (opcode == OpConstant && count == 4)
            constants[spv[word + 2]] = spv[word + 3];
        word += count;
    }
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[word]);
        if (!count || word + count > spv.size()) return false;
        if (opcode == OpTypeArray && count == 4) {
            const auto length = constants.find(spv[word + 3]);
            if (length != constants.end() && length->second == expected_length)
                return true;
        }
        word += count;
    }
    return false;
}

struct SpirvBufferAccessSummary {
    uint32_t variable = 0;
    bool aliased = false;
    bool coherent = false;
    size_t loads = 0;
    size_t volatile_loads = 0;
    size_t stores = 0;
    size_t volatile_stores = 0;
};

static SpirvBufferAccessSummary spirv_buffer_access_summary(
        const std::vector<uint32_t>& spv, uint32_t binding) {
    constexpr uint16_t OpVariable = 59, OpLoad = 61, OpStore = 62;
    constexpr uint16_t OpAccessChain = 65, OpDecorate = 71;
    constexpr uint32_t StorageBuffer = 12, DecorationAliased = 20;
    constexpr uint32_t DecorationCoherent = 23, DecorationBinding = 33;
    constexpr uint32_t MemoryAccessVolatile = 1;
    SpirvBufferAccessSummary out;
    std::map<uint32_t, uint32_t> pointer_base;
    std::set<uint32_t> storage_variables;
    std::set<uint32_t> aliased, coherent;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[word]);
        if (!count || word + count > spv.size()) return {};
        if (opcode == OpVariable && count == 4 && spv[word + 3] == StorageBuffer)
            storage_variables.insert(spv[word + 2]);
        if (opcode == OpDecorate && count >= 3) {
            if (spv[word + 2] == DecorationBinding && count >= 4 &&
                spv[word + 3] == binding)
                out.variable = spv[word + 1];
            if (spv[word + 2] == DecorationAliased) aliased.insert(spv[word + 1]);
            if (spv[word + 2] == DecorationCoherent) coherent.insert(spv[word + 1]);
        }
        if (opcode == OpAccessChain && count >= 4)
            pointer_base[spv[word + 2]] = spv[word + 3];
        word += count;
    }
    if (!storage_variables.count(out.variable)) out.variable = 0;
    out.aliased = aliased.count(out.variable) != 0;
    out.coherent = coherent.count(out.variable) != 0;
    auto root = [&](uint32_t pointer) {
        std::set<uint32_t> seen;
        while (pointer_base.count(pointer) && seen.insert(pointer).second)
            pointer = pointer_base[pointer];
        return pointer;
    };
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[word]);
        if (!count || word + count > spv.size()) return {};
        if (opcode == OpLoad && count >= 4 && root(spv[word + 3]) == out.variable) {
            ++out.loads;
            if (count >= 5 && (spv[word + 4] & MemoryAccessVolatile))
                ++out.volatile_loads;
        }
        if (opcode == OpStore && count >= 3 && root(spv[word + 1]) == out.variable) {
            ++out.stores;
            if (count >= 4 && (spv[word + 3] & MemoryAccessVolatile))
                ++out.volatile_stores;
        }
        word += count;
    }
    return out;
}

static bool spirv_storage_buffer_has_decoration(const std::vector<uint32_t>& spv,
                                                uint32_t binding,
                                                uint32_t decoration) {
    const SpirvBufferAccessSummary summary = spirv_buffer_access_summary(spv, binding);
    if (!summary.variable) return false;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        if (!count || word + count > spv.size()) return false;
        if ((spv[word] & 0xffffu) == 71u && count >= 3 &&
            spv[word + 1] == summary.variable && spv[word + 2] == decoration)
            return true;
        word += count;
    }
    return false;
}

static bool spirv_has_device_uniform_release_barrier(const std::vector<uint32_t>& spv) {
    constexpr uint16_t OpConstant = 43, OpMemoryBarrier = 225;
    constexpr uint32_t ScopeDevice = 1, MemorySemanticsUniformRelease = 0x44;
    std::map<uint32_t, uint32_t> constants;
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[word]);
        if (!count || word + count > spv.size()) return false;
        if (opcode == OpConstant && count == 4)
            constants[spv[word + 2]] = spv[word + 3];
        word += count;
    }
    for (size_t word = 5; word < spv.size();) {
        const uint32_t count = spv[word] >> 16;
        const uint16_t opcode = static_cast<uint16_t>(spv[word]);
        if (!count || word + count > spv.size()) return false;
        if (opcode == OpMemoryBarrier && count == 3 &&
            constants[spv[word + 1]] == ScopeDevice &&
            constants[spv[word + 2]] == MemorySemanticsUniformRelease)
            return true;
        word += count;
    }
    return false;
}


void run_bvh_checks() {
        const uint32_t bvh_code[] = {
            0xf1989f07u, 0x00040303u, 0x43440d3fu, 0x46424140u, 0x00004847u,
            0xbf810000u,
        };
        ShaderResourceTable bvh_rt;
        ShaderResource bvh{};
        bvh.cls = ResourceClass::ConstantBuffer;
        bvh.format = DataFormat::Uint32;
        bvh.num_components = 1;
        bvh.binding = 2;
        bvh.size = 128;
        bvh.fetch_pc = 0;
        bvh_rt.resources.push_back(bvh);

        constexpr uint32_t lanes = 64;
        constexpr uint32_t inputs_per_lane = 73;
        std::vector<float> bvh_inputs(lanes * inputs_per_lane, 0.0f);
        auto set_input_bits = [&](uint32_t lane, uint32_t vgpr, uint32_t bits) {
            bvh_inputs[lane * inputs_per_lane + vgpr] = std::bit_cast<float>(bits);
        };
        auto set_ray = [&](uint32_t lane, uint32_t node, float ox, float oy, float oz) {
            set_input_bits(lane, 3, node);
            bvh_inputs[lane * inputs_per_lane + 63] = 100.0f; // ray extent
            bvh_inputs[lane * inputs_per_lane + 13] = ox;
            bvh_inputs[lane * inputs_per_lane + 68] = oy;
            bvh_inputs[lane * inputs_per_lane + 67] = oz;
            bvh_inputs[lane * inputs_per_lane + 64] = 0.0f;
            bvh_inputs[lane * inputs_per_lane + 65] = 0.0f;
            bvh_inputs[lane * inputs_per_lane + 66] = 1.0f;
            bvh_inputs[lane * inputs_per_lane + 70] = std::numeric_limits<float>::infinity();
            bvh_inputs[lane * inputs_per_lane + 71] = std::numeric_limits<float>::infinity();
            bvh_inputs[lane * inputs_per_lane + 72] = 1.0f;
        };
        for (uint32_t lane = 0; lane < lanes; ++lane) set_ray(lane, 5u, 0.0f, 0.0f, -5.0f);

        alignas(256) static std::array<uint8_t, 256> null_bvh_bytes{};
        ShaderResourceTable null_bvh_rt = bvh_rt;
        null_bvh_rt.resources[0].gpu_addr = 0;
        null_bvh_rt.resources[0].size = static_cast<uint32_t>(null_bvh_bytes.size());
        null_bvh_rt.resources[0].host_data = null_bvh_bytes.data();
        null_bvh_rt.resources[0].host_data_size = null_bvh_bytes.size();
        const std::vector<uint32_t> null_bvh_module = recompile_valu(
            bvh_code, std::size(bvh_code), inputs_per_lane, 3, &null_bvh_rt);
        const std::vector<float> null_bvh_output = prosper::test::run_compute(
            null_bvh_module, bvh_inputs, lanes, lanes,
            std::vector<uint32_t>(null_bvh_bytes.size() / sizeof(uint32_t)));
        bool null_bvh_ok = !null_bvh_module.empty() && null_bvh_output.size() == lanes;
        for (float value : null_bvh_output)
            null_bvh_ok &= bits_of(value) == 0xffffffffu;
        CHECK(null_bvh_ok,
              "proven guarded null BVH returns architectural no-hit child IDs");

        std::vector<uint32_t> box_words(32, 0u);
        box_words[0] = 0x100u; box_words[1] = 0x200u;
        box_words[2] = 0x300u; box_words[3] = 0x400u;
        for (uint32_t child = 0; child < 4; ++child) {
            const float lo = child == 0 ? -1.0f : 10.0f + static_cast<float>(child);
            const float hi = child == 0 ?  1.0f : 11.0f + static_cast<float>(child);
            const uint32_t base = 4u + child * 6u;
            box_words[base + 0] = bits_of(lo); box_words[base + 1] = bits_of(lo);
            box_words[base + 2] = bits_of(lo); box_words[base + 3] = bits_of(hi);
            box_words[base + 4] = bits_of(hi); box_words[base + 5] = bits_of(hi);
        }
        const uint32_t box_expect[4] = {0x100u, 0xffffffffu, 0xffffffffu, 0xffffffffu};
        bool box_ok = true;
        for (uint32_t component = 0; component < 4; ++component) {
            const std::vector<uint32_t> module = recompile_valu(
                bvh_code, std::size(bvh_code), inputs_per_lane, 3u + component, &bvh_rt);
            const std::vector<float> output = prosper::test::run_compute(
                module, bvh_inputs, lanes, lanes, box_words);
            box_ok &= output.size() == lanes;
            for (uint32_t lane = 0; lane < output.size(); ++lane)
                box_ok &= bits_of(output[lane]) == box_expect[component];
        }
        CHECK(box_ok, "IMAGE_BVH_INTERSECT_RAY returns the four unsorted FP32-box child hits");

        // GTA V sets BOX_SORT_EN. Keep the same live instruction and node-production site, but make
        // every child hit at a deliberately shuffled distance. The false arm proves physical order;
        // toggling only descriptor semantics proves the production compare-swap path.
        const float shuffled_near[4] = {4.0f, 1.0f, 3.0f, 2.0f};
        for (uint32_t child = 0; child < 4; ++child) {
            const uint32_t base = 4u + child * 6u;
            box_words[base + 0] = bits_of(-1.0f);
            box_words[base + 1] = bits_of(-1.0f);
            box_words[base + 2] = bits_of(-5.0f + shuffled_near[child]);
            box_words[base + 3] = bits_of(1.0f);
            box_words[base + 4] = bits_of(1.0f);
            box_words[base + 5] = bits_of(-4.5f + shuffled_near[child]);
        }
        const uint32_t physical_expect[4] = {0x100u, 0x200u, 0x300u, 0x400u};
        const uint32_t sorted_expect[4] = {0x200u, 0x400u, 0x300u, 0x100u};
        auto box_order_matches = [&](const uint32_t expected[4]) {
            bool matches = true;
            for (uint32_t component = 0; component < 4; ++component) {
                const std::vector<uint32_t> module = recompile_valu(
                    bvh_code, std::size(bvh_code), inputs_per_lane, 3u + component, &bvh_rt);
                const std::vector<float> output = prosper::test::run_compute(
                    module, bvh_inputs, lanes, lanes, box_words);
                matches &= !module.empty() && output.size() == lanes;
                for (uint32_t lane = 0; lane < output.size(); ++lane)
                    matches &= bits_of(output[lane]) == expected[component];
            }
            return matches;
        };
        CHECK(box_order_matches(physical_expect),
              "BOX_SORT_EN=0 preserves physical box-child order");
        bvh_rt.resources[0].bvh_sort_enabled = true;
        CHECK(box_order_matches(sorted_expect),
              "BOX_SORT_EN=1 stably returns box children from nearest to furthest");
        bvh_rt.resources[0].bvh_sort_enabled = false;

        // Make the X near edge one float ULP beyond the Y far edge. BOX_GROW=0 must miss;
        // BOX_GROW=6 expands the far edge by six 2^-24 increments and retains the child.
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            set_ray(lane, 5u, 0.0f, 0.0f, 0.0f);
            bvh_inputs[lane * inputs_per_lane + 64] = 1.0f;
            bvh_inputs[lane * inputs_per_lane + 65] = 1.0f;
            bvh_inputs[lane * inputs_per_lane + 66] = 1.0f;
            bvh_inputs[lane * inputs_per_lane + 70] = 1.0f;
            bvh_inputs[lane * inputs_per_lane + 71] = 1.0f;
            bvh_inputs[lane * inputs_per_lane + 72] = 1.0f;
        }
        box_words[4] = 0x3f800001u; box_words[5] = bits_of(0.0f);
        box_words[6] = bits_of(0.0f); box_words[7] = bits_of(2.0f);
        box_words[8] = bits_of(1.0f); box_words[9] = bits_of(2.0f);
        const std::vector<uint32_t> edge_module = recompile_valu(
            bvh_code, std::size(bvh_code), inputs_per_lane, 3, &bvh_rt);
        const std::vector<float> edge_output = prosper::test::run_compute(
            edge_module, bvh_inputs, lanes, lanes, box_words);
        bool edge_zero_ok = edge_output.size() == lanes;
        for (uint32_t lane = 0; lane < edge_output.size(); ++lane)
            edge_zero_ok &= bits_of(edge_output[lane]) == 0xffffffffu;
        CHECK(edge_zero_ok, "IMAGE_BVH_INTERSECT_RAY honors zero box growth");

        bvh_rt.resources[0].bvh_box_grow = 6u;
        const std::vector<uint32_t> grown_edge_module = recompile_valu(
            bvh_code, std::size(bvh_code), inputs_per_lane, 3, &bvh_rt);
        const std::vector<float> grown_edge_output = prosper::test::run_compute(
            grown_edge_module, bvh_inputs, lanes, lanes, box_words);
        bool edge_grown_ok = grown_edge_output.size() == lanes;
        for (uint32_t lane = 0; lane < grown_edge_output.size(); ++lane)
            edge_grown_ok &= bits_of(grown_edge_output[lane]) == 0x100u;
        CHECK(edge_grown_ok,
              "IMAGE_BVH_INTERSECT_RAY applies descriptor-selected box growth");

        ShaderResourceTable bvh64_rt = bvh_rt;
        bvh64_rt.resources[0].size = 64u;
        bvh64_rt.resources[0].bvh_box_grow = 0u;

        // A 64-byte allocation cannot contain a complete FP32 box node. It must therefore return
        // no-hit even though the first child's ID and six bounds fit in the allocation and describe
        // a hit. This catches unsigned underflow in the 128-byte node bound: (64-128)/8 would make
        // every node offset appear valid while the individually-clamped loads conceal the OOB read.
        std::vector<uint32_t> short_box_words(16, 0u);
        short_box_words[0] = 0x100u;
        short_box_words[4] = bits_of(-1.0f); short_box_words[5] = bits_of(-1.0f);
        short_box_words[6] = bits_of(-1.0f); short_box_words[7] = bits_of( 1.0f);
        short_box_words[8] = bits_of( 1.0f); short_box_words[9] = bits_of( 1.0f);
        for (uint32_t lane = 0; lane < lanes; ++lane)
            set_ray(lane, 5u, 0.0f, 0.0f, -5.0f);
        bool short_box_ok = true;
        for (uint32_t component = 0; component < 4; ++component) {
            const std::vector<uint32_t> module = recompile_valu(
                bvh_code, std::size(bvh_code), inputs_per_lane, 3u + component, &bvh64_rt);
            const std::vector<float> output = module.empty() ? std::vector<float>{} :
                prosper::test::run_compute(
                    module, bvh_inputs, lanes, lanes, short_box_words);
            short_box_ok &= !module.empty() && output.size() == lanes;
            for (uint32_t lane = 0; lane < output.size(); ++lane)
                short_box_ok &= bits_of(output[lane]) == 0xffffffffu;
        }
        CHECK(short_box_ok,
              "64-byte IMAGE_BVH_INTERSECT_RAY rejects an incomplete FP32 box node");

        std::vector<uint32_t> tri_words(16, 0u);
        tri_words[0] = bits_of(9.0f); tri_words[1] = bits_of(9.0f); tri_words[2] = bits_of(9.0f);
        tri_words[3] = bits_of(0.0f); tri_words[4] = bits_of(0.0f); tri_words[5] = bits_of(0.0f); // V1
        tri_words[6] = bits_of(0.0f); tri_words[7] = bits_of(1.0f); tri_words[8] = bits_of(0.0f); // V2
        tri_words[9] = bits_of(1.0f); tri_words[10] = bits_of(0.0f); tri_words[11] = bits_of(0.0f); // V3
        tri_words[15] = 0x00000900u; // type-1 I=vertex1, J=vertex2
        for (uint32_t lane = 0; lane < lanes; ++lane) set_ray(lane, 1u, 0.25f, 0.25f, -1.0f);
        const std::vector<uint32_t> tri_module = recompile_valu(
            bvh_code, std::size(bvh_code), inputs_per_lane, 3, &bvh64_rt);
        const std::vector<float> tri_output = tri_module.empty() ? std::vector<float>{} :
            prosper::test::run_compute(
                tri_module, bvh_inputs, lanes, lanes, tri_words);
        bool tri_ok = tri_output.size() == lanes;
        for (uint32_t lane = 0; lane < tri_output.size(); ++lane)
            tri_ok &= bits_of(tri_output[lane]) == bits_of(-1.0f);
        CHECK(!tri_module.empty() && tri_ok,
              "64-byte IMAGE_BVH_INTERSECT_RAY type-1 triangle returns its exact t numerator");
    }
