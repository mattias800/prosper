// ngg_merged_lut_fixture.hpp -- inputs for the merged-NGG subgroup shell (#3135 P2), shared by the
// execution test (tests/gpu/execute/test_ngg_subgroup_shell.cpp) and tools/spv_validate.
//
// Kena's captured LUT producer: the linked program from tests/data, a hand-built stand-in for the
// resource table the live resource fold publishes for it, and that table's buffer contents. Plus a
// small synthetic two-wave program that pins the shell's own contract (launch words, GS_ALLOC_REQ,
// a guest binding 0 beside set-2 shell I/O). Header-only.
#pragma once

#include "gpu/execute/ngg_subgroup_plan.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <utility>
#include <vector>

namespace prosper::test::ngg {

using gpu::DataFormat;
using gpu::ResourceClass;
using gpu::ShaderResource;
using gpu::VertexFetchIndexMode;

// The capture's raw SPI_SHADER_PGM_RSRC2_GS: LDS_SIZE [26:19] = 17 granules = 2176 dwords,
// ES_VGPR_COMP_CNT = 3, USER_SGPR = 0.
inline constexpr uint32_t kKenaRsrc2Gs = 0x008b0000u;
inline constexpr uint32_t kKenaUserSgprs = 8;

inline std::vector<uint32_t> load_words(const std::filesystem::path& path) {
    std::vector<uint32_t> words;
    if (FILE* file = std::fopen(path.string().c_str(), "rb")) {
        uint32_t word = 0;
        while (std::fread(&word, 4, 1, file) == 1) words.push_back(word);
        std::fclose(file);
    }
    return words;
}

// The captured ES prolog and GS main linked as the hardware runs them: the prolog up to its
// `s_setpc_b64 s[6:7]`, then the main program. `data` is prosper/tests/data.
inline std::vector<uint32_t> kena_linked(const std::filesystem::path& data) {
    const auto prolog = load_words(data / "ngg_merged_es_prolog.bin");
    const auto main = load_words(data / "ngg_merged_gs_main.bin");
    const gpu::VertexPrologInfo info = gpu::rdna2_vertex_prolog_info(prolog.data(), prolog.size());
    const size_t span = gpu::rdna2_recompile_code_span(main.data(), main.size());
    if (!info.valid || !span) return {};
    std::vector<uint32_t> linked(prolog.begin(), prolog.begin() + info.prefix_dwords);
    linked.insert(linked.end(), main.begin(), main.begin() + span);
    return linked;
}

// The two packed attribute-fetch configuration words the ES prolog reads at pc 6 (slot, format,
// SOFFSET and an instance-rate bit per attribute). Both are vertex-rate.
inline constexpr std::array<uint32_t, 2> kAttributeConfig = {0u, 0x3b11a2b4u};

// A plausible vertex-buffer V# (base, 16-byte stride, 4 records, 32_32 float): what the guest
// patches is checked against nothing here, because each fetch resolves through its own table entry.
inline constexpr std::array<uint32_t, 4> kVertexSharp = {0x00300000u, 0x00100000u, 4u, 0x0002cfacu};

// The capture's resource table: two vertex streams (POS.xy and PARAM.xy halves of one 16-byte
// record per vertex) and three constant buffers. `vertices` sizes the vertex streams.
inline gpu::ShaderResourceTable kena_resources(uint32_t vertices) {
    gpu::ShaderResourceTable table;
    const auto add = [&](ResourceClass cls, uint32_t binding, uint32_t size, uint32_t components,
                         uint32_t fetch_pc, uint32_t sgpr) {
        ShaderResource resource;
        resource.cls = cls;
        resource.binding = binding;
        resource.size = size;
        resource.stride = 16;
        resource.format = DataFormat::Float32;
        resource.num_components = components;
        resource.fetch_pc = fetch_pc;
        resource.sgpr_base = sgpr;
        if (cls == ResourceClass::VertexBuffer)
            resource.fetch_index_mode = VertexFetchIndexMode::Vertex;
        table.resources.push_back(resource);
    };
    add(ResourceClass::ConstantBuffer, 2, 32, 4, UINT32_MAX, 8);
    add(ResourceClass::VertexBuffer, 3, 16 * vertices, 2, 0x34, 8);
    add(ResourceClass::VertexBuffer, 4, 16 * vertices, 2, 0x5b, 8);
    add(ResourceClass::ConstantBuffer, 5, 32, 4, 0x0c, UINT32_MAX);
    add(ResourceClass::ConstantBuffer, 6, 32, 4, 0x1a, UINT32_MAX);
    // The ES prolog's immediate two-word raw scalar load at pc 6 (s_load_dwordx2 s[20:21]) feeds a
    // later register-offset descriptor load, so it is an owned raw snapshot: a keyless fetch-PC
    // resource carrying the exact eight bytes the fold observed (#3865, #4578).
    ShaderResource snapshot;
    snapshot.cls = ResourceClass::ConstantBuffer;
    snapshot.binding = 7;
    snapshot.size = 8;
    snapshot.stride = 0;
    snapshot.format = DataFormat::Uint32;
    snapshot.num_components = 1;
    snapshot.fetch_pc = 6;
    snapshot.gpu_addr = 0x100000;
    snapshot.owned_raw_snapshot_bytes = 8;
    snapshot.host_data = reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(kAttributeConfig.data()));
    snapshot.host_data_size = 8;
    table.resources.push_back(snapshot);
    // The two V# the prolog loads from its descriptor table at a register offset (pc 33, pc 66)
    // and then patches as numbers before the fetch: raw register snapshots of the 16 bytes read.
    // The fetches themselves resolve through their fetch-PC vertex-buffer entries above.
    for (const auto& [pc, binding] : {std::pair<uint32_t, uint32_t>{33, 8}, {66, 9}}) {
        ShaderResource vsharp;
        vsharp.cls = ResourceClass::ConstantBuffer;
        vsharp.binding = binding;
        vsharp.size = 16;
        vsharp.stride = 0;
        vsharp.format = DataFormat::Uint32;
        vsharp.num_components = 1;
        vsharp.fetch_pc = pc;
        vsharp.gpu_addr = 0x200000 + 0x100 * pc;
        vsharp.raw_register_snapshot = true;
        vsharp.host_data = reinterpret_cast<uint8_t*>(const_cast<uint32_t*>(kVertexSharp.data()));
        vsharp.host_data_size = 16;
        table.resources.push_back(vsharp);
    }
    return table;
}

// The constant-buffer words bound at bindings 2, 5 and 6. Word 0 is the layer base the prolog adds to
// InstanceID (pc 12); words 4..7 are the attribute scale and bias it reads at pc 26 (identity).
inline constexpr std::array<uint32_t, 8> kKenaConstants = {
    0u, 0x3b11a2b4u, 0x43c80000u, 0xc3610000u, 0x3f800000u, 0x3f800000u, 0u, 0u};

// One 16-byte record per vertex: POS.xy then PARAM.xy, PARAM = normalized screen position.
inline std::vector<uint32_t> vertex_records(const std::vector<std::array<float, 2>>& positions) {
    std::vector<uint32_t> words;
    words.reserve(positions.size() * 4u + 2u);
    for (const auto& p : positions) {
        words.push_back(std::bit_cast<uint32_t>(p[0]));
        words.push_back(std::bit_cast<uint32_t>(p[1]));
        words.push_back(std::bit_cast<uint32_t>((p[0] + 1.0f) * 0.5f));
        words.push_back(std::bit_cast<uint32_t>((1.0f - p[1]) * 0.5f));
    }
    words.push_back(0);
    words.push_back(0);
    return words;
}

template <size_t N>
std::vector<uint32_t> vertex_records(const std::array<std::array<float, 2>, N>& positions) {
    return vertex_records(std::vector<std::array<float, 2>>(positions.begin(), positions.end()));
}

inline std::map<uint32_t, std::vector<uint32_t>>
kena_buffers(const std::vector<uint32_t>& records) {
    return {{2, {kKenaConstants.begin(), kKenaConstants.end()}},
            {3, std::vector<uint32_t>(records.begin() + 2, records.end())},
            {4, std::vector<uint32_t>(records.begin(), records.end() - 2)},
            {5, {kKenaConstants.begin(), kKenaConstants.end()}},
            {6, {kKenaConstants.begin(), kKenaConstants.end()}},
            {7, {kAttributeConfig[0], kAttributeConfig[1]}},
            {8, {kVertexSharp.begin(), kVertexSharp.end()}},
            {9, {kVertexSharp.begin(), kVertexSharp.end()}}};
}

inline constexpr std::array<std::array<float, 2>, 4> kLutQuad = {
    {{1, -1}, {1, 1}, {-1, -1}, {-1, 1}}};

inline gpu::NggSubgroupLimits kena_limits() {
    // KENA_STATUS.md: VGT_GS_ONCHIP_CNTL 0x10020040, GE_CNTL 0x8040, 192 / 3 / ITEMSIZE 4.
    return gpu::decode_ngg_subgroup_limits(0x10020040u, 0x8040u, 192u, 3u, 4u);
}

// ---- The synthetic two-wave contract ----

inline constexpr uint32_t kSyntheticCbuf = 0xabc00001u;

inline std::vector<uint32_t> synthetic_program(bool wave0_only = true, bool send = true,
                                               bool narrow_param = false) {
    std::vector<uint32_t> code = {
        0xbefe04c1u,   // s_mov_b64 exec, -1
        0x9394ff03u, 0x00040018u,   // s_bfe_u32 s20, s3, [27:24]: wave index
        0xbf068014u,   // s_cmp_eq_u32 s20, 0
        0xbf840002u,   // s_cbranch_scc0 +2 (skip the request outside wave 0)
        0xb07c3005u,   // s_movk_i32 m0, 0x3005: 5 vertices, 3 primitives
        0xbf900009u,   // s_sendmsg GS_ALLOC_REQ
        0xf4200544u, 0xfa000000u,   // s_buffer_load_dword s21, s[8:11], 0 (guest binding 0)
        0xbf8cc07fu,   // s_waitcnt lgkmcnt(0)
        0x7e140215u,   // v_mov_b32 v10, s21
        0x7e120302u,   // v_mov_b32 v9, v2
        0xf8000941u, 0x00000009u,   // exp prim v9
        0xf80000cfu, 0x03020100u,   // exp pos0 v0..v3
        0xf800020fu, 0x030a0805u,   // exp param0 v5, v8, v10, v3
        0xbf810000u,
    };
    if (!wave0_only) code[4] = 0xbf800000u;   // s_nop: every wave requests
    if (!send) code[6] = 0xbf800000u;
    if (narrow_param) {
        // v_mbcnt_lo_u32_b32 v11, -1, 0; v_cmpx_gt_u32 32, v11: lanes 32..63 drop out of EXEC.
        code.insert(code.begin() + 16, {0xd765000bu, 0x000100c1u, 0x7da816a0u});
    }
    return code;
}

inline gpu::ShaderResourceTable synthetic_resources() {
    gpu::ShaderResourceTable table;
    ShaderResource cbuf;
    cbuf.cls = ResourceClass::ConstantBuffer;
    cbuf.binding = 0;   // a guest binding the old set-0 shell buffers would have collided with
    cbuf.size = 16;
    cbuf.stride = 16;
    cbuf.num_components = 4;
    cbuf.fetch_pc = 7;
    cbuf.sgpr_base = 8;
    table.resources.push_back(cbuf);
    return table;
}

}   // namespace prosper::test::ngg
