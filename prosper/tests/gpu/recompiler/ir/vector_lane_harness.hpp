// Per-lane execution of ONE vector instruction through the real recompiler and the integer SPIR-V
// interpreter -- the harness every vector family's execution test calls.
//
// The program it builds, for `lanes` invocations of one compute dispatch:
//   v2 = S0[lane], v3 = S1[lane], v1 = old VDST[lane], v4 = active[lane]   (buffer loads)
//   v_cmpx_ne_u32 0, v4          EXEC = the lanes whose `active` word is nonzero
//   <the instruction under test, written as  op v1, v2, v3>
//   s_mov_b64 exec, -1
//   buffer_store_dword v1 -> out[lane]
// so every lane has its own operands, and an EXEC-inactive lane must come back with its OLD VDST.
// One buffer (binding 3, V# in s[8:11]) holds the four input arrays and receives the result.
#pragma once

#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace prosper::test::ir {

inline constexpr uint32_t kVdst = 1, kVsrc0 = 2, kVsrc1 = 3;   // the registers the op must name

// VOP2 `op v1, v2, v3` (SRC0 = VGPR v2 is operand 256 + 2).
inline uint32_t vop2(uint32_t op, uint32_t vdst = kVdst, uint32_t src0 = 256 + kVsrc0,
                     uint32_t vsrc1 = kVsrc1) {
    return (op << 25) | (vdst << 17) | (vsrc1 << 9) | src0;
}

// VOP1 op v1,v2, with the same per-lane source/destination used by the binary harness.
inline uint32_t vop1(uint32_t op, uint32_t vdst = kVdst, uint32_t src0 = 256 + kVsrc0) {
    return 0x7e000000u | (vdst << 17) | (op << 9) | src0;
}

struct LaneInputs {
    std::vector<uint32_t> s0, s1, old_d;
    std::vector<bool> active;
    size_t lanes() const { return s0.size(); }
};

// Edge values in the first lanes, then a fixed-seed LCG; roughly a quarter of the lanes inactive,
// including lane 0 and the last lane, so both EXEC edges are exercised.
inline LaneInputs lane_inputs(uint32_t seed, uint32_t lanes = 64) {
    const uint32_t edges[] = {0u, 1u, 31u, 32u, 33u, 0x7fffffffu, 0x80000000u, 0xffffffffu,
                              0x00ffffffu, 0x01000000u, 0xffff0000u, 0x0000ffffu};
    LaneInputs in;
    const auto next = [&] { return seed = seed * 1664525u + 1013904223u; };
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        const uint32_t e = static_cast<uint32_t>(std::size(edges));
        in.s0.push_back(lane < 2 * e ? edges[lane % e] : next());
        in.s1.push_back(lane < 2 * e ? edges[(lane * 5 + 3) % e] : next());
        in.old_d.push_back(0xd0000000u | lane);
        in.active.push_back(lane != 0 && lane != lanes - 1 && (next() >> 30) != 0);
    }
    return in;
}

struct LaneRun {
    std::vector<uint32_t> d;   // VDST per lane after the instruction; empty on failure
    std::string error;         // the recompiler's reject reason or the interpreter's error
};

inline uint64_t lane_harness_address() {
    static std::atomic<uint64_t> next{0x4937000000ull};
    return next.fetch_add(0x100, std::memory_order_relaxed);
}

// Shared by execution and the opt-in synthetic witness exporter: identical code and inputs.
struct LaneProgram {
    std::vector<uint32_t> code;
    prosper::gpu::ShaderResourceTable table;
    prosper::gpu::ComputeShaderConfig config;
};
inline LaneProgram lane_program(const std::vector<uint32_t>& op_words, uint32_t lanes,
                                uint32_t wave) {
    using namespace prosper::gpu;
    const auto load = [&](uint32_t vdata, uint32_t array) -> std::vector<uint32_t> {
        // buffer_load_dword vN, v0 (idxen), s[8:11], offset = array * lanes * 4 bytes
        return {0xe0302000u | (array * lanes * 4u), 0x80020000u | (vdata << 8)};
    };
    std::vector<uint32_t> code;
    for (const auto& words : {load(kVsrc0, 0), load(kVsrc1, 1), load(kVdst, 2), load(4, 3)})
        code.insert(code.end(), words.begin(), words.end());
    code.push_back(0xbf8c3f70u);   // s_waitcnt vmcnt(0)
    code.push_back(0x7daa0880u);   // v_cmpx_ne_u32 0, v4
    code.insert(code.end(), op_words.begin(), op_words.end());
    code.push_back(0xbefe04c1u);   // s_mov_b64 exec, -1
    code.insert(code.end(), {0xe0702000u, 0x80020100u, 0xbf810000u});   // store v1; s_endpgm

    ShaderResource r;
    r.cls = ResourceClass::ConstantBuffer;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.binding = 3;
    r.sgpr_base = 8;
    r.stride = 4;
    r.size = 4 * lanes * 4;
    ShaderResourceTable table;
    table.resources.push_back(r);
    ComputeShaderConfig config;
    config.local_x = config.threads_x = lanes;
    config.local_y = config.local_z = config.threads_y = config.threads_z = 1;
    config.wave_size = wave;
    config.dispatch_launch.known = ComputeDispatchLaunchState::kAll;
    config.dispatch_launch.initiator = 1u | (wave == 32 ? 1u << 15u : 0u);
    config.dispatch_launch.num_thread = {lanes, 1, 1};
    config.dispatch_launch.dimensions = {1, 1, 1};

    return {std::move(code), std::move(table), std::move(config)};
}

inline LaneRun run_vector_op_per_lane(const std::vector<uint32_t>& op_words, const LaneInputs& in,
                                      uint32_t wave = 64, bool half_rte_preserve_denorm = false) {
    using namespace prosper::gpu;
    const uint32_t lanes = static_cast<uint32_t>(in.lanes());
    const auto program = lane_program(op_words, lanes, wave);
    const auto& code = program.code;
    const uint64_t address = lane_harness_address();
    const auto module = recompile_compute(code.data(), code.size(), &program.table, program.config,
                                          {RecompileDiagnosticStage::Compute, address});
    if (module.empty()) return {{}, "refused: " + last_terminal_reject_reason(address)};
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d}) buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    vm.half_rte_preserve_denorm = half_rte_preserve_denorm;
    auto out = vm.run_buffers(lanes, {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(lanes);
    return {out, {}};
}

} // namespace prosper::test::ir
