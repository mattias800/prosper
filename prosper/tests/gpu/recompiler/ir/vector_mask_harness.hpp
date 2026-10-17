// The per-lane harness for the mask-reading and mask-writing VALU ops (v_cndmask_b32 and the carry
// ops, ir/ops/vector_cndmask.cpp and ir/ops/vector_carry.cpp). It wraps vector_lane_harness.hpp's
// program, so lanes have their own S0/S1/S2 and about a quarter of them are EXEC-inactive:
//   v_cmp_ne_u32_e64 <mask>, 0, v5     the selector mask v_cndmask_b32 reads, or
//   v_add_co_u32 v7, <mask>, v5, -1     the carry-in mask a _co_ci op reads (both: v5 != 0)
//   <the op under test>                 v1 = D, and a carry op writes its carry to <cout>
//   s_mov_b64 exec, -1
//   [v_cndmask_b32 v1, 0, 1, <cout>]    ReadMask only: v1 = the lane's carry bit, every lane
// so an EXEC-inactive lane keeps its old VDST and reads back 0 from the carry mask (ISA 3.9).
#pragma once

#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace prosper::test::ir {

constexpr uint32_t kCarryOut = 20, kCarryIn = 22;   // SGPR pairs s[20:21] and s[22:23]

// v_cmp_ne_u32_e64 <sdst>, 0, v5 (llvm-mc gfx1030: d4c5xxxx 00020a80).
inline std::vector<uint32_t> mask_from_s2(uint32_t sdst) {
    return {0xd4c50000u | sdst, 0x80u | ((256u + kVsrc2) << 9)};
}
// v_add_co_u32 v7, <sdst>, v5, -1 (llvm-mc gfx1030: d70fxx07 00018305): the carry is v5 != 0. A
// carry-in is produced the way guest code produces one, by an earlier carry op.
inline std::vector<uint32_t> carry_from_s2(uint32_t sdst) {
    return {0xd70f0007u | (sdst << 8), (256u + kVsrc2) | (0xc1u << 9)};
}
// s_mov_b64 exec, -1, then every lane stores its mask bit in v1. An SGPR pair is read with
// v_cndmask_b32_e64 v1, 0, 1, <mask>; VCC with v_mov_b32 v8, 1 and v_cndmask_b32_e32 v1, 0, v8,
// which reads the architectural VCC the way the VOP2 carry ops write it.
inline std::vector<uint32_t> read_mask(uint32_t mask) {
    if (mask == kVcc) return {kSMovExecAll, 0x7e100281u, 0x02021080u};
    return {kSMovExecAll, 0xd5010001u, 0x80u | (0x81u << 9) | (mask << 18)};
}
// VOP3 word 0 (VOP3B: SDST in [14:8]; VOP3A: ABS in [10:8]) and word 1 (SRC0-SRC2, NEG [31:29]).
inline std::vector<uint32_t> vop3_words(uint32_t opcode, uint32_t field_8, uint32_t src2,
                                        uint32_t neg = 0) {
    return {0xd4000000u | (opcode << 16) | (field_8 << 8) | kVdst,
            (256u + kVsrc0) | ((256u + kVsrc1) << 9) | (src2 << 18) | (neg << 29)};
}

inline std::vector<uint32_t> concat(std::initializer_list<std::vector<uint32_t>> parts) {
    std::vector<uint32_t> out;
    for (const auto& part : parts) out.insert(out.end(), part.begin(), part.end());
    return out;
}

// S2 for every lane: a mix of zero and nonzero words, so the mask the op reads differs per lane.
inline LaneInputs mask_lane_inputs(uint32_t seed) {
    LaneInputs in = lane_inputs(seed);
    uint32_t state = seed ^ 0x9e3779b9u;
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        state = state * 1664525u + 1013904223u;
        in.s2.push_back((state >> 28) < 7 ? 0u : state);
    }
    return in;
}

// One witness program and the wave sizes it is written for (a program the recompiler refuses in
// one wave size is left out there: every written program must replay).
struct MaskWitness {
    std::vector<uint32_t> words;
    std::vector<uint32_t> waves{64u, 32u};
};

// Writes each program as a replayable sidecar pair under the witness root `name`
// (PROSPER_IR_WITNESS_DIR when set). Every program loads v5. Returns the number written.
inline size_t export_mask_witness(const std::vector<MaskWitness>& programs, const char* name,
                                  uint64_t base_address) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root(name);
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        EXPECT_FALSE(error) << error.message();
    }
    EXPECT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump(name);
    if (!fs::create_directories(dump)) return 0;
    size_t count = 0;
    for (const auto& program : programs)
        for (uint32_t wave : program.waves) {
            const auto lp = lane_program(program.words, 64, wave, true);
            write_compare_witness_program(dump, base_address + count * 0x100, count, lp);
            ++count;
        }
    return count;
}

} // namespace prosper::test::ir
