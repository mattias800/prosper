// The per-lane MASK harness every VOPC compare group's execution test calls (ir/ops/
// vector_compare*.cpp). It wraps vector_lane_harness.hpp's program, so lanes have their own S0/S1
// and about a quarter of them are EXEC-inactive, and reads the MASK the compare writes:
//   v_cmp_*  vcc / s[20:21], v2, v3      (VOPC e32, or VOP3 e64 with VCC or an SGPR destination)
//   s_mov_b64 exec, -1
//   v_cndmask_b32_e64 v1, 0, 1, <mask>   -> v1 is the lane's mask bit, for EVERY lane
// so an EXEC-inactive lane must read back 0 (ISA 3.9: VCC[n] = EXEC[n] & test). A V_CMPX
// (opcode + 0x10) instead narrows EXEC; `v_mov_b32 v1, 1` after it writes 1 exactly where
// EXEC & test holds, and every other lane keeps its old v1.
#pragma once

#include "gpu/recompiler/ir/ops/vector_compare.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <system_error>
#include <vector>

namespace prosper::test::ir {

using prosper::gpu::ir::Known;
using prosper::gpu::ir::VectorCompareDef;

constexpr uint32_t kVcc = 106, kSgprPair = 20;
constexpr uint32_t kSMovExecAll = 0xbefe04c1u;   // s_mov_b64 exec, -1
constexpr uint32_t kMovV1One = 0x7e020281u;      // v_mov_b32 v1, 1

enum class Form { VopcVcc, Vop3Sgpr, Vop3Vcc, CmpxVopc, CmpxVop3 };
constexpr Form kForms[] = {Form::VopcVcc, Form::Vop3Sgpr, Form::Vop3Vcc, Form::CmpxVopc,
                           Form::CmpxVop3};

inline bool is_cmpx(Form form) { return form == Form::CmpxVopc || form == Form::CmpxVop3; }

// VOPC e32: v_cmp_* vcc, v2, v3. VOP3 e64 (llvm-mc gfx1030): word 0 carries the opcode, the ABS
// bits [10:8] and SDST [7:0]; word 1 SRC0, SRC1 and NEG [31:29].
// `vsrc1` names SRC1's VGPR: a 64-bit group passes kVsrc2, so its sources are v[2:3] and v[5:6].
inline std::vector<uint32_t> compare_words(uint32_t op, Form form, uint32_t abs = 0,
                                           uint32_t neg = 0, uint32_t vsrc1 = kVsrc1) {
    const uint32_t opcode = is_cmpx(form) ? op + 0x10u : op;
    if (form == Form::VopcVcc || form == Form::CmpxVopc)
        return {0x7c000000u | (opcode << 17) | (vsrc1 << 9) | (256 + kVsrc0)};
    const uint32_t sdst = form == Form::Vop3Sgpr ? kSgprPair : form == Form::Vop3Vcc ? kVcc : 0x7e;
    return {0xd4000000u | (opcode << 16) | (abs << 8) | sdst,
            (256 + kVsrc0) | ((256 + vsrc1) << 9) | (neg << 29)};
}

// The instruction words under test: the compare, then a read of the mask it wrote into v1.
inline std::vector<uint32_t> compare_program(uint32_t op, Form form, uint32_t abs = 0,
                                             uint32_t neg = 0, uint32_t vsrc1 = kVsrc1) {
    auto words = compare_words(op, form, abs, neg, vsrc1);
    if (is_cmpx(form)) {
        words.push_back(kMovV1One);
        return words;
    }
    const uint32_t mask = form == Form::Vop3Sgpr ? kSgprPair : kVcc;
    words.push_back(kSMovExecAll);
    words.push_back(0xd5010001u);                    // v_cndmask_b32_e64 v1, 0, 1, <mask>
    words.push_back(0x80u | (0x81u << 9) | (mask << 18));
    return words;
}

// What v1 holds after `program`, for one lane whose test result is `test`.
inline uint32_t expected_mask_read(Form form, bool active, bool test, uint32_t old_d) {
    if (is_cmpx(form)) return active && test ? 1u : old_d;
    return active && test ? 1u : 0u;
}


inline void check_compare_run(const VectorCompareDef& def, Form form, uint32_t wave, const LaneInputs& in) {
    const auto run = run_vector_op_per_lane(compare_program(def.opcode, form), in, wave);
    ASSERT_TRUE(run.error.empty()) << run.error;
    ASSERT_EQ(run.d.size(), in.lanes());
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        const Known test = prosper::gpu::ir::eval_vector_compare(def, in.s0[lane], in.s1[lane]);
        ASSERT_TRUE(test.has_value()) << "lane " << lane;
        EXPECT_EQ(run.d[lane], expected_mask_read(form, in.active[lane], *test != 0, in.old_d[lane]))
            << "form " << int(form) << " wave " << wave << " lane " << lane << " s0 " << std::hex
            << in.s0[lane] << " s1 " << in.s1[lane] << " active " << in.active[lane];
    }
}

// The witness root for `name`: PROSPER_IR_WITNESS_DIR when set, otherwise a scratch directory.
inline std::filesystem::path compare_witness_root(const char* name) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    return configured ? std::filesystem::path(configured) : prosper_test::test_scratch_dir() / name;
}
inline std::filesystem::path compare_witness_dump(const char* name) {
    return compare_witness_root(name) / "refused_shaders_20000101-000000Z_synthetic";
}

// Writes one program as a replayable sidecar pair in `dump` and checks that it replays.
inline void write_compare_witness_program(const std::filesystem::path& dump, uint64_t address,
                                          size_t index, const LaneProgram& lp) {
    const std::filesystem::path stem =
        dump / ("cs_" + std::to_string(address) + "_" + std::to_string(index));
    std::ofstream bin(stem.string() + ".bin", std::ios::binary);
    bin.write(reinterpret_cast<const char*>(lp.code.data()),
              static_cast<std::streamsize>(lp.code.size() * sizeof(uint32_t)));
    bin.close();
    EXPECT_TRUE(bin.good());
    const std::filesystem::path inputs = stem.string() + ".inputs";
    EXPECT_TRUE(prosper::gpu::write_refused_compute_inputs(inputs, address, &lp.table, lp.config,
                                                           "synthetic-witness (no live refusal)"));
    const auto read = prosper::gpu::read_refused_shader_inputs(inputs);
    EXPECT_TRUE(read.incomplete.empty()) << read.incomplete;
    const auto replay = prosper::gpu::replay_refused_shader_inputs(lp.code, read);
    EXPECT_TRUE(replay.produced) << replay.reject << " at " << address;
}

// Writes every definition of `defs` in every Form and both wave sizes as replayable sidecars, for
// the corpus gate (HOW_TO_ADD_A_FAMILY.md, "synthetic witness corpus"). PROSPER_IR_WITNESS_DIR
// names the output directory; without it the export goes to scratch, which still proves every
// program replays. Returns the number of programs written. A pair_sources definition is written
// with its 64-bit sources v[2:3] and v[5:6], and the program loads v5 and v6.
inline size_t export_compare_witness(std::span<const VectorCompareDef> defs, const char* name,
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
    for (const auto& def : defs)
        for (Form form : kForms)
            for (uint32_t wave : {64u, 32u}) {
                const uint32_t vsrc1 = def.pair_sources ? kVsrc2 : kVsrc1;
                const auto lp = lane_program(compare_program(def.opcode, form, 0, 0, vsrc1), 64,
                                             wave, def.pair_sources, def.pair_sources);
                write_compare_witness_program(dump, base_address + count * 0x100, count, lp);
                ++count;
            }
    return count;
}

} // namespace prosper::test::ir
