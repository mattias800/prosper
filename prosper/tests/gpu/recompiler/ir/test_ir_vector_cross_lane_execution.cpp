// v_readlane_b32 (ir/ops/vector_cross_lane.cpp) per lane, through the real recompiler and the
// integer SPIR-V interpreter's exact native subgroup, in Wave64 and Wave32: lanes with differing
// data, a partial EXEC (readlane ignores it, so an inactive source lane is still read), SGPR and
// inline selectors, and selectors past the wave width (S1 % W). The program is
//   <harness loads; v_cmpx_ne_u32 0, v4>
//   [s_mov_b32 s21, selector]  v_readlane_b32 s20, v2, <s21 | inline>  v_mov_b32 v1, s20
// so an active lane stores the read scalar and an inactive lane keeps its old v1.
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ops/vector_cross_lane.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;

namespace {

constexpr uint32_t kReadDst = 20, kSelectorSgpr = 21;

struct Selector {
    bool inline_constant;   // an inline integer 0..64 instead of s21
    uint32_t value;
};

std::vector<uint32_t> readlane_words(Selector selector) {
    std::vector<uint32_t> words;
    uint32_t src1 = 128u + selector.value;   // inline integer
    if (!selector.inline_constant) {
        words.insert(words.end(), {0xbe8003ffu | (kSelectorSgpr << 16), selector.value});
        src1 = kSelectorSgpr;
    }
    words.insert(words.end(), {0xd7600000u | kReadDst, (256u + kVsrc0) | (src1 << 9)});
    words.push_back(0x7e000200u | (kVdst << 17) | kReadDst);   // v_mov_b32 v1, s20
    return words;
}

LaneProgram readlane_program(Selector selector, uint32_t wave, uint32_t native) {
    LaneProgram lp = lane_program(readlane_words(selector), wave, wave);
    lp.config.native_subgroup_size = native;
    return lp;
}

bool has_shuffle(const std::vector<uint32_t>& module) {
    for (size_t i = 5; i < module.size();) {
        const uint32_t count = module[i] >> 16;
        if (!count || count > module.size() - i) return false;
        if ((module[i] & 0xffffu) == 345u) return true;   // OpGroupNonUniformShuffle
        i += count;
    }
    return false;
}

struct ReadlaneRun {
    std::vector<uint32_t> d;
    std::string error;
    bool shuffled = false;
};
ReadlaneRun run_readlane(Selector selector, uint32_t wave,
                         const prosper::test::ir::LaneInputs& in) {
    const LaneProgram lp = readlane_program(selector, wave, wave);
    const uint64_t address = lane_harness_address();
    const auto module = recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config,
                                          {RecompileDiagnosticStage::Compute, address});
    if (module.empty()) return {{}, "refused: " + last_terminal_reject_reason(address)};
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (const bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = wave;
    auto out = vm.run_buffers(wave, {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(wave);
    return {out, {}, has_shuffle(module)};
}

std::vector<Selector> selectors(uint32_t wave) {
    return {{false, 0u},          {false, 5u},          {false, wave - 1u}, {false, wave + 5u},
            {false, 0xffffffffu}, {false, 0x80000021u}, {true, 0u},         {true, 7u},
            {true, 33u},          {true, 64u}};
}

std::string describe(Selector s, uint32_t wave) {
    return std::string(s.inline_constant ? "inline " : "s21 = ") + std::to_string(s.value) +
           " wave " + std::to_string(wave);
}

TEST(IrVectorCrossLaneExecution, ReadlaneMatchesTheIsaAndItsDefinitionPerLane) {
    for (const uint32_t wave : {64u, 32u}) {
        const prosper::test::ir::LaneInputs in = lane_inputs(0x5c1au + wave, wave);
        for (const Selector selector : selectors(wave)) {
            const ReadlaneRun run = run_readlane(selector, wave, in);
            ASSERT_TRUE(run.error.empty()) << describe(selector, wave) << ": " << run.error;
            EXPECT_TRUE(run.shuffled) << describe(selector, wave) << ": not the native route";
            // The definition, evaluated over the whole subgroup.
            const Block block = [&] {
                Block b;
                Builder ir(b);
                const Value s0 = ir.input(Type::U32), s1 = ir.input(Type::U32);
                readlane_b32(ir, s0, s1, wave);
                return b;
            }();
            std::vector<InvocationInputs> inputs;
            for (uint32_t l = 0; l < wave; ++l) inputs.push_back({in.s0[l], selector.value});
            const auto eval = evaluate_lanes(block, inputs);
            for (uint32_t l = 0; l < wave; ++l) {
                // Hand-worked from AMD 70648 V_READLANE_B32: lane = S1 % W, EXEC ignored.
                const uint32_t isa = in.s0[selector.value % wave];
                const uint32_t expected = in.active[l] ? isa : in.old_d[l];
                EXPECT_EQ(run.d[l], expected) << describe(selector, wave) << " lane " << l;
                EXPECT_EQ(eval[l][block.size() - 1], isa) << describe(selector, wave);
            }
        }
    }
}

// The harness's own positive control: the selected lane is EXEC-inactive and holds a value no
// active lane has, so a readlane that masked its source by EXEC (or read this lane) would fail.
TEST(IrVectorCrossLaneExecution, AnInactiveSourceLaneIsStillRead) {
    for (const uint32_t wave : {64u, 32u}) {
        const prosper::test::ir::LaneInputs in = lane_inputs(0x5c1au + wave, wave);
        ASSERT_FALSE(in.active[0]);
        ASSERT_FALSE(in.active[wave - 1]);
        for (const uint32_t source : {0u, wave - 1u}) {
            const ReadlaneRun run = run_readlane({false, source}, wave, in);
            ASSERT_TRUE(run.error.empty()) << run.error;
            uint32_t active_lanes = 0;
            for (uint32_t l = 0; l < wave; ++l) {
                if (!in.active[l]) continue;
                ++active_lanes;
                EXPECT_EQ(run.d[l], in.s0[source]) << "wave " << wave << " lane " << l;
                if (l != source) EXPECT_NE(in.s0[l], in.s0[source]);
            }
            EXPECT_GT(active_lanes, wave / 2);
        }
    }
}

// The synthetic witness corpus (HOW_TO_ADD_A_FAMILY.md, "Cross-lane operations"): every selector
// form in both wave sizes, once with the exact native subgroup and once without it (whatever
// route the recompiler then takes is gated too). Every program must replay.
TEST(IrVectorCrossLaneExecution, ExportsReplayableWitnessPrograms) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_cross_lane_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_cross_lane_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (const uint32_t wave : {64u, 32u})
        for (const uint32_t native : {wave, 0u})
            for (const Selector selector : selectors(wave)) {
                write_compare_witness_program(dump, 0x5c1a000000ull + count * 0x100, count,
                                              readlane_program(selector, wave, native));
                ++count;
            }
    EXPECT_EQ(count, 40u);
}

}   // namespace
