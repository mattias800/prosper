// Regression #5377: AMD 70648 Table 90 FI=0 reads zero from an
// inactive source. Inputs are all loaded before EXEC narrows; values remain genuine.
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/dpp16_source_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <system_error>
#include <cstdint>
#include <iostream>

namespace {
using namespace prosper::test::ir;
LaneInputs inputs(uint32_t wave, bool partial) {
    LaneInputs in;
    for (uint32_t lane = 0; lane < wave; ++lane) {
        in.s0.push_back(0x10010000u + lane);
        in.s1.push_back(0x20020000u + lane);
        in.old_d.push_back(0xd0000000u + lane);
        in.active.push_back(!partial || (lane % 4) != 0);
    }
    return in;
}
std::filesystem::path stem_for(uint32_t ctrl, bool partial, uint32_t wave) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the launch environment is immutable.
    const char* root = std::getenv("PROSPER_DPP_PREFLIGHT_ARTIFACTS");
    if (!root) return {};
    return std::filesystem::path(root) / ("w" + std::to_string(wave) + "-ctrl" +
                                          std::to_string(ctrl) + (partial ? "-partial" : "-full"));
}
bool save_words(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()),
               static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    file.close();
    return file.good();
}
LaneRun run_native(const std::vector<uint32_t>& words, const LaneInputs& in, uint32_t wave) {
    using namespace prosper::gpu;
    auto lp = lane_program(words, wave, wave);
    lp.config.native_subgroup_size = wave;
    const uint64_t address = lane_harness_address();
    const auto module = recompile_compute(lp.code.data(), lp.code.size(), &lp.table, lp.config,
                                          {RecompileDiagnosticStage::Compute, address});
    if (module.empty()) return {{}, "refused: " + last_terminal_reject_reason(address)};
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (const bool active : in.active) buffer.push_back(active ? 1u : 0u);
    const auto stem = stem_for((words[1] >> 8) & 0x1ffu, !in.active[0], wave);
    if (!stem.empty()) {
        if (!save_words(stem.string() + ".bin", lp.code) ||
            !save_words(stem.string() + ".spv", module) ||
            !save_words(stem.string() + ".buffer", buffer))
            return {{}, "save failed"};
        if (!write_refused_compute_inputs(stem.string() + ".inputs", address, &lp.table, lp.config,
                                          "synthetic diagnostic (accepted by base)"))
            return {{}, "sidecar save failed"};
        const auto read = read_refused_shader_inputs(stem.string() + ".inputs");
        const auto replay = replay_refused_shader_inputs(lp.code, read);
        if (!read.incomplete.empty() || !replay.produced || replay.spirv != module)
            return {{}, "saved sidecar replay mismatch: " + replay.reject};
    }
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = wave;
    auto out = vm.run_buffers(wave, {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(wave);
    return {out, {}};
}
void check_mov(uint32_t ctrl, bool partial) {
    for (uint32_t wave : {32u, 64u}) {
        SCOPED_TRACE(::testing::Message() << "wave=" << wave << " ctrl=" << ctrl);
        const auto in = inputs(wave, partial);
        const auto out = run_native({vop1(1, 1, 250), 0xff080002u | (ctrl << 8)}, in, wave);
        ASSERT_TRUE(out.error.empty()) << out.error;
        ASSERT_EQ(out.d.size(), wave);
        uint32_t differences = 0;
        std::vector<uint32_t> expected_lanes;
        for (uint32_t lane = 0; lane < wave; ++lane) {
            uint32_t expected = in.old_d[lane];
            if (in.active[lane]) {
                if (ctrl == 0x111u) {
                    expected = (lane % 16 != 0 && in.active[lane - 1]) ? in.s0[lane - 1] : 0;
                } else {
                    // QUAD_PERM:[1,0,3,2] maps every lane to its pair.
                    expected = in.active[lane ^ 1u] ? in.s0[lane ^ 1u] : 0;
                }
            }
            expected_lanes.push_back(expected);
            if (out.d[lane] != expected) ++differences;
            EXPECT_EQ(out.d[lane], expected) << "lane=" << lane;
        }
        const auto stem = stem_for(ctrl, partial, wave);
        if (!stem.empty()) {
            EXPECT_TRUE(save_words(stem.string() + ".expected", expected_lanes));
            EXPECT_TRUE(save_words(stem.string() + ".actual", out.d));
        }
        std::cout << "DPP_PREFLIGHT wave=" << wave << " ctrl=" << ctrl << " partial=" << partial
                  << " differences=" << differences << "\n";
    }
}
TEST(IrDpp16InactiveSource, RowShrAllActive) {
    check_mov(0x111u, false);
}
TEST(IrDpp16InactiveSource, RowShrInactiveSource) {
    check_mov(0x111u, true);
}
TEST(IrDpp16InactiveSource, QuadPermAllActive) {
    check_mov(0xb1u, false);
}
TEST(IrDpp16InactiveSource, QuadPermInactiveSource) {
    check_mov(0xb1u, true);
}
}   // namespace

namespace {
namespace cases = prosper::test::ir::dpp16_source;
using namespace prosper::gpu;

void check_case(const cases::Case& c, const std::vector<uint32_t>& module) {
    ASSERT_FALSE(module.empty());
    for (uint32_t mask : {0u, 1u, 2u}) {
        const auto in = cases::inputs(c, mask);
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = c.wave;
        auto got = vm.run_buffers(c.wave, {{3, cases::buffer(in)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        got.resize(c.wave);
        EXPECT_EQ(got, cases::expected(c, in)) << "mask=" << mask;
    }
}

void check_catalog(bool rows) {
    for (const auto& c : cases::cases()) {
        if ((c.control >= 0x111u) != rows) continue;
        SCOPED_TRACE(::testing::Message()
                     << c.wave << ':' << c.control << ':' << int(c.op) << ':' << c.bound);
        const auto p = cases::program(c);
        const auto inst = rdna2_decode_one(p.code.data() + 10, p.code.size() - 10);
        ASSERT_TRUE(inst.has_dpp);
        ASSERT_FALSE(inst.has_modifier);
        const auto& effects = rdna2_instruction_effects(inst.fmt, inst.opcode);
        ASSERT_NE(effects.name, nullptr);
        EXPECT_EQ(effects.vdst_dwords, 1u);
        check_case(c, recompile_compute(p.code.data(), p.code.size(), &p.table, p.config));
    }
}

TEST(IrDpp16InactiveSource, EveryRowShiftKeepsBoundsIndependentOfSourceExec) {
    check_catalog(true);
}
TEST(IrDpp16InactiveSource, EveryQuadTableAndBothUnaryAndBinaryCallersReadSourceExec) {
    check_catalog(false);
}
TEST(IrDpp16InactiveSource, FetchInactiveOneRemainsRefused) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t control : {0x111u, 0xb1u}) {
            const auto p = cases::program({wave, control, cases::Op::Mov, true}, true);
            EXPECT_TRUE(
                recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
        }
}
TEST(IrDpp16InactiveSource, ExportsEveryChangedSourceShapeAndControl) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the launch environment is immutable.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const auto root = compare_witness_root("dpp16_inactive_source_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = compare_witness_dump("dpp16_inactive_source_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    ASSERT_TRUE(fs::create_directories(root / "runtime"));
    std::ofstream attributes(root / "attributes.tsv");
    size_t index = 0;
    for (const auto& c : cases::cases()) {
        const auto p = cases::program(c);
        const uint64_t address = 0x5d53770000ull + index * 0x100;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, address});
        ASSERT_FALSE(module.empty()) << index;
        const auto stem = dump / ("cs_" + std::to_string(address) + "_" + std::to_string(index));
        ASSERT_TRUE(save_words(stem.string() + ".bin", p.code));
        ASSERT_TRUE(save_words(stem.string() + ".spv", module));
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", address, &p.table,
                                                 p.config, "synthetic DPP FI0 witness"));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
        ASSERT_TRUE(replay.produced);
        ASSERT_EQ(replay.spirv, module);
        check_case(c, module);
        const auto in = cases::inputs(c, 1);
        ASSERT_TRUE(
            save_words(root / "runtime" / (std::to_string(index) + ".buffer"), cases::buffer(in)));
        ASSERT_TRUE(save_words(root / "runtime" / (std::to_string(index) + ".expected"),
                               cases::expected(c, in)));
        attributes << index << '\t' << c.wave << '\t' << c.control << '\t' << int(c.op) << '\t'
                   << c.bound << '\n';
        ++index;
    }
    attributes.close();
    EXPECT_TRUE(attributes.good());
    EXPECT_EQ(index, 1286u);
}
} // namespace
