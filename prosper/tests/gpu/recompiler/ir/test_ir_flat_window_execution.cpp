// Window loads read the original base from push constants; exercise that fixture ABI explicitly.
#include "fixtures/compute_runner.h"
#include "gpu/recompiler/ir/flat_window_cases.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
namespace fixture = prosper::test::ir::flat_window;
namespace {
bool save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    return out.good();
}
void execute(const fixture::Case& c, const prosper::test::ir::LaneProgram& p,
             const std::vector<uint32_t>& module, const std::filesystem::path& stem) {
    // The integer oracle deliberately rejects an inactive lane's out-of-window access.
    // Use the same module for all-active CPU and partial-EXEC robust Vulkan executions.
    const auto cpu = fixture::data(c, true), partial = fixture::data(c, false);
    bpermute_oracle::Interpreter vm(module);
    vm.supplied_push_constants = p.config.user_sgprs;
    const auto interpreted = vm.run_buffers(c.wave, {{2, cpu.cbuf}, {3, cpu.input}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(interpreted, cpu.expected);
    std::vector<uint32_t> actual;
    const auto dummy = prosper::test::run_compute(
        module, std::vector<float>(c.wave), c.wave, c.wave, partial.cbuf, partial.input, &actual,
        c.wave, nullptr, nullptr, nullptr, 0, nullptr, p.config.user_sgprs);
    ASSERT_EQ(dummy.size(), c.wave);
    ASSERT_TRUE(save(stem.string() + ".buffer2", partial.cbuf));
    ASSERT_TRUE(save(stem.string() + ".buffer3", partial.input));
    ASSERT_TRUE(save(stem.string() + ".actual", actual));
    ASSERT_TRUE(save(stem.string() + ".expected", partial.expected));
    ASSERT_TRUE(save(stem.string() + ".cpu_input", cpu.input));
    ASSERT_TRUE(save(stem.string() + ".cpu_actual", interpreted));
    ASSERT_TRUE(save(stem.string() + ".cpu_expected", cpu.expected));
    ASSERT_EQ(actual, partial.expected);
}
TEST(IrFlatWindowExecution, PushConstantUploadChangesValuesWithoutRecompilingTheModule) {
    constexpr uint32_t wave = 32;
    SpirvCompute b;
    b.begin(1, nullptr, wave, 1, 1, wave, 14, true, true);
    b.store_output_word(b.load_push_constant(12), 1, 0);
    const auto module = b.finish();
    for (uint32_t word : {0x3fa00000u, 0x40200000u}) {
        std::vector<uint32_t> push(14);
        push[12] = word;
        bpermute_oracle::Interpreter vm(module);
        vm.supplied_push_constants = push;
        const auto interpreted = vm.run_buffers(wave, {{1, std::vector<uint32_t>(wave)}}, 1);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(interpreted, std::vector<uint32_t>(wave, word));
        const auto actual =
            prosper::test::run_compute(module, std::vector<float>(wave), wave, wave, {}, {},
                                       nullptr, wave, nullptr, nullptr, nullptr, 0, nullptr, push);
        ASSERT_EQ(actual.size(), wave);
        for (float value : actual) EXPECT_EQ(std::bit_cast<uint32_t>(value), word);
    }
}
TEST(IrFlatWindowExecution, ExportsReplaysAndExecutesEveryLoadFormBesideMatchedRefusals) {
    namespace fs = std::filesystem;
    using namespace prosper::test::ir;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never changes its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_flat_window_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_flat_window_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream attributes(root / "attributes.jsonl");
    size_t index = 0, accepted = 0, refused = 0;
    for (const auto& c : fixture::cases()) {
        SCOPED_TRACE(index);
        const auto p = fixture::program(c);
        const uint64_t site = 0x5391000000ull + index * 0x1000;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, site});
        ASSERT_EQ(!module.empty(), c.admitted()) << last_terminal_reject_reason(site);
        const std::string reason = c.admitted() ? "synthetic-witness (no live refusal)"
                                                : last_terminal_reject_reason(site);
        const fs::path stem = dump / ("cs_" + std::to_string(site) + "_" + std::to_string(index));
        ASSERT_TRUE(save(stem.string() + ".bin", p.code));
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", site, &p.table,
                                                 p.config, reason));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_EQ(replay.produced, c.admitted()) << replay.reject << " " << replay.incomplete;
        if (c.admitted()) {
            ASSERT_EQ(replay.spirv, module);
            ASSERT_TRUE(save(stem.string() + ".spv", module));
            execute(c, p, module, stem);
            ++accepted;
        } else {
            ASSERT_FALSE(reason.empty());
            ASSERT_EQ(replay.reject, reason);
            ++refused;
        }
        attributes << "{\"index\":" << index++ << ",\"wave\":" << c.wave
                   << ",\"segment\":" << c.segment << ",\"opcode\":" << c.load.opcode
                   << ",\"bytes\":" << c.load.bytes << ",\"component\":" << c.component
                   << ",\"invalid\":" << int(c.invalid) << ",\"accepted\":" << c.admitted()
                   << ",\"stem\":\"" << stem.filename().string() << "\"}\n";
    }
    attributes.close();
    ASSERT_TRUE(attributes.good());
    EXPECT_EQ(index, 64u);
    EXPECT_EQ(accepted, 56u);
    EXPECT_EQ(refused, 8u);
}
}   // namespace
