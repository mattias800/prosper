// Actual bounded SMEM loads must retain scalar-offset, per-component, PC-table and EXEC contracts.
#include "gpu/recompiler/ir/smem_index_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "fixtures/compute_runner.h"
#include <gtest/gtest.h>
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
namespace fixture = prosper::test::ir::smem_index;
namespace {
bool save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    return out.good();
}
TEST(IrSmemIndexExecution, ExportsReplaysAndExecutesEveryWitnessBesideMatchedRefusals) {
    namespace fs = std::filesystem;
    using namespace prosper::test::ir;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the process does not mutate its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_smem_index_witness");
    if (!configured) {
        std::error_code ignored;
        fs::remove_all(root, ignored);
    }
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    const fs::path dump = compare_witness_dump("ir_smem_index_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream attributes(root / "attributes.jsonl");
    size_t index = 0, accepted = 0, refused = 0;
    for (const auto& c : fixture::cases()) {
        SCOPED_TRACE(index);
        const auto p = fixture::program(c);
        const uint64_t site = 0x5389000000ull + index * 0x1000;
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
            const auto d = fixture::data(c);
            bpermute_oracle::Interpreter vm(module);
            const auto interpreted = vm.run_buffers(c.wave, {{2, d.cbuf}, {3, d.input}}, 3);
            ASSERT_TRUE(vm.error.empty()) << vm.error;
            ASSERT_EQ(interpreted, d.expected);
            std::vector<uint32_t> actual;
            const auto dummy =
                prosper::test::run_compute(module, std::vector<float>(c.wave), c.wave, c.wave,
                                           d.cbuf, d.input, &actual, c.wave);
            ASSERT_EQ(dummy.size(), c.wave);
            ASSERT_TRUE(save(stem.string() + ".buffer2", d.cbuf));
            ASSERT_TRUE(save(stem.string() + ".buffer3", d.input));
            ASSERT_TRUE(save(stem.string() + ".actual", actual));
            ASSERT_TRUE(save(stem.string() + ".expected", d.expected));
            ASSERT_EQ(actual, d.expected);
            ++accepted;
        } else {
            ASSERT_FALSE(reason.empty());
            ASSERT_EQ(replay.reject, reason);
            ++refused;
        }
        attributes << "{\"index\":" << index++ << ",\"pcrel\":" << c.pcrel << ",\"wave\":" << c.wave
                   << ",\"width\":" << c.width << ",\"component\":" << c.component
                   << ",\"soffset\":" << c.soffset << ",\"immediate\":" << c.immediate
                   << ",\"invalid\":" << int(c.invalid) << ",\"accepted\":" << c.admitted()
                   << ",\"stem\":\"" << stem.filename().string() << "\"}\n";
    }
    attributes.close();
    ASSERT_TRUE(attributes.good());
    EXPECT_EQ(index, 628u);
    EXPECT_EQ(accepted, 620u);
    EXPECT_EQ(refused, 8u);
}
}   // namespace
