// Execute both encodings, SGPR/VGPR/overlap, varying lanes, and inactive canaries.
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include <ios>
#include <cstddef>
#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/ir/integer_float_rte_e64_cases.hpp"
#include "gpu/recompiler/ir/ops/vector_integer_float_rte_e64.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::integer_float_rte;
namespace {
class IrIntegerFloatRteExecution : public ::testing::Test {
    void SetUp() override {
        reset_float32_rte_support_for_test();
        publish_float32_rte_support(true, true, true);
    }
    void TearDown() override { reset_float32_rte_support_for_test(); }
};
std::vector<uint32_t> compile(const cases::Case& c) {
    const auto p = cases::program(c);
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
}
void execute(const cases::Case& c, const std::vector<uint32_t>& module, const LaneInputs& in) {
    bpermute_oracle::Interpreter vm(module);
    const auto actual = vm.run_buffers(64, {{3, cases::buffer(in)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(actual, cases::expected(c, in));
    const auto* def = find_vector_integer_float_rte_e64_op(Rdna2Format::VOP3, c.opcode);
    ASSERT_NE(def, nullptr);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        EXPECT_EQ(eval_conversion_op(*def, cases::source(c, in, lane)),
                  cases::reference(cases::source(c, in, lane), c.opcode == 0x185));
}
bool save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * 4));
    out.close();
    return out.good();
}
}   // namespace
TEST_F(IrIntegerFloatRteExecution, BothWavesAndPrefixesRoundVaryingLanesAndPreserveExecOff) {
    for (const auto& c : cases::cases())
        if (c.accepted) {
            const auto module = compile(c);
            ASSERT_FALSE(module.empty());
            for (uint32_t mask : {0u, 1u, 2u})
                for (uint32_t seed : {1u, 17u, 5394u})
                    execute(c, module, cases::inputs(seed, mask));
        }
}
TEST_F(IrIntegerFloatRteExecution, EveryUnsupportedModeAndModifierRemainsRefused) {
    for (const auto& c : cases::cases())
        if (!c.accepted) EXPECT_TRUE(compile(c).empty());
    for (uint32_t opcode : {0x185u, 0x186u}) {
        cases::Case c;
        c.opcode = opcode;
        reset_float32_rte_support_for_test();
        EXPECT_TRUE(compile(c).empty());
        publish_float32_rte_support(true, true, true);
        ASSERT_FALSE(compile(c).empty());
    }
}
TEST_F(IrIntegerFloatRteExecution, ExportsEveryNewFormBesideUnchangedRefusals) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never changes its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("integer_float_rte_e64_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root));
    const fs::path dump = compare_witness_dump("integer_float_rte_e64_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream attributes(root / "attributes.tsv");
    size_t index = 0, accepted = 0, refused = 0;
    for (const auto& c : cases::cases()) {
        const auto p = cases::program(c);
        const uint64_t address = 0x5394000000ull + index * 0x1000;
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, address});
        ASSERT_EQ(!module.empty(), c.accepted) << last_terminal_reject_reason(address);
        const fs::path stem =
            dump / ("cs_" + std::to_string(address) + "_" + std::to_string(index));
        ASSERT_TRUE(save(stem.string() + ".bin", p.code));
        ASSERT_TRUE(
            write_refused_compute_inputs(stem.string() + ".inputs", address, &p.table, p.config,
                                         c.accepted ? "synthetic witness (no live refusal)"
                                                    : last_terminal_reject_reason(address)));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
        ASSERT_EQ(replay.produced, c.accepted) << replay.reject;
        if (c.accepted) {
            ASSERT_EQ(replay.spirv, module);
            ASSERT_TRUE(save(stem.string() + ".spv", module));
            const auto in = cases::inputs(5394);
            ASSERT_TRUE(save(stem.string() + ".buffer3", cases::buffer(in)));
            ASSERT_TRUE(save(stem.string() + ".expected", cases::expected(c, in)));
            execute(c, replay.spirv, in);
            ++accepted;
        } else
            ++refused;
        attributes << index << '\t' << c.opcode << '\t' << c.wave << '\t' << c.source << '\t'
                   << c.legacy << '\t' << c.mode << '\t' << c.modifier << '\t' << c.accepted
                   << '\n';
        ++index;
    }
    attributes.close();
    ASSERT_TRUE(attributes.good());
    EXPECT_EQ(accepted, 24u);
    EXPECT_EQ(refused, 80u);
}
