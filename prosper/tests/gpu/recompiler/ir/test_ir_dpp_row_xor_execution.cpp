// Native row-XOR transforms only SRC0; independent lane truth also pins inactive source zeros.
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/dpp_row_xor_cases.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ops/vector_dpp_row_xor.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
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
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::dpp_row_xor;
namespace {
uint32_t shuffles(const std::vector<uint32_t>& words) {
    uint32_t count = 0;
    for (size_t i = 5; i < words.size();) {
        const uint32_t n = words[i] >> 16;
        if (!n || n > words.size() - i) return 0;
        count += (words[i] & 0xffffu) == 345u;
        i += n;
    }
    return count;
}
void definition(const cases::Case& c, const LaneInputs& in) {
    Block block;
    Builder ir(block);
    const Value source = ir.input(Type::U32), active = ir.input(Type::Bool);
    const auto d =
        dpp_row_xor_source(ir, source, active, c.control == 0x128 ? 8u : c.control - 0x160u);
    std::vector<InvocationInputs> rows;
    for (uint32_t lane = 0; lane < c.wave; ++lane)
        rows.push_back({in.s0[lane], in.active[lane] ? 1u : 0u});
    const auto got = evaluate_lanes(block, rows);
    for (uint32_t lane = 0; lane < c.wave; ++lane)
        EXPECT_EQ(got[lane][d.index], Known{cases::bounded_source(c, in, lane)}) << lane;
}
void check(const cases::Case& c, const std::vector<uint32_t>& module) {
    ASSERT_FALSE(module.empty());
    EXPECT_EQ(shuffles(module), 2u);
    EXPECT_GE(compute_spirv_min_subgroup_size(module), 16u);
    for (uint32_t mask : {0u, 1u, 2u}) {
        const auto in = cases::inputs(c, mask);
        definition(c, in);
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = c.wave;
        auto got = vm.run_buffers(c.wave, {{3, cases::buffer(in)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        got.resize(c.wave);
        EXPECT_EQ(got, cases::expected(c, in)) << "mask=" << mask;
    }
}
void save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(words.data()),
              static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    out.close();
    ASSERT_TRUE(out.good()) << path;
}
TEST(IrDppRowXorExecution, AllControlsAndCombinersPreserveInactiveDestinationsAndSourceOne) {
    for (const auto& c : cases::cases()) {
        SCOPED_TRACE(::testing::Message()
                     << c.wave << ':' << c.control << ':' << int(c.combiner) << ':' << c.overlap);
        const auto p = cases::program(c);
        check(c, recompile_compute(p.code.data(), p.code.size(), &p.table, p.config));
    }
}
TEST(IrDppRowXorExecution, BadBoundaryFetchAndRowOrBankMasksRemainRefused) {
    for (const auto& c : cases::refused_cases()) {
        const auto p = cases::program(c);
        EXPECT_TRUE(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
    }
}
TEST(IrDppRowXorExecution, ExportsEveryAdmittedControlAndMatchedRefusals) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never changes its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const auto root = compare_witness_root("dpp_row_xor_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = compare_witness_dump("dpp_row_xor_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    ASSERT_TRUE(fs::create_directories(root / "runtime"));
    std::ofstream attributes(root / "attributes.tsv");
    auto all = cases::cases();
    const auto bad = cases::refused_cases();
    all.insert(all.end(), bad.begin(), bad.end());
    size_t index = 0;
    for (const auto& c : all) {
        const uint64_t address = 0x5d53960000ull + index * 0x100;
        const auto p = cases::program(c);
        const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                              {RecompileDiagnosticStage::Compute, address});
        const bool accepted = c.bad_flags == 0;
        ASSERT_EQ(!module.empty(), accepted) << index;
        const auto stem = dump / ("cs_" + std::to_string(address) + "_" + std::to_string(index));
        save(stem.string() + ".bin", p.code);
        ASSERT_TRUE(write_refused_compute_inputs(stem.string() + ".inputs", address, &p.table,
                                                 p.config, "synthetic DPP16 witness"));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
        ASSERT_EQ(replay.produced, accepted);
        ASSERT_EQ(replay.spirv, module);
        if (accepted) {
            check(c, module);
            save(stem.string() + ".spv", module);
            const auto in = cases::inputs(c, 1);
            save(root / "runtime" / (std::to_string(index) + ".buffer"), cases::buffer(in));
            save(root / "runtime" / (std::to_string(index) + ".expected"), cases::expected(c, in));
        }
        attributes << index << '\t' << c.wave << '\t' << c.control << '\t' << int(c.combiner)
                   << '\t' << c.overlap << '\t' << c.bad_flags << '\t' << accepted << '\n';
        ++index;
    }
    attributes.close();
    EXPECT_TRUE(attributes.good());
    EXPECT_EQ(index, 416u);
}
}   // namespace
