// Native DS values through the real recompiler: differing peers, source/destination EXEC,
// byte-address carry, aliases, and raw-program/sidecar/saved-module witness replay.
#include "gpu/recompiler/ir/ds_shuffle_cases.hpp"
#include "gpu/recompiler/ir/eval_lanes.hpp"
#include "gpu/recompiler/ir/ops/ds_shuffle.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace ds = prosper::test::ir::ds_shuffle;

namespace {
uint32_t opcode_count(const std::vector<uint32_t>& module, uint32_t opcode) {
    uint32_t count = 0;
    for (size_t i = 5; i < module.size();) {
        const auto words = module[i] >> 16;
        if (!words || words > module.size() - i) return 0;
        if ((module[i] & 0xffffu) == opcode) ++count;
        i += words;
    }
    return count;
}
LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in, uint32_t wave) {
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = wave;
    auto out = vm.run_buffers(wave, {{3, ds::buffer(in)}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(wave);
    return {out, {}};
}
void check_definition(const ds::Case& c, const LaneInputs& in) {
    Block block;
    Builder ir(block);
    Value d;
    std::vector<InvocationInputs> rows;
    if (c.kind == ds::Kind::Bpermute) {
        const auto address = ir.input(Type::U32), source = ir.input(Type::U32);
        const auto active = ir.input(Type::Bool), offset = ir.input(Type::U32);
        d = ds_bpermute_b32(ir, address, source, active, offset);
        for (uint32_t lane = 0; lane < c.wave; ++lane)
            rows.push_back({in.s0[lane], in.s1[lane], in.active[lane] ? 1u : 0u, c.control});
    } else {
        const DsSwizzleInputs inputs{[&] { return ir.input(Type::U32); },
                                     [&] { return ir.input(Type::Bool); }};
        d = c.kind == ds::Kind::Quad ? ds_swizzle_quad_b32(ir, c.control, inputs)
                                     : ds_swizzle_mask_b32(ir, c.control, inputs);
        for (uint32_t lane = 0; lane < c.wave; ++lane)
            rows.push_back({in.s0[lane], in.active[lane] ? 1u : 0u});
    }
    const auto got = evaluate_lanes(block, rows);
    const auto expected = ds::expected(c, in);
    for (uint32_t lane = 0; lane < c.wave; ++lane) {
        ASSERT_TRUE(got[lane][d.index].has_value());
        EXPECT_EQ(in.active[lane] ? static_cast<uint32_t>(*got[lane][d.index]) : in.old_d[lane],
                  expected[lane])
            << "lane=" << lane;
    }
}
std::vector<uint32_t> compile(const ds::Case& c, uint64_t address) {
    const auto p = ds::program(c);
    EXPECT_EQ(std::count(p.code.begin(), p.code.end(), 0x7daa0880u),
              c.exec == ds::Exec::Unmasked ? 0 : 1);
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                             {RecompileDiagnosticStage::Compute, address});
}
void check_case(const ds::Case& c, const std::vector<uint32_t>& module) {
    ASSERT_FALSE(module.empty());
    // Exactly data and source-EXEC shuffles: the reserved event-tagged CFG service is not this arm.
    EXPECT_EQ(opcode_count(module, 345), 2u);
    EXPECT_EQ(compute_spirv_min_subgroup_size(module), c.kind == ds::Kind::Quad ? 4u : 32u);
    const auto in = ds::inputs(c);
    check_definition(c, in);
    const auto out = execute(module, in, c.wave);
    ASSERT_TRUE(out.error.empty()) << out.error;
    EXPECT_EQ(out.d, ds::expected(c, in));
}
void save(const std::filesystem::path& path, const std::vector<uint32_t>& words) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(words.data()),
               static_cast<std::streamsize>(words.size() * sizeof(uint32_t)));
    file.close();
    ASSERT_TRUE(file.good()) << path;
}

TEST(IrDsShuffleExecution, NativeBpermuteMatchesIsaAndDefinitionIncludingAliasesAndCarry) {
    for (const auto& c : ds::cases()) {
        if (c.kind != ds::Kind::Bpermute) continue;
        SCOPED_TRACE(::testing::Message()
                     << c.wave << " offset=" << c.control << " pattern=" << c.pattern
                     << " exec=" << int(c.exec) << " dst=" << c.dst);
        check_case(c, compile(c, lane_harness_address()));
    }
}
TEST(IrDsShuffleExecution, BasicSwizzlesMatchIsaAndDefinitionWithInactiveSourcesAndDestinations) {
    for (const auto& c : ds::cases()) {
        if (c.kind == ds::Kind::Bpermute) continue;
        SCOPED_TRACE(::testing::Message() << c.wave << " control=" << c.control
                                          << " exec=" << int(c.exec) << " dst=" << c.dst);
        check_case(c, compile(c, lane_harness_address()));
    }
}
TEST(IrDsShuffleExecution, GdsAndUnimplementedRotateFftKeepTheirRefusals) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t arm = 0; arm < 4; ++arm) {
            ds::Case c{wave,
                       arm < 2 ? ds::Kind::Bpermute : ds::Kind::Mask,
                       arm < 2    ? 0u
                       : arm == 2 ? 0xc000u
                                  : 0xe000u,
                       0,
                       ds::Exec::Full,
                       1};
            auto p = ds::program(c);
            if (arm < 2) p.code[10] |= arm == 0 ? 1u << 16 : 1u << 17;
            const auto address = lane_harness_address();
            const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                                  {RecompileDiagnosticStage::Compute, address});
            EXPECT_TRUE(module.empty()) << "wave=" << wave << " arm=" << arm;
            EXPECT_FALSE(last_terminal_reject_reason(address).empty());
        }
}
TEST(IrDsShuffleExecution, ExportsReplayableWitnessPrograms) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never changes its launch environment.
    const bool configured = std::getenv("PROSPER_IR_WITNESS_DIR") != nullptr;
    const fs::path root = compare_witness_root("ir_ds_shuffle_witness");
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error);
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = compare_witness_dump("ir_ds_shuffle_witness");
    ASSERT_TRUE(fs::create_directories(dump));
    ASSERT_TRUE(fs::create_directories(root / "modules"));
    ASSERT_TRUE(fs::create_directories(root / "runtime"));
    std::ofstream attributes(root / "attributes.tsv");
    size_t index = 0;
    for (const auto& c : ds::cases()) {
        const uint64_t address = 0x5d5f000000ull + index * 0x100;
        const auto p = ds::program(c);
        const auto module = compile(c, address);
        ASSERT_FALSE(module.empty()) << "case=" << index;
        check_case(c, module);
        write_compare_witness_program(dump, address, index, p);
        const fs::path stem =
            dump / ("cs_" + std::to_string(address) + "_" + std::to_string(index));
        const auto replay = replay_refused_shader_inputs(
            p.code, read_refused_shader_inputs(stem.string() + ".inputs"));
        ASSERT_TRUE(replay.produced);
        ASSERT_EQ(replay.spirv, module);
        const auto saved = root / "modules" / (std::to_string(index) + ".spv");
        save(saved, module);
        std::ifstream file(saved, std::ios::binary);
        std::vector<uint32_t> reopened(module.size());
        file.read(reinterpret_cast<char*>(reopened.data()),
                  static_cast<std::streamsize>(reopened.size() * sizeof(uint32_t)));
        ASSERT_TRUE(file.good());
        ASSERT_EQ(reopened, module);
        const auto in = ds::inputs(c);
        const auto out = execute(reopened, in, c.wave);
        ASSERT_TRUE(out.error.empty()) << out.error;
        EXPECT_EQ(out.d, ds::expected(c, in));
        save(root / "runtime" / (std::to_string(index) + ".buffer"), ds::buffer(in));
        save(root / "runtime" / (std::to_string(index) + ".expected"), ds::expected(c, in));
        attributes << index << '\t' << c.wave << '\t' << int(c.kind) << '\t' << c.control << '\t'
                   << c.pattern << '\t' << int(c.exec) << '\t' << c.dst << '\t' << address << '\n';
        ++index;
    }
    attributes.close();
    EXPECT_TRUE(attributes.good());
    EXPECT_EQ(index, 464u);
}
}   // namespace
