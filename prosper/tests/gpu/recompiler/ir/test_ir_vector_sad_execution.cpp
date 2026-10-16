// AMD 70648 12.12 VOP3 word-result SAD: emitted SPIR-V vs shared definitions,
// with three independent per-lane inputs, old D independent of S2, and partial EXEC in both waves.
#include "gpu/recompiler/ir/ops/vector_sad.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>
#include "gpu/recompiler/ir/ops/vector_ternary_integer.hpp"
#include "gpu/recompiler/ir/builder.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/ir.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
LaneInputs ternary_inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    const uint32_t edges[] = {0u, 1u, 31u, 32u, 33u, 0x7fffffffu, 0x80000000u, 0xffffffffu};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        seed = seed * 1103515245u + 12345u;
        uint32_t value = lane < 16 ? edges[(lane * 3 + 5) % 8] : seed;
        if (value == in.old_d[lane]) value ^= 1;
        in.s2.push_back(value);
    }
    return in;
}
std::vector<uint32_t> words(uint32_t opcode) {
    return {0xd4000000u | (opcode << 16) | kVdst,
            (256u + kVsrc0) | ((256u + kVsrc1) << 9) | ((256u + kVsrc2) << 18)};
}
}   // namespace
TEST(IrVectorSadExecution, EveryDefinitionMatchesEachLaneInBothWaves) {
    for (const auto& def : vector_sad_defs()) {
        SCOPED_TRACE(rdna2_instruction_effects(def.fmt, def.opcode).name);
        const auto code = words(def.opcode);
        std::vector<Rdna2Inst> decoded;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
        ASSERT_EQ(decoded.size(), 1u);
        EXPECT_EQ(decoded[0].fmt, def.fmt);
        EXPECT_EQ(decoded[0].opcode, def.opcode);
        EXPECT_EQ(decoded[0].src[2].kind, OperandKind::VGPR);
        EXPECT_EQ(decoded[0].src[2].value, int(kVsrc2));
        for (uint32_t wave : {64u, 32u})
            for (uint32_t seed = 1; seed <= 8; ++seed) {
                const auto in = ternary_inputs(seed * 5243);
                const auto run = run_vector_op_per_lane(code, in, wave);
                ASSERT_TRUE(run.error.empty()) << run.error;
                ASSERT_EQ(run.d.size(), in.lanes());
                for (size_t lane = 0; lane < in.lanes(); ++lane) {
                    const auto want =
                        eval_ternary_integer_op(def, {in.s0[lane], in.s1[lane], in.s2[lane]});
                    if (!want) {
                        ADD_FAILURE() << "unknown lane " << lane;
                        continue;
                    }
                    EXPECT_EQ(run.d[lane], in.active[lane] ? *want : in.old_d[lane])
                        << "lane " << lane << " wave " << wave;
                }
            }
    }
}
TEST(IrVectorSadExecution, HarnessSeesAllThreeInputsAndInactiveDestinations) {
    auto in = ternary_inputs(43);
    const auto code = words(0x15d);
    const auto run = run_vector_op_per_lane(code, in);
    ASSERT_TRUE(run.error.empty()) << run.error;
    std::set<uint32_t> active;
    size_t inactive = 0;
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        if (in.active[lane]) {
            active.insert(run.d[lane]);
            EXPECT_EQ(run.d[lane], (std::max(in.s0[lane], in.s1[lane]) -
                                    std::min(in.s0[lane], in.s1[lane]) + in.s2[lane]));
            EXPECT_NE(in.s2[lane], in.old_d[lane]);
        } else {
            ++inactive;
            EXPECT_EQ(run.d[lane], in.old_d[lane]);
        }
    EXPECT_GT(active.size(), 16u);
    EXPECT_GE(inactive, 2u);
    in.s2.pop_back();
    const auto bad = run_vector_op_per_lane(code, in);
    EXPECT_TRUE(bad.d.empty());
    EXPECT_EQ(bad.error, "S2 lane count mismatch");
}
TEST(IrVectorSadExecution, HandWorkedIsaUnsignedMagnitudeMaskRoleAndWraparound) {
    struct Example {
        uint32_t op, a, b, c, want;
    };
    // AMD70648 12.12: unsigned differences; accumulators wrap modulo2^32.
    // Four255-byte differences total1020; two65535-half differences total131070.
    // MSAD skips zero reference bytes in S1, including when the S0 byte is nonzero.
    for (auto x :
         {Example{0x15a, 0, 0xffffffff, 7, 0x403}, Example{0x15a, 0x01020304, 0x04030201, 0, 8},
          Example{0x15a, 0, 0xffffffff, 0xffffffff, 0x3fb},
          Example{0x15b, 0, 0xffffffff, 7, 0x03fc0007},
          Example{0x15b, 0, 0xffffffff, 0xffffffff, 0x03fbffff},
          Example{0x15b, 0x01010101, 0, 0x76543210, 0x76583210},
          Example{0x15c, 0x0000ffff, 0xffff0000, 7, 0x20005},
          Example{0x15c, 0, 0xffffffff, 0xffffffff, 0x1fffd},
          Example{0x15c, 0x12345678, 0x12345678, 0x89abcdef, 0x89abcdef},
          Example{0x15d, 0x80000000, 0x7fffffff, 2, 3}, Example{0x15d, 0xffffffff, 0, 7, 6},
          Example{0x15d, 0, 0xffffffff, 0, 0xffffffff},
          Example{0x15d, 0xabcdef01, 0xabcdef01, 0x12345678, 0x12345678},
          Example{0x171, 0xffffffff, 0, 7, 7}, Example{0x171, 0, 0x01020304, 7, 17},
          Example{0x171, 0x0000ffff, 0x00000100, 7, 0x105},
          Example{0x171, 0, 0xffffffff, 0xffffffff, 0x3fb}}) {
        SCOPED_TRACE(::testing::Message() << x.op << " " << x.a << " " << x.b << " " << x.c);
        const auto* def = find_vector_sad_op(Rdna2Format::VOP3, x.op);
        ASSERT_NE(def, nullptr);
        EXPECT_EQ(eval_ternary_integer_op(*def, {x.a, x.b, x.c}), Known(x.want));
        auto in = ternary_inputs(13);
        std::fill(in.s0.begin(), in.s0.end(), x.a);
        std::fill(in.s1.begin(), in.s1.end(), x.b);
        std::fill(in.s2.begin(), in.s2.end(), x.c);
        const auto run = run_vector_op_per_lane(words(x.op), in);
        ASSERT_TRUE(run.error.empty()) << run.error;
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane], in.active[lane] ? x.want : in.old_d[lane]);
    }
}
TEST(IrVectorSadExecution, ExistingClampRefusalRemains) {
    const auto in = ternary_inputs(151);
    for (const auto& def : vector_sad_defs()) {
        const auto good = run_vector_op_per_lane(words(def.opcode), in);
        ASSERT_TRUE(good.error.empty()) << good.error;
        auto code = words(def.opcode);
        code[0] |= 1u << 15;
        const auto bad = run_vector_op_per_lane(code, in);
        EXPECT_TRUE(bad.d.empty());
        EXPECT_TRUE(bad.error.starts_with("refused:")) << bad.error;
    }
}
TEST(IrVectorSadExecution, EffectsAndSourceSlotsAgreeWithTheDefinitionDomain) {
    std::set<Opcode> seen;
    for (const auto& def : vector_sad_defs()) {
        const auto& e = rdna2_instruction_effects(def.fmt, def.opcode);
        EXPECT_EQ(e.lane_flow, Rdna2LaneFlow::LaneWise);
        EXPECT_EQ(e.vdst_dwords, 1u);
        EXPECT_FALSE(e.vdst_half);
        EXPECT_FALSE(e.accumulates);
        EXPECT_EQ(rdna2_valu_source_count(def.fmt, def.opcode), 3u);
        Block block;
        Builder ir(block);
        std::vector<uint32_t> sources;
        def.build(ir, TernaryInputs(ir, sources));
        EXPECT_EQ(sources.size(), block.input_count());
        EXPECT_EQ(std::set<uint32_t>(sources.begin(), sources.end()),
                  (std::set<uint32_t>{0, 1, 2}));
        for (const auto& inst : block.insts()) seen.insert(inst.op);
    }
    for (auto op : {Opcode::BfeU, Opcode::UMin, Opcode::UMax, Opcode::ISub, Opcode::IAdd,
                    Opcode::Shl, Opcode::INe, Opcode::Select})
        EXPECT_TRUE(seen.contains(op));
    EXPECT_EQ(find_vector_sad_op(Rdna2Format::VOP2, 0x15a), nullptr);
    EXPECT_EQ(find_vector_sad_op(Rdna2Format::VOP3, 0x128), nullptr);
}
TEST(IrVectorSadExecution, UnknownSourceWordsRemainUnknown) {
    for (const auto& def : vector_sad_defs())
        for (unsigned source = 0; source < 3; ++source) {
            std::array<Known, 3> in{1, 2, 3};
            in[source] = std::nullopt;
            EXPECT_FALSE(eval_ternary_integer_op(def, in));
        }
}

// The recorded corpus did not observe this family's output. Export every existing plain form
// using exactly the execution harness's program and resource/launch bindings, without touching it.
TEST(IrVectorSadExecution, ExportsBothWaveFormsForTheCorpusGate) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): this process never mutates its launch environment.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "sad-witness";
    ASSERT_FALSE(fs::exists(root)) << "witness export requires a new directory";
    ASSERT_TRUE(fs::create_directories(root));
    const fs::path dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directory(dump));
    size_t count = 0;
    for (const auto& def : vector_sad_defs())
        for (uint32_t wave : {64u, 32u}) {
            const auto program = lane_program(words(def.opcode), 64, wave, true);
            const uint64_t address = 0x5243000000ull + count * 0x100;
            const fs::path stem =
                dump / ("cs_" + std::to_string(address) + "_" + std::to_string(count));
            std::ofstream bin(stem.string() + ".bin", std::ios::binary);
            bin.write(reinterpret_cast<const char*>(program.code.data()),
                      static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
            bin.close();
            ASSERT_TRUE(bin.good());
            const fs::path inputs = stem.string() + ".inputs";
            ASSERT_TRUE(write_refused_compute_inputs(inputs, address, &program.table,
                                                     program.config,
                                                     "synthetic-witness (no live refusal)"));
            const auto read = read_refused_shader_inputs(inputs);
            ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
            const auto replay = replay_refused_shader_inputs(program.code, read);
            ASSERT_TRUE(replay.produced) << replay.reject << " op " << def.opcode;
            EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
            ++count;
        }
    EXPECT_EQ(count, 10u);
}
