// #5322: min-subnormal CEIL/FLOOR must become signed zero in producing input-flush
// modes. Hand IEEE-bit expectations are independent of eval and the host-math interpreter.
#include "gpu/recompiler/ir/ops/float_input_modifier.hpp"
#include "gpu/recompiler/ir/eval.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/ir/ops/vector_float_unary.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "fixtures/test_scratch.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <array>
#include <optional>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
uint32_t expected(uint32_t bits, bool ceil, bool flush) {
    uint32_t magnitude = bits & 0x7fffffffu;
    const uint32_t sign = bits & 0x80000000u;
    if (flush && magnitude < 0x00800000u) return sign;
    if (magnitude == 0) return bits;
    const int exponent = int(magnitude >> 23) - 127;
    if (exponent < 0) return (ceil != bool(sign)) ? (sign | 0x3f800000u) : sign;
    if (exponent >= 23) return bits;
    const uint32_t unit = 1u << (23 - exponent), mask = unit - 1;
    if (!(bits & mask)) return bits;
    return (bits & ~mask) + ((ceil != bool(sign)) ? unit : 0);
}
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    constexpr uint32_t edges[] = {1,          0x80000001, 0x007fffff, 0x807fffff, 0,
                                  0x80000000, 0x00800000, 0x80800000, 0x3fa00000, 0xbfa00000,
                                  0x3f800000, 0xbf800000, 0x3f000000, 0xbf000000};
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        if (lane < 2 * std::size(edges)) in.s0[lane] = edges[lane % std::size(edges)];
        if ((in.s0[lane] & 0x7fffffffu) >= 0x7f800000u) in.s0[lane] = 0x3fa00000;
    }
    return in;
}
} // namespace
TEST(IrF32RoundInputExecution, BitFlushingHasHandWorkedBoundariesAndPreservesSpecialBits) {
    Block block;
    Builder ir(block);
    const auto value = build_flush_f32_input(ir, ir.input(Type::U32));
    struct Case {
        uint32_t source, want;
    };
    for (const auto x :
         std::array{Case{1, 0}, Case{0x80000001, 0x80000000}, Case{0x007fffff, 0},
                    Case{0x807fffff, 0x80000000}, Case{0, 0}, Case{0x80000000, 0x80000000},
                    Case{0x00800000, 0x00800000}, Case{0x80800000, 0x80800000},
                    Case{0x7f800000, 0x7f800000}, Case{0xff800000, 0xff800000},
                    Case{0x7fcabcde, 0x7fcabcde}, Case{0xff800001, 0xff800001}}) {
        const Known source[] = {x.source};
        EXPECT_EQ(evaluate(block, source)[value.index], Known(x.want));
    }
    const Known unknown[] = {std::nullopt};
    EXPECT_FALSE(evaluate(block, unknown)[value.index]);
}
TEST(IrF32RoundInputExecution, HandExpectationsDistinguishFlushPreserveAndDirection) {
    EXPECT_EQ(expected(1, true, true), 0u);
    EXPECT_EQ(expected(1, true, false), 0x3f800000u);
    EXPECT_EQ(expected(0x80000001, false, true), 0x80000000u);
    EXPECT_EQ(expected(0x80000001, false, false), 0xbf800000u);
    EXPECT_EQ(expected(0x3fa00000, true, true), 0x40000000u);
    EXPECT_EQ(expected(0xbfa00000, false, true), 0xc0000000u);
}
TEST(IrF32RoundInputExecution, BothWavesExecuteEveryModeWithDifferentInputsAndPartialExec) {
    for (uint32_t wave : {32u, 64u})
        for (uint32_t opcode : {0x22u, 0x24u})
            for (uint32_t mode = 0; mode < 256; ++mode)
                for (uint32_t seed : {5322u, 107u}) {
                    const auto in = inputs(seed);
                    const auto run =
                        run_vector_op_per_lane({vop1(opcode)}, in, wave, false, mode << 12);
                    ASSERT_TRUE(run.error.empty()) << run.error << " mode " << mode;
                    ASSERT_EQ(run.d.size(), in.lanes());
                    for (size_t lane = 0; lane < in.lanes(); ++lane)
                        EXPECT_EQ(run.d[lane],
                                  in.active[lane]
                                      ? expected(in.s0[lane], opcode == 0x22, !(mode & 0x10))
                                      : in.old_d[lane])
                            << "wave " << wave << " op " << opcode << " mode " << mode << " lane "
                            << lane;
                }
}
TEST(IrF32RoundInputExecution, IeeeAndOutputModeDoNotSuppressInputFlushing) {
    const auto in = inputs(5322);
    for (uint32_t opcode : {0x22u, 0x24u})
        for (uint32_t rsrc1 : {0u, 0x20000u, 0x800000u, 0x820000u}) {
            const auto run = run_vector_op_per_lane({vop1(opcode)}, in, 64, false, rsrc1);
            ASSERT_TRUE(run.error.empty()) << run.error;
            for (size_t lane = 0; lane < in.lanes(); ++lane)
                EXPECT_EQ(run.d[lane], in.active[lane] ? expected(in.s0[lane], opcode == 0x22, true)
                                                       : in.old_d[lane]);
        }
}
TEST(IrF32RoundInputExecution, MissingModeHasAnObservedModeAndUnrelatedOpControl) {
    const auto in = inputs(5322);
    for (uint32_t opcode : {0x22u, 0x24u}) {
        const auto no_mode = run_vector_op_per_lane({vop1(opcode)}, in, 64, false, UINT32_MAX);
        EXPECT_TRUE(no_mode.d.empty());
        EXPECT_NE(no_mode.error.find("f32-round-input-mode-unavailable"), std::string::npos);
        EXPECT_TRUE(run_vector_op_per_lane({vop1(opcode)}, in, 64, false, 0).error.empty());
        EXPECT_TRUE(run_vector_op_per_lane({vop1(opcode)}, in, 64, false, 0x30000).error.empty());
    }
    for (uint32_t opcode : {0x21u, 0x23u})
        EXPECT_TRUE(
            run_vector_op_per_lane({vop1(opcode)}, in, 64, false, UINT32_MAX).error.empty());
}

TEST(IrF32RoundInputExecution, ComposedEvaluationAgreesWithIndependentBitAnswers) {
    for (uint32_t opcode : {0x22u, 0x24u})
        for (bool flush : {false, true}) {
            const auto* def = find_vector_float_unary_op(Rdna2Format::VOP1, opcode);
            ASSERT_NE(def, nullptr);
            Block block;
            Builder ir(block);
            const auto source = ir.input(Type::U32);
            const auto transformed = flush ? build_flush_f32_input(ir, source) : source;
            const auto result = def->build(ir, {transformed, ir.input(Type::U32)});
            const auto output = flush ? build_preserve_f32_zero(ir, transformed, result) : result;
            for (uint32_t seed = 1; seed < 5; ++seed)
                for (auto bits : inputs(seed).s0) {
                    const Known operands[] = {bits, 0};
                    EXPECT_EQ(evaluate(block, operands)[output.index],
                              Known(expected(bits, opcode == 0x22, flush)));
                }
        }
}
namespace {
struct Witness {
    uint32_t opcode, wave, mode;
    bool ieee;
    size_t index;
};
void export_witness(const std::filesystem::path& dump, const LaneInputs& in, const Witness& w,
                    std::ostream& manifest) {
    auto program = lane_program({vop1(w.opcode)}, 64, w.wave);
    program.config.compute_pgm_rsrc1 = (w.mode << 12) | (uint32_t(w.ieee) << 23);
    const uint64_t address = 0x5322000000ull + w.index * 0x100;
    const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(w.index);
    std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
    bin.write(reinterpret_cast<const char*>(program.code.data()),
              static_cast<std::streamsize>(program.code.size() * sizeof(uint32_t)));
    bin.close();
    ASSERT_TRUE(bin.good());
    const auto sidecar = dump / (stem + ".inputs");
    ASSERT_TRUE(write_refused_compute_inputs(sidecar, address, &program.table, program.config,
                                             "synthetic-input-mode witness"));
    const auto read = read_refused_shader_inputs(sidecar);
    ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
    const auto replay = replay_refused_shader_inputs(program.code, read);
    ASSERT_TRUE(replay.produced) << replay.reject;
    ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
    const auto path = dump / (stem + ".spv");
    std::ofstream spv(path, std::ios::binary);
    spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
              static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
    spv.close();
    ASSERT_TRUE(spv.good());
    std::vector<uint32_t> saved(replay.spirv.size());
    std::ifstream saved_file(path, std::ios::binary);
    saved_file.read(reinterpret_cast<char*>(saved.data()),
                    static_cast<std::streamsize>(saved.size() * sizeof(uint32_t)));
    ASSERT_TRUE(saved_file.good());
    ASSERT_EQ(saved, replay.spirv);
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(saved);
    const auto out = vm.run_buffers(64, {{3, buffer}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_GE(out.size(), 64u);
    for (size_t lane = 0; lane < in.lanes(); ++lane)
        EXPECT_EQ(out[lane], in.active[lane]
                                 ? expected(in.s0[lane], w.opcode == 0x22, !(w.mode & 0x10))
                                 : in.old_d[lane]);
    manifest << stem << '\t' << w.opcode << '\t' << w.wave << '\t' << w.mode << '\t' << w.ieee
             << '\n';
}
} // namespace
TEST(IrF32RoundInputExecution, ExportsExecutesAndReplaysModeWitnesses) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input, never changed here.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "f32-round-input-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream manifest(root / "manifest.tsv");
    manifest << "stem\topcode\twave\tmode\tieee\n";
    const auto in = inputs(5322);
    size_t count = 0;
    for (uint32_t opcode : {0x22u, 0x24u})
        for (uint32_t wave : {32u, 64u})
            for (uint32_t mode : {0u, 0x10u, 0x20u, 0x30u})
                for (bool ieee : {false, true})
                    export_witness(dump, in, {opcode, wave, mode, ieee, count++}, manifest);
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 32u);
}
