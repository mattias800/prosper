// #5344: CVT_FLR supports denormals even when the host GLSL Floor flushes inputs.
// Independent finite bit expectations pin the guest result; a hand-built Floor
// module proves that the opt-in interpreter profile exposes the original defect.
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/ir/ops/vector_conversion.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <optional>
#include <string>
#include <system_error>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::gpu::ir;
using namespace prosper::test::ir;
namespace {
struct Example {
    uint32_t bits, want;
};
constexpr std::array kExamples{Example{1, 0},
                               Example{0x80000001, 0xffffffff},
                               Example{0x007fffff, 0},
                               Example{0x807fffff, 0xffffffff},
                               Example{0, 0},
                               Example{0x80000000, 0},
                               Example{0x00800000, 0},
                               Example{0x80800000, 0xffffffff},
                               Example{0x3f7fffff, 0},
                               Example{0xbf7fffff, 0xffffffff},
                               Example{0x3f800000, 1},
                               Example{0xbf800000, 0xffffffff},
                               Example{0x3f800001, 1},
                               Example{0xbf800001, 0xfffffffe},
                               Example{0x40100000, 2},
                               Example{0xc0100000, 0xfffffffd},
                               Example{0x4affffff, 0x007fffff},
                               Example{0xcaffffff, 0xff800000},
                               Example{0x4b800000, 0x01000000},
                               Example{0xcb800000, 0xff000000},
                               Example{0x4effffff, 0x7fffff80},
                               Example{0xcf000000, 0x80000000}};
LaneInputs inputs(uint32_t seed) {
    auto in = lane_inputs(seed);
    for (size_t lane = 0; lane < in.lanes(); ++lane) {
        in.s0[lane] = kExamples[(lane + seed) % kExamples.size()].bits;
        // Alias tests must read the old lane value before publishing the conversion.
        in.old_d[lane] = kExamples[(lane * 7 + seed + 3) % kExamples.size()].bits;
    }
    return in;
}
uint32_t expected(uint32_t word) {
    for (const auto x : kExamples)
        if (x.bits == word) return x.want;
    ADD_FAILURE() << "missing independent finite example " << word;
    return 0;
}
constexpr uint32_t kForms = 7;
std::vector<uint32_t> words(uint32_t form) {
    switch (form) {
        case 1: return {vop1(0x0d, kVdst, 256 + kVdst)};
        case 2: return {0xbeb203ffu, 0x80000001u, vop1(0x0d, kVdst, 50)};
        case 3: return {vop1(0x0d, kVdst, 255), 1};
        case 4: return {vop1(0x0d, kVdst, 255), 0x807fffff};
        case 5: return {vop1(0x0d, kVdst, 128)};
        case 6: return {vop1(0x0d, kVdst, 243)}; // Inline F32 -1.
        default: return {vop1(0x0d)};
    }
}
uint32_t source(const LaneInputs& in, size_t lane, uint32_t form) {
    switch (form) {
        case 1: return in.old_d[lane];
        case 2: return 0x80000001;
        case 3: return 1;
        case 4: return 0x807fffff;
        case 5: return 0;
        case 6: return 0xbf800000;
        default: return in.s0[lane];
    }
}
LaneRun execute(const std::vector<uint32_t>& module, const LaneInputs& in, bool flush) {
    std::vector<uint32_t> buffer;
    for (const auto* array : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), array->begin(), array->end());
    for (bool active : in.active) buffer.push_back(active ? 1u : 0u);
    bpermute_oracle::Interpreter vm(module);
    vm.floor_flush_denorm_input = flush;
    auto out = vm.run_buffers(static_cast<uint32_t>(in.lanes()), {{3, buffer}}, 3);
    if (!vm.error.empty()) return {{}, vm.error};
    out.resize(in.lanes());
    return {out, {}};
}
void check(const std::vector<uint32_t>& module, const LaneInputs& in, uint32_t form) {
    ASSERT_FALSE(module.empty());
    for (bool flush : {false, true}) {
        const auto run = execute(module, in, flush);
        ASSERT_TRUE(run.error.empty()) << run.error;
        ASSERT_EQ(run.d.size(), in.lanes());
        for (size_t lane = 0; lane < in.lanes(); ++lane)
            EXPECT_EQ(run.d[lane],
                      in.active[lane] ? expected(source(in, lane, form)) : in.old_d[lane])
                << "form " << form << " flush " << flush << " lane " << lane;
    }
}
// Independent storage-buffer Floor control, without the conversion builder or prepass.
std::vector<uint32_t> native_floor() {
    std::vector<uint32_t> module{0x07230203, 0x00010300, 0, 25, 0};
    const auto emit = [&](uint32_t op, std::initializer_list<uint32_t> operands) {
        module.push_back((uint32_t(operands.size() + 1) << 16) | op);
        module.insert(module.end(), operands);
    };
    emit(Op_Capability, {1});
    emit(Op_ExtInstImport, {1, 0x4c534c47, 0x6474732e, 0x3035342e, 0});
    emit(Op_MemoryModel, {0, 1});
    emit(Op_EntryPoint, {5, 16, 0x6e69616d, 0, 14});
    emit(Op_ExecutionMode, {16, 17, 64, 1, 1});
    emit(Op_Decorate, {8, 6, 4});
    emit(Op_Decorate, {9, 2});
    emit(Op_MemberDecorate, {9, 0, 35, 0});
    emit(Op_Decorate, {15, 34, 0});
    emit(Op_Decorate, {15, 33, 3});
    emit(Op_Decorate, {14, 11, 28});
    emit(Op_TypeVoid, {2});
    emit(Op_TypeFunction, {3, 2});
    emit(Op_TypeInt, {4, 32, 0});
    emit(Op_TypeFloat, {5, 32});
    emit(Op_TypeVector, {6, 4, 3});
    emit(Op_TypePointer, {7, 1, 6});
    emit(Op_TypeRuntimeArray, {8, 4});
    emit(Op_TypeStruct, {9, 8});
    emit(Op_TypePointer, {10, 12, 9});
    emit(Op_TypePointer, {11, 12, 4});
    emit(Op_Constant, {4, 12, 0});
    emit(Op_Variable, {7, 14, 1});
    emit(Op_Variable, {10, 15, 12});
    emit(Op_Function, {2, 16, 0, 3});
    emit(Op_Label, {17});
    emit(Op_Load, {6, 18, 14});
    emit(Op_CompositeExtract, {4, 19, 18, 0});
    emit(Op_AccessChain, {11, 20, 15, 12, 19});
    emit(Op_Load, {4, 21, 20});
    emit(Op_Bitcast, {5, 22, 21});
    emit(Op_ExtInst, {5, 23, 1, 8, 22}); // GLSL.std.450 Floor
    emit(Op_Bitcast, {4, 24, 23});
    emit(Op_Store, {20, 24});
    emit(Op_Return, {});
    emit(Op_FunctionEnd, {});
    return module;
}
} // namespace

TEST(IrCvtFlrDenormExecution, IndependentFloorControlProvesOnlyTheSelectedProfileFlushes) {
    const auto module = native_floor();
    const std::array cases{Example{0x80000001, 0xbf800000},
                           Example{0x807fffff, 0xbf800000},
                           Example{1, 0},
                           Example{0x007fffff, 0},
                           Example{0, 0},
                           Example{0x80000000, 0x80000000},
                           Example{0xc0100000, 0xc0400000},
                           Example{0x40100000, 0x40000000}};
    std::vector<uint32_t> data;
    for (size_t lane = 0; lane < 64; ++lane) data.push_back(cases[lane % cases.size()].bits);
    for (bool flush : {false, true}) {
        bpermute_oracle::Interpreter vm(module);
        vm.floor_flush_denorm_input = flush;
        const auto out = vm.run_buffers(64, {{3, data}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), data.size());
        for (size_t lane = 0; lane < data.size(); ++lane) {
            const auto x = cases[lane % cases.size()];
            const bool negative_subnormal = x.bits == 0x80000001 || x.bits == 0x807fffff;
            EXPECT_EQ(out[lane], flush && negative_subnormal ? 0x80000000u : x.want);
        }
    }
}
TEST(IrCvtFlrDenormExecution, PrepassChangesOnlyNonzeroSubnormalWords) {
    Block block;
    Builder ir(block);
    const auto prepared = prepare_cvt_flr_input(ir, ir.input(Type::U32));
    for (const auto x : kExamples) {
        const Known operand[] = {x.bits};
        const uint32_t magnitude = x.bits & 0x7fffffffu;
        const uint32_t want = magnitude != 0 && magnitude < 0x00800000u
                                  ? ((x.bits & 0x80000000u) ? 0xbf800000u : 0u)
                                  : x.bits;
        EXPECT_EQ(evaluate(block, operand)[prepared.index], Known(want));
    }
    // No claim about Floor's nonfinite result: only the raw input boundary is unchanged.
    for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7f800001u, 0xff812345u, 0x7fcabcdeu,
                          0xffc12345u, 0x7f7fffffu, 0xff7fffffu}) {
        const Known operand[] = {bits};
        EXPECT_EQ(evaluate(block, operand)[prepared.index], Known(bits));
    }
    const Known missing[] = {std::nullopt};
    EXPECT_FALSE(evaluate(block, missing)[prepared.index]);
}
TEST(IrCvtFlrDenormExecution, ExistingConversionKeepsIndependentAnswersInBothHostProfilesAndWaves) {
    const auto* def = find_vector_conversion_op(Rdna2Format::VOP1, 0x0d);
    ASSERT_NE(def, nullptr);
    for (const auto x : kExamples) EXPECT_EQ(eval_conversion_op(*def, x.bits), Known(x.want));
    for (uint32_t wave : {32u, 64u})
        for (uint32_t form = 0; form < kForms; ++form)
            for (uint32_t mode : {0u, 0x30000u, UINT32_MAX}) {
                auto p = lane_program(words(form), 64, wave);
                p.config.compute_pgm_rsrc1 = mode;
                const auto module =
                    recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
                for (uint32_t seed : {1u, 17u, 41u}) check(module, inputs(seed), form);
            }
}
TEST(IrCvtFlrDenormExecution, UnknownAndNonfiniteConversionResultsRemainOutsideTheOracleDomain) {
    const auto* def = find_vector_conversion_op(Rdna2Format::VOP1, 0x0d);
    ASSERT_NE(def, nullptr);
    EXPECT_FALSE(eval_conversion_op(*def, std::nullopt));
    const auto p = lane_program(words(0), 64, 64);
    const auto module = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
    ASSERT_FALSE(module.empty());
    for (uint32_t bits : {0x7f800000u, 0xff800000u, 0x7fcabcdeu, 0xff800001u}) {
        EXPECT_FALSE(eval_conversion_op(*def, bits));
        auto in = inputs(1);
        in.s0[1] = bits;
        in.active[1] = true;
        for (bool flush : {false, true})
            EXPECT_EQ(execute(module, in, flush).error, "unsupported nonfinite unary float input");
    }
}
TEST(IrCvtFlrDenormExecution, ExportsReplaysAndExecutesEachExactSavedModule) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_IR_WITNESS_DIR");
    const fs::path root = configured ? fs::path(configured)
                                     : prosper_test::test_scratch_dir() / "cvt-flr-denorm-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream manifest(root / "manifest.tsv");
    manifest << "stem\twave\tform\trsrc1\n";
    size_t count = 0;
    for (uint32_t wave : {32u, 64u})
        for (uint32_t form = 0; form < kForms; ++form)
            for (uint32_t mode : {0u, 0x30000u, UINT32_MAX}) {
                auto p = lane_program(words(form), 64, wave);
                p.config.compute_pgm_rsrc1 = mode;
                const uint64_t address = 0x5344000000ull + count * 0x100;
                const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(count);
                std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
                bin.write(reinterpret_cast<const char*>(p.code.data()),
                          static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
                bin.close();
                ASSERT_TRUE(bin.good());
                ASSERT_TRUE(write_refused_compute_inputs(dump / (stem + ".inputs"), address,
                                                         &p.table, p.config, "synthetic-cvt-flr"));
                const auto read = read_refused_shader_inputs(dump / (stem + ".inputs"));
                ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                const auto replay = replay_refused_shader_inputs(p.code, read);
                ASSERT_TRUE(replay.produced) << replay.reject;
                ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                const auto file = root / (stem + ".spv");
                std::ofstream saved(file, std::ios::binary);
                saved.write(reinterpret_cast<const char*>(replay.spirv.data()),
                            static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
                saved.close();
                ASSERT_TRUE(saved.good());
                std::vector<uint32_t> loaded(replay.spirv.size());
                std::ifstream input(file, std::ios::binary);
                input.read(reinterpret_cast<char*>(loaded.data()),
                           static_cast<std::streamsize>(loaded.size() * sizeof(uint32_t)));
                ASSERT_TRUE(input.good());
                ASSERT_EQ(loaded, replay.spirv);
                check(loaded, inputs(17), form);
                manifest << stem << '\t' << wave << '\t' << form << '\t' << mode << '\n';
                ++count;
            }
    std::ofstream control(root / "native-floor-control.spv", std::ios::binary);
    const auto module = native_floor();
    control.write(reinterpret_cast<const char*>(module.data()),
                  static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    control.close();
    ASSERT_TRUE(control.good());
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 42u);
}
