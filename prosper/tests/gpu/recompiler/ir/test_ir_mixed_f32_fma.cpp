// #5225: v_fma_mix_f32 is fused after the independent full/half source selections.
#include "fixtures/mixed_f32_fma.hpp"
#include "fixtures/test_scratch.h"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::test::mixed_f32;
namespace {
std::vector<uint32_t> compile(Shape shape, uint32_t wave) {
    auto p = program(shape, wave);
    return recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
}

void check(Shape shape, const std::vector<uint32_t>& module) {
    const auto in = inputs(shape);
    std::vector<uint32_t> buffer;
    for (const auto* values : {&in.s0, &in.s1, &in.old_d})
        buffer.insert(buffer.end(), values->begin(), values->end());
    for (bool active : in.active) buffer.push_back(active);
    buffer.insert(buffer.end(), in.s2.begin(), in.s2.end());
    bpermute_oracle::Interpreter vm(module);
    vm.half_rte_preserve_denorm = true;
    const auto actual = vm.run_buffers(64, {{3, buffer}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(actual.size(), buffer.size());
    for (std::size_t lane = 0; lane < in.lanes(); ++lane)
        EXPECT_EQ(actual[lane], expected(shape, in, lane)) << "lane " << lane;
}
}   // namespace

TEST(IrMixedF32Fma, LlvmEncodingsSelectIndependentHalfAndHighFields) {
    EXPECT_EQ(words(), (std::vector<uint32_t>{0xcc200001, 0x04160702}));
    EXPECT_EQ(words({2, 2}), (std::vector<uint32_t>{0xcc201001, 0x14160702}));
    const auto code = words({2, 2, 1, 1, true});
    const auto in = rdna2_decode_one(code.data(), code.size());
    EXPECT_EQ(in.fmt, Rdna2Format::VOP3P);
    EXPECT_EQ(in.opcode, 0x20u);
    EXPECT_EQ(in.vop3p_opsel, 2);
    EXPECT_EQ(in.vop3p_opsel_hi, 2);
    EXPECT_TRUE(in.src_abs[0]);
    EXPECT_TRUE(in.src_neg[0]);
    EXPECT_TRUE(in.clamp);
}

TEST(IrMixedF32Fma, HandWorkedFullAndMixedCancellationWitnesses) {
    EXPECT_EQ(expected({}, inputs(), 4), 0xa8800000u);   // (1+2^-23)(1-2^-23)-1 = -2^-46
    const Shape mixed{2, 2};
    // (1+2^-23)(1-2^-10)-(1-2^-10) = 2^-23-2^-33. Separate rounding gives 0x34000000.
    EXPECT_EQ(expected(mixed, inputs(mixed), 4), 0x33ffc000u);
    for (uint32_t wave : {32u, 64u})
        for (const Shape shape : {Shape{}, mixed}) {
            const auto module = compile(shape, wave);
            ASSERT_FALSE(module.empty());
            check(shape, module);
        }
}

TEST(IrMixedF32Fma, ExistingSelectorsOperandsAndModifiersKeepInactiveDestinations) {
    std::size_t index = 0;
    for (const Shape shape : shapes()) {
        SCOPED_TRACE(std::to_string(index++));
        EXPECT_FALSE(inputs(shape).active[0]);
        EXPECT_TRUE(inputs(shape).active[4]);
        for (uint32_t wave : {32u, 64u}) {
            const auto module = compile(shape, wave);
            ASSERT_FALSE(module.empty());
            check(shape, module);
        }
    }
    EXPECT_EQ(index, 84u);
}

TEST(IrMixedF32Fma, ResultIsOneFmaWithoutASeparatelyRoundedProduct) {
    for (const Shape shape : {Shape{}, Shape{2, 2}, Shape{7, 5}}) {
        const auto module = compile(shape, 64);
        ASSERT_FALSE(module.empty());
        std::size_t fma = 0, separate = 0;
        for (std::size_t pc = 5; pc < module.size(); pc += module[pc] >> 16) {
            const uint32_t count = module[pc] >> 16, opcode = module[pc] & 65535;
            ASSERT_GT(count, 0u);
            ASSERT_LE(pc + count, module.size());
            if (opcode == Op_ExtInst && count >= 5 && module[pc + 4] == Glsl_Fma) ++fma;
            if (opcode == Op_FMul || opcode == Op_FAdd) ++separate;
        }
        EXPECT_EQ(fma, 1u);
        EXPECT_EQ(separate, 0u);
    }
}

TEST(IrMixedF32Fma, ExportsAndExecutesSavedF32WitnessesWithUnchangedHalfControls) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process launch input.
    const char* configured = std::getenv("PROSPER_MIXED_F32_WITNESS_DIR");
    const auto root =
        configured ? fs::path(configured) : prosper_test::test_scratch_dir() / "mixed-f32-witness";
    if (!configured) {
        std::error_code error;
        fs::remove_all(root, error);
        ASSERT_FALSE(error) << error.message();
    }
    ASSERT_FALSE(fs::exists(root));
    const auto dump = root / "refused_shaders_20000101-000000Z_synthetic";
    ASSERT_TRUE(fs::create_directories(dump));
    std::ofstream manifest(root / "manifest.tsv");
    manifest << "stem\topcode\twave\thalf\thigh\tabs\tneg\tclamp\tform\tslot\n";
    std::size_t count = 0;
    for (uint32_t opcode : {0x20u, 0x21u, 0x22u}) {
        const auto forms =
            opcode == 0x20 ? shapes() : std::vector<Shape>{{}, {2, 2}, {7, 5}, {0, 0, 0, 0, true}};
        for (const Shape shape : forms)
            for (uint32_t wave : {32u, 64u}) {
                auto p = program(shape, wave, opcode);
                const uint64_t address = 0x5225200000ull + count * 0x100;
                const auto stem = "cs_" + std::to_string(address) + "_" + std::to_string(count);
                std::ofstream bin(dump / (stem + ".bin"), std::ios::binary);
                bin.write(reinterpret_cast<const char*>(p.code.data()),
                          static_cast<std::streamsize>(p.code.size() * sizeof(uint32_t)));
                bin.close();
                ASSERT_TRUE(bin.good());
                const auto file = dump / (stem + ".inputs");
                ASSERT_TRUE(write_refused_compute_inputs(file, address, &p.table, p.config,
                                                         "synthetic-mixed-f32-witness"));
                const auto read = read_refused_shader_inputs(file);
                ASSERT_TRUE(read.incomplete.empty()) << read.incomplete;
                const auto replay = replay_refused_shader_inputs(p.code, read);
                ASSERT_TRUE(replay.produced) << replay.reject;
                ASSERT_TRUE(replay.incomplete.empty()) << replay.incomplete;
                const auto path = root / (stem + ".spv");
                std::ofstream spv(path, std::ios::binary);
                spv.write(reinterpret_cast<const char*>(replay.spirv.data()),
                          static_cast<std::streamsize>(replay.spirv.size() * sizeof(uint32_t)));
                spv.close();
                ASSERT_TRUE(spv.good());
                std::vector<uint32_t> loaded(replay.spirv.size());
                std::ifstream saved(path, std::ios::binary);
                saved.read(reinterpret_cast<char*>(loaded.data()),
                           static_cast<std::streamsize>(loaded.size() * sizeof(uint32_t)));
                ASSERT_TRUE(saved.good());
                ASSERT_EQ(loaded, replay.spirv);
                if (opcode == 0x20) check(shape, loaded);
                // Half-result modules are byte controls, not a claim to have fixed #5225.
                manifest << stem << '\t' << opcode << '\t' << wave << '\t' << shape.half << '\t'
                         << shape.high << '\t' << shape.abs << '\t' << shape.neg << '\t'
                         << shape.clamp << '\t' << int(shape.form) << '\t' << shape.slot << '\n';
                ++count;
            }
    }
    manifest.close();
    ASSERT_TRUE(manifest.good());
    EXPECT_EQ(count, 184u);
}
