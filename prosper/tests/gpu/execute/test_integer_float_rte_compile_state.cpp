// An RTE32 verdict changes admission and module requirements: cache and replay must retain it.
#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/graphics_shader_compile.hpp"
#include "gpu/capture/fragment_compile_case.hpp"
#include "gpu/execute/shader_cache_internal.hpp"
#include "gpu/diagnostics/refused_shader_inputs.hpp"
#include "gpu/recompiler/float32_rte_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/ir/integer_float_rte_e64_cases.hpp"
#include "fixtures/test_scratch.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <cstddef>
#include <string>
using namespace prosper::gpu;
namespace cases = prosper::test::ir::integer_float_rte;
namespace {
class IntegerFloatRteCompileState : public ::testing::Test {
    void SetUp() override {
        reset_float32_rte_support_for_test();
        clear_shader_recompile_cache();
    }
    void TearDown() override {
        reset_float32_rte_support_for_test();
        clear_shader_recompile_cache();
    }
};
}   // namespace
TEST_F(IntegerFloatRteCompileState, KeysIncludeTheCapabilityForEveryCachedStage) {
    for (auto stage :
         {ShaderProgramStage::Compute, ShaderProgramStage::Fragment, ShaderProgramStage::Vertex}) {
        ShaderCompileKey a;
        a.stage = stage;
        ShaderCompileKey b = a;
        b.float32_rte = true;
        EXPECT_FALSE(a == b);
        EXPECT_NE(ShaderCompileKeyHash::compute(a), ShaderCompileKeyHash::compute(b));
        ShaderCompileKey same = b;
        EXPECT_EQ(same, b);
        EXPECT_EQ(ShaderCompileKeyHash::compute(same), ShaderCompileKeyHash::compute(b));
    }
}
TEST_F(IntegerFloatRteCompileState, CachedComputeRecompilesWhenCapabilityPublicationChanges) {
    const auto p = cases::program({0x185});
    const auto compile = [&](uint64_t& identity) {
        return recompile_compute_shader_cached(p.code.data(), p.code.size(), &p.table, p.config,
                                               &identity);
    };
    uint64_t absent = 0, supported = 0, reused = 0;
    EXPECT_TRUE(compile(absent).empty());
    publish_float32_rte_support(true, true, true);
    const auto module = compile(supported);
    ASSERT_FALSE(module.empty());
    EXPECT_EQ(compile(reused), module);
    EXPECT_NE(supported, 0u);
    EXPECT_EQ(reused, supported);
    publish_float32_rte_support(false, true, true);
    EXPECT_TRUE(compile(reused).empty());
    EXPECT_NE(reused, supported);
 // An explicitly supplied request must still control its own key and compiler read.
    auto explicit_request = p.config;
    explicit_request.choices.set(CompilerChoice::Float32Rte, 1);
    EXPECT_EQ(
        recompile_compute_shader_cached(p.code.data(), p.code.size(), &p.table, explicit_request),
        module);
}
TEST_F(IntegerFloatRteCompileState, SidecarsRetainBothCapabilityVerdictsAndLegacyAbsence) {
    auto p = cases::program({0x186});
    const auto legacy_identity = refused_compute_inputs_identity(&p.table, p.config);
    uint64_t first_identity = 0;
    for (bool capable : {false, true}) {
        reset_float32_rte_support_for_test();
        publish_float32_rte_support(!capable, true, true);
        p.config.choices.set(CompilerChoice::Float32Rte, capable);
        const auto identity = refused_compute_inputs_identity(&p.table, p.config);
        EXPECT_NE(identity, legacy_identity);
        if (capable)
            EXPECT_NE(identity, first_identity);
        else
            first_identity = identity;
        const auto path = prosper_test::test_scratch_path(capable ? "rte32-supported.inputs"
                                                                  : "rte32-unsupported.inputs");
        ASSERT_TRUE(write_refused_compute_inputs(path, 0x5394000000ull, &p.table, p.config,
                                                 "synthetic capability witness"));
        const auto saved = read_refused_shader_inputs(path);
        EXPECT_EQ(saved.compute.choices.get(CompilerChoice::Float32Rte), capable ? 1 : 0);
        const auto replay = replay_refused_shader_inputs(p.code, saved);
        EXPECT_TRUE(replay.incomplete.empty()) << replay.incomplete;
        EXPECT_EQ(replay.produced, capable);
        EXPECT_EQ(replay.spirv,
                  recompile_compute(p.code.data(), p.code.size(), &p.table, p.config));
    }
    p.config.choices = CompilerChoiceSet{};
    const auto path = prosper_test::test_scratch_path("rte32-legacy.inputs");
    ASSERT_TRUE(write_refused_compute_inputs(path, 0x5394000000ull, &p.table, p.config,
                                             "legacy capability unknown"));
    EXPECT_FALSE(read_refused_shader_inputs(path)
                     .compute.choices.get(CompilerChoice::Float32Rte)
                     .has_value());
}

TEST_F(IntegerFloatRteCompileState, GraphicsCompilerPinsItsKeyAndRetainsTheChoiceTranscript) {
    for (bool capable : {false, true}) {
        reset_float32_rte_support_for_test();
        publish_float32_rte_support(!capable, true, true);
        auto code = cases::words({0x185, 32, 50});
        code.insert(code.begin(), {0xbeb203ff, 0x01000003});
        code.insert(code.end(), {0xf800080f, 0x01010101, 0xbf810000});
        ShaderCompileKey key;
        key.stage = ShaderProgramStage::Fragment;
        key.fragment_float_mode = {true, 0};
        key.float32_rte = capable;
        key.code = std::make_shared<const std::vector<uint32_t>>(code);
        CompilerChoiceTrace trace;
        std::vector<uint32_t> module;
        {
            CompilerChoiceScope recording(trace);
            module = compile_graphics_shader(ShaderProgramStage::Fragment, key, nullptr,
                                             0x5394001000ull, nullptr);
        }
        EXPECT_EQ(!module.empty(), capable);
        size_t found = 0;
        for (const auto& read : trace.reads)
            if (read.choice == CompilerChoice::Float32Rte) {
                ++found;
                EXPECT_EQ(read.value, capable);
            }
        EXPECT_GT(found, 0u);
        FragmentCompileCase record;
        record.complete = false;
        record.reason = "synthetic capability transcript";
        record.choices = trace;
        const auto encoded = encode_fragment_compile_case(record);
        EXPECT_EQ(decode_fragment_compile_case(encoded).choices.reads, trace.reads);
    }
}
