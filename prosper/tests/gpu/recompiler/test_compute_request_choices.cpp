// ADR 0006 / spec GPU-1: recompile_compute takes its switch-type choices from the request.
//
// The probe is the program tests/gpu/test_recompile_coverage.cpp uses for #3143: one image_load whose
// descriptor cannot resolve against an empty resource table. With PROSPER_MIMG_SOFT unset the compile
// is refused; with it set the emitter substitutes a constant and compiles. That makes the switch an
// output-changing lever whose position is visible in whether a module comes back at all.
#include "gpu/recompiler/compiler_choices.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <stdexcept>
#include <vector>

using namespace prosper::gpu;
namespace {
void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

// Restores the switch however the test exits, so a failing assertion cannot leak it to a neighbour.
struct MimgSoftEnv {
    explicit MimgSoftEnv(const char* value) { set_env("PROSPER_MIMG_SOFT", value); }
    ~MimgSoftEnv() { set_env("PROSPER_MIMG_SOFT", nullptr); }
};

std::vector<uint32_t> compile_probe(const ComputeShaderConfig& config) {
    static const uint32_t probe[] = {
        0x7E080300u,               // v_mov_b32 v4, v0 (1D image coordinate)
        0xF0000F00u, 0x00000004u,  // image_load v[0:3], v4, s[0:7], dmask:0xf dim:1D
        0xBF810000u,               // s_endpgm
    };
    ShaderResourceTable empty_rt;   // no resources: s[0:7] resolves to nothing
    return recompile_compute(probe, std::size(probe), &empty_rt, config,
                             RecompileDiagnosticContext{RecompileDiagnosticStage::Compute, 0x6a1ull});
}

ComputeShaderConfig probe_config() {
    ComputeShaderConfig config;
    config.user_sgprs.resize(16);
    return config;
}
}   // namespace

// Red without ADR 0006: the emitter read the environment, so a request value changed nothing.
TEST(ComputeRequestChoices, RequestValueSelectsTheSoftPathWithTheEnvironmentUnset) {
    MimgSoftEnv env(nullptr);
    auto config = probe_config();
    EXPECT_TRUE(compile_probe(config).empty()) << "no switch anywhere: the unresolved MIMG is refused";
    config.choices.set(CompilerChoice::MimgSoft, 1);
    EXPECT_FALSE(compile_probe(config).empty()) << "the request alone must soften it";
}

// Red without ADR 0006: an armed environment overrode what the request said.
TEST(ComputeRequestChoices, ArmedEnvironmentCannotOverrideTheRequest) {
    MimgSoftEnv env("1");
    auto config = probe_config();
    config.choices.set(CompilerChoice::MimgSoft, 0);
    EXPECT_TRUE(compile_probe(config).empty()) << "request says off; the shell must not turn it on";
}

// The same request compiles to the same words whatever the shell holds.
TEST(ComputeRequestChoices, SameRequestSameModuleRegardlessOfEnvironment) {
    auto config = probe_config();
    config.choices.set(CompilerChoice::MimgSoft, 1);
    std::vector<uint32_t> unarmed, armed;
    {
        MimgSoftEnv env(nullptr);
        unarmed = compile_probe(config);
    }
    {
        MimgSoftEnv env("1");
        armed = compile_probe(config);
    }
    ASSERT_FALSE(unarmed.empty());
    EXPECT_EQ(unarmed, armed);
}

// Transitional, and pinned so its removal is deliberate: an entry the caller left unset still reads the
// environment, which is what the stages that have not migrated yet rely on. Delete this test with the
// last such stage.
TEST(ComputeRequestChoices, UnsetEntryStillReadsTheEnvironmentUntilEveryStageMigrates) {
    MimgSoftEnv env("1");
    EXPECT_FALSE(compile_probe(probe_config()).empty());
}

TEST(ComputeRequestChoices, FromEnvironmentBuildsEverySwitchChoiceFromTheShell) {
    {
        MimgSoftEnv env("1");
        const auto set = compiler_choices_from_environment();
        EXPECT_EQ(set.get(CompilerChoice::MimgSoft), 1);
        EXPECT_EQ(set.get(CompilerChoice::NormalizeSamplerCoordinates), 1);
        EXPECT_TRUE(set.get(CompilerChoice::ForcedArrayLayer).has_value());
        EXPECT_TRUE(set.get(CompilerChoice::VccScalarMerge).has_value());
        EXPECT_FALSE(set.get(CompilerChoice::FloatControls).has_value())
            << "a device capability, not a switch: it belongs to the target description";
    }
    MimgSoftEnv env(nullptr);
    EXPECT_EQ(compiler_choices_from_environment().get(CompilerChoice::MimgSoft), 0);
}

TEST(CompilerChoiceScope, SuppliedValueAnswersWithoutCallingTheLiveRead) {
    CompilerChoiceSet supplied;
    supplied.set(CompilerChoice::VccScalarMerge, 7);
    CompilerChoiceScope scope(supplied);
    const auto value = compiler_choice(CompilerChoice::VccScalarMerge, []() -> int64_t {
        throw std::logic_error("live read must not run");
    });
    EXPECT_EQ(value, 7);
}

TEST(CompilerChoiceScope, UnsetChoiceFallsThroughToTheLiveRead) {
    CompilerChoiceSet supplied;   // nothing set
    CompilerChoiceScope scope(supplied);
    EXPECT_EQ(compiler_choice(CompilerChoice::FragmentTapPc, [] { return int64_t{42}; }), 42);
}

TEST(CompilerChoiceScope, ReplayBeatsSuppliedAndSuppliedReadsAreRecorded) {
    CompilerChoiceSet supplied;
    supplied.set(CompilerChoice::MimgSoft, 1);
    CompilerChoiceTrace recorded;
    {
        CompilerChoiceScope outer(recorded);
        CompilerChoiceScope inner(supplied);
        EXPECT_EQ(compiler_choice(CompilerChoice::MimgSoft, []() -> int64_t { return 99; }), 1);
    }
    ASSERT_EQ(recorded.reads.size(), 1u);
    EXPECT_EQ(recorded.reads[0].choice, CompilerChoice::MimgSoft);
    EXPECT_EQ(recorded.reads[0].value, 1);

    CompilerChoiceTrace building;
    building.reads.push_back({CompilerChoice::MimgSoft, 5});
    const CompilerChoiceTrace replayed = building;   // const selects the replay constructor
    CompilerChoiceScope outer(supplied);
    CompilerChoiceScope replay(replayed);
    EXPECT_EQ(compiler_choice(CompilerChoice::MimgSoft, []() -> int64_t { return 99; }), 5)
        << "a replayed trace owns the sequence even under a supplied request";
}
