// SAVEEXEC into VCC must retain OLD EXEC's physical DATA before a later one-word overwrite.
// Balloting the new EXEC, or dropping the high word, changes a subsequent numeric observation.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/arbitrary_placeholder.hpp"
#include "fixtures/compute_runner.h"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr std::array<uint32_t, 10> saves{0x24, 0x25, 0x26, 0x27, 0x28,
                                         0x29, 0x2a, 0x2b, 0x37, 0x38};
constexpr uint32_t replacement = 0x13579bdfu;
std::vector<uint32_t> program(uint32_t opcode, uint32_t active = 64, bool replace = false) {
    std::vector<uint32_t> code{0xbe9404c1u}; // s[20:21]=-1
    if (active != 64) {
        code.push_back(0x7d880080u | active); // VCC=(active>v0)
        code.push_back(0xbefe046au);         // EXEC=VCC
    }
    code.insert(code.end(), {0xbeea0014u | (opcode << 8), 0xbefe04c1u});
    if (replace) code.insert(code.end(), {0xbeeb03ffu, replacement});
    // Overwrite only LO; CSELECT must keep the actual current HI, which the final VMOV reads.
    code.insert(code.end(), {0x806a8080u, 0xbf068080u, 0x856a8281u, 0x7e02026bu, 0xe0702000u,
                             0x80020100u, 0xbf810000u});
    return code;
}
std::vector<uint32_t>
compile(const std::vector<uint32_t>& code,
        const ComputeShaderConfig& config = prosper::test::native_execution_launch()) {
    const auto resources = prosper::test::arbitrary_placeholder::table();
    return recompile_compute(code.data(), code.size(), &resources, config);
}
void values(uint32_t opcode, uint32_t active, bool replace = false) {
    SCOPED_TRACE(opcode);
    SCOPED_TRACE(active);
    const auto module = compile(program(opcode, active, replace));
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    const auto output = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    const uint32_t expected = replace        ? replacement
                              : active == 64 ? UINT32_MAX
                              : active <= 32 ? 0u
                                             : (1u << (active - 32)) - 1u;
    EXPECT_EQ(output, std::vector<uint32_t>(64, expected));
}
} // namespace

TEST(NativeSaveexecVccData, EverySaveexecFormRetainsTheOldHighDword) {
    for (uint32_t opcode : saves) values(opcode, 64);
}
TEST(NativeSaveexecVccData, NarrowedExecIsCapturedBeforeTheUpdateAndRestore) {
    for (uint32_t opcode : saves)
        for (uint32_t active : {0u, 32u, 40u, 63u}) values(opcode, active);
}
TEST(NativeSaveexecVccData, GenuineLiteralReplacementEndsTheSavedGeneration) {
    for (uint32_t opcode : saves) values(opcode, 40, true);
}
TEST(NativeSaveexecVccData, UnsupportedPhysicalProfilesCannotSupplyBallotWords) {
    for (unsigned mutation : {0u, 1u, 2u}) {
        auto config = prosper::test::native_execution_launch();
        if (mutation == 0) config.dispatch_launch.known = 0;
        if (mutation == 1) config.exact_thread_extent = true;
        if (mutation == 2) config.native_subgroup_size = 0;
        for (uint32_t opcode : saves) EXPECT_TRUE(compile(program(opcode), config).empty());
    }
}
TEST(NativeSaveexecVccData, OldExecWordsExecuteOnARequiredWave64Subgroup) {
    if (!prosper::test::default_compute_required_subgroup_supported(64, 64))
        GTEST_SKIP() << "native Wave64 compute unavailable";
    for (uint32_t opcode : saves) {
        for (uint32_t active : {40u, 64u}) {
            SCOPED_TRACE(opcode);
            SCOPED_TRACE(active);
            const auto module = compile(program(opcode, active));
            ASSERT_FALSE(module.empty());
            std::vector<uint32_t> output;
            prosper::test::run_compute(module, {0.0f}, 64, 1, {},
                                       std::vector<uint32_t>(64, 0xdeadbeefu), &output, 64, nullptr,
                                       nullptr, nullptr, 64);
            EXPECT_EQ(output, std::vector<uint32_t>(64, active == 64 ? UINT32_MAX : 255u));
        }
    }
}
