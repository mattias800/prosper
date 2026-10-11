// #4958: legacy native execution fixtures need their real raw group-count launch. These
// compile the consuming instructions, so a metadata-only constructor check cannot pass
// while the compiler still refuses physical words; partial or absent launches remain refused.
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
constexpr std::array<uint32_t, 6> exec_words{
    0x7d8800a1u, // compare gives lanes 0..32, hence asymmetric low/high halves
    0xbefe046au, 0x80907e80u, 0x82917f80u, 0xbefe04c1u, 0xbf810000u,
};
size_t count_opcode(const std::vector<uint32_t>& module, uint32_t opcode) {
    size_t count = 0;
    for (size_t at = 5; at < module.size();) {
        const size_t width = module[at] >> 16;
        if (!width || width > module.size() - at) return SIZE_MAX;
        count += (module[at] & 0xffffu) == opcode;
        at += width;
    }
    return count;
}
std::vector<uint32_t> compile_exec(const ComputeShaderConfig& config, uint64_t address) {
    return recompile_compute(exec_words.data(), exec_words.size(), nullptr, config,
                             {RecompileDiagnosticStage::Compute, address});
}
void physical_words_refuse(const ComputeShaderConfig& config, uint64_t address) {
    ASSERT_FALSE(compile_exec(prosper::test::native_execution_launch(), address + 1).empty());
    EXPECT_TRUE(compile_exec(config, address).empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("saved-mask"), std::string::npos)
        << last_terminal_reject_reason(address);
}
} // namespace

TEST(NativeExecutionLaunch, BothExecWordsHaveCheckedEntryAuthority) {
    const auto module = compile_exec(prosper::test::native_execution_launch(), 0x4958000100ull);
    ASSERT_FALSE(module.empty());
    EXPECT_EQ(count_opcode(module, 339), 2u);
}

TEST(NativeExecutionLaunch, SelfVccNotCanReloadItsCheckedPhysicalWord) {
    const std::array<uint32_t, 16> words{
        0xbe800385u, 0x7d840000u, 0xbeea076au, 0xbf880000u, 0x876ac16au, 0x7e02026au,
        0xbefe04c1u, 0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
        0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u,
    };
    const auto config = prosper::test::native_execution_launch();
    EXPECT_FALSE(recompile_compute(words.data(), words.size(), nullptr, config).empty());
}

TEST(NativeExecutionLaunch, DualDomainNotKeepsItsScalarSccAcrossTheReload) {
    const std::array<uint32_t, 16> words{
        0xbe800385u, 0x7d840000u, 0xbeea046au, 0xbeea076au, 0xbf880000u, 0x85018081u,
        0x7e020201u, 0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
        0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u,
    };
    const auto config = prosper::test::native_execution_launch();
    EXPECT_FALSE(recompile_compute(words.data(), words.size(), nullptr, config).empty());
}

TEST(NativeExecutionLaunch, GeometryAloneCannotSupplyTheSameWords) {
    auto config = prosper::test::native_execution_launch();
    config.dispatch_launch = {};
    physical_words_refuse(config, 0x4958000200ull);
}

TEST(NativeExecutionLaunch, MissingRawProducingFieldsCannotSupplyTheSameWords) {
    uint64_t address = 0x4958000300ull;
    for (uint32_t bit = 1; bit <= ComputeDispatchLaunchState::kDimZ; bit <<= 1) {
        auto config = prosper::test::native_execution_launch();
        config.dispatch_launch.known &= ~bit;
        physical_words_refuse(config, address);
        address += 0x100;
    }
}

TEST(NativeExecutionLaunch, PartialGroupFlagCannotSupplyTheSameWords) {
    auto config = prosper::test::native_execution_launch();
    config.dispatch_launch.initiator |= 2u;
    physical_words_refuse(config, 0x4958001000ull);
}

TEST(NativeExecutionLaunch, ExactShortExtentCannotBorrowACompleteGroup) {
    auto config = prosper::test::native_execution_launch();
    config.exact_thread_extent = true;
    config.threads_x = 33;
    physical_words_refuse(config, 0x4958001100ull);
}
} // namespace prosper::gpu
