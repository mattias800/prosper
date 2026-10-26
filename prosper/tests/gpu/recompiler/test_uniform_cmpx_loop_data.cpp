// #5497: a uniform scalar-counter exit must retain genuine preheader DATA
// ancestry without admitting scalar live-outs or lane-varying exits.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace {
using namespace prosper::gpu;
ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0);
    c.tgid_x_en = true; // genuine uniform s16, dynamic to the compiler
    prosper::test::complete_compute_launch(c);
    return c;
}

ShaderResourceTable resources() {
    ShaderResource out;
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(out);
    return table;
}

bool has_op(const std::vector<uint32_t>& module, uint32_t opcode) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16u;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == opcode) return true;
        at += width;
    }
    return false;
}
} // namespace

namespace {
std::vector<uint32_t> uniform_counter_program() {
    return {
        0x7e040300u, // output index = lane
        0x7e020280u, // output count = 0
        0x7e0e0281u, // genuine carried v7
        0xbe84047eu, // save full EXEC
        0x7e12023cu, // v9 = absent s60 outside the eventual cohort
        0x7da6009fu, // EXEC = (31 <= lane)
        0xe0302000u, 0x80030902u, // v9 = input[lane], in this cohort only
        0xbf8c3f70u,
        0xbea90380u, // s41 = 0, uniform scalar counter
        0x7e000281u, // header pc10: replace v0 before its first read
        0x4a100f00u, // v8 = v0 + v7
        0x7e000308u, // v0 = v8
        0x7d841300u, // fresh VCC: v0 == loaded v9, before the exit condition
        0x7da306f9u, 0x86860029u, // CMPX_LT_U32 s41,3, full-word scalar SDWA
        0xbf880006u, // EXECZ -> pc23
        0x4a060281u, // v3 = count + 1
        0x02020701u, // count = VCC ? v3 : count
        0x7e00023cu, // unavailable old v0, dead before the next active read
        0x7e0e0281u, // genuine v7 on the back edge
        0x81298129u, // s41 += 1
        0xbf82fff3u, // back to pc10
        0xbefe0404u, // restore full EXEC
        0xbea90380u, // no scalar-counter live-out
        0xe0702000u, 0x80020102u, // output[lane] = count
        0xbf810000u,
    };
}
ShaderResourceTable counter_resources() {
    auto table = resources();
    auto input = table.resources.front();
    input.binding = 4;
    input.sgpr_base = 12;
    input.size = 64 * 4;
    table.resources.push_back(input);
    return table;
}

std::vector<uint32_t> counter_capped_data_program() {
    auto code = uniform_counter_program();
    const auto replace = [&](uint32_t from, uint32_t to) {
        const auto at = std::find(code.begin(), code.end(), from);
        EXPECT_NE(at, code.end());
        if (at != code.end()) *at = to;
    };
    replace(0x7da6009fu, 0x7da6008fu); // incoming cohort: lanes 15..63
    replace(0x7e000281u, 0x7e000304u); // header v0 = carried v4, before its first read
    replace(0x7d841300u, 0x7da21300u); // DATA CMPX_LT_U32 v0,v9, before scalar cap
    replace(0x4a060281u, 0x4a020281u); // count += 1 in active lanes
    replace(0x02020701u, 0x4a080881u); // v4 += 1 on the back edge
    code.insert(code.begin() + 1, 0x36080483u); // v4 = lane & 3, under full entry EXEC
    return code;
}
} // namespace

TEST(UniformCmpxLoopData, AUniformCounterExitKeepsThePreheaderDataCohort) {
    const auto code = uniform_counter_program();
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120100ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    ASSERT_TRUE(has_op(module, 246));
    ASSERT_FALSE(has_op(module, 251));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto out = vm.run_buffers(
        64, {{4, std::vector<uint32_t>(64, 2)}, {3, std::vector<uint32_t>(80, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 80u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(out[lane], lane >= 31 ? 3u : 0u) << lane;
    for (uint32_t slot = 64; slot < out.size(); ++slot) EXPECT_EQ(out[slot], 0xdeadbeefu) << slot;
}

TEST(UniformCmpxLoopData, AnUnavailableActiveLoadStillRefuses) {
    auto code = uniform_counter_program();
    code[6] = 0x7e12023cu;
    code[7] = 0xbf800000u;
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120200ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source pc=13"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(UniformCmpxLoopData, ARequiredBackedgeInputStillRefuses) {
    auto code = uniform_counter_program();
    code[20] = 0x7e0e023du; // v7 = absent s61, required before the next rewrite
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120300ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}

TEST(UniformCmpxLoopData, AMissingDescriptorCannotBorrowTheDataBound) {
    const auto code = uniform_counter_program();
    const auto c = config();
    const auto table = resources();
    constexpr uint64_t address = 0x5497120400ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=6"), std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(UniformCmpxLoopData, AScalarCounterLiveOutCannotBorrowTheDataBound) {
    auto code = uniform_counter_program();
    code[24] = 0x7e0e0229u; // observe the counter after restoring the full entry wave
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120500ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}

TEST(UniformCmpxLoopData, ALaneVaryingExitCannotBorrowTheUniformCounterProof) {
    auto code = uniform_counter_program();
    code[15] = 0x06860007u; // CMPX consumes v7 instead of scalar s41
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120600ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}

TEST(UniformCmpxLoopData, AReadfirstCounterCannotBorrowTheUniformCounterProof) {
    auto code = uniform_counter_program();
    code[20] = 0x7e520507u; // s41 = readfirstlane(v7), before its scalar increment
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120700ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}

TEST(UniformCmpxLoopData, AnAlreadyGenuineGlobalWordKeepsTheNativeRoute) {
    auto code = uniform_counter_program();
    code[4] = 0x7e120282u; // genuine v9 = 2 in the whole entry wave
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120800ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    EXPECT_TRUE(has_op(module, 246));
    EXPECT_FALSE(has_op(module, 251));
}

TEST(UniformCmpxLoopData, ADataCmpxCanExitEarlyBeforeTheSeparateScalarCap) {
    const auto code = counter_capped_data_program();
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120900ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    ASSERT_TRUE(has_op(module, 246));
    ASSERT_FALSE(has_op(module, 251));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto out = vm.run_buffers(
        64, {{4, std::vector<uint32_t>(64, 4)}, {3, std::vector<uint32_t>(80, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 80u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(out[lane], lane >= 15 ? 3u - (lane & 3u) : 0u) << lane;
    for (uint32_t slot = 64; slot < out.size(); ++slot) EXPECT_EQ(out[slot], 0xdeadbeefu) << slot;
}

TEST(UniformCmpxLoopData, ADataCmpxStillRefusesItsMissingLoadedOperand) {
    auto code = counter_capped_data_program();
    code[7] = 0x7e12023cu;
    code[8] = 0xbf800000u;
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120a00ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("fabricated-fresh-mask-source pc=14"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

TEST(UniformCmpxLoopData, ADataCmpxStillRefusesItsRequiredBackedgeInput) {
    auto code = counter_capped_data_program();
    code[21] = 0x7e0e023du; // v7 = absent s61, needed by the next current-lane rewrite
    const auto c = config();
    const auto table = counter_resources();
    constexpr uint64_t address = 0x5497120b00ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}
