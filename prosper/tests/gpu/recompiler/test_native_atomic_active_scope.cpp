// #5497: GLC returns and numeric relays must retain genuine current-EXEC dependencies.
// The interpreter observes old memory values; absent inputs must still refuse.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace {
using namespace prosper::gpu;
std::vector<uint32_t> program(bool missing_address = false, bool missing_backedge = false,
                              bool exchange = false) {
    std::vector<uint32_t> code{
        0x7e040300u, // v2 = lane
        0x36180481u, // v12 = lane & 1
        0x7e000280u, // v0 = 0, genuine initial carried word
        0x7e060280u, // v3 = 0
        0x7e020280u, // v1 = 0, observed count
        missing_address ? 0x7e0e023cu : 0x7e0e0280u, // atomic index v7 = absent s60, or zero
        0x7e0c0283u, // v6 = 3
        0x7e540506u, // s42 = readfirstlane(v6)
        0xbea90380u, // s41 = 0
        0xbe84047eu, // save entry EXEC
        0x7da6049fu, // C: EXEC &= (31 <= v2)
    };
    const auto header = code.size();
    code.insert(code.end(),
                {
                    0xbe86047eu, // save this iteration's EXEC
                    0x7e000281u, // replace current v0 before every active read
                    0x7da41881u, // D: odd lanes inside C, spanning both Wave64 halves
                    0xbf068114u, // genuine TGID X == 1
                    0xbf850001u, // inner scalar if selects the general SCC-loop route
                    0x7e140281u, // scratch v10 = 1
                    exchange ? 0xe0c06000u : 0xe0c86000u,
                    0x80020007u, // buffer_atomic_add/exchange v0, v7, s[8:11], 0, idxen glc
                    0xbf8c3f70u,
                    0x7d840100u, // fresh VCC = (v0 == v0)
                    0x4a060081u, // v3 = v0 + 1, next buffer address
                    0xe0702000u,
                    0x80030203u, // output5[v3] = lane
                    0x4a100281u, // v8 = count + 1
                    0x02021101u, // count = VCC ? v8 : count
                    0x7e00023cu, // absent v0 on backedge; next iteration rewrites it first
                    missing_backedge
                        ? 0x7e0e023du
                        : 0x7e0e0280u, // missing actual atomic-address backedge, or zero
                    0xbefe0406u, // restore this iteration's EXEC
                    0x80298129u, // s41++
                    0xbf0a2a29u, // s41 < s42
                });
    const auto back = code.size();
    code.push_back(0xbf850000u |
                   static_cast<uint16_t>(static_cast<int>(header) - static_cast<int>(back) - 1));
    code.insert(code.end(), {
                                0xbefe0404u, // restore entry EXEC
                                0xe0702000u,
                                0x80040102u, // output3[lane] = count
                                0xbf810000u,
                            });
    return code;
}
ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(20, 0);
    c.tgid_x_en = true;
    prosper::test::complete_compute_launch(c);
    return c;
}
ShaderResourceTable resources(bool atomic = true) {
    ShaderResourceTable t;
    for (auto [base, binding, words] :
         {std::array<uint32_t, 3>{8, 4, 1}, {12, 5, 128}, {16, 3, 64}}) {
        if (!atomic && base == 8) continue;
        ShaderResource r;
        r.cls = ResourceClass::ConstantBuffer;
        r.format = DataFormat::Uint32;
        r.num_components = 1;
        r.binding = binding;
        r.stride = 4;
        r.size = words * 4;
        r.sgpr_base = base;
        t.resources.push_back(r);
    }
    return t;
}
std::vector<uint32_t> compile(uint64_t address, bool missing_address = false,
                              bool missing_backedge = false, bool resource = true,
                              bool exchange = false) {
    auto code = program(missing_address, missing_backedge, exchange);
    auto c = config();
    auto table = resources(resource);
    return recompile_compute(code.data(), code.size(), &table, c,
                             {RecompileDiagnosticStage::Compute, address});
}
bool has_op(const std::vector<uint32_t>& module, uint32_t op) {
    for (size_t at = 5; at < module.size();) {
        auto count = module[at] >> 16u;
        if (!count || count > module.size() - at) return false;
        if ((module[at] & 65535u) == op) return true;
        at += count;
    }
    return false;
}
}   // namespace
TEST(NativeAtomicActiveScope, ACompleteReturnDoesNotReadItsInactiveOldGeneration) {
    constexpr uint64_t address = 0x5497090100ull;
    const auto module = compile(address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    EXPECT_TRUE(has_op(module, 246));
    EXPECT_FALSE(has_op(module, 251));
    EXPECT_TRUE(has_op(module, 234));
}
TEST(NativeAtomicActiveScope, InterpreterObservesReturnedValuesFreshPredicatesAndThreeIterations) {
    constexpr uint64_t address = 0x5497090200ull;
    const auto module = compile(address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.allow_buffer_u32_atomics = true;
    vm.extra_writable_bindings = {4, 5};
    const auto counts = vm.run_buffers(64,
                                       {{3, std::vector<uint32_t>(64, 0xdeadbeefu)},
                                        {4, {0}},
                                        {5, std::vector<uint32_t>(128, 0xdeadbeefu)}},
                                       3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(counts.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(counts[lane], lane >= 31 && (lane & 1) ? 3u : 0u) << lane;
    const auto counter = vm.writable_result(4);
    ASSERT_EQ(counter.size(), 1u);
    EXPECT_EQ(counter[0], 51u);
    const auto returned = vm.writable_result(5);
    ASSERT_EQ(returned.size(), 128u);
    std::array<uint32_t, 64> observations{};
    EXPECT_EQ(returned[0], 0xdeadbeefu);
    for (size_t i = 1; i <= 51; ++i) {
        ASSERT_GE(returned[i], 31u) << i;
        ASSERT_LT(returned[i], 64u) << i;
        EXPECT_EQ(returned[i] & 1u, 1u);
        ++observations[returned[i]];
    }
    for (uint32_t lane = 31; lane < 64; lane += 2) EXPECT_EQ(observations[lane], 3u) << lane;
    for (size_t i = 52; i < returned.size(); ++i) EXPECT_EQ(returned[i], 0xdeadbeefu) << i;
}
TEST(NativeAtomicActiveScope, AMissingAtomicAddressStillRefuses) {
    constexpr uint64_t address = 0x5497090300ull;
    EXPECT_TRUE(compile(address, true).empty());
}
TEST(NativeAtomicActiveScope, AMissingActualAddressBackedgeStillRefuses) {
    constexpr uint64_t address = 0x5497090400ull;
    EXPECT_TRUE(compile(address, false, true).empty());
}
TEST(NativeAtomicActiveScope, AMissingAtomicDescriptorStillRefuses) {
    constexpr uint64_t address = 0x5497090500ull;
    EXPECT_TRUE(compile(address, false, false, false).empty());
}

TEST(NativeAtomicActiveScope, AnExistingBroaderCohortRetainsItsCompleteCoverage) {
    const std::vector<uint32_t> code{
        0x7e040300u, // v2 = lane
        0x36180481u, // v12 = lane & 1
        0x7e020280u, // v1 = 0 everywhere
        0x7e06023cu, // v3 = missing s60 everywhere
        0xbe84047eu, // save actual entry EXEC
        0x7da6049fu, // C: lanes 31..63
        0x7e060280u, // v3 = 0 in C
        0xbe86047eu, // save C
        0x7da41881u, // D: odd lanes inside C
        0x7e060281u, // v3 = 1 in D, preserving C minus D
        0xbefe0406u, // restore C
        0x4a100681u, // v8 = v3 + 1 across the whole C
        0x7d840703u, // fresh VCC = (v3 == v3)
        0x02021103u, // v1 = VCC ? v8 : v3
        0xbefe0404u, // restore actual entry
        0xe0702000u, 0x80040102u, // output3[lane] = v1
        0xbf810000u,
    };
    const auto c = config();
    const auto table = resources();
    constexpr uint64_t address = 0x5497090600ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto values = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(values[lane], lane < 31 ? 0u : ((lane & 1) ? 2u : 1u)) << lane;
}

TEST(NativeAtomicActiveScope, ExchangeReturnsTheOldWordInsideTheSameCurrentCohort) {
    constexpr uint64_t address = 0x5497090700ull;
    const auto module = compile(address, false, false, true, true);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    ASSERT_TRUE(has_op(module, 229));
    ASSERT_FALSE(has_op(module, 234));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.allow_buffer_u32_atomics = true;
    vm.extra_writable_bindings = {4, 5};
    const auto counts = vm.run_buffers(64,
                                       {{3, std::vector<uint32_t>(64, 0xdeadbeefu)},
                                        {4, {0}},
                                        {5, std::vector<uint32_t>(128, 0xdeadbeefu)}},
                                       3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(counts.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(counts[lane], lane >= 31 && (lane & 1) ? 3u : 0u) << lane;
    const auto counter = vm.writable_result(4);
    ASSERT_EQ(counter.size(), 1u);
    EXPECT_EQ(counter[0], 1u);
    const auto returned = vm.writable_result(5);
    ASSERT_EQ(returned.size(), 128u);
    const auto participating = [](uint32_t lane) { return lane >= 31 && lane < 64 && (lane & 1); };
    EXPECT_EQ(returned[0], 0xdeadbeefu);
    EXPECT_TRUE(participating(returned[1])) << returned[1];
    EXPECT_TRUE(participating(returned[2])) << returned[2];
    for (size_t i = 3; i < returned.size(); ++i) EXPECT_EQ(returned[i], 0xdeadbeefu) << i;
}

TEST(NativeAtomicActiveScope, MissingAtomicDataAndScalarOffsetStillRefuse) {
    for (bool offset : {false, true}) {
        SCOPED_TRACE(offset ? "missing scalar offset"
                            : "expanded reader of missing inactive VDATA");
        auto code = program();
        if (offset)
            code[18] = 0x3c020007u; // the existing atomic packet with absent s60 as SOFFSET
        else {
            // Active GLC returns read memory. Restoring C also exposes inactive lanes'
            // actual old-VDATA fallback, which the atomic did not replace.
            code[12] = 0x7e00023cu;
            code.insert(code.begin() + 20, 0xbefe0406u);
            const auto back = code.size() - 5;
            code[back] = 0xbf850000u | static_cast<uint16_t>(11 - static_cast<int>(back) - 1);
        }
        const auto c = config();
        const auto table = resources();
        const uint64_t address = 0x5497090800ull + offset;
        EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                      {RecompileDiagnosticStage::Compute, address})
                        .empty());
        if (!offset)
            EXPECT_NE(
                last_terminal_reject_reason(address).find("fabricated-fresh-mask-source pc=21 "),
                std::string::npos)
                << last_terminal_reject_reason(address);
    }
}

TEST(NativeAtomicActiveScope, APreservedOuterCohortKeepsOnlyCurrentDataDependencies) {
    constexpr uint64_t address = 0x5497090900ull;
    auto code = program();
    code[2] = 0x7e00023cu; // old v0 is missing outside the pre-loop cohort
    code[3] = 0x7e06023cu; // old v3 is missing outside the pre-loop cohort
    code.insert(code.begin() + 11, {0x7e000280u, 0x7e060280u}); // genuine v0/v3 in C
    const auto poison = std::find(code.begin() + 13, code.end(), 0x7e00023cu);
    ASSERT_NE(poison, code.end());
    code.insert(poison + 1, 0x7e06023cu); // both words are dead until their next active rewrite
    const auto back = code.size() - 5;
    code[back] = 0xbf850000u | static_cast<uint16_t>(13 - static_cast<int>(back) - 1);
    const auto c = config();
    const auto table = resources();
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    ASSERT_TRUE(has_op(module, 246));
    ASSERT_FALSE(has_op(module, 251));
    ASSERT_TRUE(has_op(module, 234));
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.allow_buffer_u32_atomics = true;
    vm.extra_writable_bindings = {4, 5};
    const auto counts = vm.run_buffers(64,
                                       {{3, std::vector<uint32_t>(64, 0xdeadbeefu)},
                                        {4, {0}},
                                        {5, std::vector<uint32_t>(128, 0xdeadbeefu)}},
                                       3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(counts.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(counts[lane], lane >= 31 && (lane & 1) ? 3u : 0u) << lane;
    const auto counter = vm.writable_result(4);
    ASSERT_EQ(counter.size(), 1u);
    EXPECT_EQ(counter[0], 51u);
    const auto returned = vm.writable_result(5);
    ASSERT_EQ(returned.size(), 128u);
    std::array<uint32_t, 64> observations{};
    EXPECT_EQ(returned[0], 0xdeadbeefu);
    for (size_t i = 1; i <= 51; ++i) {
        ASSERT_GE(returned[i], 31u) << i;
        ASSERT_LT(returned[i], 64u) << i;
        EXPECT_EQ(returned[i] & 1u, 1u);
        ++observations[returned[i]];
    }
    for (uint32_t lane = 31; lane < 64; lane += 2) EXPECT_EQ(observations[lane], 3u) << lane;
    for (size_t i = 52; i < returned.size(); ++i) EXPECT_EQ(returned[i], 0xdeadbeefu) << i;
}
