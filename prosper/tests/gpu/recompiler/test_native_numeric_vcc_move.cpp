// #4967: replacing VCC with current numeric words must replace its old predicate too.
// Execute actual compiler CFG/phis and storage sinks on a strict CPU oracle; no Vulkan device.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
namespace {
using namespace prosper::gpu;
const std::vector<uint32_t> kReassembledOverStaleVcc{
    0x7e060280u, 0x7d8800a8u, 0x87946a7eu, 0xd7610012u, 0x10e15u,    0xd7610012u,
    0x10c14u,    0xbe940380u, 0xbe950380u, 0x7d880088u, 0xd7600002u, 0x10d12u,
    0xd7600003u, 0x10f12u,    0xbeea0402u, 0x7da80090u, 0xbf880001u, 0x7e060287u,
    0xbefe046au, 0x4a0606ffu, 0x64u,       0xbefe04c1u, 0xe0702000u, 0x80020300u};
const std::vector<uint32_t> kTail{0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
                                  0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u};
constexpr uint32_t mov(uint32_t dst, uint32_t src, bool pair = false) {
    return 0xbe800000u | (dst << 16u) | ((pair ? 4u : 3u) << 8u) | src;
}
void constant(std::vector<uint32_t>& p, uint32_t dst, uint32_t value) {
    if (value <= 64u)
        p.push_back(mov(dst, 128u + value));
    else if (value == UINT32_MAX)
        p.push_back(mov(dst, 193));
    else {
        p.push_back(mov(dst, 255));
        p.push_back(value);
    }
}
void finish(std::vector<uint32_t>& p) {
    p.insert(p.end(),
             {mov(126, 106, true), 0x7e060281u, mov(126, 193, true), 0xe0702000u, 0x80020300u});
    p.insert(p.end(), kTail.begin(), kTail.end());
}
std::vector<uint32_t> numeric(uint32_t low, uint32_t high, int missing = -1,
                              bool fabricated = false) {
    std::vector<uint32_t> p{0x7e060280u, 0x7e020280u, 0x7d880088u};
    if (missing != 0) constant(p, 14, low);
    if (missing != 1) {
        if (fabricated)
            p.push_back(mov(15, 60));
        else
            constant(p, 15, high);
    }
    p.push_back(mov(106, 14, true));
    finish(p);
    return p;
}
std::vector<uint32_t> stale() {
    auto p = kReassembledOverStaleVcc;
    p.push_back(0x7e020280u);
    p.insert(p.end(), kTail.begin(), kTail.end());
    return p;
}
ShaderResourceTable table() {
    ShaderResource out;
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.num_components = 1;
    out.binding = 3;
    out.stride = 4;
    out.sgpr_base = 8;
    ShaderResourceTable t;
    t.resources.push_back(out);
    return t;
}
struct Compiled {
    std::vector<uint32_t> module;
    std::string reasons;
};
Compiled compile(const std::vector<uint32_t>& p,
                 const ComputeShaderConfig& c = prosper::test::native_execution_launch()) {
    static uint64_t address = 0x4967000000ull;
    const uint64_t current = address++;
    const auto t = table();
    TerminalRejectCapture capture;
    Compiled r;
    r.module =
        recompile_compute(p.data(), p.size(), &t, c, {RecompileDiagnosticStage::Compute, current});
    for (const auto& [tag, payload] : capture.take()) r.reasons += tag + " " + payload + "\n";
    if (const char* directory = std::getenv("PROSPER_NATIVE_VCC_SPV_DIRECTORY");
        directory && !r.module.empty()) {
        std::filesystem::create_directories(directory);
        std::ofstream f(std::filesystem::path(directory) / (std::to_string(current) + ".spv"),
                        std::ios::binary);
        f.write(reinterpret_cast<const char*>(r.module.data()), r.module.size() * sizeof(uint32_t));
    }
    return r;
}
bool has_switch(const std::vector<uint32_t>& m) {
    for (size_t at = 5; at < m.size();) {
        const auto width = m[at] >> 16u;
        if (!width || width > m.size() - at) return false;
        if ((m[at] & 65535u) == 251u) return true;
        at += width;
    }
    return false;
}
std::vector<uint32_t> expected(uint64_t mask, uint32_t base = 0) {
    std::vector<uint32_t> out;
    for (uint32_t lane = 0; lane < 64; ++lane) out.push_back(base + uint32_t((mask >> lane) & 1u));
    return out;
}
void values(const std::vector<uint32_t>& p, const std::vector<uint32_t>& want,
            const ComputeShaderConfig& c = prosper::test::native_execution_launch()) {
    const auto r = compile(p, c);
    ASSERT_FALSE(r.module.empty()) << r.reasons;
    ASSERT_TRUE(has_switch(r.module));
    bpermute_oracle::Interpreter vm(r.module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto got = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeef));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(got.size(), 64u);
    ASSERT_EQ(want.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane) EXPECT_EQ(got[lane], want[lane]) << "lane " << lane;
}
void refuses(const std::vector<uint32_t>& p, const ComputeShaderConfig& c) {
    ASSERT_FALSE(compile(numeric(UINT32_MAX, 255)).module.empty()) << "matched complete control";
    const auto r = compile(p, c);
    EXPECT_TRUE(r.module.empty());
    EXPECT_FALSE(r.reasons.empty());
}
std::vector<uint32_t> branch_pair(bool complete) {
    std::vector<uint32_t> p{0x7e060280u, 0x7e020280u, 0x7d880088u};
    constant(p, 14, UINT32_MAX);
    if (complete) constant(p, 15, 255);
    p.insert(p.end(), {0xbf068004u, 0xbf840002u});
    constant(p, 15, 255);
    p.push_back(mov(106, 14, true));
    finish(p);
    return p;
}
}   // namespace
TEST(NativeNumericVccMove, SpilledPairReplacesTheStaleEightLanePredicate) {
    std::vector<uint32_t> want;
    for (uint32_t lane = 0; lane < 64; ++lane)
        want.push_back(lane < 16 ? 107u : lane < 40 ? 100u : 0u);
    values(stale(), want);
}
TEST(NativeNumericVccMove, MatchedControlWithoutTheOldCompareKeepsTheOriginalValues) {
    auto p = stale();
    p.erase(std::find(p.begin(), p.end(), 0x7d880088u));
    std::vector<uint32_t> want;
    for (uint32_t lane = 0; lane < 64; ++lane)
        want.push_back(lane < 16 ? 107u : lane < 40 ? 100u : 0u);
    values(p, want);
}
TEST(NativeNumericVccMove, BothCurrentWordsChooseTheirActualLaneBits) {
    for (uint64_t mask : {uint64_t{0}, uint64_t{UINT32_MAX}, UINT64_C(0xffffffff00000000),
                          UINT64_C(0x000000ffffffffff), UINT64_C(0x8000000000000001)})
        values(numeric(uint32_t(mask), uint32_t(mask >> 32)), expected(mask));
}
TEST(NativeNumericVccMove, SelfMoveReadsBothWordsBeforeReplacingThePredicate) {
    auto p = numeric(UINT32_MAX, 255);
    auto at = std::find(p.begin(), p.end(), mov(106, 14, true));
    ASSERT_NE(at, p.end());
    const auto index = size_t(at - p.begin());
    p.insert(p.begin() + index, {mov(106, 14, true), mov(106, 106, true)});
    p.erase(p.begin() + index + 2);
    values(p, expected(UINT64_C(0x000000ffffffffff)));
}
TEST(NativeNumericVccMove, MissingLowOrHighWordCannotReuseTheOldPredicate) {
    for (int missing : {0, 1})
        refuses(numeric(UINT32_MAX, 255, missing), prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, AWordCopiedFromAnAbsentScalarCannotBecomeAMask) {
    refuses(numeric(UINT32_MAX, 255, -1, true), prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, OnePathWordCannotBecomeAMaskOnEitherRuntimeSuccessor) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(5, 0);
        c.user_sgprs[4] = path;
        refuses(branch_pair(false), c);
    }
}
TEST(NativeNumericVccMove, BothPathsCurrentWordsKeepBothRuntimeSuccessors) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(5, 0);
        c.user_sgprs[4] = path;
        values(branch_pair(true), expected(UINT64_C(0x000000ffffffffff)), c);
    }
}
TEST(NativeNumericVccMove, AnOverwrittenInputCannotSupplyTheAbsentSibling) {
    auto c = prosper::test::native_execution_launch();
    c.user_sgprs.resize(16, 0);
    c.user_sgprs[15] = 255;
    auto p = numeric(UINT32_MAX, 255, 1);
    p.insert(p.begin() + 3, mov(14, 60, true));
    refuses(p, c);
}
TEST(NativeNumericVccMove, CurrentReplacementRetiresTheEarlierMissingPair) {
    auto c = prosper::test::native_execution_launch();
    c.user_sgprs.resize(16, 0);
    auto p = numeric(UINT32_MAX, 255);
    p.insert(p.begin() + 3, mov(14, 60, true));
    values(p, expected(UINT64_C(0x000000ffffffffff)), c);
}
TEST(NativeNumericVccMove, TheMovePreservesBothActualSccValues) {
    for (bool truth : {false, true}) {
        auto p = numeric(UINT32_MAX, 255);
        auto at = std::find(p.begin(), p.end(), mov(106, 14, true));
        p.insert(at, truth ? 0xbf068080u : 0xbf068180u);
        auto store = std::find(p.begin(), p.end(), 0xe0702000u);
        p.insert(store, {0xbf840001u, 0x4a060682u});
        values(p, expected(UINT64_C(0x000000ffffffffff), truth ? 2u : 0u));
    }
}
TEST(NativeNumericVccMove, NormalizedGeometryCannotReplaceTheRawLaunch) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch = {};
    refuses(numeric(UINT32_MAX, 255), c);
}
TEST(NativeNumericVccMove, EveryMissingRawLaunchFieldStillRefuses) {
    for (uint32_t bit = 1; bit <= ComputeDispatchLaunchState::kDimZ; bit <<= 1) {
        auto c = prosper::test::native_execution_launch();
        c.dispatch_launch.known &= ~bit;
        refuses(numeric(UINT32_MAX, 255), c);
    }
}
TEST(NativeNumericVccMove, PartialGroupsStillRefuse) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch.initiator |= 2u;
    refuses(numeric(UINT32_MAX, 255), c);
    c = prosper::test::native_execution_launch();
    c.exact_thread_extent = true;
    c.threads_x = 33;
    refuses(numeric(UINT32_MAX, 255), c);
}
TEST(NativeNumericVccMove, ARequiredWidthAloneCannotProjectAWave64Pair) {
    auto c = prosper::test::native_execution_launch();
    c.native_subgroup_size = 32;
    refuses(numeric(UINT32_MAX, 255), c);
}
TEST(NativeNumericVccMove, TheCpuOracleRequiresItsOwnExplicitNativeProfile) {
    const auto r = compile(stale());
    ASSERT_FALSE(r.module.empty());
    bpermute_oracle::Interpreter vm(r.module);
    EXPECT_TRUE(vm.run(64, std::vector<uint32_t>(64, 0xdeadbeef)).empty());
    EXPECT_FALSE(vm.error.empty());
}
