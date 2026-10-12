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
#include <initializer_list>
#include <ios>
#include <string>
#include <vector>
namespace {
using namespace prosper::gpu;
// NOLINTBEGIN(bugprone-throwing-static-initialization): immutable fixture words; an allocation
// failure terminates this hermetic test binary before it can run in either spelling.
const std::vector<uint32_t> kReassembledOverStaleVcc{
    0x7e060280u, 0x7d8800a8u, 0x87946a7eu, 0xd7610012u, 0x10e15u,    0xd7610012u,
    0x10c14u,    0xbe940380u, 0xbe950380u, 0x7d880088u, 0xd7600002u, 0x10d12u,
    0xd7600003u, 0x10f12u,    0xbeea0402u, 0x7da80090u, 0xbf880001u, 0x7e060287u,
    0xbefe046au, 0x4a0606ffu, 0x64u,       0xbefe04c1u, 0xe0702000u, 0x80020300u};
const std::vector<uint32_t> kTail{0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u,
                                  0xbf870001u, 0xbf82fffdu, 0x7e040d02u, 0xbf810000u};
// NOLINTEND(bugprone-throwing-static-initialization)
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
                 const ComputeShaderConfig& c = prosper::test::native_execution_launch(),
                 bool direct_descriptors = false) {
    static uint64_t address = 0x4967000000ull;
    const uint64_t current = address++;
    auto t = table();
    if (direct_descriptors) {
        t.resources[0].srt_offset = UINT32_MAX;
        auto second = t.resources[0];
        second.binding = 4;
        second.sgpr_base = 12;
        t.resources.push_back(second);
    }
    TerminalRejectCapture capture;
    Compiled r;
    r.module =
        recompile_compute(p.data(), p.size(), &t, c, {RecompileDiagnosticStage::Compute, current});
    for (const auto& [tag, payload] : capture.take())
        r.reasons.append(tag).append(" ").append(payload).append("\n");
    // NOLINTNEXTLINE(concurrency-mt-unsafe): fixed process environment, no concurrent mutation.
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
    out.reserve(64);
    for (uint32_t lane = 0; lane < 64; ++lane) out.push_back(base + uint32_t((mask >> lane) & 1u));
    return out;
}
void values(const std::vector<uint32_t>& p, const std::vector<uint32_t>& want,
            const ComputeShaderConfig& c = prosper::test::native_execution_launch(),
            bool direct_descriptors = false) {
    const auto r = compile(p, c, direct_descriptors);
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
    want.reserve(64);
    for (uint32_t lane = 0; lane < 64; ++lane)
        want.push_back(lane < 16 ? 107u : lane < 40 ? 100u : 0u);
    values(stale(), want);
}
TEST(NativeNumericVccMove, MatchedControlWithoutTheOldCompareKeepsTheOriginalValues) {
    auto p = stale();
    p.erase(std::find(p.begin(), p.end(), 0x7d880088u));
    std::vector<uint32_t> want;
    want.reserve(64);
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

namespace {
std::vector<uint32_t> lane_local_pair(bool add = false, bool move = false) {
    auto p = numeric(1, 0);
    auto low = std::find(p.begin(), p.end(), mov(14, 129));
    p.erase(low);
    p.insert(p.begin() + 3, add ? std::initializer_list<uint32_t>{0x7e1c0500u, 0x800e810eu}
                                : std::initializer_list<uint32_t>{0x4a040081u, 0x7e1c0502u});
    if (move) {
        auto use = std::find(p.begin(), p.end(), mov(106, 14, true));
        p.insert(use, {mov(16, 14), mov(17, 128)});
        *std::find(p.begin(), p.end(), mov(106, 14, true)) = mov(106, 16, true);
    }
    return p;
}
std::vector<uint32_t> slot_pair(bool genuine, bool replace = false) {
    auto p = lane_local_pair();
    auto use = std::find(p.begin(), p.end(), mov(106, 14, true));
    const auto index = use - p.begin();
    p.erase(use);
    std::vector<uint32_t> relay;
    if (genuine) relay.push_back(mov(14, 129));
    relay.insert(relay.end(), {0xd7610012u, 0x0001000eu, 0xd7600010u, 0x00010112u, mov(17, 128)});
    if (replace) relay.push_back(mov(16, 129));
    relay.push_back(mov(106, 16, true));
    p.insert(p.begin() + index, relay.begin(), relay.end());
    return p;
}
std::vector<uint32_t> joined_pair(bool both_genuine, bool replace = false) {
    std::vector<uint32_t> p{
        0x7e060280u,  0x7e020280u, 0x7d880088u, 0x4a040081u,
        mov(15, 128), 0xbf068004u, 0xbf840002u, both_genuine ? mov(14, 129) : 0x7e1c0502u,
        0xbf820001u,  mov(14, 129)};
    if (replace) p.push_back(mov(14, 129));
    p.push_back(mov(106, 14, true));
    finish(p);
    return p;
}
}   // namespace
TEST(NativeNumericVccMove, LaneLocalReadfirstCannotSupplyNewMaskAuthority) {
    refuses(lane_local_pair(), prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, ScalarAddCannotLaunderTheLaneLocalWord) {
    refuses(lane_local_pair(true), prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, ScalarMoveCannotLaunderTheLaneLocalWord) {
    refuses(lane_local_pair(false, true), prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, NumericSpillReadlaneKeepsItsStoredSourceUniformity) {
    refuses(slot_pair(false), prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, GenuineNumericSpillSourceKeepsTheOneLaneValue) {
    values(slot_pair(true), expected(1));
}
TEST(NativeNumericVccMove, CurrentScalarOverwriteRetiresTheUnprovedSpillWord) {
    values(slot_pair(false, true), expected(1));
}
TEST(NativeNumericVccMove, AJoinedLaneLocalWordRefusesBothRuntimeSuccessors) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(5, 0);
        c.user_sgprs[4] = path;
        refuses(joined_pair(false), c);
    }
}
TEST(NativeNumericVccMove, BothCurrentScalarArmsKeepTheOneLaneValue) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(5, 0);
        c.user_sgprs[4] = path;
        values(joined_pair(true), expected(1), c);
    }
}
TEST(NativeNumericVccMove, ACurrentOverwriteRetiresTheJoinedUnprovedWord) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(5, 0);
        c.user_sgprs[4] = path;
        values(joined_pair(false, true), expected(1), c);
    }
}
TEST(NativeNumericVccMove, InternedConstantReplacementStartsANewScalarGeneration) {
    auto p = numeric(1, 0);
    p.insert(p.begin() + 3, {0x7e040281u, 0x7e1c0502u});
    values(p, expected(1));
}
TEST(NativeNumericVccMove, AnUnrelatedReadfirstCannotPoisonTheGenuineZeroHighWord) {
    auto p = numeric(1, 0);
    p.insert(p.begin() + 3, {0x7e040280u, 0x7e780502u});
    values(p, expected(1));
}
TEST(NativeNumericVccMove, DispatcherReloadCannotReseedTheLaneLocalWord) {
    auto p = lane_local_pair();
    auto use = std::find(p.begin(), p.end(), mov(106, 14, true));
    p.insert(use, 0xbf820000u);
    refuses(p, prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, CurrentScalarReplacementAfterReloadKeepsTheOneLaneValue) {
    auto p = lane_local_pair();
    auto use = std::find(p.begin(), p.end(), mov(106, 14, true));
    p.insert(use, {0xbf820000u, mov(14, 129)});
    values(p, expected(1));
}

namespace {
std::vector<uint32_t> joined_slot(bool both_genuine) {
    auto p = slot_pair(false);
    auto read = std::find(p.begin(), p.end(), 0xd7600010u);
    p.insert(read, {0xbf068004u, 0xbf840004u, mov(14, 129), 0xd7610012u, 0x0001000eu, 0xbf820003u,
                    both_genuine ? mov(14, 129) : 0x7e1c0502u, 0xd7610012u, 0x0001000eu});
    return p;
}
}   // namespace
TEST(NativeNumericVccMove, AScalarOverwriteCannotChangeTheOlderNumericSlot) {
    auto p = slot_pair(false);
    auto read = std::find(p.begin(), p.end(), 0xd7600010u);
    p.insert(read, mov(14, 129));
    refuses(p, prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, ACurrentSlotOverwriteRetiresItsUnprovedSource) {
    auto p = slot_pair(false);
    auto read = std::find(p.begin(), p.end(), 0xd7600010u);
    p.insert(read, {mov(14, 129), 0xd7610012u, 0x0001000eu});
    values(p, expected(1));
}
TEST(NativeNumericVccMove, OnePathSlotReplacementRefusesBothRuntimeSuccessors) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(5, 0);
        c.user_sgprs[4] = path;
        refuses(joined_slot(false), c);
    }
}
TEST(NativeNumericVccMove, BothPathsSlotReplacementKeepTheOneLaneValue) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(5, 0);
        c.user_sgprs[4] = path;
        values(joined_slot(true), expected(1), c);
    }
}
TEST(NativeNumericVccMove, APartialVectorOverwriteCannotReseedTheOlderNumericSlot) {
    auto p = slot_pair(false);
    auto read = std::find(p.begin(), p.end(), 0xd7600010u);
    p.insert(read, {mov(126, 106, true), 0x7e240281u, mov(126, 193, true)});
    refuses(p, prosper::test::native_execution_launch());
}

namespace {
std::vector<uint32_t> preserving_scc(uint32_t producer, bool genuine = false,
                                     bool replace = false) {
    std::vector<uint32_t> p{
        0x7e060280u, 0x7e020280u, 0x7d880088u, genuine ? mov(16, 128) : 0x7e200500u,
        0xbf068010u, producer,    0x850e8281u, mov(15, 128)};
    if (replace) p.push_back(mov(14, 129));
    p.push_back(mov(106, 14, true));
    finish(p);
    return p;
}
}   // namespace
TEST(NativeNumericVccMove, SccPreservingDataWritersCannotLaunderTheOldCondition) {
    for (uint32_t producer : {0x930e8181u, 0x920e8081u, 0x928e8081u})
        refuses(preserving_scc(producer), prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, TheGenuineConditionSurvivesSccPreservingDataWriters) {
    for (uint32_t producer : {0x930e8181u, 0x920e8081u, 0x928e8081u})
        values(preserving_scc(producer, true), expected(1));
}
TEST(NativeNumericVccMove, AFullReplacementRetiresTheUnprovedConditionalWord) {
    for (uint32_t producer : {0x930e8181u, 0x920e8081u, 0x928e8081u})
        values(preserving_scc(producer, false, true), expected(1));
}

namespace {
std::vector<uint32_t> numeric_loop(const std::vector<uint32_t>& rewrite) {
    std::vector<uint32_t> p{0x7e060280u,  0x7e0a0280u,  0x7e000505u,  0x7d880088u, mov(14, 129),
                            mov(15, 128), mov(22, 128), mov(23, 131), 0xbf0a1716u};
    const uint32_t body = 3 + static_cast<uint32_t>(rewrite.size()) + 2;
    p.push_back(0xbf840000u | body);
    p.insert(p.end(), {mov(106, 14, true), 0xd5010003u, 0x01a9e480u});
    p.insert(p.end(), rewrite.begin(), rewrite.end());
    p.push_back(0x80168116u);
    p.push_back(0xbf820000u | ((0x10000u - (body + 2)) & 0xffffu));
    p.insert(p.end(), {0xe0702000u, 0x80020300u, 0xbf810000u});
    return p;
}
void loop_values(const std::vector<uint32_t>& p, uint64_t mask) {
    const auto r = compile(p);
    ASSERT_FALSE(r.module.empty()) << r.reasons;
    bpermute_oracle::Interpreter vm(r.module);
    vm.native_subgroup_width = 64;
    const auto got = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeef));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(got.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(got[lane], (mask & (uint64_t{1} << lane)) ? 0x3f800000u : 0u) << lane;
}
}   // namespace
TEST(NativeNumericVccMove, APlainFullyInitializedConstantReadfirstHasAUniformValue) {
    auto p = numeric(1, 0);
    const auto low = std::find(p.begin(), p.end(), mov(14, 129));
    p.erase(low);
    const auto high = std::find(p.begin(), p.end(), mov(15, 128));
    p.insert(high, {0x7e040281u, 0x7e1c0502u});
    values(p, expected(1));
}
TEST(NativeNumericVccMove, ACheckedLoopProjectsItsCurrentPairInsteadOfTheOldPredicate) {
    loop_values(numeric_loop({mov(14, 130), mov(15, 128)}), 2);
}
TEST(NativeNumericVccMove, ALoopBlanketCannotAdmitANonuniformCurrentBackedgeWord) {
    refuses(numeric_loop({0x4a040081u, 0x7e1c0502u, mov(15, 128)}),
            prosper::test::native_execution_launch());
}
TEST(NativeNumericVccMove, ACurrentUniformOverwriteRetiresTheNonuniformLoopWord) {
    loop_values(numeric_loop({0x4a040081u, 0x7e1c0502u, mov(14, 130), mov(15, 128)}), 2);
}
TEST(NativeNumericVccMove, AUniformAbsentBackedgeStillViolatesTheReadRoot) {
    const auto r = compile(numeric_loop({mov(14, 60, true)}));
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reasons.find("fabricated on the back edge"), std::string::npos) << r.reasons;
}

TEST(NativeNumericVccMove, ACurrentDriverDescriptorPairReplacesTheOldPredicate) {
    auto c = prosper::test::native_execution_launch();
    c.user_sgprs.resize(16, 0);
    c.user_sgprs[14] = 1;
    std::vector<uint32_t> p{0x7e060280u, 0x7e020280u, 0x7d880088u, mov(106, 14, true)};
    finish(p);
    values(p, expected(1), c, true);
    ASSERT_FALSE(compile(p, c, true).module.empty());
    c.dispatch_launch = {};
    EXPECT_TRUE(compile(p, c, true).module.empty());
    c = prosper::test::native_execution_launch();
    c.user_sgprs.resize(16, 0);
    c.user_sgprs[14] = 1;
    c.dispatch_launch.initiator |= 2u;
    EXPECT_TRUE(compile(p, c, true).module.empty());
}
TEST(NativeNumericVccMove, AOnePathDescriptorWriteKeepsTheCurrentWordOnBothEdges) {
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(16, 0);
        c.user_sgprs[4] = path;
        c.user_sgprs[15] = 1;
        const uint64_t high = path ? 1u : 255u;
        values(branch_pair(false), expected(UINT32_MAX | (high << 32u)), c, true);
    }
}

TEST(NativeNumericVccMove, AOnePathDestroyedDescriptorWordCannotBorrowItsOldInput) {
    auto p = branch_pair(false);
    const auto high = std::find(p.begin(), p.end(), mov(15, 255));
    ASSERT_NE(high, p.end());
    p.erase(high + 1);
    *high = mov(15, 60);
    *std::find(p.begin(), p.end(), 0xbf840002u) = 0xbf840001u;
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(16, 0);
        c.user_sgprs[4] = path;
        c.user_sgprs[15] = 1;
        ASSERT_FALSE(compile(branch_pair(false), c, true).module.empty());
        const auto r = compile(p, c, true);
        EXPECT_TRUE(r.module.empty());
        EXPECT_FALSE(r.reasons.empty());
    }
}

TEST(NativeNumericVccMove, TheDispatcherRetainsItsUntouchedDriverWordAsOrdinaryData) {
    auto p = branch_pair(false);
    const auto copy = std::find(p.begin(), p.end(), mov(106, 14, true));
    ASSERT_NE(copy, p.end());
    p.erase(copy + 1, p.end());
    p.insert(p.end(), {0x7e06026bu, 0xe0702000u, 0x80020300u});
    p.insert(p.end(), kTail.begin(), kTail.end());
    for (uint32_t path : {0u, 1u}) {
        auto c = prosper::test::native_execution_launch();
        c.user_sgprs.resize(16, 0);
        c.user_sgprs[4] = path;
        c.user_sgprs[15] = 1;
        values(p, std::vector<uint32_t>(64, path ? 1u : 255u), c, true);
    }
}
