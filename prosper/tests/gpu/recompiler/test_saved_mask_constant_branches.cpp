// #4969: optional SCC folding must not destroy an otherwise proved saved EXEC dword.
// Check the actual DATA sink, including asymmetric halves and an older mask restored to full
// EXEC before the reload; a late ballot or a fabricated zero would corrupt that sink.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr std::array<uint32_t, 15> native_data{0x7e060280u, 0xbe9e047eu, 0xd7610014u, 0x0001381eu,
                                               0xbf068004u, 0xbf840001u, 0x7e040281u, 0xd760000eu,
                                               0x00013914u, 0xbf068005u, 0xbf840001u, 0x7e040282u,
                                               0x4a06000eu, 0xe0702000u, 0x80020300u};
constexpr std::array<uint32_t, 15> data_then_mask{
    0x7e060280u, 0xbe9e047eu, 0xbe8c0387u, 0xd7610014u, 0x0001380cu,
    0xd7610014u, 0x0001381eu, 0xbf068004u, 0xbf840001u, 0x7e040281u,
    0xd760000eu, 0x00013914u, 0x4a06000eu, 0xe0702000u, 0x80020300u};
constexpr std::array<uint32_t, 9> tail{0x7e040280u, 0x7c020300u, 0xbf860001u,
                                       0x7e040281u, 0x7d840100u, 0xbf870001u,
                                       0xbf82fffdu, 0x7e040d02u, 0xbf810000u};
struct Options {
    bool second = false, immediate = true, direct = false, high = false, narrowed_save = false;
};
std::vector<uint32_t> program(Options o, uint32_t a = 0, uint32_t b = 0) {
    std::vector<uint32_t> p;
    if (o.immediate) {
        p.push_back(0xbe840380u | a); // s4=0/1, before the original branch
        if (!o.second) p.push_back(0xbe850380u | b);
    }
    const auto& prefix = o.second ? data_then_mask : native_data;
    p.insert(p.end(), prefix.begin(), prefix.end());
    if (o.narrowed_save) {
        const auto save = std::find(p.begin(), p.end(), 0xbe9e047eu);
        p.insert(save, {0x7d8800a8u, 0xbefe046au}); // EXEC lanes 0..39 before saving s[30:31]
        const auto spill = std::find(p.begin(), p.end(), 0x0001381eu);
        p.insert(spill + 1, 0xbefe04c1u); // restore current EXEC; the spilled word stays older
    }
    auto source = std::find(p.begin(), p.end(), 0x0001381eu);
    if (o.direct)
        *source = 0x0001387eu; // same stream, only direct EXEC instead of saved s30
    else if (o.high)
        *source = 0x0001381fu;
    p.push_back(0x7e020280u); // genuine v1 before the original irreducible tail
    p.insert(p.end(), tail.begin(), tail.end());
    return p;
}
ShaderResourceTable output_table() {
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
bool has_op(const std::vector<uint32_t>& module, uint32_t op) {
    for (size_t at = 5; at < module.size();) {
        const size_t width = module[at] >> 16;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == op) return true;
        at += width;
    }
    return false;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& p, const ComputeShaderConfig& c,
                              uint64_t address = 0) {
    const auto t = output_table();
    return recompile_compute(p.data(), p.size(), &t, c,
                             {RecompileDiagnosticStage::Compute, address});
}
void values(const std::vector<uint32_t>& p, const ComputeShaderConfig& c, uint32_t word) {
    TerminalRejectCapture capture;
    const auto module = compile(p, c);
    std::string reason;
    for (const auto& [tag, payload] : capture.take()) {
        reason += tag;
        reason += ' ';
        reason += payload;
        reason += '\n';
    }
    ASSERT_FALSE(module.empty()) << reason;
    ASSERT_TRUE(has_op(module, 251)) << "must use the public CFG dispatcher";
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = c.user_sgprs;
    const auto sink = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeefu)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(sink.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(sink[lane], lane + word) << "lane " << lane;
}
void all_paths(Options o) {
    for (bool second : {false, true})
        for (uint32_t a : {0u, 1u})
            for (uint32_t b : {0u, 1u}) {
                SCOPED_TRACE(::testing::Message()
                             << "second=" << second << " a=" << a << " b=" << b);
                o.second = second;
                auto c = prosper::test::native_execution_launch();
                if (!o.immediate) {
                    c.user_sgprs.resize(6, 0);
                    c.user_sgprs[4] = a;
                    c.user_sgprs[5] = b;
                }
                values(program(o, a, b), c, o.narrowed_save && o.high ? 255u : UINT32_MAX);
            }
}
void physical_words_refuse(const ComputeShaderConfig& c, uint64_t address) {
    const auto p = program({});
    ASSERT_FALSE(compile(p, prosper::test::native_execution_launch(), address + 1).empty());
    TerminalRejectCapture capture;
    EXPECT_TRUE(compile(p, c, address).empty());
    const auto reason = last_terminal_reject_reason(address);
    EXPECT_NE(reason.find("cfg-recompile-reject"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=9 words=d760000e,00013914"), std::string::npos) << reason;
    EXPECT_NE(reason.find("mode=unresolved-operand"), std::string::npos) << reason;
}
} // namespace

TEST(SavedMaskConstantBranches, ImmediateLowWordSurvivesEveryBranchOutcome) {
    all_paths({});
}
TEST(SavedMaskConstantBranches, ImmediateHighWordSurvivesEveryBranchOutcome) {
    all_paths({.high = true});
}
TEST(SavedMaskConstantBranches, GenuineDynamicEntryWordsKeepTheirOriginalProof) {
    all_paths({.immediate = false});
}
TEST(SavedMaskConstantBranches, DirectExecControlKeepsItsSuccessfulOptimizedModule) {
    all_paths({.direct = true});
}
TEST(SavedMaskConstantBranches, RestoredExecCannotReplaceAnOlderAsymmetricLowWord) {
    all_paths({.narrowed_save = true});
}
TEST(SavedMaskConstantBranches, RestoredExecCannotReplaceAnOlderAsymmetricHighWord) {
    all_paths({.high = true, .narrowed_save = true});
}
TEST(SavedMaskConstantBranches, OverwritingTheSavedSourceDoesNotChangeTheSpilledGeneration) {
    auto p = program({});
    const auto spill = std::find(p.begin(), p.end(), 0x0001381eu);
    p.insert(spill + 1, {0xbe9e0380u, 0xbe9f0380u});   // overwrite both source words after capture
    values(p, prosper::test::native_execution_launch(), UINT32_MAX);
}
TEST(SavedMaskConstantBranches, GenuineSlotReplacementPublishesItsNewNumericValue) {
    auto p = program({});
    const auto spill = std::find(p.begin(), p.end(), 0x0001381eu);
    p.insert(spill + 1, {0xd7610014u, 0x00013887u});   // overwrite the slot with ordinary seven
    values(p, prosper::test::native_execution_launch(), 7u);
}
TEST(SavedMaskConstantBranches, RetryDoesNotPublishTheDiscardedReloadRefusal) {
    constexpr uint64_t address = 0x49690100ull;
    TerminalRejectCapture capture;
    ASSERT_FALSE(compile(program({}), prosper::test::native_execution_launch(), address).empty());
    for (const auto& [tag, payload] : capture.take()) {
        EXPECT_NE(tag, "cfg-recompile-reject") << payload;
        EXPECT_EQ(payload.find("mode=unresolved-operand"), std::string::npos) << payload;
    }
    EXPECT_EQ(last_terminal_reject_reason(address).find("mode=unresolved-operand"),
              std::string::npos);
}
TEST(SavedMaskConstantBranches, GeometryWithoutRawLaunchCannotBorrowTheRetry) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch = {};
    physical_words_refuse(c, 0x49690200ull);
}
TEST(SavedMaskConstantBranches, EveryRawLaunchFieldIsStillRequired) {
    uint64_t address = 0x49690300ull;
    for (uint32_t bit = 1; bit <= ComputeDispatchLaunchState::kDimZ; bit <<= 1) {
        auto c = prosper::test::native_execution_launch();
        c.dispatch_launch.known &= ~bit;
        physical_words_refuse(c, address);
        address += 0x100;
    }
}
TEST(SavedMaskConstantBranches, PartialInitiatorCannotSupplyPhysicalWords) {
    auto c = prosper::test::native_execution_launch();
    c.dispatch_launch.initiator |= 2u;
    physical_words_refuse(c, 0x49691000ull);
}
TEST(SavedMaskConstantBranches, PartialExactExtentCannotSupplyPhysicalWords) {
    auto c = prosper::test::native_execution_launch();
    c.exact_thread_extent = true;
    c.threads_x = 33;
    physical_words_refuse(c, 0x49691100ull);
}
TEST(SavedMaskConstantBranches, OneArmSpillCannotAuthenticateAScalarBranch) {
    auto p = program({.immediate = false});
    const auto spill = std::find(p.begin(), p.end(), 0xd7610014u);
    p.insert(spill, {0xbf068004u, 0xbf840002u});   // s4=1 can bypass the only definition
    const auto reload = std::find(p.begin(), p.end(), 0x00013914u);
    p.insert(reload + 1, {0xbf06800eu, 0xbf840001u, 0x7e040282u});
    auto c = prosper::test::native_execution_launch();
    c.user_sgprs.resize(6, 0);
    constexpr uint64_t address = 0x49691200ull;
    for (uint32_t path : {0u, 1u}) {
        c.user_sgprs[4] = path;
        EXPECT_TRUE(compile(p, c, address + path).empty());
        const auto reason = last_terminal_reject_reason(address + path);
        EXPECT_NE(reason.find("poisoned-scc"), std::string::npos) << reason;
        EXPECT_NE(reason.find("pc=12 "), std::string::npos) << reason;
    }
    auto control = p;
    const auto guard = std::find(control.begin(), control.end(), 0xbf068004u);
    control.insert(guard, {0xd7610014u, 0x0001381eu});   // genuine definition on the skipped path
    for (uint32_t path : {0u, 1u}) {
        c.user_sgprs[4] = path;
        values(control, c, UINT32_MAX);
    }
}
TEST(SavedMaskConstantBranches, FailedOriginalRetryPublishesOnlyItsOwnBranchCause) {
    auto p = program({});
    const auto spill = std::find(p.begin(), p.end(), 0xd7610014u);
    p.insert(spill, {0xbf068006u, 0xbf840002u});   // dynamic s6 can bypass the only spill
    const auto reload = std::find(p.begin(), p.end(), 0x00013914u);
    p.insert(reload + 1, {0xbf06800eu, 0xbf840001u, 0x7e040282u});
    auto control = p;
    const auto guard = std::find(control.begin(), control.end(), 0xbf068006u);
    control.insert(guard, {0xd7610014u, 0x0001381eu});
    auto c = prosper::test::native_execution_launch();
    c.user_sgprs.resize(7, 0);
    for (uint32_t path : {0u, 1u}) {
        SCOPED_TRACE(path);
        c.user_sgprs[6] = path;
        values(control, c, UINT32_MAX);
        TerminalRejectCapture capture;
        EXPECT_TRUE(compile(p, c, 0x49691300ull + path).empty());
        unsigned branch_causes = 0;
        for (const auto& [tag, payload] : capture.take()) {
            EXPECT_NE(tag, "cfg-recompile-reject") << payload;
            if (tag != "compute-cfg-reject") continue;
            ++branch_causes;
            EXPECT_NE(payload.find("pc=14 reason=poisoned-scc"), std::string::npos) << payload;
        }
        EXPECT_EQ(branch_causes, 1u) << "the discarded optimized attempt must remain private";
    }
}
TEST(SavedMaskConstantBranches, ConstantDeadResourceArmRemainsOptimized) {
    const std::vector<uint32_t> p{0xbe8b03a5u, 0x876a870bu, 0x816ac46au, 0xbf0b826au,
                                  0xbf850002u, 0xe0002000u, 0x80030100u, 0xbf810000u};
    auto c = prosper::test::native_execution_launch();
    ASSERT_FALSE(recompile_compute(p.data(), p.size(), nullptr, c).empty())
        << "the constant branch skips an unbound resource; a blanket original-stream retry fails";
    auto dynamic = p;
    dynamic.erase(dynamic.begin());   // genuinely unknown s11, both resource paths reachable
    c.user_sgprs.resize(12, 0);   // genuine s11=0 takes the unbound resource path at runtime
    constexpr uint64_t address = 0x49691400ull;
    EXPECT_TRUE(recompile_compute(dynamic.data(), dynamic.size(), nullptr, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("unresolved-operand"), std::string::npos);
    EXPECT_NE(last_terminal_reject_reason(address).find("words=e0002000,80030100"),
              std::string::npos);
}
