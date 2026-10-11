// #4943: a retained physical mask dword must survive a genuine public CFG edge as DATA.
// A false Bool slot disables live lanes; promoting one dword to a B64 alias invents its sibling.
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
constexpr uint32_t end = 0xbf810000u;
constexpr uint32_t mov(uint32_t dst, uint32_t src, bool pair = false) {
    return 0xbe800000u | (dst << 16u) | ((pair ? 4u : 3u) << 8u) | src;
}
struct Options {
    bool data = false, high = false, both = false, overwrite = false, numeric_read = false,
         vcc = false, relay = false;
};
std::vector<uint32_t> program(Options o = {}) {
    std::vector<uint32_t> p{0x7e020280u}; // v1=0 before masked replacement
    if (o.data)
        p.insert(p.end(), {mov(30, 193), mov(31, 193)});
    else {
        if (o.vcc) p.push_back(0x7d8800a8u); // v_cmp_gt_u32 vcc,40,v0: LO=~0, HI=0xff
        p.push_back(mov(30, o.vcc ? 106 : 126, true));
    }
    const uint32_t source = o.high ? 31 : 30;
    p.insert(p.end(), {0xd7610014u, source | (128u << 9u)});
    if (o.both) p.insert(p.end(), {0xd7610014u, 31u | (129u << 9u)});
    if (o.overwrite) p.insert(p.end(), {0xd7610014u, 128u | (128u << 9u)});
    p.insert(p.end(), {0xbf068004u, 0xbf840001u, 0x7e040281u}); // genuine s4 comparison/edge
    p.insert(p.end(), {0xd7600000u | (o.high ? 15u : 14u), 276u | (128u << 9u)});
    if (o.both) p.insert(p.end(), {0xd760000fu, 276u | (129u << 9u)});
    if (o.relay)
        p.insert(p.end(), {0xd7610015u, 14u | (128u << 9u), 0xbf068004u, 0xbf840001u, 0x7e040281u,
                           0xd760000eu, 277u | (128u << 9u)});
    if (o.numeric_read)
        p.push_back(0x7e020200u | (o.high ? 15u : 14u));
    else
        p.insert(p.end(), {mov(126, 14, true), 0x7e020281u, mov(126, 193, true)});
    p.insert(p.end(), {0xe0702000u, 0x80020100u}); // owned binding3 output[v0]=v1
    // Existing project-owned irreducible CFG shape: the public structurizer must fall back.
    p.insert(p.end(), {0x7e040280u, 0x7c020300u, 0xbf860001u, 0x7e040281u, 0x7d840100u, 0xbf870001u,
                       0xbf82fffdu, 0x7e040d02u, end});
    return p;
}
ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.local_x = c.wave_size = c.native_subgroup_size = 64;
    c.user_sgprs.resize(16, 0); // genuine s4, s14 and s15; descriptor words owned by the table
    prosper::test::complete_compute_launch(c);
    return c;
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
std::vector<uint32_t> compile(Options o = {}, const ComputeShaderConfig& c = config()) {
    const auto p = program(o);
    const auto t = table();
    return recompile_compute(p.data(), p.size(), &t, c);
}
bool has_op(const std::vector<uint32_t>& module, uint32_t op) {
    for (size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16u;
        if (!width || width > module.size() - at) return false;
        if ((module[at] & 65535u) == op) return true;
        at += width;
    }
    return false;
}
void values(Options o, uint64_t mask, const ComputeShaderConfig& c = config()) {
    const auto module = compile(o, c);
    ASSERT_FALSE(module.empty());
    ASSERT_TRUE(has_op(module, 251)) << "must exercise the public CFG dispatcher";
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64; // explicit CPU oracle input, not inferred entry authority
    vm.push_constants = c.user_sgprs;
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(out[lane], o.numeric_read ? UINT32_MAX : uint32_t((mask >> lane) & 1u))
            << "lane " << lane;
}
void refuses(const ComputeShaderConfig& c) {
    TerminalRejectCapture capture;
    EXPECT_TRUE(compile({.numeric_read = true}, c).empty());
    std::string reason;
    for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
    EXPECT_NE(reason.find("cfg-recompile-reject"), std::string::npos) << reason;
    EXPECT_NE(reason.find("pc=7 words=d760000e"), std::string::npos) << reason;
}
enum class SccUse { Immediate, AcrossEdge, ScalarRelay, ReplacedSlot };
std::vector<uint32_t> slot_scc_program(SccUse use, bool all_paths) {
    auto p = program({.numeric_read = true});
    const auto slot = std::find(p.begin(), p.end(), 0xd7610014u);
    // s4=1 skips the only spill, so the joined DATA slot is not a MUST-defined word.
    p.insert(slot, {0xbf068004u, 0xbf840002u});
    if (all_paths) p.insert(p.begin() + 2, {0xd7610014u, 30u | (128u << 9u)});
    if (use == SccUse::ReplacedSlot) {
        const auto reload = std::find(p.begin(), p.end(), 0xd760000eu);
        p.insert(reload, {0xd7610014u, 193u | (128u << 9u)});
    }
    auto at = std::find(p.begin(), p.end(), 0x7e02020eu);
    std::vector<uint32_t> condition;
    if (use == SccUse::ScalarRelay) condition.push_back(mov(16, 14));
    condition.push_back(use == SccUse::ScalarRelay ? 0xbf068010u : 0xbf06800eu);
    // EXEC branches end native fused chains without rewriting the scalar condition.
    if (use == SccUse::AcrossEdge) condition.insert(condition.end(), {0xbf880001u, 0x7e040281u});
    condition.insert(condition.end(), {0xbf840001u, 0x7e040281u});
    p.insert(at, condition.begin(), condition.end());
    return p;
}
void all_ones(const std::vector<uint32_t>& p) {
    const auto t = table();
    for (uint32_t path : {0u, 1u}) {
        auto c = config();
        c.user_sgprs[4] = path;
        TerminalRejectCapture capture;
        const auto module = recompile_compute(p.data(), p.size(), &t, c);
        std::string reason;
        for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
        ASSERT_FALSE(module.empty()) << "path " << path << ' ' << reason;
        ASSERT_TRUE(has_op(module, 251));
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.push_constants = c.user_sgprs;
        const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane)
            EXPECT_EQ(out[lane], UINT32_MAX) << "path " << path << " lane " << lane;
    }
}
void slot_scc_values(SccUse use, bool all_paths) {
    all_ones(slot_scc_program(use, all_paths));
}
std::vector<uint32_t> scalar_scc_program(bool defined) {
    auto p = program({.data = true, .numeric_read = true});
    const auto at = std::find(p.begin(), p.end(), 0x7e02020eu);
    std::vector<uint32_t> condition;
    if (defined) condition.push_back(mov(60, 128)); // current B32 writer, outside launch SGPRs
    condition.insert(condition.end(), {0xbf06803cu, 0xbf840001u, 0x7e040281u});
    p.insert(at, condition.begin(), condition.end());
    return p;
}
void slot_scc_refuses(SccUse use) {
    const auto p = slot_scc_program(use, false);
    const auto t = table();
    TerminalRejectCapture capture;
    EXPECT_TRUE(recompile_compute(p.data(), p.size(), &t, config()).empty());
    std::string reason;
    for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
    EXPECT_NE(reason.find("poisoned-scc"), std::string::npos) << reason;
}
} // namespace

TEST(NativeCfgSavedMaskData, LowSpillKeepsDefinedZeroHighWord) {
    values({}, UINT32_MAX);
    auto c = config();
    c.user_sgprs[4] = 1; // take the other genuine SCC successor
    values({}, UINT32_MAX, c);
}
TEST(NativeCfgSavedMaskData, HighSpillKeepsDefinedZeroLowWord) {
    values({.high = true}, uint64_t(UINT32_MAX) << 32u);
}
TEST(NativeCfgSavedMaskData, BothPhysicalSpillsRestoreTheCompletePair) {
    values({.both = true}, UINT64_MAX);
}
TEST(NativeCfgSavedMaskData, UnequalVccHighWordKeepsItsPhysicalIndex) {
    values({.high = true, .vcc = true}, uint64_t(0xff) << 32u);
}
TEST(NativeCfgSavedMaskData, UnequalVccWordsRestoreTheirCompletePair) {
    values({.both = true, .vcc = true}, (uint64_t(0xff) << 32u) | UINT32_MAX);
}
TEST(NativeCfgSavedMaskData, TypedDataReloadCanBeSpilledAcrossAnotherEdge) {
    values({.relay = true}, UINT32_MAX);
}
TEST(NativeCfgSavedMaskData, GenuineScalarDataControlUsesTheSameDispatcher) {
    values({.data = true}, UINT32_MAX);
}
TEST(NativeCfgSavedMaskData, RetainedPhysicalWordIsAlsoCurrentNumericData) {
    values({.numeric_read = true}, 0);
}
TEST(NativeCfgSavedMaskData, LatestSlotWriteReplacesTheRetainedMaskDword) {
    values({.overwrite = true}, 0);
}
TEST(NativeCfgSavedMaskData, CurrentIndependentSiblingDeterminesItsOwnLaneBits) {
    auto c = config();
    c.user_sgprs[15] = 0x80000001u;
    values({}, (uint64_t(c.user_sgprs[15]) << 32u) | UINT32_MAX, c);
}
TEST(NativeCfgSavedMaskData, MissingSiblingCannotBecomeHistoricalCompleteBool) {
    auto c = config();
    c.user_sgprs.resize(15);
    EXPECT_TRUE(compile({}, c).empty());
}
TEST(NativeCfgSavedMaskData, UnknownRawLaunchCannotMaterializeMaskData) {
    auto c = config();
    c.dispatch_launch = {};
    refuses(c);
}
TEST(NativeCfgSavedMaskData, PartialRawLaunchCannotMaterializeMaskData) {
    auto c = config();
    c.dispatch_launch.initiator |= 2u;
    refuses(c);
}
TEST(NativeCfgSavedMaskData, MismatchedGeometryCannotBorrowProducerAuthority) {
    auto c = config();
    c.threads_x += 64;
    refuses(c);
}
TEST(NativeCfgSavedMaskData, UnknownLaunchCannotPromoteOneHalfToCompleteBool) {
    auto c = config();
    c.dispatch_launch = {};
    EXPECT_TRUE(compile({}, c).empty());
}
TEST(NativeCfgSavedMaskData, PartialLaunchCannotPromoteOneHalfToCompleteBool) {
    auto c = config();
    c.dispatch_launch.initiator |= 2u;
    EXPECT_TRUE(compile({}, c).empty());
}
TEST(NativeCfgSavedMaskData, OracleRefusesMissingInputsAndUndeclaredNativeWidth) {
    const auto module = compile({.data = true});
    ASSERT_FALSE(module.empty());
    for (bool width : {false, true}) {
        bpermute_oracle::Interpreter vm(module);
        if (width)
            vm.native_subgroup_width = 64;
        else
            vm.push_constants = config().user_sgprs;
        EXPECT_TRUE(vm.run_buffers(64, {{3, std::vector<uint32_t>(64)}}, 3).empty());
        EXPECT_FALSE(vm.error.empty());
    }
}
TEST(NativeCfgSavedMaskData, OracleObservesAChangedPhysicalSibling) {
    const auto module = compile({.data = true});
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config().user_sgprs;
    vm.push_constants[15] =
        UINT32_MAX; // hand-built adverse input for the original zero-high oracle
    const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(out.size(), 64u);
    uint32_t differences = 0;
    for (uint32_t lane = 0; lane < 64; ++lane) differences += out[lane] != uint32_t(lane < 32);
    EXPECT_EQ(differences, 32u);
}
TEST(NativeCfgSavedMaskData, OneArmMissingSlotCannotEstablishSccBranch) {
    slot_scc_refuses(SccUse::Immediate);
}
TEST(NativeCfgSavedMaskData, MissingSlotCannotPersistSccAcrossAnotherEdge) {
    slot_scc_refuses(SccUse::AcrossEdge);
}
TEST(NativeCfgSavedMaskData, MissingSlotScalarRelayCannotEstablishSccBranch) {
    slot_scc_refuses(SccUse::ScalarRelay);
}
TEST(NativeCfgSavedMaskData, AnAllPathSlotCanEstablishAndPersistScc) {
    for (auto use : {SccUse::Immediate, SccUse::AcrossEdge, SccUse::ScalarRelay})
        slot_scc_values(use, true);
}
TEST(NativeCfgSavedMaskData, ACurrentDefiniteSlotWriteEndsMissingArmProvenance) {
    slot_scc_values(SccUse::ReplacedSlot, false);
}

TEST(NativeCfgSavedMaskData, AbsentScalarInputCannotEstablishDispatcherScc) {
    const auto p = scalar_scc_program(false);
    const auto t = table();
    TerminalRejectCapture capture;
    EXPECT_TRUE(recompile_compute(p.data(), p.size(), &t, config()).empty());
    std::string reason;
    for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
    EXPECT_NE(reason.find("poisoned-scc"), std::string::npos) << reason;
}
TEST(NativeCfgSavedMaskData, AnExplicitScalarProducerEstablishesDispatcherScc) {
    all_ones(scalar_scc_program(true));
}
