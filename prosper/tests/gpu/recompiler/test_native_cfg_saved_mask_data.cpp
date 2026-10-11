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
#include <initializer_list>
#include <map>
#include <string>
#include <utility>
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

namespace {
enum class FreshMaskUse {
    Immediate,
    ScalarRelay,
    SccRelay,
    AcrossEdge,
    VectorRelay,
    Cmpx,
    SavedPair,
    Carry
};
std::vector<uint32_t> slot_mask_program(FreshMaskUse use, bool all_paths,
                                        bool replacement = false) {
    auto p = program({.numeric_read = true});
    const auto slot = std::find(p.begin(), p.end(), 0xd7610014u);
    p.insert(slot, {0xbf068004u, 0xbf840002u}); // s4=1 skips the sole spill
    if (all_paths) p.insert(p.begin() + 2, {0xd7610014u, 30u | (128u << 9u)});
    auto sink = std::find(p.begin(), p.end(), 0x7e02020eu);
    const size_t at = sink - p.begin();
    p.erase(sink);
    std::vector<uint32_t> mask;
    if (replacement) mask.push_back(mov(14, 193)); // current word ends missing-slot provenance
    if (use == FreshMaskUse::ScalarRelay) mask.push_back(mov(16, 14));
    if (use == FreshMaskUse::SccRelay)
        mask.push_back(0xbf06800eu); // scalar cmp -> explicit SCC word
    if (use == FreshMaskUse::AcrossEdge) mask.insert(mask.end(), {0xbf880001u, 0x7e040281u});
    if (use == FreshMaskUse::VectorRelay) mask.push_back(0x7e06020eu); // v3=s14
    const uint32_t source = use == FreshMaskUse::ScalarRelay   ? 16u
                            : use == FreshMaskUse::SccRelay    ? 253u
                            : use == FreshMaskUse::VectorRelay ? 259u
                                                               : 14u;
    if (use == FreshMaskUse::Carry)
        mask.insert(mask.end(), {0xd70f6a03u, source | (256u << 9u)}); // v3,VCC=source+v0
    else if (use == FreshMaskUse::SavedPair)
        mask.insert(mask.end(), {0xd4c20016u, source | (256u << 9u)}); // cmp_eq -> s[22:23]
    else
        mask.push_back((use == FreshMaskUse::Cmpx ? 0x7da40000u : 0x7d840000u) | source);
    if (use != FreshMaskUse::Cmpx)
        mask.push_back(mov(126, use == FreshMaskUse::SavedPair ? 22 : 106, true));
    mask.insert(mask.end(), {0x7e020281u, mov(126, 193, true)});
    p.insert(p.begin() + at, mask.begin(), mask.end());
    return p;
}
void mask_refuses(const std::vector<uint32_t>& p, const ComputeShaderConfig& base = config()) {
    const auto t = table();
    for (uint32_t path : {0u, 1u}) {
        auto c = base;
        c.user_sgprs[4] = path;
        TerminalRejectCapture capture;
        EXPECT_TRUE(recompile_compute(p.data(), p.size(), &t, c).empty());
        std::string reason;
        for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
        EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    }
}
void mask_values(const std::vector<uint32_t>& p, bool carry = false, bool first_lane = false) {
    const auto t = table();
    for (uint32_t path : {0u, 1u}) {
        auto c = config();
        c.user_sgprs[4] = path;
        TerminalRejectCapture capture;
        const auto module = recompile_compute(p.data(), p.size(), &t, c);
        std::string reason;
        for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
        ASSERT_FALSE(module.empty()) << reason;
        ASSERT_TRUE(has_op(module, 251));
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.push_constants = c.user_sgprs;
        const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane)
            EXPECT_EQ(out[lane], uint32_t((carry && lane != 0) || (first_lane && lane == 0)))
                << "path " << path << " lane " << lane;
    }
}
} // namespace

TEST(NativeCfgSavedMaskData, MissingSlotCannotObtainMaskAuthorityThroughFreshVectorCompare) {
    mask_refuses(slot_mask_program(FreshMaskUse::Immediate, false));
}
TEST(NativeCfgSavedMaskData, MissingSlotScalarRelayCannotEstablishFreshMask) {
    mask_refuses(slot_mask_program(FreshMaskUse::ScalarRelay, false));
}
TEST(NativeCfgSavedMaskData, MissingSlotSccRelayCannotEstablishFreshMask) {
    mask_refuses(slot_mask_program(FreshMaskUse::SccRelay, false));
}
TEST(NativeCfgSavedMaskData, MissingSlotCannotEstablishFreshMaskAcrossAnotherEdge) {
    mask_refuses(slot_mask_program(FreshMaskUse::AcrossEdge, false));
}
TEST(NativeCfgSavedMaskData, MissingSlotValuRelayCannotEstablishFreshMask) {
    mask_refuses(slot_mask_program(FreshMaskUse::VectorRelay, false));
}
TEST(NativeCfgSavedMaskData, MissingSlotCannotEstablishFreshCmpxExec) {
    mask_refuses(slot_mask_program(FreshMaskUse::Cmpx, false));
}
TEST(NativeCfgSavedMaskData, MissingSlotCannotEstablishFreshSavedPair) {
    mask_refuses(slot_mask_program(FreshMaskUse::SavedPair, false));
}
TEST(NativeCfgSavedMaskData, MissingSlotCannotEstablishFreshCarryMask) {
    mask_refuses(slot_mask_program(FreshMaskUse::Carry, false));
}
TEST(NativeCfgSavedMaskData, AllPathSlotFreshMasksUseGenuinePhysicalData) {
    for (auto use : {FreshMaskUse::Immediate, FreshMaskUse::ScalarRelay, FreshMaskUse::SccRelay,
                     FreshMaskUse::AcrossEdge, FreshMaskUse::VectorRelay, FreshMaskUse::Cmpx,
                     FreshMaskUse::SavedPair, FreshMaskUse::Carry})
        mask_values(slot_mask_program(use, true), use == FreshMaskUse::Carry,
                    use == FreshMaskUse::SccRelay);
}
TEST(NativeCfgSavedMaskData, CurrentWordReplacementEndsMissingSlotFreshMaskProvenance) {
    for (auto use : {FreshMaskUse::Immediate, FreshMaskUse::ScalarRelay, FreshMaskUse::SccRelay,
                     FreshMaskUse::AcrossEdge, FreshMaskUse::VectorRelay, FreshMaskUse::Cmpx,
                     FreshMaskUse::SavedPair, FreshMaskUse::Carry})
        mask_values(slot_mask_program(use, false, true), use == FreshMaskUse::Carry,
                    use == FreshMaskUse::SccRelay);
}

TEST(NativeCfgSavedMaskData, WideFreshCompareCannotInventItsMissingScalarHighWord) {
    auto p = slot_mask_program(FreshMaskUse::Immediate, true);
    const auto cmp = std::find(p.begin(), p.end(), 0x7d84000eu);
    const size_t at = cmp - p.begin();
    p.erase(cmp);
    p.insert(p.begin() + at, {0xd4e2006au, 14u | (256u << 9u)}); // cmp_eq_u64 s[14:15],v[0:1]
    auto c = config();
    c.user_sgprs.resize(15); // the low reload is genuine; the numeric high sibling is absent
    mask_refuses(p, c);
}
TEST(NativeCfgSavedMaskData, WideFreshCompareConsumesBothGenuineScalarWords) {
    auto p = slot_mask_program(FreshMaskUse::Immediate, true);
    const auto cmp = std::find(p.begin(), p.end(), 0x7d84000eu);
    const size_t at = cmp - p.begin();
    p.erase(cmp);
    p.insert(p.begin() + at, {0xd4e2006au, 14u | (256u << 9u)});
    mask_values(p);
}

namespace {
struct LoopProgram {
    std::vector<uint32_t> p;
    std::map<std::string, size_t> labels;
    std::vector<std::pair<size_t, std::string>> fixups;
    void add(std::initializer_list<uint32_t> words) { p.insert(p.end(), words); }
    void label(const std::string& name) { labels[name] = p.size(); }
    void branch(uint32_t op, const std::string& target) {
        fixups.emplace_back(p.size(), target);
        add({op});
    }
    std::vector<uint32_t> finish() {
        for (const auto& [pc, target] : fixups) p[pc] |= uint16_t(labels.at(target) - pc - 1);
        return p;
    }
};
// A compute value witness for the direct-EXEC-break scalar phi shape. The fragment validation
// sample has the same scalar producers/control edges; this fixture has an owned buffer sink.
std::vector<uint32_t> numeric_direct_break_loop(unsigned mode) {
    LoopProgram a;
    a.add({0x7e040300u}); // preserve the genuine guest lane in v2 for the output index
    if (mode != 1) a.add({mov(0, 128)});
    if (mode == 3 || mode == 4 || mode == 5) {
        a.add({mov(30, 135)}); // ordinary DATA=7, independent of old mask authority
        if (mode == 4) a.add({0xd7610014u, 30u | (128u << 9u)});
        a.add({0xbf068004u});
        a.branch(0xbf840000u, "slot-join");
        a.add({0xd7610014u, 30u | (128u << 9u)});
        a.label("slot-join");
    }
    if (mode == 6) a.add({mov(16, 128)});
    a.add({0x7e000280u, 0x7e020284u, 0x816a8180u, 0x876b8380u, mov(4, 126, true)});
    a.label("check");
    if (mode == 6) a.add({mov(0, 16)});   // check word depends on a different, genuine carried root
    a.branch(0xbf880000u, "exit");
    a.add(
        {mode == 0 ? 0x7d820200u : 0x7d820284u});   // scalar counter, or a defined false predicate
    a.add({0x87fe6a7eu});
    if (mode == 2) a.add({mov(17, 60), mov(0, 17)});   // a new fabricated scalar generation
    if (mode == 3 || mode == 4 || mode == 5) a.add({0xd7600000u, 276u | (128u << 9u)});
    if (mode == 5) a.add({mov(0, 135)});   // current replacement ends the missing slot generation
    if (mode == 6)
        a.add({mov(17, 60),
               mov(0, 17)});   // body replaces the current check generation with fabrication
    a.branch(0xbf880000u, "exit");
    if (mode == 1) a.add({mov(0, 135)});   // sole producer lies after the direct break
    a.add({0x7e000280u, mode == 6 ? 0x80108110u : 0x80008100u});
    a.branch(0xbf820000u, "check");
    a.label("exit");
    a.add({mov(126, 4, true), 0x7d820200u, 0x7e020200u, 0xe0702000u, 0x80020102u, end});
    return a.finish();
}
ComputeShaderConfig loop_config(unsigned mode, uint32_t path) {
    auto c = config();
    c.user_sgprs[4] = path;
    if (mode == 1) c.user_sgprs.clear();   // no launch s0 before the sole body producer
    return c;
}
void loop_refuses(unsigned mode) {
    const auto p = numeric_direct_break_loop(mode);
    const auto resources = table();
    for (uint32_t path : {0u, 1u}) {
        TerminalRejectCapture capture;
        EXPECT_TRUE(
            recompile_compute(p.data(), p.size(), &resources, loop_config(mode, path)).empty());
        std::string reason;
        for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
        EXPECT_NE(reason.find("fabricated-fresh-mask-source"), std::string::npos) << reason;
    }
}
void loop_values(unsigned mode, uint32_t expected) {
    const auto p = numeric_direct_break_loop(mode);
    const auto resources = table();
    for (uint32_t path : {0u, 1u}) {
        const auto c = loop_config(mode, path);
        const auto module = recompile_compute(p.data(), p.size(), &resources, c);
        ASSERT_FALSE(module.empty());
        ASSERT_TRUE(has_op(module, 246)) << "must exercise the structured loop join";
        ASSERT_FALSE(has_op(module, 251)) << "dispatcher must not hide the exit join";
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.push_constants = c.user_sgprs;
        const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane)
            EXPECT_EQ(out[lane], expected) << "path " << path << " lane " << lane;
    }
}
}   // namespace

TEST(NativeCfgSavedMaskData, NumericDirectBreakKeepsTheCurrentScalarValue) {
    loop_values(0, 4);
}
TEST(NativeCfgSavedMaskData, EarlyDirectBreakCannotInventAProducerThatHasNotRun) {
    loop_refuses(1);
}
TEST(NativeCfgSavedMaskData, DirectBreakBodyCannotBorrowPreheaderAuthorityAfterScalarFabrication) {
    loop_refuses(2);
}
TEST(NativeCfgSavedMaskData, OneArmReadlaneCannotReplaceAGenuineLoopScalar) {
    loop_refuses(3);
}
TEST(NativeCfgSavedMaskData, AllPathReadlaneReplacesTheLoopScalarWithGenuineCurrentData) {
    loop_values(4, 7);
}
TEST(NativeCfgSavedMaskData, CurrentOverwriteEndsFabricatedBodyScalarGeneration) {
    loop_values(5, 7);
}
TEST(NativeCfgSavedMaskData, DirectBreakCannotBorrowDifferentRootForAFabricatedCurrentWord) {
    loop_refuses(6);
}

namespace {
enum class SccRelay {
    Scalar,
    Vector,
    ScalarSpill,
    ExplicitSpill,
    VectorLaneRead,
    ScalarSecondSource,
    VectorThirdSource,
    AcrossEdge,
    IndexedLaneRead
};
std::vector<uint32_t> explicit_scc_relay_program(SccRelay use, bool all_paths,
                                                 bool replacement = false) {
    auto p = slot_mask_program(FreshMaskUse::SccRelay, all_paths);
    const auto cmp = std::find(p.begin(), p.end(), 0x7d8400fdu);
    const size_t at = cmp - p.begin();
    const bool vector = use == SccRelay::Vector || use == SccRelay::VectorThirdSource;
    *cmp = 0x7d840000u | (vector ? 259u : 16u);
    std::vector<uint32_t> relay;
    if (use == SccRelay::Vector || use == SccRelay::VectorLaneRead)
        relay.push_back(0x7e0602fdu);   // v3=SCC
    else if (use == SccRelay::ScalarSecondSource)
        relay.push_back(0x8010fd80u);   // s_add_u32 s16,0,SCC: explicit second source
    else if (use == SccRelay::VectorThirdSource)
        relay.insert(relay.end(), {0xd5430003u, 128u | (128u << 9u) | (253u << 18u)});
    else if (use != SccRelay::ExplicitSpill && use != SccRelay::IndexedLaneRead)
        relay.push_back(mov(16, 253));   // exact independent review's scalar SCC copy
    if (use == SccRelay::VectorLaneRead) relay.push_back(0x7e200503u);   // s16=readfirstlane(v3)
    if (use == SccRelay::IndexedLaneRead)
        relay.insert(relay.end(), {0xd7600010u, 256u | (253u << 9u)});   // s16=readlane(v0,SCC)
    if (use == SccRelay::ScalarSpill || use == SccRelay::ExplicitSpill) {
        relay.insert(relay.end(),
                     {0xd7610015u, (use == SccRelay::ExplicitSpill ? 253u : 16u) | (128u << 9u),
                      0xbf880001u, 0x7e040281u, 0xd7600010u, 277u | (128u << 9u)});
    }
    if (use == SccRelay::AcrossEdge) relay.insert(relay.end(), {0xbf880001u, 0x7e040281u});
    if (replacement) {
        // A vector overwrite defines every old lane only after full EXEC is restored.
        if (vector) relay.push_back(mov(126, 193, true));
        relay.push_back(vector ? 0x7e060280u : mov(16, 128));
    }
    p.insert(p.begin() + at, relay.begin(), relay.end());
    return p;
}
}   // namespace

TEST(NativeCfgSavedMaskData, MissingSccScalarMoveCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::Scalar, false));
}
TEST(NativeCfgSavedMaskData, MissingSccVectorMoveCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::Vector, false));
}
TEST(NativeCfgSavedMaskData, MissingSccScalarSpillCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::ScalarSpill, false));
}
TEST(NativeCfgSavedMaskData, MissingExplicitSccSpillCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::ExplicitSpill, false));
}
TEST(NativeCfgSavedMaskData, MissingSccVectorLaneReadCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::VectorLaneRead, false));
}
TEST(NativeCfgSavedMaskData, MissingSccSecondSourceCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::ScalarSecondSource, false));
}
TEST(NativeCfgSavedMaskData, MissingSccThirdSourceCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::VectorThirdSource, false));
}
TEST(NativeCfgSavedMaskData, MissingSccRelayAcrossAnEdgeCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::AcrossEdge, false));
}
TEST(NativeCfgSavedMaskData, MissingSccLaneSelectorCannotAuthenticateFreshMask) {
    mask_refuses(explicit_scc_relay_program(SccRelay::IndexedLaneRead, false));
}
TEST(NativeCfgSavedMaskData, AllPathSccRelaysUseTheirActualNumericValue) {
    for (auto use :
         {SccRelay::Scalar, SccRelay::Vector, SccRelay::ScalarSpill, SccRelay::ExplicitSpill,
          SccRelay::VectorLaneRead, SccRelay::ScalarSecondSource, SccRelay::VectorThirdSource,
          SccRelay::AcrossEdge, SccRelay::IndexedLaneRead}) {
        SCOPED_TRACE(static_cast<unsigned>(use));
        mask_values(explicit_scc_relay_program(use, true), false, true);
    }
}
TEST(NativeCfgSavedMaskData, CurrentReplacementEndsMissingSccRelayGeneration) {
    for (auto use :
         {SccRelay::Scalar, SccRelay::Vector, SccRelay::ScalarSpill, SccRelay::ExplicitSpill,
          SccRelay::VectorLaneRead, SccRelay::ScalarSecondSource, SccRelay::VectorThirdSource,
          SccRelay::AcrossEdge, SccRelay::IndexedLaneRead}) {
        SCOPED_TRACE(static_cast<unsigned>(use));
        mask_values(explicit_scc_relay_program(use, false, true), false, true);
    }
}

namespace {
void current_scc_values(const std::vector<uint32_t>& p) {
    const auto resources = table();
    for (uint32_t path : {0u, 1u}) {
        auto c = config();
        c.user_sgprs[4] = path;
        TerminalRejectCapture capture;
        const auto module = recompile_compute(p.data(), p.size(), &resources, c);
        std::string reason;
        for (const auto& [tag, payload] : capture.take()) reason += tag + payload;
        ASSERT_FALSE(module.empty()) << reason;
        ASSERT_TRUE(has_op(module, 251));
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        vm.push_constants = c.user_sgprs;
        const auto out = vm.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(out.size(), 64u);
        for (uint32_t lane = 0; lane < 64; ++lane)
            EXPECT_EQ(out[lane], uint32_t(lane == (path == 0 ? 1u : 0u)))
                << "path " << path << " lane " << lane;
    }
}
}   // namespace

TEST(NativeCfgSavedMaskData, ANewGenuineCompareEndsMissingSccConditionGeneration) {
    for (auto use :
         {SccRelay::Scalar, SccRelay::Vector, SccRelay::ScalarSpill, SccRelay::ExplicitSpill,
          SccRelay::VectorLaneRead, SccRelay::ScalarSecondSource, SccRelay::VectorThirdSource,
          SccRelay::AcrossEdge, SccRelay::IndexedLaneRead}) {
        SCOPED_TRACE(static_cast<unsigned>(use));
        auto p = explicit_scc_relay_program(use, false);
        const auto old = std::find(p.begin(), p.end(), 0xbf06800eu);
        p.insert(old + 1, 0xbf068004u);   // current SCC=(genuine s4==0), independent of the reload
        current_scc_values(p);
    }
}

TEST(NativeCfgSavedMaskData, AnUnrelatedMarkedMovePreservesGenuineSccAcrossAnEdge) {
    auto p = explicit_scc_relay_program(SccRelay::AcrossEdge, false);
    const auto compare = std::find(p.begin(), p.end(), 0xbf06800eu);
    *compare = 0xbf068004u;   // genuine current SCC=(s4==0)
    // This scalar copy has a missing-slot input, but S_MOV_B32 does not rewrite SCC.
    p.insert(compare + 1, mov(17, 14));
    current_scc_values(p);
}
