#include "bpermute_spirv_oracle.hpp"
#include "fixtures/portable_entry_exec.hpp"
#include "gpu/recompiler/rdna2_entry_exec_constants.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

using namespace prosper::gpu;
namespace fixture = prosper::test::portable_entry_exec;
namespace {
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& p) {
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(p.data(), p.size(), ins), p.size());
    return ins;
}
void execute(uint32_t lanes, bool phased) {
    const auto p = fixture::program(lanes, phased);
    const auto address = fixture::address();
    const auto module = fixture::compile(p, fixture::config(lanes), address);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    const auto got = vm.run_buffers(
        lanes, {{3, std::vector<uint32_t>(size_t{lanes} * fixture::planes, fixture::sentinel)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    EXPECT_EQ(got, fixture::expected(lanes))
        << "physical low/high, unsigned borrow, lane0 restore, full restore and phase scalar";
}
void no_authority(const ComputeShaderConfig& c) {
    const auto ins = decode(fixture::program());
    EXPECT_TRUE(portable_entry_exec_constant_reads(ins, c).empty());
}
void refuse(const std::vector<uint32_t>& p, const ComputeShaderConfig& c = fixture::config()) {
    const auto address = fixture::address();
    EXPECT_TRUE(fixture::compile(p, c, address).empty());
    EXPECT_FALSE(last_terminal_reject_reason(address).empty());
}
} // namespace

TEST(PortableEntryExec, FullWavePreservesBothWordsBorrowAndRestoreValues) {
    execute(64, false);
}
TEST(PortableEntryExec, FourWavesRetainIndependentGuestMasks) {
    execute(256, false);
}
TEST(PortableEntryExec, OriginalEntryConstantsSurviveAuthenticatedFirstBarrierPhase) {
    execute(256, true);
}
TEST(PortableEntryExec, CompleteThreeDimensionalLaunchHasFullPhysicalEntryWords) {
    auto c = fixture::config();
    c.local_x = 16;
    c.local_y = 16;
    c.threads_x = 1920;
    c.threads_y = 1088;
    c.dispatch_launch.num_thread = {16, 16, 1};
    c.dispatch_launch.dimensions = {120, 68, 1};
    EXPECT_EQ(portable_entry_exec_constant_reads(decode(fixture::program()), c),
              (std::vector<uint32_t>{0, 1}));
    // Structural compilation only: the CPU oracle's coordinate model is one-dimensional.
    EXPECT_FALSE(fixture::compile(fixture::program(), c).empty());
}
TEST(PortableEntryExec, PartialLastGuestWaveCannotSupplyFullWords) {
    auto c = fixture::config(96);
    no_authority(c);
    refuse(fixture::program(96), c);
}
TEST(PortableEntryExec, LessThanOneGuestWaveCannotSupplyFullWords) {
    auto c = fixture::config(32);
    no_authority(c);
    refuse(fixture::program(32), c);
}
TEST(PortableEntryExec, IncompleteGlobalAxesRemainOutsideGroupCountContract) {
    for (uint32_t axis = 0; axis < 3; ++axis) {
        auto c = fixture::config();
        c.local_y = c.local_z = 2;
        c.threads_y = c.threads_z = 2;
        if (axis == 0) --c.threads_x;
        if (axis == 1) --c.threads_y;
        if (axis == 2) --c.threads_z;
        no_authority(c);
    }
}
TEST(PortableEntryExec, UnspecifiedExtentIsNotAnAllLanesContract) {
    no_authority(ComputeShaderConfig{});
    auto c = fixture::config();
    c.threads_x = 0;
    no_authority(c);
    refuse(fixture::program(), c);
}
TEST(PortableEntryExec, ZeroLocalAxisCannotBorrowNormalizedDimensions) {
    auto c = fixture::config();
    c.local_y = 0;
    no_authority(c);
    refuse(fixture::program(), c);
}
TEST(PortableEntryExec, ThreeDimensionalProductOverflowIsRefusedBeforeMultiplication) {
    auto c = fixture::config();
    c.local_x = c.local_y = c.local_z = std::numeric_limits<uint32_t>::max();
    c.threads_x = c.threads_y = c.threads_z = std::numeric_limits<uint32_t>::max();
    no_authority(c);
}
TEST(PortableEntryExec, ExactExtentModeDoesNotAcquireThisRootAuthority) {
    auto c = fixture::config();
    c.exact_thread_extent = true;
    no_authority(c);
    refuse(fixture::program(), c);
    --c.threads_x;
    no_authority(c);
}
TEST(PortableEntryExec, NativeAndWave32DomainsKeepTheirExistingSeparateRules) {
    auto c = fixture::config();
    c.native_subgroup_size = 64;
    no_authority(c);
    c.native_subgroup_size = 0;
    c.wave_size = 32;
    no_authority(c);
    refuse(fixture::program(), c);
}
TEST(PortableEntryExec, EarlierExecWritePreventsNewEntryConstantsEvenWhenRestoringFull) {
    auto p = fixture::program();
    p.insert(p.begin(), 0xbefe04c1u);
    EXPECT_TRUE(portable_entry_exec_constant_reads(decode(p), fixture::config()).empty());
    refuse(p);
}
TEST(PortableEntryExec, LaterNumericReadCannotReuseTheEntryGeneration) {
    auto p = fixture::program();
    p.insert(p.end() - 1, {0x80927e80u, 0x82937f80u});
    EXPECT_EQ(portable_entry_exec_constant_reads(decode(p), fixture::config()),
              (std::vector<uint32_t>{0, 1}));
    refuse(p);
}
TEST(PortableEntryExec, IncomingBackedgeRevokesTheOriginalReadGeneration) {
    auto p = fixture::program();
    const auto pc = static_cast<int32_t>(p.size() - 1);
    p.insert(p.end() - 1, 0xbf820000u | static_cast<uint16_t>(-pc - 1));
    EXPECT_TRUE(portable_entry_exec_constant_reads(decode(p), fixture::config()).empty());
    refuse(p);
}
TEST(PortableEntryExec, BranchBeforeReadDoesNotProveEntryParticipation) {
    auto p = fixture::program();
    p.insert(p.begin(), 0xbf840000u);
    EXPECT_TRUE(portable_entry_exec_constant_reads(decode(p), fixture::config()).empty());
    refuse(p);
}
TEST(PortableEntryExec, IndirectPcAndDebugEscapeRemainUntrusted) {
    for (const uint32_t escape : {0xbe802000u, 0xbf920001u, 0xbf970000u}) {
        auto p = fixture::program();
        p.insert(p.end() - 1, escape);
        EXPECT_TRUE(portable_entry_exec_constant_reads(decode(p), fixture::config()).empty());
    }
}
TEST(PortableEntryExec, SyntheticTerminatorAndNonzeroSliceAreNotOriginalPrograms) {
    auto ins = decode(fixture::program());
    ins.back().synthetic_terminator = true;
    EXPECT_TRUE(portable_entry_exec_constant_reads(ins, fixture::config()).empty());
    ins = decode(fixture::program());
    ins.erase(ins.begin());
    EXPECT_TRUE(portable_entry_exec_constant_reads(ins, fixture::config()).empty());
}
TEST(PortableEntryExec, UnknownAndMalformedDecodedStreamsNeverAcquireAuthority) {
    auto ins = decode(fixture::program());
    ins.front().fmt = Rdna2Format::Unknown;
    EXPECT_TRUE(portable_entry_exec_constant_reads(ins, fixture::config()).empty());
    ins = decode(fixture::program());
    ++ins[1].pc;
    EXPECT_TRUE(portable_entry_exec_constant_reads(ins, fixture::config()).empty());
}
TEST(PortableEntryExec, FloatSourcesAndModifiersAreNotBitEquivalentSpecializations) {
    auto ins = decode(fixture::program());
    ins[0].fmt = Rdna2Format::VOP2;
    ins[1].opcode = kSop2OpcodeAddU32;
    EXPECT_TRUE(portable_entry_exec_constant_reads(ins, fixture::config()).empty());
    ins = decode(fixture::program());
    ins[0].has_modifier = ins[1].has_modifier = true;
    EXPECT_TRUE(portable_entry_exec_constant_reads(ins, fixture::config()).empty());
}
TEST(PortableEntryExec, NonIntegerExecSourceCannotReceiveTheIntegerBitSubstitution) {
    auto p = fixture::program();
    p.insert(p.begin(), 0x060c007eu); // v_add_f32 v6,EXEC_LO,v0: not an audited integer SOP2
    refuse(p);
}
TEST(PortableEntryExec, APreviousCompilationDoesNotSeedModifiedCodeOrAnotherLaunch) {
    const auto p = fixture::program();
    EXPECT_FALSE(fixture::compile(p).empty());
    auto changed = p;
    changed.insert(changed.begin(), 0xbefe04c1u);
    refuse(changed);
    auto c = fixture::config();
    c.threads_y = 0;
    refuse(p, c);
    EXPECT_FALSE(fixture::compile(p).empty());
}
TEST(PortableEntryExec, RootCannotBeConstructedByBodyOrPhaseCallers) {
    EXPECT_FALSE((std::is_constructible_v<PortableEntryExecRoot, const uint32_t*, size_t,
                                          const ComputeShaderConfig&>));
}
TEST(PortableEntryExec, BareDispatcherAndSlicedPhaseCannotInferWordsFromTrueBool) {
    const auto code = fixture::program();
    auto ins = decode(code);
    for (bool slice : {false, true}) {
        SpirvCompute b;
        b.begin(4, nullptr, 256, 1, 1, 64, 0, true, true);
        b.native_logical_uniform_launch = true;
        RegState rs;
        rs.exec = b.btrue();
        rs.vcc = rs.scc = b.bfalse();
        rs.vreg[0] = b.localid;
        if (slice) ins.back().synthetic_terminator = true;
        EXPECT_FALSE(emit_cfg_state_machine(b, rs, ins, {}, nullptr, true, false, {}, code.data(),
                                            code.size()));
    }
}
TEST(PortableEntryExec, NgGWrapperCannotMintOrdinaryGuestEntryAuthority) {
    const auto p = fixture::program();
    EXPECT_TRUE(
        recompile_ngg_exports_for_test(p.data(), p.size(), 0, 0, nullptr, 4, 0, {}, true).empty());
}
TEST(PortableEntryExec, IndependentExpectedValuesDetectWrongWordAndWrongBorrow) {
    const auto wanted = fixture::expected();
    EXPECT_NE(wanted[0], 0u);
    EXPECT_NE(wanted[256], std::numeric_limits<uint32_t>::max());
    EXPECT_NE(wanted[512], 0u);
    EXPECT_EQ(wanted[768], 9u);
    EXPECT_EQ(wanted[769], fixture::sentinel);
}

TEST(PortableEntryExec, LegacyNormalizedLaunchDoesNotProveEntryWords) {
    auto c = fixture::config();
    c.dispatch_launch = {};
    no_authority(c);
    refuse(fixture::program(), c);
}
TEST(PortableEntryExec, EveryProducingWordHasIndependentRequiredPresence) {
    for (uint32_t bit = 1; bit <= ComputeDispatchLaunchState::kDimZ; bit <<= 1) {
        auto c = fixture::config();
        c.dispatch_launch.known &= ~bit;
        no_authority(c);
        refuse(fixture::program(), c);
    }
}
TEST(PortableEntryExec, PartialThreadgroupControlIsIndependentOfDimensionMode) {
    auto c = fixture::config();
    c.dispatch_launch.initiator |= 2u; // group-count mode still has PARTIAL_TG_EN
    c.dispatch_launch.num_thread[0] |= 31u << 16u;
    no_authority(c);
    refuse(fixture::program(), c);
}
TEST(PortableEntryExec, DisabledPartialSettingsAreIgnoredWithoutLosingTheirRawWords) {
    auto c = fixture::config();
    for (auto& word : c.dispatch_launch.num_thread) word |= 0xffff0000u;
    EXPECT_EQ(portable_entry_exec_constant_reads(decode(fixture::program()), c),
              (std::vector<uint32_t>{0, 1}));
    const auto module = fixture::compile(fixture::program(), c);
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter vm(module);
    EXPECT_EQ(vm.run_buffers(
                  256, {{3, std::vector<uint32_t>(256u * fixture::planes, fixture::sentinel)}}, 3),
              fixture::expected());
    EXPECT_TRUE(vm.error.empty()) << vm.error;
}
TEST(PortableEntryExec, RawThreadModeAndWidthCannotBorrowNormalizedConfig) {
    for (const uint64_t flag :
         {uint64_t{32u}, uint64_t{1u << 15u}, uint64_t{1u << 14u}, uint64_t{1u << 13u},
          uint64_t{1u << 12u}, uint64_t{1u << 31u}, uint64_t{1} << 32u}) {
        auto c = fixture::config();
        c.dispatch_launch.initiator |= flag;
        no_authority(c);
        refuse(fixture::program(), c);
    }
    auto c = fixture::config();
    c.dispatch_launch.initiator = 0;
    no_authority(c);
    refuse(fixture::program(), c);
}
TEST(PortableEntryExec, RawDimensionsAndRegisterWordsCannotBeNormalizedIntoAuthority) {
    for (uint32_t axis = 0; axis < 3; ++axis) {
        auto c = fixture::config();
        ++c.dispatch_launch.num_thread[axis];
        no_authority(c);
        c = fixture::config();
        ++c.dispatch_launch.dimensions[axis];
        no_authority(c);
        c = fixture::config();
        c.dispatch_launch.num_thread[axis] = 0;
        no_authority(c);
    }
}
TEST(PortableEntryExec, CheckedRawGeometryRejectsSaturationAndLocalProductOverflow) {
    auto c = fixture::config();
    c.threads_x = UINT32_MAX;
    c.dispatch_launch.dimensions[0] = UINT32_MAX;
    no_authority(c);
    c.local_x = c.local_y = c.local_z = 65535;
    c.threads_x = c.threads_y = c.threads_z = 65535;
    c.dispatch_launch.num_thread = {65535, 65535, 65535};
    c.dispatch_launch.dimensions = {1, 1, 1};
    no_authority(c);
}
