// Pure decoded effects pin CURRENT's all-path DATA and the actual Wave32 physical slot aliases.
// These tests deliberately isolate the MUST join from independent emitter placeholder guards.
#include "gpu/recompiler/rdna2_wave32_active_data.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include <gtest/gtest.h>
#include <initializer_list>
#include <vector>
namespace {
using namespace prosper::gpu;
void step(Wave32ActiveData& data, std::initializer_list<uint32_t> words, bool genuine = true) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.begin(), words.size(), ins);
    ASSERT_EQ(ins.size(), 1u);
    std::vector<std::pair<int, uint32_t>> scalars;
    std::vector<int> vectors;
    for_each_scalar_write(
        ins[0], [&](int base, uint32_t width) { scalars.emplace_back(base, width); }, true);
    for_each_possible_vector_write(ins[0], 255, [&](int word) { vectors.push_back(word); });
    advance_wave32_active_data(
        data, ins[0], scalars, vectors,
        ins[0].fmt == Rdna2Format::VOPC && !vopc_is_cmpx(ins[0].opcode) ? 106 : -1, genuine);
}
void produce(Wave32ActiveData& data) {
    step(data, {0x7e040281u});
}
void save(Wave32ActiveData& data) {
    step(data, {0xd7610013u, 0x00014a7eu});
    step(data, {0xd7610013u, 0x0001507fu});
}
void reload(Wave32ActiveData& data) {
    step(data, {0xd760000eu, 0x00014b13u});
    step(data, {0xd760000fu, 0x00015113u});
}
void restore(Wave32ActiveData& data) {
    step(data, {0xbefe040eu});
}
Wave32ActiveData saved() {
    Wave32ActiveData data;
    produce(data);
    save(data);
    return data;
}
SpirvCompute builder() {
    SpirvCompute b;
    b.is_compute = true;
    b.wave_size = b.native_subgroup_size = 32;
    b.guest_stage = GuestShaderStage::Compute;
    b.physical_domain = PhysicalExecutionDomain::Workgroup;
    b.local_count = 64;
    b.guest_local_size = {64, 1, 1};
    b.guest_compute_launch.wave_size = 32;
    b.guest_compute_launch.local = {64, 1, 1};
    return b;
}
}   // namespace
TEST(Wave32ActiveData, ActualDecodedSelectorsAreThirtySevenAndForty) {
    const uint32_t low[]{0xd7610013u, 0x00014a7eu};
    const uint32_t high[]{0xd7610013u, 0x0001507fu};
    const auto l = rdna2_decode_one(low, 2), h = rdna2_decode_one(high, 2);
    EXPECT_EQ(l.src[1].kind, OperandKind::InlineInt);
    EXPECT_EQ(l.src[1].value, 37);
    EXPECT_EQ(h.src[1].kind, OperandKind::InlineInt);
    EXPECT_EQ(h.src[1].value, 40);
}
TEST(Wave32ActiveData, GenuineWriterPublishesCurrentData) {
    Wave32ActiveData data;
    produce(data);
    EXPECT_TRUE(data.current.contains(2));
}
TEST(Wave32ActiveData, AWriterWithMissingInputRevokesTheOldWord) {
    auto data = saved();
    step(data, {0x7e040309u}, false);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, APartialDestinationCannotCreateTheCompleteWord) {
    Wave32ActiveData data;
    const uint32_t words[]{0x7e0402f9u, 0x06061480u};
    const auto instruction = rdna2_decode_one(words, 2);
    ASSERT_TRUE(instruction.has_sdwa);
    ASSERT_EQ(instruction.sdwa_dst_sel, 4u);
    ASSERT_EQ(instruction.sdwa_dst_unused, 2u);
    step(data, {words[0], words[1]});
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, ACurrentCmpxOnlyShrinksTheWordDomain) {
    auto data = saved();
    step(data, {0x7da40080u});
    EXPECT_TRUE(data.current.contains(2));
}
TEST(Wave32ActiveData, AnyUnprovedFullExecReplacementEndsCurrentData) {
    auto data = saved();
    step(data, {0xbefe03c1u});
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, BothActualWordsMakeTheRestoreASemanticNoOp) {
    auto data = saved();
    reload(data);
    restore(data);
    EXPECT_TRUE(data.current.contains(2));
}
TEST(Wave32ActiveData, AHighOnlyTransportCannotBorrowTheLowWord) {
    Wave32ActiveData data;
    produce(data);
    step(data, {0xd7610013u, 0x0001507fu});
    reload(data);
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, AScalarLowOverwriteEndsTheCurrentEquality) {
    auto data = saved();
    reload(data);
    step(data, {0xbe8e0380u});
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, AScalarHighOverwriteEndsTheCurrentEquality) {
    auto data = saved();
    reload(data);
    step(data, {0xbe8f0380u});
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, PhysicalLaneFiveRetiresTheRawThirtySevenTransport) {
    auto data = saved();
    step(data, {0xd7610013u, 0x00010a81u});
    reload(data);
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, PhysicalLaneEightRetiresTheRawFortyTransport) {
    auto data = saved();
    step(data, {0xd7610013u, 0x00011081u});
    reload(data);
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, ReadingADifferentRawAliasCannotBorrowUnsupportedLowering) {
    auto data = saved();
    step(data, {0xd760000eu, 0x00010b13u}); // selector5, not the stored raw37
    step(data, {0xd760000fu, 0x00011113u}); // selector8, not raw40
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, ASeparatePhysicalSlotPreservesTheRealTransport) {
    auto data = saved();
    step(data, {0xd7610013u, 0x00010c81u}); // physical6 differs from5/8
    reload(data);
    restore(data);
    EXPECT_TRUE(data.current.contains(2));
}
TEST(Wave32ActiveData, ASelectorFromScalarDataRetiresEveryPossibleAlias) {
    auto data = saved();
    step(data, {0xd7610013u, 0x00000080u});
    reload(data);
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, AnOutOfRangeLiteralRetiresEveryPossibleAlias) {
    auto data = saved();
    step(data, {0xd7610013u, 0x0001fe81u, 69u});
    reload(data);
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, OrdinaryValuReplacementRetiresItsOldSlotArray) {
    auto data = saved();
    step(data, {0x7e260280u});
    reload(data);
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, ReloadedScalarCopiesAreIndependentOfLaterSlotReplacement) {
    auto data = saved();
    reload(data);
    step(data, {0x7e260280u});
    restore(data);
    EXPECT_TRUE(data.current.contains(2));
}
TEST(Wave32ActiveData, NewExecGenerationCannotUseOldWordEqualities) {
    auto data = saved();
    step(data, {0x7da40080u});
    reload(data);
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, EveryPredecessorMustSupplyTheCurrentWord) {
    auto left = saved(), right = left;
    step(right, {0x7e040309u}, false);
    meet_wave32_active_data(left, right);
    EXPECT_FALSE(left.current.contains(2));
}
TEST(Wave32ActiveData, BothGenuinePredecessorsKeepTheCurrentWord) {
    auto left = saved(), right = left;
    meet_wave32_active_data(left, right);
    EXPECT_TRUE(left.current.contains(2));
}
TEST(Wave32ActiveData, BothMissingPredecessorsCannotCreateTheCurrentWord) {
    auto left = saved(), right = left;
    step(left, {0x7e040309u}, false);
    step(right, {0x7e040309u}, false);
    meet_wave32_active_data(left, right);
    EXPECT_FALSE(left.current.contains(2));
}
TEST(Wave32ActiveData, OnePredecessorSlotOverwriteEndsTheRestoreProof) {
    auto left = saved(), right = left;
    step(right, {0xd7610013u, 0x00010a81u});
    meet_wave32_active_data(left, right);
    reload(left);
    restore(left);
    EXPECT_FALSE(left.current.contains(2));
}
TEST(Wave32ActiveData, DistinctPathExecValuesCanEachDefineTheirOwnCurrentData) {
    auto left = saved(), right = left;
    step(left, {0x7da40080u});
    produce(left);
    save(left);
    reload(left);
    step(right, {0x7dea0081u});
    produce(right);
    save(right);
    reload(right);
    meet_wave32_active_data(left, right);
    restore(left);
    EXPECT_TRUE(left.current.contains(2))
        << "relative CURRENT equality does not assert identical path masks";
}
TEST(Wave32ActiveData, AConstantHighZeroIsNotAnAuthenticatedCurrentTransport) {
    auto data = saved();
    reload(data);
    step(data, {0xbe8f0380u});
    restore(data);
    EXPECT_FALSE(data.current.contains(2));
}
TEST(Wave32ActiveData, DataTrackingDoesNotSupplyUnknownRawLaunchWords) {
    const auto b = builder();
    EXPECT_TRUE(tracks_wave32_active_data(b));
    EXPECT_FALSE(compute_dispatch_has_complete_groups(b.guest_compute_launch));
}
TEST(Wave32ActiveData, APortableProfileDoesNotBorrowTheNativeDataRoute) {
    auto b = builder();
    b.native_subgroup_size = 0;
    EXPECT_FALSE(tracks_wave32_active_data(b));
}
TEST(Wave32ActiveData, APhysicalWidthMismatchDoesNotBorrowTheGuestRoute) {
    auto b = builder();
    b.native_subgroup_size = 64;
    EXPECT_FALSE(tracks_wave32_active_data(b));
}
TEST(Wave32ActiveData, APartialNativeSubgroupGeometryDoesNotEnableThisRoute) {
    auto b = builder();
    b.local_count = 63;
    b.guest_local_size = {63, 1, 1};
    b.guest_compute_launch.local = b.guest_local_size;
    EXPECT_FALSE(tracks_wave32_active_data(b));
}
TEST(Wave32ActiveData, AChangedDeclaredGeometryCannotBorrowTheBuilderShape) {
    auto b = builder();
    b.guest_compute_launch.local = {32, 2, 1};
    EXPECT_FALSE(tracks_wave32_active_data(b));
}
TEST(Wave32ActiveData, AFragmentDomainCannotUseTheComputeDataRoute) {
    auto b = builder();
    b.is_fragment = true;
    EXPECT_FALSE(tracks_wave32_active_data(b));
}

TEST(Wave32ActiveData, OverflowingWorkgroupDimensionsDoNotSupplyADomain) {
    auto b = builder();
    b.guest_compute_launch.local = b.guest_local_size = {0xffffffffu, 0xffffffffu, 64};
    EXPECT_FALSE(tracks_wave32_active_data(b));
}

TEST(Wave32ActiveData, InitialVccStorageCannotAuthenticateTheBranch) {
    Wave32ActiveData data;
    produce(data);
    step(data, {0xbf860000u});
    EXPECT_FALSE(data.control);
    EXPECT_TRUE(data.current.empty());
    step(data, {0x7d840080u});
    produce(data);
    EXPECT_FALSE(data.vcc_produced);
    EXPECT_TRUE(data.current.empty());
}
TEST(Wave32ActiveData, ARealCompareAuthenticatesTheCurrentVccBranch) {
    Wave32ActiveData data;
    step(data, {0x7d840080u});
    EXPECT_TRUE(data.vcc_produced);
    step(data, {0xbf860000u});
    produce(data);
    EXPECT_TRUE(data.control);
    EXPECT_TRUE(data.current.contains(2));
}
TEST(Wave32ActiveData, AScalarOverlapRetiresTheVccControlProducer) {
    Wave32ActiveData data;
    step(data, {0x7d840080u});
    step(data, {0xbeea0380u});
    EXPECT_FALSE(data.vcc_produced);
    step(data, {0xbf860000u});
    EXPECT_FALSE(data.control);
}
TEST(Wave32ActiveData, BothPredecessorsMustAuthenticateTheVccProducer) {
    Wave32ActiveData genuine;
    step(genuine, {0x7d840080u});
    auto missing = genuine;
    step(missing, {0xbeea0380u});
    meet_wave32_active_data(genuine, missing);
    EXPECT_FALSE(genuine.vcc_produced);
    step(genuine, {0xbf860000u});
    EXPECT_FALSE(genuine.control);
}
TEST(Wave32ActiveData, ARejectedControlChoiceCannotBorrowTheOtherPredecessor) {
    Wave32ActiveData genuine;
    step(genuine, {0x7d840080u});
    produce(genuine);
    Wave32ActiveData missing;
    step(missing, {0xbf860000u});
    meet_wave32_active_data(genuine, missing);
    EXPECT_FALSE(genuine.control);
    produce(genuine);
    EXPECT_TRUE(genuine.current.empty());
}

TEST(Wave32ActiveData, AnImplicitCarryCannotAuthenticateItsInitialVccInput) {
    Wave32ActiveData data;
    step(data, {0x50020280u});
    EXPECT_FALSE(data.vcc_produced);
    step(data, {0xbf860000u});
    EXPECT_FALSE(data.control);
}
TEST(Wave32ActiveData, AnImplicitCarryDoesNotKeepAnOldIndependentCompareProducer) {
    Wave32ActiveData data;
    step(data, {0x7d840080u});
    ASSERT_TRUE(data.vcc_produced);
    step(data, {0x50020280u});
    EXPECT_FALSE(data.vcc_produced);
}
TEST(Wave32ActiveData, BothGenuineControlPredecessorsRetainTheProducer) {
    Wave32ActiveData left, right;
    step(left, {0x7d840080u});
    step(right, {0x7d840080u});
    meet_wave32_active_data(left, right);
    EXPECT_TRUE(left.vcc_produced);
    step(left, {0xbf860000u});
    produce(left);
    EXPECT_TRUE(left.control);
    EXPECT_TRUE(left.current.contains(2));
}

TEST(Wave32ActiveData, ARelayedCompareCannotAuthenticateTheControlProducer) {
    Wave32ActiveData data;
    step(data, {0x7d840080u}, false);
    EXPECT_FALSE(data.vcc_produced);
    step(data, {0xbf860000u});
    produce(data);
    EXPECT_FALSE(data.control);
    EXPECT_TRUE(data.current.empty());
}
TEST(Wave32ActiveData, ARejectedBackedgeControlCannotBorrowTheEntryProducer) {
    Wave32ActiveData entry;
    step(entry, {0x7d840080u});
    produce(entry);
    auto backedge = entry;
    step(backedge, {0x7d840080u}, false);
    step(backedge, {0xbf860000u});
    meet_wave32_active_data(entry, backedge);
    step(entry, {0x7d840080u});
    produce(entry);
    EXPECT_FALSE(entry.control);
    EXPECT_FALSE(entry.vcc_produced);
    EXPECT_TRUE(entry.current.empty());
}

TEST(Wave32ActiveData, AnImplicitMaskReadLosesRecoveryBeforeAnyLaterControl) {
    Wave32ActiveData data;
    produce(data);
    step(data, {0x02040680u}, false);
    EXPECT_FALSE(data.control);
    step(data, {0x7d840480u});
    produce(data);
    EXPECT_FALSE(data.vcc_produced);
    EXPECT_TRUE(data.current.empty());
}
TEST(Wave32ActiveData, ARealVccProducerKeepsTheOrdinaryImplicitMaskControl) {
    Wave32ActiveData data;
    step(data, {0x7d840080u});
    step(data, {0x02040680u}, false);
    EXPECT_TRUE(data.control);
    EXPECT_TRUE(data.vcc_produced);
}
TEST(Wave32ActiveData, AnExplicitInitialMaskReadCannotLaunderLaterControl) {
    Wave32ActiveData data;
    step(data, {0xd1010002u, 0x01aa0680u}, false);
    step(data, {0x7d840480u});
    EXPECT_FALSE(data.control);
    EXPECT_FALSE(data.vcc_produced);
}
TEST(Wave32ActiveData, AScalarCopyOfInitialVccCannotAuthenticateNumericData) {
    Wave32ActiveData data;
    step(data, {0xbe94036au});
    step(data, {0x7d840014u});
    EXPECT_FALSE(data.control);
    EXPECT_FALSE(data.vcc_produced);
}
TEST(Wave32ActiveData, ALogicalLowProducerDoesNotGrantOrdinaryHighWordData) {
    Wave32ActiveData data;
    step(data, {0x7d840080u});
    ASSERT_TRUE(data.vcc_produced);
    step(data, {0xbe94036bu});
    EXPECT_FALSE(data.control);
}

TEST(Wave32ActiveData, AnImplicitScalarDestinationReadNeedsItsIndependentVccData) {
    const uint32_t word = 0xb7ea0001u; // ADDK reads old VCC_LO as scalar DATA
    const auto instruction = rdna2_decode_one(&word, 1);
    ASSERT_EQ(instruction.fmt, Rdna2Format::SOPK);
    ASSERT_EQ(instruction.opcode, 0x0fu);
    ASSERT_EQ(instruction.dst.value, 106);
    Wave32ActiveData data;
    step(data, {0x7d840080u});
    step(data, {word});
    EXPECT_FALSE(data.control);
}
