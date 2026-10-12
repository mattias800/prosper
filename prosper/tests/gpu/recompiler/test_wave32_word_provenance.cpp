// Wave32 ordinary DATA provenance is independent of the one-word Bool mask domain.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <ios>
#include <initializer_list>
#include <string>
#include <vector>
using namespace prosper::gpu;
namespace {
using Words = std::vector<uint32_t>;
std::atomic<uint64_t> next_address{0xa4964001};
thread_local std::string reason;
thread_local std::string artifact_stem;
Words compile(const Words& body, bool values = false, int project = 44) {
    Words code{0xbefe04c1u};
    if (values) code.push_back(0x7e060280u); // genuine output v3=0
    code.insert(code.end(), body.begin(), body.end());
    code.push_back(0xbead03c1u); // original four-case high-word initializer
    code.push_back(0xbefe0300u | static_cast<uint32_t>(project)); // B32 EXEC_LO
    if (values)
        code.insert(code.end(), {0x7e060281u, 0xbefe03c1u});
    else
        code.push_back(0x7e0602f2u);
    code.insert(code.end(), {0xe0702000u, 0x80020300u, 0xbf810000u});
    ShaderResourceTable table;
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    table.resources.push_back(output);
    ComputeShaderConfig config;
    config.local_x = config.wave_size = config.native_subgroup_size = 32;
    prosper::test::complete_compute_launch(config, 2);
    config.user_sgprs.resize(4);
    config.tgid_x_en = true;
    const uint64_t address = next_address.fetch_add(1, std::memory_order_relaxed);
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    reason = last_terminal_reject_reason(address);
    artifact_stem.clear();
    // The environment is fixed before the test process starts; no test mutates it.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* directory = std::getenv("PROSPER_WAVE32_PROVENANCE_ARTIFACTS")) {
        artifact_stem = std::string(directory) + "/" +
                        ::testing::UnitTest::GetInstance()->current_test_info()->name() + "-" +
                        std::to_string(address);
        std::ofstream guest(artifact_stem + ".guest.u32", std::ios::binary);
        guest.write(reinterpret_cast<const char*>(code.data()), code.size() * sizeof(uint32_t));
        if (!module.empty()) {
            std::ofstream spv(artifact_stem + ".spv", std::ios::binary);
            spv.write(reinterpret_cast<const char*>(module.data()),
                      module.size() * sizeof(uint32_t));
        }
    }
    return module;
}
void refused(const Words& body, const char* tag = "scalar-fabricated-lane-mask", int project = 44) {
    EXPECT_TRUE(compile(body, false, project).empty()) << reason;
    EXPECT_NE(reason.find(tag), std::string::npos) << reason;
}
void admitted(const Words& body) {
    EXPECT_FALSE(compile(body).empty()) << reason;
}
void values(const Words& body, uint32_t expected, bool dispatcher = false) {
    const auto module = compile(body, true);
    ASSERT_FALSE(module.empty()) << reason;
    if (dispatcher) {
        bool has_switch = false;
        for (size_t pc = 5; pc < module.size();) {
            const uint32_t count = module[pc] >> 16;
            ASSERT_GT(count, 0u);
            ASSERT_LE(pc + count, module.size());
            has_switch |= (module[pc] & 0xffffu) == 251u;
            pc += count;
        }
        ASSERT_TRUE(has_switch) << "crossing scalar branches must exercise dispatcher reloads";
    }
    for (uint32_t group : {0u, 1u}) {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 32;
        vm.native32_workgroup_count = 2;
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        const auto actual = vm.run_buffers(32, {{3, Words(64, 0xdeadbeefu)}}, 3, group);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(actual.size(), 64u);
        if (!artifact_stem.empty()) {
            std::ofstream data(artifact_stem + ".group" + std::to_string(group) + ".actual.u32",
                               std::ios::binary);
            data.write(reinterpret_cast<const char*>(actual.data()),
                       actual.size() * sizeof(uint32_t));
        }
        for (uint32_t lane = 0; lane < 32; ++lane)
            EXPECT_EQ(actual[lane], (expected >> lane) & 1u)
                << "group=" << group << " lane=" << lane;
    }
}
// A scalar loop with an optional early exit or one-arm current-generation replacement.
Words scalar_loop(bool entry_word, bool early_exit, bool missing_overwrite,
                  bool exit_replacement = false) {
    Words code{0xbe900380u, 0xbe910383u}; // s16=0; s17=3
    if (entry_word) code.push_back(0xbea80380u); // current s40=0
    const uint32_t header = static_cast<uint32_t>(code.size());
    code.push_back(0xbf068004u); // genuine GroupIdX s4 == 0
    const uint32_t branch = static_cast<uint32_t>(code.size());
    code.push_back(0xbf840001u);
    code.push_back(missing_overwrite ? 0xbea8033cu : 0xbea80380u);
    code.push_back(0x81108110u); // current counter++
    code.push_back(0xbf0a1110u); // counter < 3
    const uint32_t latch = static_cast<uint32_t>(code.size());
    code.push_back(0xbf850000u | ((header - (latch + 1u)) & 0xffffu));
    const uint32_t exit = static_cast<uint32_t>(code.size());
    if (early_exit) code[branch] = 0xbf840000u | (exit - (branch + 1u));
    if (exit_replacement) code.push_back(0xbea80380u);
    code.insert(code.end(), {0xbf068028u, 0xbf840002u, 0xbeac0380u, 0xbf820001u, 0xbeac03c1u});
    return code;
}
Words crossing_scalar_edges(bool source_defined, bool current_replacement = false) {
    Words code;
    code.reserve(20);
    if (source_defined) code.push_back(0x7e280280u); // current v20=0
    code.push_back(0x7e0c0314u); // v6=v20
    if (current_replacement) code.push_back(0x7e0c0280u); // current whole v6=0
    // A genuine QUAD_PERM OR selects the existing public compute dispatcher. Its three words
    // are independently initialized; it neither consumes nor defines the v6 under test.
    code.insert(code.end(), {0x7e0e0280u, 0x7e100280u, 0x7e120280u, 0x381210fau, 0xff08b107u});
    // The two forward regions cross. Group0 follows the second branch and Group1 the first.
    code.insert(code.end(), {0xbf068004u, 0xbf840003u, 0xbeb20387u, 0xbf068104u, 0xbf840001u,
                             0xbeb30387u, 0x7e580506u});
    return code;
}
}   // namespace
// These are the four unchanged project-owned body streams from the coherent raw2 replay.
TEST(Wave32WordProvenance, MissingDirect) {
    refused({0x7e0c0314u, 0x7e580506u});
}
TEST(Wave32WordProvenance, GenuineDirect) {
    admitted({0x7e280280u, 0x7e0c0314u, 0x7e580506u});
}
TEST(Wave32WordProvenance, MissingAcrossEdge) {
    refused({0x7e0c0314u, 0xbf880000u, 0x7e580506u});
}
TEST(Wave32WordProvenance, GenuineAcrossEdge) {
    admitted({0x7e280280u, 0x7e0c0314u, 0xbf880000u, 0x7e580506u});
}
TEST(Wave32WordProvenance, CurrentZeroActualValues) {
    values({0x7e0c0280u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, CurrentAllOnesActualValues) {
    values({0x7e0c02c1u, 0x7e580506u}, UINT32_MAX);
}
TEST(Wave32WordProvenance, GenuineEntryActualValues) {
    values({0x7e0c0300u, 0xbf880000u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, CompleteOverwriteRetiresMissingWord) {
    values({0x7e0c0314u, 0x7e0c0280u, 0xbf880000u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, ScalarOverwriteRetiresMissingWord) {
    values({0x7e0c0314u, 0x7e580506u, 0xbeac0380u, 0xbf880000u}, 0);
}
TEST(Wave32WordProvenance, CurrentMissingWordReplacesHistoricalGenuineWord) {
    refused({0x7e0c0280u, 0x7e0c0314u, 0xbf880000u, 0x7e580506u});
}
TEST(Wave32WordProvenance, MissingScalarRelayAcrossEdge) {
    refused({0x7e0c0314u, 0x7e580506u, 0xbeae032cu, 0xbf880000u, 0xbeac032eu});
}
TEST(Wave32WordProvenance, GenuineScalarRelayAcrossEdge) {
    values({0x7e0c0280u, 0x7e580506u, 0xbeae032cu, 0xbf880000u, 0xbeac032eu}, 0);
}
TEST(Wave32WordProvenance, PartialWriteRetainsMissingOldBits) {
    refused({0x7e0c02f9u, 0x00051400u, 0xbf880000u, 0x7e580506u});
}
TEST(Wave32WordProvenance, PartialWriteRetainsGenuineOldBits) {
    values({0x7e0c0280u, 0x7e0c02f9u, 0x00051400u, 0xbf880000u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, PredicatedWriteRetainsMissingInactiveLanes) {
    refused({0xbefe0380u, 0x7e0c0280u, 0xbefe03c1u, 0xbf880000u, 0x7e580506u});
}
TEST(Wave32WordProvenance, PredicatedWriteRetainsGenuineInactiveLanes) {
    values({0x7e0c0280u, 0xbefe0380u, 0x7e0c0280u, 0xbefe03c1u, 0xbf880000u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, LowWordFullRestoreAllowsCurrentWave32Overwrite) {
    values({0xbefe0380u, 0xbefe03c1u, 0xbf880000u, 0x7e0c0280u, 0xbf880000u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, OneArmWriterIsNotAnEntryValue) {
    refused({0xbf068004u, 0xbf840001u, 0x7e280280u, 0x7e0c0314u, 0x7e580506u});
}
TEST(Wave32WordProvenance, BothArmsRetainTheCurrentWord) {
    values({0x7e280280u, 0xbf068004u, 0xbf840001u, 0x7e280280u, 0x7e0c0314u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, MissingFreshPredicateDirect) {
    refused({0x7e0c0314u, 0x7d840106u, 0xbeac036au}, "fabricated-fresh-mask-source");
}
TEST(Wave32WordProvenance, MissingFreshPredicateAcrossEdge) {
    refused({0x7e0c0314u, 0xbf880000u, 0x7d840106u, 0xbeac036au}, "fabricated-fresh-mask-source");
}
TEST(Wave32WordProvenance, GenuineFreshPredicateAcrossEdge) {
    admitted({0x7e0c0280u, 0xbf880000u, 0x7d840106u, 0xbeac036au});
}
TEST(Wave32WordProvenance, MissingCompareSccScalarRelayAcrossEdge) {
    refused({0x7e580514u, 0xbf06802cu, 0xbeac03fdu, 0xbf880000u});
}
TEST(Wave32WordProvenance, GenuineCompareSccScalarRelayAcrossEdge) {
    values({0x7e280280u, 0x7e580514u, 0xbf06802cu, 0xbeac03fdu, 0xbf880000u}, 1);
}
TEST(Wave32WordProvenance, FullExecHelperRequiresActualWidth) {
    std::vector<Rdna2Inst> decoded;
    const uint32_t code = 0xbefe03c1u;
    ASSERT_EQ(rdna2_walk(&code, 1, decoded), 1u);
    ASSERT_EQ(decoded.size(), 1u);
    EXPECT_EQ(decoded.front().opcode, kSop1OpcodeMovB32);
    EXPECT_TRUE(vector_exec_is_full_after_write(decoded.front(), false, 32));
    EXPECT_FALSE(vector_exec_is_full_after_write(decoded.front(), false, 64));
}

TEST(Wave32WordProvenance, B32MaskDoesNotDefineItsNeighbor) {
    refused({0x7e0c0280u, 0x7d8400f9u, 0x0606ac06u, 0x7e0c022du, 0xbf880000u, 0x7e580506u});
}
TEST(Wave32WordProvenance, GenuineNeighborOfB32MaskRemainsData) {
    values(
        {0xbead0380u, 0x7e0c0280u, 0x7d8400f9u, 0x0606ac06u, 0x7e0c022du, 0xbf880000u, 0x7e580506u},
        0);
}
TEST(Wave32WordProvenance, SelectedConstantSlotStaysGenuineBesideMissingWord) {
    values(
        {0xd7610014u, 0x0001003cu, 0xd7610014u, 0x00010280u, 0xbf880000u, 0xd760002cu, 0x00010314u},
        0);
}
TEST(Wave32WordProvenance, MissingSelectedSlotCannotGainAuthority) {
    refused({0xd7610014u, 0x0001003cu, 0xd7610014u, 0x00010280u, 0xbf880000u, 0xd760002cu,
             0x00010114u});
}
TEST(Wave32WordProvenance, DynamicAbsentSelectorAcrossEdgeCannotGainAuthority) {
    refused({0xd760002cu, 0x00005100u, 0xbf880000u});
}
TEST(Wave32WordProvenance, GenuineDynamicSelectorAcrossEdgeRemainsData) {
    values({0xbea80380u, 0xd760002cu, 0x00005100u, 0xbf880000u}, 0);
}

TEST(Wave32WordProvenance, ExplicitB32MaskEncodingHasOneCurrentWord) {
    Words code{0x7d8400f9u, 0x0606ac06u, 0xd760002cu, 0x00005100u};
    std::vector<Rdna2Inst> decoded;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), decoded), code.size());
    ASSERT_EQ(decoded.size(), 2u);
    EXPECT_EQ(decoded[0].dst.kind, OperandKind::SGPR);
    EXPECT_EQ(decoded[0].dst.value, 44);
    EXPECT_EQ(decoded[1].src[1].kind, OperandKind::SGPR);
    EXPECT_EQ(decoded[1].src[1].value, 40);
}
TEST(Wave32WordProvenance, AbsentHighVccWordIsNotTheCurrentLowPredicate) {
    refused({0x7e0c026bu, 0x7e580506u});
}
TEST(Wave32WordProvenance, GenuineHighVccWordRemainsCurrentData) {
    values({0xbeeb0380u, 0x7e0c026bu, 0xbf880000u, 0x7e580506u}, 0);
}
TEST(Wave32WordProvenance, AbsentHighVccWordAcrossEdgeIsNotTheCurrentLowPredicate) {
    refused({0x7e0c026bu, 0xbf880000u, 0x7e580506u});
}

TEST(Wave32WordProvenance, MissingSccBranchCannotSelectGenuineMaskConstants) {
    refused({0x7e500514u, 0xbf068028u, 0xbf840002u, 0xbeac0380u, 0xbf820001u, 0xbeac03c1u},
            "poisoned-scc");
}
TEST(Wave32WordProvenance, GenuineSccBranchSelectsTheActualZeroMask) {
    values(
        {0x7e280280u, 0x7e500514u, 0xbf068028u, 0xbf840002u, 0xbeac0380u, 0xbf820001u, 0xbeac03c1u},
        0);
}
TEST(Wave32WordProvenance, MissingSccBranchAcrossEdgeCannotSelectGenuineMaskConstants) {
    refused(
        {0x7e500514u, 0xbf068028u, 0xbf880000u, 0xbf840002u, 0xbeac0380u, 0xbf820001u, 0xbeac03c1u},
        "poisoned-scc");
}
TEST(Wave32WordProvenance, GenuineSccBranchAcrossEdgeSelectsTheActualZeroMask) {
    values({0x7e280280u, 0x7e500514u, 0xbf068028u, 0xbf880000u, 0xbf840002u, 0xbeac0380u,
            0xbf820001u, 0xbeac03c1u},
           0);
}

TEST(Wave32WordProvenance, UnprovidedEntrySccCannotSupplyNumericMaskAuthority) {
    refused({0xbeac03fdu, 0xbf880000u});
}
TEST(Wave32WordProvenance, ActualCompareReplacesUnprovidedEntryScc) {
    values({0xbf060404u, 0xbeac03fdu, 0xbf880000u}, 1);
}
TEST(Wave32WordProvenance, OneArmScalarCannotSupplyBranchAuthority) {
    refused({0xbf068004u, 0xbf840001u, 0xbea80380u, 0xbf068028u, 0xbf840002u, 0xbeac0380u,
             0xbf820001u, 0xbeac03c1u},
            "poisoned-scc");
}
TEST(Wave32WordProvenance, BothPathsScalarSuppliesTheActualBranchValue) {
    values({0xbea80380u, 0xbf068004u, 0xbf840001u, 0xbea80380u, 0xbf068028u, 0xbf840002u,
            0xbeac0380u, 0xbf820001u, 0xbeac03c1u},
           0);
}
TEST(Wave32WordProvenance, RelayedMissingSccCannotSupplyBranchAuthority) {
    refused({0x7e500514u, 0xbf068028u, 0xbeae03fdu, 0xbf06802eu, 0xbf840002u, 0xbeac0380u,
             0xbf820001u, 0xbeac03c1u},
            "poisoned-scc");
}
TEST(Wave32WordProvenance, GenuineRelayedSccSuppliesTheActualBranchValue) {
    values({0x7e280280u, 0x7e500514u, 0xbf068028u, 0xbeae03fdu, 0xbf06802eu, 0xbf840002u,
            0xbeac0380u, 0xbf820001u, 0xbeac03c1u},
           UINT32_MAX);
}
TEST(Wave32WordProvenance, InitializedLoopConditionKeepsItsCheckedSourceRoots) {
    values(scalar_loop(true, false, false), 0);
}
TEST(Wave32WordProvenance, LoopOneArmScalarDoesNotBecomeAnExitInput) {
    refused(scalar_loop(false, false, false), "poisoned-scc");
}
TEST(Wave32WordProvenance, LoopMissingCurrentOverwriteCannotBorrowItsEntryWord) {
    refused(scalar_loop(true, false, true), "poisoned-scc");
}
TEST(Wave32WordProvenance, LoopEarlyExitBeforeWriterDoesNotDefineTheWord) {
    refused(scalar_loop(false, true, false), "poisoned-scc");
}
TEST(Wave32WordProvenance, LoopEarlyExitRetainsGenuineEntryData) {
    values(scalar_loop(true, true, false), 0);
}
TEST(Wave32WordProvenance, CurrentExitReplacementRepairsTheMissingLoopWord) {
    // An unconditional current writer repairs the same one-arm loop as the negative above.
    // The separately retained early SCC-break shape is an unsupported route calibration.
    values(scalar_loop(false, false, false, true), 0);
}

TEST(Wave32WordProvenance, UnsignedHighMultiplyProducesActualCurrentMaskBits) {
    // UINT_MAX squared has high word UINT_MAX-1; this also exercises the oracle's product pair.
    const uint32_t instruction = 0x80000000u | (0x35u << 23) | (44u << 16) | (193u << 8) | 193u;
    values({instruction}, UINT32_MAX - 1u);
}

TEST(Wave32WordProvenance, SccPreservingDataInstructionsCannotRefreshAMissingCondition) {
    for (uint32_t op : {0x24u, 0x26u, 0x32u, 0x35u}) {
        const uint32_t instruction = 0x80000000u | (op << 23) | (50u << 16) | (129u << 8) | 128u;
        refused({0x7e500514u, 0xbf068028u, instruction, 0xbf880000u, 0xbf840002u, 0xbeac0380u,
                 0xbf820001u, 0xbeac03c1u},
                "poisoned-scc");
    }
}
TEST(Wave32WordProvenance, SccPreservingDataInstructionsKeepTheGenuineCurrentCondition) {
    for (uint32_t op : {0x24u, 0x26u, 0x32u, 0x35u}) {
        const uint32_t instruction = 0x80000000u | (op << 23) | (50u << 16) | (129u << 8) | 128u;
        values({0xbf060404u, instruction, 0xbf880000u, 0xbf840002u, 0xbeac0380u, 0xbf820001u,
                0xbeac03c1u},
               0);
    }
}
TEST(Wave32WordProvenance, UnrepresentedMaskSccCannotBecomeScalarAtAnEdge) {
    refused(
        {0x7d840100u, 0xbe983c6au, 0xbf880000u, 0xbf840002u, 0xbeac0380u, 0xbf820001u, 0xbeac03c1u},
        "poisoned-scc");
}
TEST(Wave32WordProvenance, FreshScalarCompareReplacesUnrepresentedMaskScc) {
    values({0x7d840100u, 0xbe983c6au, 0xbf880000u, 0xbf060404u, 0xbf840002u, 0xbeac0380u,
            0xbf820001u, 0xbeac03c1u},
           0);
}

TEST(Wave32WordProvenance, DispatcherReloadCannotAuthenticateAMissingOrdinaryWord) {
    refused(crossing_scalar_edges(false));
}
TEST(Wave32WordProvenance, DispatcherReloadKeepsTheGenuineCurrentOrdinaryWord) {
    values(crossing_scalar_edges(true), 0, true);
}
TEST(Wave32WordProvenance, DispatcherReloadKeepsTheCurrentWholeReplacement) {
    values(crossing_scalar_edges(false, true), 0, true);
}

// A B32 mask writes one physical word; a stored or PHI neighbor keeps its current provenance.
TEST(Wave32WordProvenance, B32CompareCannotClearAStoredMissingNeighbor) {
    refused({0xbead033cu, 0x7e0c0280u, 0x7d8400f9u, 0x0606ac06u, 0x7e0c022du, 0x7e580506u});
    EXPECT_NE(reason.find("pc=8"), std::string::npos) << reason;
}
TEST(Wave32WordProvenance, B32CompareKeepsTheGenuineStoredNeighbor) {
    values(
        {0xbebc0387u, 0xbead033cu, 0x7e0c0280u, 0x7d8400f9u, 0x0606ac06u, 0x7e0c022du, 0x7e580506u},
        7);
}
TEST(Wave32WordProvenance, B32CompareAllowsAGenuineCurrentNeighborReplacement) {
    values(
        {0xbead033cu, 0x7e0c0280u, 0x7d8400f9u, 0x0606ac06u, 0xbead0387u, 0x7e0c022du, 0x7e580506u},
        7);
}
TEST(Wave32WordProvenance, B32CompareCannotClearAOneArmNeighborPhi) {
    refused({0xbf068004u, 0xbf840001u, 0xbead0380u, 0x7e0c0280u, 0x7d8400f9u, 0x0606ac06u,
             0x7e0c022du, 0x7e580506u});
    EXPECT_NE(reason.find("pc=10"), std::string::npos) << reason;
}
TEST(Wave32WordProvenance, B32CompareKeepsTheAllPathNeighborPhi) {
    values({0xbead0380u, 0xbf068004u, 0xbf840001u, 0xbead0380u, 0x7e0c0280u, 0x7d8400f9u,
            0x0606ac06u, 0x7e0c022du, 0x7e580506u},
           0);
}
TEST(Wave32WordProvenance, B32CarrySdstCannotClearAStoredMissingNeighbor) {
    refused({0xbead033cu, 0x7e0c0280u, 0xd70f2c07u, 0x00010080u, 0x7e0c022du, 0x7e580506u});
    EXPECT_NE(reason.find("pc=8"), std::string::npos) << reason;
}
TEST(Wave32WordProvenance, B32CarrySdstKeepsTheGenuineStoredNeighbor) {
    values(
        {0xbebc0387u, 0xbead033cu, 0x7e0c0280u, 0xd70f2c07u, 0x00010080u, 0x7e0c022du, 0x7e580506u},
        7);
}
TEST(Wave32WordProvenance, B32CarrySdstAllowsAGenuineCurrentNeighborReplacement) {
    values(
        {0xbead033cu, 0x7e0c0280u, 0xd70f2c07u, 0x00010080u, 0xbead0387u, 0x7e0c022du, 0x7e580506u},
        7);
}

// The CPU instance profile must observe the actual group and global-id inputs, rather
// than executing workgroup zero twice. These controls do not alter production admission.
TEST(Wave32WordProvenance, Native32OracleChecksBothActualWorkgroupIds) {
    const auto module = compile({0xbeac0304u}, true);   // current s44 = genuine GroupIdX s4
    ASSERT_FALSE(module.empty()) << reason;
    for (uint32_t group : {0u, 1u}) {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 32;
        vm.native32_workgroup_count = 2;
        const auto actual = vm.run_buffers(32, {{3, Words(64, 0xdeadbeefu)}}, 3, group);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(actual.size(), 64u);
        for (uint32_t lane = 0; lane < 32; ++lane)
            EXPECT_EQ(actual[lane], group == 1 && lane == 0 ? 1u : 0u) << group << ':' << lane;
    }
}

TEST(Wave32WordProvenance, Native32OracleSuppliesGlobalInvocationId) {
    // Independent hand-built SPIR-V control: sink[GlobalInvocationId.x] = 1.
    // No compiler-generated address expression can hide an incorrect supplied GID.
    Words module{0x07230203u, 0x00010300u, 0, 19, 0};
    const auto instruction = [&](uint32_t op, std::initializer_list<uint32_t> operands) {
        module.push_back((static_cast<uint32_t>(operands.size() + 1) << 16) | op);
        module.insert(module.end(), operands);
    };
    instruction(17, {1});   // Shader capability
    instruction(14, {0, 1});   // Logical GLSL450 memory model
    instruction(15, {5, 10, 0x6e69616du, 0, 8});   // GLCompute main, Input interface
    instruction(16, {10, 17, 32, 1, 1});
    instruction(71, {8, 11, 28});   // GlobalInvocationId
    instruction(71, {9, 34, 0});
    instruction(71, {9, 33, 3});
    instruction(71, {4, 6, 4});
    instruction(71, {5, 2});
    instruction(72, {5, 0, 35, 0});
    instruction(19, {1});
    instruction(21, {2, 32, 0});
    instruction(23, {3, 2, 3});
    instruction(29, {4, 2});
    instruction(30, {5, 4});
    instruction(32, {6, 1, 3});
    instruction(32, {7, 12, 5});
    instruction(32, {11, 12, 2});
    instruction(33, {12, 1});
    instruction(43, {2, 13, 0});
    instruction(43, {2, 14, 1});
    instruction(59, {6, 8, 1});
    instruction(59, {7, 9, 12});
    instruction(54, {1, 10, 0, 12});
    instruction(248, {15});
    instruction(61, {3, 16, 8});
    instruction(81, {2, 17, 16, 0});
    instruction(65, {11, 18, 9, 13, 17});
    instruction(62, {18, 14});
    instruction(253, {});
    instruction(56, {});
    for (uint32_t group : {0u, 1u}) {
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 32;
        vm.native32_workgroup_count = 2;
        const auto actual = vm.run_buffers(32, {{3, Words(64, 0xdeadbeefu)}}, 3, group);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        ASSERT_EQ(actual.size(), 64u);
        for (uint32_t index = 0; index < 64; ++index)
            EXPECT_EQ(actual[index], index / 32 == group ? 1u : 0xdeadbeefu)
                << group << ':' << index;
    }
}

TEST(Wave32WordProvenance, Native32OracleRefusesMissingOrOutOfRangeInstanceAuthority) {
    const auto module = compile({0xbeac03c1u}, true);
    ASSERT_FALSE(module.empty()) << reason;
    for (uint32_t width : {0u, 32u, 64u}) {
        for (uint32_t groups : {0u, 1u, 2u}) {
            if (width != 32 && groups == 0) continue;   // legacy Wave64 is a separate profile
            for (uint32_t count : {31u, 32u, 64u}) {
                bpermute_oracle::Interpreter vm(module);
                vm.native_subgroup_width = width;
                vm.native32_workgroup_count = groups;
                // Group 2 lies outside both bounded extents, and has no legacy authority.
                EXPECT_TRUE(vm.run_buffers(count, {{3, Words(96, 0xdeadbeefu)}}, 3, 2).empty());
                EXPECT_FALSE(vm.error.empty());
            }
        }
    }
    bpermute_oracle::Interpreter missing(module);
    missing.native_subgroup_width = 32;
    EXPECT_TRUE(missing.run_buffers(32, {{3, Words(64, 0xdeadbeefu)}}, 3, 1).empty());
    EXPECT_EQ(missing.error, "unproved native32 instance");
    for (uint32_t count : {31u, 64u}) {
        bpermute_oracle::Interpreter wrong_count(module);
        wrong_count.native_subgroup_width = 32;
        wrong_count.native32_workgroup_count = 2;
        EXPECT_TRUE(wrong_count.run_buffers(count, {{3, Words(64, 0xdeadbeefu)}}, 3, 1).empty());
        EXPECT_EQ(wrong_count.error, "unproved native32 instance");
    }
    bpermute_oracle::Interpreter overflow(module);
    overflow.native_subgroup_width = 32;
    overflow.native32_workgroup_count = UINT32_MAX;
    EXPECT_TRUE(
        overflow.run_buffers(32, {{3, Words(64, 0xdeadbeefu)}}, 3, UINT32_MAX / 32 + 1).empty());
    EXPECT_EQ(overflow.error, "unproved native32 instance");
}

TEST(Wave32WordProvenance, Native32OracleRefusesUnmatchedEntrypointLocalSize) {
    const auto original = compile({0xbeac03c1u}, true);
    ASSERT_FALSE(original.empty()) << reason;
    size_t local = 0;
    for (size_t pc = 5; pc < original.size(); pc += original[pc] >> 16) {
        if ((original[pc] & 0xffffu) == 16 && original[pc] >> 16 == 6 && original[pc + 2] == 17) {
            ASSERT_EQ(local, 0u);
            local = pc;
        }
    }
    ASSERT_NE(local, 0u);
    for (uint32_t variant = 0; variant < 8; ++variant) {
        auto module = original;
        if (variant == 0)
            module.erase(module.begin() + local, module.begin() + local + 6);
        else if (variant == 1)
            module[local + 3] = 64;   // wrong invocation count
        else if (variant == 2) {
            module[local + 3] = 16;
            module[local + 4] = 2;   // same count, unsupported multidimensional geometry
        } else if (variant == 3)
            module[local + 5] = 2;
        else if (variant == 4)
            module[local + 1] = 0;   // another function's declaration supplies no entry authority
        else if (variant == 5) {
            const Words duplicate(original.begin() + local, original.begin() + local + 6);
            module.insert(module.begin() + local, duplicate.begin(), duplicate.end());
        } else if (variant == 6) {
            // Specialization-dependent geometry lies outside this literal-size profile.
            const Words specialized{0x0006014bu, original[local + 1], 38, 1, 1, 1};
            module.insert(module.begin() + local, specialized.begin(), specialized.end());
        } else {
            // WorkgroupSize takes precedence over LocalSize; never use the latter as proof.
            const Words override_size{0x00040047u, 0, 11, 25};
            module.insert(module.begin() + local, override_size.begin(), override_size.end());
        }
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 32;
        vm.native32_workgroup_count = 2;
        EXPECT_TRUE(vm.run_buffers(32, {{3, Words(64, 0xdeadbeefu)}}, 3, 1).empty()) << variant;
        EXPECT_FALSE(vm.error.empty()) << variant;
    }
}
