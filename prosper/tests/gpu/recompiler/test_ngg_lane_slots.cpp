// #4912: an NGG scalar spill initializes one physical lane, never the launch VGPR's other peers.
// Portable shell modules execute against owned launch/export words in the fail-closed CPU
// oracle. Native modules compile and validate; no native topology or GPU execution is inferred.
#include "gpu/recompiler/ngg_subgroup_abi.hpp"
#include "gpu/recompiler/ngg_subgroup_shell.hpp"
#include "fixtures/ngg_lane_slots.hpp"
#include "bpermute_spirv_oracle.hpp"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace f = prosper::test::ngg_slots;
namespace {
NggSubgroupAbiFacts analyze(const std::vector<uint32_t>& code, bool wave32 = false) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    NggSubgroupAbiLaunch launch;
    launch.wave32 = wave32;
    return analyze_ngg_subgroup_abi(ins, launch);
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, bool native = false,
                              uint32_t lanes = 64, NggExportRecordLayout* layout = nullptr) {
    static std::atomic<uint64_t> serial{0};
    NggSubgroupShellConfig config;
    config.waves = 4;
    config.wave_lanes = lanes;
    config.native_wave64 = native;
    std::string refusal;
    const auto module = recompile_ngg_subgroup(
        code.data(), code.size(), nullptr, config, layout,
        {RecompileDiagnosticStage::Vertex, 0x49120000u + 4096u * serial.fetch_add(1)}, &refusal);
    EXPECT_FALSE(module.empty()) << refusal;
    return module;
}
std::vector<uint32_t> launch(uint32_t lanes) {
    std::vector<uint32_t> words(size_t(lanes) * 4u * kNggLaunchWordsPerLane);
    for (uint32_t thread = 0; thread < lanes * 4u; ++thread) {
        const size_t first = size_t(thread) * kNggLaunchWordsPerLane;
        for (uint32_t reg = 0; reg < 9; ++reg) words[first + reg] = 0xdead0000u + thread + reg;
        words[first + kNggLaunchS3Word] = 0x40000000u | ((thread / lanes) << 24) | lanes;
    }
    return words;
}
void expect_values(const std::vector<uint32_t>& code, const std::array<uint32_t, 4>& wanted,
                   uint32_t lanes = 64) {
    for (bool native : {false, true}) {
        SCOPED_TRACE(native);
        NggExportRecordLayout layout;
        const auto module = compile(code, native, lanes, &layout);
        ASSERT_FALSE(module.empty());
        // The native envelope needs SubgroupId/SubgroupLocalInvocationId, which this CPU
        // interpreter does not supply. Keep actual native compilation separate from values.
        if (native) continue;
        bpermute_oracle::Interpreter oracle(module);
        oracle.use_ngg_shell_storage();
        const auto output = oracle.run_buffers(
            lanes * 4u,
            {{kNggShellLaunchBinding, launch(lanes)},
             {kNggShellExportBinding, std::vector<uint32_t>(layout.block_words(4, lanes))}},
            kNggShellExportBinding);
        ASSERT_TRUE(oracle.error.empty()) << oracle.error;
        ASSERT_EQ(output.size(), layout.block_words(4, lanes));
        EXPECT_EQ(output[kNggHeaderVertsAlloc], 5u);
        EXPECT_EQ(output[kNggHeaderPrimsAlloc], 3u);
        EXPECT_EQ(output[kNggHeaderAllocRequests], 1u);
        // This deliberate four-wave fixture sends from every wave. Its raw exports are an
        // emitter oracle; the raster consumer would reject these three stray requests.
        EXPECT_EQ(output[kNggHeaderStrayRequests], 3u);
        EXPECT_EQ(output[kNggHeaderLaunchMismatches], 0u);
        for (uint32_t thread = 0; thread < lanes * 4u; ++thread) {
            const uint32_t record = layout.record_offset(4, 0, thread, lanes);
            ASSERT_EQ(output[record], kNggFlagPrim | kNggFlagPos0) << thread;
            ASSERT_EQ(output[record + kNggRecordPrimWord], wanted[thread / lanes]) << thread;
            for (uint32_t channel = 0; channel < 4; ++channel)
                ASSERT_EQ(output[record + kNggRecordPos0Word + channel],
                          0xdead0000u + thread + channel)
                    << thread;
        }
    }
}
std::vector<uint32_t> body_program(std::initializer_list<uint32_t> body) {
    auto code = f::begin();
    f::scalar(code, 20, f::kFirst);
    f::write(code);
    f::append(code, body);
    f::read(code);
    f::finish(code);
    return code;
}
void refused_peer(const std::vector<uint32_t>& code, uint32_t pc) {
    const auto facts = analyze(code);
    EXPECT_EQ(facts.reason, "ngg-abi-read-v4") << facts.refusal;
    EXPECT_NE(facts.refusal.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << facts.refusal;
}
}   // namespace

TEST(NggLaneSlots, ConstantSpillAndReloadExecuteEveryWave) {
    expect_values(f::straight(), {f::kFirst, f::kFirst, f::kFirst, f::kFirst});
}
TEST(NggLaneSlots, EachUnprovidedLaunchRegisterHasSeparatePeers) {
    for (uint32_t reg : {4u, 6u, 7u}) {
        SCOPED_TRACE(reg);
        expect_values(f::straight(reg), {f::kFirst, f::kFirst, f::kFirst, f::kFirst});
    }
}
TEST(NggLaneSlots, WritingOnePeerDoesNotInitializeItsNeighbour) {
    auto code = f::straight();
    code[7] = (256u + 4u) | (136u << 9);   // READLANE lane8, not the written lane7
    refused_peer(code, 6);
    EXPECT_TRUE(analyze(f::straight()).ok());
}
TEST(NggLaneSlots, SlotWriteDoesNotInitializeAWholeVgpr) {
    const auto bad = body_program({0x7e120304u});   // ordinary v_mov v9,v4
    refused_peer(bad, 6);
    const auto good = body_program({0x7e120300u});   // supplied v0 is ordinary data
    EXPECT_TRUE(analyze(good).ok());
    EXPECT_FALSE(compile(good).empty());
}
TEST(NggLaneSlots, MultiplePeersPreserveTheirOwnValues) {
    auto code = f::begin();
    f::scalar(code, 20, f::kFirst);
    f::write(code);
    f::scalar(code, 20, f::kSecond);
    f::write(code, 4, 8);
    f::read(code);
    f::finish(code);
    expect_values(code, {f::kFirst, f::kFirst, f::kFirst, f::kFirst});
}
TEST(NggLaneSlots, RewritingOnePeerReplacesOnlyItsValue) {
    auto code = f::begin();
    f::scalar(code, 20, f::kFirst);
    f::write(code);
    f::scalar(code, 20, f::kSecond);
    f::write(code);
    f::read(code);
    f::finish(code);
    expect_values(code, {f::kSecond, f::kSecond, f::kSecond, f::kSecond});
}
TEST(NggLaneSlots, ScalarSlotOperationsIgnoreEmptyExec) {
    auto code = f::straight();
    code.insert(code.begin() + 4, 0xbefe0480u);   // EXEC=0 before WRITELANE and READLANE
    code.insert(code.begin() + 9, f::kExecFull);   // restore before observable exports
    expect_values(code, {f::kFirst, f::kFirst, f::kFirst, f::kFirst});
}
TEST(NggLaneSlots, CanonicalSelectorsRespectActualWaveWidth) {
    expect_values(f::straight(4, 31), {f::kFirst, f::kFirst, f::kFirst, f::kFirst}, 32);
    expect_values(f::straight(4, 63), {f::kFirst, f::kFirst, f::kFirst, f::kFirst});
    EXPECT_EQ(analyze(f::straight(4, 32), true).reason, "ngg-abi-read-v4");
    EXPECT_TRUE(analyze(f::straight(4, 32)).ok());
}
TEST(NggLaneSlots, AReadDoesNotWrapAnUnsupportedWave32SlotName) {
    auto code = f::straight(4, 0);
    code[7] = (256u + 4u) | (160u << 9);   // lane32 is not the emitter's lane0 name
    EXPECT_EQ(analyze(code, true).reason, "ngg-abi-read-v4");
    EXPECT_TRUE(analyze(f::straight(4, 0), true).ok());
}
TEST(NggLaneSlots, UnknownScalarSourceDoesNotAcquireSlotAuthority) {
    auto code = f::straight();
    code[5] = 30u | (135u << 9);   // s30 was never written
    const auto facts = analyze(code);
    EXPECT_EQ(facts.reason, "ngg-abi-read-undefined-sgpr");
    EXPECT_EQ(facts.refused_sgpr, 30);
    EXPECT_TRUE(analyze(f::straight()).ok());
}
TEST(NggLaneSlots, DynamicSelectorDoesNotBorrowAConstantPeer) {
    auto code = f::straight();
    code[7] = (256u + 4u) | (20u << 9);   // dynamic selector s20
    refused_peer(code, 6);
    EXPECT_TRUE(analyze(f::straight()).ok());
}
TEST(NggLaneSlots, NoncanonicalLaneWriteKeepsItsOldDestinationCharge) {
    auto code = f::straight();
    code[4] |= 0x8000u;   // unsupported CLAMP modifier
    EXPECT_FALSE(analyze(code).ok());
    EXPECT_TRUE(analyze(f::straight()).ok());
}
TEST(NggLaneSlots, BothScalarArmsCarryTheSelectedSlotValue) {
    expect_values(f::joined(), {f::kFirst, f::kSecond, f::kFirst, f::kSecond});
}
TEST(NggLaneSlots, OneArmCannotInventTheOtherArmsSlot) {
    const auto bad = analyze(f::joined(false));
    EXPECT_EQ(bad.reason, "ngg-abi-read-v4") << bad.refusal;
    EXPECT_TRUE(analyze(f::joined()).ok());
}
TEST(NggLaneSlots, LoopCarriesSlotValueAcrossEveryBackEdge) {
    expect_values(f::counted(), {f::kFirst + 3u, f::kFirst + 3u, f::kFirst + 3u, f::kFirst + 3u});
}
TEST(NggLaneSlots, LoopBodyWriteDoesNotSeedTheFirstHeaderRead) {
    const auto bad = analyze(f::counted(false));
    EXPECT_EQ(bad.reason, "ngg-abi-read-v4") << bad.refusal;
    EXPECT_TRUE(analyze(f::counted()).ok());
}
TEST(NggLaneSlots, FullOrdinaryOverwriteEndsTheNamedLifetime) {
    auto code = body_program({0x7e080214u});   // v_mov_b32 v4,s20 under full EXEC
    expect_values(code, {f::kFirst, f::kFirst, f::kFirst, f::kFirst}, 64);
}
TEST(NggLaneSlots, PartialOrdinaryOverwriteCannotLaunderAnInactivePeer) {
    const auto code = body_program({0xbefe04ffu, 1u, 0x7e080214u});   // EXEC=1; v4=s20
    refused_peer(code, 9);
    EXPECT_TRUE(analyze(body_program({0x7e080214u})).ok());
}
TEST(NggLaneSlots, FullOverwriteCannotResurrectAnEarlierDifferentSlot) {
    const auto code = body_program({0xbe9403ffu, f::kSecond, 0x7e080214u});
    // This read is ordinary data after overwrite, so it yields the actual new value.
    expect_values(code, {f::kSecond, f::kSecond, f::kSecond, f::kSecond}, 64);
}
TEST(NggLaneSlots, OrdinaryReadUnderCurrentExecStillRequiresARealDefinition) {
    const auto bad = body_program({0xbefe04ffu, 1u, 0x7e120304u});   // narrow then whole v4 read
    refused_peer(bad, 8);
    EXPECT_TRUE(analyze(body_program(
                            {0xbefe04ffu, 1u, 0x7e080214u, 0x7e120304u, f::kExecFull, 0x7e080214u}))
                    .ok());
}

TEST(NggLaneSlots, SavedExecCannotResurrectWholeVgprAuthorityAfterASpill) {
    auto code = f::begin();
    f::scalar(code, 20, f::kFirst);
    code.push_back(0x7e080214u);   // full ordinary v4
    code.push_back(0xbe98047eu);   // s_mov_b64 s[24:25],exec
    f::write(code);
    code.push_back(0xbefe0418u);   // restore saved EXEC
    const uint32_t consumer = static_cast<uint32_t>(code.size());
    code.push_back(0x7e120304u);   // whole v4 cannot reappear from saved current-EXEC facts
    f::read(code);
    f::finish(code);
    refused_peer(code, consumer);
    code[consumer] = 0x7e120300u;
    EXPECT_TRUE(analyze(code).ok());
}

TEST(NggLaneSlots, NamedLifetimeMayFactSurvivesAFullPartialBranchJoin) {
    auto code = f::begin();
    f::scalar(code, 20, f::kFirst);
    f::write(code);
    f::append(code, {0x90169803u, 0x87168116u, 0xbf068016u});
    const size_t to_partial = f::branch(code, 0x04);
    code.push_back(0x7e080214u);   // full write retires named lifetime on this arm only
    f::append(code, {0xbefe04ffu, 1u});
    const size_t to_join = f::branch(code, 0x02);
    f::target(code, to_partial, code.size());
    f::append(code, {0xbefe04ffu, 1u, 0x7e080214u});   // only lane0 is defined on other arm
    f::target(code, to_join, code.size());
    const uint32_t consumer = static_cast<uint32_t>(code.size());
    f::read(code);
    f::finish(code);
    refused_peer(code, consumer);
    // A fresh full overwrite after the join proves every ordinary peer again.
    code.insert(code.begin() + consumer, {f::kExecFull, 0x7e080214u});
    EXPECT_TRUE(analyze(code).ok());
}

TEST(NggLaneSlots, PartialLoopOverwriteCannotKeepItsPreheaderSlot) {
    auto code = f::begin();
    f::scalar(code, 20, f::kFirst);
    f::write(code);
    f::scalar(code, 22, 3);
    const uint32_t header = static_cast<uint32_t>(code.size());
    f::read(code);
    f::append(code, {0xbefe04ffu, 1u, 0x7e080214u, f::kExecFull, 0x80968116u, 0xbf068016u});
    const size_t again = f::branch(code, 0x04);
    f::target(code, again, header);
    f::read(code);
    f::finish(code);
    refused_peer(code, header);
    EXPECT_TRUE(analyze(f::counted()).ok());
}

TEST(NggLaneSlots, AnOverlappingAppendedTfeStatusRetiresThePeer) {
    auto code = f::begin();
    f::scalar(code, 20, f::kFirst);
    f::write(code);
    for (uint32_t reg = 16; reg < 20; ++reg) f::scalar(code, reg, 0);
    f::append(code, {0xe0300000u, 0x80840300u});   // BUFFER_LOAD_DWORD v3, status=v4
    const uint32_t consumer = static_cast<uint32_t>(code.size());
    f::read(code);
    f::finish(code);
    refused_peer(code, consumer);
    code[consumer - 1] &= ~(1u << 23);   // no status result overlaps v4
    EXPECT_TRUE(analyze(code).ok());
}

TEST(NggLaneSlots, PartialCrossLaneReadDoesNotBorrowAFormerSlot) {
    const auto code = body_program(
        {0xbefe04ffu, 1u, 0x7e080214u, 0xdacc0000u, 0x00000400u});   // DS_BPERMUTE v0,v0,v4
    const auto facts = analyze(code);
    EXPECT_EQ(facts.reason, "ngg-abi-read-v4") << facts.refusal;
    EXPECT_NE(facts.refusal.find("pc=9 "), std::string::npos) << facts.refusal;
    EXPECT_TRUE(analyze(body_program({0x7e080214u})).ok());
}

TEST(NggLaneSlots, OracleDefaultProfileRejectsActualSetTwoStorage) {
    NggExportRecordLayout layout;
    const auto module = compile(f::straight(), false, 64, &layout);
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter oracle(module);
    EXPECT_TRUE(
        oracle
            .run_buffers(256, {{0, launch(64)}, {1, std::vector<uint32_t>(layout.block_words(4))}},
                         1)
            .empty());
    EXPECT_EQ(oracle.error, "unsupported live storage ABI");
}
TEST(NggLaneSlots, OracleRejectsWrongAtomicScopeOrSemantics) {
    NggExportRecordLayout layout;
    const auto original = compile(f::straight(), false, 64, &layout);
    ASSERT_FALSE(original.empty());
    for (uint32_t operand : {4u, 5u}) {
        auto module = original;
        bool found = false;
        for (size_t pc = 5; pc < module.size(); pc += module[pc] >> 16) {
            if ((module[pc] & 0xffffu) != 234) continue;
            ASSERT_EQ(module[pc] >> 16, 7u);
            module[pc + operand] = module[pc + (operand == 4 ? 5u : 4u)];
            found = true;
        }
        ASSERT_TRUE(found);
        bpermute_oracle::Interpreter oracle(module);
        oracle.use_ngg_shell_storage();
        EXPECT_TRUE(
            oracle
                .run_buffers(
                    256, {{0, launch(64)}, {1, std::vector<uint32_t>(layout.block_words(4))}}, 1)
                .empty());
        EXPECT_EQ(oracle.error, "unsupported AtomicIAdd profile");
    }
}

TEST(NggLaneSlots, OracleRejectsNativeModuleWithoutTopologyInputs) {
    NggExportRecordLayout layout;
    const auto module = compile(f::straight(), true, 64, &layout);
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter oracle(module);
    oracle.use_ngg_shell_storage();
    EXPECT_TRUE(
        oracle
            .run_buffers(256, {{0, launch(64)}, {1, std::vector<uint32_t>(layout.block_words(4))}},
                         1)
            .empty());
    EXPECT_EQ(oracle.error, "unsupported builtin");
}
