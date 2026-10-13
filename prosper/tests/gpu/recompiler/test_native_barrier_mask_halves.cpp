// #4966: a real guest barrier preserves the current physical EXEC dwords in WRITELANE slots.
// A persisted Bool/uint placeholder cannot authorize a reload or a later partial-route ballot.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/native_execution_launch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
constexpr uint32_t barrier = 0xbf8a0000u;
// Keep the physical half producers, both reloads, original sink and irreducible tail from
// CfgSpillSlotDomain.AnExecSpilledAcrossABarrierIsRestored. Supply only genuine v1/launch inputs.
constexpr std::array<uint32_t, 18> body{
    0x7e060280u, 0x7d8800a8u, 0xbefe046au, 0xd7610014u, 0x0001007eu, 0xd7610014u,
    0x0001027fu, 0xbefe04c1u, barrier,     0xd760000eu, 0x00010114u, 0xd760000fu,
    0x00010314u, 0xbefe040eu, 0x7e060281u, 0xbefe04c1u, 0xe0702000u, 0x80020300u};
constexpr std::array<uint32_t, 9> tail{0x7e040280u, 0x7c020300u, 0xbf860001u,
                                       0x7e040281u, 0x7d840100u, 0xbf870001u,
                                       0xbf82fffdu, 0x7e040d02u, 0xbf810000u};

std::vector<uint32_t> program() {
    std::vector<uint32_t> words(body.begin(), body.end());
    words.push_back(0x7e020280u); // genuine v1 before the unchanged tail
    words.insert(words.end(), tail.begin(), tail.end());
    return words;
}

std::vector<uint32_t>
compile(const std::vector<uint32_t>& words,
        const ComputeShaderConfig& config = prosper::test::native_execution_launch()) {
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    ShaderResourceTable resources;
    resources.resources.push_back(output);
    return recompile_compute(words.data(), words.size(), &resources, config,
                             {RecompileDiagnosticStage::Compute, 0x49660001u});
}

void values(const std::vector<uint32_t>& words, const std::array<uint32_t, 64>& expected,
            const ComputeShaderConfig& config = prosper::test::native_execution_launch()) {
    const auto module = compile(words, config);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0x49660001u);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = config.user_sgprs;
    const auto actual = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(actual.size(), expected.size());
    for (uint32_t lane = 0; lane < expected.size(); ++lane)
        EXPECT_EQ(actual[lane], expected[lane]) << "physical lane=" << lane;
}

std::array<uint32_t, 64> first_lanes(uint32_t count) {
    std::array<uint32_t, 64> expected{};
    for (uint32_t lane = 0; lane < count; ++lane) expected[lane] = 1;
    return expected;
}
} // namespace

TEST(NativeBarrierMaskHalves, OriginalBarrierRetainsBothPhysicalWordsAndExactSink) {
    values(program(), first_lanes(40));
}

TEST(NativeBarrierMaskHalves, RemovingOnlyTheBarrierPreservesTheSameValueControl) {
    auto words = program();
    words.erase(words.begin() + 8);
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, TwoConsecutiveRealBarriersPreserveCurrentSlots) {
    auto words = program();
    words.insert(words.begin() + 8, barrier);
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, RawLaunchPresenceRemainsRequired) {
    auto config = prosper::test::native_execution_launch();
    config.dispatch_launch.known = 0;
    EXPECT_TRUE(compile(program(), config).empty());
    EXPECT_NE(last_terminal_reject_reason(0x49660001u).find("pc=9 words=d760000e,00010114"),
              std::string::npos);
    values(program(), first_lanes(40));
}

TEST(NativeBarrierMaskHalves, NonemptyIntermediatePhasePreservesCurrentHalves) {
    auto words = program();
    words.insert(words.begin() + 9, {0xbf800000u, barrier});
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, LowPhysicalWordIsAvailableAsNumericData) {
    auto words = program();
    words[13] = 0xbefe04c1u; // full EXEC for the numeric observation
    words[14] = 0x4a06000eu; // v3 = s14 + local_x
    std::array<uint32_t, 64> expected{};
    for (uint32_t lane = 0; lane < expected.size(); ++lane) expected[lane] = UINT32_MAX + lane;
    values(words, expected);
}

TEST(NativeBarrierMaskHalves, HighPhysicalWordIsAvailableAsNumericData) {
    auto words = program();
    words[13] = 0xbefe04c1u;
    words[14] = 0x4a06000fu; // v3 = s15 + local_x; mask40 has HI = 0xff
    std::array<uint32_t, 64> expected{};
    for (uint32_t lane = 0; lane < expected.size(); ++lane) expected[lane] = 0xffu + lane;
    values(words, expected);
}

TEST(NativeBarrierMaskHalves, GenuineHalfAliasReSpillsKeepTheirPhysicalIdentity) {
    auto words = program();
    words[10] = 0x00010115u; // final reloads use v21
    words[12] = 0x00010315u;
    words.insert(words.begin() + 8, {0xd760000eu, 0x00010114u, 0xd760000fu, 0x00010314u,
                                     0xd7610015u, 0x0001000eu, 0xd7610015u, 0x0001020fu});
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, AFabricatedScalarCannotBecomeAGenuineHalfByReSpilling) {
    auto words = program();
    words[10] = 0x00010115u;
    words[12] = 0x00010315u;
    words.insert(words.begin() + 8, {0xd760000eu, 0x00010114u, 0xd760000fu, 0x00010314u,
                                     0xd7610015u, 0x00010010u, 0xd7610015u, 0x0001020fu});
    EXPECT_TRUE(compile(words).empty()); // absent s16 cannot authenticate v21 lane0
    words[13] = 0x0001000eu;           // only the re-spill source becomes genuine s14
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, ADynamicWritelaneCannotGainASupportedSelectorFromTheHandoff) {
    auto words = program();
    // s18=0 dynamically addresses v20; all historical slots die before the fresh named LO.
    words.insert(words.begin() + 8,
                 {0xbe920380u, 0xd7610014u, 0x00002480u, 0xd7610014u, 0x0001007eu});
    EXPECT_TRUE(compile(words).empty());
    EXPECT_NE(last_terminal_reject_reason(0x49660001u).find("pc=9 words=d7610014,00002480"),
              std::string::npos); // dynamic WRITELANE remains a separate unsupported emitter
    words[10] = 0x00010080u;     // only the selector becomes the supported static lane0
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, AFullVectorOverwriteThenOnlyLowSpillCannotReviveHigh) {
    auto words = program();
    words.insert(words.begin() + 8, {0x7e280280u, 0xd7610014u, 0x0001007eu});
    EXPECT_TRUE(compile(words).empty());
    words.insert(words.begin() + 11, {0xd7610014u, 0x0001027fu});
    values(words, first_lanes(64));
}

TEST(NativeBarrierMaskHalves, DifferentHistoricalHalfValuesRemainIndependent) {
    auto words = program();
    words.insert(words.begin() + 5, 0xbefe04c1u); // LO from mask40, HI from full EXEC
    values(words, first_lanes(64));
}

TEST(NativeBarrierMaskHalves, SwappedPhysicalWordsCannotReuseTheOrderedBoolAlias) {
    auto words = program();
    words[10] = 0x00010314u;
    words[12] = 0x00010114u;
    std::array<uint32_t, 64> expected{};
    for (uint32_t lane = 0; lane < expected.size(); ++lane)
        expected[lane] = lane < 8 || lane >= 32 ? 1u : 0u;
    values(words, expected);
}

TEST(NativeBarrierMaskHalves, AFullOverwriteThenGenuineNewSpillsEndsTheOldGeneration) {
    auto words = program();
    words.insert(words.begin() + 8,
                 {0x7e280280u, 0xd7610014u, 0x0001007eu, 0xd7610014u, 0x0001027fu});
    values(words, first_lanes(64));
}

TEST(NativeBarrierMaskHalves, MissingLowOrHighSpillDoesNotGainAPlaceholderDefinition) {
    for (uint32_t at : {3u, 5u}) {
        auto words = program();
        words.erase(words.begin() + at, words.begin() + at + 2);
        EXPECT_TRUE(compile(words).empty()) << "missing physical spill at original dword=" << at;
        EXPECT_NE(
            last_terminal_reject_reason(0x49660001u)
                .find(at == 3 ? "pc=7 words=d760000e,00010114" : "pc=9 words=d760000f,00010314"),
            std::string::npos);
    }
    values(program(), first_lanes(40));
}

TEST(NativeBarrierMaskHalves, OrdinaryVgprOverwriteCannotReviveHistoricalSlots) {
    auto words = program();
    words.insert(words.begin() + 8, 0x7e280280u); // v20=0 retires both physical half slots
    EXPECT_TRUE(compile(words).empty());
    EXPECT_NE(
        last_terminal_reject_reason(0x49660001u).find("pc=10 reason=invalidated-vgpr-lane-slot"),
        std::string::npos);
    words.erase(words.begin() + 8);
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, APostBarrierBranchCannotAuthorizeALatePartialBallot) {
    auto words = program();
    // Native READFIRST's generic spelling is lane-local here. Its scalar branch could route
    // only lanes0..31 into the two reloads. A saved half tag cannot prove ballot participation.
    words.insert(words.begin() + 9, {0x7e240500u, 0xbf0aa012u, 0xbf840006u});
    EXPECT_TRUE(compile(words).empty());
    words[11] = 0xbf800000u;   // removing only the branch restores physical participation
    values(words, first_lanes(40));
}

TEST(NativeBarrierMaskHalves, PartialOrShortRawLaunchCannotInheritTheHandoff) {
    auto config = prosper::test::native_execution_launch();
    config.dispatch_launch.initiator |= 2u;
    EXPECT_TRUE(compile(program(), config).empty());
    EXPECT_NE(last_terminal_reject_reason(0x49660001u).find("pc=9 words=d760000e,00010114"),
              std::string::npos);
    config = prosper::test::native_execution_launch();
    config.exact_thread_extent = true;
    config.threads_x = 33;
    EXPECT_TRUE(compile(program(), config).empty());
    EXPECT_NE(last_terminal_reject_reason(0x49660001u).find("pc=13 words=befe040e"),
              std::string::npos);
    values(program(), first_lanes(40));
}

TEST(NativeBarrierMaskHalves, GenuineNumericSpillRemainsAUsableScalarBranchCondition) {
    std::vector<uint32_t> words{0x7e060280u, 0x7e280280u, 0xd7610014u, 0x00010081u, barrier,
                                0xd760000eu, 0x00010114u, 0xbf06810eu, 0xbf850001u, 0x7e060282u,
                                0xe0702000u, 0x80020300u, 0x7e020280u};
    auto numeric_tail = tail;
    // This separate SCC handoff control uses the genuine entry v0 in both operands. The old
    // dispatcher otherwise loses the new v1 definition past this SCC edge (#4972), obscuring
    // whether the numeric spill condition itself survived. The real mask test retains its tail.
    numeric_tail[1] = 0x7c020100u;
    words.insert(words.end(), numeric_tail.begin(), numeric_tail.end());
    values(words, first_lanes(0));
    words.erase(words.begin() + 4);
    values(words, first_lanes(0));
}

TEST(NativeBarrierMaskHalves, NumericReSpillAcrossASecondBarrierRequiresItsGenuineScalar) {
    std::vector<uint32_t> words{0x7e060280u, 0x7e280280u, 0xd7610014u, 0x00010081u, barrier,
                                0xd760000eu, 0x00010114u, 0xd7610015u, 0x00010010u, barrier,
                                0xd760000eu, 0x00010115u, 0xbf06810eu, 0xbf850001u, 0x7e060282u,
                                0xe0702000u, 0x80020300u, 0x7e020280u};
    auto numeric_tail = tail;
    numeric_tail[1] = 0x7c020100u;   // same independent SCC fixture's genuine entry operands
    words.insert(words.end(), numeric_tail.begin(), numeric_tail.end());
    EXPECT_TRUE(compile(words).empty());   // absent s16 remains absent across the second phase
    words[8] = 0x0001000eu;   // only the numeric re-spill source becomes real s14
    values(words, first_lanes(0));
}

TEST(NativeBarrierMaskHalves, AOneEdgeMaskOverwriteCannotBorrowAnImportedNumericDomain) {
    auto words = program();
    words[4] = 0x00010087u;   // original phase: v20 lane0 contains genuine DATA 7
    words.erase(words.begin() + 11, words.begin() + 13);   // this control never consumes HI
    words[11] = 0xbefe04c1u;
    words[12] = 0x4a06000eu;   // numeric observation s14 + local_x, with full EXEC
    words[18] = 0x7c020100u;   // independent SCC fixture uses genuine entry v0 in its tail
    words.insert(words.begin() + 9, {0xbf068012u, 0xbf840002u, 0xd7610014u, 0x0001007eu});
    auto config = prosper::test::native_execution_launch();
    config.user_sgprs.resize(19, 0);
    EXPECT_TRUE(compile(words, config).empty());   // one edge writes a mask over the DATA slot
    words[12] = 0x00010087u;   // only that source becomes genuine DATA 7 on both paths
    std::array<uint32_t, 64> expected{};
    for (uint32_t lane = 0; lane < expected.size(); ++lane) expected[lane] = lane + 7;
    for (uint32_t s18 : {0u, 1u}) {
        config.user_sgprs[18] = s18;
        values(words, expected, config);
    }
}
}   // namespace prosper::gpu
