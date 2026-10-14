// Current-EXEC DATA must survive an exact low/high lane spill and reload of that same EXEC.
// Losing its domain refuses genuine comparisons before the saved FF1/BCNT consumer is reached.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "fixtures/compute_coverage_inputs.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_exec_spill_equality.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
enum class Reduction { Ff1, Bcnt };
std::vector<uint32_t> persisted(Reduction reduction) {
    // These are the complete unchanged legacy fixtures, including the real mask producer,
    // irreducible tail and final reduction. Neither normalized geometry nor a guessed mask is
    // substituted for their actual inputs.
    const bool ff1 = reduction == Reduction::Ff1;
    std::vector<uint32_t> code =
        ff1 ? std::vector<uint32_t>{0xbe900381u, 0xbe910382u, 0x7e060210u, 0xbe8203a8u,
                                    0x7d840002u, 0xbefe046au, 0xbe90246au}
            : std::vector<uint32_t>{0xbe860381u, 0xbe870382u, 0x7e060206u, 0xbe8203a8u,
                                    0x7d840002u, 0xbefe046au, 0xbe86246au};
    const auto& tail = prosper::test::coverage_inputs::dispatch_tail;
    code.insert(code.end(), std::begin(tail), std::end(tail) - 1);
    code.insert(
        code.end(),
        ff1 ? std::initializer_list<uint32_t>{0xbefe0410u, 0xbeea1410u, 0x7e04026au, 0xbf810000u}
            : std::initializer_list<uint32_t>{0xbefe0406u, 0xbeea1006u, 0x7e04026au, 0xbf810000u});
    return code;
}
ComputeShaderConfig config() {
    ComputeShaderConfig c;
    c.wave_size = c.native_subgroup_size = 64;
    prosper::test::complete_compute_launch(c);
    return c;
}
ShaderResourceTable resources() {
    ShaderResourceTable table;
    ShaderResource resource{};
    resource.cls = ResourceClass::VertexBuffer;
    resource.binding = 3;
    resource.sgpr_base = 8;
    resource.stride = 16;
    resource.format = DataFormat::Float32;
    resource.num_components = 4;
    table.resources.push_back(resource);
    return table;
}
void retain(const std::vector<uint32_t>& module, uint64_t address) {
    if (module.empty()) return;
    // This CTest process reads a fixed launch environment and never mutates it.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    if (const char* directory = std::getenv("PROSPER_EXEC_SPILL_SPV_DIRECTORY")) {
        std::ofstream output(std::filesystem::path(directory) / (std::to_string(address) + ".spv"),
                             std::ios::binary);
        output.write(reinterpret_cast<const char*>(module.data()),
                     static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
        EXPECT_TRUE(output.good()) << "retain the actual translated SOURCE for strict validation";
    }
}
void genuine(Reduction reduction, uint64_t address) {
    const auto code = persisted(reduction);
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(code.data(), code.size(), instructions);
    bool comparison = false;
    for (const auto& in : instructions)
        if (in.pc == 24) {
            EXPECT_EQ(in.fmt, Rdna2Format::VOPC);
            EXPECT_EQ(in.opcode, 0x81u);
            EXPECT_EQ(in.src[1].kind, OperandKind::VGPR);
            EXPECT_EQ(in.src[1].value, 1);
            comparison = true;
        }
    ASSERT_TRUE(comparison);
    const auto table = resources();
    const auto module = recompile_compute(code.data(), code.size(), &table, config(),
                                          {RecompileDiagnosticStage::Compute, address});
    retain(module, address);
    EXPECT_FALSE(module.empty()) << last_terminal_reject_reason(address);
}
} // namespace

TEST(ExecSpillData, PersistedFf1RetainsTheGenuineNarrowWord) {
    genuine(Reduction::Ff1, 0x5127000010ull);
}
TEST(ExecSpillData, PersistedBcntRetainsTheGenuineNarrowWord) {
    genuine(Reduction::Bcnt, 0x5127000020ull);
}

namespace {
bool step(ExecSpillEquality& equality, std::initializer_list<uint32_t> words) {
    std::vector<Rdna2Inst> instructions;
    rdna2_walk(words.begin(), words.size(), instructions);
    if (instructions.size() != 1) {
        ADD_FAILURE() << "expected one actual decoded instruction";
        return false;
    }
    const auto& in = instructions.front();
    std::vector<std::pair<int, uint32_t>> scalars;
    std::vector<int> vectors;
    for_each_scalar_write(in, [&](int base, uint32_t width) { scalars.emplace_back(base, width); });
    for_each_possible_vector_write(in, 255, [&](int vector) { vectors.push_back(vector); });
    return advance_exec_spill_equality(equality, in, scalars, vectors, -1);
}
void low(ExecSpillEquality& state) {
    step(state, {0xd7610013u, 0x00014a7eu});
}
void high(ExecSpillEquality& state) {
    step(state, {0xd7610013u, 0x0001507fu});
}
void reload(ExecSpillEquality& state) {
    step(state, {0xd760000eu, 0x00014b13u});
    step(state, {0xd760000fu, 0x00015113u});
}
bool restore(ExecSpillEquality& state) {
    return step(state, {0xbefe040eu});
}
void empty(const std::vector<uint32_t>& code, uint64_t address) {
    const auto table = resources();
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, config(),
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
}
} // namespace

TEST(ExecSpillData, MissingVectorInputIsNotRepairedByTheMaskEquality) {
    auto code = persisted(Reduction::Ff1);
    code[10] = 0xbf800000u; // missing v1 producer; the mask transport is otherwise complete
    empty(code, 0x5127001010ull);
    EXPECT_NE(last_terminal_reject_reason(0x5127001010ull).find("pc=24 "), std::string::npos);
}
TEST(ExecSpillData, RealTwoWordTransportProvesTheRestoreIsUnchanged) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    EXPECT_TRUE(restore(state));
    EXPECT_TRUE(restore(state)) << "a semantic no-op cannot end the current word lifetimes";
}
TEST(ExecSpillData, MissingLowWordCannotBorrowTheHighWord) {
    ExecSpillEquality state;
    high(state);
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, MissingHighWordCannotBorrowTheLowWord) {
    ExecSpillEquality state;
    low(state);
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, LowAndHighPositionsMustNotBeSwapped) {
    ExecSpillEquality state;
    low(state);
    high(state);
    step(state, {0xd760000eu, 0x00015113u});
    step(state, {0xd760000fu, 0x00014b13u});
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, UnrelatedSlotWriteRetainsTheActualWords) {
    ExecSpillEquality state;
    low(state);
    high(state);
    step(state, {0xd7610013u, 0x00014680u}); // v19[35]=0, separate from37/40
    reload(state);
    EXPECT_TRUE(restore(state));
}
TEST(ExecSpillData, SameSlotWriteEndsItsOldEquality) {
    ExecSpillEquality state;
    low(state);
    high(state);
    step(state, {0xd7610013u, 0x00014a81u}); // v19[37]=1, CURRENT_LOW was not established as1
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, DynamicSelectorRetiresEveryPossibleAliasedSlot) {
    ExecSpillEquality state;
    low(state);
    high(state);
    step(state, {0xd7610013u, 0x00000080u}); // scalar s0 selector, hardware wave-width masked
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, OutOfRangeLiteralCannotKeepACanonicalSlotAlias) {
    ExecSpillEquality state;
    low(state);
    high(state);
    step(state, {0xd7610013u, 0x0001fe81u, 101u}); //101&63 aliases37 on a masked selector path
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, OrdinaryVectorOverwriteRetiresAllOldSlots) {
    ExecSpillEquality state;
    low(state);
    high(state);
    step(state, {0x7e260280u}); // v19=0, no spill-array identity survives
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, CopiedScalarWordsOutliveTheirOldSlotStorage) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state, {0x7e260280u});
    EXPECT_TRUE(restore(state)) << "the actual SGPR copies are independent of later slot writes";
}
TEST(ExecSpillData, LowScalarOverwriteEndsTheCompleteRestoreProof) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state, {0xbe8e0381u});
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, HighScalarOverwriteEndsTheCompleteRestoreProof) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state, {0xbe8f0380u});
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, OneWordCopiesRetainBothGenuineTransportedValues) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state, {0xbe8c030eu});
    step(state, {0xbe8d030fu});
    step(state, {0xbe8e0381u}); // overwriting original copies cannot poison independent copies
    EXPECT_TRUE(step(state, {0xbefe040cu}));
}
TEST(ExecSpillData, SelfAliasedPairCopySnapshotsBothOldWords) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state,
         {0xbe8e040eu}); //s[14:15]=s[14:15], a genuine even-aligned source/destination alias
    EXPECT_TRUE(restore(state));
}
TEST(ExecSpillData, UnalignedPairSourceCannotBorrowAdjacentTransportedWords) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state, {0xbe90030fu}); //copy the high word before overwriting s15
    step(state, {0xbe8f030eu});   //s15/s16 now hold genuine LOW/HIGH values, but form no legal pair
    EXPECT_FALSE(step(state, {0xbefe040fu}));
}
TEST(ExecSpillData, UnalignedPairDestinationDoesNotPublishAnArchitecturalAlias) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state, {0xbe8f040eu});
    EXPECT_TRUE(state.scalars.empty());
}
TEST(ExecSpillData, DirectUntransportedSpecialWordsDoNotOriginateSpillFacts) {
    ExecSpillEquality state;
    step(state, {0xbe8e037eu});
    step(state, {0xbe8f037fu});
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, AdjacentNullSourceDoesNotInventAnExecWord) {
    ExecSpillEquality state;
    step(state, {0xbe8c047du});   //s[12:13]=NULL, no arithmetic adjacency to EXEC_LO
    EXPECT_TRUE(state.scalars.empty());
}
TEST(ExecSpillData, NarrowingBetweenHalfWritesEndsTheOldGeneration) {
    ExecSpillEquality state;
    low(state);
    step(state, {0x7dea0080u});   //ordinary CMPX replaces EXEC, preserving VCC
    high(state);
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, BothWordsWrittenAfterNarrowingHaveTheNewCurrentValue) {
    ExecSpillEquality state;
    low(state);
    step(state, {0x7dea0080u});
    low(state);
    high(state);
    reload(state);
    EXPECT_TRUE(restore(state));
}
TEST(ExecSpillData, PartialExecWriteEndsAllOlderCurrentEqualities) {
    ExecSpillEquality state;
    low(state);
    high(state);
    reload(state);
    step(state, {0xbefe0380u});
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, ARepeatedProducerPcIsNotAValueGeneration) {
    ExecSpillEquality state;
    low(state);
    high(state);
    step(state, {0x7dea0080u});
    high(state);   //same decoded pc and slot as the first high write, different CURRENT generation
    reload(state);
    EXPECT_FALSE(restore(state));
}
TEST(ExecSpillData, MustMeetRetainsTwoGenuineIncomingTransports) {
    ExecSpillEquality first, second;
    low(first);
    high(first);
    low(second);
    high(second);
    meet_exec_spill_equality(first, second);
    reload(first);
    EXPECT_TRUE(restore(first));
}
TEST(ExecSpillData, MustMeetRejectsOneIncomingLostLowWord) {
    ExecSpillEquality first, second;
    low(first);
    high(first);
    low(second);
    high(second);
    step(second, {0xd7610013u, 0x00014a81u});
    meet_exec_spill_equality(first, second);
    reload(first);
    EXPECT_FALSE(restore(first));
}
TEST(ExecSpillData, MustMeetKeepsTwoMissingIncomingWordsMissing) {
    ExecSpillEquality first, second;
    high(first);
    high(second);
    meet_exec_spill_equality(first, second);
    reload(first);
    EXPECT_FALSE(restore(first));
}
TEST(ExecSpillData, ScalarMeetRejectsOneIncomingLostHighWord) {
    ExecSpillEquality first, second;
    low(first);
    high(first);
    reload(first);
    low(second);
    high(second);
    reload(second);
    step(second, {0xbe8f0380u});
    meet_exec_spill_equality(first, second);
    EXPECT_FALSE(restore(first));
}

namespace {
std::vector<uint32_t> mixed_generation(bool renew_low) {
    // Save ENTRY's low word, then narrow to physical lane40 and define v1 only there.
    // An old-low/new-high restore enables all lower32 lanes unless LOW is rewritten in B.
    std::vector<uint32_t> code{
        0xd7610013u,
        0x00014a7eu,
        0x7da400a8u,   //CMPX_EQ_U32 40,v0: CURRENT B has LOW=0/HIGH=1<<8
        0x7e020280u,
    };
    if (renew_low) code.insert(code.end(), {0xd7610013u, 0x00014a7eu});
    code.insert(code.end(),
                {0xd7610013u, 0x0001507fu, 0xd760000eu, 0x00014b13u, 0xd760000fu, 0x00015113u,
                 0xbefe040eu,
                 0x7d020280u,   //compare 0,v1 observes the new mask's actual enabled lanes
                 0xbf810000u});
    return code;
}
ComputeShaderConfig dispatcher_config() {
    auto c = config();
    // Select the existing exact guest-wave dispatcher explicitly, avoiding an independent
    // straight-line scalar/mask proof that would conceal the new CFG CURRENT-equality boundary.
    c.force_exchange_dispatcher = true;
    return c;
}
}   // namespace
TEST(ExecSpillData, CurrentWordTransportAfterTheTransitionHasAUsableComparison) {
    const auto code = mixed_generation(true);
    const auto table = resources();
    const auto module = recompile_compute(code.data(), code.size(), &table, dispatcher_config(),
                                          {RecompileDiagnosticStage::Compute, 0x5127002010ull});
    retain(module, 0x5127002010ull);
    EXPECT_FALSE(module.empty()) << last_terminal_reject_reason(0x5127002010ull);
}
TEST(ExecSpillData, MixedGenerationRestoreCannotExposeAnUninitializedLowerLane) {
    const auto code = mixed_generation(false);
    const auto table = resources();
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, dispatcher_config(),
                                  {RecompileDiagnosticStage::Compute, 0x5127002020ull})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(0x5127002020ull).find("pc=11 "), std::string::npos)
        << last_terminal_reject_reason(0x5127002020ull);
}

namespace {
void actual_lanes(uint32_t selected, uint64_t address) {
    // Real lane-varying local IDs produce a low-only, high-only, or empty architectural EXEC.
    // The observable store is after the exact two-word restore, with v1 defined only in B.
    std::vector<uint32_t> code{0x7da40080u | selected,
                               0x7e020285u,
                               0xd7610013u,
                               0x00014a7eu,
                               0xd7610013u,
                               0x0001507fu,
                               0xd760000eu,
                               0x00014b13u,
                               0xd760000fu,
                               0x00015113u,
                               0xbefe040eu,
                               0x7d020285u,
                               0xe0702000u,
                               0x80020100u,
                               0xbf810000u};
    // 64 is a literal scalar compare source, rather than inline encoding 192 (-0).
    if (selected == 64) {
        code[0] = 0x7da400ffu;
        code.insert(code.begin() + 1, 64u);
    }
    auto table = resources();
    auto& out = table.resources.front();
    out.cls = ResourceClass::ConstantBuffer;
    out.format = DataFormat::Uint32;
    out.stride = 4;
    out.num_components = 1;
    const auto module = recompile_compute(code.data(), code.size(), &table, dispatcher_config(),
                                          {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    retain(module, address);
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;   // explicit complete oracle domain, not host inference
    const auto values = vm.run(64, std::vector<uint32_t>(64, 0xdeadbeefu));
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    ASSERT_EQ(values.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(values[lane], lane == selected ? 5u : 0xdeadbeefu) << "lane=" << lane;
}
}   // namespace
TEST(ExecSpillData, LowOnlyCurrentMaskRetainsItsActualStoreLanes) {
    actual_lanes(5, 0x5127003010ull);
}
TEST(ExecSpillData, HighOnlyCurrentMaskRetainsItsActualStoreLanes) {
    actual_lanes(40, 0x5127003020ull);
}
TEST(ExecSpillData, EmptyCurrentMaskDoesNotFabricateAnyEnabledStoreLane) {
    actual_lanes(64, 0x5127003030ull);
}
TEST(ExecSpillData, CurrentEqualityDoesNotSupplyAnAbsentPhysicalLaunch) {
    const auto code = persisted(Reduction::Ff1);
    const auto table = resources();
    auto c = config();
    c.dispatch_launch = {};
    constexpr uint64_t address = 0x5127004010ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_TRUE(module.empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("words=d760000e,00014b13"),
              std::string::npos)
        << last_terminal_reject_reason(address);
}

namespace {
bool signed_pc_comparison(uint32_t left, uint32_t right, bool defined = true) {
    const auto code = mixed_generation(true);
    const auto table = resources();
    constexpr uint64_t address = 0x5127005010ull;
    const auto module = recompile_compute(code.data(), code.size(), &table, dispatcher_config(),
                                          {RecompileDiagnosticStage::Compute, address});
    EXPECT_FALSE(module.empty()) << last_terminal_reject_reason(address);
    bpermute_oracle::Interpreter vm(module);
    const auto found = std::find_if(vm.ins.begin(), vm.ins.end(),
                                    [](const auto& instruction) { return instruction.op == 177; });
    if (found == vm.ins.end()) {
        ADD_FAILURE() << "the actual dispatcher has no signed PC bound";
        return false;
    }
    // Replacing vm.ins below invalidates its references; retain this actual instruction.
    const auto comparison = *found;   // NOLINT(performance-unnecessary-copy-initialization)
    const auto& args = comparison.a;
    EXPECT_NE(args[2], args[3]);
    uint32_t integer = 0;
    for (const auto& [id, type] : vm.types)
        if (type.op == 21 && type.a == std::vector<uint32_t>{32, 1}) integer = id;
    EXPECT_NE(integer, 0u);
    vm.ins = {comparison, {253, {}}};   // evaluate that typed instruction, then return
    bpermute_oracle::Lane lane;
    lane.v[args[2]] = {integer, {left}, defined};
    lane.v[args[3]] = {integer, {right}, true};
    vm.advance(lane);
    return vm.boolean(lane, args[1]);
}
}   // namespace
TEST(ExecSpillData, SignedDispatcherOracleDoesNotCompareNegativePcAsUnsigned) {
    EXPECT_TRUE(signed_pc_comparison(UINT32_MAX, 0));   // -1 < 0
    EXPECT_FALSE(signed_pc_comparison(0, UINT32_MAX));
}
TEST(ExecSpillData, SignedDispatcherOracleDoesNotPublishAPoisonComparison) {
    EXPECT_THROW(signed_pc_comparison(UINT32_MAX, 0, false), std::runtime_error);
}

TEST(ExecSpillData, EachIncomingPathMayResaveItsOwnDifferentCurrentMask) {
    ExecSpillEquality first, second;
    step(first, {0x7da40085u});   // CURRENT A: physical lane5
    low(first);
    high(first);
    reload(first);
    step(second, {0x7da400a8u});   // CURRENT B: physical lane40
    low(second);
    high(second);
    reload(second);
    meet_exec_spill_equality(first, second);
    EXPECT_TRUE(restore(first)) << "the joined equalities are relative to each actual CURRENT";
}
namespace {
std::vector<uint32_t> scalar_branch(bool overwrite_low) {
    std::vector<uint32_t> code{0xbf068000u, 0xbf840000u};   // genuine dynamic s0 branch DATA
    code.insert(code.end(),
                {0x7da40085u, 0x7e020280u, 0xd7610013u, 0x00014a7eu, 0xd7610013u, 0x0001507fu});
    if (overwrite_low) code.insert(code.end(), {0xd7610013u, 0x00014a81u});
    const auto jump = code.size();
    code.push_back(0xbf820000u);
    const auto second = code.size();
    code.insert(code.end(),
                {0x7da400a8u, 0x7e020280u, 0xd7610013u, 0x00014a7eu, 0xd7610013u, 0x0001507fu});
    const auto joined = code.size();
    code[1] |= static_cast<uint32_t>(second - 2);
    code[jump] |= static_cast<uint32_t>(joined - jump - 1);
    code.insert(code.end(), {0xd760000eu, 0x00014b13u, 0xd760000fu, 0x00015113u, 0xbefe040eu,
                             0x7d020280u, 0xbf810000u});
    return code;
}
}   // namespace
TEST(ExecSpillData, BothScalarBranchArmsSupplyUsableCurrentTransports) {
    const auto code = scalar_branch(false);
    const auto table = resources();
    std::vector<uint32_t> first;
    for (uint32_t s0 : {0u, 1u}) {
        auto c = dispatcher_config();
        c.user_sgprs = {s0};   // supplied push DATA, never resource or launch authority
        const uint64_t address = 0x5127006010ull + s0;
        const auto module = recompile_compute(code.data(), code.size(), &table, c,
                                              {RecompileDiagnosticStage::Compute, address});
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
        retain(module, address);
        if (first.empty())
            first = module;
        else
            EXPECT_EQ(module, first)
                << "the API supplies dynamic DATA; both branches stay reachable";
    }
}
TEST(ExecSpillData, OneArmCannotBorrowTheOtherArmsUnmodifiedLowSlot) {
    const auto code = scalar_branch(true);
    const auto table = resources();
    auto c = dispatcher_config();
    c.user_sgprs = {0};
    constexpr uint64_t address = 0x5127006020ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    // Existing mask typing rejects the mixed one-arm pair at MOV-to-EXEC, before DATA use.
    ASSERT_EQ(code[code.size() - 3], 0xbefe040eu);
    EXPECT_NE(last_terminal_reject_reason(address).find("wave64-ambiguous-mask-read"),
              std::string::npos)
        << last_terminal_reject_reason(address);
    EXPECT_NE(
        last_terminal_reject_reason(address).find("pc=" + std::to_string(code.size() - 3) + " "),
        std::string::npos)
        << last_terminal_reject_reason(address);
}
TEST(ExecSpillData, NumericPushWordsDoNotOriginateCurrentEquality) {
    auto code = mixed_generation(true);
    // Every source exists as numeric API DATA. That does not certify either as CURRENT EXEC.
    code[1] = 0x00014a00u;
    code[5] = 0x00014a00u;
    code[7] = 0x00015001u;
    const auto table = resources();
    auto c = dispatcher_config();
    c.user_sgprs = {0, 256};
    constexpr uint64_t address = 0x5127006030ull;
    EXPECT_TRUE(recompile_compute(code.data(), code.size(), &table, c,
                                  {RecompileDiagnosticStage::Compute, address})
                    .empty());
    EXPECT_NE(last_terminal_reject_reason(address).find("pc=13 "), std::string::npos)
        << last_terminal_reject_reason(address);
}
