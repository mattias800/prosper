// #5116: two current vector sources can be genuine in nested EXEC domains without sharing an
// owner SSA value. Losing their confinement rejects a real active-lane read; inventing a subset
// would instead let absent lanes form a predicate. Emit actual CMPX/writers, including old-word
// accumulators, and keep CFG joins, carried obligations and later generations separate.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;
constexpr uint32_t cmpx_a = 0x7da6009fu; // v_cmpx_le_u32 31, v0
constexpr uint32_t cmpx_b = 0x7da600afu; // v_cmpx_le_u32 47, v0
constexpr uint32_t cmpx_c = 0x7da600b7u; // v_cmpx_le_u32 55, v0
constexpr uint32_t mov6 = 0x7e0c0281u, mov7 = 0x7e0e0282u;
constexpr uint32_t xor8 = 0x3a100f06u, xor8_reverse = 0x3a100d07u;
constexpr uint32_t compare8 = 0x7d841108u, compare6 = 0x7d840d06u;
constexpr uint32_t full_exec = 0xbefe04c1u, end = 0xbf810000u;
constexpr uint32_t fmac6 = 0x560c0ef2u; // v_fmac_f32 v6, 1.0, v7; old v6 is implicit
// #5185: a word relayed from, or accumulated over, a never-written VGPR compute does not load is
// arbitrary, so the public refusals below need a HARD producer: the old LDS word an atomic returns
// at the never-written address v20 (ds_add_rtn_u32 vD, v20, v0). An atomic at an arbitrary address
// stays hard (#5193); a plain LDS read there would be arbitrary. (They used v_mov_b32 v7, v20 or
// an absent accumulator, which are admitted as arbitrary now.)
Words lds_hard(uint32_t vgpr) {
    return {0xd8800000u, (vgpr << 24) | 20u};
}

Words cat(std::initializer_list<Words> pieces) {
    Words code;
    for (const auto& piece : pieces) code.insert(code.end(), piece.begin(), piece.end());
    return code;
}
std::vector<Rdna2Inst> decode(const Words& code) {
    std::vector<Rdna2Inst> instructions;
    EXPECT_EQ(rdna2_walk(code.data(), code.size(), instructions), code.size());
    return instructions;
}
Words nested(uint32_t combine = xor8, const Words& middle = {}) {
    return cat({{cmpx_a, mov6, cmpx_b, mov7}, middle, {combine}});
}
void retain(const Words& module, const std::string& arm) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): inherited before this single-threaded test process.
    if (const char* directory = std::getenv("PROSPER_CONFINED_UNION_SPV_DIRECTORY")) {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        static uint64_t serial = 0;
        const auto path =
            std::filesystem::path(directory) /
            (std::string(info->name()) + "-" + arm + "-" + std::to_string(++serial) + ".spv");
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(module.data()),
                     static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
        EXPECT_TRUE(output.good()) << "retain the actual producing module";
    }
}

struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.begin(4, nullptr, 64, 1, 1, 64, 0, true, true);
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.vreg[0] = b.localid_comp[0];
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.max_vgpr = 20;
        // Inventory absence before a writer can hide it behind an SSA select.
        const auto inventory = decode({cmpx_a, mov6, cmpx_b, mov7, xor8, compare8, 0x7e260314u});
        seed_vector_word_definedness(rs, inventory, false, 64);
    }
    bool emit(const Words& code) {
        for (const auto& in : decode(code)) {
            const auto before = snapshot_saved_b64_masks(b, rs, in);
            bool ok = true;
            const bool handled = emit_alu(b, rs, in, ok, true);
            EXPECT_TRUE(handled);
            if (!handled || !ok) return false;
            record_scalar_write(rs, in, false, before);
        }
        return true;
    }
    ScalarSourceMarks source(uint32_t instruction = xor8) const {
        return scalar_source_marks(rs, rdna2_decode_one(&instruction, 1));
    }
    bool defined(uint32_t comparison = compare8) {
        return fresh_mask_sources_defined(b, rs, rdna2_decode_one(&comparison, 1));
    }
};

struct Result {
    Words module;
    std::string reason;
};
Result compile(const Words& body, uint32_t native = 64, bool raw_launch = true) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = native;
    config.user_sgprs.resize(4);
    config.tgid_x_en = true;   // s4 is a genuine dynamic selector for both-edge controls
    prosper::test::complete_compute_launch(config, 2);
    if (!raw_launch) config.dispatch_launch = {};
    ShaderResourceTable table;
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 3;
    output.stride = 4;
    output.sgpr_base = 8;
    table.resources.push_back(output);
    const Words code = cat({body, {0x7e12028du, 0xe0702000u, 0x80020900u, end}});
    static uint64_t next_address = 0x5116000000ull;
    const uint64_t address = ++next_address;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    if (!module.empty()) retain(module, "public");
    return {std::move(module), last_terminal_reject_reason(address)};
}
void accepted(const Result& result) {
    EXPECT_FALSE(result.module.empty()) << result.reason;
}
void refused(const Result& result, uint32_t pc,
             const std::string& reason = "fabricated-fresh-mask-source") {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find(reason), std::string::npos) << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reason;
}
Words loop(const Words& rewrite) {
    // A scalar, top-tested loop carries v6. It reads the seed before the backedge rewrite.
    const Words read{compare6};
    const uint32_t body_size = static_cast<uint32_t>(read.size() + rewrite.size() + 2);
    return cat({{0x7e0c0281u, 0xbe960380u, 0xbe970383u, 0xbf0a1716u, 0xbf840000u | body_size},
                read,
                rewrite,
                {0x80168116u, 0xbf820000u | ((0x10000u - body_size - 2u) & 0xffffu)}});
}
}   // namespace

TEST(ConfinedSourceUnion, EncodingsNameRealNestedWritersAndImplicitAccumulator) {
    const auto instructions = decode(nested());
    ASSERT_EQ(instructions.size(), 5u);
    for (int index : {0, 2}) {
        EXPECT_TRUE(vopc_is_cmpx(instructions[index].opcode));
        EXPECT_EQ(instructions[index].opcode, 0xd3u);
        EXPECT_EQ(instructions[index].src[1].value, 0);
    }
    EXPECT_EQ(instructions[4].src[0].value, 6);
    EXPECT_EQ(instructions[4].src[1].value, 7);
    EXPECT_EQ(instructions[4].opcode, 0x1du);
    const auto accumulator = decode({fmac6});
    ASSERT_EQ(accumulator.size(), 1u);
    EXPECT_EQ(accumulator[0].opcode, 0x2bu);
    EXPECT_EQ(accumulator[0].dst.value, 6);
    EXPECT_EQ(accumulator[0].src[1].value, 7);
}

TEST(ConfinedSourceUnion, CurrentExplicitSourcesRetainTheNarrowerEmittedOwner) {
    for (uint32_t combine : {xor8, xor8_reverse}) {
        Emitter e;
        ASSERT_TRUE(e.emit({cmpx_a, mov6}));
        const uint32_t a = e.rs.exec;
        ASSERT_TRUE(e.emit({cmpx_b, mov7}));
        const uint32_t b = e.rs.exec;
        ASSERT_NE(a, b);
        ASSERT_EQ(e.rs.exec_subset_of.at(b), a);
        const auto source = e.source(combine);
        EXPECT_TRUE(source.placeholder);
        EXPECT_TRUE(source.hard);
        EXPECT_EQ(source.outside_exec, b);
        ASSERT_TRUE(e.emit({combine}));
        EXPECT_TRUE(e.defined());
        EXPECT_TRUE(e.rs.vreg_source_marks.at(8).placeholder)
            << "active-lane definedness does not authenticate the missing outer lanes";
    }
}

TEST(ConfinedSourceUnion, ATransitiveNestedOwnerRetainsCurrentConfinement) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6, cmpx_b, cmpx_c, mov7, xor8}));
    EXPECT_TRUE(e.defined());
    EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, e.rs.exec);
}

TEST(ConfinedSourceUnion, ActualEmittedIntegerValuesMatchOnlyTheEnabledLanes) {
    Emitter e;
    ASSERT_TRUE(e.emit(nested()));
    ASSERT_TRUE(e.defined());
    e.b.store_output_word(e.rs.vreg.at(8), 1, 0, e.rs.exec);
    const auto module = e.b.finish();
    retain(module, "emitter");
    bpermute_oracle::Interpreter vm(module);
    const auto actual = vm.run_buffers(64, {{1, Words(64, 0xdeadbeefu)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    Words expected(64, 0xdeadbeefu);
    for (uint32_t lane = 47; lane < 64; ++lane) expected[lane] = 3;
    EXPECT_EQ(actual, expected);
}

TEST(ConfinedSourceUnion, ARelayPreservesTheNestedSourceDomain) {
    Emitter e;
    ASSERT_TRUE(e.emit(nested(xor8, {0x7e0e0307u})));
    EXPECT_TRUE(e.defined());
}

TEST(ConfinedSourceUnion, ImplicitAccumulatorMeetsItsOldWordWithTheNewSource) {
    Emitter e;
    ASSERT_TRUE(e.emit(nested(fmac6)));
    EXPECT_TRUE(e.defined(compare6));
    EXPECT_EQ(e.rs.vreg_source_marks.at(6).outside_exec, e.rs.exec);
}

TEST(ConfinedSourceUnion, AnAbsentAccumulatorOldWordRemainsUnconfined) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6, cmpx_b, mov7, 0x56100ef2u}));
    EXPECT_FALSE(e.defined());
    EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, 0u);
}

TEST(ConfinedSourceUnion, IncomparableEmittedOwnersCannotGrantConfinement) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6, full_exec, cmpx_b, mov7}));
    EXPECT_EQ(e.source().outside_exec, 0u);
    ASSERT_TRUE(e.emit({xor8}));
    EXPECT_FALSE(e.defined());
}

TEST(ConfinedSourceUnion, DisjointEmittedOwnersCannotGrantConfinement) {
    Emitter e;
    ASSERT_TRUE(e.emit({0x7da40080u, mov6, full_exec, 0x7daa0080u, mov7, xor8}));
    EXPECT_FALSE(e.defined());
}

TEST(ConfinedSourceUnion, ARemovedSubsetFactKeepsThePositiveShapePoisoned) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6, cmpx_b, mov7}));
    e.rs.exec_subset_of.erase(e.rs.exec);   // model the absent emitter proof, never insert a fact
    EXPECT_EQ(e.source().outside_exec, 0u);
    ASSERT_TRUE(e.emit({xor8}));
    EXPECT_FALSE(e.defined());
}

TEST(ConfinedSourceUnion, AnUnconfinedRelayCannotBorrowTheOtherSourcesOwner) {
    Emitter e;
    ASSERT_TRUE(e.emit(nested(xor8, {0x7e0e0314u})));   // v7 = absent v20
    EXPECT_FALSE(e.defined());
    EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, 0u);
}

TEST(ConfinedSourceUnion, AnUnproducedScalarRelayRemainsUnconfined) {
    Emitter e;
    ASSERT_TRUE(e.emit(nested(xor8, {0x7e0e0214u})));   // v7 = absent s20
    EXPECT_FALSE(e.defined());
}

TEST(ConfinedSourceUnion, RestoringFullExecCannotReadTheMissingOuterLanes) {
    Emitter e;
    ASSERT_TRUE(e.emit(cat({nested(), {full_exec}})));
    EXPECT_FALSE(e.defined());
}

TEST(ConfinedSourceUnion, WideningTheAccumulatorCannotRebaseItsMissingLanes) {
    Emitter e;
    ASSERT_TRUE(e.emit(cat({nested(fmac6), {full_exec, cmpx_a, fmac6}})));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConfinedSourceUnion, ContextFreePredecessorJoinStillRequiresEqualOwners) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6, cmpx_b, mov7}));
    VectorWordMarks joined{{6, e.rs.vreg_source_marks.at(6)}};
    join_vector_word_marks(joined, {{6, e.rs.vreg_source_marks.at(7)}});
    EXPECT_TRUE(joined.at(6).hard);
    EXPECT_EQ(joined.at(6).outside_exec, 0u);
}

TEST(ConfinedSourceUnion, ContextFreeJoinKeepsAnUnconfinedPredecessorPoisoned) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6}));
    VectorWordMarks joined{{6, e.rs.vreg_source_marks.at(6)}};
    join_vector_word_marks(joined, {{6, e.rs.vreg_source_marks.at(20)}});
    EXPECT_EQ(joined.at(6).outside_exec, 0u);
}

TEST(ConfinedSourceUnion, NestedOwnerUnionKeepsBothActualLoopCarryObligations) {
    for (bool poison_backedge : {false, true}) {
        Emitter e;
        ASSERT_TRUE(e.emit({cmpx_a, mov6, cmpx_b, mov7}));
        LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
        const auto body = decode({mov6, mov7});
        mark_vector_loop_carried(e.rs, body, 0, 2, check.carried, check.confined);
        ASSERT_TRUE(e.emit({xor8}));
        ASSERT_TRUE(e.defined());
        for (int reg : {6, 7})
            EXPECT_TRUE(
                e.b.loop_blanket_roots_read.contains(blanket_root(0, kVectorBlanketBase + reg)));
        if (poison_backedge) ASSERT_TRUE(e.emit({0x7e0e0314u}));
        auto exits = loop_check_marks(e.rs);
        EXPECT_EQ(check.holds(e.b, e.rs, exits), !poison_backedge);
        if (poison_backedge) {
            EXPECT_EQ(exits.vectors.at(8).outside_exec, 0u);
            EXPECT_TRUE(exits.vectors.at(8).blanket_roots.empty());
        }
    }
}

// #5167 reversed this case's expectation. v7 is a loop assumption: the guard itself admits a
// predicate on v7 alone, recording its root for the backedge check, and admits v6 alone under B.
// Their lane-wise combination is no less defined, so refusing it was an over-refusal. The
// assumption still adds its root to v6's owner, and a poisoned backedge still fails the check.
TEST(ConfinedSourceUnion, ALoopAssumptionAddsItsRootToANestedHardOwner) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov7, cmpx_a, mov6}));
    const uint32_t a = e.rs.exec;
    ASSERT_TRUE(e.emit({cmpx_b}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    mark_vector_loop_carried(e.rs, decode({mov7}), 0, 1, check.carried, check.confined);
    EXPECT_TRUE(e.rs.vreg_source_marks.at(7).placeholder);
    EXPECT_FALSE(e.rs.vreg_source_marks.at(7).hard);
    const uint64_t root = blanket_root(0, kVectorBlanketBase + 7);
    EXPECT_EQ(e.source().outside_exec, a);
    EXPECT_TRUE(e.source().blanket_roots.contains(root));
    ASSERT_TRUE(e.emit({xor8}));
    ASSERT_TRUE(e.defined());
    EXPECT_TRUE(e.b.loop_blanket_roots_read.contains(root));
    ASSERT_TRUE(e.emit({0x7e0e0314u}));   // the backedge rewrites v7 from absent v20
    auto exits = loop_check_marks(e.rs);
    EXPECT_FALSE(check.holds(e.b, e.rs, exits));
}

TEST(ConfinedSourceUnion, PublicCompilerProducesTheNestedExplicitSourceModule) {
    accepted(compile(cat({nested(), {compare8}})));
    accepted(compile(cat({nested(xor8_reverse), {compare8}})));
}

TEST(ConfinedSourceUnion, PublicCompilerProducesTheNestedAccumulatorModule) {
    accepted(compile(cat({nested(fmac6), {compare6}})));
}

TEST(ConfinedSourceUnion, PublicCompilerRejectsAnUnconfinedRelayAtItsFreshCompare) {
    accepted(compile(cat({nested(xor8, {0x7e0e0307u}), {compare8}})));
    accepted(compile(cat({nested(xor8, {0x7e0e0314u}), {compare8}})));   // relays arbitrary v20
    refused(compile(cat({nested(xor8, lds_hard(7)), {compare8}})), 7);
}

TEST(ConfinedSourceUnion, PublicCompilerRejectsAnAbsentImplicitAccumulator) {
    accepted(compile(cat({nested(fmac6), {compare6}})));
    // An absent accumulator is arbitrary (#5185); a hard one is refused.
    accepted(compile({cmpx_a, mov6, cmpx_b, mov7, 0x56100ef2u, compare8}));
    refused(compile(cat({lds_hard(8), {cmpx_a, mov6, cmpx_b, mov7, 0x56100ef2u, compare8}})), 7);
}

TEST(ConfinedSourceUnion, PublicCompilerRejectsIncomparableOwnersAtTheCompare) {
    accepted(compile(cat({nested(), {compare8}})));
    // v6's lanes of B outside A were never written: arbitrary, so admitted (#5185). With a hard
    // old v6 those lanes reach the compare.
    accepted(compile({cmpx_a, mov6, full_exec, cmpx_b, mov7, xor8, compare8}));
    refused(compile(cat({lds_hard(6), {cmpx_a, mov6, full_exec, cmpx_b, mov7, xor8, compare8}})),
            8);
}

TEST(ConfinedSourceUnion, PublicCompilerRejectsAFullExecReadAfterNestedWrites) {
    accepted(compile(cat({nested(), {compare8}})));
    // Outside B, v8 was never written and compute never loads it: #5174 admits that read as an
    // arbitrary value (a refusal before the decision made it an over-refusal), and #5185 admits
    // it when v8 first relays another such word. When v8 first holds a hard value, its lanes
    // outside B keep that old value and the read stays refused.
    accepted(compile(cat({nested(), {full_exec, compare8}})));
    accepted(
        compile(cat({{0x7e10031fu /* v_mov_b32 v8, v31 */}, nested(), {full_exec, compare8}})));
    refused(compile(cat({lds_hard(8), nested(), {full_exec, compare8}})), 8);
}

TEST(ConfinedSourceUnion, OneArmWriterCannotSupplyTheOtherPredecessor) {
    // Dynamic s4 makes both edges reachable; only the writer differs between these two shapes.
    accepted(compile(
        {0xbf068004u, 0xbf840002u, mov7, 0xbf820001u, mov7, cmpx_a, mov6, cmpx_b, xor8, compare8}));
    // The other edge leaves v7 never written: arbitrary (#5185). Leaving it hard is refused.
    accepted(compile({0xbf068004u, 0xbf840002u, mov7, 0xbf820001u, 0xbf800000u, cmpx_a, mov6,
                      cmpx_b, xor8, compare8}));
    refused(compile(cat({lds_hard(7),
                         {0xbf068004u, 0xbf840002u, mov7, 0xbf820001u, 0xbf800000u, cmpx_a, mov6,
                          cmpx_b, xor8, compare8}})),
            11);
}

TEST(ConfinedSourceUnion, ALoopBackedgeCannotLaunderAnAbsentCurrentGeneration) {
    accepted(compile(loop({mov6})));
    // v6 = s30, which no launch loads: hard. A never-written source (v20) is arbitrary, which an
    // arbitrary back edge now holds weakly (#5193).
    const auto poisoned = compile(loop({0x7e0c021eu}));
    EXPECT_TRUE(poisoned.module.empty());
    EXPECT_NE(poisoned.reason.find("fabricated on the back edge"), std::string::npos)
        << poisoned.reason;
}

TEST(ConfinedSourceUnion, FreshProfilesAndCodeDoNotReuseTheEarlierWordGeneration) {
    accepted(compile(cat({nested(), {compare8}})));
    for (uint32_t native : {0u, 32u, 64u})
        refused(compile(cat({nested(xor8, lds_hard(7)), {compare8}}), native), 7);
    refused(compile(cat({nested(xor8, lds_hard(7)), {compare8}}), 64, false), 7);
    accepted(compile(cat({nested(), {compare8}})));
}

// #5128: a complete narrowed writer defines its active domain even when the inactive old
// generation was conditional on a loop backedge. Publishing that domain must retire only
// unread old-word roots; MAC and other preserving writers still consume their old generation.
namespace {
void conditional_words(Emitter& e, LoopBlanketCheck& check, const Words& writers) {
    const auto instructions = decode(writers);
    mark_vector_loop_carried(e.rs, instructions, 0, static_cast<uint32_t>(writers.size()),
                             check.carried, check.confined);
}
constexpr uint32_t mov6_from7 = 0x7e0c0307u;
constexpr uint32_t mov8 = 0x7e100281u;
constexpr uint32_t fmac6_from8 = 0x560c10f2u;
constexpr uint32_t xor9_from6and8 = 0x3a121106u;
constexpr uint32_t compare9 = 0x7d841309u;
Words publication_loop(const Words& body, const Words& seed = {mov6}) {
    const uint32_t size = static_cast<uint32_t>(body.size() + 2u);
    return cat({seed,
                {0xbe960380u, 0xbe970383u, 0xbf0a1716u, 0xbf840000u | size},
                body,
                {0x80168116u, 0xbf820000u | ((0x10000u - size - 2u) & 0xffffu)}});
}
void structured_public(const Result& result) {
    ASSERT_FALSE(result.module.empty()) << result.reason;
    bool loop = false, dispatcher = false;
    for (std::size_t pc = 5; pc < result.module.size();) {
        const uint32_t count = result.module[pc] >> 16u;
        ASSERT_NE(count, 0u);
        ASSERT_LE(pc + count, result.module.size());
        loop |= (result.module[pc] & 0xffffu) == Op_LoopMerge;
        dispatcher |= (result.module[pc] & 0xffffu) == Op_Switch;
        pc += count;
    }
    EXPECT_TRUE(loop);
    EXPECT_FALSE(dispatcher);
}
}   // namespace

TEST(ConditionalWordPublication, CompleteWriterPublishesTheSameExecDomainBeforeMac) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_FALSE(e.rs.vreg_source_marks.at(6).hard);
    ASSERT_TRUE(e.emit({cmpx_a, mov6, mov8, fmac6_from8}));
    EXPECT_TRUE(e.defined(compare6));
    EXPECT_EQ(e.rs.vreg_source_marks.at(6).outside_exec, e.rs.exec);
    EXPECT_TRUE(e.rs.vreg_source_marks.at(6).blanket_roots.empty());
}

TEST(ConditionalWordPublication, CompleteWriterCanFeedAnExplicitSourceAndRelay) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov6, mov8, xor9_from6and8, 0x7e120309u}));
    ASSERT_TRUE(e.defined(compare9));
    e.b.store_output_word(e.rs.vreg.at(9), 1, 0, e.rs.exec);
    const auto module = e.b.finish();
    retain(module, "conditional-value");
    bpermute_oracle::Interpreter vm(module);
    const auto actual = vm.run_buffers(64, {{1, Words(64, 0xdeadbeefu)}}, 1);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    Words expected(64, 0xdeadbeefu);
    for (uint32_t lane = 31; lane < 64; ++lane) expected[lane] = 0;
    EXPECT_EQ(actual, expected);
}

TEST(ConditionalWordPublication, ACompleteWriteRetiresOnlyItsUnreadOldRoot) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6, mov7}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6, mov7});
    ASSERT_TRUE(e.emit({cmpx_a, mov6_from7}));
    const auto& marks = e.rs.vreg_source_marks.at(6);
    EXPECT_FALSE(marks.hard) << "the original global conditional proof must survive";
    EXPECT_EQ(marks.outside_exec, e.rs.exec);
    EXPECT_EQ(marks.blanket_roots, (std::set<uint64_t>{blanket_root(0, kVectorBlanketBase + 6),
                                                       blanket_root(0, kVectorBlanketBase + 7)}));
    ASSERT_TRUE(e.defined(compare6));
    EXPECT_FALSE(e.b.loop_blanket_roots_read.contains(blanket_root(0, kVectorBlanketBase + 6)));
    EXPECT_TRUE(e.b.loop_blanket_roots_read.contains(blanket_root(0, kVectorBlanketBase + 7)));
    // The rewritten old destination may be unavailable; only the consumed source must hold.
    ASSERT_TRUE(e.emit({0x7e0c0314u, full_exec, mov7}));
    auto exits = loop_check_marks(e.rs);
    EXPECT_TRUE(check.holds(e.b, e.rs, exits));
}

TEST(ConditionalWordPublication, AConsumedSourceRootStillRejectsAPoisonedBackedge) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6, mov7}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6, mov7});
    ASSERT_TRUE(e.emit({cmpx_a, mov6_from7}));
    ASSERT_TRUE(e.defined(compare6));
    ASSERT_TRUE(e.emit({0x7e0e0314u}));
    auto exits = loop_check_marks(e.rs);
    EXPECT_FALSE(check.holds(e.b, e.rs, exits));
    EXPECT_EQ(exits.vectors.at(6).outside_exec, 0u);
    EXPECT_TRUE(exits.vectors.at(6).blanket_roots.empty());
}

TEST(ConditionalWordPublication, MacStillReadsItsConditionalOldWordRoot) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({mov8, fmac6_from8}));
    ASSERT_TRUE(e.defined(compare6));
    EXPECT_TRUE(e.b.loop_blanket_roots_read.contains(blanket_root(0, kVectorBlanketBase + 6)));
    ASSERT_TRUE(e.emit({0x7e0c0314u}));
    auto exits = loop_check_marks(e.rs);
    EXPECT_FALSE(check.holds(e.b, e.rs, exits));
}

TEST(ConditionalWordPublication, MacCannotBorrowAnOwnerForAnUnconfinedSoftOldWord) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov8, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, AnAbsentAccumulatorCannotBorrowTheNewSourcesOwner) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov8, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, ACompleteWriterCannotAuthenticateAnAbsentScalarRelay) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, 0x7e0c0214u, mov8, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, RestoredExecKeepsTheOriginalGlobalBackedgeObligation) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov6, full_exec}));
    ASSERT_TRUE(e.defined(compare6));
    EXPECT_TRUE(e.b.loop_blanket_roots_read.contains(blanket_root(0, kVectorBlanketBase + 6)));
    ASSERT_TRUE(e.emit({0x7e0c0314u}));
    auto exits = loop_check_marks(e.rs);
    EXPECT_FALSE(check.holds(e.b, e.rs, exits));
}

TEST(ConditionalWordPublication, RestoredExecCannotReadAHardAccumulatorOutsideItsOwner) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov6, mov8, fmac6_from8}));
    ASSERT_TRUE(e.defined(compare6));
    ASSERT_TRUE(e.emit({full_exec}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, ADisjointOwnerCannotReuseThePublishedGeneration) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({0x7da40080u, mov6, full_exec, 0x7daa0080u, mov8, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, APartialWriterKeepsItsConsumedOldGeneration) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov8, 0x7e0c02f9u, 0x00851408u, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, AnOverwriteWithAnAbsentRelayStartsANewUnconfinedGeneration) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov6, mov8, fmac6_from8}));
    ASSERT_TRUE(e.defined(compare6));
    ASSERT_TRUE(e.emit({0x7e0c0314u, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, AContextFreeJoinCannotAuthenticateOneMissingArm) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    const VectorWordMarks missing{{6, e.rs.vreg_source_marks.at(20)}};
    ASSERT_TRUE(e.emit({cmpx_a, mov6}));
    join_vector_word_marks(e.rs.vreg_source_marks, missing);
    ASSERT_TRUE(e.emit({mov8, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, PublicLoopPublishesItsCurrentGenerationBeforeTheAccumulator) {
    structured_public(
        compile(publication_loop({cmpx_a, mov6, mov8, fmac6_from8, compare6, full_exec})));
}

// #5193: these twins seed v6 = s30 (no launch loads it), so an unreplaced old v6 is HARD. With
// the defined seed v6 = 1 it is a loop assumption, which an arbitrary meet now keeps with its
// obligations: no hard lane reaches the compare.
constexpr uint32_t hard_seed6 = 0x7e0c021eu;   // v_mov_b32 v6, s30

TEST(ConditionalWordPublication, PublicLoopRejectsAnUnreadButUnreplacedOldAccumulator) {
    structured_public(compile(publication_loop(
        {cmpx_a, mov6, mov8, fmac6_from8, compare6, full_exec}, {hard_seed6})));
    refused(compile(publication_loop(
                {cmpx_a, 0xbf800000u, mov8, fmac6_from8, compare6, full_exec}, {hard_seed6})),
            9);
}

TEST(ConditionalWordPublication, PublicLoopRejectsANewAbsentSourceAtItsFreshCompare) {
    refused(
        compile(publication_loop({cmpx_a, 0x7e0c021eu, mov8, fmac6_from8, compare6, full_exec})),
        9);   // v6 = s30, never loaded: hard (a never-written v20 would be arbitrary, #5193)
}

TEST(ConditionalWordPublication, PublicLoopRequiresBothWriterArms) {
    const auto body = [](uint32_t other) {
        return Words{cmpx_a, 0xbf068004u, 0xbf840002u, mov6,     0xbf820001u,
                     other,  mov8,        fmac6_from8, compare6, full_exec};
    };
    structured_public(compile(publication_loop(body(mov6), {hard_seed6})));
    refused(compile(publication_loop(body(0xbf800000u), {hard_seed6})), 13);
}

TEST(ConditionalWordPublication, PublicLoopChecksTheConsumedSourceOnItsActualBackedge) {
    const auto body = [](uint32_t rewrite) {
        return Words{cmpx_a, mov6_from7, mov8, fmac6_from8, compare6, full_exec, rewrite};
    };
    structured_public(compile(publication_loop(body(mov7), {mov6, mov7})));
    // v7 = s30, never loaded: hard (a never-written v20 would be arbitrary, #5193).
    const auto poisoned = compile(publication_loop(body(0x7e0e021eu), {mov6, mov7}));
    EXPECT_TRUE(poisoned.module.empty());
    EXPECT_NE(poisoned.reason.find("loop-carried v7 is fabricated on the back edge"),
              std::string::npos)
        << poisoned.reason;
}

TEST(ConditionalWordPublication, ANewSoftHeaderCannotReuseThePriorWritersScopedGeneration) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck first(e.rs, std::set<int>{}, 0);
    conditional_words(e, first, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov6}));
    LoopBlanketCheck next(e.rs, std::set<int>{}, 10);
    auto instructions = decode({mov6});
    instructions[0].pc = 10;
    mark_vector_loop_carried(e.rs, instructions, 10, 11, next.carried, next.confined);
    EXPECT_EQ(e.rs.vreg_source_marks.at(6).outside_exec, 0u);
    ASSERT_TRUE(e.emit({mov8, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(ConditionalWordPublication, AnUnreadFailedGlobalRootKeepsAnIndependentScopedRelay) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a, mov6, 0x7e120306u}));
    ASSERT_TRUE(e.defined(compare9));
    ASSERT_TRUE(e.emit({0x7e0c0314u}));
    auto exits = loop_check_marks(e.rs);
    ASSERT_TRUE(check.holds(e.b, e.rs, exits));
    EXPECT_TRUE(e.defined(compare9));
    ASSERT_TRUE(e.emit({full_exec}));
    EXPECT_FALSE(e.defined(compare9));
}

TEST(ConditionalWordPublication, AMemoryResultDoesNotPublishAnArithmeticScopedProof) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    conditional_words(e, check, {mov6});
    ASSERT_TRUE(e.emit({cmpx_a}));
    // Test the mark recorder's boundary without claiming a memory emitter or resource admission.
    // The existing global conditional roots survive; memory requires its own producing proof.
    const auto instructions = decode({0xe0502000u, 0x80020600u});
    ASSERT_EQ(instructions.size(), 1u);
    ASSERT_EQ(instructions[0].fmt, Rdna2Format::MUBUF);
    ASSERT_EQ(instructions[0].dst.value, 6);
    const auto source = scalar_source_marks(e.rs, instructions[0]);
    EXPECT_TRUE(source.memory);
    record_vector_word_marks(e.rs, instructions[0], source);
    EXPECT_FALSE(e.rs.vreg_source_marks.at(6).hard);
    ASSERT_TRUE(e.emit({mov8, fmac6_from8}));
    EXPECT_FALSE(e.defined(compare6));
}
