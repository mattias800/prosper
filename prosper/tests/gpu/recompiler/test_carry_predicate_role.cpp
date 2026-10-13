// A genuine current carry predicate is a per-lane Bool even when its physical SGPR words
// are unavailable. Charging those words loses valid arithmetic; accepting an old or fabricated
// predicate can change a 64-bit address. Keep source lifetime and arithmetic value controls paired.
#include "bpermute_spirv_oracle.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_structured_saved_masks.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr std::array<uint32_t, 3> operations{0x128, 0x129, 0x12a};
std::vector<uint32_t> carry(uint32_t op, uint32_t condition = 32, uint32_t destination = 3,
                            uint32_t left = 129, uint32_t right = 257) {
    return {0xd4003000u | (op << 16u) | destination, left | (right << 9u) | (condition << 18u)};
}
std::vector<uint32_t> program(uint32_t op, const std::vector<uint32_t>& middle = {},
                              bool numeric_observer = false) {
    // SDWA V_CMP_LT_U32 s[32:33],31,v0; SRC2 consumes that predicate, never its physical words.
    std::vector<uint32_t> words{0x7d8200f9u, 0x0686a09fu};
    words.insert(words.end(), middle.begin(), middle.end());
    const auto arithmetic = carry(op);
    words.insert(words.end(), arithmetic.begin(), arithmetic.end());
    if (numeric_observer) words.push_back(0x7e080220u);   // numeric v2 = s32
    words.push_back(end);
    return words;
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    return ins;
}
void retain_module(const std::vector<uint32_t>& module, uint32_t variant) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): the observer is configured before this single-threaded test.
    const char* directory = std::getenv("PROSPER_TEST_CARRY_SPV_DIRECTORY");
    if (!directory || !*directory || module.empty()) return;
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    const auto path = std::filesystem::path(directory) /
                      (std::string(info->name()) + "-" + std::to_string(variant) + ".spv");
    std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(module.data()),
                 static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    ASSERT_TRUE(output.good());
}
struct Program {
    std::vector<uint32_t> words;
    std::map<std::string, size_t> labels;
    std::vector<std::pair<size_t, std::string>> edges;
    void add(std::initializer_list<uint32_t> values) { words.insert(words.end(), values); }
    void label(const std::string& name) { labels[name] = words.size(); }
    void branch(uint32_t op, const std::string& name) {
        edges.emplace_back(words.size(), name);
        add({op});
    }
    std::vector<uint32_t> finish() {
        for (const auto& [pc, name] : edges)
            words[pc] |= static_cast<uint16_t>(labels.at(name) - pc - 1);
        return words;
    }
};
enum class Control {
    Genuine,
    Absent,
    Poison,
    Relay,
    Low,
    High,
    Replacement,
    OneArm,
    BothArms,
    AcrossEdge,
    LoopGenuine,
    LoopPoison
};
std::vector<uint32_t> production_program(uint32_t op, Control control = Control::Genuine) {
    Program p;
    p.add({0x7e0202ffu, UINT32_MAX});   // genuine v1, independent of the condition's producer
    if (control == Control::LoopGenuine || control == Control::LoopPoison)
        p.add({0xbe840380u});   // genuine scalar loop counter
    const bool split = control == Control::OneArm || control == Control::BothArms;
    if (split) {
        p.add({0xbf068004u});   // genuine user s4
        p.branch(0xbf840000u, "other");
    }
    if (control == Control::Relay) p.add({0x7e0e0214u});   // absent s20 -> v7
    if (control != Control::Absent)
        p.add({control == Control::Poison || control == Control::Relay ? 0x7d820ef9u : 0x7d8200f9u,
               0x0686a09fu});
    if (split) {
        p.branch(0xbf820000u, "merge");
        p.label("other");
        if (control == Control::BothArms) p.add({0x7d8200f9u, 0x0686a080u});
        p.label("merge");
    }
    if (control == Control::Low) p.add({0xbea00380u});
    if (control == Control::High || control == Control::Replacement) p.add({0xbea10380u});
    if (control == Control::Replacement) p.add({0x7d8200f9u, 0x0686a080u});
    if (control == Control::AcrossEdge) {
        p.add({0xbf068004u});
        p.branch(0xbf840000u, "carry");
        p.add({0x7e0c0280u});   // unrelated genuine word on the other predecessor
    }
    p.label("carry");
    const auto arithmetic = carry(op);
    p.add({arithmetic[0], arithmetic[1]});
    if (control == Control::LoopGenuine || control == Control::LoopPoison) {
        p.add({0x80048104u});   // ++s4
        if (control == Control::LoopPoison) p.add({0xbea10380u});   // lose the saved generation
        p.add({0xbf0a8204u});   // s4 < 2
        p.branch(0xbf850000u, "carry");
    }
    p.add({0xd5010004u, 0x00c10280u});   // v4 = 0/1 from the actual carry-out s48
    p.add({0xe0742000u, 0x80020300u, end});   // output[v0] = {v3,v4}
    return p.finish();
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& words, uint64_t address,
                              uint32_t path = 0) {
    ComputeShaderConfig config;
    config.local_x = config.wave_size = config.native_subgroup_size = 64;
    config.user_sgprs.resize(5);
    config.user_sgprs[4] = path;
    // Unknown physical launch authority is intentional: per-lane Bool reads do not grant words.
    ShaderResource output;
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 2;
    output.binding = 3;
    output.stride = 8;
    output.sgpr_base = 8;
    ShaderResourceTable table;
    table.resources.push_back(output);
    return recompile_compute(words.data(), words.size(), &table, config,
                             {RecompileDiagnosticStage::Compute, address});
}
void check_production_values(const std::vector<uint32_t>& module, uint32_t op, uint32_t threshold,
                             uint32_t path = 0) {
    bpermute_oracle::Interpreter vm(module);
    vm.native_subgroup_width = 64;
    vm.push_constants = std::vector<uint32_t>(5);
    vm.push_constants[4] = path;
    const auto actual = vm.run_buffers(64, {{3, std::vector<uint32_t>(128, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(vm.error.empty()) << vm.error;
    std::vector<uint32_t> expected;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const uint32_t cin = lane > threshold;
        const uint64_t a = op == 0x12a ? UINT32_MAX : 1;
        const uint64_t c = op == 0x12a ? 1 : UINT32_MAX;
        expected.push_back(op == 0x128 ? static_cast<uint32_t>(a + c + cin)
                                       : static_cast<uint32_t>(a - c - cin));
        expected.push_back(op == 0x128 ? (a + c + cin > UINT32_MAX) : (a < c + cin));
    }
    EXPECT_EQ(actual, expected) << op;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    explicit Emitter(uint32_t wave = 64) {
        b.begin(4, nullptr, wave, 1, 1, wave, 0, true, true);
        b.native_subgroup_size = wave;
        b.allow_b32_masks = wave == 32;
        b.diagnostic = {RecompileDiagnosticStage::Compute, 0xa5097000ull};
        rs.exec = b.btrue();
        rs.vcc = rs.scc = b.bfalse();
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.vreg[0] = b.subgroup_local_id();
        rs.vreg[1] = b.uconst(UINT32_MAX);
        rs.vreg[3] = b.uconst(17);
    }
    void prove(const std::vector<uint32_t>& words) {
        const auto ins = decode(words);
        seed_vector_word_definedness(rs, ins, false, b.wave_size);
        initialize_saved_mask_word_proof(b, ins, words.data(), words.size(), 0, &rs);
    }
    bool emit(const Rdna2Inst& in) {
        prepare_saved_mask_word_write(b, rs, in);
        const auto before = snapshot_saved_b64_masks(b, rs, in);
        bool ok = true;
        if (!emit_alu(b, rs, in, ok, true) || !ok) return false;
        record_scalar_write(rs, in, false, before);
        retain_saved_mask_words(b, rs, in);
        return true;
    }
};
void unavailable_predicate(Emitter& e, const std::vector<uint32_t>& words) {
    e.prove(words);
    ASSERT_TRUE(e.emit(decode(words).front()));
    ASSERT_TRUE(e.rs.sreg_bool.contains(32));
    ASSERT_TRUE(e.rs.unavailable_saved_mask_words.contains(32));
    ASSERT_TRUE(e.rs.unavailable_saved_mask_words.contains(33));
    ASSERT_FALSE(e.rs.sreg.contains(32));
    ASSERT_FALSE(e.rs.sreg.contains(33));
}
}   // namespace

TEST(CarryPredicateRole, AllThreeOpcodesReadTheCurrentBoolWithoutPhysicalWords) {
    for (uint32_t op : operations) {
        SCOPED_TRACE(op);
        Emitter e;
        const auto words = program(op);
        unavailable_predicate(e, words);
        const auto in = decode(words)[1];
        EXPECT_FALSE(scalar_source_marks(e.rs, in).placeholder);
        EXPECT_TRUE(e.emit(in));
        EXPECT_FALSE(e.rs.sreg.contains(32));
        EXPECT_FALSE(e.rs.sreg.contains(33));
    }
}

TEST(CarryPredicateRole, EmittedAdditionAndBothBorrowOrdersHaveExactLaneValues) {
    for (uint32_t op : operations) {
        Emitter e;
        const auto words = program(op);
        unavailable_predicate(e, words);
        ASSERT_TRUE(e.emit(decode(words)[1]));
        e.b.store_output_word(e.rs.vreg.at(3), 2, 0);
        e.b.store_output_word(e.b.sel(e.rs.sreg_bool.at(48), e.b.uconst(1), e.b.uconst(0)), 2, 1);
        const auto module = e.b.finish();
        retain_module(module, op);
        bpermute_oracle::Interpreter vm(module);
        vm.native_subgroup_width = 64;
        const auto actual = vm.run_buffers(64, {{1, std::vector<uint32_t>(128, 0xdeadbeef)}}, 1);
        ASSERT_TRUE(vm.error.empty()) << vm.error;
        std::vector<uint32_t> expected;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const uint32_t cin = lane > 31;
            const uint64_t a = op == 0x12a ? UINT32_MAX : 1;
            const uint64_t c = op == 0x12a ? 1 : UINT32_MAX;
            expected.push_back(op == 0x128 ? static_cast<uint32_t>(a + c + cin)
                                           : static_cast<uint32_t>(a - c - cin));
            expected.push_back(op == 0x128 ? (a + c + cin > UINT32_MAX) : (a < c + cin));
        }
        EXPECT_EQ(actual, expected) << "operation " << op;
    }
}

TEST(CarryPredicateRole, NumericVectorDestinationAliasUsesItsOldGenuineInput) {
    for (uint32_t op : operations) {
        Emitter e;
        const auto words = program(op);
        unavailable_predicate(e, words);
        const auto alias = carry(op, 32, 1);
        const auto in = decode(alias).front();
        EXPECT_FALSE(scalar_source_marks(e.rs, in).placeholder);
        EXPECT_TRUE(e.emit(in));
    }
}

TEST(CarryPredicateRole, MissingCurrentBoolCannotBeInventedFromPhysicalStorage) {
    for (uint32_t op : operations) {
        Emitter e;
        const auto words = program(op);
        unavailable_predicate(e, words);
        e.rs.sreg_bool.erase(32);
        e.rs.sreg[32] = e.rs.sreg[33] = e.b.uconst(0);
        EXPECT_FALSE(e.emit(decode(words)[1]));
    }
}

TEST(CarryPredicateRole, EitherMergePoisonWordRetainsTheFreshPredicateGuard) {
    for (uint32_t op : operations)
        for (int word : {32, 33}) {
            Emitter e;
            const auto words = program(op);
            unavailable_predicate(e, words);
            e.rs.sreg_merge_placeholder.insert(word);
            EXPECT_TRUE(scalar_source_marks(e.rs, decode(words)[1]).placeholder);
            EXPECT_FALSE(e.emit(decode(words)[1]));
            EXPECT_EQ(e.b.stage_reject_reason, "fabricated-fresh-mask-source");
        }
}

TEST(CarryPredicateRole, NumericSourceRolesStillRequireTheirOwnWords) {
    for (uint32_t op : operations)
        for (uint32_t source = 0; source < 2; ++source) {
            Emitter e;
            const auto words = program(op);
            unavailable_predicate(e, words);
            const auto arithmetic =
                carry(op, 32, 3, source == 0 ? 32 : 129, source == 1 ? 32 : 257);
            EXPECT_FALSE(e.emit(decode(arithmetic).front()));
            EXPECT_EQ(e.b.stage_reject_reason, "fabricated-fresh-mask-source");
        }
}

TEST(CarryPredicateRole, APhysicalNumericReadStillHasNoMaskWordAuthority) {
    Emitter e;
    const auto words = program(0x128, {}, true);
    unavailable_predicate(e, words);
    const auto ins = decode(words);
    ASSERT_TRUE(e.emit(ins[1]));
    EXPECT_FALSE(e.emit(ins[2]));
}

TEST(CarryPredicateRole, EitherNumericHalfOverwriteRetiresTheOldBoolGeneration) {
    for (uint32_t op : operations)
        for (uint32_t word : {32u, 33u}) {
            Emitter e;
            const auto words = program(op, {0xbe800380u | (word << 16u)});
            unavailable_predicate(e, words);
            const auto ins = decode(words);
            ASSERT_TRUE(e.emit(ins[1]));
            EXPECT_FALSE(e.rs.sreg_bool.contains(32));
            EXPECT_FALSE(e.emit(ins[2]));
        }
}

TEST(CarryPredicateRole, AGenuineLaterCompareEstablishesOnlyItsNewGeneration) {
    Emitter e;
    const auto words = program(0x128, {0xbea10380u, 0x7d8200f9u, 0x0686a080u});
    unavailable_predicate(e, words);
    const auto old = e.rs.sreg_bool.at(32);
    const auto ins = decode(words);
    ASSERT_TRUE(e.emit(ins[1]));
    ASSERT_FALSE(e.rs.sreg_bool.contains(32));
    ASSERT_TRUE(e.emit(ins[2]));
    EXPECT_NE(e.rs.sreg_bool.at(32), old);
    EXPECT_TRUE(e.emit(ins[3]));
}

TEST(CarryPredicateRole, AFabricatedVectorProducerCannotPublishItsPredicate) {
    for (uint32_t op : operations) {
        Emitter e;
        const auto words = program(op);
        e.rs.vreg.erase(0);
        e.prove(words);
        EXPECT_FALSE(e.emit(decode(words).front()));
        EXPECT_FALSE(e.rs.sreg_bool.contains(32));
    }
}

TEST(CarryPredicateRole, AFabricatedScalarRelayCannotPublishItsPredicate) {
    Emitter e;
    const std::vector<uint32_t> words{0x7e000214u, 0x7d8200f9u, 0x0686a09fu,
                                      0xd5283003u, 0x00820281u, end};
    e.prove(words);
    const auto ins = decode(words);
    ASSERT_TRUE(e.emit(ins[0]));
    EXPECT_FALSE(e.emit(ins[1]));
    EXPECT_FALSE(e.rs.sreg_bool.contains(32));
}

TEST(CarryPredicateRole, BothGenuineEdgesRetainTheCurrentPredicateAtAJoin) {
    Emitter e;
    const auto words = program(0x128);
    unavailable_predicate(e, words);
    const auto edge = structured_saved_mask_edge(e.b, e.rs);
    EXPECT_TRUE(join_structured_saved_masks(e.b, e.rs, edge, edge, e.b.cur_block, e.b.cur_block,
                                            decode(words), 0, 2));
    EXPECT_TRUE(e.emit(decode(words)[1]));
}

TEST(CarryPredicateRole, AMissingOrOverwrittenEdgeCannotSupplyAMustBoolJoin) {
    for (bool overwrite : {false, true}) {
        Emitter e;
        const auto words = program(0x128);
        unavailable_predicate(e, words);
        const auto genuine = structured_saved_mask_edge(e.b, e.rs);
        if (overwrite) {
            ASSERT_TRUE(e.emit(decode({0xbea10380u}).front()));
        } else {
            e.rs.sreg_bool.erase(32);
        }
        const auto absent = structured_saved_mask_edge(e.b, e.rs);
        EXPECT_FALSE(join_structured_saved_masks(e.b, e.rs, genuine, absent, e.b.cur_block,
                                                 e.b.cur_block, decode(words), 0, 2));
        EXPECT_EQ(e.b.stage_reject_reason, "structured-saved-mask-domain-missing-edge");
    }
}

TEST(CarryPredicateRole, AConfinedNumericVectorInputRequiresTheSameExecOrItsSubset) {
    for (bool contained : {false, true}) {
        Emitter e;
        const auto words = program(0x128);
        unavailable_predicate(e, words);
        const uint32_t owner = e.b.ucmp(Op_ULessThan, e.rs.vreg.at(0), e.b.uconst(32));
        e.rs.exec = contained ? owner : e.b.logical_not(owner);
        e.rs.exec_narrowed = true;
        auto& marks = e.rs.vreg_source_marks[1];
        marks.placeholder = marks.hard = true;
        marks.outside_exec = owner;
        EXPECT_EQ(e.emit(decode(words)[1]), contained);
    }
}

TEST(CarryPredicateRole, ProductionModulesUseTheSavedPredicateAndBothCarryResults) {
    for (uint32_t op : operations) {
        const auto words = production_program(op);
        const auto module = compile(words, 0xa5097100ull + op);
        ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0xa5097100ull + op);
        retain_module(module, op);
        check_production_values(module, op, 31);
    }
}

TEST(CarryPredicateRole, ProductionMissingAndFabricatedProducersRemainRefused) {
    for (Control control : {Control::Absent, Control::Poison, Control::Relay}) {
        const uint64_t address = 0xa5097200ull + static_cast<uint32_t>(control);
        EXPECT_TRUE(compile(production_program(0x128, control), address).empty());
        const auto reason = last_terminal_reject_reason(address);
        EXPECT_NE(
            reason.find(control == Control::Absent ? "pc=2 " : "fabricated-fresh-mask-source"),
            std::string::npos)
            << reason;
    }
}

TEST(CarryPredicateRole, ProductionPartialOverwritesKeepTheOldGenerationRefused) {
    for (Control control : {Control::Low, Control::High}) {
        const uint64_t address = 0xa5097300ull + static_cast<uint32_t>(control);
        EXPECT_TRUE(compile(production_program(0x128, control), address).empty());
        EXPECT_NE(last_terminal_reject_reason(address).find("pc=5 "), std::string::npos)
            << last_terminal_reject_reason(address);
    }
}

TEST(CarryPredicateRole, ProductionReplacementHasANewCurrentPredicate) {
    const auto module = compile(production_program(0x128, Control::Replacement), 0xa5097400ull);
    ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(0xa5097400ull);
    retain_module(module, 0);
    check_production_values(module, 0x128, 0);
}

TEST(CarryPredicateRole, ProductionOneArmCannotSupplyTheCondition) {
    const auto words = production_program(0x128, Control::OneArm);
    EXPECT_TRUE(compile(words, 0xa5097500ull).empty());
    EXPECT_NE(last_terminal_reject_reason(0xa5097500ull).find("mask"), std::string::npos)
        << last_terminal_reject_reason(0xa5097500ull);
}

TEST(CarryPredicateRole, ProductionBothArmsAndPreservedEdgeKeepGenuineConditions) {
    for (Control control : {Control::BothArms, Control::AcrossEdge}) {
        const auto words = production_program(0x128, control);
        for (uint32_t path : {0u, 1u}) {
            const uint64_t address = 0xa5097600ull + static_cast<uint64_t>(control) * 2 + path;
            const auto module = compile(words, address, path);
            ASSERT_FALSE(module.empty()) << last_terminal_reject_reason(address);
            retain_module(module, static_cast<uint32_t>(control) * 2 + path);
            check_production_values(module, 0x128,
                                    control == Control::BothArms && path == 1 ? 0 : 31, path);
        }
    }
}

TEST(CarryPredicateRole, ProductionLoopRetainsOnlyAnUnchangedGenuinePredicate) {
    const auto genuine = compile(production_program(0x128, Control::LoopGenuine), 0xa5097700ull);
    ASSERT_FALSE(genuine.empty()) << last_terminal_reject_reason(0xa5097700ull);
    retain_module(genuine, 0);
    check_production_values(genuine, 0x128, 31);
    EXPECT_TRUE(compile(production_program(0x128, Control::LoopPoison), 0xa5097701ull).empty());
    EXPECT_NE(last_terminal_reject_reason(0xa5097701ull).find("mask"), std::string::npos)
        << last_terminal_reject_reason(0xa5097701ull);
}
TEST(CarryPredicateRole, Wave32UsesOnlyTheCompleteBoolWordBesideAnUnrelatedHighPoison) {
    for (uint32_t op : operations) {
        Emitter e(32);
        const auto words = program(op);
        e.prove(words);
        ASSERT_TRUE(e.emit(decode(words).front()));
        ASSERT_TRUE(e.rs.sreg_bool_b32.contains(32));
        e.rs.sreg_merge_placeholder.insert(33);
        EXPECT_FALSE(scalar_source_marks(e.rs, decode(words)[1]).placeholder);
        EXPECT_TRUE(e.emit(decode(words)[1]));
        EXPECT_TRUE(e.rs.sreg_merge_placeholder.contains(33));
    }
}

TEST(CarryPredicateRole, SpecialVccSpellingRetainsItsExistingPhysicalGuard) {
    Emitter e;
    const auto words = program(0x128);
    unavailable_predicate(e, words);
    e.rs.vcc = e.b.btrue();
    e.rs.unavailable_saved_mask_words.insert(106);
    const auto in = decode(carry(0x128, 106)).front();
    ASSERT_EQ(in.src[2].kind, OperandKind::Special);
    EXPECT_FALSE(e.emit(in));
    EXPECT_EQ(e.b.stage_reject_reason, "fabricated-fresh-mask-source");
}

TEST(CarryPredicateRole, AContainedExecSubsetPreservesTheNumericInputObligation) {
    Emitter e;
    const auto words = program(0x128);
    unavailable_predicate(e, words);
    const auto owner = e.b.ucmp(Op_ULessThan, e.rs.vreg.at(0), e.b.uconst(48));
    e.rs.exec = e.b.land(owner, e.b.ucmp(Op_ULessThan, e.rs.vreg.at(0), e.b.uconst(16)));
    e.rs.exec_subset_of[e.rs.exec] = owner;
    e.rs.exec_narrowed = true;
    auto& marks = e.rs.vreg_source_marks[1];
    marks.placeholder = marks.hard = true;
    marks.outside_exec = owner;
    marks.blanket_roots.insert(0x5097);
    EXPECT_TRUE(e.emit(decode(words)[1]));
    EXPECT_TRUE(e.b.loop_blanket_roots_read.contains(0x5097));
}

TEST(CarryPredicateRole, AFabricatedReplacementCannotReuseTheEarlierGenuinePredicate) {
    auto words = production_program(0x128);
    words.insert(words.begin() + 4, {0x7d820ef9u, 0x0686a09fu});
    EXPECT_TRUE(compile(words, 0xa5097800ull).empty());
    EXPECT_NE(last_terminal_reject_reason(0xa5097800ull).find("fabricated-fresh-mask-source pc=4"),
              std::string::npos)
        << last_terminal_reject_reason(0xa5097800ull);
}
}   // namespace prosper::gpu
