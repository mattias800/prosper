// #3984: validated PC-relative specialization must retain the original Bool-copy
// generation proof through its real counted postlude, without granting DATA words.
#include "fixtures/fragment_dispatch_bool_copy.hpp"
#include "fragment_constant_output_oracle.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::fragment_dispatch_bool_copy;
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return ins;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint32_t target = 17,
                              const ShaderResourceTable* resources = nullptr) {
    return recompile_fragment(code.data(), code.size(), resources, nullptr, target);
}
size_t ballots(const std::vector<uint32_t>& module) {
    size_t count = 0;
    for (size_t at = 5; at < module.size();) {
        const uint32_t width = module[at] >> 16;
        if (!width || width > module.size() - at) return SIZE_MAX;
        count += (module[at] & 0xffffu) == 339;
        at += width;
    }
    return count;
}
bool proves(const std::vector<uint32_t>& code, const std::vector<Rdna2Inst>& ins, uint32_t pc) {
    SpirvCompute b;
    b.wave_size = 64;
    b.begin_fragment();
    b.saved_mask_original_words = code;
    RegState rs;
    rs.exec = b.btrue();
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    initialize_native_logical_data_demand(b, ins, &rs);
    return b.fragment_bool_only_vcc_copy_pcs.contains(pc);
}
// Hand-built SSA records calibrate the evaluator, not an executable SPIR-V module.
std::vector<uint32_t> oracle_graph() {
    return {0x07230203u,
            0x00010600u,
            0u,
            32u,
            0u,
            (4u << 16) | 21u,
            1,
            32,
            0,
            (3u << 16) | 22u,
            2,
            32,
            (2u << 16) | 20u,
            3,
            (4u << 16) | 23u,
            4,
            2,
            4,
            (4u << 16) | 32u,
            5,
            3,
            4,
            (4u << 16) | 59u,
            5,
            6,
            3,
            (4u << 16) | 43u,
            1,
            7,
            1,
            (4u << 16) | 43u,
            1,
            8,
            9,
            (4u << 16) | 124u,
            2,
            9,
            7,
            (4u << 16) | 124u,
            2,
            10,
            8,
            (3u << 16) | 41u,
            3,
            11,
            (6u << 16) | 169u,
            2,
            12,
            11,
            10,
            9,
            (7u << 16) | 80u,
            4,
            13,
            12,
            12,
            12,
            12,
            (3u << 16) | 62u,
            6,
            13};
}
size_t instruction_at(const std::vector<uint32_t>& graph, uint32_t op) {
    for (size_t at = 5; at < graph.size(); at += graph[at] >> 16)
        if ((graph[at] & 0xffffu) == op) return at;
    return graph.size();
}
} // namespace

TEST(FragmentDispatchBoolCopy, ActualDispatchCountedPostludePreservesTheSelectedValue) {
    for (bool equal : {false, true}) {
        for (uint32_t iterations : {0u, 3u}) {
            const auto code = fixture::program(fixture::Tail::Bool, equal, iterations);
            const auto recipe = rdna2_pcrel_dispatch_info(code.data(), code.size());
            ASSERT_TRUE(recipe.valid);
            auto selected = decode(code);
            ASSERT_TRUE(rdna2_specialize_pcrel_dispatch(selected, recipe, 17));
            ASSERT_TRUE(detect_counted_loop(selected).found);
            const auto copy = std::find_if(selected.begin(), selected.end(), [](const auto& in) {
                return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                       in.dst.value == 106 && in.src[0].value == 20;
            });
            ASSERT_NE(copy, selected.end());
            ASSERT_TRUE(proves(code, decode(code), copy->pc))
                << "original MUST source is reachable";
            ASSERT_TRUE(proves(code, selected, copy->pc));
            const auto module = compile(code);
            ASSERT_FALSE(module.empty());
            const auto actual = prosper::test::fragment_constant_output(module);
            if (actual.has_value()) {
                EXPECT_EQ(*actual, std::vector<uint32_t>(4, equal ? 9u : 1u));
            } else {
                FAIL() << "constant SSA oracle refused, not a value verdict";
            }
            EXPECT_EQ(ballots(module), 0u);
        }
    }
}

TEST(FragmentDispatchBoolCopy, ExistingMaskEarlyOutDoesNotGrantPhysicalWords) {
    const auto code = fixture::program(fixture::Tail::Bool, false, 3, true);
    const auto module = compile(code);
    ASSERT_FALSE(module.empty());
    EXPECT_EQ(ballots(module), 0u);
    TerminalRejectCapture rejection;
    EXPECT_TRUE(compile(fixture::program(fixture::Tail::HighData, false, 3, true)).empty());
    const auto records = rejection.take();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NE(records[0].second.find("saved-mask-copy-words-unavailable"), std::string::npos);
}

TEST(FragmentDispatchBoolCopy, SameGenerationSurvivorGateRetainsTheKill) {
    const auto code =
        fixture::program(fixture::Tail::Bool, false, 3, true, fixture::EarlyOut::Killed);
    const auto module = compile(code);
    ASSERT_FALSE(module.empty());
    size_t kills = 0;
    for (size_t at = 5; at < module.size(); at += module[at] >> 16)
        kills += (module[at] & 0xffffu) == 252; // structural OpKill; no CPU raster claim
    EXPECT_EQ(kills, 1u);
    EXPECT_EQ(ballots(module), 0u);
}

TEST(FragmentDispatchBoolCopy, ZeroSurvivorCannotEnterANonterminatingInteriorLoop) {
    const auto code = fixture::program(fixture::Tail::Bool, false, 3, true,
                                       fixture::EarlyOut::NonTerminatingInteriorLoop);
    auto selected = decode(code);
    const auto recipe = rdna2_pcrel_dispatch_info(code.data(), code.size());
    ASSERT_TRUE(recipe.valid);
    ASSERT_TRUE(rdna2_specialize_pcrel_dispatch(selected, recipe, 17));
    const auto loop = detect_counted_loop(selected);
    ASSERT_TRUE(loop.found) << "the shared detector proves shape, not termination";
    const auto step = std::find_if(selected.begin(), selected.end(), [&](const auto& in) {
        return in.pc > loop.exit_branch_pc && in.pc < loop.backedge_pc &&
               in.fmt == Rdna2Format::SOP2 && in.opcode == 0 && in.dst.value == 2;
    });
    ASSERT_NE(step, selected.end());
    ASSERT_EQ(step->src[1].kind, OperandKind::InlineInt);
    ASSERT_EQ(step->src[1].value, 0);
    const auto copy = std::find_if(selected.begin(), selected.end(), [](const auto& in) {
        return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
               in.dst.value == 106 && in.src[0].value == 20;
    });
    ASSERT_NE(copy, selected.end());
    ASSERT_TRUE(proves(code, decode(code), copy->pc));
    ASSERT_TRUE(proves(code, selected, copy->pc));
    TerminalRejectCapture rejection;
    EXPECT_TRUE(compile(code).empty());
    const auto records = rejection.take();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NE(
        records[0].second.find("saved-mask-copy-words-unavailable pc=" + std::to_string(copy->pc)),
        std::string::npos);
}

TEST(FragmentDispatchBoolCopy, OverflowWrongSignAndReplacedInductionCannotGrantContinuation) {
    for (auto early :
         {fixture::EarlyOut::OverflowInteriorLoop, fixture::EarlyOut::WrongSignInteriorLoop,
          fixture::EarlyOut::ReplacedInductionWord}) {
        SCOPED_TRACE(static_cast<int>(early));
        const auto code = fixture::program(fixture::Tail::Bool, false, 3, true, early);
        auto selected = decode(code);
        ASSERT_TRUE(rdna2_specialize_pcrel_dispatch(
            selected, rdna2_pcrel_dispatch_info(code.data(), code.size()), 17));
        ASSERT_TRUE(detect_counted_loop(selected).found);
        const auto copy = std::find_if(selected.begin(), selected.end(), [](const auto& in) {
            return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                   in.dst.value == 106 && in.src[0].value == 20;
        });
        ASSERT_NE(copy, selected.end());
        ASSERT_TRUE(proves(code, decode(code), copy->pc));
        ASSERT_TRUE(proves(code, selected, copy->pc));
        TerminalRejectCapture rejection;
        EXPECT_TRUE(compile(code).empty());
        const auto records = rejection.take();
        ASSERT_EQ(records.size(), 1u);
        EXPECT_NE(records[0].second.find("saved-mask-copy-words-unavailable pc=" +
                                         std::to_string(copy->pc)),
                  std::string::npos);
    }
}

TEST(FragmentDispatchBoolCopy, BackedgePathsCannotSkipAnInductionProducer) {
    for (auto early : {fixture::EarlyOut::SkippedInductionStep,
                       fixture::EarlyOut::SkippedInductionInitializer}) {
        SCOPED_TRACE(static_cast<int>(early));
        const auto code = fixture::program(fixture::Tail::Bool, false, 3, true, early);
        auto selected = decode(code);
        ASSERT_TRUE(rdna2_specialize_pcrel_dispatch(
            selected, rdna2_pcrel_dispatch_info(code.data(), code.size()), 17));
        // These are decoded, closed guest edges. Existing emitters may also refuse
        // them; no claim that this is the first refusal or an isolated permit control.
        EXPECT_TRUE(compile(code).empty());
    }
}

TEST(FragmentDispatchBoolCopy, SignedAndUnsignedUnitStepsReachTheirBoundsBeforeWrap) {
    for (bool signed_compare : {false, true}) {
        for (uint32_t bound : {131u, 193u}) { // 3, or signed -1 / unsigned UINT32_MAX
            auto code =
                fixture::program(fixture::Tail::Bool, false, 3, true, fixture::EarlyOut::Killed);
            const auto original = decode(code);
            const auto loop = detect_counted_loop(original);
            ASSERT_TRUE(loop.found);
            const auto comparison =
                std::find_if(original.begin(), original.end(), [&](const auto& in) {
                    return in.pc >= loop.header_pc && in.pc < loop.exit_branch_pc &&
                           in.fmt == Rdna2Format::SOPC;
                });
            ASSERT_NE(comparison, original.end());
            code[comparison->pc] = (signed_compare ? 0xbf040002u : 0xbf0a0002u) | (bound << 8);
            // Offline compilation only: the maximal unsigned bound is not executed.
            const auto module = compile(code);
            ASSERT_FALSE(module.empty());
            EXPECT_EQ(ballots(module), 0u);
            size_t kills = 0;
            for (size_t at = 5; at < module.size(); at += module[at] >> 16)
                kills += (module[at] & 0xffffu) == 252;
            EXPECT_EQ(kills, 1u);
        }
    }
}

TEST(FragmentDispatchBoolCopy, BranchPolarityAndSurvivorGenerationCannotGrantLineage) {
    for (auto early :
         {fixture::EarlyOut::MissingRestore, fixture::EarlyOut::StaleSurvivor,
          fixture::EarlyOut::HighSurvivorOverwrite, fixture::EarlyOut::OverlappingSurvivorOverwrite,
          fixture::EarlyOut::ConditionalSurvivorOverwrite, fixture::EarlyOut::BypassFinalGate,
          fixture::EarlyOut::NumericScc, fixture::EarlyOut::Scc1,
          fixture::EarlyOut::ExportBeforeGate, fixture::EarlyOut::ReenableAfterGate}) {
        SCOPED_TRACE(static_cast<int>(early));
        const auto code = fixture::program(fixture::Tail::Bool, false, 3, true, early);
        const auto instructions = decode(code);
        const auto copy =
            std::find_if(instructions.begin(), instructions.end(), [](const auto& in) {
                return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                       in.dst.value == 106 && in.src[0].value == 20;
            });
        ASSERT_NE(copy, instructions.end());
        ASSERT_TRUE(proves(code, instructions, copy->pc)) << "otherwise genuine original copy";
        TerminalRejectCapture rejection;
        EXPECT_TRUE(compile(code).empty()) << static_cast<int>(early);
        const auto records = rejection.take();
        EXPECT_EQ(records.size(), 1u);
        if (records.size() != 1u) continue;
        EXPECT_NE(records[0].second.find("saved-mask-copy-words-unavailable pc=" +
                                         std::to_string(copy->pc)),
                  std::string::npos)
            << "an earlier unrelated refusal does not validate the lineage guard";
    }
}

TEST(FragmentDispatchBoolCopy, RawTailReentryAndPhysicalWordDemandAreNotTerminalKills) {
    for (auto early : {fixture::EarlyOut::TailReentry, fixture::EarlyOut::TailData}) {
        SCOPED_TRACE(static_cast<int>(early));
        const auto code = fixture::program(fixture::Tail::Bool, false, 3, true, early);
        auto instructions = decode(code);
        const auto recipe = rdna2_pcrel_dispatch_info(code.data(), code.size());
        ASSERT_TRUE(recipe.valid);
        ASSERT_TRUE(rdna2_specialize_pcrel_dispatch(instructions, recipe, 17));
        const auto copy =
            std::find_if(instructions.begin(), instructions.end(), [](const auto& in) {
                return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                       in.dst.value == 106 && in.src[0].value == 20;
            });
        ASSERT_NE(copy, instructions.end());
        const uint32_t copy_pc = copy->pc;
        const bool closed = rdna2_append_closed_tail_blocks(code.data(), code.size(), instructions);
        EXPECT_EQ(closed, early == fixture::EarlyOut::TailData);
        TerminalRejectCapture rejection;
        EXPECT_TRUE(compile(code).empty());
        const auto records = rejection.take();
        EXPECT_EQ(records.size(), 1u);
        if (records.size() != 1u) continue;
        EXPECT_NE(records[0].second.find("saved-mask-copy-words-unavailable pc=" +
                                         std::to_string(copy_pc)),
                  std::string::npos);
    }
}

TEST(FragmentDispatchBoolCopy, SkippedBodyAndTerminalTailStoresCannotUseLateDiscard) {
    ShaderResourceTable resources;
    ShaderResource output{};
    output.cls = ResourceClass::ConstantBuffer;
    output.format = DataFormat::Uint32;
    output.num_components = 1;
    output.binding = 0;
    output.stride = 4;
    output.sgpr_base = 8;
    resources.resources.push_back(output);
    for (auto early : {fixture::EarlyOut::LateStore, fixture::EarlyOut::TailStore}) {
        SCOPED_TRACE(static_cast<int>(early));
        const auto code = fixture::program(fixture::Tail::Bool, false, 3, true, early);
        auto instructions = decode(code);
        ASSERT_TRUE(rdna2_append_closed_tail_blocks(code.data(), code.size(), instructions));
        const auto copy =
            std::find_if(instructions.begin(), instructions.end(), [](const auto& in) {
                return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                       in.dst.value == 106 && in.src[0].value == 20;
            });
        ASSERT_NE(copy, instructions.end());
        TerminalRejectCapture rejection;
        EXPECT_TRUE(compile(code, 17, &resources).empty());
        const auto records = rejection.take();
        EXPECT_EQ(records.size(), 1u);
        if (records.size() != 1u) continue;
        EXPECT_NE(records[0].second.find("saved-mask-copy-words-unavailable pc=" +
                                         std::to_string(copy->pc)),
                  std::string::npos);
    }
}

TEST(FragmentDispatchBoolCopy, OriginalSetpcDemandSurvivesRemovedSetupAndAlternateEntry) {
    const auto code = fixture::retained_copy_before_dispatch();
    const auto recipe = rdna2_pcrel_dispatch_info(code.data(), code.size());
    ASSERT_TRUE(recipe.valid);
    ASSERT_NE(std::find(recipe.target_pcs.begin(), recipe.target_pcs.end(), 23),
              recipe.target_pcs.end());
    auto selected = decode(code);
    ASSERT_TRUE(rdna2_specialize_pcrel_dispatch(selected, recipe, 25));
    EXPECT_FALSE(proves(code, decode(code), 6)) << "SETPC demands all original current words";
    EXPECT_TRUE(proves(code, selected, 6)) << "selected-only deadness is deliberately insufficient";
    TerminalRejectCapture rejection;
    EXPECT_TRUE(compile(code, 25).empty());
    const auto records = rejection.take();
    ASSERT_EQ(records.size(), 1u);
    EXPECT_NE(records[0].second.find("saved-mask-copy-words-unavailable pc=6"), std::string::npos);
    // Same dispatch/alternate-entry shape with real physical source words is supported.
    EXPECT_FALSE(compile(fixture::retained_copy_before_dispatch(true), 25).empty());
}

TEST(FragmentDispatchBoolCopy, NumericRelayAndOneArmReplacementStillRequireWords) {
    for (auto tail : {fixture::Tail::LowData, fixture::Tail::HighData, fixture::Tail::RelayData,
                      fixture::Tail::OneArmReplacement}) {
        const auto code = fixture::program(tail);
        TerminalRejectCapture rejection;
        EXPECT_TRUE(compile(code).empty());
        const auto records = rejection.take();
        ASSERT_EQ(records.size(), 1u);
        const auto ins = decode(code);
        const auto copy = std::find_if(ins.begin(), ins.end(), [](const auto& in) {
            return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                   in.dst.value == 106 && in.src[0].value == 20;
        });
        ASSERT_NE(copy, ins.end());
        EXPECT_NE(records[0].second.find("saved-mask-copy-words-unavailable pc=" +
                                         std::to_string(copy->pc)),
                  std::string::npos);
    }
}

TEST(FragmentDispatchBoolCopy, GenuinePairAndCompleteReplacementRemainAvailable) {
    for (auto tail : {fixture::Tail::NumericPair, fixture::Tail::CompleteReplacement})
        EXPECT_FALSE(compile(fixture::program(tail)).empty());
}

TEST(FragmentDispatchBoolCopy, PublicSelectedVectorCannotManufactureThePrivateLineage) {
    const auto code = fixture::program();
    auto original = decode(code);
    auto selected = original;
    ASSERT_TRUE(rdna2_specialize_pcrel_dispatch(
        selected, rdna2_pcrel_dispatch_info(code.data(), code.size()), 17));
    SpirvCompute b;
    b.wave_size = 64;
    b.begin_fragment();
    RegState rs;
    rs.exec = b.btrue();
    rs.vcc = b.bfalse();
    rs.scc = b.bfalse();
    rs.native_saved_mask_words = true;
    rs.absent_sgpr_reads_fabricated_zero = true;
    initialize_saved_mask_word_proof(b, original, code.data(), code.size(), 0, &rs);
    EXPECT_FALSE(
        emit_body(b, rs, selected, {}, nullptr, true, false, {}, code.data(), code.size()));
    const auto copy = std::find_if(selected.begin(), selected.end(), [](const auto& in) {
        return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
               in.dst.value == 106 && in.src[0].value == 20;
    });
    ASSERT_NE(copy, selected.end());
    EXPECT_EQ(b.stage_reject_pc, copy->pc);
    EXPECT_EQ(b.stage_reject_reason, "saved-mask-copy-words-unavailable");
    EXPECT_TRUE(b.fragment_bool_only_vcc_copy_pcs.empty());
}

TEST(FragmentDispatchBoolCopy, InvalidTargetOrRawRecipeDoesNotBorrowAnotherSelection) {
    auto code = fixture::program();
    EXPECT_TRUE(compile(code, 18).empty());
    code.back() = 0xffffffffu; // corrupt a qword target's high word
    EXPECT_TRUE(compile(code).empty());
}

TEST(FragmentDispatchBoolCopy, OrdinaryScalarBranchCannotUseTheMaskEarlyOutPolicy) {
    auto code = fixture::program(fixture::Tail::Bool, false, 3, true);
    const auto original = decode(code);
    const auto branch = std::find_if(original.begin(), original.end(), [](const auto& in) {
        return in.fmt == Rdna2Format::SOPP && in.opcode == 0x04 && in.simm16 > 2;
    });
    ASSERT_NE(branch, original.end());
    const auto mask_pcs = mask_test_branches_for_test(code.data(), code.size(), false);
    ASSERT_NE(std::find(mask_pcs.begin(), mask_pcs.end(), branch->pc), mask_pcs.end());
    code[branch->pc - 1] = fixture::mov(44, 129); // ordinary scalar replacement, same target
    const auto ordinary_pcs = mask_test_branches_for_test(code.data(), code.size(), false);
    EXPECT_EQ(std::find(ordinary_pcs.begin(), ordinary_pcs.end(), branch->pc), ordinary_pcs.end());
    EXPECT_TRUE(compile(code).empty());
}

TEST(FragmentDispatchBoolCopy, ConstantOracleKnownValuesChangeWithTheCondition) {
    auto graph = oracle_graph();
    EXPECT_EQ(prosper::test::fragment_constant_output(graph), std::vector<uint32_t>(4, 9u));
    graph[instruction_at(graph, 41)] = (3u << 16) | 42u;
    EXPECT_EQ(prosper::test::fragment_constant_output(graph), std::vector<uint32_t>(4, 1u));
}

TEST(FragmentDispatchBoolCopy, ConstantOracleDeclinesUnsupportedAndMalformedGraphs) {
    EXPECT_FALSE(prosper::test::fragment_constant_output({}).has_value());
    const std::vector<uint32_t> malformed{0x07230203u, 0, 0, 0, 0, 0};
    EXPECT_FALSE(prosper::test::fragment_constant_output(malformed).has_value());
    for (unsigned mutation : {0u, 1u, 2u}) {
        auto graph = oracle_graph();
        if (mutation == 0)
            graph[instruction_at(graph, 169) + 3] = 7; // numeric condition
        else if (mutation == 1)
            graph[instruction_at(graph, 124) + 3] = 12; // cycle through the unselected arm
        else {
            const auto at = instruction_at(graph, 124);
            graph[at] = (4u << 16) | 83u; // valid typed CopyObject is deliberately unsupported
            graph[at + 3] = 10;
        }
        EXPECT_FALSE(prosper::test::fragment_constant_output(graph).has_value());
    }
}
} // namespace prosper::gpu
