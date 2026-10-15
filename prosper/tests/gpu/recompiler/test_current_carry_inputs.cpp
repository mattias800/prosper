// #5092: carry-in arithmetic consumes two B32 values and a current Bool predicate.
// Requiring an unrelated high numeric word or ballot storage refuses genuine address math;
// treating an absent/retired predicate as carry zero would silently produce the wrong address.
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
using namespace prosper::gpu;
constexpr std::array<uint32_t, 3> carry_ops{0x128u, 0x129u, 0x12au};
constexpr uint32_t end = 0xbf810000u;
struct Result {
    std::vector<uint32_t> module;
    std::string reject;
};
Result compile(const std::vector<uint32_t>& code, uint32_t native_width = 64) {
    ComputeShaderConfig config;
    config.user_sgprs.assign(1, 0);
    config.tgid_x_en = true; // genuine dynamic s1 for all-predecessor controls
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = native_width;
    config.force_exchange_dispatcher = true;
    prosper::test::complete_compute_launch(config, 2);
    static uint64_t next = 0x5092000100ull;
    const uint64_t address = next++;
    auto module = recompile_compute(code.data(), code.size(), nullptr, config,
                                    {RecompileDiagnosticStage::Compute, address});
    if (!module.empty()) {
        if (const char* directory = std::getenv("PROSPER_CARRY_INPUT_SPV_DIRECTORY")) {
            const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
            const auto name = std::string(info->name()) + "-" + std::to_string(native_width) + "-" +
                              std::to_string(address) + ".spv";
            std::ofstream output(std::filesystem::path(directory) / name, std::ios::binary);
            output.write(reinterpret_cast<const char*>(module.data()),
                         static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
            EXPECT_TRUE(output.good())
                << "retain the actual translated module for strict validation";
        }
    }
    return {std::move(module), last_terminal_reject_reason(address)};
}
void accepted(const Result& r) {
    ASSERT_FALSE(r.module.empty()) << r.reject;
    bool dispatcher = false;
    for (std::size_t at = 5; at < r.module.size();) {
        const uint32_t count = r.module[at] >> 16;
        ASSERT_GT(count, 0u);
        ASSERT_LE(at + count, r.module.size());
        dispatcher |= (r.module[at] & 0xffffu) == 251u;
        at += count;
    }
    EXPECT_TRUE(dispatcher) << "the real dispatcher must consume the current predicate";
}
void refused(const Result& r, uint32_t pc, const std::string& reason = "mode=unresolved-operand") {
    EXPECT_TRUE(r.module.empty());
    EXPECT_NE(r.reject.find(reason), std::string::npos) << r.reject;
    EXPECT_NE(r.reject.find("pc=" + std::to_string(pc) + " "), std::string::npos) << r.reject;
}
std::vector<uint32_t> program(uint32_t op, uint32_t carry = 12, uint32_t out = 106) {
    return {
        0x7e020281u, // v1 = 1, genuine numeric source
        0xbe900381u, // s16 = 1, genuine explicit SDWA compare source
        0xbe910381u, // s17 = 1; unrelated s18 is deliberately absent
        0x7d8602f9u,
        0x06868c10u, // exact SDWA compare writes fresh s[12:13] Bool
        0xd4000001u | (op << 16u) | (out << 8u),
        17u | (257u << 9u) | (carry << 18u),
        end,
    };
}
}   // namespace

TEST(CurrentCarryInputs, NumericOperandsReadOnlyTheirB32Word) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        const auto code = program(op);
        const auto in = rdna2_decode_one(code.data() + 5, code.size() - 5);
        const auto compare = rdna2_decode_one(code.data() + 3, code.size() - 3);
        ASSERT_EQ(compare.src[0].kind, OperandKind::SGPR);
        ASSERT_EQ(compare.src[0].value, 16);
        ASSERT_EQ(compare.src[1].kind, OperandKind::VGPR);
        ASSERT_EQ(compare.src[1].value, 1);
        ASSERT_EQ(compare.dst.value, 12);
        ASSERT_EQ(in.opcode, op);
        ASSERT_EQ(in.n_src, 3u);
        EXPECT_EQ(in.src[0].value, 17);
        EXPECT_EQ(in.src[2].value, 12);
        EXPECT_EQ(scalar_alu_source_words(in, 0), 1u);
        EXPECT_EQ(scalar_alu_source_words(in, 1), 1u);
        EXPECT_EQ(scalar_alu_source_words(in, 2), 2u);
    }
}
TEST(CurrentCarryInputs, FreshSdwaBoolAndSingleNumericWordCompile) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        accepted(compile(program(op)));
    }
}
TEST(CurrentCarryInputs, PortableFreshSdwaBoolAndSingleNumericWordCompile) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        accepted(compile(program(op), 0));
    }
}
TEST(CurrentCarryInputs, MissingNumericSourceCannotBorrowItsNeighbor) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code[2] = 0xbe920381u; // s18 present, but actual s17 absent
        refused(compile(code), 5);
    }
}
TEST(CurrentCarryInputs, MissingVectorSourceCannotBorrowTheFreshPredicate) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        // The second source is the absent s18: never loaded, so hard on every route. (A never-
        // written v2 would not do: under #5185 it is arbitrary on every route, #5039.)
        code[6] = 17u | (18u << 9u) | (12u << 18u);
        refused(compile(code), 5);
    }
}
TEST(CurrentCarryInputs, AbsentPredicateCannotBeInventedFromPhysicalData) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op, 22);
        code.insert(code.begin() + 5,
                    {0xbe960381u, 0xbe970380u}); // genuine s[22:23] DATA, no Bool
        refused(compile(code), 7);
    }
}
TEST(CurrentCarryInputs, MissingPredicateCannotBorrowAnUnrelatedFreshMask) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        refused(compile(program(op, 22)), 5);
    }
}
TEST(CurrentCarryInputs, HighWordReplacementRetiresTheOldB64Predicate) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code.insert(code.begin() + 5, 0xbe8d0380u); // s13 numeric overwrite retires pair12
        refused(compile(code), 6);
    }
}
TEST(CurrentCarryInputs, CarryOutputAliasReadsTheOldPredicateBeforeWriting) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        accepted(compile(program(op, 12, 12)));
    }
}
TEST(CurrentCarryInputs, BothDynamicPredecessorsSupplyTheirCurrentPredicate) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code.insert(code.begin() + 3, {0xbf068001u, 0xbf840002u, 0x7d8602f9u, 0x06868c10u});
        accepted(compile(code));
    }
}
TEST(CurrentCarryInputs, OneDynamicPredecessorCannotBorrowTheOthersPredicate) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code.insert(code.begin() + 3, {0xbf068001u, 0xbf840002u});
        refused(compile(code), 7, "reason=wave64-ambiguous-mask-read");
    }
}

TEST(CurrentCarryInputs, SuppliedNumericNeighborIsAConservativeBaselineControl) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        // Inventory old numeric-mask storage before an independent SDWA compare overwrites it.
        code.insert(code.begin() + 3, 0x878c7e7eu);   // s[12:13] = EXEC & EXEC
        code.insert(code.begin() + 6,
                    {0xbe920380u, 0xbf820000u});   // genuine s18 isolates Bool role
        accepted(compile(code));
    }
}
TEST(CurrentCarryInputs, PortableSuppliedNumericNeighborIsAConservativeBaselineControl) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code.insert(code.begin() + 3, 0x878c7e7eu);
        code.insert(code.begin() + 6, {0xbe920380u, 0xbf820000u});
        accepted(compile(code, 0));
    }
}
TEST(CurrentCarryInputs, CopiedPredicateKeepsItsIndependentCurrentGeneration) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op, 20);
        code.insert(code.begin() + 5, {0xbe94040cu, 0xbe8d0380u});   // copy then retire original12
        accepted(compile(code));
    }
}
TEST(CurrentCarryInputs, RelayFromAnAbsentPredicateDoesNotCreateCarryAuthority) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op, 20);
        code.insert(code.begin() + 5, 0xbe940418u);   // absent s[24:25] relayed into s[20:21]
        refused(compile(code), 6);
    }
}

TEST(CurrentCarryInputs, ScalarNumericSrc1ReadsOnlyItsActualB32Word) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code[6] = 257u | (17u << 9u) | (12u << 18u);
        accepted(compile(code));
        accepted(compile(code, 0));
    }
}
TEST(CurrentCarryInputs, MissingScalarNumericSrc1CannotBorrowItsNeighbor) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code[2] = 0xbe920381u;
        code[6] = 257u | (17u << 9u) | (12u << 18u);
        refused(compile(code), 5);
    }
}
TEST(CurrentCarryInputs, AnUnrelatedDirtyNumericNeighborDoesNotPoisonTheRealSource) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code.insert(code.begin() + 5, 0xbe920309u);   // missing s9 poisons s18, never reads s17
        accepted(compile(code));
    }
}
TEST(CurrentCarryInputs, CarryDestinationAliasReadsItsGenuineNumericInputBeforeReplacement) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        accepted(compile(program(op, 12, 17)));
    }
}
TEST(CurrentCarryInputs, ANewMissingNumericGenerationCannotUseAnEarlierGenuineValue) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        code.insert(code.begin() + 5, 0xbe910309u);   // replace actual s17 with missing s9
        refused(compile(code), 6);
    }
}
namespace {
std::vector<uint32_t> numeric_join(uint32_t op, bool genuine) {
    auto code = program(op);
    code.erase(code.begin() + 2);   // the original numeric definition is supplied by the two arms
    code.insert(code.begin() + 2, {0xbf068001u, 0xbf840002u, 0xbe910381u, 0xbf820001u,
                                   genuine ? 0xbe910381u : 0xbe910309u});
    return code;
}
std::vector<uint32_t> numeric_loop(uint32_t op, bool genuine) {
    auto code = program(op);
    code.insert(code.begin() + 5, 0xbe860380u);   // genuine counter before the header carry
    code.insert(code.end() - 1,
                {genuine ? 0xbe910381u : 0xbe910309u, 0x80068106u, 0xbf0a8206u, 0xbf85fffau});
    return code;
}
}   // namespace
TEST(CurrentCarryInputs, EveryDynamicPredecessorSuppliesTheGenuineNumericWord) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        accepted(compile(numeric_join(op, true)));
    }
}
TEST(CurrentCarryInputs, OneDynamicPredecessorCannotBorrowTheOthersNumericWord) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        refused(compile(numeric_join(op, false)), 9);
    }
}
TEST(CurrentCarryInputs, EveryLoopBackedgeSuppliesTheCurrentGenuineNumericGeneration) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        accepted(compile(numeric_loop(op, true)));
    }
}
TEST(CurrentCarryInputs, ALoopBackedgeCannotReviveTheEarlierGenuineNumericGeneration) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        refused(compile(numeric_loop(op, false)), 6);
    }
}
TEST(CurrentCarryInputs, ThePredicateStillConsumesBothMaskHalvesAtAMerge) {
    for (uint32_t op : carry_ops) {
        SCOPED_TRACE(op);
        auto code = program(op);
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data() + 5, code.size() - 5, ins);
        ASSERT_EQ(ins.size(), 2u);
        EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 12));
        EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 13));
        EXPECT_TRUE(sgpr_dead_at_merge(ins, 0, 14));   // unrelated neighbor remains dead
    }
}
