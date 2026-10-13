// #4957: a missing Bool spelling at a structured merge is not a zero physical mask.
// Refuse live mixed domains; preserve actual incoming predicates on both dynamic paths.
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr uint64_t incoming = 0x80000001abcdef01ull;
constexpr uint64_t left = 0x1122334455667788ull, right = 0xfedcba9876543210ull;
constexpr uint32_t mov(uint32_t dst, uint32_t src, bool pair = false) {
    return 0xbe800000u | (dst << 16u) | ((pair ? 4u : 3u) << 8u) | src;
}
constexpr uint32_t logical(uint32_t op, uint32_t dst = 24) {
    return 0x80000000u | (op << 23u) | (dst << 16u) | (32u << 8u) | 30u;
}
void select(std::vector<uint32_t>& code, uint32_t root = 24) {
    code.insert(code.end(), {0xd5010000u, 129u | (137u << 9u) | (root << 18u)});
}
std::vector<uint32_t> one_arm(bool relay = false) {
    std::vector<uint32_t> code{0xbf068014u, 0xbf840001u, logical(0x13)};
    if (relay) code.push_back(mov(36, 24, true));
    select(code, relay ? 36 : 24);
    code.push_back(end);
    return code;
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    uint32_t storage;
    Emitter() {
        b.wave_size = 64;
        b.begin_fragment();
        b.diagnostic = {RecompileDiagnosticStage::Fragment, 0xa4957001ull};
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.scc = b.btrue();
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.vreg[0] = b.fconst(0.5f);
        rs.sreg_bool[30] = constant_mask_bit(b, uint32_t(left), uint32_t(left >> 32u));
        rs.sreg_bool[32] = constant_mask_bit(b, uint32_t(right), uint32_t(right >> 32u));
        b.declare_cbufs();
        storage = b.cbuf_load(b.uconst(0), 2);
        rs.sreg[20] = storage;
    }
    void physical() {
        rs.sreg[24] = b.uconst(uint32_t(incoming));
        rs.sreg[25] = b.uconst(uint32_t(incoming >> 32u));
    }
    void predicate(uint64_t bits = incoming) {
        rs.sreg_bool[24] = constant_mask_bit(b, uint32_t(bits), uint32_t(bits >> 32u));
    }
    bool run(const std::vector<uint32_t>& code) {
        std::vector<Rdna2Inst> ins;
        rdna2_walk(code.data(), code.size(), ins);
        return emit_body(
            b, rs, ins, {}, nullptr, true, false, [](RegState&, const Rdna2Inst&) { return true; },
            code.data(), code.size());
    }
    void refused(const std::vector<uint32_t>& code, uint32_t branch_pc = 1) {
        EXPECT_FALSE(run(code));
        EXPECT_EQ(b.stage_reject_pc, branch_pc);
        EXPECT_EQ(b.stage_reject_reason, "structured-saved-mask-domain-missing-edge");
    }
};

// Execute the actual emitted acyclic CFG, including predecessor-selected phis and the
// real storage load. This narrow CPU oracle rejects unknown operations and undefined reads.
// It does not execute fragment entry/pixel mapping or claim device equivalence.
std::optional<uint32_t> value(const Emitter& e, uint32_t storage_word, uint32_t lane) {
    struct Inst {
        uint32_t op;
        std::vector<uint32_t> a;
    };
    std::vector<Inst> code;
    std::map<uint32_t, uint32_t> values, types, labels;
    const auto decode = [](const std::vector<uint32_t>& words, std::vector<Inst>& out) {
        for (size_t at = 0; at < words.size();) {
            const size_t width = words[at] >> 16u;
            if (!width || width > words.size() - at) return false;
            out.push_back(
                {words[at] & 0xffffu, {words.begin() + at + 1, words.begin() + at + width}});
            at += width;
        }
        return true;
    };
    std::vector<Inst> declarations;
    if (!decode(e.b.types, declarations) || !decode(e.b.code, code)) return {};
    for (const auto& in : declarations) {
        if (in.op != Op_Constant && in.op != Op_ConstantTrue && in.op != Op_ConstantFalse) continue;
        if (in.a.size() < 2) return {};
        types[in.a[1]] = in.a[0];
        if (in.op == Op_Constant) {
            if (in.a.size() != 3) return {};
            values[in.a[1]] = in.a[2];
        } else
            values[in.a[1]] = in.op == Op_ConstantTrue;
    }
    for (size_t i = 0; i < code.size(); ++i)
        if (code[i].op == Op_Label && code[i].a.size() == 1)
            labels.emplace(code[i].a[0], static_cast<uint32_t>(i));
    std::set<uint32_t> storage_pointers;
    uint32_t current = 0, predecessor = 0;
    size_t pc = 0, steps = 0;
    try {
        while (pc < code.size() && ++steps < 512) {
            const auto& in = code.at(pc++);
            const auto& a = in.a;
            if (in.op == Op_Function || in.op == Op_SelectionMerge) continue;
            if (in.op == Op_Label) {
                current = a.at(0);
                continue;
            }
            if (in.op == Op_AccessChain) {
                if (a.size() != 5 || a[2] != e.b.v_cbuf || values.at(a[3]) != 0 ||
                    values.at(a[4]) != 0)
                    return {};
                storage_pointers.insert(a[1]);
                continue;
            }
            if (in.op == Op_Branch || in.op == Op_BranchConditional) {
                const auto target = in.op == Op_Branch ? a.at(0) : a.at(values.at(a.at(0)) ? 1 : 2);
                if (in.op == Op_BranchConditional && types.at(a[0]) != e.b.t_bool) return {};
                predecessor = current;
                pc = labels.at(target);
                continue;
            }
            if (a.size() < 3) return {};
            types[a[1]] = a[0];
            if (in.op == Op_Load) {
                if (a[1] == e.storage && storage_pointers.contains(a[2]))
                    values[a[1]] = storage_word;
                else if (a[2] == e.b.v_subgroup_localid)
                    values[a[1]] = lane;
                else
                    return {};
            } else if (in.op == Op_Phi) {
                bool found = false;
                for (size_t i = 2; i + 1 < a.size(); i += 2)
                    if (a[i + 1] == predecessor) {
                        if (found || types.at(a[i]) != a[0]) return {};
                        values[a[1]] = values.at(a[i]);
                        found = true;
                    }
                if (!found) return {};
            } else if (in.op == Op_Bitcast)
                values[a[1]] = values.at(a[2]);
            else if (in.op == Op_Select) {
                if (a.size() != 5 || types.at(a[2]) != e.b.t_bool) return {};
                values[a[1]] = values.at(a[values.at(a[2]) ? 3 : 4]);
            } else if (a.size() == 3 && in.op == Op_LogicalNot)
                values[a[1]] = !values.at(a[2]);
            else if (a.size() == 4) {
                const uint32_t x = values.at(a[2]), y = values.at(a[3]);
                switch (in.op) {
                    case Op_BitwiseAnd: values[a[1]] = x & y; break;
                    case Op_ShiftRightLogical:
                        if (y >= 32) return {};
                        values[a[1]] = x >> y;
                        break;
                    case Op_IEqual: values[a[1]] = x == y; break;
                    case Op_INotEqual: values[a[1]] = x != y; break;
                    case Op_UGreaterThanEqual: values[a[1]] = x >= y; break;
                    case Op_LogicalAnd: values[a[1]] = x && y; break;
                    case Op_LogicalOr: values[a[1]] = x || y; break;
                    case 165u: values[a[1]] = x != y; break; // OpLogicalNotEqual
                    case Op_FOrdLessThan:
                        values[a[1]] = std::bit_cast<float>(x) < std::bit_cast<float>(y);
                        break;
                    default: return {};
                }
            } else
                return {};
        }
        if (steps >= 512) return {};
        return values.at(e.rs.vreg.at(0));
    } catch (const std::out_of_range&) { return {}; }
}
void expect_values(const Emitter& e, uint64_t taken, uint64_t skipped) {
    for (uint32_t storage : {0u, 1u}) {
        const uint64_t expected = storage ? skipped : taken;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto actual = value(e, storage, lane);
            if (!actual.has_value()) {
                ADD_FAILURE() << "storage=" << storage << " lane=" << lane;
                return;
            }
            EXPECT_EQ(*actual, expected & (uint64_t(1) << lane) ? 9u : 1u)
                << "storage=" << storage << " lane=" << lane;
        }
    }
}
void retain_module(Emitter& e, const char* name) {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): inherited before tests; no test changes this.
    const char* directory = std::getenv("PROSPER_STRUCTURED_MASK_JOIN_SPV_DIRECTORY");
    if (!directory) return;
    const auto module = e.b.finish();
    ASSERT_FALSE(module.empty());
    std::ofstream out(std::filesystem::path(directory) / name, std::ios::binary);
    ASSERT_TRUE(out.good());
    out.write(reinterpret_cast<const char*>(module.data()),
              static_cast<std::streamsize>(module.size() * sizeof(uint32_t)));
    ASSERT_TRUE(out.good());
}
} // namespace

TEST(StructuredSavedMaskJoin, GenuinePhysicalIncomingPairCannotBecomeFalse) {
    Emitter e;
    e.physical();
    e.refused({0xbf068014u, 0xbf840002u, 0x7c0200f9u, 0x06869880u, 0xd5010000u, 0x00611281u, end});
}
TEST(StructuredSavedMaskJoin, MissingIncomingMaskRefuses) {
    Emitter e;
    e.refused(one_arm());
}
TEST(StructuredSavedMaskJoin, IncompleteIncomingPhysicalPairRefuses) {
    Emitter e;
    e.physical();
    e.rs.sreg.erase(25);
    e.refused(one_arm());
}
TEST(StructuredSavedMaskJoin, FabricatedPhysicalPairRefuses) {
    Emitter e;
    e.physical();
    e.rs.sreg_merge_placeholder.insert(25);
    e.refused(one_arm());
}
TEST(StructuredSavedMaskJoin, MemoryPatternPairRefuses) {
    Emitter e;
    e.physical();
    e.rs.sreg_memory_pattern.insert(24);
    e.refused(one_arm());
}
TEST(StructuredSavedMaskJoin, GenuineIncomingPredicatePreservesBothDynamicPaths) {
    Emitter e;
    e.predicate();
    ASSERT_TRUE(e.run(one_arm()));
    expect_values(e, left ^ right, incoming);
    retain_module(e, "incoming-predicate.spv");
}
TEST(StructuredSavedMaskJoin, GenuinePredicateRelayPreservesBothDynamicPaths) {
    Emitter e;
    e.predicate();
    ASSERT_TRUE(e.run(one_arm(true)));
    expect_values(e, left ^ right, incoming);
    retain_module(e, "predicate-relay.spv");
}
TEST(StructuredSavedMaskJoin, GenuineFalsePredicateIsStillDefined) {
    Emitter e;
    e.predicate(0);
    ASSERT_TRUE(e.run(one_arm()));
    expect_values(e, left ^ right, 0);
    retain_module(e, "genuine-zero-predicate.spv");
}
TEST(StructuredSavedMaskJoin, BothArmsProduceMasksWithoutAnIncomingPredicate) {
    Emitter e;
    std::vector<uint32_t> code{0xbf068014u, 0xbf840002u, logical(0x13), 0xbf820001u, logical(0x11)};
    select(code);
    code.push_back(end);
    ASSERT_TRUE(e.run(code));
    expect_values(e, left ^ right, left | right);
    retain_module(e, "two-defined-arms.spv");
}
TEST(StructuredSavedMaskJoin, OneArmReplacingPredicateWithDataRefuses) {
    Emitter e;
    e.predicate();
    e.rs.sreg[26] = e.b.uconst(0xabcdef01u);
    e.rs.sreg[27] = e.b.uconst(0x80000001u);
    // Inline zero has a genuine Bool spelling; use a complete DATA-only source instead.
    std::vector<uint32_t> code{0xbf068014u, 0xbf840001u, mov(24, 26, true)};
    select(code);
    code.push_back(end);
    e.refused(code);
}
TEST(StructuredSavedMaskJoin, IfElseWithOneDataArmRefuses) {
    Emitter e;
    e.physical();
    e.rs.sreg[26] = e.b.uconst(0xabcdef01u);
    e.rs.sreg[27] = e.b.uconst(0x80000001u);
    std::vector<uint32_t> code{0xbf068014u, 0xbf840002u, logical(0x13), 0xbf820001u,
                               mov(24, 26, true)};
    select(code);
    code.push_back(end);
    e.refused(code);
}
TEST(StructuredSavedMaskJoin, CompletelyDeadOneArmMaskNeedsNoPhi) {
    Emitter e;
    ASSERT_TRUE(e.run({0xbf068014u, 0xbf840001u, logical(0x13), end}));
    EXPECT_FALSE(e.rs.sreg_bool.contains(24));
}
TEST(StructuredSavedMaskJoin, InterpolationDoesNotKeepDeadOneArmMaskLive) {
    Emitter e;
    ASSERT_TRUE(e.run({0xbf068014u, 0xbf840001u, logical(0x13), 0xc8100000u, end}));
    EXPECT_FALSE(e.rs.sreg_bool.contains(24));
    for (uint32_t storage : {0u, 1u})
        for (uint32_t lane = 0; lane < 64; ++lane)
            EXPECT_EQ(value(e, storage, lane), std::bit_cast<uint32_t>(0.5f));
    retain_module(e, "interpolation-dead-mask.spv");
}
TEST(StructuredSavedMaskJoin, InterpolationPreservesGenuinePredicatesOnBothPaths) {
    Emitter e;
    e.predicate();
    auto code = one_arm();
    code.insert(code.begin() + 3, 0xc8100000u);
    ASSERT_TRUE(e.run(code));
    expect_values(e, left ^ right, incoming);
    retain_module(e, "interpolation-genuine-predicate.spv");
}
TEST(StructuredSavedMaskJoin, InterpolationCannotSupplyMissingLiveMaskOrRelay) {
    for (bool relay : {false, true}) {
        Emitter e;
        auto code = one_arm(relay);
        code.insert(code.begin() + 3, 0xc8100000u);
        e.refused(code);
    }
}
TEST(StructuredSavedMaskJoin, FullReplacementAfterJoinNeedsNoOldMask) {
    Emitter e;
    std::vector<uint32_t> code{0xbf068014u, 0xbf840001u, logical(0x13), logical(0x11)};
    select(code);
    code.push_back(end);
    ASSERT_TRUE(e.run(code));
    expect_values(e, left | right, left | right);
    retain_module(e, "complete-replacement.spv");
}
TEST(StructuredSavedMaskJoin, DeadB32MaskPreservesIndependentNeighbor) {
    for (uint32_t root : {24u, 106u}) {
        Emitter e;
        e.b.wave_size = 32;
        e.b.allow_b32_masks = true;
        e.rs.sreg[root + 1] = e.b.uconst(0x12345678u);
        const std::vector<uint32_t> code{
            0xbf068014u, 0xbf840001u, mov(root, 126), mov(root, 133), 0x7e000200u | (root + 1),
            end};
        ASSERT_TRUE(e.run(code)) << "root=" << root;
        EXPECT_FALSE(e.rs.sreg_bool.contains(root));
        for (uint32_t storage : {0u, 1u})
            for (uint32_t lane = 0; lane < 32; ++lane)
                EXPECT_EQ(value(e, storage, lane), 0x12345678u)
                    << "root=" << root << " storage=" << storage << " lane=" << lane;
        retain_module(e, root == 24 ? "b32-independent-neighbor.spv" : "b32-vcc-neighbor.spv");
    }
}
TEST(StructuredSavedMaskJoin, B64MaskCannotRetireAfterOnlyLowReplacement) {
    Emitter e;
    e.rs.sreg[25] = e.b.uconst(0x12345678u);
    e.refused({0xbf068014u, 0xbf840001u, mov(24, 126, true), mov(24, 133), 0x7e000219u, end});
}
TEST(StructuredSavedMaskJoin, CompleteB64ReplacementRetiresOldMask) {
    Emitter e;
    e.rs.sreg[25] = e.b.uconst(0x12345678u);
    ASSERT_TRUE(e.run({0xbf068014u, 0xbf840001u, mov(24, 126, true), mov(24, 133), mov(25, 135),
                       0x7e000219u, end}));
    EXPECT_FALSE(e.rs.sreg_bool.contains(24));
}
TEST(StructuredSavedMaskJoin, WidthConflictCannotRetireLiveB64High) {
    Emitter e;
    e.b.wave_size = 32;
    e.b.allow_b32_masks = true;
    e.physical();
    e.predicate();
    e.refused({0xbf068014u, 0xbf840001u, mov(24, 126), mov(24, 133), 0x7e000219u, end});
}
TEST(StructuredSavedMaskJoin, Wave32CarryDoesNotReplaceMissingB64High) {
    Emitter e;
    e.b.wave_size = 32;
    e.b.allow_b32_masks = true;
    e.rs.sreg[25] = e.b.uconst(0x12345678u);
    // ADD_CO writes one mask word in this emitter mode; s25 remains independently live.
    e.refused({0xbf068014u, 0xbf840001u, logical(0x13), 0xd70f1801u, 129u | (129u << 9u),
               0x7e000219u, end});
}
TEST(StructuredSavedMaskJoin, CarryDeathProofUsesActualOutputWidthAndChecksInputs) {
    const std::vector<uint32_t> code{0xd70f1801u, 129u | (129u << 9u), 0x7e000219u, end};
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    ASSERT_EQ(ins.front().sdst.value, 24);
    EXPECT_TRUE(sgpr_dead_at_merge(ins, 0, 24, ScalarMergeProof::AnyRead, nullptr, false,
                                   {false, true, true}));
    EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 25, ScalarMergeProof::AnyRead, nullptr, false,
                                    {false, true, true}));
    EXPECT_TRUE(sgpr_dead_at_merge(ins, 0, 25));
    const std::vector<uint32_t> carry_in{0xd5281801u, 129u | (129u << 9u) | (24u << 18u), end};
    ins.clear();
    rdna2_walk(carry_in.data(), carry_in.size(), ins);
    EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 24));
    EXPECT_FALSE(sgpr_dead_at_merge(ins, 0, 25));
}
TEST(StructuredSavedMaskJoin, Wave32ImplicitCompareCannotRetireLiveVccHigh) {
    Emitter e;
    e.b.wave_size = 32;
    e.b.allow_b32_masks = true;
    e.rs.sreg[107] = e.b.uconst(0x12345678u);
    e.refused({0xbf068014u, 0xbf840001u, mov(107, 126), 0x7c0200f0u, 0x7e00026bu, end});
}
TEST(StructuredSavedMaskJoin, CompleteHighReplacementAfterWave32CompareCompiles) {
    Emitter e;
    e.b.wave_size = 32;
    e.b.allow_b32_masks = true;
    e.rs.sreg[107] = e.b.uconst(0x12345678u);
    ASSERT_TRUE(e.run(
        {0xbf068014u, 0xbf840001u, mov(107, 126), 0x7c0200f0u, mov(107, 133), 0x7e00026bu, end}));
    for (uint32_t storage : {0u, 1u})
        for (uint32_t lane = 0; lane < 32; ++lane) EXPECT_EQ(value(e, storage, lane), 5u);
}
TEST(StructuredSavedMaskJoin, Wave32ExplicitVccComparePreservesOtherWord) {
    for (uint32_t destination : {106u, 107u}) {
        Emitter e;
        e.b.wave_size = 32;
        e.b.allow_b32_masks = true;
        const uint32_t neighbor = destination == 106 ? 107 : 106;
        e.rs.sreg[neighbor] = e.b.uconst(0x12345678u);
        e.refused({0xbf068014u, 0xbf840001u, mov(neighbor, 126), 0xd4010000u | destination,
                   0x000200f0u, 0x7e000200u | neighbor, end});
    }
}
TEST(StructuredSavedMaskJoin, CompleteWave64CompareReplacementCompiles) {
    Emitter e;
    std::vector<uint32_t> code{0xbf068014u, 0xbf840001u, logical(0x13), 0xd4010018u, 0x000200f0u};
    select(code);
    code.push_back(end);
    ASSERT_TRUE(e.run(code));
    expect_values(e, 0, 0);
    retain_module(e, "complete-wave64-compare-replacement.spv");
}
TEST(StructuredSavedMaskJoin, Wave32CompareCannotRetireLiveB64High) {
    Emitter e;
    e.b.wave_size = 32;
    e.b.allow_b32_masks = true;
    e.rs.sreg[25] = e.b.uconst(0x12345678u);
    e.refused(
        {0xbf068014u, 0xbf840001u, logical(0x13), 0xd4010018u, 0x000200f0u, 0x7e000219u, end});
}
TEST(StructuredSavedMaskJoin, ArmLocalExecSaveCannotManufactureSkippedEdgeValue) {
    Emitter e;
    // The old spv_validate fixture restored s[2:3] even when EXECZ skipped its only writer.
    e.refused({0x7e020280u, 0x7e0002f2u, 0x7c2200f0u, 0xbf880001u, 0xbe82047eu, 0xbefe0402u, end},
              3);
}
TEST(StructuredSavedMaskJoin, ExactOldExecSaveExportStreamRefuses) {
    // Keep the complete former validator fixture, including its real export lowering.
    const uint32_t code[] = {0x7e020280u, 0x7e0002f2u, 0x7c2200f0u, 0xbf880001u, 0xbe82047eu,
                             0xbefe0402u, 0x7e0202f2u, 0xf800180fu, 0x01010101u, end};
    constexpr uint64_t address = 0xa4957002ull;
    EXPECT_TRUE(recompile_fragment(code, sizeof(code) / sizeof(*code), nullptr, nullptr, UINT32_MAX,
                                   nullptr, false, {RecompileDiagnosticStage::Fragment, address})
                    .empty());
    EXPECT_NE(
        last_terminal_reject_reason(address).find("structured-saved-mask-domain-missing-edge"),
        std::string::npos);
}
TEST(StructuredSavedMaskJoin, DefinedZeroBeforeExecSaveJoinCompiles) {
    Emitter e;
    ASSERT_TRUE(e.run({0xbe820480u, 0x7e020280u, 0x7e0002f2u, 0x7c2200f0u, 0xbf880001u, 0xbe82047eu,
                       0xbefe0402u, end}));
    retain_module(e, "defined-exec-save.spv");
}
TEST(StructuredSavedMaskJoin, ValueOracleRejectsUnknownOperation) {
    Emitter e;
    e.predicate();
    ASSERT_TRUE(e.run(one_arm()));
    EXPECT_EQ(value(e, 1, 0), 9u);
    e.b.code.push_back((1u << 16u) | 65000u);
    EXPECT_FALSE(value(e, 1, 0).has_value());
}
TEST(StructuredSavedMaskJoin, ValueOracleRejectsUndefinedResult) {
    Emitter e;
    e.predicate();
    ASSERT_TRUE(e.run(one_arm()));
    EXPECT_EQ(value(e, 1, 0), 9u);
    e.rs.vreg[0] = e.b.id();
    EXPECT_FALSE(value(e, 1, 0).has_value());
}
} // namespace prosper::gpu
