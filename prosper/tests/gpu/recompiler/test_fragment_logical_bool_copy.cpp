// #4924: fragment B64 logical results are genuine Bool sources for a later
// predicate-only saved copy. Missing original MUST transfer falsely demands
// physical words; numeric generations and incomplete source domains still refuse.
#include "fragment_predicate_oracle.hpp"
#include "fixtures/fragment_compare_kill.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <gtest/gtest.h>

namespace prosper::gpu {
namespace {
constexpr uint32_t end = 0xbf810000u;
constexpr uint64_t left_mask = 0x80000001abcdef01ull, right_mask = 0x7fedcba810203040ull;
constexpr uint32_t mov(uint32_t dst, uint32_t source, bool pair = false) {
    return 0xbe800000u | (dst << 16) | ((pair ? 4u : 3u) << 8) | source;
}
constexpr uint32_t logical(uint32_t op, uint32_t dst = 24, uint32_t a = 20, uint32_t c = 22) {
    return 0x80000000u | (op << 23) | (dst << 16) | (c << 8) | a;
}
std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& code) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(code.data(), code.size(), ins);
    return ins;
}
std::vector<uint32_t> program(uint32_t op = 0x17, const std::vector<uint32_t>& tail = {},
                              uint32_t dst = 24, uint32_t a = 20, uint32_t c = 22) {
    std::vector<uint32_t> code{mov(30, 106), logical(op, dst, a, c), mov(106, dst, true),
                               0xd5010000u, 129u | (137u << 9) | (106u << 18)};
    code.insert(code.end(), tail.begin(), tail.end());
    code.push_back(end);
    return code;
}
size_t ballots(const SpirvCompute& b) {
    size_t count = 0;
    for (size_t at = 0; at < b.code.size();) {
        const size_t width = b.code[at] >> 16;
        if (!width || width > b.code.size() - at) return SIZE_MAX;
        count += (b.code[at] & 0xffffu) == 339u;
        at += width;
    }
    return count;
}
uint64_t expected_mask(uint32_t op) {
    switch (op) {
        case 0x0f: return left_mask & right_mask;
        case 0x11: return left_mask | right_mask;
        case 0x13: return left_mask ^ right_mask;
        case 0x15: return left_mask & ~right_mask;
        case 0x17: return left_mask | ~right_mask;
        case 0x19: return ~(left_mask & right_mask);
        case 0x1b: return ~(left_mask | right_mask);
        case 0x1d: return ~(left_mask ^ right_mask);
        default: return 0;
    }
}
struct Emitter {
    SpirvCompute b;
    RegState rs;
    Emitter() {
        b.wave_size = 64;
        b.begin_fragment();
        b.diagnostic = {RecompileDiagnosticStage::Fragment, 0xa4924001ull};
        rs.exec = b.btrue();
        rs.vcc = b.bfalse();
        rs.native_saved_mask_words = true;
        rs.absent_sgpr_reads_fabricated_zero = true;
        rs.sreg[106] = b.uconst(3);
        rs.sreg[107] = b.uconst(5);
        rs.sreg_bool[20] = constant_mask_bit(b, uint32_t(left_mask), uint32_t(left_mask >> 32));
        rs.sreg_bool[22] = constant_mask_bit(b, uint32_t(right_mask), uint32_t(right_mask >> 32));
        rs.vreg[0] = b.uconst(0);
    }
    void prove(const std::vector<uint32_t>& code) {
        const auto ins = decode(code);
        initialize_saved_mask_word_proof(b, ins, code.data(), code.size(), b.btrue(), &rs);
        EXPECT_TRUE(b.saved_mask_word_roots.contains(106)) << "exercise the physical-copy guard";
    }
    bool emit(const Rdna2Inst& in) {
        prepare_saved_mask_word_write(b, rs, in);
        const auto before = snapshot_saved_b64_masks(rs, in);
        bool ok = true;
        EXPECT_TRUE(emit_alu(b, rs, in, ok, true));
        if (ok) {
            record_scalar_write(rs, in, allows_compute_scalar_vcc_bridge(b), before);
            retain_saved_mask_words(b, rs, in);
        }
        return ok;
    }
    void expect_copy_refusal(uint32_t pc) const {
        EXPECT_EQ(b.stage_reject_pc, pc);
        EXPECT_EQ(b.stage_reject_reason, "saved-mask-copy-words-unavailable");
        EXPECT_EQ(ballots(b), 0u);
    }
};
void expect_operation(uint32_t op) {
    Emitter e;
    const auto code = program(op);
    const auto ins = decode(code);
    e.prove(code);
    EXPECT_TRUE(e.b.fragment_bool_only_vcc_copy_pcs.contains(2));
    ASSERT_TRUE(e.emit(ins[0]));
    ASSERT_TRUE(e.emit(ins[1]));
    ASSERT_TRUE(e.rs.sreg_bool.contains(24));
    ASSERT_TRUE(e.emit(ins[2]));
    EXPECT_EQ(e.rs.vcc, e.rs.sreg_bool.at(24));
    EXPECT_FALSE(e.rs.sreg.contains(106));
    EXPECT_FALSE(e.rs.sreg.contains(107));
    ASSERT_TRUE(e.emit(ins[3]));
    fragment_predicate_oracle::Values vm(e.b);
    const uint64_t expected = expected_mask(op);
    for (uint32_t lane = 0; lane < 64; ++lane) {
        const auto value = vm.evaluate(e.rs.vreg.at(0), lane);
        ASSERT_TRUE(value.has_value()) << vm.error;
        EXPECT_EQ(value.value_or(UINT64_MAX), (expected >> lane) & 1u ? 9u : 1u) << "lane " << lane;
    }
    EXPECT_EQ(ballots(e.b), 0u);
}
std::vector<uint32_t> production_program(uint32_t op = 0x17,
                                         const std::vector<uint32_t>& tail = {}) {
    std::vector<uint32_t> code{mov(106, 129, true),
                               mov(30, 106),
                               0x7e000281u,
                               0x7e020280u,
                               0x7d840300u,
                               0xbf068080u,
                               logical(0x0b, 106, 126, 128),
                               logical(op, 24, 106, 126),
                               mov(106, 24, true),
                               0xd5010000u,
                               129u | (137u << 9) | (106u << 18)};
    code.insert(code.end(), tail.begin(), tail.end());
    code.insert(code.end(), {0xf800080fu, 0x00000000u, end});
    return code;
}
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address) {
    return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, nullptr,
                              false, {RecompileDiagnosticStage::Fragment, address});
}
} // namespace

TEST(FragmentLogicalBoolCopy, AndValues) {
    expect_operation(0x0f);
}
TEST(FragmentLogicalBoolCopy, OrValues) {
    expect_operation(0x11);
}
TEST(FragmentLogicalBoolCopy, XorValues) {
    expect_operation(0x13);
}
TEST(FragmentLogicalBoolCopy, Andn2Values) {
    expect_operation(0x15);
}
TEST(FragmentLogicalBoolCopy, Orn2Values) {
    expect_operation(0x17);
}
TEST(FragmentLogicalBoolCopy, NandValues) {
    expect_operation(0x19);
}
TEST(FragmentLogicalBoolCopy, NorValues) {
    expect_operation(0x1b);
}
TEST(FragmentLogicalBoolCopy, XnorValues) {
    expect_operation(0x1d);
}

TEST(FragmentLogicalBoolCopy, FullFragmentCselectLogicalCopyCompiles) {
    for (uint32_t op : {0x0fu, 0x11u, 0x13u, 0x15u, 0x17u, 0x19u, 0x1bu, 0x1du})
        EXPECT_FALSE(compile(production_program(op), 0xa4924100ull + op).empty()) << "op " << op;
}

TEST(FragmentLogicalBoolCopy, CurrentLowAndHighDataDemandStillRefuse) {
    for (uint32_t source : {106u, 107u}) {
        Emitter e;
        const auto code = program(0x17, {mov(31, source)});
        const auto ins = decode(code);
        e.prove(code);
        EXPECT_FALSE(e.b.fragment_bool_only_vcc_copy_pcs.contains(2));
        ASSERT_TRUE(e.emit(ins[1]));
        EXPECT_FALSE(e.emit(ins[2]));
        e.expect_copy_refusal(2);
    }
}

TEST(FragmentLogicalBoolCopy, LaterCompleteReplacementOwnsTheNumericRead) {
    for (uint32_t source : {106u, 107u}) {
        Emitter e;
        const auto code = program(0x17, {mov(106, 129, true), mov(31, source)});
        const auto ins = decode(code);
        e.prove(code);
        ASSERT_TRUE(e.emit(ins[1]));
        EXPECT_TRUE(e.emit(ins[2]));
        EXPECT_EQ(ballots(e.b), 0u);
    }
}

TEST(FragmentLogicalBoolCopy, OutputAliasesRemainPhysicalDataReads) {
    for (uint32_t source : {32u, 33u}) {
        Emitter e;
        const auto code = program(0x17, {mov(32, 106, true), mov(31, source)});
        const auto ins = decode(code);
        e.prove(code);
        EXPECT_FALSE(e.b.fragment_bool_only_vcc_copy_pcs.contains(2));
        ASSERT_TRUE(e.emit(ins[1]));
        EXPECT_FALSE(e.emit(ins[2]));
        e.expect_copy_refusal(2);
    }
}

TEST(FragmentLogicalBoolCopy, PartialAndConditionalVccWritesAreNotCompleteKills) {
    for (const auto& tail : {std::vector<uint32_t>{mov(106, 129), mov(31, 107)},
                             std::vector<uint32_t>{0xbeea0681u, mov(31, 107)}}) {
        Emitter e;
        const auto code = program(0x17, tail);
        const auto ins = decode(code);
        e.prove(code);
        ASSERT_TRUE(e.emit(ins[1]));
        EXPECT_FALSE(e.emit(ins[2]));
        e.expect_copy_refusal(2);
    }
}

TEST(FragmentLogicalBoolCopy, OneArmLogicalWriterCannotSupplyAMustSource) {
    Emitter e;
    const std::vector<uint32_t> code{mov(30, 106),
                                     0xbf840001u,
                                     logical(0x17),
                                     mov(106, 24, true),
                                     0xd5010000u,
                                     129u | (137u << 9) | (106u << 18),
                                     end};
    const auto ins = decode(code);
    e.prove(code);
    EXPECT_FALSE(e.b.fragment_bool_only_vcc_copy_pcs.contains(3));
    ASSERT_TRUE(e.emit(ins[2]));
    EXPECT_FALSE(e.emit(ins[3]));
    e.expect_copy_refusal(3);
}

TEST(FragmentLogicalBoolCopy, ALoopBackedgeDemandsTheCurrentGenerationUntilAFullKill) {
    for (bool replace : {false, true}) {
        Emitter e;
        std::vector<uint32_t> tail;
        if (replace) tail.push_back(mov(106, 129, true));
        tail.push_back(replace ? 0xbf82fff9u : 0xbf82fffau);
        const auto code = program(0x17, tail);
        const auto ins = decode(code);
        e.prove(code);
        EXPECT_EQ(e.b.fragment_bool_only_vcc_copy_pcs.contains(2), replace);
        ASSERT_TRUE(e.emit(ins[1]));
        EXPECT_EQ(e.emit(ins[2]), replace);
        if (!replace) e.expect_copy_refusal(2);
    }
}

TEST(FragmentLogicalBoolCopy, IndependentB32AndHighWordInputsCannotBecomeB64Sources) {
    for (uint32_t source : {20u, 107u, 127u}) {
        Emitter e;
        if (source == 20) e.rs.sreg_bool_b32.insert(20);
        const auto code = program(0x17, {}, 24, source, 22);
        const auto ins = decode(code);
        e.prove(code);
        EXPECT_FALSE(e.b.fragment_bool_only_vcc_copy_pcs.contains(2));
        ASSERT_TRUE(e.emit(ins[1]));
        EXPECT_FALSE(e.emit(ins[2]));
        e.expect_copy_refusal(2);
    }
}

TEST(FragmentLogicalBoolCopy, CurrentB32LogicalResultStillRefuses) {
    Emitter e;
    const auto code = program();
    const auto ins = decode(code);
    e.prove(code);
    EXPECT_TRUE(e.b.fragment_bool_only_vcc_copy_pcs.contains(2));
    ASSERT_TRUE(e.emit(ins[1]));
    e.rs.sreg_bool_b32.insert(24);
    EXPECT_FALSE(e.emit(ins[2]));
    e.expect_copy_refusal(2);
}

namespace {
void expect_full_source_relay(bool select) {
    for (uint32_t source : {106u, 107u, 126u, 127u}) {
        Emitter e;
        e.rs.scc = e.b.btrue();
        const uint32_t relay = select ? logical(0x0b, 20, source, 128) : mov(20, source, true);
        const std::vector<uint32_t> code{mov(30, 106),
                                         relay,
                                         logical(0x17),
                                         mov(106, 24, true),
                                         0xd5010000u,
                                         129u | (137u << 9) | (106u << 18),
                                         end};
        const auto ins = decode(code);
        e.prove(code);
        const bool complete = source == 106 || source == 126;
        EXPECT_EQ(e.b.fragment_bool_only_vcc_copy_pcs.contains(3), complete);
        ASSERT_TRUE(e.emit(ins[1]));
        ASSERT_TRUE(e.emit(ins[2]));
        EXPECT_EQ(e.emit(ins[3]), complete);
        if (!complete) {
            e.expect_copy_refusal(3);
            continue;
        }
        ASSERT_TRUE(e.emit(ins[4]));
        fragment_predicate_oracle::Values vm(e.b);
        const uint64_t expected = (source == 126 ? UINT64_MAX : 0u) | ~right_mask;
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto value = vm.evaluate(e.rs.vreg.at(0), lane);
            ASSERT_TRUE(value.has_value()) << vm.error;
            EXPECT_EQ(value.value_or(UINT64_MAX), (expected >> lane) & 1u ? 9u : 1u)
                << "lane " << lane;
        }
        EXPECT_EQ(ballots(e.b), 0u);
    }
}
} // namespace

TEST(FragmentLogicalBoolCopy, MovRelayCannotPromoteAnIndependentHighWord) {
    expect_full_source_relay(false);
}

TEST(FragmentLogicalBoolCopy, CselectRelayCannotPromoteAnIndependentHighWord) {
    expect_full_source_relay(true);
}

TEST(FragmentLogicalBoolCopy, B32EmitterModeCannotPublishAFullFragmentCopy) {
    Emitter e;
    e.b.allow_b32_masks = true;
    e.rs.vreg[1] = e.b.uconst(0);
    const std::vector<uint32_t> code{mov(30, 106),
                                     prosper::test::fragment_compare_kill::compare(),
                                     prosper::test::fragment_compare_kill::compare_sd(20),
                                     logical(0x17),
                                     mov(106, 24, true),
                                     0xd5010000u,
                                     129u | (137u << 9) | (106u << 18),
                                     end};
    const auto ins = decode(code);
    e.prove(code);
    EXPECT_FALSE(e.b.fragment_bool_only_vcc_copy_pcs.contains(4));
    ASSERT_TRUE(e.emit(ins[1]));
    ASSERT_TRUE(e.rs.sreg_bool_b32.contains(20));
    ASSERT_TRUE(e.emit(ins[2]));
    EXPECT_FALSE(e.emit(ins[3]));
    e.expect_copy_refusal(4);
}

TEST(FragmentLogicalBoolCopy, ChangedB32ModeRevokesTheOriginalProof) {
    Emitter e;
    const auto code = program();
    const auto ins = decode(code);
    const auto entry = e.rs;
    e.prove(code);
    ASSERT_TRUE(e.emit(ins[1]));
    e.b.allow_b32_masks = true;
    EXPECT_FALSE(native_vcc_copy_execution_domain_matches(e.b));
    initialize_saved_mask_word_proof(e.b, ins, code.data(), code.size(), 0, &entry);
    EXPECT_FALSE(e.b.fragment_bool_only_vcc_copy_pcs.contains(2));
    EXPECT_FALSE(e.emit(ins[2]));
    e.expect_copy_refusal(2);
}

TEST(FragmentLogicalBoolCopy, ChangedRawAndDomainInvalidateTheOriginalLogicalProof) {
    for (bool change_bytes : {false, true}) {
        Emitter e;
        auto code = program();
        const auto ins = decode(code);
        const auto entry = e.rs;
        e.prove(code);
        ASSERT_TRUE(e.emit(ins[1]));
        if (change_bytes)
            code[1] = logical(0x11);
        else
            e.b.is_vertex = true;
        const auto changed = decode(code);
        initialize_saved_mask_word_proof(e.b, changed, code.data(), code.size(), 0, &entry);
        EXPECT_FALSE(e.emit(ins[2]));
        e.expect_copy_refusal(2);
    }
}
} // namespace prosper::gpu
