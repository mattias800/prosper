// test_ir_cfg_exec_values -- IR_CONTROL_FLOW_PLAN step 7a. The static EXEC values and relations
// (exec_values.hpp) on synthetic streams, one per rule, and the shadow census
// (exec_value_census.hpp) over every committed fixture: no exec_within answer or complement read
// may be one the SPIR-V-id relations prove and the static relations do not.
#include "gpu/recompiler/ir/cfg/exec_values.hpp"

#include "gpu/recompiler/ir/cfg/exec_value_census.hpp"

#include "gpu/recompiler/ir/cfg_fixtures.hpp"
#include "gpu/recompiler/ir/cfg_streams.hpp"
#include "gpu/recompiler/ir/vector_lane_harness.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace prosper::gpu;
using namespace prosper::gpu::ir::cfg;
using prosper::test::cfg_streams::kEndpgm;
using prosper::test::cfg_streams::kExecnz;
using prosper::test::cfg_streams::kExecz;
using prosper::test::cfg_streams::kNop;
using prosper::test::cfg_streams::sopp;
using prosper::test::cfg_streams::walk;

namespace {

constexpr uint32_t kCmpVcc = 0x7C080080u;   // v_cmp_gt_f32 vcc, 0, v0
constexpr uint32_t kCmpxExec = 0x7C280080u;   // v_cmpx_gt_f32 exec, 0, v0
constexpr uint32_t kAndSaveexec = 0xBE80246Au;   // s_and_saveexec_b64 s[0:1], vcc
constexpr uint32_t kAndn2Exec = 0x8AFE7E00u;   // s_andn2_b64 exec, s[0:1], exec
constexpr uint32_t kOrExec = 0x88FE007Eu;   // s_or_b64 exec, exec, s[0:1]
constexpr uint32_t kMovExecS0 = 0xBEFE0400u;   // s_mov_b64 exec, s[0:1]
constexpr uint32_t kMovExecAllOnes = 0xBEFE04C1u;   // s_mov_b64 exec, -1

struct Built {
    std::vector<Rdna2Inst> ins;
    GuestCfg cfg;
    ExecValues values;
};

Built build(const std::vector<uint32_t>& words, ExecValueOptions options = {}) {
    Built out;
    out.ins = walk(words);
    out.cfg = build_guest_cfg(out.ins, SplitPolicy{});
    out.values = build_exec_values(out.cfg, out.ins, options);
    return out;
}

ExecValue::Kind kind(const Built& b, uint32_t value) {
    return b.values.values.at(value).kind;
}

// The encodings above, decoded: a wrong constant would test some other instruction.
TEST(IrExecValues, TheTestEncodingsDecodeAsIntended) {
    const auto ins = walk({kCmpVcc, kCmpxExec, kAndSaveexec, kAndn2Exec, kOrExec, kMovExecS0,
                           kMovExecAllOnes, kEndpgm});
    ASSERT_EQ(ins.size(), 8u);
    EXPECT_EQ(ins[0].fmt, Rdna2Format::VOPC);
    EXPECT_FALSE(vopc_is_cmpx(ins[0].opcode));
    EXPECT_EQ(ins[1].fmt, Rdna2Format::VOPC);
    EXPECT_TRUE(vopc_is_cmpx(ins[1].opcode));
    EXPECT_EQ(ins[2].fmt, Rdna2Format::SOP1);
    EXPECT_EQ(ins[2].opcode, 0x24u);
    EXPECT_EQ(exec_mask_location(ins[2].src[0]), kVccLocation);
    EXPECT_EQ(ins[3].fmt, Rdna2Format::SOP2);
    EXPECT_EQ(ins[3].opcode, 0x15u);
    EXPECT_EQ(exec_mask_location(ins[3].dst), kExecLocation);
    EXPECT_EQ(exec_mask_location(ins[3].src[1]), kExecLocation);
    EXPECT_EQ(ins[4].opcode, 0x11u);
    EXPECT_EQ(ins[5].opcode, 0x04u);
    EXPECT_EQ(exec_mask_location(ins[5].dst), kExecLocation);
    EXPECT_EQ(ins[6].src[0].kind, OperandKind::InlineInt);
}

TEST(IrExecValues, AVcmpxIsAWriteWithinTheEntryValue) {
    const auto b = build({kCmpxExec, kEndpgm});
    const uint32_t entry = b.values.exec_before[0], narrowed = b.values.exec_after[0];
    EXPECT_EQ(kind(b, entry), ExecValue::Kind::Entry);
    EXPECT_EQ(kind(b, narrowed), ExecValue::Kind::Write);
    EXPECT_TRUE(b.values.within(narrowed, entry));
    EXPECT_FALSE(b.values.within(entry, narrowed));
}

// The if/else lowering: saveexec narrows, andn2 runs the second arm under the complement, and the
// or restores the saved mask.
TEST(IrExecValues, AnIfElseLoweringHasItsSubsetsAndComplement) {
    const auto b = build({kCmpVcc, kAndSaveexec, kNop, kAndn2Exec, kNop, kOrExec, kEndpgm});
    const uint32_t entry = b.values.exec_before[0];
    const uint32_t mask = b.values.exec_after[0] == entry ? b.values.defined[0][0].second : kNone;
    const uint32_t then_exec = b.values.exec_after[1], else_exec = b.values.exec_after[3];
    EXPECT_TRUE(b.values.within(mask, entry));   // VCC = EXEC & cmp
    EXPECT_TRUE(b.values.within(then_exec, entry));
    EXPECT_TRUE(b.values.within(else_exec, entry));
    EXPECT_FALSE(b.values.within(else_exec, then_exec));
    const ExecComplement fact = b.values.complement(else_exec);
    ASSERT_TRUE(fact.present());
    EXPECT_EQ(b.values.representative(fact.whole), entry);
    EXPECT_EQ(b.values.representative(fact.part), then_exec);
    EXPECT_TRUE(b.values.within(b.values.exec_after[5], entry));   // OR with a superset
    EXPECT_FALSE(b.values.complement(then_exec).present());
}

TEST(IrExecValues, AMoveIntoExecIsTheSameValue) {
    const auto b = build({kCmpVcc, kAndSaveexec, kMovExecS0, kEndpgm});
    const uint32_t entry = b.values.exec_before[0], restored = b.values.exec_after[2];
    EXPECT_EQ(kind(b, restored), ExecValue::Kind::Write);
    EXPECT_EQ(b.values.values[restored].same_as, entry);
    EXPECT_TRUE(b.values.within(entry, restored));
    EXPECT_TRUE(b.values.within(restored, entry));
}

// v_cmp; s_and_saveexec; s_cbranch_execz over one v_cmpx; merge.
std::vector<uint32_t> one_armed_if() {
    return {kCmpVcc, kAndSaveexec, sopp(kExecz, 1), kCmpxExec, kNop, kEndpgm};
}

TEST(IrExecValues, AOneArmedIfMergeIsAPhiWithinTheSkippedValue) {
    const auto b = build(one_armed_if());
    const uint32_t skipped = b.values.exec_after[1], taken = b.values.exec_after[3];
    const uint32_t merged = b.values.exec_before[4];
    EXPECT_EQ(kind(b, merged), ExecValue::Kind::Phi);
    EXPECT_TRUE(b.values.within(taken, skipped));
    EXPECT_TRUE(b.values.within(merged, skipped));
    EXPECT_FALSE(b.values.within(skipped, merged));
}

TEST(IrExecValues, ADispatchReloadEqualsTheValueReachingItsCase) {
    const auto b = build(one_armed_if(), {true});
    const uint32_t arm = b.values.exec_before[3];
    ASSERT_EQ(kind(b, arm), ExecValue::Kind::DispatchReload);
    EXPECT_EQ(b.values.representative(arm), b.values.exec_after[1]);
    EXPECT_EQ(kind(b, b.values.exec_before[0]), ExecValue::Kind::DispatchReload);
    // The merge's reload stands for its phi, so the phi's relation survives the reload.
    const uint32_t merged = b.values.exec_before[4];
    ASSERT_EQ(kind(b, merged), ExecValue::Kind::DispatchReload);
    EXPECT_EQ(kind(b, b.values.representative(merged)), ExecValue::Kind::Phi);
    EXPECT_TRUE(b.values.within(merged, arm));
}

// s_nop; header: <body>; s_cbranch_execnz header; s_endpgm.
std::vector<uint32_t> bottom_tested_loop(uint32_t body) {
    return {kNop, body, sopp(kExecnz, -2), kEndpgm};
}

TEST(IrExecValues, ALoopThatOnlyNarrowsKeepsItsHeaderWithinThePreheader) {
    const auto b = build(bottom_tested_loop(kCmpxExec));
    const uint32_t header = b.values.exec_before[1];
    EXPECT_EQ(kind(b, header), ExecValue::Kind::Phi);
    EXPECT_TRUE(b.values.within(header, b.values.exec_before[0]));
    EXPECT_TRUE(b.values.within(b.values.exec_after[1], b.values.exec_before[0]));
}

TEST(IrExecValues, ALoopThatSetsExecRefusesTheHeaderInduction) {
    const auto b = build(bottom_tested_loop(kMovExecAllOnes));
    const uint32_t header = b.values.exec_before[1];
    EXPECT_EQ(kind(b, header), ExecValue::Kind::Phi);
    EXPECT_FALSE(b.values.within(header, b.values.exec_before[0]));
}

struct CensusTotal {
    size_t within = 0, equal = 0, stronger = 0, complement = 0, complement_equal = 0;
    std::vector<std::string> weaker;
    void add(const ExecValueCensus& census, const std::string& where) {
        within += census.within().total();
        equal += census.within().counts[size_t(ExecCompare::Equal)];
        stronger += census.within().counts[size_t(ExecCompare::StaticStronger)];
        complement += census.complement().total();
        complement_equal += census.complement().counts[size_t(ExecCompare::Equal)];
        for (const auto& c : census.weaker())
            weaker.push_back(where + " " + c.site + " pc=" + std::to_string(c.pc) + " " + c.detail);
    }
};

// Every exec_within answer and complement read of every committed fixture, in each stage.
TEST(IrExecValuesCensus, NoFixtureComparisonIsStaticWeaker) {
    CensusTotal total;
    for (const auto& path : prosper::test::cfg_fixtures::guest_program_fixtures(__FILE__)) {
        const auto words = prosper::test::cfg_fixtures::load_words(path);
        const std::string name = path.filename().string();
        {
            ExecValueCensus census;
            (void)recompile_vertex(words.data(), words.size());
            total.add(census, name + " vertex");
        }
        {
            ExecValueCensus census;
            (void)recompile_fragment(words.data(), words.size());
            total.add(census, name + " fragment");
        }
        {
            ExecValueCensus census;
            (void)recompile_compute(words.data(), words.size(), nullptr, ComputeShaderConfig{});
            total.add(census, name + " compute");
        }
    }
    for (const auto& line : total.weaker) ADD_FAILURE() << line;
    RecordProperty("within", static_cast<int>(total.within));
    RecordProperty("static_stronger", static_cast<int>(total.stronger));
    RecordProperty("complement", static_cast<int>(total.complement));
}

// The table-less fixtures reach few exec_within calls, so the same census also runs over table-
// backed lane programs that exercise each recorder: V_CMPX, saveexec, the andn2 partition and its
// complement read, the or restore, and a one-armed execz if whose merge is a phi. Dropping the
// compare rule or the complement rule from exec_values makes this test report static-weaker.
constexpr uint32_t kCmpGtU32 = 0x7D880702u;   // v_cmp_gt_u32 vcc, v2, v3
constexpr uint32_t kCmpxGtU32 = 0x7DA80702u;   // v_cmpx_gt_u32 exec, v2, v3
constexpr uint32_t kSaveS0 = 0xBE80047Eu;   // s_mov_b64 s[0:1], exec
constexpr uint32_t kMovV20V2 = 0x7E280302u, kMovV20V3 = 0x7E280303u;   // v_mov_b32 v20, v2/v3
constexpr uint32_t kMovV1V20 = 0x7E020314u;   // v_mov_b32 v1, v20

std::vector<std::vector<uint32_t>> recorder_programs() {
    return {
        {kCmpGtU32, kAndSaveexec, kMovV20V2, kAndn2Exec, kMovV20V3, kOrExec, kMovV1V20},
        {kSaveS0, kCmpxGtU32, sopp(kExecz, 1), kMovV20V2, kMovExecS0, kMovV1V20},
        {kCmpGtU32, kAndSaveexec, sopp(kExecz, 1), kMovV20V2, kAndn2Exec, sopp(kExecz, 1),
         kMovV20V3, kOrExec, kMovV1V20},
        // A word written under EXEC, then again under a V_CMPX narrowing of it: the second write
        // asks whether the new EXEC is within the first write's owner.
        {kSaveS0, kMovV20V2, kCmpxGtU32, kMovV20V3, kMovExecS0, kMovV1V20},
    };
}

TEST(IrExecValuesCensus, NoRecorderProgramComparisonIsStaticWeaker) {
    CensusTotal total;
    size_t index = 0;
    for (const auto& ops : recorder_programs()) {
        for (uint32_t wave : {32u, 64u}) {
            const auto p = prosper::test::ir::lane_program(ops, 64, wave);
            ExecValueCensus census;
            (void)recompile_compute(p.code.data(), p.code.size(), &p.table, p.config);
            total.add(census, "program " + std::to_string(index) + " wave" + std::to_string(wave));
        }
        ++index;
    }
    for (const auto& line : total.weaker) ADD_FAILURE() << line;
    EXPECT_GT(total.equal, 0u) << "no recorder program reached exec_within";
    EXPECT_GT(total.complement_equal, 0u) << "no recorder program read exec_complement_in";
    RecordProperty("within", static_cast<int>(total.within));
    RecordProperty("static_stronger", static_cast<int>(total.stronger));
    RecordProperty("complement", static_cast<int>(total.complement));
}

}   // namespace
