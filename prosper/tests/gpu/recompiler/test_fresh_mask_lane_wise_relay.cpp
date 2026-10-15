// #5167: the fabricated-fresh-mask-source guard (#4949) loses EXEC confinement where a value whose
// active lanes are completely defined flows through lane-wise arithmetic, so a predicate that
// never observes the undefined inactive lanes is refused. The four loss paths below were traced in
// Kena: Bridge of Spirits' refused title-screen programs. Each accepted case has a refused twin in
// which the undefined lanes CAN be observed, so the guard's soundness is what is being kept:
//   1. a loop-carried (soft) word meets a confined one (cs 0x5006f70000 pc52, v_mac_f32);
//   2. an LDS or typed-buffer read under a narrowed EXEC is a complete active-lane writer
//      (cs 0x500a910000 pc85, ds_read_b32; PPSA01826/PPSA03416 tbuffer_load_format);
//   3. the second arm of an if/else (S_ANDN2 EXEC, saved, EXEC) completes the saved mask
//      (ps 0x5006bd0000 pc85-pc105);
//   4. a structured merge meets two nested owners at the narrower one (ps 0x500aed0000).
#include "fixtures/checked_compute_launch.hpp"
#include "gpu/recompiler/rdna2_active_vector_words.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prosper::gpu;
using Words = std::vector<uint32_t>;

constexpr uint32_t cmpx_a = 0x7da6009fu;   // v_cmpx_le_u32 31, v0: EXEC A = EXEC & (v0 >= 31)
constexpr uint32_t cmpx_b = 0x7da600afu;   // v_cmpx_le_u32 47, v0: EXEC B, a proved subset of A
constexpr uint32_t cmpx_c = 0x7da600b7u;   // v_cmpx_le_u32 55, v0
constexpr uint32_t full_exec = 0xbefe04c1u;   // s_mov_b64 exec, -1
constexpr uint32_t cmp_vcc_47 = 0x7d8600afu;   // v_cmp_le_u32 vcc, 47, v0
constexpr uint32_t and_saveexec_vcc = 0xbe84246au;   // s_and_saveexec_b64 s[4:5], vcc
constexpr uint32_t andn2_exec_saved = 0x8afe7e04u;   // s_andn2_b64 exec, s[4:5], exec
constexpr uint32_t restore_exec = 0xbefe0404u;   // s_mov_b64 exec, s[4:5]
constexpr uint32_t mov6 = 0x7e0c0281u, mov7 = 0x7e0e0282u, mov6_3 = 0x7e0c0283u;
constexpr uint32_t xor8 = 0x3a100f06u;   // v8 = v6 ^ v7
constexpr uint32_t fmac6 = 0x560c0ef2u;   // v_fmac_f32 v6, 1.0, v7 (reads the old v6)
constexpr uint32_t compare6 = 0x7d840d06u, compare8 = 0x7d841108u;
constexpr uint32_t ds_read_v8 = 0xd8d80000u;   // ds_read_b32 v8, v0 (dword1 below)
constexpr uint32_t end = 0xbf810000u;

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

// The producing emitter, not a test-built mark: emit_alu records the EXEC facts and the marks.
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
        const auto inventory = decode({mov6, mov7, xor8, compare8, 0x7e260314u});
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
    // Only the mark transfer, for memory forms the compute fixture has no resource for.
    void record(const Words& code) {
        for (const auto& in : decode(code))
            record_vector_word_marks(rs, in, scalar_source_marks(rs, in));
    }
    bool defined(uint32_t comparison) {
        return fresh_mask_sources_defined(b, rs, rdna2_decode_one(&comparison, 1));
    }
};

struct Result {
    Words module;
    std::string reason;
};
Result compile(const Words& body) {
    ComputeShaderConfig config;
    config.local_x = 64;
    config.wave_size = 64;
    config.native_subgroup_size = 64;
    config.user_sgprs.resize(4);
    config.tgid_x_en = true;   // s4 is a genuine dynamic selector for the one-arm controls
    prosper::test::complete_compute_launch(config, 2);
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
    static uint64_t next_address = 0x5167000000ull;
    const uint64_t address = ++next_address;
    auto module = recompile_compute(code.data(), code.size(), &table, config,
                                    {RecompileDiagnosticStage::Compute, address});
    return {std::move(module), last_terminal_reject_reason(address)};
}
void accepted(const Result& result) {
    EXPECT_FALSE(result.module.empty()) << result.reason;
}
void refused(const Result& result, uint32_t pc) {
    EXPECT_TRUE(result.module.empty());
    EXPECT_NE(result.reason.find("fabricated-fresh-mask-source"), std::string::npos)
        << result.reason;
    EXPECT_NE(result.reason.find("pc=" + std::to_string(pc) + " "), std::string::npos)
        << result.reason;
}
// A scalar top-tested loop of three iterations, whose body is emitted once per iteration.
Words counted_loop(const Words& preheader, const Words& body) {
    const uint32_t size = static_cast<uint32_t>(body.size() + 2u);
    return cat({preheader,
                {0xbe960380u, 0xbe970383u, 0xbf0a1716u, 0xbf840000u | size},
                body,
                {0x80168116u, 0xbf820000u | ((0x10000u - size - 2u) & 0xffffu)}});
}
}   // namespace

TEST(FreshMaskLaneWiseRelay, EncodingsNameTheFormsTheyClaim) {
    const auto partition = decode({cmp_vcc_47, and_saveexec_vcc, andn2_exec_saved, restore_exec});
    ASSERT_EQ(partition.size(), 4u);
    EXPECT_EQ(partition[0].fmt, Rdna2Format::VOPC);
    EXPECT_FALSE(vopc_is_cmpx(partition[0].opcode));
    EXPECT_EQ(partition[1].fmt, Rdna2Format::SOP1);
    EXPECT_EQ(partition[1].opcode, 0x24u);
    EXPECT_EQ(partition[2].fmt, Rdna2Format::SOP2);
    EXPECT_EQ(partition[2].opcode, 0x15u);
    EXPECT_EQ(partition[2].src[0].value, 4);
    EXPECT_EQ(partition[2].src[1].value, 126);
    EXPECT_EQ(partition[3].opcode, 0x04u);
    const auto read = decode({ds_read_v8, 0x08000000u});
    ASSERT_EQ(read.size(), 1u);
    EXPECT_EQ(read[0].fmt, Rdna2Format::DS);
    EXPECT_EQ(read[0].opcode, 0x36u);
    EXPECT_EQ(read[0].dst.value, 8);
    EXPECT_EQ(read[0].src[0].value, 0);
}

// ---- 1. A loop-carried explicit source does not veto a confined owner ------------------------
// Only the explicit operands of one instruction: they are read in the same lane under the same
// EXEC. The old-destination meet and CFG joins keep their veto (MacCannotBorrowAnOwnerForAn-
// UnconfinedSoftOldWord in test_confined_source_union.cpp is that control).

TEST(FreshMaskLaneWiseRelay, ALoopCarriedWordAddsItsRootsToTheConfinedOwner) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov7, cmpx_a, mov6}));
    const uint32_t a = e.rs.exec;
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    mark_vector_loop_carried(e.rs, decode({mov7}), 0, 1, check.carried, check.confined);
    ASSERT_TRUE(e.rs.vreg_source_marks.at(7).placeholder);
    ASSERT_FALSE(e.rs.vreg_source_marks.at(7).hard);
    ASSERT_TRUE(e.emit({xor8}));
    const auto& relay = e.rs.vreg_source_marks.at(8);
    EXPECT_TRUE(relay.hard) << "the missing outer lanes of v6 are still missing";
    EXPECT_EQ(relay.outside_exec, a);
    const uint64_t root = blanket_root(0, kVectorBlanketBase + 7);
    EXPECT_TRUE(relay.blanket_roots.contains(root));
    ASSERT_TRUE(e.defined(compare8));
    EXPECT_TRUE(e.b.loop_blanket_roots_read.contains(root))
        << "the accepted read still owes the loop its backedge check";
}

TEST(FreshMaskLaneWiseRelay, ALoopCarriedWordCannotLetAWiderExecReadTheMissingLanes) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov7, cmpx_a, mov6}));
    LoopBlanketCheck check(e.rs, std::set<int>{}, 0);
    mark_vector_loop_carried(e.rs, decode({mov7}), 0, 1, check.carried, check.confined);
    ASSERT_TRUE(e.emit({xor8, full_exec, cmpx_b}));
    // B is derived from the new full EXEC, not from A: v6's missing lanes may be enabled.
    EXPECT_FALSE(e.defined(compare8));
}

TEST(FreshMaskLaneWiseRelay, ASoftWordWithoutRootsStillVetoesTheOwner) {
    Emitter e;
    ASSERT_TRUE(e.emit({mov7, cmpx_a, mov6}));
    auto& unproved = e.rs.vreg_source_marks[7];
    unproved = {};
    unproved.placeholder = true;   // soft, but no loop assumption names its supply
    ASSERT_TRUE(e.emit({xor8}));
    EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, 0u);
    EXPECT_FALSE(e.defined(compare8));
}

TEST(FreshMaskLaneWiseRelay, PublicCompilerAdmitsTheLoopRelayAndRefusesItsWiderTwin) {
    // v7 is rewritten every iteration, so the loop carries it; v6 is complete only under A.
    accepted(compile(counted_loop({mov7}, {cmpx_a, mov6, xor8, compare8, full_exec, mov7})));
    // The same relay read after EXEC was widened again observes v6's missing lanes.
    refused(compile(counted_loop({mov7}, {cmpx_a, mov6, full_exec, xor8, compare8, mov7})), 9);
}

// ---- 2. Ordinary memory reads complete their active lanes --------------------------------------

TEST(FreshMaskLaneWiseRelay, AnLdsReadUnderANarrowedExecConfinesItsMissingLanes) {
    for (uint32_t opcode : {0x36u, 0x37u, 0x38u, 0x76u, 0x77u, 0xfeu, 0xffu}) {
        Emitter e;
        ASSERT_TRUE(e.emit({cmpx_a}));
        e.record({0xd8000000u | (opcode << 18u), 0x08000000u});
        EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, e.rs.exec) << opcode;
        EXPECT_TRUE(e.defined(compare8)) << opcode;
        ASSERT_TRUE(e.emit({full_exec}));
        EXPECT_FALSE(e.defined(compare8)) << "full EXEC reads the never-written lanes";
    }
}

TEST(FreshMaskLaneWiseRelay, CrossLaneAndGlobalDataShareReadsStayIncomplete) {
    // ds_swizzle_b32 (0x35) and ds_bpermute_b32 (0xb3) read other lanes; GDS is device-global.
    for (const Words& words : {Words{0xd8d40000u, 0x08000000u}, Words{0xdacc0000u, 0x08000100u},
                               Words{0xd8da0000u, 0x08000000u}}) {
        Emitter e;
        ASSERT_TRUE(e.emit({cmpx_a}));
        e.record(words);
        EXPECT_FALSE(e.defined(compare8)) << std::hex << words[0];
    }
}

TEST(FreshMaskLaneWiseRelay, AnLdsReadWithAnUnconfinedAddressConfinesNothing) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a}));
    e.record({ds_read_v8, 0x08000014u});   // address v20 was never written
    EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, 0u);
    EXPECT_FALSE(e.defined(compare8));
}

TEST(FreshMaskLaneWiseRelay, ATypedBufferLoadIsCompleteButItsTfeAndD16FormsAreNot) {
    // tbuffer_load_format_x..xyzw v8, v0 offen, 32_FLOAT: the per-lane format-load path MUBUF uses.
    for (uint32_t opcode : {0u, 1u, 2u, 3u}) {
        Emitter e;
        ASSERT_TRUE(e.emit({cmpx_a}));
        e.record({0xe8b01000u | (opcode << 16u), 0x80020800u});
        EXPECT_TRUE(e.defined(compare8)) << opcode;
    }
    for (const Words& words : {Words{0xe8b01000u, 0x80820800u},   // TFE status word
                               Words{0xe8b01000u, 0x80220800u}}) {   // tbuffer_load_format_d16_x
        Emitter e;
        ASSERT_TRUE(e.emit({cmpx_a}));
        e.record(words);
        EXPECT_FALSE(e.defined(compare8)) << std::hex << words[0] << " " << words[1];
    }
}

TEST(FreshMaskLaneWiseRelay, PublicCompilerAdmitsAnLdsReadRelayAndRefusesItsWiderTwin) {
    accepted(compile({cmpx_a, ds_read_v8, 0x08000000u, compare8}));
    // #5174: outside A, v8 was never written and compute never loads it, so the wider read is
    // admitted as an arbitrary value (a refusal before that decision made it an over-refusal).
    accepted(compile({cmpx_a, ds_read_v8, 0x08000000u, full_exec, compare8}));
    // #5185: an old v8 relayed from v19, itself never written, is arbitrary outside A too.
    accepted(compile({0x7e100313u, cmpx_a, ds_read_v8, 0x08000000u, full_exec, compare8}));
    // A hard old v8 (LDS data read at the never-written address v19) survives outside A: refused.
    refused(
        compile({ds_read_v8, 0x08000013u, cmpx_a, ds_read_v8, 0x08000000u, full_exec, compare8}),
        6);
}

// ---- 3. The second arm of an if/else completes the saved mask ----------------------------------

Words if_else(const Words& then_arm, const Words& else_arm) {
    return cat({{cmpx_a, cmp_vcc_47, and_saveexec_vcc},
                then_arm,
                {andn2_exec_saved},
                else_arm,
                {restore_exec}});
}

TEST(FreshMaskLaneWiseRelay, BothArmsOfAPartitionDefineTheSavedMask) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a}));
    const uint32_t outer = e.rs.exec;
    ASSERT_TRUE(e.emit({cmp_vcc_47, and_saveexec_vcc, mov6, andn2_exec_saved}));
    const auto partition = e.rs.exec_complement_in.find(e.rs.exec);
    ASSERT_NE(partition, e.rs.exec_complement_in.end());
    EXPECT_EQ(partition->second.first, outer);
    ASSERT_TRUE(e.emit({mov6_3, restore_exec}));
    ASSERT_EQ(e.rs.exec, outer);
    EXPECT_EQ(e.rs.vreg_source_marks.at(6).outside_exec, outer);
    EXPECT_TRUE(e.defined(compare6));
}

TEST(FreshMaskLaneWiseRelay, OneArmOfAPartitionCannotDefineTheOther) {
    Emitter e;
    ASSERT_TRUE(e.emit(if_else({mov6}, {mov7})));
    EXPECT_FALSE(e.defined(compare6)) << "v6 was never written in the second arm's lanes";
    Emitter second;
    ASSERT_TRUE(second.emit(if_else({mov7}, {mov6})));
    EXPECT_FALSE(second.defined(compare6)) << "nor in the first arm's";
}

TEST(FreshMaskLaneWiseRelay, ANarrowerSecondArmDoesNotCompleteThePartition) {
    Emitter e;
    ASSERT_TRUE(e.emit(if_else({mov6}, {cmpx_c, mov6_3})));
    EXPECT_FALSE(e.defined(compare6));
}

TEST(FreshMaskLaneWiseRelay, PublicCompilerAdmitsThePartitionAndRefusesItsOneArmTwin) {
    accepted(compile(cat({if_else({mov6}, {mov6_3}), {compare6}})));
    // #5174: v6 was never written in the second arm's lanes and compute never loads it, so the
    // one-arm read is admitted as an arbitrary value (an over-refusal under that decision).
    accepted(compile(cat({if_else({mov6}, {mov7}), {compare6}})));
    // #5185: an old v6 relayed from v19, itself never written, is arbitrary there too.
    accepted(compile(cat({{0x7e0c0313u}, if_else({mov6}, {mov7}), {compare6}})));
    // With a hard old v6 (LDS data read at the never-written address v19) the second arm's lanes
    // keep it: refused.
    refused(compile(cat({{0xd8d80000u, 0x06000013u}, if_else({mov6}, {mov7}), {compare6}})), 9);
}

// ---- 4. A structured merge meets nested owners at the narrower one ------------------------------

TEST(FreshMaskLaneWiseRelay, AMergeOfNestedOwnersKeepsTheNarrowerOne) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6, cmpx_b, mov7}));
    const uint32_t b = e.rs.exec;
    VectorWordMarks skipped{{6, e.rs.vreg_source_marks.at(6)}};   // owner A
    ASSERT_TRUE(e.emit({fmac6}));   // owner B on the taken arm
    ASSERT_EQ(e.rs.vreg_source_marks.at(6).outside_exec, b);
    VectorWordMarks merged = e.rs.vreg_source_marks;
    join_vector_word_marks(merged, skipped, &e.rs);
    EXPECT_EQ(merged.at(6).outside_exec, b);
    VectorWordMarks context_free = e.rs.vreg_source_marks;
    join_vector_word_marks(context_free, skipped);
    EXPECT_EQ(context_free.at(6).outside_exec, 0u) << "without the EXEC facts, owners must match";
}

TEST(FreshMaskLaneWiseRelay, AMergeOfUnrelatedOwnersStaysUnconfined) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a, mov6, full_exec, cmpx_b, mov7}));
    VectorWordMarks skipped{{6, e.rs.vreg_source_marks.at(6)}};
    ASSERT_TRUE(e.emit({0x7e0c0307u}));   // v6 = v7 under B, unrelated to A
    VectorWordMarks merged = e.rs.vreg_source_marks;
    join_vector_word_marks(merged, skipped, &e.rs);
    EXPECT_EQ(merged.at(6).outside_exec, 0u);
}

TEST(FreshMaskLaneWiseRelay, PublicCompilerAdmitsTheNestedMergeAndRefusesItsUnrelatedTwin) {
    // s_cmp_eq_u32 s4, 0; s_cbranch_scc1 over one instruction: a structured one-arm region.
    const Words one_arm{0xbf068004u, 0xbf840001u, fmac6};
    accepted(compile(cat({{cmpx_a, mov6, cmpx_b, mov7}, one_arm, {compare6}})));
    // v6's lanes of B outside A were never written: arbitrary, so the unrelated owners no longer
    // refuse (#5185). With a hard old v6 (LDS data at the never-written address v19) they do.
    accepted(compile(cat({{cmpx_a, mov6, full_exec, cmpx_b, mov7}, one_arm, {compare6}})));
    refused(compile(cat({{0xd8d80000u, 0x06000013u, cmpx_a, mov6, full_exec, cmpx_b, mov7},
                         one_arm,
                         {compare6}})),
            10);
}

// ---- 6. ds_read_addtid_b32 completes its active lanes when M0 is genuine (#5183) ----------------
// Each EXEC lane reads LDS at M0 + offset + its own lane id: no VGPR address, no other lane. M0 is
// the whole address input, so the read is complete exactly when M0 is genuine. Kena cs
// 0x5008e80000 pc1248/pc1260 (under s_movk_i32 m0, 0) fed its last refusal at pc1335.

namespace {
constexpr uint32_t addtid_v8 = 0xdac40000u;   // ds_read_addtid_b32 v8 (dword1 below)
constexpr uint32_t addtid_v8_dst = 0x08000000u;
constexpr uint32_t m0_zero = 0xb07c0000u;   // s_movk_i32 m0, 0
}   // namespace

TEST(FreshMaskLaneWiseRelay, AddtidEncodingNamesTheFormItClaims) {
    const auto read = decode({addtid_v8, addtid_v8_dst, m0_zero});
    ASSERT_EQ(read.size(), 2u);
    EXPECT_EQ(read[0].fmt, Rdna2Format::DS);
    EXPECT_EQ(read[0].opcode, 0xb1u);
    EXPECT_FALSE(read[0].ds_gds);
    EXPECT_EQ(read[0].dst.value, 8);
    EXPECT_EQ(read[1].fmt, Rdna2Format::SOPK);
    EXPECT_EQ(read[1].dst.value, 124);
}

TEST(FreshMaskLaneWiseRelay, AnAddtidReadWithAGenuineM0ConfinesItsMissingLanes) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a}));
    e.rs.sreg[124] = e.b.uconst(0);
    e.record({addtid_v8, addtid_v8_dst});
    EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, e.rs.exec);
    EXPECT_TRUE(e.defined(compare8));
    ASSERT_TRUE(e.emit({full_exec}));
    EXPECT_FALSE(e.defined(compare8)) << "full EXEC reads the never-written lanes";
}

TEST(FreshMaskLaneWiseRelay, AnAddtidReadWithAMissingOrFabricatedM0ConfinesNothing) {
    for (const int arm : {0, 1, 2}) {
        Emitter e;
        ASSERT_TRUE(e.emit({cmpx_a}));
        if (arm == 1) {   // one path never wrote M0
            e.rs.sreg[124] = e.b.uconst(0);
            e.rs.sreg_merge_placeholder.insert(124);
        } else if (arm == 2) {   // M0 holds a saved EXEC half no dword can stand for, even
            e.rs.sreg[124] = e.b.uconst(0);   // under a loop assumption
            e.rs.unavailable_saved_mask_words.insert(124);
            e.rs.sreg_loop_blanket[124] = {blanket_root(0, 124)};
        }   // arm 0: M0 was never written
        e.record({addtid_v8, addtid_v8_dst});
        EXPECT_EQ(e.rs.vreg_source_marks.at(8).outside_exec, 0u) << arm;
        EXPECT_FALSE(e.defined(compare8)) << arm;
    }
}

TEST(FreshMaskLaneWiseRelay, AnAddtidReadOnALoopAssumedM0OwesTheLoopItsCheck) {
    Emitter e;
    ASSERT_TRUE(e.emit({cmpx_a}));
    e.rs.sreg[124] = e.b.uconst(0);
    e.rs.sreg_merge_placeholder.insert(124);
    const uint64_t root = blanket_root(0, 124);
    e.rs.sreg_loop_blanket[124] = {root};
    e.record({addtid_v8, addtid_v8_dst});
    ASSERT_TRUE(e.defined(compare8));
    EXPECT_TRUE(e.b.loop_blanket_roots_read.contains(root));
}

TEST(FreshMaskLaneWiseRelay, AddtidWritesAndGdsReadsStayIncomplete) {
    // ds_write_addtid_b32 v8 (0xb0) writes no VGPR; the GDS-bit read is device-global data.
    for (const Words& words :
         {Words{0xdac00000u, 0x00000800u}, Words{0xdac60000u, addtid_v8_dst}}) {
        const auto in = decode(words);
        ASSERT_EQ(in.size(), 1u);
        EXPECT_FALSE(vector_read_completes_active_word(in[0])) << std::hex << words[0];
    }
    EXPECT_TRUE(vector_read_completes_active_word(decode({addtid_v8, addtid_v8_dst})[0]));
}

TEST(FreshMaskLaneWiseRelay, ADispatcherAddtidReadNeedsAGenuineUnrelayedM0) {
    const auto read = decode({addtid_v8, addtid_v8_dst});
    ASSERT_EQ(read.size(), 1u);
    const ActiveVectorWords state;
    const std::set<int> none;
    EXPECT_TRUE(active_vector_sources_defined(state, read[0], none, none, {124}, none, true));
    EXPECT_FALSE(active_vector_sources_defined(state, read[0], none, none, {}, none, true))
        << "M0 is not a MUST scalar word";
    EXPECT_FALSE(active_vector_sources_defined(state, read[0], none, none, {124}, {124}, true))
        << "M0 was relayed from a possibly fabricated source";
}

TEST(FreshMaskLaneWiseRelay, PublicCompilerAdmitsAnAddtidReadOverAHardOldWord) {
    // v8 first holds LDS data read at the never-written address v19 (hard, #5185); under A the
    // ADDTID read replaces every active lane, so the compare under A reads only genuine lanes.
    accepted(
        compile({ds_read_v8, 0x08000013u, m0_zero, cmpx_a, addtid_v8, addtid_v8_dst, compare8}));
    // Read under the full EXEC again, the hard lanes outside A are observed.
    refused(compile({ds_read_v8, 0x08000013u, m0_zero, cmpx_a, addtid_v8, addtid_v8_dst, full_exec,
                     compare8}),
            7);
    // M0 written on one path only (s_cmp_eq_u32 s4, 0; s_cbranch_scc1 over the write): the read
    // may address through a fabricated M0, so it completes nothing.
    refused(compile({ds_read_v8, 0x08000013u, 0xbf068004u, 0xbf840001u, m0_zero, cmpx_a, addtid_v8,
                     addtid_v8_dst, compare8}),
            8);
    // M0 written on both paths with different values: a genuine phi, admitted.
    accepted(
        compile({ds_read_v8, 0x08000013u, m0_zero, 0xbf068004u, 0xbf840001u,
                 0xb07c0004u /* s_movk_i32 m0, 4 */, cmpx_a, addtid_v8, addtid_v8_dst, compare8}));
}
