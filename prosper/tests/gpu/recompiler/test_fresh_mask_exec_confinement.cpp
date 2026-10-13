// A fresh VOPC predicate reads only the lanes its own EXEC enables. A never-written VGPR is
// confined to the lanes outside EXEC G only when a COMPLETE full-word write under G defined every
// lane inside G. Then a compare under the same G never observes the missing lanes.
//
// Positive: complete full-word writes under G, compared under G, are defined.
// Controls (all must refuse, all under the same G): a different EXEC value; a WRITELANE that
// replaces one selected lane; an SDWA form that preserves destination bits; a MOVRELD whose
// destination set is indexed; a compare that also reads an absent scalar.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

using namespace prosper::gpu;

namespace {
// v_mov_b32 v6, <literal>: a complete full-word write.
constexpr std::array kMovV6 = {0x7e0c02ffu, 0x00001234u};
// VOP2 op 0x1b: v8 = <literal> op v6 (complete full-word write, reads v6).
constexpr std::array kV8FromV6 = {0x36100cffu, 0x00003fffu};
// v_mov_b32_sdwa v6, 0 with WORD_0 / UNUSED_PRESERVE: keeps the upper destination bits.
constexpr std::array kSdwaV6 = {0x7e0c02f9u, 0x00851480u};
// v_writelane_b32 v6, s2, lane 0: replaces one selected lane.
constexpr std::array kWritelaneV6 = {0xd7610006u, 0x00010002u};
// v_movreld_b32 with VDST v6 (indexed through M0), inline-zero source.
constexpr std::array kMovrelV6 = {0x7e0c8480u};
// FMAC_F32 (0x12b) and the decoder-admitted packed FMAC lowering (0x13c) both read the old
// destination. The latter is not a verified architectural e64 encoding.
// SRC2 encodes s2: the decoder exposes a third source field for these opcodes, so the test makes
// it genuine, leaving the old-destination accumulator as the only missing input.
constexpr std::array kFmacV6 = {0xd12b0006u, 0x000a0501u};
constexpr std::array kPkFmacV6 = {0xd13c0006u, 0x000a0501u};
// v_mac_f32 v6, v1, v2 (VOP2 0x1f): lane-wise accumulator reading its own lane's old word.
constexpr std::array kMacV6 = {0x3e0c0501u};
// Synthetic forms accepted by prosper's existing SDWA decoder/lowering. These are not captured
// PS5 instructions or LLVM-supported SDWA encodings; their old destination still participates
// in the implemented operation and therefore cannot be certified as a complete replacement.
constexpr std::array kSdwaMacV6 = {0x3e0c04f9u, 0x06060601u};
constexpr std::array kSdwaFmacV6 = {0x560c04f9u, 0x06060601u};
constexpr std::array kSdwaPkFmacV6 = {0x780c04f9u, 0x06060601u};
constexpr std::array kSdwaFullMovV6 = {0x7e0c02f9u, 0x00060601u};
// VOPC compare, src0 = v6, vsrc1 = v6.
constexpr std::array kCmpV6V6 = {0x7c200d06u};
// VOPC compare, src0 = v6, vsrc1 = v8.
constexpr std::array kCmpV8 = {0x7c201106u};
// VOPC compare, src0 = s20 (absent scalar), vsrc1 = v6.
constexpr std::array kCmpS20V6 = {0x7c200c14u};
// v_cmpx_eq_u32 0, v0: narrows EXEC from a genuine, independently supplied v0.
constexpr std::array kCmpxV0 = {0x7da40080u};

std::vector<Rdna2Inst> decode(std::span<const uint32_t> words) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    return ins;
}

std::vector<uint32_t> cat(std::initializer_list<std::span<const uint32_t>> parts) {
    std::vector<uint32_t> out;
    for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

struct Harness {
    SpirvCompute b;
    RegState rs;
    Harness() {
        b.wave_size = 64;
        b.begin_fragment();
        rs.vector_wave_size = 64;
        rs.vector_full_exec = false;
        rs.max_vgpr = 7;
    }
};

// Writes `writes` under EXEC G, then asks whether `reader` (a predicate) is defined under the
// same G, or under another EXEC value when `same_exec` is false.
bool predicate_defined(std::span<const uint32_t> writes, std::span<const uint32_t> reader,
                       bool same_exec = true,
                       const std::function<void(RegState&, SpirvCompute&)>& setup = {}) {
    Harness h;
    if (setup) setup(h.rs, h.b);
    // G is a narrowed EXEC with a value of its own. The seed reads narrowedness, so it must be set
    // first.
    const uint32_t g = h.b.btrue();
    h.rs.exec = g;
    h.rs.exec_narrowed = true;
    const auto write_ins = decode(writes);
    const auto read_ins = decode(reader);
    if (write_ins.empty() || read_ins.size() != 1) return false;
    std::vector<Rdna2Inst> all = write_ins;
    all.push_back(read_ins[0]);
    seed_vector_word_definedness(h.rs, all, false, 64);
    for (const auto& in : write_ins)
        record_vector_word_marks(h.rs, in, scalar_source_marks(h.rs, in));
    h.rs.exec = same_exec ? g : h.b.uconst(78);
    h.rs.exec_narrowed = true;
    return fresh_mask_sources_defined(h.b, h.rs, read_ins[0]);
}

// Exercise the producing emitter, rather than inserting the subset fact in the test. Removing
// emit_alu's CMPX provenance assignment must break the positive case. The unrelated-parent control
// still emits successfully; only its reader's fresh-mask admission must fail.
bool predicate_defined_in_subset(std::span<const uint32_t> writes, std::span<const uint32_t> reader,
                                 bool subset_of_writer) {
    Harness h;
    h.rs.exec = h.b.btrue();
    h.rs.vreg[0] = h.b.uconst(0);
    const auto write_ins = decode(writes);
    const auto read_ins = decode(reader);
    const auto cmpx = decode(kCmpxV0);
    EXPECT_EQ(cmpx.size(), 1u);
    if (cmpx.size() != 1) return false;
    EXPECT_EQ(cmpx[0].fmt, Rdna2Format::VOPC);
    EXPECT_EQ(cmpx[0].opcode, 0xd2u);
    EXPECT_TRUE(vopc_is_cmpx(cmpx[0].opcode));
    if (write_ins.empty() || read_ins.size() != 1) return false;
    std::vector<Rdna2Inst> all = write_ins;
    all.push_back(read_ins[0]);
    all.push_back(cmpx[0]);
    seed_vector_word_definedness(h.rs, all, false, 64);
    const auto emit = [&](const Rdna2Inst& in) {
        const auto before = snapshot_saved_b64_masks(h.b, h.rs, in);
        bool ok = true;
        const bool handled = emit_alu(h.b, h.rs, in, ok, /*allow_exec_update=*/true);
        EXPECT_TRUE(handled);
        EXPECT_TRUE(ok) << h.b.stage_reject_reason;
        if (!handled || !ok) return false;
        record_scalar_write(h.rs, in, false, before);
        return true;
    };
    if (!emit(cmpx[0])) return false;
    const uint32_t g = h.rs.exec;
    EXPECT_TRUE(h.rs.exec_narrowed);
    for (const auto& in : write_ins)
        if (!emit(in)) return false;
    // An unrelated Bool value has no proven relation to the writer's EXEC. CMPX still records a
    // true subset relation to that value; it cannot authenticate the writer's missing lanes.
    if (!subset_of_writer) h.rs.exec = h.b.ucmp(Op_IEqual, h.b.uconst(1), h.b.uconst(2));
    const uint32_t parent = h.rs.exec;
    if (!emit(cmpx[0])) return false;
    EXPECT_NE(h.rs.exec, parent);
    if (subset_of_writer) EXPECT_EQ(parent, g);
    return fresh_mask_sources_defined(h.b, h.rs, read_ins[0]);
}
void sgpr_genuine(RegState& rs, SpirvCompute& b, int r) {
    rs.sreg[r] = b.uconst(5);
}
void vgpr_genuine(RegState& rs, SpirvCompute& b, int r) {
    rs.vreg[r] = b.uconst(7);
}

// v6 holds a placeholder confined to G. A lane-wise accumulator under M combines its own lane's
// clean sources with its own lane's old word, so the meet stays G: it is accepted under G, and it
// is NOT rebased to a different writer EXEC M.
bool accumulator_meet(bool mac_under_writer_exec, std::span<const uint32_t> words = kMacV6) {
    Harness h;
    h.rs.vreg[1] = h.b.uconst(3);
    h.rs.vreg[2] = h.b.uconst(4);
    const uint32_t g = h.b.btrue();
    h.rs.exec = g;
    h.rs.exec_narrowed = true;
    const auto mov = decode(kMovV6);
    const auto mac = decode(words);
    const auto cmp = decode(kCmpV6V6);
    if (mov.size() != 1 || mac.size() != 1 || cmp.size() != 1) return false;
    std::vector<Rdna2Inst> all = mov;
    all.insert(all.end(), mac.begin(), mac.end());
    all.insert(all.end(), cmp.begin(), cmp.end());
    seed_vector_word_definedness(h.rs, all, false, 64);
    record_vector_word_marks(h.rs, mov[0], scalar_source_marks(h.rs, mov[0]));
    h.rs.exec = mac_under_writer_exec ? g : h.b.uconst(79);
    h.rs.exec_narrowed = true;
    record_vector_word_marks(h.rs, mac[0], scalar_source_marks(h.rs, mac[0]));
    return fresh_mask_sources_defined(h.b, h.rs, cmp[0]);
}
}   // namespace

TEST(FreshMaskExecConfinement, CompleteWriteCompareUnderTheWritersExecIsDefined) {
    // Lanes outside G keep the never-written v8; a compare under the same G never reads them.
    EXPECT_TRUE(predicate_defined(cat({kMovV6, kV8FromV6}), kCmpV8, true));
    EXPECT_TRUE(predicate_defined(kMovV6, kCmpV6V6, true));
}

TEST(FreshMaskExecConfinement, ControlEncodingsDecodeAsTheFormsTheyClaim) {
    // A control that failed to decode would refuse for the wrong reason. Pin each form.
    const auto wl = decode(kWritelaneV6);
    ASSERT_EQ(wl.size(), 1u);
    EXPECT_EQ(wl[0].fmt, Rdna2Format::VOP3);
    EXPECT_EQ(wl[0].opcode, 0x361u);
    const auto sd = decode(kSdwaV6);
    ASSERT_EQ(sd.size(), 1u);
    EXPECT_TRUE(sd[0].has_sdwa);
    const auto mr = decode(kMovrelV6);
    ASSERT_EQ(mr.size(), 1u);
    EXPECT_EQ(mr[0].fmt, Rdna2Format::VOP1);
    EXPECT_EQ(mr[0].opcode, kVop1OpcodeMovreldB32);
    const auto s20 = decode(kCmpS20V6);
    ASSERT_EQ(s20.size(), 1u);
    EXPECT_EQ(s20[0].src[0].kind, OperandKind::SGPR);
    EXPECT_EQ(s20[0].src[0].value, 20);
    for (const auto words : {kFmacV6, kPkFmacV6}) {
        const auto fmac = decode(words);
        ASSERT_EQ(fmac.size(), 1u);
        EXPECT_EQ(fmac[0].fmt, Rdna2Format::VOP3);
        EXPECT_EQ(fmac[0].dst.value, 6);
    }
    EXPECT_EQ(decode(kFmacV6)[0].opcode, 0x12bu);
    EXPECT_EQ(decode(kPkFmacV6)[0].opcode, 0x13cu);
    const auto mac = decode(kMacV6);
    ASSERT_EQ(mac.size(), 1u);
    EXPECT_EQ(mac[0].fmt, Rdna2Format::VOP2);
    EXPECT_EQ(mac[0].opcode, 0x1fu);
    EXPECT_EQ(mac[0].dst.value, 6);
}

TEST(FreshMaskExecConfinement, CompareUnderAnotherExecStillRefuses) {
    // A different EXEC value does not prove its lanes were written: the refusal must remain.
    EXPECT_FALSE(predicate_defined(kMovV6, kCmpV6V6, /*same_exec=*/false));
}

TEST(FreshMaskExecConfinement, WritelaneReplacingOneLaneLeavesOtherLanesUnconfined) {
    // The WRITELANE writes lane 0 only; the other active lanes of v6 were never written.
    EXPECT_FALSE(predicate_defined(kWritelaneV6, kCmpV6V6, true,
                                   [](RegState& rs, SpirvCompute& b) { sgpr_genuine(rs, b, 2); }));
}

TEST(FreshMaskExecConfinement, SdwaPreservedBitsStillRefuse) {
    // The SDWA form keeps the destination's upper 16 bits in every active lane.
    EXPECT_FALSE(predicate_defined(kSdwaV6, kCmpV6V6, true));
}

TEST(FreshMaskExecConfinement, RelativeMovrelDestinationsStillRefuse) {
    // MOVRELD writes only the register M0 selects, not every register up to max_vgpr.
    EXPECT_FALSE(predicate_defined(kMovrelV6, kCmpV6V6, true, [](RegState& rs, SpirvCompute& b) {
        sgpr_genuine(rs, b, 124);
    }));
}

TEST(FreshMaskExecConfinement, MixedAbsentScalarInputStillRefuses) {
    // The compare also reads s20, which was never written and reads as fabricated zero.
    EXPECT_FALSE(predicate_defined(kMovV6, kCmpS20V6, true, [](RegState& rs, SpirvCompute&) {
        rs.absent_sgpr_reads_fabricated_zero = true;
    }));
}

TEST(FreshMaskExecConfinement, CompareUnderACmpxSubsetOfTheWritersExecIsDefined) {
    // v_cmpx sets EXEC' = EXEC & cmp, so EXEC' lies inside the writer's EXEC: the placeholder lanes
    // outside the writer's EXEC are never in EXEC'.
    EXPECT_TRUE(predicate_defined_in_subset(cat({kMovV6, kV8FromV6}), kCmpV8, true));
}

TEST(FreshMaskExecConfinement, CompareUnderAnUnrelatedExecRecordStillRefuses) {
    // A recorded parent that is not the writer's EXEC proves nothing about the reader's lanes.
    EXPECT_FALSE(predicate_defined_in_subset(cat({kMovV6, kV8FromV6}), kCmpV8, false));
}

TEST(FreshMaskExecConfinement, AccumulatorKeepsSharedConfinement) {
    EXPECT_TRUE(accumulator_meet(/*mac_under_writer_exec=*/true));
}

TEST(FreshMaskExecConfinement, AccumulatorMeetDoesNotRebaseToWriterExec) {
    EXPECT_FALSE(accumulator_meet(/*mac_under_writer_exec=*/false));
}

TEST(FreshMaskExecConfinement, FmacAccumulatorReadsStillRefuse) {
    // s2 is genuine, and so is s3: the decoder exposes a third source field for these opcodes
    // and scalar_source_words reads the s[2:3] pair, so s3 must not be the missing input.
    const auto setup = [](RegState& rs, SpirvCompute& b) {
        vgpr_genuine(rs, b, 1);
        vgpr_genuine(rs, b, 2);
        sgpr_genuine(rs, b, 2);
        sgpr_genuine(rs, b, 3);
    };
    EXPECT_FALSE(predicate_defined(kFmacV6, kCmpV6V6, true, setup));
    EXPECT_FALSE(predicate_defined(kPkFmacV6, kCmpV6V6, true, setup));
}

TEST(FreshMaskExecConfinement, SdwaAccumulatorCannotConfineAnUndefinedOldWord) {
    for (const auto words : {kSdwaMacV6, kSdwaFmacV6, kSdwaPkFmacV6}) {
        const auto ins = decode(words);
        ASSERT_EQ(ins.size(), 1u);
        ASSERT_EQ(ins[0].fmt, Rdna2Format::VOP2);
        ASSERT_TRUE(ins[0].has_sdwa);
        ASSERT_FALSE(ins[0].has_modifier);
        ASSERT_EQ(ins[0].sdwa_dst_sel, 6u);
        ASSERT_TRUE(vector_write_depends_on_old_word(ins[0], true));
        EXPECT_FALSE(predicate_defined(words, kCmpV6V6, true,
                                       [](RegState& rs, SpirvCompute& b) {
                                           vgpr_genuine(rs, b, 1);
                                           vgpr_genuine(rs, b, 2);
                                       }))
            << "opcode " << ins[0].opcode;
    }
}

TEST(FreshMaskExecConfinement, SdwaAccumulatorWithGenuineOldWordRemainsDefined) {
    for (const auto words : {kSdwaMacV6, kSdwaFmacV6, kSdwaPkFmacV6}) {
        EXPECT_TRUE(predicate_defined(words, kCmpV6V6, true, [](RegState& rs, SpirvCompute& b) {
            vgpr_genuine(rs, b, 1);
            vgpr_genuine(rs, b, 2);
            vgpr_genuine(rs, b, 6);
        }));
    }
}

TEST(FreshMaskExecConfinement, SdwaAccumulatorRetainsOnlyItsOriginalConfinement) {
    for (const auto words : {kSdwaMacV6, kSdwaFmacV6, kSdwaPkFmacV6}) {
        EXPECT_TRUE(accumulator_meet(true, words));
        EXPECT_FALSE(accumulator_meet(false, words));
    }
}

TEST(FreshMaskExecConfinement, SdwaFullWordMoveStillConfinesInactiveMissingLanes) {
    const auto setup = [](RegState& rs, SpirvCompute& b) { vgpr_genuine(rs, b, 1); };
    EXPECT_TRUE(predicate_defined(kSdwaFullMovV6, kCmpV6V6, true, setup));
    EXPECT_FALSE(predicate_defined(kSdwaFullMovV6, kCmpV6V6, false, setup));
}
