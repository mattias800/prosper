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

#include <cstdint>
#include <functional>
#include <vector>

using namespace prosper::gpu;

namespace {
// v_mov_b32 v6, <literal>: a complete full-word write.
const std::vector<uint32_t> kMovV6 = {0x7e0c02ffu, 0x00001234u};
// VOP2 op 0x1b: v8 = <literal> op v6 (complete full-word write, reads v6).
const std::vector<uint32_t> kV8FromV6 = {0x36100cffu, 0x00003fffu};
// v_mov_b32_sdwa v6, 0 with WORD_0 / UNUSED_PRESERVE: keeps the upper destination bits.
const std::vector<uint32_t> kSdwaV6 = {0x7e0c02f9u, 0x00851480u};
// v_writelane_b32 v6, s2, lane 0: replaces one selected lane.
const std::vector<uint32_t> kWritelaneV6 = {0xd7610006u, 0x00010002u};
// v_movreld_b32 with VDST v6 (indexed through M0), inline-zero source.
const std::vector<uint32_t> kMovrelV6 = {0x7e0c8480u};
// v_fmac_f32_e64 (0x12b) and v_pk_fmac_f16_e64 (0x13c): the VOP3 forms of VOP2 0x2b/0x3c, which
// accumulate the old destination like 0x11f (the e64 form of 0x1f).
// SRC2 encodes s2: the decoder exposes a third source field for these opcodes, so the test makes
// it genuine, leaving the old-destination accumulator as the only missing input.
const std::vector<uint32_t> kFmacV6 = {0xd12b0006u, 0x000a0501u};
const std::vector<uint32_t> kPkFmacV6 = {0xd13c0006u, 0x000a0501u};
// VOPC compare, src0 = v6, vsrc1 = v6.
const std::vector<uint32_t> kCmpV6V6 = {0x7c200d06u};
// VOPC compare, src0 = v6, vsrc1 = v8.
const std::vector<uint32_t> kCmpV8 = {0x7c201106u};
// VOPC compare, src0 = s20 (absent scalar), vsrc1 = v6.
const std::vector<uint32_t> kCmpS20V6 = {0x7c200c14u};

std::vector<Rdna2Inst> decode(const std::vector<uint32_t>& words) {
    std::vector<Rdna2Inst> ins;
    rdna2_walk(words.data(), words.size(), ins);
    return ins;
}

std::vector<uint32_t> cat(std::initializer_list<std::vector<uint32_t>> parts) {
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
        b.diagnostic = {RecompileDiagnosticStage::Fragment, 0xa4924001ull};
        rs.vector_wave_size = 64;
        rs.vector_full_exec = false;
        rs.max_vgpr = 7;
        rs.vcc = b.bfalse();
    }
};

// Writes `writes` under EXEC G, then asks whether `reader` (a predicate) is defined under the
// same G, or under another EXEC value when `same_exec` is false.
bool predicate_defined(const std::vector<uint32_t>& writes, const std::vector<uint32_t>& reader,
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

// v_cmpx_eq_u32 v1, v2 (VOPC e32, op 0xd2): EXEC' = EXEC & (v1 == v2).
const std::vector<uint32_t> kCmpxV1V2 = {0x7da40501u};

// Emits a real v_cmpx through emit_alu under the current EXEC, so the EXEC parent relationship
// comes from the production path. False when the emission itself refuses.
bool emit_cmpx(Harness& h) {
    const auto cmpx = decode(kCmpxV1V2);
    if (cmpx.size() != 1) return false;
    const auto before = snapshot_saved_b64_masks(h.rs, cmpx[0]);
    bool ok = true;
    if (!emit_alu(h.b, h.rs, cmpx[0], ok, true) || !ok) return false;
    record_scalar_write(h.rs, cmpx[0], allows_compute_scalar_vcc_bridge(h.b), before);
    return true;
}

// The reader runs under the EXEC a real emitted v_cmpx produced. When the cmpx runs under the
// writer's EXEC the predicate is defined; running it under an unrelated EXEC still refuses. The
// test asserts the recorded parent, so a missing provenance assignment in emit_alu fails it.
bool predicate_defined_after_cmpx(const std::vector<uint32_t>& writes,
                                  const std::vector<uint32_t>& reader,
                                  bool cmpx_under_writer_exec) {
    Harness h;
    const uint32_t g = h.b.btrue();
    h.rs.exec = g;
    h.rs.exec_narrowed = true;
    const auto write_ins = decode(writes);
    const auto read_ins = decode(reader);
    const auto cmpx = decode(kCmpxV1V2);
    if (write_ins.empty() || read_ins.size() != 1 || cmpx.size() != 1) return false;
    h.rs.vreg[1] = h.b.uconst(11);
    h.rs.vreg[2] = h.b.uconst(11);
    std::vector<Rdna2Inst> all = write_ins;
    all.push_back(cmpx[0]);
    all.push_back(read_ins[0]);
    seed_vector_word_definedness(h.rs, all, false, 64);
    for (const auto& in : write_ins)
        record_vector_word_marks(h.rs, in, scalar_source_marks(h.rs, in));
    if (!cmpx_under_writer_exec) h.rs.exec = h.b.btrue();   // an unrelated EXEC value
    const uint32_t parent = h.rs.exec;
    if (!emit_cmpx(h)) return false;
    if (h.rs.exec_subset_of.count(h.rs.exec) == 0) return false;
    if (h.rs.exec_subset_of.at(h.rs.exec) != parent) return false;
    h.rs.exec_narrowed = true;
    return fresh_mask_sources_defined(h.b, h.rs, read_ins[0]);
}
void sgpr_genuine(RegState& rs, SpirvCompute& b, int r) {
    rs.sreg[r] = b.uconst(5);
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
    const auto cx = decode(kCmpxV1V2);
    ASSERT_EQ(cx.size(), 1u);
    EXPECT_EQ(cx[0].fmt, Rdna2Format::VOPC);
    EXPECT_EQ(cx[0].opcode, 0xd2u);
    EXPECT_EQ(cx[0].src[0].kind, OperandKind::VGPR);
    EXPECT_EQ(cx[0].src[0].value, 1);
    EXPECT_EQ(cx[0].src[1].kind, OperandKind::VGPR);
    EXPECT_EQ(cx[0].src[1].value, 2);
    for (const auto& words : {kFmacV6, kPkFmacV6}) {
        const auto fmac = decode(words);
        ASSERT_EQ(fmac.size(), 1u);
        EXPECT_EQ(fmac[0].fmt, Rdna2Format::VOP3);
        EXPECT_EQ(fmac[0].dst.value, 6);
    }
    EXPECT_EQ(decode(kFmacV6)[0].opcode, 0x12bu);
    EXPECT_EQ(decode(kPkFmacV6)[0].opcode, 0x13cu);
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
    EXPECT_TRUE(predicate_defined_after_cmpx(cat({kMovV6, kV8FromV6}), kCmpV8, true));
}

TEST(FreshMaskExecConfinement, CompareUnderAnUnrelatedExecRecordStillRefuses) {
    // A cmpx under an unrelated EXEC proves nothing about the writer's lanes.
    EXPECT_FALSE(predicate_defined_after_cmpx(cat({kMovV6, kV8FromV6}), kCmpV8, false));
}

void vgpr_genuine(RegState& rs, SpirvCompute& b, int r) {
    rs.vreg[r] = b.uconst(7);
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