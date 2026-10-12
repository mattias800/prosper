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
        rs.vector_wave_size = 64;
        rs.vector_full_exec = false;
        rs.max_vgpr = 7;
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
