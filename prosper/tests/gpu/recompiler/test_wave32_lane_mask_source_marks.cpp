// #5202: the dispatcher's source marks (spilled_vector_source_marks) charge a Wave32 lane-mask
// source ONE SGPR, as the structured walk's scalar_source_marks does. ISA 70648 6.2.3 reads "two
// consecutive SGPRs" for a mask in Wave64 only; in Wave32 a mask's upper half is ignored (2.1).
// Charged the pair, a carry-in or selector whose own word is defined was marked a fabricated relay
// whenever the word above it was never written, and indirect_pointer_descriptor_range's dispatcher
// refused a defined carry at pc507. test_ir_vector_carry_execution covers the structured walk.
// VCC_LO is the exception (scalar_source_mark_words), pinned below.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <set>

namespace {
using namespace prosper::gpu;

constexpr int kMask = 20;   // s20 holds the mask; s21 is never written
// SRC0 = 0 (inline), SRC1 = v32, SRC2 = s20.
constexpr uint32_t kSources = 0x80u | (0x120u << 9) | (static_cast<uint32_t>(kMask) << 18);
// v_add_co_ci_u32_e64 v44, vcc_lo, 0, v32, s20 and v_cndmask_b32_e64 v1, 0, v32, s20.
constexpr uint32_t kCarryIn[2] = {0xd5286a2cu, kSources};
constexpr uint32_t kCndmask[2] = {0xd5010001u, kSources};

SpilledVectorSourceMarks marks(const uint32_t (&words)[2], const std::set<int>& scalar_words,
                               uint32_t wave) {
    const Rdna2Inst in = rdna2_decode_one(words, 2);
    EXPECT_EQ(in.fmt, Rdna2Format::VOP3);
    return spilled_vector_source_marks(SpilledMaskHalves{}, in, {}, scalar_words, true, wave);
}

TEST(Wave32LaneMaskSourceMarks, DispatcherChargesOneSgpr) {
    for (const auto* words : {&kCarryIn, &kCndmask}) {
        SCOPED_TRACE((*words)[0]);
        const auto wave32 = marks(*words, {kMask}, 32);
        EXPECT_FALSE(wave32.vector_relay);
        EXPECT_FALSE(wave32.nonarbitrary);
        // Control: in Wave64 the mask IS the pair, so the unwritten s21 is a missing input.
        const auto wave64 = marks(*words, {kMask}, 64);
        EXPECT_TRUE(wave64.vector_relay);
        EXPECT_TRUE(wave64.nonarbitrary);
        // Control: the one word Wave32 does read is still checked.
        const auto absent = marks(*words, {}, 32);
        EXPECT_TRUE(absent.vector_relay);
        EXPECT_TRUE(absent.nonarbitrary);
    }
}

// VCC_LO keeps charging VCC_HI in Wave32 (scalar_source_mark_words): the never-written VCC_HI is
// what marks a read of the unprovided entry VCC. VCC_HI itself is one word.
TEST(Wave32LaneMaskSourceMarks, VccLoStillChargesVccHi) {
    constexpr uint32_t kVccLo = 0x80u | (0x120u << 9) | (106u << 18);
    constexpr uint32_t kVccHi = 0x80u | (0x120u << 9) | (107u << 18);
    for (const uint32_t op : {0xd5286a2cu, 0xd5010001u}) {
        SCOPED_TRACE(op);
        const uint32_t from_lo[2] = {op, kVccLo};
        const uint32_t from_hi[2] = {op, kVccHi};
        EXPECT_TRUE(marks(from_lo, {106}, 32).vector_relay);
        EXPECT_FALSE(marks(from_lo, {106, 107}, 32).vector_relay);
        EXPECT_FALSE(marks(from_hi, {107}, 32).vector_relay);
    }
}

}   // namespace
