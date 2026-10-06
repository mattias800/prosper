// test_predicated_jump — the polarity of a packet-predicated sceAgcDcbJump under a BOOL64
// sceAgcDcbSetPredication window.
//
// The guest opens the window with sceAgcDcbSetPredication(dcb, 1, 3, 1, cond), records a Jump to a
// side segment, marks it with sceAgcSetPacketPredication, and closes the window with (dcb, 1, 0, 1,
// 0). PRED_OP 3 is the hardware BOOL64 predicate, which runs the guarded work only when the 64-bit
// condition is NON-ZERO. Dragon Quest VII Reimagined (PPSA17942) guards each of AGC's "Decompress
// Htile" helpers this way with a per-surface "needs it" word; the inverted rule ran the helper for a
// 1x1 scratch surface, whose 1x1 viewport then stayed bound for the rest of the frame.
//
// The polarity is the hardware PRED_BOOL, carried by one of the call's two flag arguments (both 1 on
// every observed call), not by the op. The builder keeps both, and the fold decides on them.
//
// These tests drive the real HLE builders, so the guest's argument order and the fold-time decision
// are tested together: a register the segment writes must land exactly when the window's rule says
// so, and an unpredicated jump in the same window must run either way.
#include "hle/dispatch/dispatch.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/cond_indirect_buffer.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <gtest/gtest.h>

#include <string>

#include <cstdint>

using namespace prosper;
namespace P = prosper::agc::Pm4;

namespace {

// The game's Dcb struct (must match hle_agc.cpp's AgcDcb byte-for-byte).
struct Dcb {
    uint32_t* bottom;
    uint32_t* top;
    uint32_t* cursor_up;
    uint32_t* cursor_down;
    void* callback;
    void* user_data;
    uint32_t reserved_dw;
    uint32_t pad;
};
using Hle6 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

struct Builders {
    Hle6 set_cx = nullptr, set_pred = nullptr, jump = nullptr, mark_pred = nullptr;
    bool ok() const { return set_cx && set_pred && jump && mark_pred; }
};

Builders builders() {
    register_builtin_hle();
    Builders b;
    b.set_cx = reinterpret_cast<Hle6>(Hle::lookup("LHFXRrlTPD8"));     // DcbSetCxRegisterDirect
    b.set_pred = reinterpret_cast<Hle6>(Hle::lookup("bbFueFP+J4k"));   // DcbSetPredication
    b.jump = reinterpret_cast<Hle6>(Hle::lookup("xSAR0LTcRKM"));       // DcbJump
    b.mark_pred = reinterpret_cast<Hle6>(Hle::lookup("w6Dj1VJt5qY"));  // SetPacketPredication
    return b;
}

struct Cb {
    uint32_t words[256]{};
    Dcb dcb{};
    Cb() {
        dcb.bottom = words;
        dcb.top = words + 256;
        dcb.cursor_up = words;
        dcb.cursor_down = words + 256;
    }
    uint64_t handle() { return (uint64_t)(uintptr_t)&dcb; }
    uint32_t used() const { return (uint32_t)(dcb.cursor_up - words); }
};

uint64_t addr_of(const void* p) { return (uint64_t)(uintptr_t)p; }
uint64_t reg(uint32_t offset, uint32_t value) { return ((uint64_t)value << 32) | offset; }

constexpr uint32_t kGuarded = P::PA_CL_VPORT_XSCALE;   // what an AGC helper segment rewrites
constexpr uint32_t kSegmentValue = 0x3f000000u;          // 0.5f: a 1x1 viewport's X scale
constexpr uint32_t kParentValue = 0x44f00000u;           // 1920.0f: the parent's 3840-wide one

// Run: parent writes kParentValue, then opens a BOOL64 window on `cond` and makes a predicated
// jump to a segment that writes kSegmentValue. Returns the register after the fold.
uint32_t fold_predicated_jump(const Builders& b, const volatile uint64_t& cond, bool predicated,
                              uint64_t a1 = 1, uint64_t a3 = 1) {
    Cb segment;
    b.set_cx(segment.handle(), reg(kGuarded, kSegmentValue), 0, 0, 0, 0);

    Cb parent;
    b.set_cx(parent.handle(), reg(kGuarded, kParentValue), 0, 0, 0, 0);
    b.set_pred(parent.handle(), a1, 3, a3, addr_of((const void*)&cond), 0);
    const uint64_t jump =
        b.jump(parent.handle(), 0, 0, addr_of(segment.words), segment.used(), 0);
    EXPECT_NE(jump, 0u);
    if (predicated) b.mark_pred(jump, 0, 0, 0, 0, 0);
    b.set_pred(parent.handle(), 1, 0, 1, 0, 0);

    gpu::GpuState st;
    gpu::run_command_buffer(parent.words, parent.used(), st);
    const auto it = st.cx.find(kGuarded);
    return it == st.cx.end() ? 0u : it->second;
}

// The observed call shape, sceAgcDcbSetPredication(dcb, 1, 3, 1, word): DRAW_VISIBLE.
TEST(PredicatedJump, Bool64RunsTheSegmentOnlyOnANonZeroCondition) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    volatile uint64_t cond = 0;
    EXPECT_EQ(fold_predicated_jump(b, cond, true), kParentValue)
        << "a zero BOOL64 condition discards the segment: its register write must not land";
    cond = 1;
    EXPECT_EQ(fold_predicated_jump(b, cond, true), kSegmentValue)
        << "a non-zero condition runs the segment";
    cond = 0x100000000ull;
    EXPECT_EQ(fold_predicated_jump(b, cond, true), kSegmentValue)
        << "the condition is 64-bit: a value only in the high half is non-zero";
}

TEST(PredicatedJump, AnUnmarkedJumpRunsWhateverTheCondition) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    volatile uint64_t cond = 0;
    EXPECT_EQ(fold_predicated_jump(b, cond, false), kSegmentValue)
        << "only a jump marked by SetPacketPredication participates in the window";
}

// Both flag slots clear: whichever one is PRED_BOOL, it says DRAW_NOT_VISIBLE, so the polarity
// inverts. CONFIDENCE: MED -- the hardware contract, with no title seen calling it this way.
TEST(PredicatedJump, ClearFlagsInvertThePolarity) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    volatile uint64_t cond = 0;
    EXPECT_EQ(fold_predicated_jump(b, cond, true, 0, 0), kSegmentValue)
        << "DRAW_NOT_VISIBLE runs the segment on a zero word";
    cond = 1;
    EXPECT_EQ(fold_predicated_jump(b, cond, true, 0, 0), kParentValue)
        << "...and discards it on a non-zero word";
}

// The builder keeps the op and both flags in the packet; nothing is dropped.
TEST(PredicatedJump, TheBuilderKeepsBothFlagArguments) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    Cb cb;
    ASSERT_NE(b.set_pred(cb.handle(), 0x12, 3, 0x34, 0x1000, 0), 0u);
    ASSERT_EQ(cb.used(), 4u) << "the packet size is unchanged";
    EXPECT_EQ(cb.words[3], gpu::pack_set_predication_control(3, 0x12, 0x34));
    EXPECT_EQ(cb.words[3], 0x80341203u);
}

TEST(PredicatedJump, ControlWordRules) {
    using gpu::pack_set_predication_control;
    using gpu::predicated_jump_skips;
    const uint32_t observed = pack_set_predication_control(3, 1, 1);
    EXPECT_TRUE(predicated_jump_skips(observed, 0));
    EXPECT_FALSE(predicated_jump_skips(observed, 1));
    EXPECT_FALSE(predicated_jump_skips(observed, ~0ull));
    const uint32_t not_visible = pack_set_predication_control(3, 0, 0);
    EXPECT_FALSE(predicated_jump_skips(not_visible, 0));
    EXPECT_TRUE(predicated_jump_skips(not_visible, 1));
    // The flag slots disagree: PRED_BOOL is not identified, so the observed polarity is kept.
    EXPECT_TRUE(predicated_jump_skips(pack_set_predication_control(3, 1, 0), 0));
    EXPECT_TRUE(predicated_jump_skips(pack_set_predication_control(3, 0, 1), 0));
    // A packet recorded before the flags were kept (bit 31 clear) is of the observed shape.
    EXPECT_TRUE(predicated_jump_skips(3u, 0));
    EXPECT_FALSE(predicated_jump_skips(3u, 1));
    // Ops with no title evidence keep the previous rule; pinned so a change to it is deliberate.
    EXPECT_FALSE(predicated_jump_skips(pack_set_predication_control(1, 1, 1), 0));
    EXPECT_TRUE(predicated_jump_skips(pack_set_predication_control(1, 1, 1), 1));
    EXPECT_TRUE(predicated_jump_skips(pack_set_predication_control(4, 1, 1), 1))
        << "BOOL32 is not given BOOL64's rule without title evidence";
}

// Every unobserved shape is reported once per distinct shape. A single process-wide flag would
// silence a second, different shape.
TEST(PredicatedJump, UnobservedShapesAreReportedOncePerShape) {
    using gpu::pack_set_predication_control;
    const uint32_t first = pack_set_predication_control(3, 7, 7);
    const uint32_t second = pack_set_predication_control(5, 1, 1);
    testing::internal::CaptureStderr();
    gpu::predicated_jump_skips(first, 0);
    gpu::predicated_jump_skips(first, 1);
    gpu::predicated_jump_skips(second, 0);
    gpu::predicated_jump_skips(pack_set_predication_control(3, 1, 1), 0);   // observed: silent
    const std::string out = testing::internal::GetCapturedStderr();
    const auto count = [&](const std::string& needle) {
        size_t n = 0;
        for (size_t at = out.find(needle); at != std::string::npos; at = out.find(needle, at + 1))
            ++n;
        return n;
    };
    EXPECT_EQ(count("op=3 a1=7 a3=7"), 1u) << out;
    EXPECT_EQ(count("op=5 a1=1 a3=1"), 1u) << out;
    EXPECT_EQ(count("op=3 a1=1 a3=1"), 0u) << out;
}

}  // namespace
