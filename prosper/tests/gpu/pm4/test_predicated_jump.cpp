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
// These tests drive the real HLE builders, so the guest's argument order and the fold-time decision
// are tested together: a register the segment writes must land exactly when the condition is
// non-zero, and an unpredicated jump in the same window must run either way.
#include "hle/dispatch/dispatch.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "gpu/pm4/cond_indirect_buffer.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include <gtest/gtest.h>

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
uint32_t fold_predicated_jump(const Builders& b, const volatile uint64_t& cond, bool predicated) {
    Cb segment;
    b.set_cx(segment.handle(), reg(kGuarded, kSegmentValue), 0, 0, 0, 0);

    Cb parent;
    b.set_cx(parent.handle(), reg(kGuarded, kParentValue), 0, 0, 0, 0);
    b.set_pred(parent.handle(), 1, 3, 1, addr_of((const void*)&cond), 0);
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

TEST(PredicatedJump, PolarityRuleByOp) {
    EXPECT_TRUE(gpu::predicated_jump_skips(3, 0));
    EXPECT_FALSE(gpu::predicated_jump_skips(3, 1));
    EXPECT_FALSE(gpu::predicated_jump_skips(3, ~0ull));
    // Ops with no title evidence keep the previous rule (reported once); pinned so a change to it
    // is deliberate.
    EXPECT_FALSE(gpu::predicated_jump_skips(1, 0));
    EXPECT_TRUE(gpu::predicated_jump_skips(1, 1));
}

}  // namespace
