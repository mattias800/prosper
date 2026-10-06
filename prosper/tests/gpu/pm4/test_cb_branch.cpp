// test_cb_branch — sceAgcCbBranch is a command-buffer builder, not a submit (#4540).
//
// The firmware (libSceAgc.sprx, file offset 0x5e80) appends one 14-dword type-3 packet, header
// 0xC00C3F00, and returns its address; it submits nothing. prosper used to fold the then-target at
// the moment the guest recorded the branch, so the target's draws ran against whatever pipeline the
// previous fold left bound, before the state the guest recorded ahead of the branch. These tests
// pin the builder's exact layout and the ordering contract that follows from it:
//   * recording a branch executes nothing;
//   * submitting the buffer that carries the branch executes the target at the branch's position,
//     so a register written before the branch is visible to the target's draw;
//   * the condition selects the then- or else-target, and an empty else-target is a no-op;
//   * GetSize and the three BranchPatch* patchers match the firmware.
#include "hle/dispatch/dispatch.hpp"
#include "gpu/pm4/command_processor.hpp"
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

extern "C" void prosper_agc_submit_stats(uint64_t* submits, uint64_t* draws);

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
struct Packet {
    uint32_t* addr;
    uint32_t dw_num;
    uint8_t pad[4];
};
using Hle6 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
using Hle12 = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                           uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);

constexpr uint32_t kShReg = 0x4c;   // an SH user-data register offset; any value works

struct Builders {
    Hle6 reset = nullptr, setsh = nullptr, draw = nullptr, submit = nullptr, get_size = nullptr;
    Hle6 patch_compare = nullptr, patch_then = nullptr, patch_else = nullptr;
    Hle12 branch = nullptr;
    bool ok() const {
        return reset && setsh && draw && submit && get_size && patch_compare && patch_then &&
               patch_else && branch;
    }
};

Builders builders() {
    register_builtin_hle();
    Builders b;
    b.reset = reinterpret_cast<Hle6>(Hle::lookup("TRO721eVt4g"));   // DcbResetQueue
    b.setsh = reinterpret_cast<Hle6>(Hle::lookup("pFLArOT53+w"));   // DcbSetShRegisterDirect
    b.draw = reinterpret_cast<Hle6>(Hle::lookup("Yw0jKSqop+E"));   // DcbDrawIndexAuto
    b.submit = reinterpret_cast<Hle6>(Hle::lookup("UglJIZjGssM"));   // DriverSubmitDcb
    b.branch = reinterpret_cast<Hle12>(Hle::lookup("w1KFAHVqpaU"));   // sceAgcCbBranch
    b.get_size = reinterpret_cast<Hle6>(Hle::lookup("uZW-mqsxkrM"));   // sceAgcCbBranchGetSize
    b.patch_compare = reinterpret_cast<Hle6>(Hle::lookup("GXBlM-ekzrI"));
    b.patch_then = reinterpret_cast<Hle6>(Hle::lookup("xb8VgcXQhvI"));
    b.patch_else = reinterpret_cast<Hle6>(Hle::lookup("QmfvaYpsOcI"));
    return b;
}

struct Cb {
    uint32_t words[512]{};
    Dcb dcb{};
    explicit Cb(uint32_t capacity = 512) {
        dcb.bottom = words;
        dcb.top = words + capacity;
        dcb.cursor_up = words;
        dcb.cursor_down = words + capacity;
    }
    uint64_t handle() { return (uint64_t)(uintptr_t)&dcb; }
    uint32_t used() const { return (uint32_t)(dcb.cursor_up - words); }
};

uint64_t addr_of(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

TEST(CbBranch, BuilderWritesTheFirmwarePacket) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    Cb cb;
    const uint64_t pkt = b.branch(cb.handle(), /*mode*/ 5, /*function*/ 0xb,
                                  /*compare*/ 0x1122334455667787ull, /*mask*/ 0xa1a2a3a4b1b2b3b4ull,
                                  /*reference*/ 0xc1c2c3c4d1d2d3d4ull, /*then flags*/ 6,
                                  /*then*/ 0x0000005566778803ull, /*then dw*/ 0x123456,
                                  /*else flags*/ 5, /*else*/ 0x000000998877660full,
                                  /*else dw*/ 0xfedcba);
    ASSERT_EQ(pkt, addr_of(cb.words)) << "returns the packet address";
    ASSERT_EQ(cb.used(), 14u) << "appends exactly 14 dwords";
    const uint32_t* w = cb.words;
    EXPECT_EQ(w[0], 0xC00C3F00u);
    EXPECT_EQ(w[1], (5u & 3u) | ((0xbu & 7u) << 8));
    EXPECT_EQ(w[2], 0x55667780u) << "compare address low, low 3 bits cleared";
    EXPECT_EQ(w[3], 0x11223344u);
    EXPECT_EQ(w[4], 0xb1b2b3b4u);
    EXPECT_EQ(w[5], 0xa1a2a3a4u);
    EXPECT_EQ(w[6], 0xd1d2d3d4u);
    EXPECT_EQ(w[7], 0xc1c2c3c4u);
    EXPECT_EQ(w[8], 0x66778800u) << "then address low, low 2 bits cleared";
    EXPECT_EQ(w[9], 0x00000055u);
    EXPECT_EQ(w[10], 0x23456u | (2u << 28)) << "then size masked to 20 bits, flags & 3 at 28";
    EXPECT_EQ(w[11], 0x8877660cu) << "else address low, low 2 bits cleared";
    EXPECT_EQ(w[12], 0x00000099u);
    EXPECT_EQ(w[13], 0xedcbau | (1u << 28)) << "else size masked to 20 bits, flags & 3 at 28";
    EXPECT_EQ(b.get_size(0, 0, 0, 0, 0, 0), 0x38u) << "firmware GetSize: 14 dwords, in bytes";
}

TEST(CbBranch, FullBufferWritesNothing) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    Cb cb(13);   // one dword short, and no overflow callback installed
    EXPECT_EQ(b.branch(cb.handle(), 1, 0, 0, 0, 0, 0, 0x1000, 4, 0, 0, 0), 0u);
    EXPECT_EQ(cb.used(), 0u);
}

TEST(CbBranch, PatchersRewriteOnlyTheirField) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    Cb cb;
    const uint64_t pkt = b.branch(cb.handle(), 1, 3, 0x1007, 0, 0, 2, 0x2003, 8, 1, 0x3003, 9);
    ASSERT_NE(pkt, 0u);
    uint32_t before[14];
    std::memcpy(before, cb.words, sizeof before);

    EXPECT_EQ(b.patch_compare(pkt, 0xaabbccdd11223347ull, 0, 0, 0, 0), 0u);
    EXPECT_EQ(cb.words[2], 0x11223340u | (before[2] & 7u));
    EXPECT_EQ(cb.words[3], 0xaabbccddu);
    EXPECT_EQ(cb.words[1], before[1]) << "mode/function untouched";

    EXPECT_EQ(b.patch_then(pkt, 0x0000000144445557ull, 0x77777, 0, 0, 0), 0u);
    EXPECT_EQ(cb.words[8], 0x44445554u | (before[8] & 3u));
    EXPECT_EQ(cb.words[9], 1u);
    EXPECT_EQ(cb.words[10], (before[10] & 0xfff00000u) | 0x77777u) << "then flags kept";

    EXPECT_EQ(b.patch_else(pkt, 0x0000000266667777ull, 0x1fffff, 0, 0, 0), 0u);
    EXPECT_EQ(cb.words[11], 0x66667774u | (before[11] & 3u));
    EXPECT_EQ(cb.words[12], 2u);
    EXPECT_EQ(cb.words[13], (before[13] & 0xfff00000u) | 0xfffffu) << "else flags kept";

    // A packet with another opcode is refused untouched (firmware returns 0x8a6c000c).
    uint32_t other[14] = {0xC00C1000u};
    uint32_t copy[14];
    std::memcpy(copy, other, sizeof copy);
    EXPECT_EQ(b.patch_compare(addr_of(other), 0x1000, 0, 0, 0, 0), 0x8a6c000cu);
    EXPECT_EQ(b.patch_then(addr_of(other), 0x1000, 4, 0, 0, 0), 0x8a6c000cu);
    EXPECT_EQ(b.patch_else(addr_of(other), 0x1000, 4, 0, 0, 0), 0x8a6c000cu);
    EXPECT_EQ(std::memcmp(other, copy, sizeof copy), 0);
}

// The regression for #4540: the branch must run where it sits in the carrying buffer.
TEST(CbBranch, TargetRunsWhenTheCarryingBufferRunsAndSeesStateRecordedBeforeIt) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    Cb target;
    b.draw(target.handle(), 3, 0, 0, 0, 0);
    const uint32_t target_dw = target.used();

    Cb parent;
    b.reset(parent.handle(), 0x3ff, 0, 0, 0, 0);
    b.setsh(parent.handle(), ((uint64_t)0xabcd1234u << 32) | kShReg, 0, 0, 0, 0);
    uint64_t s0 = 0, d0 = 0;
    prosper_agc_submit_stats(&s0, &d0);
    ASSERT_NE(
        b.branch(parent.handle(), 1, 0, 0, 0, 0, 0, addr_of(target.words), target_dw, 0, 0, 0), 0u);
    uint64_t s1 = 0, d1 = 0;
    prosper_agc_submit_stats(&s1, &d1);
    EXPECT_EQ(s1, s0) << "recording a branch submits nothing";
    EXPECT_EQ(d1, d0) << "recording a branch executes no draw";

    // The command processor reaches the branch after the SH write and runs the target there.
    gpu::GpuState st;
    gpu::run_command_buffer(parent.words, parent.used(), st);
    ASSERT_EQ(st.draws.size(), 1u) << "the target's draw executes inside the carrying buffer";
    ASSERT_TRUE(st.draws[0].state);
    const auto it = st.draws[0].state->sh.find(kShReg);
    ASSERT_NE(it, st.draws[0].state->sh.end());
    EXPECT_EQ(it->second, 0xabcd1234u) << "the target sees the register written before the branch";

    // And the real submit path executes it once.
    Packet pkt{parent.words, parent.used(), {}};
    EXPECT_EQ(b.submit(addr_of(&pkt), 0, 0, 0, 0, 0), 0u);
    uint64_t s2 = 0, d2 = 0;
    prosper_agc_submit_stats(&s2, &d2);
    EXPECT_EQ(s2, s1 + 1);
    EXPECT_EQ(d2, d1 + 1);
}

TEST(CbBranch, ConditionSelectsThenOrElse) {
    const Builders b = builders();
    ASSERT_TRUE(b.ok());
    Cb then_target, else_target;
    b.draw(then_target.handle(), 3, 0, 0, 0, 0);
    b.draw(else_target.handle(), 3, 0, 0, 0, 0);
    b.draw(else_target.handle(), 3, 0, 0, 0, 0);
    alignas(8) static uint64_t value = 0x0000000700000005ull;

    struct Arm {
        uint64_t function, mask, reference;
        bool with_else;
        size_t draws;
        const char* what;
    };
    const Arm arms[] = {
        {0, 0, 0, true, 1, "function 0 always takes the then-target"},
        {3, 0xffffffffull, 5, true, 1, "== on the masked value takes then"},
        {3, 0xffffffffull, 6, true, 2, "a false compare takes the else-target"},
        {4, ~0ull, 5, true, 1, "!= on the full 64-bit value takes then"},
        {1, 0xffffffffull, 5, true, 2, "< is false for equal values"},
        {3, 0xffffffffull, 6, false, 0, "a false compare with no else-target runs nothing"},
    };
    for (const Arm& a : arms) {
        Cb parent;
        ASSERT_NE(b.branch(parent.handle(), 1, a.function, addr_of(&value), a.mask, a.reference, 0,
                           addr_of(then_target.words), then_target.used(), 0,
                           a.with_else ? addr_of(else_target.words) : 0,
                           a.with_else ? else_target.used() : 0),
                  0u);
        gpu::GpuState st;
        gpu::run_command_buffer(parent.words, parent.used(), st);
        EXPECT_EQ(st.draws.size(), a.draws) << a.what;
    }
}

}   // namespace
