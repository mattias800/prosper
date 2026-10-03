// #4265: native Windows recovery must resume the correct frame, preserve host state and continue
// the real initializer loop after a foreign-frame fault. No game dump, GPU, or debugger is involved.
#include "host/fault/win_recovery.hpp"
#include "host/image/exec_image.hpp"
#include <windows.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <vector>

namespace {
struct RegisterWitness {
    uint64_t gpr[8];
    uint64_t xmm[10][2];
    uint32_t mxcsr;
    uint16_t fpcsr;
    uint64_t rsp;
};
static_assert(offsetof(RegisterWitness, xmm) == 64);
static_assert(offsetof(RegisterWitness, mxcsr) == 224);
static_assert(offsetof(RegisterWitness, fpcsr) == 228);
static_assert(offsetof(RegisterWitness, rsp) == 232);
extern "C" int prosper_test_win_recovery_witness(prosper::host::WinRecoveryState*,
                                                 RegisterWitness*);

class GuestCode {
public:
    GuestCode()
        : bytes_(static_cast<uint8_t*>(
              VirtualAlloc(nullptr, 4096, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE))) {}
    ~GuestCode() {
        if (bytes_) VirtualFree(bytes_, 0, MEM_RELEASE);
    }
    GuestCode(const GuestCode&) = delete;
    GuestCode& operator=(const GuestCode&) = delete;
    uint64_t address() const { return reinterpret_cast<uint64_t>(bytes_); }
    void append(std::initializer_list<uint8_t> bytes) {
        for (uint8_t byte : bytes) bytes_[used_++] = byte;
    }
    void immediate(uint64_t value) {
        memcpy(bytes_ + used_, &value, sizeof(value));
        used_ += sizeof(value);
    }
    void finish() { FlushInstructionCache(GetCurrentProcess(), bytes_, used_); }

private:
    uint8_t* bytes_;
    size_t used_ = 0;
};
}   // namespace

TEST(WinRecovery, RestoresPhysicalNonvolatileState) {
    prosper::host::WinRecoveryState state{};
    RegisterWitness witness{};
    ASSERT_EQ(prosper_test_win_recovery_witness(&state, &witness), 1);
    for (uint64_t n = 0; n < 8; ++n)
        EXPECT_EQ(witness.gpr[n], 0x1111000011110001ull + n) << "nonvolatile GPR " << n;
    for (uint64_t n = 0; n < 10; ++n) {
        const uint64_t lane = n + 6;
        const uint64_t expected = lane | (lane << 32);
        EXPECT_EQ(witness.xmm[n][0], expected) << "XMM" << lane << " low word";
        EXPECT_EQ(witness.xmm[n][1], expected) << "XMM" << lane << " high word";
    }
    EXPECT_EQ(witness.mxcsr, 0x3f80u);
    EXPECT_EQ(witness.fpcsr, 0x077fu);
    EXPECT_EQ(witness.rsp, state.rsp);
}

TEST(WinRecovery, ResumesCompiledFrameLocals) {
    prosper::host::WinRecoveryState state{};
    volatile uint64_t marker = 0x87654321abcdef01ull;
    volatile int entered = 0;
    if (prosper::host::prosper_win_recovery_save(&state) == 0) {
        entered = 1;
        prosper::host::prosper_win_recovery_restore(&state);
    }
    EXPECT_EQ(entered, 1);
    EXPECT_EQ(marker, 0x87654321abcdef01ull);
}

TEST(WinRecovery, FaultedInitializerContinuesThroughValidInitializers) {
    prosper::install_trap_handler();
    GuestCode success, fault;
    ASSERT_NE(success.address(), 0u);
    ASSERT_NE(fault.address(), 0u);
    volatile uint64_t calls = 0;
    // SysV leaf: increment an owned counter, return normally. A successful call is a positive
    // control for the actual initializer trampoline, not another synthetic fault response.
    success.append({0x48, 0xb8});
    success.immediate(reinterpret_cast<uint64_t>(&calls));
    success.append({0x48, 0xff, 0x00, 0xc3});
    // Destroy a common callee-save frame register before an actual #UD. Recovery must not rely on
    // the foreign guest restoring it, nor on the guest/trampoline epilogues executing.
    fault.append({0x31, 0xed, 0x0f, 0x0b});
    success.finish();
    fault.finish();
    const std::vector<uint64_t> inits{success.address(), fault.address(), success.address(), 0,
                                      success.address()};
    EXPECT_EQ(prosper::run_guest_inits(inits), 3u);
    EXPECT_EQ(calls, 3u) << "the loop must advance after both #UD and a null target";
    EXPECT_EQ(prosper::recovery_thunk_call_rsp_mod16(), 0);
}

TEST(WinRecovery, EntryFaultRestoresNativeStackBounds) {
    prosper::install_trap_handler();
    GuestCode fault;
    ASSERT_NE(fault.address(), 0u);
    fault.append({0x31, 0xed, 0x0f, 0x0b});
    fault.finish();
    ULONG_PTR before_lo = 0, before_hi = 0, after_lo = 0, after_hi = 0;
    GetCurrentThreadStackLimits(&before_lo, &before_hi);
    prosper::LoadedImage image;
    image.entry = fault.address();
    const auto result = prosper::run_entry(image);
    GetCurrentThreadStackLimits(&after_lo, &after_hi);
    EXPECT_EQ(result.kind, 3);
    EXPECT_EQ(result.fault_rip, fault.address() + 2);
    EXPECT_EQ(before_lo, after_lo);
    EXPECT_EQ(before_hi, after_hi);
}
