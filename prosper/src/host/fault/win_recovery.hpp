// Windows guest faults cross foreign frames: recover host state without an SEH unwind (#4265).
#pragma once

#ifdef _WIN32
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace prosper::host {

// Private to the two assembly entry points below, not a compiler/CRT jmp_buf. Offsets are checked
// against their instructions. Capturing the actual caller RBP avoids compiler SEH frame biases.
struct alignas(16) WinRecoveryState {
    uint64_t rbx, rbp, rdi, rsi, rsp, r12, r13, r14, r15, rip;
    uint32_t mxcsr;
    uint16_t fpcsr;
    uint8_t reserved[10];
    uint64_t xmm[10][2]; // Microsoft-x64 nonvolatile low 128 bits of XMM6--XMM15.
};

static_assert(std::is_standard_layout_v<WinRecoveryState>);
static_assert(alignof(WinRecoveryState) == 16 && sizeof(WinRecoveryState) == 256);
static_assert(offsetof(WinRecoveryState, rbx) == 0);
static_assert(offsetof(WinRecoveryState, rbp) == 8);
static_assert(offsetof(WinRecoveryState, rdi) == 16);
static_assert(offsetof(WinRecoveryState, rsi) == 24);
static_assert(offsetof(WinRecoveryState, rsp) == 32);
static_assert(offsetof(WinRecoveryState, r12) == 40);
static_assert(offsetof(WinRecoveryState, r13) == 48);
static_assert(offsetof(WinRecoveryState, r14) == 56);
static_assert(offsetof(WinRecoveryState, r15) == 64);
static_assert(offsetof(WinRecoveryState, rip) == 72);
static_assert(offsetof(WinRecoveryState, mxcsr) == 80);
static_assert(offsetof(WinRecoveryState, fpcsr) == 84);
static_assert(offsetof(WinRecoveryState, xmm) == 96);

// save returns 0 normally and 1 after restore. Its calling frame and state must remain alive.
// As with setjmp, locals modified between save and restore must be volatile if read afterward.
// No C++ cleanup is run across the discarded foreign frames. Never use this for C++ exceptions.
extern "C" __attribute__((returns_twice)) int prosper_win_recovery_save(WinRecoveryState* state);
extern "C" __attribute__((noreturn)) void
prosper_win_recovery_restore(const WinRecoveryState* state);

} // namespace prosper::host
#endif
