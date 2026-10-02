// Only the existing guest variadic capture frames. No TLS, statics, formatting or checkpoints.
#include "hle/libc/libc_variadic_capture.hpp"
#include <cstring>
#include <cstdio>
#include <cstdarg>
namespace prosper {
#if defined(_WIN32)
using abi::SysvVaList;
#endif
// The printf family: REAL C variadic functions, and PROSPER_GUEST_ABI so the import stub is the same
// bare tail-jump on EVERY platform. That is what puts the guest's own System V frame — integer
// registers, xmm registers, AL and the overflow area alike — in front of the compiler's variadic
// prologue, which is the only thing that can capture an argument list the format string decides at
// run time. On Linux and macOS this is exactly what already happened and the tag expands to nothing;
// on Windows it replaces a fixed integer shuffle that could not deliver a floating-point argument at
// all and displaced every argument behind one (#3246).
//
// Guest pointers are identity-mapped (guest VA == host VA), so the buffer, the format string and any
// %s argument are usable host pointers directly — no P() translation, because nothing re-places the
// guest's values on the way in.
//
// Each body is deliberately trivial, and that is a REQUIREMENT rather than a style: a guest-ABI frame
// cannot carry SEH unwind data on Windows, so it may own no object with a destructor and must call
// nothing that could be inlined into it carrying one (PROSPER_GUEST_ABI in dispatch.hpp). Capture,
// delegate, return.
//
// THE ONE CAVEAT, and it is on LINUX rather than on the new path. When guest %fs TLS is on — which is
// the DEFAULT (`guest_tls.cpp` opts out only through PROSPER_NO_GUEST_FS) — the import stub is not a
// tail-jump at all but the swap stub, which interposes a CALL so it can restore the guest's %fs
// afterwards. To keep the callee's stack arguments where SysV puts them it re-pushes the guest's
// spilled words... and it re-pushes exactly FOUR of them (`emit_swap_stub`, exec_image_linux.cpp:
// "push original arg10/9/8/7"). Immediately behind those sits the stub's saved r11, then the guest's
// return address, then a shifted duplicate of the same four words.
//
// So a variadic call whose overflow area holds a FIFTH word reads the stub's saved r11 as that
// argument, and everything after it is wrong too. Reaching it needs more than six integer-class or
// more than eight floating-point variadic arguments — rare for a format call, and unchanged by this
// commit, which is why it is recorded rather than fixed here (#3271). It is NOT a Windows problem: there the
// guest-ABI stub really is a bare tail-jump with no interposed frame, so the overflow area arrives
// whole. macOS never emits the swap stub either (`stub_swap_mode()` returns false there).
//
// CONFIDENCE: HIGH on Linux/macOS for register and xmm arguments, which is the overwhelmingly common
// case and the long-standing behaviour with the tag empty; MED for Linux stack-spilled arguments past
// the fourth, per the paragraph above. HIGH on the Windows mechanism — the shape below was assembled
// by MinGW GCC 16.1.1 at every optimization level and executed under wine, delivering a
// twelve-argument mixed call including System V overflow-area integers and five xmm doubles, and the
// whole suite runs on the Windows MinGW CI host (#3246). What is NOT verified is a live guest calling
// it on a Windows host; nobody here has one.
PROSPER_GUEST_ABI uint64_t h_snprintf(void* buf, size_t n, const char* fmt, ...) {
#if defined(_WIN32)
    __builtin_sysv_va_list ap; __builtin_sysv_va_start(ap, fmt);
    prosper::abi::SysvVaList captured; memcpy(&captured, &ap, sizeof captured);
    __builtin_sysv_va_end(ap);
    return win_variadic_call(WinVariadicSink::Snprintf, buf, n, nullptr, fmt, captured,
                             /*run_checkpoint=*/true);
#else
    va_list ap; va_start(ap, fmt); int r = vsnprintf((char*)buf, n, fmt, ap); va_end(ap);
    return (uint64_t)(int64_t)r;
#endif
}
PROSPER_GUEST_ABI uint64_t h_sprintf(void* buf, const char* fmt, ...) {
#if defined(_WIN32)
    __builtin_sysv_va_list ap; __builtin_sysv_va_start(ap, fmt);
    prosper::abi::SysvVaList captured; memcpy(&captured, &ap, sizeof captured);
    __builtin_sysv_va_end(ap);
    return win_variadic_call(WinVariadicSink::Sprintf, buf, 0, nullptr, fmt, captured,
                             /*run_checkpoint=*/true);
#else
    va_list ap; va_start(ap, fmt); int r = vsprintf((char*)buf, fmt, ap); va_end(ap);
    return (uint64_t)(int64_t)r;
#endif
}
PROSPER_GUEST_ABI uint64_t h_printf(const char* fmt, ...) {
#if defined(_WIN32)
    __builtin_sysv_va_list ap; __builtin_sysv_va_start(ap, fmt);
    prosper::abi::SysvVaList captured; memcpy(&captured, &ap, sizeof captured);
    __builtin_sysv_va_end(ap);
    return win_variadic_call(WinVariadicSink::Printf, nullptr, 0, nullptr, fmt, captured,
                             /*run_checkpoint=*/true);
#else
    va_list ap; va_start(ap, fmt); int r = capture_aware_vprintf(fmt, ap); va_end(ap);
    return (uint64_t)(int64_t)r;
#endif
}
} // namespace prosper
