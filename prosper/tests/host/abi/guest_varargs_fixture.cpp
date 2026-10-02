// test_guest_varargs — a guest System V variadic call, re-expressed for a Microsoft x64 host (#3246).
//
// #2955 made the import bridge signature-driven, and #3246 is the one shape a signature cannot
// describe: a real C variadic, whose argument list is whatever the format string says at run time.
// Three independent things have to hold, and only the last needs a Windows host:
//
//   (1) READING the guest's list. `sysv_va_arg` walks the System V register save area and overflow
//       area. Checked against a REAL `va_start` frame produced by the compiler — a positive instance
//       built outside the machinery under test, including arguments that spill past both files.
//   (2) CLASSIFYING the arguments. `plan_format` decides, per conversion, which System V file the
//       argument came from. A table, asserted directly, on any platform.
//   (3) WRITING the Microsoft list. The packed slots are consumed by the COMPILER's own Microsoft
//       va_arg (`__builtin_ms_va_list`), which is available on Linux too — so the layout claim is
//       checked against the toolchain rather than against this file's own belief about it. On
//       Windows the real CRT reads the same bytes, and the formatted string is asserted as well.
//
// Plus the two structural facts the wiring depends on: a guest-ABI handler's stub is a bare
// tail-jump, and the printf family really is registered that way.
//
// What NONE of this checks is a live guest calling printf on a Windows host. Nothing here pretends to.
#include "host/abi/guest_varargs.hpp"
#include "host/abi/sysv_ms_bridge.hpp"
#include "guest_varargs_fixture.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>

using namespace prosper;
using namespace prosper::abi;

#if defined(__x86_64__) || defined(_M_X64)
#define PROSPER_TEST_X86_64 1
#else
#define PROSPER_TEST_X86_64 0
#endif

#if PROSPER_TEST_X86_64
// The producer side must build a SYSTEM V frame. On Linux that is what an ordinary variadic function
// already does; on Windows it needs the same tag the real handlers carry.
#if defined(_WIN32)
#define TEST_GUEST_ABI      __attribute__((sysv_abi))
#define TEST_GUEST_VA_LIST  __builtin_sysv_va_list
#define TEST_GUEST_VA_START __builtin_sysv_va_start
#define TEST_GUEST_VA_END   __builtin_sysv_va_end
#define TEST_MS_VA_LIST     va_list
#else
#define TEST_GUEST_ABI
#define TEST_GUEST_VA_LIST  va_list
#define TEST_GUEST_VA_START va_start
#define TEST_GUEST_VA_END   va_end
#define TEST_MS_VA_LIST     __builtin_ms_va_list
#endif
#endif

namespace prosper_varargs_fixture {
#if PROSPER_TEST_X86_64
// The guest side: an ordinary C variadic call, placed by System V. What it captures is exactly what
// h_printf captures.
TEST_GUEST_ABI void guest_call(const char* fmt, ...) {
    TEST_GUEST_VA_LIST ap;
    TEST_GUEST_VA_START(ap, fmt);
    SysvVaList captured;
    memcpy(&captured, &ap, sizeof captured);
    TEST_GUEST_VA_END(ap);
    capture_and_pack(fmt, captured);
}

// The same frame, read by the COMPILER's own System V va_arg instead of by sysv_va_arg — the
// independently produced positive instance that stops arm (1) being checked against itself.
TEST_GUEST_ABI void guest_call_reference(const char* fmt, ...) {
    TEST_GUEST_VA_LIST ap;
    TEST_GUEST_VA_START(ap, fmt);
    for (unsigned i = 0; i < g_ref_n; ++i) {
        if (g_cls[i] == VarargClass::Sse) g_ref_d[i] = __builtin_va_arg(ap, double);
        else                              g_ref_u[i] = __builtin_va_arg(ap, uint64_t);
    }
    TEST_GUEST_VA_END(ap);
}

#endif  // PROSPER_TEST_X86_64

} // namespace prosper_varargs_fixture
