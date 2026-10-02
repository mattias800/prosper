#pragma once
#include "hle/dispatch/dispatch.hpp"
#include "host/abi/guest_varargs.hpp"
#include <cstdarg>
#include <cstddef>
#include <cstdint>

namespace prosper {
// The producer owns only the guest variadic frame. All formatting, log capture and
// exception checkpoints remain in the host-ABI delegate, compiled by the main compiler.
PROSPER_GUEST_ABI uint64_t h_snprintf(void* buf, size_t n, const char* fmt, ...);
PROSPER_GUEST_ABI uint64_t h_sprintf(void* buf, const char* fmt, ...);
PROSPER_GUEST_ABI uint64_t h_printf(const char* fmt, ...);

#if defined(_WIN32)
enum class WinVariadicSink { Printf, Sprintf, Snprintf, Sscanf };
__attribute__((noinline))
uint64_t win_variadic_call(WinVariadicSink sink, void* buf, size_t n, const char* src,
                          const char* fmt, const abi::SysvVaList& ap,
                          bool run_checkpoint) noexcept;
#else
int capture_aware_vprintf(const char* format, va_list args);
#endif
} // namespace prosper
