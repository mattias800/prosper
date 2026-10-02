#pragma once
#include "host/abi/guest_varargs.hpp"
#include <cstdint>

namespace prosper_varargs_fixture {
extern prosper::abi::VarargClass g_cls[prosper::abi::kMaxFormatArgs];
extern unsigned g_ref_n;
extern uint64_t g_ref_u[prosper::abi::kMaxFormatArgs];
extern double g_ref_d[prosper::abi::kMaxFormatArgs];
void capture_and_pack(const char* fmt, const prosper::abi::SysvVaList& ap);
#if defined(_WIN32)
__attribute__((sysv_abi)) void guest_call(const char* fmt, ...);
__attribute__((sysv_abi)) void guest_call_reference(const char* fmt, ...);
#else
void guest_call(const char* fmt, ...);
void guest_call_reference(const char* fmt, ...);
#endif
} // namespace prosper_varargs_fixture
