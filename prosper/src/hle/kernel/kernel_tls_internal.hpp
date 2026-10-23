#pragma once
#include "hle/dispatch/dispatch.hpp"

namespace prosper {
void tls_dtv_purge_current_thread();
size_t tls_dtv_thread_count();
void set_tls_modules(const TlsModuleDesc* descs, size_t count);
PROSPER_SYSV_ABI uint64_t k_tls_get_addr(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                                         uint64_t a4, uint64_t a5);
} // namespace prosper
