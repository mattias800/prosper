#include "hle/kernel/kernel_tls_internal.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "host/platform/immortal.hpp"
#include <stddef.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace prosper {
// clang-format off: preserve the byte-identical moved definitions.
namespace {
// NOLINTNEXTLINE(bugprone-throwing-static-initialization): preserve the existing process-lifetime Immortal holder.
Immortal<std::vector<TlsModuleDesc>> g_tls_mods;
// NOLINTNEXTLINE(bugprone-throwing-static-initialization): preserve the existing process-lifetime Immortal holder.
Immortal<std::mutex> g_tls_mods_mx;   // guards g_tls_mods: a runtime sceKernelLoadStartModule re-runs
                            // set_tls_modules (assign, may realloc) while a worker is in k_tls_get_addr (#344).
// Per-thread DTV (thread -> module id -> TLS block). MUST NOT use a host `thread_local`: guest threads
// run under the GUEST %fs, and host thread_local storage is %fs-relative, so it ALIASES guest memory —
// reads come back as garbage (an unordered_map whose bucket_count reads 0 → `hash % 0` → SIGFPE).
// This is the same host↔guest %fs-aliasing landmine as the GfxDevice boot wall. Key by
// std::thread::id in a mutex-guarded global map instead. CONFIDENCE: HIGH (root-caused via gdb:
// SIGFPE in k_tls_get_addr with a corrupt thread_local map under a guest %fs).
// The map is PURGED on every HLE-controlled thread-exit path (thread_trampoline return +
// scePthreadExit/pthread_exit) via tls_dtv_purge_current_thread(): glibc recycles pthread ids, so a
// stale entry would hand a NEW thread the dead thread's dirty TLS blocks instead of the fresh
// zero/tdata-initialized state the ABI guarantees — and the blocks would leak per thread churn (#68).
// NOLINTNEXTLINE(bugprone-throwing-static-initialization): preserve the existing process-lifetime Immortal holder.
Immortal<std::mutex> g_dtv_mx;
// NOLINTNEXTLINE(bugprone-throwing-static-initialization): preserve the existing process-lifetime Immortal holder.
Immortal<std::unordered_map<std::thread::id, std::unordered_map<uint64_t, void*>>> g_dtv;
static_assert(std::is_trivially_destructible_v<decltype(g_tls_mods)> &&
              std::is_trivially_destructible_v<decltype(g_tls_mods_mx)> &&
              std::is_trivially_destructible_v<decltype(g_dtv_mx)> &&
              std::is_trivially_destructible_v<decltype(g_dtv)>,
              "#2613: the TLS module list and per-thread DTV must register no static destructor -- "
              "guest threads reach both while the process is exiting");
} // namespace
void tls_dtv_purge_current_thread() {
    std::unordered_map<uint64_t, void*> mine;
    { std::lock_guard<std::mutex> lk(*g_dtv_mx);
      auto it = g_dtv->find(std::this_thread::get_id());
      if (it == g_dtv->end()) return;   // main/host threads may never have touched __tls_get_addr
      mine = std::move(it->second);
      g_dtv->erase(it);
    }
    for (auto& kv : mine) free(kv.second);   // free the blocks outside the lock
}
size_t tls_dtv_thread_count() {   // test/diagnostic introspection: threads with live DTV entries
    std::lock_guard<std::mutex> lk(*g_dtv_mx);
    return g_dtv->size();
}
void set_tls_modules(const TlsModuleDesc* descs, size_t count) {
    { std::lock_guard<std::mutex> lk(*g_tls_mods_mx); g_tls_mods->assign(descs, descs + count); }
    // NOLINTNEXTLINE(concurrency-mt-unsafe): preserve the existing per-publication diagnostic read.
    if (getenv("PROSPER_TLSLOG"))
        for (size_t i = 0; i < count; i++)
            fprintf(stderr, "[tls] module %zu: init_va=0x%llx filesz=0x%llx memsz=0x%llx\n", i,
                    (unsigned long long)descs[i].init_va, (unsigned long long)descs[i].filesz,
                    (unsigned long long)descs[i].memsz);
}
PROSPER_SYSV_ABI uint64_t k_tls_get_addr(uint64_t a0, uint64_t a1, uint64_t a2,
                                       uint64_t a3, uint64_t a4, uint64_t a5) {   // __tls_get_addr(tls_index* ti): ti[0]=module id, ti[1]=offset-in-block
    const uint64_t* ti = (const uint64_t*)(uintptr_t)a0;
    if (!ti) return 0;
    uint64_t modid = ti[0], off = ti[1];
    std::thread::id tid = std::this_thread::get_id();   // portable per-OS-thread key (no %fs, no syscall)
    { std::lock_guard<std::mutex> lk(*g_dtv_mx);
      auto& dtv = (*g_dtv)[tid];
      auto it = dtv.find(modid);
      if (it != dtv.end()) return (uint64_t)(uintptr_t)it->second + off;
    }
    size_t memsz = 64, filesz = 0; uint64_t init_va = 0;
    { std::lock_guard<std::mutex> lk(*g_tls_mods_mx);   // #344: safe vs a concurrent set_tls_modules realloc
      if (modid < g_tls_mods->size()) {
          memsz  = (*g_tls_mods)[modid].memsz ? (*g_tls_mods)[modid].memsz : 64;
          filesz = (*g_tls_mods)[modid].filesz;
          init_va = (*g_tls_mods)[modid].init_va;
      } }
    void* blk = calloc(1, memsz);   // zero-init (tbss), then copy the tdata image
    if (init_va && filesz) memcpy(blk, (const void*)(uintptr_t)init_va, filesz);
    { std::lock_guard<std::mutex> lk(*g_dtv_mx); (*g_dtv)[tid][modid] = blk; }
    return (uint64_t)(uintptr_t)blk + off;
}
// clang-format on

} // namespace prosper
