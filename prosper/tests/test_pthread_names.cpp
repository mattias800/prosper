// test_pthread_names -- guest-visible names must round-trip instead of returning success with an
// untouched output buffer. The contract is intentionally independent of host name limits/APIs.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static std::atomic<bool> worker_started{false};
static std::atomic<bool> worker_release{false};
// The body lives OUTSIDE the ABI shim, and on Windows that is load-bearing rather than tidy (#2142).
//
// WinLibs MinGW-w64 UCRT gcc 16.1.0 fails to ASSEMBLE a `sysv_abi` function that contains a call
// needing unwind info:
//
//     ccXXXX.s: Error: .seh_handlerdata used outside of .seh_proc block
//
// gcc emits `.seh_handlerdata` for the EH region while the non-native-ABI prologue opens no
// `.seh_proc`. That aborted `cmake --build all`, so every target after this one was skipped and a
// full local ctest needed `-k 0` -- while CI stayed green, because its Windows job uses MSYS2's gcc.
//
// Reduced to two lines by hand, outside this file, so the trigger is named rather than guessed:
//
//     #include <thread>
//     extern "C" __attribute__((sysv_abi)) void* f(void*) {
//         std::this_thread::yield(); return nullptr; }
//
// Fails at -O2, compiles at -O0. `sysv_abi` alone is NOT enough -- an empty one assembles fine --
// so the attribute is not the thing to remove.
//
// Two workarounds were tried and rejected before this one:
//   * -fno-asynchronous-unwind-tables on the target: assembles, then fails to LINK with
//     `undefined reference to __gxx_personality_v0`. It trades one toolchain error for another.
//   * -O0 on the target: works, but silently un-optimises a test to dodge a codegen bug.
// Moving the work out of the shim needs no flags at all and leaves the ABI boundary trivially
// correct: a shim that only forwards has nothing to unwind, which is what the ABI already requires.
//
// `noinline` is the part that must not be removed -- without it -O2 merges the body back in and the
// assembler error returns.
static __attribute__((noinline)) void* named_worker_body() {
    worker_started.store(true, std::memory_order_release);
    while (!worker_release.load(std::memory_order_acquire)) std::this_thread::yield();
    return (void*)0x386;
}
#ifdef _WIN32
extern "C" __attribute__((sysv_abi)) void* named_worker(void*) {
#else
static void* named_worker(void*) {
#endif
    return named_worker_body();
}

int main() {
    printf("== test_pthread_names ==\n");
    register_builtin_hle();

    HleFn self = Hle::lookup(nid_hash("scePthreadSelf"));
    HleFn getname = Hle::lookup(nid_hash("scePthreadGetname"));
    HleFn rename = Hle::lookup(nid_hash("scePthreadRename"));
    HleFn set_name = Hle::lookup(nid_hash("scePthreadSetName"));
    HleFn posix_get = Hle::lookup(nid_hash("pthread_getname_np"));
    HleFn posix_rename = Hle::lookup(nid_hash("pthread_rename_np"));
    HleFn posix_set_name = Hle::lookup(nid_hash("pthread_set_name_np"));
    HleFn create = Hle::lookup(nid_hash("scePthreadCreate"));
    HleFn join = Hle::lookup(nid_hash("scePthreadJoin"));
    CHECK(self && getname && rename && set_name && posix_get && posix_rename && posix_set_name &&
              create && join,
          "Sony and POSIX thread-name functions are registered");
    if (!self || !getname || !rename || !set_name || !posix_get || !posix_rename ||
        !posix_set_name || !create || !join)
        return 1;

    const uint64_t thread = self(0, 0, 0, 0, 0, 0);
    CHECK(thread != 0, "scePthreadSelf returns a usable thread handle");

    std::array<unsigned char, 40> sony_out{};
    sony_out.fill(0xa5);
    CHECK(getname(thread, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0) == 0,
          "unnamed current thread has a deterministic Sony result");
    CHECK(sony_out[0] == 0 && sony_out[31] == 0,
          "Sony getter writes and terminates its full 32-byte name field");
    CHECK(sony_out[32] == 0xa5 && sony_out[39] == 0xa5,
          "Sony getter does not write beyond the 32-byte ABI field");

    static const char long_name[] = "render-worker-name-that-is-longer-than-thirty-one";
    CHECK(rename(thread, (uint64_t)(uintptr_t)long_name, 0, 0, 0, 0) == 0,
          "scePthreadRename accepts the current thread");
    sony_out.fill(0xa5);
    CHECK(getname(thread, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0) == 0,
          "scePthreadGetname reads the renamed thread");
    CHECK(std::string((const char*)sony_out.data()) == std::string(long_name).substr(0, 31),
          "Sony name round-trip truncates deterministically to 31 bytes");
    CHECK(sony_out[31] == 0 && sony_out[32] == 0xa5,
          "truncated Sony name is terminated without overrunning the field");

    std::array<unsigned char, 40> posix_out{};
    posix_out.fill(0x6c);
    CHECK(posix_get(thread, (uint64_t)(uintptr_t)posix_out.data(), 0, 0, 0, 0) == 0,
          "pthread_getname_np uses the fixed-width two-argument guest ABI");
    CHECK(strcmp((const char*)posix_out.data(), (const char*)sony_out.data()) == 0,
          "Sony and POSIX getters share one guest-visible name");
    CHECK(posix_out[31] == 0 && posix_out[32] == 0x6c,
          "POSIX getter writes exactly the 32-byte guest name field");

    static const char renamed_again[] = "io-worker";
    CHECK(posix_rename(thread, (uint64_t)(uintptr_t)renamed_again, 0, 0, 0, 0) == 0 &&
          posix_set_name(thread, (uint64_t)(uintptr_t)renamed_again, 0, 0, 0, 0) == 0 &&
          set_name(thread, (uint64_t)(uintptr_t)renamed_again, 0, 0, 0, 0) == 0,
          "POSIX rename/set-name and scePthreadSetName share the rename contract");
    sony_out.fill(0xa5);
    getname(thread, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0);
    CHECK(strcmp((const char*)sony_out.data(), renamed_again) == 0,
          "set-name aliases update the Sony getter result");
    CHECK(rename(thread, 0, 0, 0, 0, 0) == 0,
          "a null Sony rename is an accepted no-op");
    sony_out.fill(0xa5);
    getname(thread, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0);
    CHECK(strcmp((const char*)sony_out.data(), renamed_again) == 0,
          "null rename preserves the previous name");

    // #2178: scePthreadRename's failure is ENCODED, on the family rule -- every Sony libkernel
    // spelling returns 0x80020000|errno -- and NOT on evidence specific to this function.
    // CONFIDENCE: MED, and the rationale block above SCE_PTHREAD_ALIAS(k_sce_pthread_rename) in
    // hle_kernel.cpp says why, including what the guest disassembly does and does not establish.
    //
    // Read this arm as guarding a DECISION, not as citing a measurement. It is here rather than
    // left to the sweep because the Sony/POSIX split is a REGISTRATION property: pointing the name
    // at the raw body again would restore the bare value silently, and every other arm in this file
    // would still pass. If the decision is later shown wrong, this arm is where it is changed --
    // which is the point of asserting it explicitly rather than leaving it implied.
    CHECK(rename(0, (uint64_t)(uintptr_t)renamed_again, 0, 0, 0, 0) == 0x80020016ull,
          "#2178: scePthreadRename on a null thread returns ENCODED EINVAL (0x80020016)");
    CHECK(posix_rename(0, (uint64_t)(uintptr_t)renamed_again, 0, 0, 0, 0) == 22,
          "#2178: ...while the POSIX spelling still returns the BARE errno, so the encoding went on "
          "the Sony wrapper rather than the shared body");

    // NOT swept, and this is deliberate: Getname's contract differs (its failures are lookup
    // failures on a call whose success path writes through a buffer), and no guest caller found so
    // far discriminates -- its only one does `test eax,eax`, where 3 and 0x80020003 are the same.
    // These two assertions stand until a caller that branches on the value turns up (#2178).
    CHECK(getname(0, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0) == 3,
          "null thread returns ESRCH");
    CHECK(getname(thread, 0, 0, 0, 0, 0) == 14,
          "null Sony output returns EFAULT");
    CHECK(getname(0x1122334455667788ull, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0) == 3,
          "unknown thread returns ESRCH instead of success with garbage output");

    // Creation publishes the requested name before returning. The worker stays alive while the
    // parent reads and renames it, then normal exit must retire the generation-tagged registry row.
    static const char created_name[] = "guest-created-render-worker";
    uint64_t worker = 0;
    worker_started.store(false, std::memory_order_relaxed);
    worker_release.store(false, std::memory_order_relaxed);
    CHECK(create((uint64_t)(uintptr_t)&worker, 0, (uint64_t)(uintptr_t)&named_worker, 0,
                 (uint64_t)(uintptr_t)created_name, 0) == 0 && worker != 0,
          "scePthreadCreate publishes a named worker");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!worker_started.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    CHECK(worker_started.load(std::memory_order_acquire), "named worker entered its guest function");
    sony_out.fill(0xa5);
    CHECK(getname(worker, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0) == 0 &&
          strcmp((const char*)sony_out.data(), created_name) == 0,
          "created thread's initial name is immediately guest-visible");
    static const char worker_renamed[] = "renamed-worker";
    CHECK(rename(worker, (uint64_t)(uintptr_t)worker_renamed, 0, 0, 0, 0) == 0,
          "a live created thread can be renamed");
    sony_out.fill(0xa5);
    getname(worker, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0);
    CHECK(strcmp((const char*)sony_out.data(), worker_renamed) == 0,
          "created thread rename round-trips");
    worker_release.store(true, std::memory_order_release);
    void* worker_result = nullptr;
    CHECK(join(worker, (uint64_t)(uintptr_t)&worker_result, 0, 0, 0, 0) == 0 &&
          worker_result == (void*)0x386,
          "named worker exits and joins normally");
    CHECK(getname(worker, (uint64_t)(uintptr_t)sony_out.data(), 0, 0, 0, 0) == 3,
          "exited thread name is retired before pthread_t reuse");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
