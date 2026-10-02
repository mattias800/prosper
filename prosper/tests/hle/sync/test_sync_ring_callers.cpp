// PROSPER_SYNC_RING_CALLERS: every sync-ring event can carry the GUEST call site that reached it.
//
// WHY. A total guest deadlock (every thread parked on a condition variable) leaves the ring with
// "thread X signalled object Y" and no way to say WHICH guest function did it. Without the call site
// nobody can go and read the code that was supposed to wake the stuck thread. This adds it, and the
// dump names it as <module>+0x<offset> like every other guest address in prosper's diagnostics.
//
// The ring is sized from the environment at static-initialisation time, so this test is registered
// with ENVIRONMENT "PROSPER_SYNC_RING=256;PROSPER_SYNC_RING_CALLERS=1".
//
// THE FIXTURE. A real guest frame is just "a return address in a guest module's range with a `call`
// instruction ending right before it". The test places a four-instruction stub at a guest code
// address and calls it, and the stub calls back into a host thunk that emits the sync event:
//
//     sub rsp,0x28 ; mov rax,<thunk> ; call rax ; add rsp,0x28 ; ret
//
// so the expected caller is exactly stub+16 (4 + 10 + 2 bytes), which is computed here rather than
// read back from the code under test.
//
// WHAT EACH ARM KILLS (mutations that must turn this test red):
//   M1  sync_trace stops capturing callers (or capture is left off)         -> the guest-event arm
//   M2  the dump omits the caller text                                      -> the guest-event arm
//   M3  capture reports a stale/dead stack value as a caller (no `call`     -> the CONTROL arm: an
//       validation)                                                            event from plain host
//                                                                              code must carry none
// The control event is emitted BEFORE the guest stub runs, deliberately: the stub leaves a dead
// return address in stack slots below main()'s live frames, and an event recorded afterwards from
// the same depth could re-validate it.
#include "hle/sync/sync_futex.hpp"
#include "host/image/boot_program.hpp"
#include "host/image/exec_image.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <windows.h>
#include <pthread.h>
#endif

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("  [FAIL] %s\n", msg); fails++; } \
                              else std::printf("  [ok]   %s\n", msg); } while (0)

#ifdef _WIN32
namespace {
pthread_cond_t g_cond = PTHREAD_COND_INITIALIZER;

// Plain host function (Microsoft x64) reached from the guest-address stub.
void host_thunk() { prosper::interruptible_cond_signal(&g_cond); }

std::string read_all(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return {};
    std::string out;
    char buffer[4096];
    size_t got;
    while ((got = std::fread(buffer, 1, sizeof buffer, f)) > 0) out.append(buffer, got);
    std::fclose(f);
    return out;
}

// The text after "kind=signal " ... up to the end of that event line, for the Nth signal event.
std::string nth_signal_line(const std::string& dump, int n) {
    size_t at = 0;
    for (int i = 0; i <= n; ++i) {
        at = dump.find("kind=signal", at);
        if (at == std::string::npos) return {};
        if (i < n) at += 1;
    }
    const size_t end = dump.find('\n', at);
    return dump.substr(at, end == std::string::npos ? std::string::npos : end - at);
}
}   // namespace
#endif

int main() {
#ifndef _WIN32
    std::printf("  [skip] PROSPER_SYNC_RING is a Windows-only diagnostic\n");
    std::printf("== PASS ==\n");
    return 0;
#else
    const char* path = "test_sync_ring_callers.out";
    std::remove(path);

    // CONTROL: an event recorded from plain host code, before any guest stub has run.
    prosper::interruptible_cond_signal(&g_cond);

    // Place the stub at a guest code address.
    const uint64_t stub_address = prosper::BOOT_EBOOT + 0x00100000ull;
    void* page = VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(stub_address)), 0x1000,
                              MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE);
    CHECK(page == reinterpret_cast<void*>(static_cast<uintptr_t>(stub_address)),
          "the stub page is mapped at a guest code address");
    if (page != reinterpret_cast<void*>(static_cast<uintptr_t>(stub_address))) return 1;

    uint8_t code[] = {
        0x48, 0x83, 0xEC, 0x28,                                            // sub rsp,0x28
        0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0,                                // mov rax, imm64
        0xFF, 0xD0,                                                        // call rax
        0x48, 0x83, 0xC4, 0x28,                                            // add rsp,0x28
        0xC3,                                                              // ret
    };
    const uint64_t thunk = reinterpret_cast<uint64_t>(&host_thunk);
    std::memcpy(code + 6, &thunk, sizeof thunk);
    std::memcpy(page, code, sizeof code);
    FlushInstructionCache(GetCurrentProcess(), page, sizeof code);
    const uint64_t expected_return = stub_address + 16;   // 4 + 10 + 2: just past `call rax`

    // GUEST EVENT: the same signal, now reached through a guest call site.
    reinterpret_cast<void (*)()>(page)();

    prosper::dump_guest_sync_trace(path);
    const std::string dump = read_all(path);
    CHECK(!dump.empty(), "the ring dump wrote something");

    const std::string control = nth_signal_line(dump, 0);
    const std::string guest = nth_signal_line(dump, 1);
    CHECK(!control.empty() && !guest.empty(), "both signal events were recorded");

    const std::string expected = std::string("caller=") + prosper::describe_code_address(expected_return);
    CHECK(guest.find(expected) != std::string::npos,
          "an event reached through a guest call site names that call site (M1, M2)");
    CHECK(control.find("caller=") == std::string::npos,
          "an event from plain host code carries no caller, not a stale stack value (M3)");

    std::remove(path);
    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
#endif
}
