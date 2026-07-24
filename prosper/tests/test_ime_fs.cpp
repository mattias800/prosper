// test_ime_fs — sceImeUpdate drains the IME key queue and invokes the guest's registered keyboard
// event handler, which is GUEST code that reads guest thread-local state (its per-thread allocator
// etc.). But sceImeUpdate is itself an HLE handler, so the generated import stub has already swapped
// this thread to the HOST %fs before it ran (#1155). If the guest handler is called directly here it
// runs on host %fs, its thread-local reads hit host glibc garbage, and the first allocation it drives
// (the menu->gameplay transition on a key press) faults near null — Evergate's SIGSEGV at eboot+0xaf8431
// (#1286). The fix recovers the caller's guest %fs from the import-stub frame (via the entry-rsp
// trampoline) and restores it around each per-key dispatch.
//
// This test drives the REAL sceImeUpdate import path end to end — generated import stub -> entry-rsp
// trampoline -> guest-%fs recovery -> per-key dispatch — on an activated guest TCB, and asserts the
// guest handler ran on the GUEST %fs (not the host %fs) and that %fs is left restored afterward. It is
// the regression guard for #1286: revert the ime_deliver swap and the handler observes the host %fs.
// The %fs swap is Linux-only (Windows/macOS never swap hardware %fs at the import boundary), so the
// test is a no-op pass elsewhere.
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include "../src/hle/ime_input.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else       { std::printf("  [ok]   %s\n", m); } } while (0)

#if defined(__linux__) && !defined(__APPLE__)
// The "guest" IME event handler: record the %fs base it is invoked on and the decoded event. Runs on
// the guest TCB, so it must not call libc; it has no stack buffers, so no -fstack-protector canary is
// emitted (and the guest TCB carries the host canary at +0x28 regardless — guest_tls.cpp:135). SysV
// ABI matches ime_deliver's call cast on Linux (the default ABI).
extern "C" {
volatile uint64_t g_ime_observed_fs = 0;   // %fs base seen inside the guest handler
volatile uint32_t g_ime_observed_kind = 0;  // event[0x00]: 0x101 = key down, 0x102 = key up
volatile uint16_t g_ime_observed_hid = 0;   // event[0x08]: USB HID usage id
volatile int      g_ime_calls = 0;
}
static void test_ime_handler(uint64_t /*arg0*/, void* ev) {
    uint64_t fs = 0;
    __asm__ volatile("rdfsbase %0" : "=r"(fs));
    g_ime_observed_fs = fs;
    g_ime_observed_kind = *(const uint32_t*)((const uint8_t*)ev + 0x00);
    g_ime_observed_hid  = *(const uint16_t*)((const uint8_t*)ev + 0x08);
    ++g_ime_calls;
}

using GuestIme = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
#endif

int main() {
    std::printf("== test_ime_fs ==\n");

#if !defined(__linux__) || defined(__APPLE__)
    std::printf("  [skip] guest-%%fs swap at the import boundary is Linux-only; nothing to guard here\n");
    std::printf("== PASS ==\n");
    return 0;
#else
    // Enable the Linux guest initial-exec %fs path with a single empty TLS module (as the guest-fs
    // stub-args test does), then confirm it is active. The Linux suite already runs a guest-fs test,
    // so the environment supports user-mode FSGSBASE.
    setenv("PROSPER_GUEST_FS", "1", 1);
    TlsModuleDesc empty_module{};
    guest_tls_set_templates(&empty_module, 1);
    CHECK(guest_tls_enabled(), "Linux guest initial-exec %fs path enabled");

    // Register the built-in HLE table (which registers sceImeUpdate -> the s_ime_update_entry
    // trampoline) and generate a resolved import stub for it, exactly as the guest calls it.
    register_builtin_hle();
    const std::vector<ImportSlot> slots = { {"libSceIme", "-4GCfYdNF1s"} };  // sceImeUpdate
    dispatch_init(&slots, nullptr);
    std::string err;
    // Stubs MUST live in [0x600000000, 0x700000000): callback_guest_fs_from_entry_stack validates that
    // the return-into-stub address falls in that range before trusting the saved guest %fs, matching
    // where the real loader places import stubs. A base outside it makes recovery fail closed (return 0).
    CHECK(install_stubs(slots, 0x680000000ull, 96, &err),
          "generated the sceImeUpdate import stub");
    if (fails) {
        if (!err.empty()) std::printf("  install error: %s\n", err.c_str());
        std::printf("== FAIL: %d ==\n", fails);
        return 1;
    }
    auto ime_update = reinterpret_cast<GuestIme>(static_cast<uintptr_t>(stub_addr(0)));

    // Queue one Space (HID 0x2c) key press. Do this BEFORE activating the guest TCB: ime_push_key
    // takes a mutex (host libc), which must run on the host TCB.
    ime_push_key(0x2c, /*down=*/true);

    // Read the host %fs base, then activate this thread's guest TCB. From here until the restore below
    // the thread runs on the guest TCB; avoid libc. The generated stub swaps to host %fs for the HLE
    // body and the fix swaps back to the guest %fs only around the per-key handler dispatch.
    uint64_t host_fs = 0;
    __asm__ volatile("rdfsbase %0" : "=r"(host_fs));
    const uint64_t guest_fs = guest_tls_activate_thread();

    uint64_t handler = (uint64_t)(uintptr_t)&test_ime_handler;
    ime_update(handler, 0, 0, 0, 0, 0);
    // Immediately after the import call returns (still before any libc), the import boundary must have
    // restored the CALLER's guest %fs: the stub swapped guest->host on entry and back on exit. Reading
    // it here proves the whole swap is balanced, not merely that we can force host %fs afterward.
    uint64_t fs_at_return = 0;
    __asm__ volatile("rdfsbase %0" : "=r"(fs_at_return));

    // Back to the host TCB so the CHECKs (libc printf) run correctly.
    guest_fs_enter_host_for_signal();

    CHECK(guest_fs != 0, "activated a guest %fs base for the call");
    CHECK(guest_fs != host_fs, "guest %fs base differs from the host %fs base");
    CHECK(g_ime_calls == 1, "sceImeUpdate dispatched the queued key exactly once");
    CHECK(g_ime_observed_kind == 0x101u && g_ime_observed_hid == 0x2cu,
          "guest handler received the decoded key-down event (0x101, HID 0x2c)");
    // The core #1286 assertion: the guest handler ran on the caller's GUEST %fs, not the host %fs the
    // import stub swapped in. Reverting the ime_deliver swap makes g_ime_observed_fs == host_fs here.
    CHECK(g_ime_observed_fs == guest_fs,
          "guest key handler ran on the caller's GUEST %fs (not host %fs)");
    CHECK(g_ime_observed_fs != host_fs,
          "guest key handler did NOT run on the host %fs (the #1286 crash condition)");
    CHECK(fs_at_return == guest_fs,
          "import boundary left %fs on the caller's guest TCB after sceImeUpdate returned (swap balanced)");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
#endif
}
