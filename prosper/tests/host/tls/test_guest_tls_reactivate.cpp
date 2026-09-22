// test_guest_tls_reactivate (#3635) — a second guest_tls_activate_thread() on one host thread must
// keep that thread's guest TCB, not install a fresh zeroed one.
//
// The defect: on Linux the activation unconditionally mmapped a new block, zeroed it, re-copied the
// modules' tdata and wrfsbase'd it. So a second call silently discarded everything the guest had
// stored in its static TLS since the first (initial-exec variables, libc.prx's allocator state) and
// leaked the old block. macOS and Windows were already idempotent. Nothing crashes at the moment of
// the second call -- the damage surfaces later as a guest reading back zero from a thread-local it
// wrote -- which is why this is pinned by a test rather than left to a boot to notice.
//
// What the test does, with no guest: activate, write two recognisable values into the guest's
// static TLS (one over a tdata byte, one into tbss), swap back to the host %fs, activate again, and
// check the second call returned the SAME thread pointer and both values survived. Against the old
// code the second TP differs, the tbss byte reads 0 and the tdata byte is back to its template value.
//
// Nothing between an activation and the swap back may touch host TLS: with the guest TCB installed,
// errno and thread_locals resolve into guest memory. Every host-libc call is outside those windows.
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

#if defined(__linux__) && defined(__x86_64__)
static inline uint64_t rd_fs() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
static inline void     wr_fs(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }

// The module's PT_TLS initialisation image. Its first byte is what a fresh block re-copies, so a
// value written over it distinguishes "the same block" from "a new block initialised from this".
alignas(16) static const uint8_t kTdata[16] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
                                               0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xf0, 0x01};
#endif

int main() {
    printf("== test_guest_tls_reactivate ==\n");
#if !defined(__linux__) || !defined(__x86_64__)
    printf("  [skip] the wrfsbase activation path is x86-64 Linux only\n== PASS ==\n");
    return 0;
#else
    unsetenv("PROSPER_NO_GUEST_FS");   // the Linux path is on by default; make sure a caller's env
    unsetenv("PROSPER_TLSLOG");        // cannot turn it off, and keep the activation free of stderr

    prosper::TlsModuleDesc mods[2]{};  // index 0 is reserved
    mods[1].init_va = (uint64_t)(uintptr_t)kTdata;
    mods[1].filesz = sizeof kTdata;
    mods[1].memsz = 0x100;             // tdata (16 bytes) + tbss
    mods[1].align = 16;
    prosper::guest_tls_set_templates(mods, 2);
    CHECK(prosper::guest_tls_enabled(), "CONTROL: guest TLS is enabled for this thread");

    const uint64_t below = prosper::guest_tls_module_below(1);
    const uint64_t host_fs = rd_fs();
    constexpr uint8_t kTdataMark = 0xA7;   // overwrites kTdata[0]
    constexpr uint8_t kTbssMark = 0xC3;    // written into the zero-initialised tail
    constexpr size_t kTbssOff = 0x80;

    // ---- first activation; no host TLS until wr_fs(host_fs) ----
    const uint64_t tp1 = prosper::guest_tls_activate_thread();
    const uint64_t fs1 = rd_fs();
    uint8_t tdata_initial = 0, tbss_initial = 0xff;
    if (tp1) {
        volatile uint8_t* block = (volatile uint8_t*)(uintptr_t)(tp1 - below);
        tdata_initial = block[0];
        tbss_initial = block[kTbssOff];
        block[0] = kTdataMark;
        block[kTbssOff] = kTbssMark;
    }
    wr_fs(host_fs);

    CHECK(tp1 != 0 && tp1 != host_fs, "CONTROL: the first activation installs a distinct guest TCB");
    CHECK(fs1 == tp1, "CONTROL: ...and %fs points at it");
    CHECK(tdata_initial == kTdata[0] && tbss_initial == 0,
          "CONTROL: the fresh block carries the tdata template and a zeroed tbss, so the marks "
          "below are distinguishable from a re-initialised block");
    CHECK(prosper::guest_tls_own_tp() == tp1, "CONTROL: the thread records the TCB as its own");

    // ---- second activation on the SAME host thread ----
    const uint64_t tp2 = prosper::guest_tls_activate_thread();
    const uint64_t fs2 = rd_fs();
    uint8_t tdata_after = 0, tbss_after = 0;
    if (tp2) {
        volatile uint8_t* block = (volatile uint8_t*)(uintptr_t)(tp2 - below);
        tdata_after = block[0];
        tbss_after = block[kTbssOff];
    }
    wr_fs(host_fs);

    CHECK(tp2 == tp1, "#3635: a second activation returns the SAME thread pointer");
    CHECK(fs2 == tp1, "#3635: ...and re-applies it as %fs, as the Windows arm does");
    CHECK(tbss_after == kTbssMark,
          "#3635: a value the guest stored in tbss survives the second activation");
    CHECK(tdata_after == kTdataMark,
          "#3635: a value the guest stored over tdata is not re-initialised from the template");
    CHECK(prosper::guest_tls_own_tp() == tp1, "#3635: the thread's own TCB is unchanged");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
#endif
}
