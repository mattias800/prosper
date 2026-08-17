// test_trap_arbitration — PROSPER_HWBP must not swallow a SIGTRAP that PROSPER_BP owns (#1932).
//
// prosper's Linux fault_handler is ONE SIGTRAP consumer chain shared by several in-process debug
// instruments (the same kind of instrumentation gdb or Wine's relay debugging provides, pointed at
// this emulator's own process). The hardware-breakpoint branch runs first and used to accept ANY
// SIGTRAP while an HWBP/HWWATCH fd was live, so with PROSPER_BP armed as well the int3 hit was
// consumed by the WRONG state machine: it fabricated an "[hwbp] rip=<bp addr>+1" record, BP's
// restore / back-up-RIP / single-step / re-arm sequence never ran, the guest's original instruction
// was skipped, and the title died on the corruption that followed.
//
// Two levels of coverage here, and the second is the one that would have caught it:
//
//   1. A decision table over prosper::host::hardware_breakpoint_claims_sigtrap() — every producer
//      state a SIGTRAP can arrive in, including the per-TID HWBP step ownership that must survive.
//   2. A LIVE end-to-end arm: a real perf hardware execute breakpoint on one hand-assembled stub, a
//      real int3 patched into another, and a SIGTRAP handler whose dispatch mirrors production's
//      order (HWBP branch first, BP branch second). It runs TWICE — once through the shipped
//      predicate, and once through hardware_breakpoint_instrument_live(), which is the verbatim
//      pre-#1932 condition `(g_hwbp_on && g_hwbp_fd >= 0) || g_hwwatch_fd >= 0`. The second run is
//      the mutation arm and MUST reproduce the defect, so a predicate that cannot tell the two
//      instruments apart fails this test in both directions rather than passing silently.
//
// The BP stub is built so "the original instruction was skipped" is directly observable instead of
// inferred: the patched byte is a one-byte `cdq`, so skipping it lands exactly on the next
// instruction (no misdecode, nothing unsafe) and simply leaves edx holding 0 instead of the
// sign-extended 0xffffffff the stub then stores. Correct routing therefore stores 0xffffffff;
// the defect stores 0.
//
// The perf arm needs no elevated privilege — a per-task breakpoint on the caller's own process is
// permitted at kernel.perf_event_paranoid=2. If it is refused anyway the test says so with the
// exact errno and continues with a non-perf fd so the routing arms (which only require "an HWBP fd
// is live", exactly as production does) still run; it never turns an unavailable instrument into a
// silent pass.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "../src/host/trap_arbitration.hpp"

#if defined(__linux__)

#include <cerrno>
#include <fcntl.h>
#include <linux/hw_breakpoint.h>
#include <linux/perf_event.h>
#include <signal.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <ucontext.h>
#include <unistd.h>

#include "../src/host/posix_shim.hpp"   // PROSPER_GREGS + the REG_* indices

namespace {

int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", (message)); ++failures; } \
} while (0)

using prosper::host::HardwareBreakpointTrapState;
using prosper::host::SoftwareBreakpointTrapState;
using prosper::host::hardware_breakpoint_claims_sigtrap;
using prosper::host::hardware_breakpoint_instrument_live;

constexpr long long kTrapFlag = 0x100ll;

// ---------------------------------------------------------------------------------------------
// 1. Decision table.
// ---------------------------------------------------------------------------------------------

void test_decision_table() {
    constexpr uint64_t kBpAddr = 0x400100;
    const SoftwareBreakpointTrapState bp_off{};
    const SoftwareBreakpointTrapState bp_armed{ .armed = true, .stepping = false, .addr = kBpAddr };
    const SoftwareBreakpointTrapState bp_stepping{ .armed = true, .stepping = true, .addr = kBpAddr };

    const HardwareBreakpointTrapState hw_off{};
    const HardwareBreakpointTrapState hw_bp{ .armed = true, .breakpoint_fd = 7 };
    const HardwareBreakpointTrapState hw_watch{ .watchpoint_fd = 9 };
    const HardwareBreakpointTrapState hw_bp_stepping{
        .armed = true, .breakpoint_fd = 7, .stepping_this_thread = true };
    // g_hwbp_on set but perf_event_open failed: production leaves g_hwbp_fd at -1 and the branch
    // must stay out of the chain entirely.
    const HardwareBreakpointTrapState hw_arm_failed{ .armed = true, .breakpoint_fd = -1 };

    // No instrument live -> the branch never claims anything, whatever BP is doing.
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_off, bp_off, 0x4000, false),
          "table: no HWBP/HWWATCH fd must not claim a trap");
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_arm_failed, bp_off, 0x4000, false),
          "table: g_hwbp_on with a failed arm (fd<0) must not claim a trap");
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_arm_failed, bp_armed, kBpAddr + 1, false),
          "table: a failed HWBP arm must leave the int3 hit to PROSPER_BP");

    // HWBP live, BP not armed -> every trap is the hardware instrument's.
    CHECK(hardware_breakpoint_claims_sigtrap(hw_bp, bp_off, 0x4000, false),
          "table: HWBP live and BP disarmed must claim a perf hit");
    CHECK(hardware_breakpoint_claims_sigtrap(hw_bp, bp_off, kBpAddr + 1, false),
          "table: with BP disarmed even bp_addr+1 belongs to HWBP");

    // THE DEFECT, both halves.
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_bp, bp_armed, kBpAddr + 1, false),
          "table: the int3 hit (rip == bp_addr+1) must be left to PROSPER_BP");
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_bp, bp_stepping, kBpAddr + 5, true),
          "table: BP's own single-step completion must be left to PROSPER_BP");
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_watch, bp_armed, kBpAddr + 1, false),
          "table: a live HWWATCH fd alone must also leave the int3 hit to PROSPER_BP");
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_watch, bp_stepping, kBpAddr + 5, true),
          "table: a live HWWATCH fd alone must also leave BP's step completion to PROSPER_BP");

    // A one-byte original instruction: the step completion lands on the SAME rip as the hit. Both
    // are BP's, and BP's own branch separates them by g_bp_stepping.
    CHECK(!hardware_breakpoint_claims_sigtrap(hw_bp, bp_stepping, kBpAddr + 1, true),
          "table: a one-byte original's step completion is still PROSPER_BP's");

    // Per-TID HWBP ownership must survive the new exclusion.
    CHECK(hardware_breakpoint_claims_sigtrap(hw_bp_stepping, bp_stepping, kBpAddr + 5, true),
          "table: this thread's own HWBP step completion stays with HWBP");
    CHECK(hardware_breakpoint_claims_sigtrap(hw_bp_stepping, bp_armed, kBpAddr + 1, true),
          "table: a thread mid-HWBP-step keeps its completion even at bp_addr+1");
    // Another thread being mid-BP-step must NOT divert a genuine asynchronous perf hit here: that
    // context has TF clear, because this thread never set it.
    CHECK(hardware_breakpoint_claims_sigtrap(hw_bp, bp_stepping, 0x4000, false),
          "table: a perf hit on another thread must not be stolen by a global g_bp_stepping");
    // The HWBP hit address itself is never bp_addr+1 (a perf execute breakpoint traps BEFORE its
    // instruction), so an ordinary hit while BP is armed still belongs to HWBP.
    CHECK(hardware_breakpoint_claims_sigtrap(hw_bp, bp_armed, kBpAddr, false),
          "table: a perf hit at the bp address itself is still HWBP's");
}

// ---------------------------------------------------------------------------------------------
// 2. Live end-to-end arm.
// ---------------------------------------------------------------------------------------------

// stub_hw: `mov eax, 7 ; ret` — the hardware execute breakpoint's target.
const uint8_t kStubHw[] = { 0xb8, 0x07, 0x00, 0x00, 0x00, 0xc3 };

// stub_bp(uint32_t* out):
//   +0  31 d2              xor  edx, edx
//   +2  b8 ff ff ff ff     mov  eax, 0xffffffff
//   +7  99                 cdq                     <- the byte replaced by 0xCC
//   +8  89 17              mov  [rdi], edx
//   +10 c3                 ret
// `cdq` sign-extends eax into edx, so a correctly restored + stepped instruction stores
// 0xffffffff. It is one byte, so if the trap is misrouted and RIP is left one past the 0xCC the
// CPU lands exactly on the `mov` with edx still 0 — the skip is observable and cannot misdecode.
const uint8_t kStubBp[] = { 0x31, 0xd2, 0xb8, 0xff, 0xff, 0xff, 0xff, 0x99, 0x89, 0x17, 0xc3 };
constexpr size_t kStubBpPatchOffset = 7;
constexpr size_t kStubHwOffset = 0x00;
constexpr size_t kStubBpOffset = 0x40;

// --- instrument state, mirroring the file-scope globals in exec_image_linux.cpp ---
bool                  g_hw_armed = false;
int                   g_hw_fd = -1;
bool                  g_hw_fd_is_perf = false;
int                   g_hwwatch_fd = -1;
volatile sig_atomic_t g_hw_stepping = 0;

bool                  g_bp_armed = false;
volatile sig_atomic_t g_bp_stepping = 0;
uint64_t              g_bp_addr = 0;
uint8_t               g_bp_orig = 0;

// Which state machine consumed what.
volatile sig_atomic_t g_hwbp_events = 0;
volatile sig_atomic_t g_bp_events = 0;
volatile sig_atomic_t g_unclaimed_traps = 0;
volatile uint64_t     g_last_hwbp_rip = 0;

// The mutation arm: when set, the HWBP branch uses the pre-#1932 condition.
volatile sig_atomic_t g_defect_mode = 0;

uint64_t g_page = 0;

// Mirrors bp_write_byte() in exec_image_linux.cpp: reopen the code page for write, patch, flush.
void patch_byte(uint64_t addr, uint8_t value) {
    const uint64_t page = addr & ~(uint64_t)0xfff;
    mprotect((void*)page, 0x2000, PROT_READ | PROT_WRITE | PROT_EXEC);
    *(volatile uint8_t*)addr = value;
    __builtin___clear_cache((char*)addr, (char*)addr + 1);
}

// The production dispatch order, reduced to the two branches this issue is about. Everything else
// in fault_handler either runs before both (the dmem write-trace step and the mb3 head watch, both
// of which match on their own per-TID/fd state) or after both.
void trap_handler(int sig, siginfo_t* si, void* uctx) {
    (void)si;
    auto* uc = (ucontext_t*)uctx;
    const uint64_t rip = (uint64_t)PROSPER_GREGS(uc)[REG_RIP];
    const bool trap_flag = (PROSPER_GREGS(uc)[REG_EFL] & kTrapFlag) != 0;

    const HardwareBreakpointTrapState hw{
        .armed = g_hw_armed, .breakpoint_fd = g_hw_fd, .watchpoint_fd = g_hwwatch_fd,
        .stepping_this_thread = g_hw_stepping != 0 };
    const SoftwareBreakpointTrapState bp{
        .armed = g_bp_armed, .stepping = g_bp_stepping != 0, .addr = g_bp_addr };

    // Consumer 1 — PROSPER_HWBP / PROSPER_HWWATCH.
    const bool hwbp_claims = g_defect_mode
        ? hardware_breakpoint_instrument_live(hw)                        // the pre-#1932 condition
        : hardware_breakpoint_claims_sigtrap(hw, bp, rip, trap_flag);    // the shipped one
    if (sig == SIGTRAP && hwbp_claims) {
        if (g_hw_stepping) {                       // step completion: re-enable and clear TF
            if (g_hw_fd_is_perf) ioctl(g_hw_fd, PERF_EVENT_IOC_ENABLE, 0);
            PROSPER_GREGS(uc)[REG_EFL] &= ~kTrapFlag;
            g_hw_stepping = 0;
            return;
        }
        g_hwbp_events = g_hwbp_events + 1;         // the "[hwbp] #N rip=..." record
        g_last_hwbp_rip = rip;
        if (g_hw_fd_is_perf) ioctl(g_hw_fd, PERF_EVENT_IOC_DISABLE, 0);
        PROSPER_GREGS(uc)[REG_EFL] |= kTrapFlag;   // step off the breakpoint address
        g_hw_stepping = 1;
        return;
    }

    // Consumer 2 — PROSPER_BP.
    if (sig == SIGTRAP && g_bp_armed) {
        if (g_bp_stepping) {                       // (a) mid-step: re-insert the 0xCC, clear TF
            patch_byte(g_bp_addr, 0xCC);
            PROSPER_GREGS(uc)[REG_EFL] &= ~kTrapFlag;
            g_bp_stepping = 0;
            return;
        }
        if (rip == g_bp_addr + 1) {                // (b) hit: restore, back up RIP, step, re-arm
            g_bp_events = g_bp_events + 1;         // the "[bp] #N ..." record
            patch_byte(g_bp_addr, g_bp_orig);
            PROSPER_GREGS(uc)[REG_RIP] = (greg_t)g_bp_addr;
            PROSPER_GREGS(uc)[REG_EFL] |= kTrapFlag;
            g_bp_stepping = 1;
            return;
        }
    }

    // Nothing owns it. Returning would spin forever on the same instruction, so fail loudly.
    g_unclaimed_traps = g_unclaimed_traps + 1;
    static const char message[] = "FAIL: a SIGTRAP was claimed by no consumer\n";
    (void)!write(2, message, sizeof message - 1);
    _exit(97);
}

long perf_execute_breakpoint_open(uint64_t addr) {
    // Byte-for-byte the shape perf_bp_open() uses in exec_image_linux.cpp (x86 requires
    // bp_len == sizeof(long) for an execute breakpoint).
    struct perf_event_attr pe;
    std::memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_BREAKPOINT;
    pe.size = sizeof pe;
    pe.bp_type = HW_BREAKPOINT_X;
    pe.bp_addr = addr;
    pe.bp_len = sizeof(long);
    pe.sample_period = 1;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0UL);
}

void reset_counters() {
    g_hwbp_events = 0;
    g_bp_events = 0;
    g_unclaimed_traps = 0;
    g_last_hwbp_rip = 0;
    g_hw_stepping = 0;
    g_bp_stepping = 0;
}

// Runs the full two-instrument sequence. `defect_mode` selects the pre-#1932 HWBP condition.
// Returns false if the arm itself could not be set up (already reported).
bool run_live_arm(bool defect_mode) {
    const uint64_t stub_hw = g_page + kStubHwOffset;
    const uint64_t stub_bp = g_page + kStubBpOffset;
    const char* arm = defect_mode ? "defect arm (pre-#1932 condition)" : "fixed arm (shipped predicate)";

    reset_counters();
    g_defect_mode = defect_mode ? 1 : 0;

    // --- arm the hardware execute breakpoint on stub_hw -------------------------------------
    long fd = perf_execute_breakpoint_open(stub_hw);
    if (fd >= 0) {
        g_hw_fd = (int)fd;
        g_hw_fd_is_perf = true;
        fcntl(g_hw_fd, F_SETFL, O_ASYNC);
        fcntl(g_hw_fd, F_SETSIG, SIGTRAP);
        struct f_owner_ex ow;
        ow.type = F_OWNER_TID;
        ow.pid = (pid_t)syscall(SYS_gettid);
        fcntl(g_hw_fd, F_SETOWN_EX, &ow);
        ioctl(g_hw_fd, PERF_EVENT_IOC_ENABLE, 0);
    } else {
        const int e = errno;
        std::fprintf(stderr,
                     "[hwbp] perf_event_open refused (errno=%d %s) — the hardware arm of the %s is "
                     "not exercised; the routing assertions below still run against a live fd, "
                     "which is exactly the production precondition\n",
                     e, std::strerror(e), arm);
        g_hw_fd = open("/dev/null", O_RDONLY);
        g_hw_fd_is_perf = false;
        if (g_hw_fd < 0) {
            std::fprintf(stderr, "FAIL: could not open a stand-in fd for the %s\n", arm);
            ++failures;
            return false;
        }
    }
    g_hw_armed = true;

    // --- arm the software int3 on stub_bp ---------------------------------------------------
    g_bp_addr = stub_bp + kStubBpPatchOffset;
    g_bp_orig = *(volatile uint8_t*)g_bp_addr;
    patch_byte(g_bp_addr, 0xCC);
    g_bp_armed = true;

    // --- 1. execute the hardware breakpoint's target ----------------------------------------
    auto call_hw = (int (*)())(uintptr_t)stub_hw;
    const int hw_result = call_hw();
    CHECK(hw_result == 7, "live: the hardware-breakpoint stub must still return 7");
    if (g_hw_fd_is_perf) {
        CHECK(g_hwbp_events == 1, "live: executing the hw stub must produce exactly one hwbp event");
        CHECK(g_last_hwbp_rip == stub_hw,
              "live: the hwbp event must be reported at the breakpoint address itself");
    }
    const int hwbp_events_before_bp = (int)g_hwbp_events;

    // --- 2. execute the software breakpoint's target ----------------------------------------
    auto call_bp = (void (*)(uint32_t*))(uintptr_t)stub_bp;
    uint32_t out = 0x5a5a5a5au;
    call_bp(&out);

    if (!defect_mode) {
        CHECK(g_bp_events == 1, "live: the int3 hit must be consumed by the PROSPER_BP branch");
        CHECK(g_hwbp_events == hwbp_events_before_bp,
              "live: the int3 hit must not manufacture an extra hwbp record");
        CHECK(out == 0xffffffffu,
              "live: PROSPER_BP must re-execute the original instruction it replaced");
        CHECK(*(volatile uint8_t*)g_bp_addr == 0xCC,
              "live: PROSPER_BP must re-arm its int3 after stepping off it");
        CHECK(g_bp_stepping == 0, "live: PROSPER_BP's step sequence must have completed");

        // --- 3. the breakpoint must still fire, and the hardware instrument must still work ---
        uint32_t out2 = 0x5a5a5a5au;
        call_bp(&out2);
        CHECK(g_bp_events == 2, "live: the re-armed int3 must fire again");
        CHECK(out2 == 0xffffffffu, "live: the second pass must also execute the original instruction");
        const int hwbp_events_before_second_hw = (int)g_hwbp_events;
        CHECK(call_hw() == 7, "live: the hw stub must still return 7 after the bp sequence");
        if (g_hw_fd_is_perf)
            CHECK(g_hwbp_events == hwbp_events_before_second_hw + 1,
                  "live: the hardware breakpoint must still fire after PROSPER_BP ran");
    } else {
        // The mutation arm. Without the exclusion the HWBP branch consumes the int3 trap: it
        // fabricates a second [hwbp] record at bp_addr+1, PROSPER_BP never sees the hit, and the
        // guest's original instruction is skipped. If ANY of these now hold the fixed behaviour,
        // the predicate under test is not the thing making the fixed arm pass.
        CHECK(g_bp_events == 0,
              "mutation: without the exclusion the PROSPER_BP branch must never see the int3 hit");
        CHECK(g_hwbp_events == hwbp_events_before_bp + 1,
              "mutation: without the exclusion the HWBP branch must fabricate an extra record");
        CHECK(g_last_hwbp_rip == g_bp_addr + 1,
              "mutation: the fabricated record must be reported one byte past the patched address");
        CHECK(out == 0u,
              "mutation: without the exclusion the original instruction must be skipped");
        std::fprintf(stderr,
                     "[mutation] reproduced #1932: bp events=%d, hwbp events=%d (last rip=+1 past "
                     "the int3), stored value=0x%08x instead of 0xffffffff\n",
                     (int)g_bp_events, (int)g_hwbp_events, out);
    }
    CHECK(g_unclaimed_traps == 0, "live: every SIGTRAP must be claimed by exactly one consumer");

    // --- disarm -----------------------------------------------------------------------------
    g_bp_armed = false;
    patch_byte(g_bp_addr, g_bp_orig);
    g_bp_stepping = 0;
    g_hw_armed = false;
    if (g_hw_fd >= 0) {
        if (g_hw_fd_is_perf) ioctl(g_hw_fd, PERF_EVENT_IOC_DISABLE, 0);
        close(g_hw_fd);
        g_hw_fd = -1;
    }
    g_hw_fd_is_perf = false;
    g_defect_mode = 0;
    return true;
}

void test_live_arms() {
    void* page = mmap(nullptr, 0x2000, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        std::fprintf(stderr, "FAIL: could not map the stub page (errno=%d %s)\n",
                     errno, std::strerror(errno));
        ++failures;
        return;
    }
    g_page = (uint64_t)page;
    std::memcpy((uint8_t*)page + kStubHwOffset, kStubHw, sizeof kStubHw);
    std::memcpy((uint8_t*)page + kStubBpOffset, kStubBp, sizeof kStubBp);
    if (mprotect(page, 0x2000, PROT_READ | PROT_EXEC) != 0) {
        std::fprintf(stderr, "FAIL: could not make the stub page executable (errno=%d %s)\n",
                     errno, std::strerror(errno));
        ++failures;
        return;
    }

    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = trap_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGTRAP, &sa, nullptr) != 0) {
        std::fprintf(stderr, "FAIL: could not install the SIGTRAP handler (errno=%d)\n", errno);
        ++failures;
        return;
    }

    // The mutation arm runs FIRST: if it were to pass the fixed expectations, the fixed arm's
    // success would prove nothing about the predicate.
    run_live_arm(/*defect_mode=*/true);
    run_live_arm(/*defect_mode=*/false);

    signal(SIGTRAP, SIG_DFL);
    munmap(page, 0x2000);
}

// ---------------------------------------------------------------------------------------------
// 3. Wiring check: the production handler must actually consult the shared predicate. Neither of
//    the arms above can see a call site deleted from exec_image_linux.cpp, so read it.
// ---------------------------------------------------------------------------------------------

void test_production_call_site() {
#ifdef PROSPER_EXEC_IMAGE_LINUX_SOURCE
    // The path is supplied by the build (CMAKE_CURRENT_SOURCE_DIR); it is deliberately not printed,
    // because it is an absolute path on whoever's machine ran the build.
    std::FILE* f = std::fopen(PROSPER_EXEC_IMAGE_LINUX_SOURCE, "rb");
    if (!f) {
        std::fprintf(stderr, "FAIL: cannot read the Linux fault-handler source configured at build "
                             "time, so the #1932 call site could not be confirmed\n");
        ++failures;
        return;
    }
    std::string text;
    char buffer[8192];
    size_t got = 0;
    while ((got = std::fread(buffer, 1, sizeof buffer, f)) > 0) text.append(buffer, got);
    std::fclose(f);
    // Match the branch GUARD, not merely the symbol: the surrounding comment names the function
    // too, so a bare symbol search still passes after the call site is reverted (measured — the
    // first version of this check did exactly that). If the guard is ever reformatted, update the
    // literal below rather than weakening it.
    CHECK(text.find("if (sig == SIGTRAP && prosper::host::hardware_breakpoint_claims_sigtrap(")
              != std::string::npos,
          "wiring: the Linux fault handler's hardware-breakpoint branch must be guarded by "
          "prosper::host::hardware_breakpoint_claims_sigtrap()");
#else
    std::fprintf(stderr, "FAIL: PROSPER_EXEC_IMAGE_LINUX_SOURCE was not defined by the build\n");
    ++failures;
#endif
}

}  // namespace

int main() {
    test_decision_table();
    test_live_arms();
    test_production_call_site();
    if (failures == 0) std::printf("test_trap_arbitration: OK\n");
    return failures == 0 ? 0 : 1;
}

#else   // !__linux__

int main() {
    // The arbitration itself is header-only and platform-independent, but PROSPER_HWBP is a Linux
    // perf capability and this handler only exists on Linux. Keep a compile-time reference so the
    // header still builds elsewhere.
    (void)prosper::host::hardware_breakpoint_claims_sigtrap({}, {}, 0, false);
    std::printf("test_trap_arbitration: skipped (Linux-only handler)\n");
    return 0;
}

#endif
