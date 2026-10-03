// Windows data write-watchpoint; see win_data_watch.hpp for the contract.
#ifdef _WIN32
#include "host/image/win_data_watch.hpp"
#include "host/image/exec_image.hpp"
#include <windows.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace prosper {
namespace {

struct WatchConfig {
    std::atomic<uint64_t> addr{0};
    std::atomic<unsigned> len{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> max_logged{64};
    std::atomic<uint64_t> last_rip{0}, last_value{0};
    std::atomic<unsigned long> last_tid{0};
};
WatchConfig& cfg() {
    static WatchConfig* c = new WatchConfig;   // immortal: the VEH can run during shutdown
    return *c;
}

bool readable(uint64_t p, size_t n) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(reinterpret_cast<void*>(static_cast<uintptr_t>(p)), &mbi, sizeof(mbi)))
        return false;
    if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_NOACCESS | PAGE_GUARD))) return false;
    return p + n <= reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
}

uint64_t load_masked(uint64_t addr, unsigned len) {
    if (!readable(addr, len)) return 0;
    uint64_t v = 0;
    std::memcpy(&v, reinterpret_cast<const void*>(static_cast<uintptr_t>(addr)), len);
    return v;
}

LONG CALLBACK watch_veh(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP)
        return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT* c = ep->ContextRecord;
    if (!(c->Dr6 & 1))
        return EXCEPTION_CONTINUE_SEARCH;   // not our slot: leave other single-steps alone
    WatchConfig& w = cfg();
    if (!w.addr.load()) {
        // This thread was armed for a watch that has since been dropped. Disarm it, and do not
        // count the trap: a stale debug register must not fabricate a hit.
        c->Dr7 &= ~1ull;
        c->Dr6 &= ~0xfull;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    const uint64_t addr = w.addr.load(), n = w.hits.fetch_add(1) + 1;
    const unsigned len = w.len.load();
    const uint64_t value = load_masked(addr, len);
    w.last_rip = c->Rip;
    w.last_value = value;
    w.last_tid = GetCurrentThreadId();
    if (n <= w.max_logged.load()) {
        // A data breakpoint is a TRAP: rip is the instruction AFTER the write.
        fprintf(stderr,
                "[hwwatch] write #%llu tid=%lu addr=0x%llx len=%u value=0x%llx "
                "rip=0x%llx (%s, after the writing instruction)\n",
                (unsigned long long)n, GetCurrentThreadId(), (unsigned long long)addr, len,
                (unsigned long long)value, (unsigned long long)c->Rip,
                describe_code_address(c->Rip).c_str());
        fprintf(stderr,
                "[hwwatch]   rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx "
                "r8=%llx r9=%llx r12=%llx r13=%llx r14=%llx r15=%llx rbp=%llx rsp=%llx\n",
                (unsigned long long)c->Rax, (unsigned long long)c->Rbx, (unsigned long long)c->Rcx,
                (unsigned long long)c->Rdx, (unsigned long long)c->Rsi, (unsigned long long)c->Rdi,
                (unsigned long long)c->R8, (unsigned long long)c->R9, (unsigned long long)c->R12,
                (unsigned long long)c->R13, (unsigned long long)c->R14, (unsigned long long)c->R15,
                (unsigned long long)c->Rbp, (unsigned long long)c->Rsp);
        uint64_t bp = c->Rbp;
        for (int i = 0; i < 8 && bp > 0x10000 && readable(bp, 16); i++) {
            const uint64_t ret = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(bp + 8));
            fprintf(stderr, "[hwwatch]   frame %d: 0x%llx (%s)\n", i, (unsigned long long)ret,
                    describe_code_address(ret).c_str());
            const uint64_t next = *reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(bp));
            if (next <= bp) break;
            bp = next;
        }
    } else if (n == w.max_logged.load() + 1) {
        fprintf(stderr, "[hwwatch] further hits not logged (PROSPER_HWWATCH_MAX=%llu)\n",
                (unsigned long long)w.max_logged.load());
    }
    c->Dr6 &= ~0xfull;
    return EXCEPTION_CONTINUE_EXECUTION;
}

void install_veh_once() {
    static std::once_flag once;
    std::call_once(once, [] { AddVectoredExceptionHandler(1 /*first*/, watch_veh); });
}

// Set slot 0 on `target` from a helper thread while `target` waits in join(): SetThreadContext on
// a thread's own running context is not a documented use.
bool program_thread(HANDLE target, uint64_t addr, uint64_t dr7) {
    bool ok = false;
    if (SuspendThread(target) == (DWORD)-1) return false;
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    if (GetThreadContext(target, &ctx)) {
        ctx.Dr0 = addr;
        ctx.Dr6 = 0;
        ctx.Dr7 = dr7;
        ok = SetThreadContext(target, &ctx) != 0;
    }
    ResumeThread(target);
    return ok;
}

}  // namespace

uint64_t win_data_watch_dr7(unsigned len) {
    uint64_t len_bits;
    switch (len) {
        case 1: len_bits = 0; break;
        case 2: len_bits = 1; break;
        case 4: len_bits = 3; break;
        case 8: len_bits = 2; break;
        default: return 0;
    }
    // L0 (bit 0), RW0 = 01 (data writes, bits 16-17), LEN0 (bits 18-19).
    return 1ull | (1ull << 16) | (len_bits << 18);
}

bool win_data_watch_addr_ok(uint64_t addr, unsigned len) {
    return win_data_watch_dr7(len) != 0 && addr >= 0x10000 && (addr % len) == 0;
}

bool win_data_watch_parse(const char* spec, uint64_t* addr, unsigned* len) {
    if (!spec || !*spec) return false;
    char* end = nullptr;
    const unsigned long long a = std::strtoull(spec, &end, 0);
    if (end == spec) return false;
    unsigned l = 8;
    if (*end == ':') {
        char* end2 = nullptr;
        const unsigned long v = std::strtoul(end + 1, &end2, 0);
        if (end2 == end + 1 || *end2) return false;
        l = (unsigned)v;
    } else if (*end) {
        return false;
    }
    if (!win_data_watch_addr_ok(a, l)) return false;
    *addr = a;
    *len = l;
    return true;
}

bool win_data_watch_configure(uint64_t addr, unsigned len) {
    if (!win_data_watch_addr_ok(addr, len)) return false;
    WatchConfig& w = cfg();
    w.hits = 0;
    w.len = len;
    w.addr = addr;   // last: a non-zero addr is what "configured" means to arm_current_thread
    install_veh_once();
    return true;
}

void win_data_watch_reset_for_test() {
    WatchConfig& w = cfg();
    w.addr = 0;
    w.len = 0;
    w.hits = 0;
}

void win_data_watch_arm_current_thread() {
    static std::once_flag env_once;
    std::call_once(env_once, [] {
        const char* spec = std::getenv("PROSPER_HWWATCH_ABS");
        if (!spec) return;
        uint64_t a = 0;
        unsigned l = 0;
        if (!win_data_watch_parse(spec, &a, &l)) {
            fprintf(stderr,
                    "[hwwatch] PROSPER_HWWATCH_ABS=%s ignored: need 0xADDR[:LEN] with LEN 1/2/4/8 "
                    "and an address aligned to LEN\n",
                    spec);
            return;
        }
        if (const char* m = std::getenv("PROSPER_HWWATCH_MAX")) {
            char* e = nullptr;
            const unsigned long long v = std::strtoull(m, &e, 0);
            if (e != m && !*e) cfg().max_logged = v;
        }
        win_data_watch_configure(a, l);
        fprintf(stderr, "[hwwatch] armed: write watch on 0x%llx (%u bytes) on every guest thread\n",
                (unsigned long long)a, l);
    });
    WatchConfig& w = cfg();
    const uint64_t addr = w.addr.load();
    if (!addr) return;
    HANDLE self = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE,
                         0)) {
        fprintf(stderr, "[hwwatch] DuplicateHandle failed (%lu); thread %lu is NOT watched\n",
                GetLastError(), GetCurrentThreadId());
        return;
    }
    bool ok = false;
    const uint64_t dr7 = win_data_watch_dr7(w.len.load());
    std::thread([&] { ok = program_thread(self, addr, dr7); }).join();
    CloseHandle(self);
    if (!ok)
        fprintf(stderr,
                "[hwwatch] could not program debug registers on thread %lu; it is NOT watched\n",
                GetCurrentThreadId());
}

uint64_t win_data_watch_hit_count() {
    return cfg().hits.load();
}

bool win_data_watch_last_hit(WinDataWatchHit* out) {
    WatchConfig& w = cfg();
    if (!w.hits.load()) return false;
    out->rip = w.last_rip.load();
    out->addr = w.addr.load();
    out->value = w.last_value.load();
    out->tid = w.last_tid.load();
    return true;
}

}  // namespace prosper
#endif
