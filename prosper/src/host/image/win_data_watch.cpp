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
    std::atomic<uint64_t> addr[kWinDataWatchSlots] = {};
    std::atomic<unsigned> len[kWinDataWatchSlots] = {};
    std::atomic<unsigned> count{0};   // non-zero is what "configured" means to arm_current_thread
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> max_logged{64};
    std::atomic<uint64_t> last_rip{0}, last_value{0}, last_addr{0};
    std::atomic<unsigned long> last_tid{0};
    std::atomic<unsigned> last_slot{0};
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

// DR7 enable bits (L0-L3) for every slot.
constexpr uint64_t kAllLocalEnables = 0x55;

LONG CALLBACK watch_veh(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP)
        return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT* c = ep->ContextRecord;
    if (!(c->Dr6 & 0xf))
        return EXCEPTION_CONTINUE_SEARCH;   // not one of our slots: leave other single-steps alone
    WatchConfig& w = cfg();
    if (!w.count.load()) {
        // This thread was armed for a watch that has since been dropped. Disarm it, and do not
        // count the trap: a stale debug register must not fabricate a hit.
        c->Dr7 &= ~kAllLocalEnables;
        c->Dr6 &= ~0xfull;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    unsigned slot = 0;
    while (slot < kWinDataWatchSlots && !(c->Dr6 & (1ull << slot))) slot++;
    const uint64_t addr = w.addr[slot].load(), n = w.hits.fetch_add(1) + 1;
    const unsigned len = w.len[slot].load();
    const uint64_t value = load_masked(addr, len);
    w.last_rip = c->Rip;
    w.last_value = value;
    w.last_addr = addr;
    w.last_tid = GetCurrentThreadId();
    w.last_slot = slot;
    if (n <= w.max_logged.load()) {
        // A data breakpoint is a TRAP: rip is the instruction AFTER the write.
        fprintf(stderr,
                "[hwwatch] write #%llu tid=%lu slot=%u addr=0x%llx len=%u value=0x%llx "
                "rip=0x%llx (%s, after the writing instruction)\n",
                (unsigned long long)n, GetCurrentThreadId(), slot, (unsigned long long)addr, len,
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

// Set the slots on `target` from a helper thread while `target` waits in join(): SetThreadContext
// on a thread's own running context is not a documented use.
bool program_thread(HANDLE target, const uint64_t addrs[kWinDataWatchSlots], uint64_t dr7) {
    bool ok = false;
    if (SuspendThread(target) == (DWORD)-1) return false;
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    if (GetThreadContext(target, &ctx)) {
        ctx.Dr0 = addrs[0];
        ctx.Dr1 = addrs[1];
        ctx.Dr2 = addrs[2];
        ctx.Dr3 = addrs[3];
        ctx.Dr6 = 0;
        ctx.Dr7 = dr7;
        ok = SetThreadContext(target, &ctx) != 0;
    }
    ResumeThread(target);
    return ok;
}

}   // namespace

uint64_t win_data_watch_dr7_slot(unsigned slot, unsigned len) {
    if (slot >= kWinDataWatchSlots) return 0;
    uint64_t len_bits;
    switch (len) {
        case 1: len_bits = 0; break;
        case 2: len_bits = 1; break;
        case 4: len_bits = 3; break;
        case 8: len_bits = 2; break;
        default: return 0;
    }
    // Ln (bit 2n), RWn = 01 (data writes, bits 16+4n..17+4n), LENn (bits 18+4n..19+4n).
    return (1ull << (2 * slot)) | (1ull << (16 + 4 * slot)) | (len_bits << (18 + 4 * slot));
}

uint64_t win_data_watch_dr7(unsigned len) {
    return win_data_watch_dr7_slot(0, len);
}

bool win_data_watch_addr_ok(uint64_t addr, unsigned len) {
    return win_data_watch_dr7_slot(0, len) != 0 && addr >= 0x10000 && (addr % len) == 0;
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

unsigned win_data_watch_parse_list(const char* spec, WinDataWatchSpec out[kWinDataWatchSlots]) {
    if (!spec || !*spec) return 0;
    unsigned n = 0;
    std::string rest(spec);
    size_t pos = 0;
    while (true) {
        const size_t comma = rest.find(',', pos);
        const std::string item = rest.substr(pos, comma == std::string::npos ? comma : comma - pos);
        if (n >= kWinDataWatchSlots)
            return 0;   // more entries than debug registers: refuse rather than drop some
        uint64_t a = 0;
        unsigned l = 0;
        if (!win_data_watch_parse(item.c_str(), &a, &l)) return 0;
        out[n].addr = a;
        out[n].len = l;
        n++;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return n;
}

bool win_data_watch_configure_list(const WinDataWatchSpec* specs, unsigned count) {
    if (!specs || count == 0 || count > kWinDataWatchSlots) return false;
    for (unsigned i = 0; i < count; i++)
        if (!win_data_watch_addr_ok(specs[i].addr, specs[i].len)) return false;
    WatchConfig& w = cfg();
    w.hits = 0;
    for (unsigned i = 0; i < kWinDataWatchSlots; i++) {
        w.addr[i] = i < count ? specs[i].addr : 0;
        w.len[i] = i < count ? specs[i].len : 0;
    }
    w.count = count;   // last: a non-zero count is what "configured" means
    install_veh_once();
    return true;
}

bool win_data_watch_configure(uint64_t addr, unsigned len) {
    const WinDataWatchSpec s{addr, len};
    return win_data_watch_configure_list(&s, 1);
}

void win_data_watch_reset_for_test() {
    WatchConfig& w = cfg();
    w.count = 0;
    w.hits = 0;
    for (unsigned i = 0; i < kWinDataWatchSlots; i++) {
        w.addr[i] = 0;
        w.len[i] = 0;
    }
}

void win_data_watch_arm_current_thread() {
    static std::once_flag env_once;
    std::call_once(env_once, [] {
        const char* spec = std::getenv("PROSPER_HWWATCH_ABS");
        if (!spec) return;
        WinDataWatchSpec specs[kWinDataWatchSlots];
        const unsigned n = win_data_watch_parse_list(spec, specs);
        if (!n) {
            fprintf(stderr,
                    "[hwwatch] PROSPER_HWWATCH_ABS=%s ignored: need up to %u comma-separated "
                    "0xADDR[:LEN] entries, LEN 1/2/4/8, each address aligned to its LEN\n",
                    spec, kWinDataWatchSlots);
            return;
        }
        if (const char* m = std::getenv("PROSPER_HWWATCH_MAX")) {
            char* e = nullptr;
            const unsigned long long v = std::strtoull(m, &e, 0);
            if (e != m && !*e) cfg().max_logged = v;
        }
        win_data_watch_configure_list(specs, n);
        for (unsigned i = 0; i < n; i++)
            fprintf(stderr,
                    "[hwwatch] armed: slot %u write watch on 0x%llx (%u bytes) on every "
                    "guest thread\n",
                    i, (unsigned long long)specs[i].addr, specs[i].len);
    });
    WatchConfig& w = cfg();
    const unsigned count = w.count.load();
    if (!count) return;
    uint64_t addrs[kWinDataWatchSlots] = {};
    uint64_t dr7 = 0;
    for (unsigned i = 0; i < count; i++) {
        addrs[i] = w.addr[i].load();
        dr7 |= win_data_watch_dr7_slot(i, w.len[i].load());
    }
    HANDLE self = nullptr;
    if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &self,
                         THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_SET_CONTEXT, FALSE,
                         0)) {
        fprintf(stderr, "[hwwatch] DuplicateHandle failed (%lu); thread %lu is NOT watched\n",
                GetLastError(), GetCurrentThreadId());
        return;
    }
    bool ok = false;
    std::thread([&] { ok = program_thread(self, addrs, dr7); }).join();
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
    out->addr = w.last_addr.load();
    out->value = w.last_value.load();
    out->tid = w.last_tid.load();
    out->slot = w.last_slot.load();
    return true;
}

}   // namespace prosper
#endif
