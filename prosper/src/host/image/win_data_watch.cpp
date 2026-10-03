// Windows data write-watchpoint; see win_data_watch.hpp for the contract.
#ifdef _WIN32
#include "host/image/win_data_watch.hpp"
#include "host/image/exec_image.hpp"
#include "host/memory/guest_memory_copy.hpp"
#include "diagnostics/env_cache.hpp"
#include <windows.h>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

namespace prosper {
namespace {

struct WatchConfig {
    std::mutex mutex;   // configuration/snapshot only; never acquired in the VEH
    std::atomic<uint64_t> addr[kWinDataWatchSlots] = {};
    std::atomic<unsigned> len[kWinDataWatchSlots] = {};
    std::atomic<unsigned> count{0};   // non-zero is what "configured" means to arm_current_thread
    std::atomic<uint64_t> generation{0};
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> max_logged{64};
    std::atomic<uint64_t> last_rip{0}, last_value{0}, last_addr{0};
    std::atomic<unsigned long> last_tid{0};
    std::atomic<unsigned> last_slot{0};
    std::atomic<bool> last_value_available{false};
};
WatchConfig& cfg() {
    static WatchConfig* c = new WatchConfig;   // immortal: the VEH can run during shutdown
    return *c;
}

struct WatchSnapshot {
    WinDataWatchSpec specs[kWinDataWatchSlots]{};
    unsigned count = 0;
    uint64_t generation = 0;
};

// The helper publishes this caller-owned record while the caller is suspended, after successfully
// setting its context and before resuming it. The VEH never attributes another thread's registers
// to the process-wide configuration. Atomic fields also keep publication out of the C++ data-race
// domain; no registry lock or allocation is needed in the handler.
struct ThreadWatch {
    std::atomic<uint64_t> addr[kWinDataWatchSlots]{};
    std::atomic<unsigned> len[kWinDataWatchSlots]{};
    std::atomic<unsigned> count{0};
    std::atomic<uint64_t> generation{0};
};
thread_local ThreadWatch t_watch;
std::atomic<WinDataWatchRegistrationHook> g_registration_hook{nullptr};

WatchSnapshot thread_snapshot() {
    WatchSnapshot result;
    result.count = t_watch.count.load();
    result.generation = t_watch.generation.load();
    for (unsigned i = 0; i < result.count; ++i)
        result.specs[i] = {t_watch.addr[i].load(), t_watch.len[i].load()};
    return result;
}

void publish_thread_watch(ThreadWatch& watch, const WatchSnapshot& next) {
    watch.count = 0;
    for (unsigned i = 0; i < next.count; ++i) {
        watch.addr[i] = next.specs[i].addr;
        watch.len[i] = next.specs[i].len;
    }
    watch.generation = next.generation;
    watch.count = next.count;
}

unsigned enabled_slots(uint64_t control) {
    unsigned result = 0;
    for (unsigned i = 0; i < kWinDataWatchSlots; ++i)
        if (control & (3ull << (2 * i))) result |= 1u << i;
    return result;
}

bool unsigned_prefix(const char* text, char** end, uint64_t* value) {
    if (!text) return false;
    const char* start = text;
    while (std::isspace(static_cast<unsigned char>(*start))) ++start;
    if (*start == '+' || *start == '-') return false;
    const int saved_errno = errno;
    errno = 0;
    const auto result = std::strtoull(text, end, 0);
    const bool ok = *end != text && errno != ERANGE;
    errno = saved_errno;
    if (ok) *value = result;
    return ok;
}

LONG CALLBACK watch_veh(EXCEPTION_POINTERS* ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP)
        return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT* c = ep->ContextRecord;
    const WatchSnapshot owned = thread_snapshot();
    const uint64_t addrs[kWinDataWatchSlots] = {c->Dr0, c->Dr1, c->Dr2, c->Dr3};
    const unsigned owned_slots =
        win_data_watch_matching_slots(0xf, c->Dr7, addrs, owned.specs, owned.count);
    const unsigned matched = owned_slots & static_cast<unsigned>(c->Dr6);
    if (!matched) return EXCEPTION_CONTINUE_SEARCH;
    WatchConfig& w = cfg();
    if (!w.count.load() || w.generation.load() != owned.generation) {
        // This thread was armed for a watch that has since been dropped. Disarm it, and do not
        // count the trap: a stale debug register must not fabricate a hit.
        for (unsigned i = 0; i < kWinDataWatchSlots; ++i)
            if (owned_slots & (1u << i)) c->Dr7 &= ~(1ull << (2 * i));
        c->Dr6 &= ~static_cast<uint64_t>(matched);
        t_watch.count = 0;
        return win_data_watch_needs_other_handler(c->Dr6, c->Dr7, c->EFlags)
                   ? EXCEPTION_CONTINUE_SEARCH
                   : EXCEPTION_CONTINUE_EXECUTION;
    }
    // One instruction may match several slots. Each matching word gets its own report, rather
    // than selecting the first bit and acknowledging the unreported words along with it.
    for (unsigned slot = 0; slot < owned.count; ++slot) {
        if (!(matched & (1u << slot))) continue;
        const uint64_t addr = owned.specs[slot].addr, n = w.hits.fetch_add(1) + 1;
        const unsigned len = owned.specs[slot].len;
        uint64_t value = 0;
        const bool available = host::guest_read_exact(addr, &value, len);
        if (!available) value = 0;   // never consume a partial read
        w.last_rip = c->Rip;
        w.last_value = value;
        w.last_addr = addr;
        w.last_tid = GetCurrentThreadId();
        w.last_slot = slot;
        w.last_value_available = available;
        if (n <= w.max_logged.load()) {
            char value_text[32];
            if (available) {
                std::snprintf(value_text, sizeof(value_text), "0x%llx", (unsigned long long)value);
            } else {
                std::strcpy(value_text, "unavailable");
            }
            // A data breakpoint is a TRAP: rip is the instruction AFTER the write.
            fprintf(stderr,
                    "[hwwatch] write #%llu tid=%lu slot=%u addr=0x%llx len=%u value=%s "
                    "rip=0x%llx (%s, after the writing instruction)\n",
                    (unsigned long long)n, GetCurrentThreadId(), slot, (unsigned long long)addr,
                    len, value_text, (unsigned long long)c->Rip,
                    describe_code_address(c->Rip).c_str());
            fprintf(
                stderr,
                "[hwwatch]   rax=%llx rbx=%llx rcx=%llx rdx=%llx rsi=%llx rdi=%llx "
                "r8=%llx r9=%llx r12=%llx r13=%llx r14=%llx r15=%llx rbp=%llx rsp=%llx\n",
                (unsigned long long)c->Rax, (unsigned long long)c->Rbx, (unsigned long long)c->Rcx,
                (unsigned long long)c->Rdx, (unsigned long long)c->Rsi, (unsigned long long)c->Rdi,
                (unsigned long long)c->R8, (unsigned long long)c->R9, (unsigned long long)c->R12,
                (unsigned long long)c->R13, (unsigned long long)c->R14, (unsigned long long)c->R15,
                (unsigned long long)c->Rbp, (unsigned long long)c->Rsp);
            uint64_t bp = c->Rbp;
            for (int i = 0; i < 8 && bp > 0x10000; i++) {
                uint64_t frame[2];
                if (!host::guest_read_exact(bp, frame, sizeof(frame))) break;
                const uint64_t ret = frame[1];
                fprintf(stderr, "[hwwatch]   frame %d: 0x%llx (%s)\n", i, (unsigned long long)ret,
                        describe_code_address(ret).c_str());
                const uint64_t next = frame[0];
                if (next <= bp) break;
                bp = next;
            }
        } else if (n == w.max_logged.load() + 1) {
            fprintf(stderr, "[hwwatch] further hits not logged (PROSPER_HWWATCH_MAX=%llu)\n",
                    (unsigned long long)w.max_logged.load());
        }
    }
    c->Dr6 &= ~static_cast<uint64_t>(matched);
    return win_data_watch_needs_other_handler(c->Dr6, c->Dr7, c->EFlags)
               ? EXCEPTION_CONTINUE_SEARCH
               : EXCEPTION_CONTINUE_EXECUTION;
}

bool install_veh_once() {
    // The seam may refuse registration; it cannot fabricate an installed handler.
    if (const auto hook = g_registration_hook.load(); hook && !hook()) return false;
    static std::once_flag once;
    static bool installed = false;
    std::call_once(
        once, [] { installed = AddVectoredExceptionHandler(1 /*first*/, watch_veh) != nullptr; });
    return installed;
}

// Set the slots on `target` from a helper thread while `target` waits in join(): SetThreadContext
// on a thread's own running context is not a documented use.
bool program_thread(HANDLE target, const WatchSnapshot& next, const WatchSnapshot& previous,
                    ThreadWatch* record) {
    bool ok = false;
    if (SuspendThread(target) == (DWORD)-1) return false;
    CONTEXT ctx{};
    ctx.ContextFlags = CONTEXT_DEBUG_REGISTERS;
    if (GetThreadContext(target, &ctx)) {
        uint64_t addrs[kWinDataWatchSlots] = {ctx.Dr0, ctx.Dr1, ctx.Dr2, ctx.Dr3};
        const unsigned old_slots =
            win_data_watch_matching_slots(0xf, ctx.Dr7, addrs, previous.specs, previous.count);
        // An opt-in watch must not silently steal a debugger's or another observer's slots.
        if (!(enabled_slots(ctx.Dr7) & ~old_slots)) {
            const unsigned new_slots = (1u << next.count) - 1;
            for (unsigned i = 0; i < kWinDataWatchSlots; ++i) {
                if (!((old_slots | new_slots) & (1u << i))) continue;
                ctx.Dr7 &= ~((3ull << (2 * i)) | (0xfull << (16 + 4 * i)));
                addrs[i] = i < next.count ? next.specs[i].addr : 0;
                if (i < next.count) ctx.Dr7 |= win_data_watch_dr7_slot(i, next.specs[i].len);
            }
            ctx.Dr0 = addrs[0];
            ctx.Dr1 = addrs[1];
            ctx.Dr2 = addrs[2];
            ctx.Dr3 = addrs[3];
            ctx.Dr6 &= ~static_cast<uint64_t>(old_slots | new_slots);
            ok = SetThreadContext(target, &ctx) != 0;
            if (ok) publish_thread_watch(*record, next);
        }
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

unsigned win_data_watch_matching_slots(uint64_t status, uint64_t control,
                                       const uint64_t addrs[kWinDataWatchSlots],
                                       const WinDataWatchSpec* owned, unsigned count) {
    if (!addrs || !owned || count > kWinDataWatchSlots) return 0;
    unsigned result = 0;
    for (unsigned i = 0; i < count; ++i) {
        const uint64_t expected = win_data_watch_dr7_slot(i, owned[i].len);
        const uint64_t fields = (3ull << (2 * i)) | (0xfull << (16 + 4 * i));
        if ((status & (1ull << i)) && expected && (control & fields) == expected &&
            addrs[i] == owned[i].addr)
            result |= 1u << i;
    }
    return result;
}

bool win_data_watch_needs_other_handler(uint64_t status, uint64_t control, uint64_t flags) {
    // DR6.BS may remain set after an earlier exception; EFLAGS.TF authenticates a currently
    // requested single step. In particular, let the existing Windows int3 re-arm handler see it.
    return (status & enabled_slots(control) & 0xf) ||
           ((status & (1ull << 14)) && (flags & 0x100)) || (status & ((1ull << 13) | (1ull << 15)));
}

bool win_data_watch_addr_ok(uint64_t addr, unsigned len) {
    return win_data_watch_dr7_slot(0, len) != 0 && addr >= 0x10000 && (addr % len) == 0;
}

bool win_data_watch_parse(const char* spec, uint64_t* addr, unsigned* len) {
    if (!spec || !*spec || !addr || !len) return false;
    char* end = nullptr;
    uint64_t a = 0;
    if (!unsigned_prefix(spec, &end, &a)) return false;
    unsigned l = 8;
    if (*end == ':') {
        char* end2 = nullptr;
        uint64_t v = 0;
        if (!unsigned_prefix(end + 1, &end2, &v) || *end2 || v > 8) return false;
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
    if (!spec || !*spec || !out) return 0;
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
    if (!install_veh_once()) {
        fprintf(stderr, "[hwwatch] handler registration failed; no watch configured\n");
        return false;
    }
    WatchConfig& w = cfg();
    std::lock_guard<std::mutex> lock(w.mutex);
    w.count = 0;
    w.generation.fetch_add(1);
    w.hits = 0;
    for (unsigned i = 0; i < kWinDataWatchSlots; i++) {
        w.addr[i] = i < count ? specs[i].addr : 0;
        w.len[i] = i < count ? specs[i].len : 0;
    }
    w.count = count;   // last: a non-zero count is what "configured" means
    return true;
}

bool win_data_watch_configure(uint64_t addr, unsigned len) {
    const WinDataWatchSpec s{addr, len};
    return win_data_watch_configure_list(&s, 1);
}

void win_data_watch_reset_for_test() {
    WatchConfig& w = cfg();
    std::lock_guard<std::mutex> lock(w.mutex);
    w.count = 0;
    w.generation.fetch_add(1);
    w.hits = 0;
    for (unsigned i = 0; i < kWinDataWatchSlots; i++) {
        w.addr[i] = 0;
        w.len[i] = 0;
    }
}

void win_data_watch_set_registration_hook_for_test(WinDataWatchRegistrationHook hook) {
    g_registration_hook = hook;
}

long win_data_watch_handle_for_test(::_EXCEPTION_POINTERS* ep) {
    return watch_veh(ep);
}

void win_data_watch_arm_current_thread() {
    static std::once_flag env_once;
    std::call_once(env_once, [] {
        const char* spec = PROSPER_ENV_VALUE("PROSPER_HWWATCH_ABS");
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
        if (const char* m = PROSPER_ENV_VALUE("PROSPER_HWWATCH_MAX")) {
            char* e = nullptr;
            uint64_t v = 0;
            if (unsigned_prefix(m, &e, &v) && !*e)
                cfg().max_logged = v;
            else
                fprintf(stderr, "[hwwatch] malformed PROSPER_HWWATCH_MAX ignored\n");
        }
        if (!win_data_watch_configure_list(specs, n)) return;
        for (unsigned i = 0; i < n; i++)
            fprintf(stderr,
                    "[hwwatch] armed: slot %u write watch on 0x%llx (%u bytes) on every "
                    "guest thread\n",
                    i, (unsigned long long)specs[i].addr, specs[i].len);
    });
    WatchConfig& w = cfg();
    if (!w.count.load()) return;   // keep the unset-variable path free of the snapshot mutex
    WatchSnapshot next;
    {
        std::lock_guard<std::mutex> lock(w.mutex);
        next.count = w.count.load();
        if (!next.count) return;
        next.generation = w.generation.load();
        for (unsigned i = 0; i < next.count; ++i)
            next.specs[i] = {w.addr[i].load(), w.len[i].load()};
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
    const WatchSnapshot previous = thread_snapshot();
    ThreadWatch* record = &t_watch;
    std::thread([&] { ok = program_thread(self, next, previous, record); }).join();
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
    if (!out || !w.hits.load()) return false;
    out->rip = w.last_rip.load();
    out->addr = w.last_addr.load();
    out->value = w.last_value.load();
    out->tid = w.last_tid.load();
    out->slot = w.last_slot.load();
    out->value_available = w.last_value_available.load();
    return true;
}

}   // namespace prosper
#endif
