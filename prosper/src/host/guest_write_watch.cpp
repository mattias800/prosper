#include "guest_write_watch.hpp"

#include <atomic>
#include <utility>

namespace prosper::host {

uint32_t guest_dmem_write_trace_decode_contiguous_store_size(const uint8_t* code) {
    if (!code) return 0;
    size_t cursor = 0;
    bool operand16 = false;
    bool rex_w = false;
    bool saw_rex = false;
    for (; cursor < 14; ++cursor) {
        const uint8_t prefix = code[cursor];
        if (prefix == 0x66) {
            if (saw_rex) return 0; // a legacy prefix after REX makes REX interpretation ambiguous
            operand16 = true;
        } else if (prefix >= 0x40 && prefix <= 0x4f) {
            saw_rex = true;
            rex_w = (prefix & 0x08) != 0;
        } else if (prefix == 0x26 || prefix == 0x2e || prefix == 0x36 || prefix == 0x3e ||
                   prefix == 0x64 || prefix == 0x65 || prefix == 0x67) {
            if (saw_rex) return 0;
            // Segment/address-size prefixes do not change these MOV widths.
        } else if (prefix == 0xf0 || prefix == 0xf2 || prefix == 0xf3) {
            return 0; // do not claim one span for LOCK/REP-prefixed behavior
        } else {
            break;
        }
    }
    if (cursor >= 14) return 0;
    const uint8_t opcode = code[cursor++];
    if (opcode != 0x88 && opcode != 0x89 && opcode != 0xc6 && opcode != 0xc7)
        return 0;
    const uint8_t modrm = code[cursor];
    if ((modrm >> 6) == 3) return 0;
    if ((opcode == 0xc6 || opcode == 0xc7) && ((modrm >> 3) & 7) != 0) return 0;
    if (opcode == 0x88 || opcode == 0xc6) return 1;
    return rex_w ? 8 : operand16 ? 2 : 4;
}

} // namespace prosper::host

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace prosper::host {
namespace {

constexpr uint64_t kPageSize = 0x1000;

struct AliasRange {
    uint64_t addr = 0;
    uint64_t size = 0;
    uint64_t phys = 0;
    DWORD protection = 0;
};

struct PageAlias {
    uint64_t addr = 0;
    DWORD protection = 0;
};

struct WatchedPage {
    uint64_t phys = 0;
    uint64_t generation = 0;
    uint32_t references = 0;
    bool armed = false;
    std::vector<PageAlias> aliases;
};

struct RegistrationPage {
    WatchedPage* page = nullptr;
    uint64_t generation = 0;
};

struct Registration {
    std::vector<RegistrationPage> pages;
};

struct WatchState {
    std::mutex mutex;
    uint64_t next_id = 0;
    std::vector<AliasRange> aliases;
    std::unordered_map<uint64_t, std::unique_ptr<WatchedPage>> pages_by_phys;
    std::unordered_map<uint64_t, WatchedPage*> pages_by_addr;
    std::unordered_map<uint64_t, Registration> registrations;
};

WatchState& state() {
    // Watches can outlive other translation-unit statics during process teardown.
    static WatchState* value = new WatchState;
    return *value;
}

struct AtomicStats {
    std::atomic<uint64_t> create_attempts{0};
    std::atomic<uint64_t> registrations{0};
    std::atomic<uint64_t> registered_pages{0};
    std::atomic<uint64_t> create_no_mapping{0};
    std::atomic<uint64_t> create_incomplete_aliases{0};
    std::atomic<uint64_t> create_oversized{0};
    std::atomic<uint64_t> create_bytes_le_1m{0};
    std::atomic<uint64_t> create_bytes_le_8m{0};
    std::atomic<uint64_t> create_bytes_le_32m{0};
    std::atomic<uint64_t> create_bytes_gt_32m{0};
    std::atomic<uint64_t> create_protect_failures{0};
    std::atomic<uint64_t> queries{0};
    std::atomic<uint64_t> unchanged{0};
    std::atomic<uint64_t> dirty{0};
    std::atomic<uint64_t> unknown{0};
    std::atomic<uint64_t> faults{0};
    std::atomic<uint64_t> stale_faults{0};
    std::atomic<uint64_t> physical_writes{0};
    std::atomic<uint64_t> rearms{0};
};

AtomicStats& stats() {
    static AtomicStats value;
    return value;
}

void log_create_failure(const char* reason, uint64_t addr, uint64_t size, uint64_t page) {
    if (!std::getenv("PROSPER_WRITE_WATCH_LOG")) return;
    static std::atomic<uint32_t> count{0};
    if (count.fetch_add(1, std::memory_order_relaxed) >= 128) return;
    std::fprintf(stderr,
                 "[write-watch] create failed reason=%s addr=0x%llx bytes=0x%llx page=0x%llx\n",
                 reason, (unsigned long long)addr, (unsigned long long)size,
                 (unsigned long long)page);
}

bool writable_protection(DWORD protection) {
    const DWORD base = protection & 0xffu;
    return base == PAGE_READWRITE || base == PAGE_WRITECOPY ||
           base == PAGE_EXECUTE_READWRITE || base == PAGE_EXECUTE_WRITECOPY;
}

DWORD read_only_protection(DWORD protection) {
    const DWORD base = protection & 0xffu;
    if (base == PAGE_EXECUTE || base == PAGE_EXECUTE_READ ||
        base == PAGE_EXECUTE_READWRITE || base == PAGE_EXECUTE_WRITECOPY)
        return PAGE_EXECUTE_READ;
    return PAGE_READONLY;
}

bool restore_page(WatchedPage& page) {
    bool ok = true;
    for (const PageAlias& alias : page.aliases) {
        DWORD ignored = 0;
        if (!VirtualProtect(reinterpret_cast<void*>(static_cast<uintptr_t>(alias.addr)),
                            kPageSize, alias.protection, &ignored))
            ok = false;
    }
    page.armed = false;
    return ok;
}

struct ProtectionRun {
    uint64_t addr = 0;
    uint64_t size = 0;
    DWORD from = 0;
    DWORD to = 0;
};

std::vector<ProtectionRun> protection_runs(const std::vector<WatchedPage*>& pages,
                                           bool arm) {
    std::vector<ProtectionRun> runs;
    for (WatchedPage* page : pages) {
        if (!page || page->armed != !arm) continue;
        for (const PageAlias& alias : page->aliases) {
            if (!writable_protection(alias.protection)) continue;
            runs.push_back({alias.addr, kPageSize,
                            arm ? alias.protection : read_only_protection(alias.protection),
                            arm ? read_only_protection(alias.protection) : alias.protection});
        }
    }
    std::sort(runs.begin(), runs.end(), [](const ProtectionRun& a, const ProtectionRun& b) {
        if (a.from != b.from) return a.from < b.from;
        if (a.to != b.to) return a.to < b.to;
        return a.addr < b.addr;
    });
    std::vector<ProtectionRun> merged;
    merged.reserve(runs.size());
    for (const ProtectionRun& run : runs) {
        if (!merged.empty() && merged.back().from == run.from && merged.back().to == run.to &&
            merged.back().addr + merged.back().size == run.addr) {
            merged.back().size += run.size;
        } else {
            merged.push_back(run);
        }
    }
    return merged;
}

bool set_pages_armed(const std::vector<WatchedPage*>& pages, bool armed) {
    std::vector<ProtectionRun> runs = protection_runs(pages, armed);
    size_t changed = 0;
    for (const ProtectionRun& run : runs) {
        DWORD ignored = 0;
        if (!VirtualProtect(reinterpret_cast<void*>(static_cast<uintptr_t>(run.addr)),
                            static_cast<SIZE_T>(run.size), run.to, &ignored)) {
            while (changed) {
                const ProtectionRun& prior = runs[--changed];
                VirtualProtect(reinterpret_cast<void*>(static_cast<uintptr_t>(prior.addr)),
                               static_cast<SIZE_T>(prior.size), prior.from, &ignored);
            }
            return false;
        }
        ++changed;
    }
    for (WatchedPage* page : pages)
        if (page && page->armed == !armed) page->armed = armed;
    return true;
}

void clear_locked(WatchState& watch) {
    std::vector<WatchedPage*> pages;
    pages.reserve(watch.pages_by_phys.size());
    for (auto& [phys, page] : watch.pages_by_phys) {
        (void)phys;
        pages.push_back(page.get());
    }
    set_pages_armed(pages, false);
    watch.registrations.clear();
    watch.pages_by_addr.clear();
    watch.pages_by_phys.clear();
}

bool watched_phys_overlaps(const WatchState& watch, uint64_t begin, uint64_t end) {
    for (const auto& [phys, page] : watch.pages_by_phys) {
        (void)page;
        if (phys < end && phys + kPageSize > begin) return true;
    }
    return false;
}

bool watched_addr_overlaps(const WatchState& watch, uint64_t begin, uint64_t end) {
    for (const auto& [addr, page] : watch.pages_by_addr) {
        (void)page;
        if (addr < end && addr + kPageSize > begin) return true;
    }
    return false;
}

const AliasRange* alias_at(const WatchState& watch, uint64_t addr) {
    for (const AliasRange& alias : watch.aliases)
        if (addr >= alias.addr && addr < alias.addr + alias.size) return &alias;
    return nullptr;
}

bool collect_aliases(const WatchState& watch, uint64_t phys, std::vector<PageAlias>& out) {
    for (const AliasRange& alias : watch.aliases) {
        if (phys < alias.phys || phys + kPageSize > alias.phys + alias.size) continue;
        const uint64_t addr = alias.addr + (phys - alias.phys);
        if (alias.protection & (PAGE_GUARD | PAGE_NOACCESS))
            return false;
        out.push_back({addr, alias.protection});
    }
    return !out.empty();
}

void release_registration_locked(WatchState& watch,
                                 std::unordered_map<uint64_t, Registration>::iterator reg) {
    std::vector<WatchedPage*> released;
    for (const RegistrationPage& registered : reg->second.pages) {
        WatchedPage* page = registered.page;
        if (!page || !page->references) continue;
        if (--page->references) continue;
        released.push_back(page);
    }
    set_pages_armed(released, false);
    for (WatchedPage* page : released) {
        for (const PageAlias& alias : page->aliases) watch.pages_by_addr.erase(alias.addr);
        watch.pages_by_phys.erase(page->phys);
    }
    watch.registrations.erase(reg);
}

} // namespace

GuestWriteWatch::~GuestWriteWatch() { reset(); }

GuestWriteWatch::GuestWriteWatch(GuestWriteWatch&& other) noexcept
    : id_(std::exchange(other.id_, 0)) {}

GuestWriteWatch& GuestWriteWatch::operator=(GuestWriteWatch&& other) noexcept {
    if (this != &other) {
        reset();
        id_ = std::exchange(other.id_, 0);
    }
    return *this;
}

GuestWriteWatch GuestWriteWatch::create(uint64_t addr, uint64_t size) {
    stats().create_attempts.fetch_add(1, std::memory_order_relaxed);
    (void)addr;
    if (size <= (1ull << 20))
        stats().create_bytes_le_1m.fetch_add(1, std::memory_order_relaxed);
    else if (size <= (8ull << 20))
        stats().create_bytes_le_8m.fetch_add(1, std::memory_order_relaxed);
    else if (size <= (32ull << 20))
        stats().create_bytes_le_32m.fetch_add(1, std::memory_order_relaxed);
    else
        stats().create_bytes_gt_32m.fetch_add(1, std::memory_order_relaxed);
    // Windows constructs an exception-dispatch frame below the interrupted RSP before a vectored
    // handler can restore a watched page. Guest code follows the SysV ABI and may keep live locals
    // in that 128-byte red zone, so page-protection write watches can silently corrupt the guest.
    // Report unsupported and let renderer callers use their exact byte-comparison fallback.
    return {};
}

GuestWriteWatchQuery GuestWriteWatch::query() const {
    stats().queries.fetch_add(1, std::memory_order_relaxed);
    if (!id_) {
        stats().unknown.fetch_add(1, std::memory_order_relaxed);
        return GuestWriteWatchQuery::Unknown;
    }
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    auto found = watch.registrations.find(id_);
    if (found == watch.registrations.end()) {
        stats().unknown.fetch_add(1, std::memory_order_relaxed);
        return GuestWriteWatchQuery::Unknown;
    }
    for (const RegistrationPage& registered : found->second.pages) {
        if (!registered.page || !registered.page->armed ||
            registered.page->generation != registered.generation) {
            stats().dirty.fetch_add(1, std::memory_order_relaxed);
            return GuestWriteWatchQuery::Dirty;
        }
    }
    stats().unchanged.fetch_add(1, std::memory_order_relaxed);
    return GuestWriteWatchQuery::Unchanged;
}

bool GuestWriteWatch::rearm() {
    return false;
}

void GuestWriteWatch::reset() {
    if (!id_) return;
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    auto found = watch.registrations.find(id_);
    if (found != watch.registrations.end()) release_registration_locked(watch, found);
    id_ = 0;
}

GuestWriteWatchStats guest_write_watch_stats() {
    AtomicStats& value = stats();
    return {
        value.create_attempts.load(std::memory_order_relaxed),
        value.registrations.load(std::memory_order_relaxed),
        value.registered_pages.load(std::memory_order_relaxed),
        value.create_no_mapping.load(std::memory_order_relaxed),
        value.create_incomplete_aliases.load(std::memory_order_relaxed),
        value.create_oversized.load(std::memory_order_relaxed),
        value.create_bytes_le_1m.load(std::memory_order_relaxed),
        value.create_bytes_le_8m.load(std::memory_order_relaxed),
        value.create_bytes_le_32m.load(std::memory_order_relaxed),
        value.create_bytes_gt_32m.load(std::memory_order_relaxed),
        value.create_protect_failures.load(std::memory_order_relaxed),
        value.queries.load(std::memory_order_relaxed),
        value.unchanged.load(std::memory_order_relaxed),
        value.dirty.load(std::memory_order_relaxed),
        value.unknown.load(std::memory_order_relaxed),
        value.faults.load(std::memory_order_relaxed),
        value.stale_faults.load(std::memory_order_relaxed),
        value.physical_writes.load(std::memory_order_relaxed),
        value.rearms.load(std::memory_order_relaxed),
    };
}

void guest_write_watch_notify_direct_mapping_added(uint64_t addr, uint64_t size, uint64_t phys,
                                                    uint32_t protection) {
    if (!addr || !size || addr > UINT64_MAX - size || phys > UINT64_MAX - size) return;
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    if (watched_phys_overlaps(watch, phys, phys + size)) clear_locked(watch);
    watch.aliases.push_back({addr, size, phys, static_cast<DWORD>(protection)});
}

void guest_write_watch_notify_direct_mapping_protection(uint64_t addr, uint64_t size,
                                                        uint32_t protection) {
    if (!addr || !size || addr > UINT64_MAX - size) return;
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    const uint64_t end = addr + size;
    if (watched_addr_overlaps(watch, addr, end)) clear_locked(watch);
    std::vector<AliasRange> updated;
    updated.reserve(watch.aliases.size() + 2);
    for (const AliasRange& alias : watch.aliases) {
        const uint64_t alias_end = alias.addr + alias.size;
        if (end <= alias.addr || addr >= alias_end) {
            updated.push_back(alias);
            continue;
        }
        const uint64_t overlap_begin = std::max(addr, alias.addr);
        const uint64_t overlap_end = std::min(end, alias_end);
        if (alias.addr < overlap_begin)
            updated.push_back({alias.addr, overlap_begin - alias.addr,
                               alias.phys, alias.protection});
        updated.push_back({overlap_begin, overlap_end - overlap_begin,
                           alias.phys + (overlap_begin - alias.addr),
                           static_cast<DWORD>(protection)});
        if (overlap_end < alias_end)
            updated.push_back({overlap_end, alias_end - overlap_end,
                               alias.phys + (overlap_end - alias.addr), alias.protection});
    }
    watch.aliases.swap(updated);
}

void guest_write_watch_notify_direct_mapping_removed(uint64_t addr, uint64_t size) {
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    const uint64_t end = addr <= UINT64_MAX - size ? addr + size : UINT64_MAX;
    if (watched_addr_overlaps(watch, addr, end)) clear_locked(watch);
    watch.aliases.erase(
        std::remove_if(watch.aliases.begin(), watch.aliases.end(), [&](const AliasRange& alias) {
            return addr < alias.addr + alias.size && end > alias.addr;
        }),
        watch.aliases.end());
}

void guest_write_watch_notify_physical_write(uint64_t phys, uint64_t size) {
    if (!size || phys > UINT64_MAX - size) return;
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    const uint64_t end = phys + size;
    std::vector<WatchedPage*> dirty;
    for (const auto& [page_phys, page] : watch.pages_by_phys) {
        if (page_phys >= end || page_phys + kPageSize <= phys) continue;
        ++page->generation;
        dirty.push_back(page.get());
    }
    if (!dirty.empty()) {
        set_pages_armed(dirty, false);
        stats().physical_writes.fetch_add(1, std::memory_order_relaxed);
    }
}

void guest_write_watch_invalidate_all() {
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    clear_locked(watch);
}

// Windows never arms pages (create() always refuses), so a host/kernel write can never EFAULT here.
void guest_write_watch_notify_host_write(uint64_t, uint64_t) {}
void guest_write_watch_notify_gpu_write(uint64_t, uint64_t) {}

bool guest_write_watch_handle_fault(uint64_t addr) {
    (void)addr;
    return false;
}

GuestWriteWatchFaultAction guest_write_watch_handle_fault_ex(uint64_t, uint64_t, int64_t) {
    return GuestWriteWatchFaultAction::NotHandled;
}
GuestDmemWriteTraceStepAction guest_dmem_write_trace_complete_step(
        int64_t, uint64_t, GuestDmemWriteTraceEvent&) {
    return GuestDmemWriteTraceStepAction::NotHandled;
}
void guest_dmem_write_trace_init_from_environment() {}
bool guest_dmem_write_trace_configure(const GuestDmemWriteTraceConfig&) { return false; }
bool guest_dmem_write_trace_enabled() { return false; }
GuestDmemWriteTraceSnapshot guest_dmem_write_trace_snapshot() { return {}; }
void guest_dmem_write_trace_report() {}
void guest_dmem_write_trace_notify_allocation(uint64_t, uint64_t, uint32_t) {}
void guest_dmem_write_trace_set_contention_hook_for_test(
        GuestDmemWriteTraceContentionHookForTest) {}
void guest_dmem_write_trace_lock_state_for_test() {}
void guest_dmem_write_trace_unlock_state_for_test() {}

void guest_write_watch_set_fault_onstack(bool) {}   // Windows never arms page-protection watches

} // namespace prosper::host

#else   // ---- Linux / macOS: real page-protection dirty-tracking (mprotect + SIGSEGV) --------------

#include <sched.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace prosper::host {
namespace {

constexpr uint64_t kPage = 0x1000;   // host page size (mprotect granularity)

// Guest (Sony) protection flags -> host mmap prot, matching hle_kernel_mem.cpp host_prot():
// bit0=CPU_READ, bit1=CPU_WRITE (implies read), bit2=EXEC. 0 == PROT_NONE.
int host_prot(uint32_t p) {
    int hp = 0;
    if (p & 0x1) hp |= PROT_READ;
    if (p & 0x2) hp |= PROT_READ | PROT_WRITE;
    if (p & 0x4) hp |= PROT_EXEC;
    return hp;
}
bool cpu_writable(uint32_t p) { return (p & 0x2) != 0; }

struct AliasRange { uint64_t addr = 0, size = 0, phys = 0; uint32_t prot = 0; };
struct PageAlias  { uint64_t addr = 0; uint32_t prot = 0; };
// One physical page and every guest VA that currently maps it. `armed` == the pages are mprotect'd
// read-only; a write fault clears it and bumps `generation` so any registration referencing this page
// at the old generation reads Dirty.
struct WatchedPage {
    uint64_t phys = 0, generation = 0;
    uint32_t references = 0;
    bool armed = false;
    std::vector<PageAlias> aliases;
};
struct RegistrationPage { WatchedPage* page = nullptr; uint64_t generation = 0; };
struct Registration {
    std::vector<RegistrationPage> pages;
    uint64_t begin = 0, end = 0;
    bool gpu_dirty = false;
};

struct DmemTracePage {
    uint64_t phys = 0;
    std::vector<PageAlias> aliases;
};

struct DmemTraceState {
    GuestDmemWriteTraceConfig config{};
    GuestDmemWriteTraceStatus status = GuestDmemWriteTraceStatus::Disabled;
    GuestDmemWriteTraceInvalidReason invalid_reason = GuestDmemWriteTraceInvalidReason::None;
    uint64_t allocation_matches = 0;
    uint64_t mapping_matches = 0;
    uint64_t page_faults = 0;
    uint64_t selected_faults = 0;
    uint64_t selection_uncertain_faults = 0;
    uint64_t completed_steps = 0;
    uint64_t rearms = 0;
    uint64_t coverage_gaps = 0;
    uint64_t overflow_events = 0;
    uint64_t selected_occurrence = 0;
    uint64_t allocation_phys = 0;
    uint64_t target_phys = 0;
    uint64_t target_end_phys = 0;
    uint64_t target_addr = 0;       // canonical VA from the first complete covering mapping
    bool armed = false;
    bool report_emitted = false;
    bool step_invalidated = false; // retain TF ownership until the stepping thread consumes its trap
    int64_t stepping_tid = 0;       // one global RW single-step window at a time
    GuestDmemWriteTraceEvent pending{};
    std::array<uint8_t, kGuestDmemWriteTraceMaxBytes> initial{};
    uint32_t event_count = 0;
    std::array<GuestDmemWriteTraceEvent, kGuestDmemWriteTraceMaxEvents> events{};
    std::vector<DmemTracePage> pages;   // at most two: selected byte count is <= 128
};

struct WatchState {
    std::mutex mutex;
    uint64_t next_id = 0;
    std::vector<AliasRange> aliases;                                       // live dmem topology
    std::unordered_map<uint64_t, std::unique_ptr<WatchedPage>> pages_by_phys;
    std::unordered_map<uint64_t, WatchedPage*> pages_by_addr;              // page-aligned VA -> page
    std::unordered_map<uint64_t, Registration> registrations;
    std::atomic<bool> fault_onstack{false};                               // red-zone-safe gate
    // Lock-free ownership gate for SIGTRAP coexistence. Most TRAP_TRACE deliveries belong to other
    // debugger/profiler machinery; they must not touch (or wait for) the dmem state mutex unless this
    // exact thread owns the diagnostic's one-instruction step.
    std::atomic<int64_t> pending_trace_step_tid{0};
    DmemTraceState trace;
};
static_assert(std::atomic<int64_t>::is_always_lock_free,
              "the signal-path pending-step discriminator must be lock-free");
WatchState& state() { static WatchState* value = new WatchState; return *value; }
std::atomic<GuestDmemWriteTraceContentionHookForTest> trace_contention_hook_for_test{nullptr};

struct AtomicStats {
    std::atomic<uint64_t> create_attempts{0}, registrations{0}, registered_pages{0},
        create_no_mapping{0}, create_incomplete_aliases{0}, create_oversized{0},
        create_bytes_le_1m{0}, create_bytes_le_8m{0}, create_bytes_le_32m{0},
        create_bytes_gt_32m{0},
        create_protect_failures{0},
        queries{0}, unchanged{0}, dirty{0}, unknown{0}, faults{0}, stale_faults{0},
        physical_writes{0}, rearms{0};
};
AtomicStats& stats() { static AtomicStats* value = new AtomicStats; return *value; }
inline void bump(std::atomic<uint64_t>& c) { c.fetch_add(1, std::memory_order_relaxed); }

// Does any recorded alias overlap [begin, end)? O(alias ranges). Lets the notify hooks skip the O(watched
// pages) purge on the common path: a fresh VA (no reuse) or a non-dmem munmap (every guest heap free).
bool va_range_has_alias(const WatchState& w, uint64_t begin, uint64_t end) {
    for (const AliasRange& a : w.aliases)
        if (a.addr < end && a.addr + a.size > begin) return true;
    return false;
}
struct ProtectionRun {
    uint64_t addr = 0;
    uint64_t size = 0;
    int from = 0;
    int to = 0;
};

// Coalesce adjacent pages before changing their protection. Texture and compute-cache watches commonly
// cover tens of MiB, so issuing one mprotect per 4 KiB page makes registration spend thousands of
// syscalls and global TLB shootdowns on a range the kernel can protect in one operation.
bool trace_covers_phys_locked(const WatchState& w, uint64_t phys) {
    return std::any_of(w.trace.pages.begin(), w.trace.pages.end(),
                       [&](const DmemTracePage& page) { return page.phys == phys; });
}

// Page protection is jointly owned by the production dirty watch and the diagnostic overlay. The
// diagnostic's one-instruction Stepping window is the deliberate exception: every covered alias must
// stay RW until the owning TID consumes its TF trap, so a concurrent production re-arm is refused.
bool page_requires_read_only_locked(const WatchState& w, const WatchedPage& page,
                                    bool production_armed) {
    const bool trace_page = trace_covers_phys_locked(w, page.phys);
    if (trace_page && w.trace.status == GuestDmemWriteTraceStatus::Stepping) return false;
    return production_armed ||
           (trace_page && w.trace.status == GuestDmemWriteTraceStatus::Armed);
}

std::vector<ProtectionRun> protection_runs(const WatchState& w,
                                           const std::vector<WatchedPage*>& pages, bool arm) {
    std::vector<ProtectionRun> runs;
    for (WatchedPage* page : pages) {
        if (!page) continue;
        const bool prior_read_only = page_requires_read_only_locked(w, *page, page->armed);
        const bool wanted_read_only = page_requires_read_only_locked(w, *page, arm);
        if (prior_read_only == wanted_read_only) continue;
        for (const PageAlias& alias : page->aliases) {
            if (!cpu_writable(alias.prot)) continue;
            const int full = host_prot(alias.prot);
            runs.push_back({alias.addr, kPage,
                            prior_read_only ? (full & ~PROT_WRITE) : full,
                            wanted_read_only ? (full & ~PROT_WRITE) : full});
        }
    }
    std::sort(runs.begin(), runs.end(), [](const ProtectionRun& a, const ProtectionRun& b) {
        if (a.from != b.from) return a.from < b.from;
        if (a.to != b.to) return a.to < b.to;
        return a.addr < b.addr;
    });
    std::vector<ProtectionRun> merged;
    merged.reserve(runs.size());
    for (const ProtectionRun& run : runs) {
        if (!merged.empty() && merged.back().from == run.from && merged.back().to == run.to &&
            merged.back().addr + merged.back().size == run.addr) {
            merged.back().size += run.size;
        } else {
            merged.push_back(run);
        }
    }
    return merged;
}

// mprotect every writable alias of `pages` to read-only (arm) or back to its guest prot (disarm).
// Read-only / PROT_NONE aliases are left untouched (a CPU store can't dirty them). Returns false and
// rolls back on the first mprotect failure so the guest is never left with a wrong protection.
bool set_pages_armed(WatchState& w, const std::vector<WatchedPage*>& pages, bool arm) {
    if (arm && w.trace.status == GuestDmemWriteTraceStatus::Stepping &&
        std::any_of(pages.begin(), pages.end(), [&](const WatchedPage* page) {
            return page && trace_covers_phys_locked(w, page->phys);
        }))
        return false;
    const std::vector<ProtectionRun> runs = protection_runs(w, pages, arm);
    size_t changed = 0;
    for (const ProtectionRun& run : runs) {
        if (mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(run.addr)),
                     static_cast<size_t>(run.size), run.to) != 0) {
            while (changed) {
                const ProtectionRun& prior = runs[--changed];
                mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(prior.addr)),
                         static_cast<size_t>(prior.size), prior.from);
            }
            return false;
        }
        ++changed;
    }
    for (WatchedPage* page : pages) if (page && page->armed != arm) page->armed = arm;
    return true;
}

void release_registration_locked(WatchState& w,
                                 std::unordered_map<uint64_t, Registration>::iterator reg) {
    std::vector<WatchedPage*> released;
    for (const RegistrationPage& rp : reg->second.pages) {
        WatchedPage* page = rp.page;
        if (!page || !page->references) continue;
        if (--page->references) continue;
        released.push_back(page);
    }
    set_pages_armed(w, released, false);
    for (WatchedPage* page : released) {
        for (const PageAlias& al : page->aliases) w.pages_by_addr.erase(al.addr);
        w.pages_by_phys.erase(page->phys);
    }
    w.registrations.erase(reg);
}

// A dmem topology/protection change makes every currently-watched page overlapping the changed range
// Dirty (bump generation) and unmapped from the address index; the next query reads Dirty/Unknown and
// the renderer re-establishes a fresh watch. Called with the lock held.
void invalidate_phys_range_locked(WatchState& w, uint64_t phys_begin, uint64_t phys_end) {
    std::vector<WatchedPage*> hit;
    for (auto& [phys, page] : w.pages_by_phys)
        if (phys < phys_end && phys + kPage > phys_begin) hit.push_back(page.get());
    set_pages_armed(w, hit, false);
    for (WatchedPage* page : hit) page->generation++;
}

// Drop every page-granular alias whose VA falls in [begin, end) from its WatchedPage: remove it from the
// page's alias list AND from the address index, and bump the page's generation (topology changed -> the
// next query reads Dirty). This is what resolves review B4: no stale VA survives in pages_by_addr for a
// later handle_fault to mprotect after the VA has been unmapped/reused. Does NOT mprotect the vanishing
// VAs (the caller is unmapping/remapping them; the kernel drops their protection) and — critically — does
// NOT free the WatchedPage: page lifetime is reference-counted and owned by release_registration_locked,
// so freeing one here would dangle a live Registration's raw pointer. A page that loses its last alias
// just lingers, disarmed and Dirty, until its owning registration resets. Called with the lock held.
void purge_va_range_locked(WatchState& w, uint64_t begin, uint64_t end, bool remove_topology) {
    for (auto& [phys, pageptr] : w.pages_by_phys) {
        (void)phys;
        WatchedPage* page = pageptr.get();
        bool changed = false;
        for (auto ai = page->aliases.begin(); ai != page->aliases.end();) {
            const uint64_t apage = ai->addr & ~(kPage - 1);
            if (apage < end && apage + kPage > begin) {
                w.pages_by_addr.erase(apage);
                ai = page->aliases.erase(ai);
                changed = true;
            } else {
                ++ai;
            }
        }
        if (changed) page->generation++;
    }
    if (remove_topology)
        w.aliases.erase(std::remove_if(w.aliases.begin(), w.aliases.end(),
                                       [&](const AliasRange& a) {
                                           return a.addr < end && a.addr + a.size > begin;
                                       }),
                        w.aliases.end());
}

const char* trace_status_name(GuestDmemWriteTraceStatus status) {
    switch (status) {
    case GuestDmemWriteTraceStatus::Disabled: return "disabled";
    case GuestDmemWriteTraceStatus::WaitingAllocation: return "waiting-allocation";
    case GuestDmemWriteTraceStatus::WaitingMapping: return "waiting-mapping";
    case GuestDmemWriteTraceStatus::Armed: return "armed";
    case GuestDmemWriteTraceStatus::Stepping: return "stepping";
    case GuestDmemWriteTraceStatus::Invalid: return "invalid";
    case GuestDmemWriteTraceStatus::Overflow: return "overflow";
    }
    return "invalid-enum";
}

const char* trace_reason_name(GuestDmemWriteTraceInvalidReason reason) {
    switch (reason) {
    case GuestDmemWriteTraceInvalidReason::None: return "none";
    case GuestDmemWriteTraceInvalidReason::InvalidConfig: return "invalid-config";
    case GuestDmemWriteTraceInvalidReason::CallerDiagnosticDisabled:
        return "PROSPER_DMEM_CALLER-disabled";
    case GuestDmemWriteTraceInvalidReason::UnsupportedPlatform: return "unsupported-platform";
    case GuestDmemWriteTraceInvalidReason::NoAlternateSignalStack: return "no-alt-signal-stack";
    case GuestDmemWriteTraceInvalidReason::AmbiguousAllocation: return "ambiguous-allocation";
    case GuestDmemWriteTraceInvalidReason::AddressOverflow: return "address-overflow";
    case GuestDmemWriteTraceInvalidReason::MappingTopologyChanged: return "mapping-topology-changed";
    case GuestDmemWriteTraceInvalidReason::MappingProtectionChanged:
        return "mapping-protection-changed";
    case GuestDmemWriteTraceInvalidReason::HostWrite: return "host-write";
    case GuestDmemWriteTraceInvalidReason::PhysicalWrite: return "physical-write";
    case GuestDmemWriteTraceInvalidReason::ProtectFailed: return "protect-failed";
    case GuestDmemWriteTraceInvalidReason::RearmFailed: return "rearm-failed";
    }
    return "invalid-enum";
}

bool trace_config_valid(const GuestDmemWriteTraceConfig& config) {
    if (!config.caller_chain || !config.allocation_size || !config.size ||
        config.size > kGuestDmemWriteTraceMaxBytes || !config.max_events ||
        config.max_events > kGuestDmemWriteTraceMaxEvents ||
        config.allocation_occurrence > kGuestDmemWriteTraceMaxAllocationOccurrence)
        return false;
    return config.offset <= config.allocation_size &&
           config.size <= config.allocation_size - config.offset;
}

bool trace_page_production_armed(const WatchState& w, uint64_t phys) {
    const auto found = w.pages_by_phys.find(phys);
    return found != w.pages_by_phys.end() && found->second && found->second->armed;
}

// Change the diagnostic page protections without lying to the production dirty tracker. A normal
// diagnostic reset leaves a page read-only if GuestWriteWatch still owns it. A fault path passes
// force_writable=true only after it has marked that production page Dirty/disarmed.
bool set_trace_armed_locked(WatchState& w, bool armed, bool force_writable) {
    DmemTraceState& trace = w.trace;
    struct Changed { PageAlias alias; int from = 0; };
    std::vector<Changed> changed;
    for (const DmemTracePage& page : trace.pages) {
        const bool production_armed = trace_page_production_armed(w, page.phys);
        for (const PageAlias& alias : page.aliases) {
            if (!cpu_writable(alias.prot)) continue;
            const int full = host_prot(alias.prot);
            const int prior = (trace.armed || production_armed) ? (full & ~PROT_WRITE) : full;
            const int wanted = armed || (!force_writable && production_armed)
                                   ? (full & ~PROT_WRITE)
                                   : full;
            if (mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(alias.addr)), kPage,
                         wanted) != 0) {
                while (!changed.empty()) {
                    const Changed rollback = changed.back();
                    changed.pop_back();
                    (void)mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(rollback.alias.addr)),
                                   kPage, rollback.from);
                }
                return false;
            }
            changed.push_back({alias, prior});
        }
    }
    trace.armed = armed;
    return true;
}

void trace_invalidate_locked(WatchState& w, GuestDmemWriteTraceInvalidReason reason,
                             bool force_writable = false) {
    DmemTraceState& trace = w.trace;
    if (trace.status == GuestDmemWriteTraceStatus::Disabled ||
        trace.status == GuestDmemWriteTraceStatus::Invalid ||
        trace.status == GuestDmemWriteTraceStatus::Overflow)
        return;
    if (trace.status == GuestDmemWriteTraceStatus::Stepping) {
        // The faulting thread still has TF set even when a sibling normal-context operation makes the
        // diagnostic invalid. Retain the published TID and Stepping state so only that thread consumes
        // the imminent TRAP_TRACE, clears TF, records an explicit invalid completion, and skips re-arm.
        trace.step_invalidated = true;
        if (trace.invalid_reason == GuestDmemWriteTraceInvalidReason::None)
            trace.invalid_reason = reason;
        return;
    }
    w.pending_trace_step_tid.store(0, std::memory_order_release);
    if (trace.armed) (void)set_trace_armed_locked(w, false, force_writable);
    trace.stepping_tid = 0;
    trace.status = GuestDmemWriteTraceStatus::Invalid;
    trace.invalid_reason = reason;
}

bool trace_copy_target_locked(const DmemTraceState& trace,
                              std::array<uint8_t, kGuestDmemWriteTraceMaxBytes>& out) {
    if (!trace.config.size || trace.pages.empty()) return false;
    uint64_t phys = trace.target_phys;
    size_t copied = 0;
    while (copied < trace.config.size) {
        const uint64_t page_phys = phys & ~(kPage - 1);
        const auto page = std::find_if(trace.pages.begin(), trace.pages.end(),
                                       [&](const DmemTracePage& p) { return p.phys == page_phys; });
        if (page == trace.pages.end() || page->aliases.empty()) return false;
        const uint64_t in_page = phys - page_phys;
        const size_t chunk = std::min<size_t>(trace.config.size - copied, kPage - in_page);
        const uint64_t source = page->aliases.front().addr + in_page;
        std::memcpy(out.data() + copied,
                    reinterpret_cast<const void*>(static_cast<uintptr_t>(source)), chunk);
        phys += chunk;
        copied += chunk;
    }
    return true;
}

// Rebuild every writable VA alias for the selected physical pages. This runs only in mapping/allocation
// normal context; the signal path consumes the already-complete bounded vectors without allocating.
bool trace_collect_pages_locked(WatchState& w) {
    DmemTraceState& trace = w.trace;
    std::vector<DmemTracePage> pages;
    for (uint64_t phys = trace.target_phys & ~(kPage - 1);
         phys < trace.target_end_phys; phys += kPage) {
        DmemTracePage page;
        page.phys = phys;
        for (const AliasRange& alias : w.aliases) {
            if (!cpu_writable(alias.prot) || phys < alias.phys ||
                phys > UINT64_MAX - kPage || phys + kPage > alias.phys + alias.size)
                continue;
            page.aliases.push_back({alias.addr + (phys - alias.phys), alias.prot});
        }
        if (page.aliases.empty()) return false;
        pages.push_back(std::move(page));
    }

    uint64_t canonical = 0;
    for (const AliasRange& alias : w.aliases) {
        if (!cpu_writable(alias.prot) || trace.target_phys < alias.phys ||
            trace.target_end_phys > alias.phys + alias.size)
            continue;
        canonical = alias.addr + (trace.target_phys - alias.phys);
        break;
    }
    if (!canonical) return false;
    trace.pages = std::move(pages);
    trace.target_addr = canonical;
    return true;
}

void trace_maybe_arm_locked(WatchState& w) {
    DmemTraceState& trace = w.trace;
    if (trace.status != GuestDmemWriteTraceStatus::WaitingMapping) return;
    if (!w.fault_onstack.load(std::memory_order_acquire)) {
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::NoAlternateSignalStack);
        return;
    }
    if (!trace_collect_pages_locked(w)) return;
    if (!set_trace_armed_locked(w, true, false)) {
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::ProtectFailed);
        return;
    }
    trace.status = GuestDmemWriteTraceStatus::Armed;
    if (!trace_copy_target_locked(trace, trace.initial)) {
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::MappingTopologyChanged);
        return;
    }
    std::fprintf(stderr,
                 "[dmem-write-trace] armed caller-chain=%u allocation=0x%llx offset=0x%llx"
                 " bytes=%u target=0x%llx target-phys=0x%llx pages=%zu\n",
                 trace.config.caller_chain,
                 static_cast<unsigned long long>(trace.config.allocation_size),
                 static_cast<unsigned long long>(trace.config.offset), trace.config.size,
                 static_cast<unsigned long long>(trace.target_addr),
                 static_cast<unsigned long long>(trace.target_phys), trace.pages.size());
    constexpr char hex[] = "0123456789abcdef";
    char line[64 + kGuestDmemWriteTraceMaxBytes * 2] = {};
    const char prefix[] = "[dmem-write-trace] initial=";
    size_t used = sizeof(prefix) - 1;
    std::memcpy(line, prefix, used);
    for (uint32_t i = 0; i < trace.config.size; ++i) {
        line[used++] = hex[trace.initial[i] >> 4];
        line[used++] = hex[trace.initial[i] & 0xf];
    }
    line[used++] = '\n';
    (void)std::fwrite(line, 1, used, stderr);
}

bool trace_va_to_phys_locked(const DmemTraceState& trace, uint64_t addr, uint64_t& phys) {
    const uint64_t page_addr = addr & ~(kPage - 1);
    for (const DmemTracePage& page : trace.pages) {
        for (const PageAlias& alias : page.aliases) {
            if ((alias.addr & ~(kPage - 1)) != page_addr) continue;
            phys = page.phys + (addr - page_addr);
            return true;
        }
    }
    return false;
}

} // namespace

GuestWriteWatch::~GuestWriteWatch() { reset(); }
GuestWriteWatch::GuestWriteWatch(GuestWriteWatch&& other) noexcept
    : id_(std::exchange(other.id_, 0)) {}
GuestWriteWatch& GuestWriteWatch::operator=(GuestWriteWatch&& other) noexcept {
    if (this != &other) { reset(); id_ = std::exchange(other.id_, 0); }
    return *this;
}

GuestWriteWatch GuestWriteWatch::create(uint64_t addr, uint64_t size) {
    bump(stats().create_attempts);
    if (!addr || !size || addr > UINT64_MAX - size) return {};
    WatchState& w = state();
    // Red-zone safety: a write fault runs the SIGSEGV handler; without SA_ONSTACK the kernel writes the
    // signal frame into the faulting store's red zone. Refuse until exec_image_linux confirms onstack.
    if (!w.fault_onstack.load(std::memory_order_acquire)) { bump(stats().create_no_mapping); return {}; }

    const uint64_t begin = addr & ~(kPage - 1);
    const uint64_t raw_end = addr + size;
    if (raw_end > UINT64_MAX - (kPage - 1)) return {};
    const uint64_t end = (raw_end + kPage - 1) & ~(kPage - 1);
    const uint64_t watch_bytes = end - begin;
    if (watch_bytes <= (1ull << 20))
        bump(stats().create_bytes_le_1m);
    else if (watch_bytes <= (8ull << 20))
        bump(stats().create_bytes_le_8m);
    else if (watch_bytes <= (32ull << 20))
        bump(stats().create_bytes_le_32m);
    else
        bump(stats().create_bytes_gt_32m);
    // Every watched page carries alias/generation state and must be indexed for the SIGSEGV path.
    // For large streaming textures that fixed per-page cost is substantially higher than the callers'
    // exact byte-comparison fallback (and can register millions of pages while a level loads). Keep
    // dirty tracking for ranges where it amortizes; zero makes the diagnostic limit unbounded.
    static const uint64_t max_watch_bytes = [] {
        const char* value = std::getenv("PROSPER_WRITE_WATCH_MAX_KB");
        const uint64_t kib = value ? std::strtoull(value, nullptr, 10) : 0ull;
        if (!kib) return uint64_t{UINT64_MAX};
        return kib > UINT64_MAX / 1024ull ? uint64_t{UINT64_MAX} : uint64_t{kib * 1024ull};
    }();
    if (watch_bytes > max_watch_bytes) {
        bump(stats().create_oversized);
        return {};
    }
    std::lock_guard lock(w.mutex);

    // Resolve the watched VA pages and every physical alias in linear passes over the mapping table.
    // The old implementation called alias_at() and collect_aliases() for every 4 KiB page; registering
    // a 16 MiB video volume therefore rescanned the complete guest mapping table about eight thousand
    // times. Astro Bot could spend longer creating that watch than rendering the whole intro.
    //
    // Any gap (non-dmem heap, PROT_NONE alias, or partial physical alias) still aborts to Unknown
    // before changing protections. Keep one Resolved entry per watched VA page so duplicate aliases
    // preserve the existing registration/reference-count contract.
    struct Resolved { uint64_t phys; };
    std::vector<Resolved> resolved;
    resolved.reserve(static_cast<size_t>(watch_bytes / kPage));

    std::vector<const AliasRange*> va_aliases;
    for (const AliasRange& alias : w.aliases) {
        if (alias.addr < end && alias.addr + alias.size > begin)
            va_aliases.push_back(&alias);
    }
    std::sort(va_aliases.begin(), va_aliases.end(),
              [](const AliasRange* a, const AliasRange* b) {
                  return a->addr < b->addr;
              });
    size_t va_alias_index = 0;
    for (uint64_t va = begin; va < end; va += kPage) {
        while (va_alias_index < va_aliases.size() &&
               va_aliases[va_alias_index]->addr + va_aliases[va_alias_index]->size <= va)
            ++va_alias_index;
        if (va_alias_index == va_aliases.size() ||
            va < va_aliases[va_alias_index]->addr) {
            bump(stats().create_no_mapping);
            return {};
        }
        const AliasRange& alias = *va_aliases[va_alias_index];
        resolved.push_back({alias.phys + (va - alias.addr)});
    }

    std::vector<uint64_t> watched_phys;
    watched_phys.reserve(resolved.size());
    for (const Resolved& page : resolved) watched_phys.push_back(page.phys);
    std::sort(watched_phys.begin(), watched_phys.end());
    watched_phys.erase(std::unique(watched_phys.begin(), watched_phys.end()),
                       watched_phys.end());
    // A trace-owned single-step deliberately holds all selected aliases RW. A production watch
    // created over those pages cannot truthfully claim to be armed until the TF trap completes, so
    // refuse it and let the caller keep its exact byte-comparison fallback.
    if (w.trace.status == GuestDmemWriteTraceStatus::Stepping &&
        std::any_of(watched_phys.begin(), watched_phys.end(),
                    [&](uint64_t phys) { return trace_covers_phys_locked(w, phys); }))
        return {};

    struct PhysicalAliases {
        std::vector<PageAlias> aliases;
        bool inaccessible = false;
    };
    std::unordered_map<uint64_t, PhysicalAliases> aliases_by_phys;
    aliases_by_phys.reserve(watched_phys.size());
    for (const AliasRange& alias : w.aliases) {
        const uint64_t alias_phys_end = alias.phys + alias.size;
        auto physical = std::lower_bound(watched_phys.begin(), watched_phys.end(), alias.phys);
        while (physical != watched_phys.end() && *physical < alias_phys_end) {
            const uint64_t page_phys = *physical++;
            if (page_phys > UINT64_MAX - kPage || page_phys + kPage > alias_phys_end)
                continue;
            PhysicalAliases& set = aliases_by_phys[page_phys];
            if (alias.prot == 0)
                set.inaccessible = true;
            else
                set.aliases.push_back({alias.addr + (page_phys - alias.phys), alias.prot});
        }
    }
    for (uint64_t phys : watched_phys) {
        const auto found = aliases_by_phys.find(phys);
        if (found == aliases_by_phys.end() || found->second.inaccessible ||
            found->second.aliases.empty()) {
            bump(stats().create_incomplete_aliases);
            return {};
        }
    }

    // Second pass: get-or-create the WatchedPage per phys, arm newly-referenced pages, build the reg.
    const uint64_t id = ++w.next_id;
    Registration reg;
    reg.begin = addr;
    reg.end = raw_end;
    std::vector<WatchedPage*> to_arm;
    for (Resolved& r : resolved) {
        auto it = w.pages_by_phys.find(r.phys);
        WatchedPage* page;
        if (it == w.pages_by_phys.end()) {
            auto up = std::make_unique<WatchedPage>();
            up->phys = r.phys;
            up->aliases = std::move(aliases_by_phys[r.phys].aliases);
            page = up.get();
            w.pages_by_phys.emplace(r.phys, std::move(up));
            for (const PageAlias& al : page->aliases) w.pages_by_addr[al.addr & ~(kPage - 1)] = page;
            to_arm.push_back(page);
        } else {
            page = it->second.get();
        }
        page->references++;
        reg.pages.push_back({page, page->generation});
    }
    if (!set_pages_armed(w, to_arm, true)) {
        // Undo references + any pages we just created; report protect failure -> Unknown.
        for (const RegistrationPage& rp : reg.pages)
            if (rp.page && rp.page->references) rp.page->references--;
        for (WatchedPage* page : to_arm) {
            for (const PageAlias& al : page->aliases) w.pages_by_addr.erase(al.addr & ~(kPage - 1));
            w.pages_by_phys.erase(page->phys);
        }
        bump(stats().create_protect_failures);
        return {};
    }
    bump(stats().registrations);
    stats().registered_pages.fetch_add(reg.pages.size(), std::memory_order_relaxed);
    w.registrations.emplace(id, std::move(reg));
    return GuestWriteWatch(id);
}

GuestWriteWatchQuery GuestWriteWatch::query() const {
    bump(stats().queries);
    if (!id_) { bump(stats().unknown); return GuestWriteWatchQuery::Unknown; }
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    auto found = w.registrations.find(id_);
    if (found == w.registrations.end()) { bump(stats().unknown); return GuestWriteWatchQuery::Unknown; }
    if (found->second.gpu_dirty) {
        bump(stats().dirty);
        return GuestWriteWatchQuery::Dirty;
    }
    for (const RegistrationPage& rp : found->second.pages) {
        if (!rp.page || !rp.page->armed || rp.page->generation != rp.generation) {
            bump(stats().dirty);
            return GuestWriteWatchQuery::Dirty;
        }
    }
    bump(stats().unchanged);
    return GuestWriteWatchQuery::Unchanged;
}

bool GuestWriteWatch::rearm() {
    if (!id_) return false;
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    auto found = w.registrations.find(id_);
    if (found == w.registrations.end()) return false;
    std::vector<WatchedPage*> pages;
    for (const RegistrationPage& rp : found->second.pages) if (rp.page) pages.push_back(rp.page);
    if (!set_pages_armed(w, pages, true)) return false;   // couldn't re-protect -> caller re-creates
    for (RegistrationPage& rp : found->second.pages) if (rp.page) rp.generation = rp.page->generation;
    found->second.gpu_dirty = false;
    bump(stats().rearms);
    return true;
}

void GuestWriteWatch::reset() {
    if (!id_) return;
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    auto found = w.registrations.find(id_);
    if (found != w.registrations.end()) release_registration_locked(w, found);
    id_ = 0;
}

GuestWriteWatchStats guest_write_watch_stats() {
    AtomicStats& v = stats();
    return {v.create_attempts.load(), v.registrations.load(), v.registered_pages.load(),
            v.create_no_mapping.load(), v.create_incomplete_aliases.load(),
            v.create_oversized.load(), v.create_bytes_le_1m.load(),
            v.create_bytes_le_8m.load(), v.create_bytes_le_32m.load(),
            v.create_bytes_gt_32m.load(), v.create_protect_failures.load(),
            v.queries.load(), v.unchanged.load(), v.dirty.load(),
            v.unknown.load(), v.faults.load(), v.stale_faults.load(),
            v.physical_writes.load(), v.rearms.load()};
}

bool guest_dmem_write_trace_configure(const GuestDmemWriteTraceConfig& config) {
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    if (w.trace.armed) (void)set_trace_armed_locked(w, false, false);
    w.pending_trace_step_tid.store(0, std::memory_order_release);
    w.trace = {};
    w.trace.config = config;
    if (!trace_config_valid(config)) {
        w.trace.status = GuestDmemWriteTraceStatus::Invalid;
        w.trace.invalid_reason = GuestDmemWriteTraceInvalidReason::InvalidConfig;
        return false;
    }
#if !defined(__linux__)
    w.trace.status = GuestDmemWriteTraceStatus::Invalid;
    w.trace.invalid_reason = GuestDmemWriteTraceInvalidReason::UnsupportedPlatform;
    return false;
#else
    w.trace.status = GuestDmemWriteTraceStatus::WaitingAllocation;
    std::fprintf(stderr,
                 "[dmem-write-trace] configured caller-chain=%u allocation=0x%llx"
                 " occurrence-mode=%s requested-occurrence=%u offset=0x%llx"
                 " bytes=%u max-events=%u\n",
                 config.caller_chain, static_cast<unsigned long long>(config.allocation_size),
                 config.allocation_occurrence ? "ordinal" : "unique",
                 config.allocation_occurrence,
                 static_cast<unsigned long long>(config.offset), config.size, config.max_events);
    return true;
#endif
}

void guest_dmem_write_trace_init_from_environment() {
    const char* value = std::getenv("PROSPER_DMEM_WRITE_TRACE");
    if (!value || !*value) return;

    GuestDmemWriteTraceConfig config{};
    uint64_t parsed[5] = {};
    const char* cursor = value;
    bool valid = true;
    bool terminated = false;
    int fields = 0;
    while (valid && fields < 5) {
        char* end = nullptr;
        parsed[fields++] = std::strtoull(cursor, &end, 0);
        if (end == cursor || (*end != ':' && *end != '\0')) {
            valid = false;
            break;
        }
        if (*end == '\0') {
            terminated = true;
            break;
        }
        cursor = end + 1;
    }
    if (!terminated) valid = false; // a sixth field or trailing separator
    if (fields != 4 && fields != 5) valid = false;
    config.caller_chain = parsed[0] <= UINT32_MAX ? static_cast<uint32_t>(parsed[0]) : 0;
    config.allocation_size = parsed[1];
    const int offset_field = fields == 5 ? 3 : 2;
    const int size_field = fields == 5 ? 4 : 3;
    config.offset = parsed[offset_field];
    config.size = parsed[size_field] <= UINT32_MAX
                      ? static_cast<uint32_t>(parsed[size_field]) : 0;
    if (fields == 5) {
        if (!parsed[2] || parsed[2] > kGuestDmemWriteTraceMaxAllocationOccurrence)
            valid = false;
        else
            config.allocation_occurrence = static_cast<uint32_t>(parsed[2]);
    }
    config.max_events = 32;
    if (const char* max_value = std::getenv("PROSPER_DMEM_WRITE_TRACE_MAX_EVENTS")) {
        char* end = nullptr;
        const unsigned long parsed_max = std::strtoul(max_value, &end, 0);
        if (end == max_value || *end != '\0' || parsed_max > UINT32_MAX)
            valid = false;
        else
            config.max_events = static_cast<uint32_t>(parsed_max);
    }
    if (!valid) config = {};
    const bool configured = guest_dmem_write_trace_configure(config);
    if (configured && !std::getenv("PROSPER_DMEM_CALLER")) {
        WatchState& w = state();
        std::lock_guard lock(w.mutex);
        w.trace.status = GuestDmemWriteTraceStatus::Invalid;
        w.trace.invalid_reason = GuestDmemWriteTraceInvalidReason::CallerDiagnosticDisabled;
        std::fprintf(stderr,
                     "[dmem-write-trace] invalid reason=PROSPER_DMEM_CALLER-disabled\n");
    } else if (!configured) {
        std::fprintf(stderr,
                     "[dmem-write-trace] invalid reason=invalid-config"
                     " expected=<chain>:<allocation-size>:<offset>:<bytes>"
                     " or <chain>:<allocation-size>:<occurrence>:<offset>:<bytes>\n");
    }
    static const bool report_registered = [] {
        return std::atexit(guest_dmem_write_trace_report) == 0;
    }();
    (void)report_registered;
}

bool guest_dmem_write_trace_enabled() {
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    return w.trace.status != GuestDmemWriteTraceStatus::Disabled;
}

GuestDmemWriteTraceSnapshot guest_dmem_write_trace_snapshot() {
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    const DmemTraceState& trace = w.trace;
    GuestDmemWriteTraceSnapshot out;
    out.config = trace.config;
    out.status = trace.status;
    out.invalid_reason = trace.invalid_reason;
    out.allocation_matches = trace.allocation_matches;
    out.mapping_matches = trace.mapping_matches;
    out.page_faults = trace.page_faults;
    out.selected_faults = trace.selected_faults;
    out.selection_uncertain_faults = trace.selection_uncertain_faults;
    out.completed_steps = trace.completed_steps;
    out.rearms = trace.rearms;
    out.coverage_gaps = trace.coverage_gaps;
    out.overflow_events = trace.overflow_events;
    out.selected_occurrence = trace.selected_occurrence;
    out.target_phys = trace.target_phys;
    out.target_addr = trace.target_addr;
    out.initial = trace.initial;
    out.event_count = trace.event_count;
    out.events = trace.events;
    return out;
}

void guest_dmem_write_trace_report() {
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    DmemTraceState& trace = w.trace;
    if (trace.status == GuestDmemWriteTraceStatus::Disabled || trace.report_emitted) return;
    trace.report_emitted = true;
    const bool valid_negative = trace.status == GuestDmemWriteTraceStatus::Armed &&
                                trace.selected_faults == 0 &&
                                trace.selection_uncertain_faults == 0 &&
                                trace.coverage_gaps == 0;
    const char* result = trace.selected_faults ? "writer-observed"
                         : valid_negative ? "no-selected-write-observed"
                                          : "undetermined";
    std::fprintf(stderr,
                 "[dmem-write-trace] summary status=%s reason=%s result=%s"
                 " allocation-matches=%llu mapping-matches=%llu page-faults=%llu"
                 " selected-faults=%llu selection-uncertain-faults=%llu"
                 " steps=%llu rearms=%llu coverage-gaps=%llu"
                 " overflow=%llu occurrence-mode=%s requested-occurrence=%u"
                 " observed-occurrences=%llu selected-occurrence=%llu events=%u/%u\n",
                 trace_status_name(trace.status), trace_reason_name(trace.invalid_reason), result,
                 static_cast<unsigned long long>(trace.allocation_matches),
                 static_cast<unsigned long long>(trace.mapping_matches),
                 static_cast<unsigned long long>(trace.page_faults),
                 static_cast<unsigned long long>(trace.selected_faults),
                 static_cast<unsigned long long>(trace.selection_uncertain_faults),
                 static_cast<unsigned long long>(trace.completed_steps),
                 static_cast<unsigned long long>(trace.rearms),
                 static_cast<unsigned long long>(trace.coverage_gaps),
                 static_cast<unsigned long long>(trace.overflow_events),
                 trace.config.allocation_occurrence ? "ordinal" : "unique",
                 trace.config.allocation_occurrence,
                 static_cast<unsigned long long>(trace.allocation_matches),
                 static_cast<unsigned long long>(trace.selected_occurrence), trace.event_count,
                 trace.config.max_events);
}

void guest_dmem_write_trace_notify_allocation(uint64_t phys, uint64_t size, uint32_t chain_id) {
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    DmemTraceState& trace = w.trace;
    const bool observing_default_ambiguity =
        trace.status == GuestDmemWriteTraceStatus::Invalid &&
        trace.invalid_reason == GuestDmemWriteTraceInvalidReason::AmbiguousAllocation &&
        trace.config.allocation_occurrence == 0;
    if (!observing_default_ambiguity &&
        trace.status != GuestDmemWriteTraceStatus::WaitingAllocation &&
        trace.status != GuestDmemWriteTraceStatus::WaitingMapping &&
        trace.status != GuestDmemWriteTraceStatus::Armed &&
        trace.status != GuestDmemWriteTraceStatus::Stepping)
        return;
    if (chain_id != trace.config.caller_chain || size != trace.config.allocation_size) return;
    ++trace.allocation_matches;
    if (observing_default_ambiguity) return;
    if (!trace.config.allocation_occurrence && trace.allocation_matches != 1) {
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::AmbiguousAllocation,
                                trace.status == GuestDmemWriteTraceStatus::Stepping);
        std::fprintf(stderr,
                     "[dmem-write-trace] invalid reason=ambiguous-allocation"
                     " caller-chain=%u allocation=0x%llx occurrence-mode=unique"
                     " observed-occurrences=%llu selected-occurrence=%llu\n",
                     chain_id, static_cast<unsigned long long>(size),
                     static_cast<unsigned long long>(trace.allocation_matches),
                     static_cast<unsigned long long>(trace.selected_occurrence));
        return;
    }
    if (trace.config.allocation_occurrence &&
        trace.allocation_matches != trace.config.allocation_occurrence)
        return;
    if (phys > UINT64_MAX - trace.config.offset ||
        phys + trace.config.offset > UINT64_MAX - trace.config.size) {
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::AddressOverflow);
        return;
    }
    trace.allocation_phys = phys;
    trace.selected_occurrence = trace.allocation_matches;
    trace.target_phys = phys + trace.config.offset;
    trace.target_end_phys = trace.target_phys + trace.config.size;
    trace.status = GuestDmemWriteTraceStatus::WaitingMapping;
    std::fprintf(stderr,
                 "[dmem-write-trace] allocation-match caller-chain=%u phys=0x%llx"
                 " allocation=0x%llx occurrence-mode=%s requested-occurrence=%u"
                 " observed-occurrences=%llu selected-occurrence=%llu target-phys=0x%llx\n",
                 chain_id, static_cast<unsigned long long>(phys),
                 static_cast<unsigned long long>(size),
                 trace.config.allocation_occurrence ? "ordinal" : "unique",
                 trace.config.allocation_occurrence,
                 static_cast<unsigned long long>(trace.allocation_matches),
                 static_cast<unsigned long long>(trace.selected_occurrence),
                 static_cast<unsigned long long>(trace.target_phys));
    trace_maybe_arm_locked(w);
}

void guest_write_watch_set_fault_onstack(bool on_altstack) {
#if defined(__linux__)
    WatchState& w = state();
    w.fault_onstack.store(on_altstack, std::memory_order_release);
    if (!on_altstack) {
        std::lock_guard lock(w.mutex);
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::NoAlternateSignalStack);
    }
#else
    // Only Linux wires handle_fault to the write-protect fault (SIGSEGV). Darwin delivers a
    // write-to-read-only-page fault as SIGBUS, which this handler is NOT hooked for, so an armed page's
    // first guest store would crash. macOS uses SA_ONSTACK unconditionally, so without this guard the
    // watch would arm and crash there. Keep fault_onstack false off Linux -> create() refuses to arm and
    // callers keep the exact byte-comparison fallback (matching Windows). This is also the single switch
    // that makes the whole watch (arming + the notify bookkeeping below) a no-op when the feature is off.
    (void)on_altstack;
#endif
}

void guest_write_watch_notify_direct_mapping_added(uint64_t addr, uint64_t size, uint64_t phys,
                                                    uint32_t protection) {
    if (!addr || !size || addr > UINT64_MAX - size || phys > UINT64_MAX - size) return;
    WatchState& w = state();
    if (!w.fault_onstack.load(std::memory_order_acquire)) return;   // feature off -> table is never used
    std::lock_guard lock(w.mutex);
    const uint64_t begin = addr & ~(kPage - 1);
    const uint64_t end = (addr + size + kPage - 1) & ~(kPage - 1);
    // VA-reuse safety: only if this VA range still carries stale coverage (rare: a MAP_FIXED reuse of a VA
    // never unmapped) do we purge it and its topology before recording the new alias, so a later watch
    // cannot resolve to the old phys. The common fresh-VA path skips the O(watched pages) scan.
    if (va_range_has_alias(w, begin, end)) purge_va_range_locked(w, begin, end, /*remove_topology=*/true);
    w.aliases.push_back({addr, size, phys, protection});

    DmemTraceState& trace = w.trace;
    const bool trace_overlap =
        (trace.status == GuestDmemWriteTraceStatus::WaitingMapping ||
         trace.status == GuestDmemWriteTraceStatus::Armed ||
         trace.status == GuestDmemWriteTraceStatus::Stepping) &&
        phys < trace.target_end_phys && phys + size > trace.target_phys;
    if (trace_overlap) {
        ++trace.mapping_matches;
        if (trace.status == GuestDmemWriteTraceStatus::Stepping) {
            trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::MappingTopologyChanged,
                                    true);
        } else if (trace.status == GuestDmemWriteTraceStatus::Armed) {
            // The new alias was writable between mmap and this notification. Even if we protect it
            // now, another thread could already have written through that process-wide coverage gap.
            // Invalidate instead of pretending the original arm remained complete.
            trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::MappingTopologyChanged);
        } else {
            trace_maybe_arm_locked(w);
        }
    }

    // A new alias to an already-watched physical page must be armed too, or a write through it would not
    // fault and the renderer would trust a stale texture (review B2). Same-phys aliasing does not change
    // content, so we do NOT bump generation — we extend the page's alias set and arm the new VA in place.
    if (!cpu_writable(protection) || w.pages_by_phys.empty()) return;   // nothing armed to extend
    for (uint64_t off = 0; off < size; off += kPage) {
        auto it = w.pages_by_phys.find(phys + off);
        if (it == w.pages_by_phys.end()) continue;
        WatchedPage* page = it->second.get();
        const uint64_t va = (addr + off) & ~(kPage - 1);
        page->aliases.push_back({va, protection});
        w.pages_by_addr[va] = page;
        if (page->armed &&
            mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(va)), kPage,
                     host_prot(protection) & ~PROT_WRITE) != 0)
            page->generation++;   // couldn't arm the new alias -> mark Dirty so the renderer re-creates
    }
}

void guest_write_watch_notify_direct_mapping_removed(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    WatchState& w = state();
    if (!w.fault_onstack.load(std::memory_order_acquire)) return;   // feature off -> nothing tracked
    std::lock_guard lock(w.mutex);
    const uint64_t begin = addr & ~(kPage - 1);
    const uint64_t end = (addr + size + kPage - 1) & ~(kPage - 1);
    if (w.trace.status == GuestDmemWriteTraceStatus::Armed ||
        w.trace.status == GuestDmemWriteTraceStatus::Stepping) {
        bool overlap = false;
        for (const DmemTracePage& page : w.trace.pages)
            for (const PageAlias& alias : page.aliases)
                if (alias.addr < end && alias.addr + kPage > begin) overlap = true;
        if (overlap)
            trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::MappingTopologyChanged,
                                    w.trace.status == GuestDmemWriteTraceStatus::Stepping);
    }
    // k_munmap calls this for EVERY unmap (including non-dmem heap frees); skip the O(watched pages) scan
    // unless a dmem alias actually overlaps. When one does, drop coverage for EVERY page of the removed
    // range (not just the first) and delete emptied pages so no stale WatchedPage / pages_by_addr entry
    // survives to mprotect a reused VA later (review B4).
    if (va_range_has_alias(w, begin, end)) purge_va_range_locked(w, begin, end, /*remove_topology=*/true);
}

void guest_write_watch_notify_direct_mapping_protection(uint64_t addr, uint64_t size,
                                                        uint32_t protection) {
    if (!addr || !size || addr > UINT64_MAX - size) return;
    WatchState& w = state();
    if (!w.fault_onstack.load(std::memory_order_acquire)) return;   // feature off -> nothing tracked
    std::lock_guard lock(w.mutex);
    const uint64_t end = addr + size;
    if (w.trace.status == GuestDmemWriteTraceStatus::Armed ||
        w.trace.status == GuestDmemWriteTraceStatus::Stepping) {
        bool overlap = false;
        for (const DmemTracePage& page : w.trace.pages)
            for (const PageAlias& alias : page.aliases)
                if (alias.addr < end && alias.addr + kPage > addr) overlap = true;
        if (overlap)
            trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::MappingProtectionChanged,
                                    w.trace.status == GuestDmemWriteTraceStatus::Stepping);
    }
    // A guest re-protect may cover only PART of an alias. Overwriting the whole AliasRange.prot would
    // mis-record the untouched remainder: if the change makes part read-only, the still-writable part
    // would read as non-writable, a future create() would skip arming it, and a CPU write there would go
    // undetected -> stale texture (review N2). Split each overlapping alias so ONLY the re-protected
    // sub-range takes the new prot; the remainder keeps its own.
    std::vector<AliasRange> split_out;
    for (auto it = w.aliases.begin(); it != w.aliases.end();) {
        const AliasRange a = *it;
        const uint64_t a_end = a.addr + a.size;
        if (!(a.addr < end && a_end > addr)) { ++it; continue; }
        it = w.aliases.erase(it);
        const uint64_t lo = std::max(a.addr, addr);   // re-protected sub-range [lo, hi)
        const uint64_t hi = std::min(a_end, end);
        if (a.addr < lo) split_out.push_back({a.addr, lo - a.addr, a.phys, a.prot});
        split_out.push_back({lo, hi - lo, a.phys + (lo - a.addr), protection});
        if (hi < a_end) split_out.push_back({hi, a_end - hi, a.phys + (hi - a.addr), a.prot});
        invalidate_phys_range_locked(w, a.phys + (lo - a.addr), a.phys + (hi - a.addr));
    }
    for (const AliasRange& r : split_out) w.aliases.push_back(r);
}

void guest_write_watch_notify_physical_write(uint64_t phys, uint64_t size) {
    if (!size) return;
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    if ((w.trace.status == GuestDmemWriteTraceStatus::Armed ||
         w.trace.status == GuestDmemWriteTraceStatus::Stepping) &&
        phys < w.trace.target_end_phys && phys + size > w.trace.target_phys)
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::PhysicalWrite,
                                w.trace.status == GuestDmemWriteTraceStatus::Stepping);
    bump(stats().physical_writes);
    invalidate_phys_range_locked(w, phys, phys + size);
}

void guest_write_watch_notify_host_write(uint64_t addr, uint64_t size) {
    if (!addr || !size || addr > UINT64_MAX - size) return;
    WatchState& w = state();
    // Hot path: the HLE calls this before EVERY read()/pread(), mostly into non-dmem heap buffers. When
    // the feature is off (the default) nothing is ever armed, so skip without even taking the lock.
    if (!w.fault_onstack.load(std::memory_order_acquire)) return;
    std::lock_guard lock(w.mutex);
    // A host/kernel store into an armed (read-only) guest page would EFAULT — e.g. sceKernelPread
    // streaming texture bytes straight into a watched dmem buffer returns an I/O error where real
    // hardware succeeds (review B5). The HLE calls this by guest VA range BEFORE such a write: restore
    // write on the overlapping pages so the store lands, and bump their generation so the watch reads
    // Dirty (the bytes are about to change). Runs in normal context, so set_pages_armed may allocate.
    const uint64_t begin = addr & ~(kPage - 1);
    const uint64_t end = (addr + size + kPage - 1) & ~(kPage - 1);
    if (!va_range_has_alias(w, begin, end)) return;   // not a dmem buffer -> skip the per-page scan
    if (w.trace.status == GuestDmemWriteTraceStatus::Armed ||
        w.trace.status == GuestDmemWriteTraceStatus::Stepping) {
        uint64_t ignored_phys = 0;
        bool overlap = false;
        for (uint64_t va = begin; va < end && !overlap; va += kPage)
            overlap = trace_va_to_phys_locked(w.trace, va, ignored_phys);
        if (overlap)
            trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::HostWrite,
                                    true);
    }
    std::vector<WatchedPage*> hit;
    for (uint64_t va = begin; va < end; va += kPage) {
        auto it = w.pages_by_addr.find(va);
        if (it != w.pages_by_addr.end() && it->second) hit.push_back(it->second);
    }
    if (hit.empty()) return;
    set_pages_armed(w, hit, false);
    for (WatchedPage* page : hit) page->generation++;
}

void guest_write_watch_notify_gpu_write(uint64_t addr, uint64_t size) {
    if (!addr || !size || addr > UINT64_MAX - size) return;
    WatchState& w = state();
    if (!w.fault_onstack.load(std::memory_order_acquire)) return;
    const uint64_t end = addr + size;
    std::lock_guard lock(w.mutex);
    if (w.trace.status == GuestDmemWriteTraceStatus::Armed ||
        w.trace.status == GuestDmemWriteTraceStatus::Stepping) {
        bool overlap = false;
        for (const DmemTracePage& page : w.trace.pages) {
            for (const PageAlias& alias : page.aliases) {
                const uint64_t selected_begin = alias.addr +
                    (w.trace.target_phys > page.phys ? w.trace.target_phys - page.phys : 0);
                const uint64_t selected_end = alias.addr +
                    (w.trace.target_end_phys < page.phys + kPage
                         ? w.trace.target_end_phys - page.phys : kPage);
                if (addr < selected_end && end > selected_begin) overlap = true;
            }
        }
        if (overlap)
            trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::PhysicalWrite,
                                    w.trace.status == GuestDmemWriteTraceStatus::Stepping);
    }
    for (auto& [id, registration] : w.registrations) {
        (void)id;
        if (registration.begin < end && addr < registration.end)
            registration.gpu_dirty = true;
    }
}

void guest_write_watch_invalidate_all() {
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    std::vector<WatchedPage*> all;
    all.reserve(w.pages_by_phys.size());
    for (auto& [phys, page] : w.pages_by_phys) { (void)phys; all.push_back(page.get()); }
    set_pages_armed(w, all, false);
    for (WatchedPage* page : all) page->generation++;
}

// Unified page-fault path for the production dirty bit and the opt-in dmem provenance overlay. The
// latter single-steps exactly one faulting store so it can retain before/after bytes; all vectors are
// built before arming, so this signal path performs no allocation.
GuestWriteWatchFaultAction guest_write_watch_handle_fault_ex(uint64_t addr, uint64_t writer_rip,
                                                             int64_t tid) {
    WatchState& w = state();
    if (!w.fault_onstack.load(std::memory_order_acquire))
        return GuestWriteWatchFaultAction::NotHandled;
    std::unique_lock<std::mutex> lock(w.mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        // Contended. We CANNOT decide ownership without the lock (pages_by_addr is being mutated), and
        // returning false here would fall through the caller's fault handler to a fatal _exit — even
        // though this may well be a watched page whose store just raced an arm/rearm. Instead resume:
        // the page is still read-only, so the store re-executes and re-faults, retrying until the lock
        // frees. The lock is only ever held for bounded, non-blocking work (create/rearm/query/notify),
        // so it always frees; a genuine (non-watched) fault then re-enters, wins the lock, and takes the
        // false path below into the real fault handler. sched_yield keeps the retry from spinning hot.
        sched_yield();
        return GuestWriteWatchFaultAction::Resume;
    }
    const uint64_t fault_page = addr & ~(kPage - 1);
    auto production_found = w.pages_by_addr.find(fault_page);
    WatchedPage* production_page = production_found == w.pages_by_addr.end()
                                       ? nullptr : production_found->second;
    uint64_t trace_fault_phys = 0;
    const bool trace_page =
        (w.trace.status == GuestDmemWriteTraceStatus::Armed ||
         w.trace.status == GuestDmemWriteTraceStatus::Stepping) &&
        trace_va_to_phys_locked(w.trace, addr, trace_fault_phys);
    if (!production_page && !trace_page) return GuestWriteWatchFaultAction::NotHandled;

    bool production_handled = false;
    auto disarm_production_page = [&](WatchedPage* page) {
        if (!page || !page->armed) return;
        bool restored = true;
        bool faulting_alias_restored = page != production_page;
        for (const PageAlias& alias : page->aliases) {
            if (!cpu_writable(alias.prot)) continue;
            const bool ok = mprotect(
                reinterpret_cast<void*>(static_cast<uintptr_t>(alias.addr)), kPage,
                host_prot(alias.prot)) == 0;
            if (!ok)
                restored = false;
            if ((alias.addr & ~(kPage - 1)) == fault_page && ok)
                faulting_alias_restored = true;
        }
        page->generation++;
        // Preserve #1144's stale-alias contract: a failure on a non-faulting alias must not leave the
        // successfully restored faulting VA logically armed. For diagnostic-only sibling pages there
        // is no faulting VA, so require the complete restore instead.
        if (page == production_page ? faulting_alias_restored : restored) page->armed = false;
        production_handled = true;
    };

    if (production_page && !production_page->armed && !trace_page) {
        // Two guest workers can take the same RO-page fault before either signal handler runs. The
        // first handler restores RW and clears `armed`; the second delivery is already queued and used
        // to fall through as a fatal guest fault despite the store now being safe to retry (Cobra's
        // parallel vertex jobs hit exactly this race). Accept the stale delivery only while the live
        // topology still describes this VA as CPU-writable. Protection changes update `w.aliases`, and
        // unmaps remove both the alias and address index, so real read-only/unmapped faults stay fatal.
        const bool live_cpu_writable = std::any_of(
            w.aliases.begin(), w.aliases.end(), [&](const AliasRange& alias) {
                return fault_page >= alias.addr && fault_page < alias.addr + alias.size &&
                       cpu_writable(alias.prot);
            });
        if (!live_cpu_writable) return GuestWriteWatchFaultAction::NotHandled;
        bump(stats().faults);
        bump(stats().stale_faults);
        return GuestWriteWatchFaultAction::Resume;
    }

    if (trace_page && w.trace.status == GuestDmemWriteTraceStatus::Stepping) {
        // A sibling already had this page fault queued before the first handler opened the global RW
        // single-step window. It can now resume, but the window means later negative coverage is void.
        if (w.trace.stepping_tid != tid) ++w.trace.coverage_gaps;
        disarm_production_page(production_page);
        if (production_handled) bump(stats().faults);
        return GuestWriteWatchFaultAction::Resume;
    }

    if (!trace_page) {
        disarm_production_page(production_page);
        if (production_handled) bump(stats().faults);
        return production_handled ? GuestWriteWatchFaultAction::Resume
                                  : GuestWriteWatchFaultAction::NotHandled;
    }

    // The diagnostic opens every selected physical page for the one stepped instruction. Mark any
    // production registrations for those pages Dirty first, so the overlay cannot make a cache watch
    // silently miss the same store.
    for (const DmemTracePage& trace_page_entry : w.trace.pages) {
        const auto found = w.pages_by_phys.find(trace_page_entry.phys);
        if (found != w.pages_by_phys.end()) disarm_production_page(found->second.get());
    }
    if (production_handled) bump(stats().faults);

    ++w.trace.page_faults;
    if (w.trace.event_count >= w.trace.config.max_events) {
        for (const DmemTracePage& page : w.trace.pages)
            for (const PageAlias& alias : page.aliases)
                if (cpu_writable(alias.prot))
                    (void)mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(alias.addr)),
                                   kPage, host_prot(alias.prot));
        w.trace.armed = false;
        w.pending_trace_step_tid.store(0, std::memory_order_release);
        w.trace.status = GuestDmemWriteTraceStatus::Overflow;
        ++w.trace.overflow_events;
        ++w.trace.coverage_gaps;
        return GuestWriteWatchFaultAction::Resume;
    }

    GuestDmemWriteTraceEvent pending{};
    pending.ordinal = w.trace.page_faults;
    pending.fault_addr = addr;
    pending.fault_phys = trace_fault_phys;
    pending.writer_rip = writer_rip;
    pending.tid = tid;
    pending.size = w.trace.config.size;
    pending.decoded_write_size = guest_dmem_write_trace_decode_contiguous_store_size(
        reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(writer_rip)));
    if (pending.decoded_write_size &&
        trace_fault_phys <= UINT64_MAX - pending.decoded_write_size) {
        pending.selected = trace_fault_phys < w.trace.target_end_phys &&
                           trace_fault_phys + pending.decoded_write_size >
                               w.trace.target_phys;
    } else {
        pending.decoded_write_size = 0;
        pending.selected = trace_fault_phys >= w.trace.target_phys &&
                           trace_fault_phys < w.trace.target_end_phys;
    }
    pending.coverage_valid_before = w.trace.coverage_gaps == 0;
    if (!trace_copy_target_locked(w.trace, pending.before)) {
        trace_invalidate_locked(w, GuestDmemWriteTraceInvalidReason::MappingTopologyChanged, true);
        return production_handled ? GuestWriteWatchFaultAction::Resume
                                  : GuestWriteWatchFaultAction::NotHandled;
    }

    bool faulting_alias_restored = false;
    bool all_restored = true;
    for (const DmemTracePage& page : w.trace.pages) {
        for (const PageAlias& alias : page.aliases) {
            if (!cpu_writable(alias.prot)) continue;
            const bool ok = mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(alias.addr)),
                                     kPage, host_prot(alias.prot)) == 0;
            if (!ok) all_restored = false;
            if ((alias.addr & ~(kPage - 1)) == fault_page && ok) faulting_alias_restored = true;
        }
    }
    if (!faulting_alias_restored) {
        w.trace.armed = false;
        w.trace.status = GuestDmemWriteTraceStatus::Invalid;
        w.trace.invalid_reason = GuestDmemWriteTraceInvalidReason::ProtectFailed;
        return production_handled ? GuestWriteWatchFaultAction::Resume
                                  : GuestWriteWatchFaultAction::NotHandled;
    }
    if (!all_restored) {
        w.trace.invalid_reason = GuestDmemWriteTraceInvalidReason::ProtectFailed;
        ++w.trace.coverage_gaps;
    }
    w.trace.pending = pending;
    w.trace.armed = false;
    w.trace.step_invalidated = false;
    w.trace.stepping_tid = tid;
    w.trace.status = GuestDmemWriteTraceStatus::Stepping;
    // Publish only after all pending event state is complete and immediately before telling the
    // signal caller to set TF. complete_step's acquire load is the lock-free discriminator for the
    // following TRAP_TRACE.
    w.pending_trace_step_tid.store(tid, std::memory_order_release);
    return GuestWriteWatchFaultAction::SingleStep;
}

bool guest_write_watch_handle_fault(uint64_t addr) {
    return guest_write_watch_handle_fault_ex(addr, 0, 0) !=
           GuestWriteWatchFaultAction::NotHandled;
}

GuestDmemWriteTraceStepAction guest_dmem_write_trace_complete_step(
        int64_t tid, uint64_t next_rip, GuestDmemWriteTraceEvent& event) {
    WatchState& w = state();
    if (w.pending_trace_step_tid.load(std::memory_order_acquire) != tid)
        return GuestDmemWriteTraceStepAction::NotHandled;
    std::unique_lock<std::mutex> lock(w.mutex, std::defer_lock);
    // The fault handler released this mutex before returning to the one stepped store, so this thread
    // cannot own it. A different thread may briefly hold it in a normal mapping/query operation. Do
    // not return to guest code while TF is set and every selected alias is RW: doing so would let an
    // arbitrary number of instructions execute before the purported post-store snapshot. Instead,
    // wait here for a bounded number of non-blocking acquisitions. Exhaustion is a fatal diagnostic
    // verdict in the signal caller, not a retry that silently widens the capture window.
    constexpr uint32_t kMaxStepLockAttempts = 1u << 20;
    bool contention_hook_called = false;
    for (uint32_t attempt = 0; attempt < kMaxStepLockAttempts && !lock.try_lock(); ++attempt) {
        if (!contention_hook_called) {
            contention_hook_called = true;
            if (auto hook = trace_contention_hook_for_test.load(std::memory_order_acquire)) hook();
        }
        sched_yield();
    }
    if (!lock.owns_lock()) return GuestDmemWriteTraceStepAction::LockTimeout;
    DmemTraceState& trace = w.trace;
    if (trace.status != GuestDmemWriteTraceStatus::Stepping || trace.stepping_tid != tid)
        return GuestDmemWriteTraceStepAction::NotHandled;

    trace.pending.next_rip = next_rip;
    const bool invalidated_during_step = trace.step_invalidated;
    // A topology invalidation may have removed the retained canonical alias. Do not dereference or
    // re-protect it; consuming this owned trap and clearing TF is still mandatory.
    const bool copied = !invalidated_during_step &&
                        trace_copy_target_locked(trace, trace.pending.after);
    bool rearmed = copied && trace.invalid_reason == GuestDmemWriteTraceInvalidReason::None;
    if (rearmed) {
        for (const DmemTracePage& page : trace.pages) {
            for (const PageAlias& alias : page.aliases) {
                if (!cpu_writable(alias.prot)) continue;
                if (mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(alias.addr)), kPage,
                             host_prot(alias.prot) & ~PROT_WRITE) != 0)
                    rearmed = false;
            }
        }
    }
    if (invalidated_during_step) {
        trace.armed = false;
        trace.status = GuestDmemWriteTraceStatus::Invalid;
        if (trace.invalid_reason == GuestDmemWriteTraceInvalidReason::None)
            trace.invalid_reason = GuestDmemWriteTraceInvalidReason::MappingTopologyChanged;
    } else if (!rearmed) {
        // A partial re-arm is worse than no watch: restore every alias RW and make the invalidity explicit.
        for (const DmemTracePage& page : trace.pages)
            for (const PageAlias& alias : page.aliases)
                if (cpu_writable(alias.prot))
                    (void)mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(alias.addr)),
                                   kPage, host_prot(alias.prot));
        trace.armed = false;
        trace.status = GuestDmemWriteTraceStatus::Invalid;
        if (trace.invalid_reason == GuestDmemWriteTraceInvalidReason::None)
            trace.invalid_reason = copied ? GuestDmemWriteTraceInvalidReason::RearmFailed
                                          : GuestDmemWriteTraceInvalidReason::MappingTopologyChanged;
    } else {
        trace.armed = true;
        trace.status = GuestDmemWriteTraceStatus::Armed;
        ++trace.rearms;
    }
    if (!trace.pending.selected && !trace.pending.decoded_write_size) {
        if (copied && std::memcmp(trace.pending.before.data(), trace.pending.after.data(),
                                  trace.pending.size) != 0) {
            // CR2 begins before the selected interval for a crossing store. A post-step byte change is
            // positive evidence that the completed instruction overlapped the selected bytes.
            trace.pending.selected = true;
        } else {
            // With no complete x86 operand decoder, an unchanged same-page store could still have
            // crossed the boundary while writing the existing value. Report unknown, never "no".
            trace.pending.selection_uncertain = true;
        }
    }
    if (trace.pending.selected)
        ++trace.selected_faults;
    else if (trace.pending.selection_uncertain)
        ++trace.selection_uncertain_faults;
    trace.pending.rearmed = rearmed;
    ++trace.completed_steps;
    ++trace.coverage_gaps;   // process-wide RO coverage is not provable during the RW single-step window
    trace.events[trace.event_count++] = trace.pending;
    event = trace.pending;
    trace.pending = {};
    trace.step_invalidated = false;
    trace.stepping_tid = 0;
    w.pending_trace_step_tid.store(0, std::memory_order_release);
    return trace.status == GuestDmemWriteTraceStatus::Invalid
               ? GuestDmemWriteTraceStepAction::CompleteInvalid
               : GuestDmemWriteTraceStepAction::Complete;
}

void guest_dmem_write_trace_set_contention_hook_for_test(
        GuestDmemWriteTraceContentionHookForTest hook) {
    trace_contention_hook_for_test.store(hook, std::memory_order_release);
}

void guest_dmem_write_trace_lock_state_for_test() { state().mutex.lock(); }
void guest_dmem_write_trace_unlock_state_for_test() { state().mutex.unlock(); }

} // namespace prosper::host

#endif
