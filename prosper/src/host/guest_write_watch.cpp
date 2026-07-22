#include "guest_write_watch.hpp"

#include <atomic>
#include <utility>

#ifdef _WIN32

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
    std::atomic<uint64_t> create_protect_failures{0};
    std::atomic<uint64_t> queries{0};
    std::atomic<uint64_t> unchanged{0};
    std::atomic<uint64_t> dirty{0};
    std::atomic<uint64_t> unknown{0};
    std::atomic<uint64_t> faults{0};
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
    (void)size;
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
        value.create_protect_failures.load(std::memory_order_relaxed),
        value.queries.load(std::memory_order_relaxed),
        value.unchanged.load(std::memory_order_relaxed),
        value.dirty.load(std::memory_order_relaxed),
        value.unknown.load(std::memory_order_relaxed),
        value.faults.load(std::memory_order_relaxed),
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

bool guest_write_watch_handle_fault(uint64_t addr) {
    (void)addr;
    return false;
}

void guest_write_watch_set_fault_onstack(bool) {}   // Windows never arms page-protection watches

} // namespace prosper::host

#else   // ---- Linux / macOS: real page-protection dirty-tracking (mprotect + SIGSEGV) --------------

#include <sys/mman.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
struct Registration { std::vector<RegistrationPage> pages; };

struct WatchState {
    std::mutex mutex;
    uint64_t next_id = 0;
    std::vector<AliasRange> aliases;                                       // live dmem topology
    std::unordered_map<uint64_t, std::unique_ptr<WatchedPage>> pages_by_phys;
    std::unordered_map<uint64_t, WatchedPage*> pages_by_addr;              // page-aligned VA -> page
    std::unordered_map<uint64_t, Registration> registrations;
    std::atomic<bool> fault_onstack{false};                               // red-zone-safe gate
};
WatchState& state() { static WatchState* value = new WatchState; return *value; }

struct AtomicStats {
    std::atomic<uint64_t> create_attempts{0}, registrations{0}, registered_pages{0},
        create_no_mapping{0}, create_incomplete_aliases{0}, create_protect_failures{0},
        queries{0}, unchanged{0}, dirty{0}, unknown{0}, faults{0}, physical_writes{0}, rearms{0};
};
AtomicStats& stats() { static AtomicStats* value = new AtomicStats; return *value; }
inline void bump(std::atomic<uint64_t>& c) { c.fetch_add(1, std::memory_order_relaxed); }

const AliasRange* alias_at(const WatchState& w, uint64_t addr) {
    for (const AliasRange& a : w.aliases)
        if (addr >= a.addr && addr < a.addr + a.size) return &a;
    return nullptr;
}
// All guest VAs currently mapping `phys` (page-granular). Returns false (leaving `out` empty) if the
// page is unmapped or any alias is inaccessible (PROT_NONE) -> caller falls back to Unknown.
bool collect_aliases(const WatchState& w, uint64_t phys, std::vector<PageAlias>& out) {
    for (const AliasRange& a : w.aliases) {
        if (phys < a.phys || phys + kPage > a.phys + a.size) continue;
        if (a.prot == 0) return false;                    // PROT_NONE alias: cannot safely protect
        out.push_back({a.addr + (phys - a.phys), a.prot});
    }
    return !out.empty();
}

// mprotect every writable alias of `pages` to read-only (arm) or back to its guest prot (disarm).
// Read-only / PROT_NONE aliases are left untouched (a CPU store can't dirty them). Returns false and
// rolls back on the first mprotect failure so the guest is never left with a wrong protection.
bool set_pages_armed(const std::vector<WatchedPage*>& pages, bool arm) {
    std::vector<std::pair<uint64_t, int>> done;   // (addr, prot-to-restore-on-rollback)
    auto rollback = [&] {
        for (auto it = done.rbegin(); it != done.rend(); ++it)
            mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(it->first)), kPage, it->second);
    };
    for (WatchedPage* page : pages) {
        if (!page || page->armed == arm) continue;
        for (const PageAlias& al : page->aliases) {
            if (!cpu_writable(al.prot)) continue;
            const int full = host_prot(al.prot);
            const int want = arm ? (full & ~PROT_WRITE) : full;
            const int back = arm ? full : (full & ~PROT_WRITE);
            if (mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(al.addr)), kPage, want) != 0) {
                rollback();
                return false;
            }
            done.push_back({al.addr, back});
        }
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
    set_pages_armed(released, false);
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
    set_pages_armed(hit, false);
    for (WatchedPage* page : hit) page->generation++;
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
    const uint64_t end = (addr + size + kPage - 1) & ~(kPage - 1);
    std::lock_guard lock(w.mutex);

    // First pass: resolve every guest page to a physical page whose full alias set is known. Any gap
    // (non-dmem heap, PROT_NONE alias, unmapped) aborts to Unknown WITHOUT touching protections.
    struct Resolved { uint64_t phys; std::vector<PageAlias> aliases; };
    std::vector<Resolved> resolved;
    for (uint64_t va = begin; va < end; va += kPage) {
        const AliasRange* a = alias_at(w, va);
        if (!a) { bump(stats().create_no_mapping); return {}; }
        const uint64_t phys = a->phys + (va - a->addr);
        std::vector<PageAlias> al;
        if (!collect_aliases(w, phys, al)) { bump(stats().create_incomplete_aliases); return {}; }
        resolved.push_back({phys, std::move(al)});
    }

    // Second pass: get-or-create the WatchedPage per phys, arm newly-referenced pages, build the reg.
    const uint64_t id = ++w.next_id;
    Registration reg;
    std::vector<WatchedPage*> to_arm;
    for (Resolved& r : resolved) {
        auto it = w.pages_by_phys.find(r.phys);
        WatchedPage* page;
        if (it == w.pages_by_phys.end()) {
            auto up = std::make_unique<WatchedPage>();
            up->phys = r.phys; up->aliases = std::move(r.aliases);
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
    if (!set_pages_armed(to_arm, true)) {
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
    if (!set_pages_armed(pages, true)) return false;   // couldn't re-protect -> caller re-creates
    for (RegistrationPage& rp : found->second.pages) if (rp.page) rp.generation = rp.page->generation;
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
            v.create_protect_failures.load(), v.queries.load(), v.unchanged.load(), v.dirty.load(),
            v.unknown.load(), v.faults.load(), v.physical_writes.load(), v.rearms.load()};
}

void guest_write_watch_set_fault_onstack(bool on_altstack) {
    state().fault_onstack.store(on_altstack, std::memory_order_release);
}

void guest_write_watch_notify_direct_mapping_added(uint64_t addr, uint64_t size, uint64_t phys,
                                                    uint32_t protection) {
    if (!addr || !size || addr > UINT64_MAX - size || phys > UINT64_MAX - size) return;
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    // A new mapping over a watched phys range changes topology -> invalidate; then record the alias.
    invalidate_phys_range_locked(w, phys, phys + size);
    w.aliases.push_back({addr, size, phys, protection});
}

void guest_write_watch_notify_direct_mapping_removed(uint64_t addr, uint64_t size) {
    if (!addr || !size) return;
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    const uint64_t end = addr + size;
    std::vector<uint64_t> dead_phys;
    for (const AliasRange& a : w.aliases)
        if (a.addr < end && a.addr + a.size > addr) dead_phys.push_back(a.phys);
    for (uint64_t phys : dead_phys) invalidate_phys_range_locked(w, phys, phys + kPage);
    w.aliases.erase(std::remove_if(w.aliases.begin(), w.aliases.end(),
                                   [&](const AliasRange& a) {
                                       return a.addr < end && a.addr + a.size > addr;
                                   }),
                    w.aliases.end());
    // Pages whose every alias just vanished can never fault again; drop them from the addr index so a
    // stale mprotect can't linger (their generation was already bumped above -> queries read Dirty).
}

void guest_write_watch_notify_direct_mapping_protection(uint64_t addr, uint64_t size,
                                                        uint32_t protection) {
    if (!addr || !size) return;
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    const uint64_t end = addr + size;
    for (AliasRange& a : w.aliases)
        if (a.addr < end && a.addr + a.size > addr) {
            a.prot = protection;
            invalidate_phys_range_locked(w, a.phys, a.phys + a.size);
        }
}

void guest_write_watch_notify_physical_write(uint64_t phys, uint64_t size) {
    if (!size) return;
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    bump(stats().physical_writes);
    invalidate_phys_range_locked(w, phys, phys + size);
}

void guest_write_watch_invalidate_all() {
    WatchState& w = state();
    std::lock_guard lock(w.mutex);
    std::vector<WatchedPage*> all;
    all.reserve(w.pages_by_phys.size());
    for (auto& [phys, page] : w.pages_by_phys) { (void)phys; all.push_back(page.get()); }
    set_pages_armed(all, false);
    for (WatchedPage* page : all) page->generation++;
}

// Called from the SIGSEGV handler (exec_image_linux fault_handler) for every write fault. Returns true
// ONLY for an address we armed: it disarms that physical page's aliases (so the store re-executes and
// succeeds) and bumps the generation so the owning registration reads Dirty. Returns false otherwise,
// so a genuine guest fault is untouched. Must be async-signal-safe: a plain mutex is not, so this uses
// try_lock and, on contention, leaves the page armed (the store re-faults and retries) rather than
// blocking in signal context.
bool guest_write_watch_handle_fault(uint64_t addr) {
    WatchState& w = state();
    if (!w.fault_onstack.load(std::memory_order_acquire)) return false;
    std::unique_lock<std::mutex> lock(w.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return false;   // contended: not ours to resolve right now, let it re-fault
    auto it = w.pages_by_addr.find(addr & ~(kPage - 1));
    if (it == w.pages_by_addr.end() || !it->second || !it->second->armed) return false;
    WatchedPage* page = it->second;
    // Allocation-free (signal context): restore write directly on the page's aliases. mprotect is a
    // syscall; the alias vector is already allocated and mutation-protected by the held lock. Do NOT
    // call set_pages_armed here — its rollback vector would malloc, which can deadlock a thread caught
    // mid-allocation. If an mprotect fails we still mark dirty; the worst case is a stale RO alias that
    // re-faults into this same path.
    for (const PageAlias& al : page->aliases)
        if (cpu_writable(al.prot))
            mprotect(reinterpret_cast<void*>(static_cast<uintptr_t>(al.addr)), kPage, host_prot(al.prot));
    page->armed = false;
    page->generation++;
    bump(stats().faults);
    return true;
}

} // namespace prosper::host

#endif
