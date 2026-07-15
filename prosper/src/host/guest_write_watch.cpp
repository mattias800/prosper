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
    if (!addr || !size || addr > UINT64_MAX - size) return {};
    const uint64_t begin = addr & ~(kPageSize - 1);
    const uint64_t range_end = addr + size;
    if (range_end > UINT64_MAX - (kPageSize - 1)) return {};
    const uint64_t end = (range_end + kPageSize - 1) & ~(kPageSize - 1);
    if (end <= begin) return {};

    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);

    Registration registration;
    registration.pages.reserve(static_cast<size_t>((end - begin) / kPageSize));
    auto rollback = [&] {
        for (auto it = registration.pages.rbegin(); it != registration.pages.rend(); ++it) {
            WatchedPage* page = it->page;
            if (!page || !page->references || --page->references) continue;
            if (page->armed) restore_page(*page);
            for (const PageAlias& alias : page->aliases) watch.pages_by_addr.erase(alias.addr);
            watch.pages_by_phys.erase(page->phys);
        }
        registration.pages.clear();
    };
    for (uint64_t page_addr = begin; page_addr < end; page_addr += kPageSize) {
        const AliasRange* source = alias_at(watch, page_addr);
        if (!source) {
            stats().create_no_mapping.fetch_add(1, std::memory_order_relaxed);
            log_create_failure("no-mapping", addr, size, page_addr);
            rollback();
            return {};
        }
        const uint64_t phys = source->phys + (page_addr - source->addr);
        auto found = watch.pages_by_phys.find(phys);
        WatchedPage* page = found == watch.pages_by_phys.end() ? nullptr : found->second.get();
        if (!page) {
            auto owned = std::make_unique<WatchedPage>();
            owned->phys = phys;
            if (!collect_aliases(watch, phys, owned->aliases)) {
                stats().create_incomplete_aliases.fetch_add(1, std::memory_order_relaxed);
                log_create_failure("incomplete-aliases", addr, size, page_addr);
                rollback();
                return {};
            }
            const bool source_alias_present = std::any_of(
                owned->aliases.begin(), owned->aliases.end(),
                [&](const PageAlias& alias) { return alias.addr == page_addr; });
            if (!source_alias_present) {
                stats().create_incomplete_aliases.fetch_add(1, std::memory_order_relaxed);
                log_create_failure("source-alias-missing", addr, size, page_addr);
                rollback();
                return {};
            }
            page = owned.get();
            watch.pages_by_phys.emplace(phys, std::move(owned));
            for (const PageAlias& alias : page->aliases)
                watch.pages_by_addr[alias.addr] = page;
        }
        ++page->references;
        registration.pages.push_back({page, page->generation});
    }

    std::vector<WatchedPage*> pages;
    pages.reserve(registration.pages.size());
    for (const RegistrationPage& registered : registration.pages)
        pages.push_back(registered.page);
    if (!set_pages_armed(pages, true)) {
        stats().create_protect_failures.fetch_add(1, std::memory_order_relaxed);
        log_create_failure("protect", addr, size, begin);
        rollback();
        return {};
    }

    uint64_t id = ++watch.next_id;
    if (!id) id = ++watch.next_id;
    watch.registrations.emplace(id, std::move(registration));
    stats().registrations.fetch_add(1, std::memory_order_relaxed);
    stats().registered_pages.fetch_add((end - begin) / kPageSize, std::memory_order_relaxed);
    return GuestWriteWatch(id);
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
    if (!id_) return false;
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    auto found = watch.registrations.find(id_);
    if (found == watch.registrations.end()) return false;
    std::vector<WatchedPage*> pages;
    pages.reserve(found->second.pages.size());
    for (const RegistrationPage& registered : found->second.pages)
        pages.push_back(registered.page);
    if (!set_pages_armed(pages, true)) return false;
    for (RegistrationPage& registered : found->second.pages) {
        if (!registered.page) return false;
        registered.generation = registered.page->generation;
    }
    stats().rearms.fetch_add(1, std::memory_order_relaxed);
    return true;
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
    WatchState& watch = state();
    std::lock_guard lock(watch.mutex);
    const uint64_t page_addr = addr & ~(kPageSize - 1);
    auto found = watch.pages_by_addr.find(page_addr);
    if (found == watch.pages_by_addr.end() || !found->second || !found->second->armed) return false;
    WatchedPage& page = *found->second;
    ++page.generation;
    restore_page(page);
    stats().faults.fetch_add(1, std::memory_order_relaxed);
    return true;
}

} // namespace prosper::host

#else

namespace prosper::host {

GuestWriteWatch::~GuestWriteWatch() = default;
GuestWriteWatch::GuestWriteWatch(GuestWriteWatch&& other) noexcept
    : id_(std::exchange(other.id_, 0)) {}
GuestWriteWatch& GuestWriteWatch::operator=(GuestWriteWatch&& other) noexcept {
    if (this != &other) id_ = std::exchange(other.id_, 0);
    return *this;
}
GuestWriteWatch GuestWriteWatch::create(uint64_t, uint64_t) { return {}; }
GuestWriteWatchQuery GuestWriteWatch::query() const { return GuestWriteWatchQuery::Unknown; }
bool GuestWriteWatch::rearm() { return false; }
void GuestWriteWatch::reset() { id_ = 0; }
GuestWriteWatchStats guest_write_watch_stats() { return {}; }
void guest_write_watch_notify_direct_mapping_added(uint64_t, uint64_t, uint64_t, uint32_t) {}
void guest_write_watch_notify_direct_mapping_removed(uint64_t, uint64_t) {}
void guest_write_watch_notify_direct_mapping_protection(uint64_t, uint64_t, uint32_t) {}
void guest_write_watch_notify_physical_write(uint64_t, uint64_t) {}
void guest_write_watch_invalidate_all() {}
bool guest_write_watch_handle_fault(uint64_t) { return false; }

} // namespace prosper::host

#endif
