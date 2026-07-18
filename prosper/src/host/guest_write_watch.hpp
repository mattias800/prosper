#pragma once

#include <cstddef>
#include <cstdint>

namespace prosper::host {

enum class GuestWriteWatchQuery {
    Unchanged,
    Dirty,
    Unknown,
};

struct GuestWriteWatchStats {
    uint64_t create_attempts = 0;
    uint64_t registrations = 0;
    uint64_t registered_pages = 0;
    uint64_t create_no_mapping = 0;
    uint64_t create_incomplete_aliases = 0;
    uint64_t create_protect_failures = 0;
    uint64_t queries = 0;
    uint64_t unchanged = 0;
    uint64_t dirty = 0;
    uint64_t unknown = 0;
    uint64_t faults = 0;
    uint64_t physical_writes = 0;
    uint64_t rearms = 0;
};

// Windows direct memory is section-backed so MEM_WRITE_WATCH cannot observe it. Page-protection
// watches are deliberately unsupported: Windows writes its exception-dispatch frame into the guest
// SysV red zone before a vectored handler can restore the page, corrupting valid guest locals.
// Callers receive Unknown and retain their exact byte-comparison fallback.
class GuestWriteWatch {
public:
    GuestWriteWatch() = default;
    ~GuestWriteWatch();
    GuestWriteWatch(const GuestWriteWatch&) = delete;
    GuestWriteWatch& operator=(const GuestWriteWatch&) = delete;
    GuestWriteWatch(GuestWriteWatch&& other) noexcept;
    GuestWriteWatch& operator=(GuestWriteWatch&& other) noexcept;

    static GuestWriteWatch create(uint64_t addr, uint64_t size);
    explicit operator bool() const { return id_ != 0; }
    GuestWriteWatchQuery query() const;
    bool rearm();
    void reset();

private:
    explicit GuestWriteWatch(uint64_t id) : id_(id) {}
    uint64_t id_ = 0;
};

GuestWriteWatchStats guest_write_watch_stats();

// Direct-memory mapping notifications. A topology/protection change invalidates existing watches;
// their next query is Unknown and the exact path may establish a fresh complete registration.
void guest_write_watch_notify_direct_mapping_added(uint64_t addr, uint64_t size, uint64_t phys,
                                                    uint32_t protection);
void guest_write_watch_notify_direct_mapping_removed(uint64_t addr, uint64_t size);
void guest_write_watch_notify_direct_mapping_protection(uint64_t addr, uint64_t size,
                                                        uint32_t protection);
void guest_write_watch_notify_physical_write(uint64_t phys, uint64_t size);
void guest_write_watch_invalidate_all();

// Retained for the platform-neutral VEH interface; Windows currently returns false because
// page-fault write watches are unsafe for SysV guest code.
bool guest_write_watch_handle_fault(uint64_t addr);

} // namespace prosper::host
