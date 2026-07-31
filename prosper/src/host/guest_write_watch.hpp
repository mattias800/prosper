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
    uint64_t create_oversized = 0;
    uint64_t create_bytes_le_1m = 0;
    uint64_t create_bytes_le_8m = 0;
    uint64_t create_bytes_le_32m = 0;
    uint64_t create_bytes_gt_32m = 0;
    uint64_t create_protect_failures = 0;
    uint64_t queries = 0;
    uint64_t unchanged = 0;
    uint64_t dirty = 0;
    uint64_t unknown = 0;
    uint64_t faults = 0;
    uint64_t stale_faults = 0;
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

// Called by the HLE, by guest VA range, immediately BEFORE a host/kernel store into guest memory (e.g. a
// read()/pread() that streams bytes straight into a guest dmem buffer). Restores write on any armed pages
// the range overlaps and marks them Dirty. No-op on Windows/macOS (no pages are ever armed there).
void guest_write_watch_notify_host_write(uint64_t addr, uint64_t size);

// Device/DMA writes already carry an exact guest VA range. Mark only registrations whose logical
// source overlaps that range; unlike a CPU protection fault, an adjacent write on the same host page
// need not create a false dirty result.
void guest_write_watch_notify_gpu_write(uint64_t addr, uint64_t size);

// Retained for the platform-neutral VEH interface; Windows currently returns false because
// page-fault write watches are unsafe for SysV guest code.
bool guest_write_watch_handle_fault(uint64_t addr);

// The Linux page-protection watch (mprotect + SIGSEGV) is only red-zone-safe when the SIGSEGV handler
// runs on a sigaltstack (SA_ONSTACK): without it the kernel writes the signal frame into the guest's
// SysV red zone, corrupting live locals of the faulting store. exec_image_linux reports whether the
// default-on handler has SA_ONSTACK; a diagnostic opt-out makes `create` refuse to arm so callers
// retain the exact byte-comparison fallback. No-op on other platforms.
void guest_write_watch_set_fault_onstack(bool on_altstack);

} // namespace prosper::host
