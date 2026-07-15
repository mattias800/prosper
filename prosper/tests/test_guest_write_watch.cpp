#include "host/guest_write_watch.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>

using prosper::host::GuestWriteWatch;
using prosper::host::GuestWriteWatchQuery;

namespace {
int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); ++failures; } \
} while (0)

LONG CALLBACK watch_veh(EXCEPTION_POINTERS* exception) {
    const EXCEPTION_RECORD* record = exception->ExceptionRecord;
    if (record->ExceptionCode == EXCEPTION_ACCESS_VIOLATION &&
        record->NumberParameters >= 2 && record->ExceptionInformation[0] == 1 &&
        prosper::host::guest_write_watch_handle_fault(
            static_cast<uint64_t>(record->ExceptionInformation[1])))
        return EXCEPTION_CONTINUE_EXECUTION;
    return EXCEPTION_CONTINUE_SEARCH;
}
} // namespace

int main() {
    constexpr uint64_t kSize = 0x10000;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                        0, static_cast<DWORD>(kSize), nullptr);
    CHECK(section != nullptr, "paging-file section created");
    if (!section) return 1;
    auto* first = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kSize));
    auto* second = static_cast<uint8_t*>(MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, kSize));
    CHECK(first && second, "two aliases mapped");
    if (!first || !second) {
        if (first) UnmapViewOfFile(first);
        if (second) UnmapViewOfFile(second);
        CloseHandle(section);
        return 1;
    }

    void* handler = AddVectoredExceptionHandler(1, watch_veh);
    CHECK(handler != nullptr, "test write-watch VEH installed");
    constexpr uint64_t kPhys = 0x10000000;
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(first), kSize, kPhys, PAGE_READWRITE);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(second), kSize, kPhys, PAGE_READWRITE);

    first[0x180] = 0x11;
    GuestWriteWatch watch = GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(first + 0x100), 0x2100);
    CHECK(static_cast<bool>(watch), "section-backed range is watchable");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged,
          "new registration starts unchanged");

    second[0x180] = 0x22;
    CHECK(first[0x180] == 0x22, "write through second alias updates first alias");
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
          "write through any physical alias dirties the registration");
    CHECK(watch.rearm(), "dirty registration rearms after exact validation");
    CHECK(watch.query() == GuestWriteWatchQuery::Unchanged,
          "rearmed registration has a fresh generation");

    first[0x1180] = 0x33;
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
          "write through the original alias dirties another covered page");
    CHECK(watch.rearm(), "registration rearms before a host physical write");
    prosper::host::guest_write_watch_notify_physical_write(kPhys + 0x1000, 0x1000);
    CHECK(watch.query() == GuestWriteWatchQuery::Dirty,
          "temporary physical-section writes dirty overlapping registrations");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(second), kSize);
    CHECK(watch.query() == GuestWriteWatchQuery::Unknown,
          "alias topology changes invalidate existing registrations");
    watch.reset();

    first[0x180] = 0x44;
    CHECK(second[0x180] == 0x44, "invalidation restores writable page protections");

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(first), kSize);
    if (handler) RemoveVectoredExceptionHandler(handler);
    UnmapViewOfFile(second);
    UnmapViewOfFile(first);
    CloseHandle(section);

    const auto stats = prosper::host::guest_write_watch_stats();
    CHECK(stats.faults >= 2, "write faults were handled");
    CHECK(stats.rearms >= 1, "rearm was counted");
    CHECK(stats.physical_writes >= 1, "physical write was counted");
    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}
