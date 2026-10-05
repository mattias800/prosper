#pragma once

#include <cstdint>

namespace prosper::host {

// Where the HOST's own memory sits relative to the guest's address range (#4426).
//
// The guest owns [16 GiB, 1008 GiB): fixed module bases from 16 GiB, automatic maps above 64 GiB.
// Code that takes a guest address also assumes, in places, that anything at or below 4 GiB is not
// one. So the host's own allocations belong in [4 GiB, 16 GiB) — above the values the guest's
// consumers read as immediates, below everything the guest places. Linux gets this for free (the
// kernel maps top-down from the far end); on Windows it takes the two halves below plus a link
// flag (`prosper_hosts_a_guest` in prosper/CMakeLists.txt).

// Reserve whatever is still free in [64 KiB, 4 GiB), so the process's later bottom-up allocations
// — heaps, and every thread stack created from here on, guest threads' included — start at 4 GiB.
// What already lives below 4 GiB stays there: with high-entropy ASLR off that is the main thread's
// stack and whatever the process allocated before this call. Idempotent; returns the bytes this
// call reserved, and 0 on a host that needs none of this.
uint64_t confine_host_allocations_above_4gib();

// "What is standing in this part of the guest's range?" — every host allocation that overlaps
// [lo, hi), and the largest free gap between them. An allocation is identified by its allocation
// base, so one host allocation is one occupant however many regions it is split into.
//
// The result carries its own scope, as every answer from this folder must: `available` is false on
// a host that cannot enumerate its address space this way (nothing was looked at), and `complete`
// is false when the walk stopped before `hi` (the gap and the count are then lower bounds).
struct GuestRangeOccupant {
    uint64_t base = 0, size = 0;
    uint32_t state = 0, type = 0; // the host's own MEM_* values, printed raw
};
struct GuestRangeOccupancy {
    static constexpr int kMaxShown = 8;
    bool available = false;
    bool complete = false;
    uint64_t largest_free_gap = 0;
    int occupants = 0;                   // all of them, including those not in `shown`
    GuestRangeOccupant shown[kMaxShown]; // the first min(occupants, kMaxShown), ascending
};
GuestRangeOccupancy query_guest_range_occupancy(uint64_t lo, uint64_t hi);

// A reservation of `len` bytes that could not be placed anywhere in [lo, hi) is fatal to the title
// and used to be silent: UE4 receives a null allocator, re-enters FMemory::GCreateMalloc from
// inside its own initializer and blocks forever on that function's __cxa_guard, before its first
// print. This names the cause on stderr — the room the range has and what is standing in it — once
// per process, and returns nullptr so the refusing call site can return it directly. Always on: it
// is the report of a guest-visible ENOMEM, not a trace.
void* report_unplaceable(uint64_t len, uint64_t lo, uint64_t hi);

} // namespace prosper::host
