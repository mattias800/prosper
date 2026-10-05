// test_windows_guest_window — on Windows the guest's address range must start out EMPTY in a
// process that boots a guest, the host's later allocations must stay above 4 GiB and below that
// range, and a huge reservation that cannot be placed must say why (#4426).
//
// The defect: every prosper executable was linked with IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA,
// under which Windows picks the base of the process's bottom-up allocations (main-thread stack,
// PEB/TEBs, heaps, then every later VirtualAlloc(NULL)) from the low terabyte. The guest's range
// IS that terabyte, so the host's own allocations sat inside it on almost every launch. A UE4
// title's 512 GiB MallocBinned3 arena must cover [496 GiB, 640 GiB) wherever it lands in the
// 880 GiB huge band; with the host cluster there the reservation fails, the title gets a null
// allocator and deadlocks in FMemory::GCreateMalloc before its first print. Measured on a UE4
// title: 6 of 28 launches hung this way, all six with the host's stacks inside that span (and
// outside it in every booted launch that was checked), against 0 of 12 with only this flag cleared
// in the same binary.
//
// A wrong value is harmful because nothing else notices: the reservation is the title's first
// allocation, the failure is silent, and the symptom (a hang with no output) names nothing.
//
// This executable is linked the way a guest-hosting one is (prosper_hosts_a_guest in
// prosper/CMakeLists.txt), so what it measures about its own address space is what prosper-app,
// boot_trace and screenshot get. Windows-only by construction — on Linux the kernel maps top-down
// from the 128 TiB end, nowhere near the range, which is why the same titles never showed this.
#include <gtest/gtest.h>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "host/memory/host_address_layout.hpp"

using namespace prosper;

namespace {

// Mirrors boot_program.hpp / hle_kernel_mem.cpp. The lowest fixed guest address is the direct-memory
// aperture at 16 GiB (below the module bases at 0x410000000); the range ends at the guest libc's
// accepted ceiling. Spelled out rather than included so a silent change to either constant cannot
// move this test's subject without somebody reading it.
constexpr uint64_t kFourGiB = 0x100000000ull;
constexpr uint64_t kGuestRangeLo = 0x400000000ull;   // 16 GiB
constexpr uint64_t kGuestRangeHi = 0xfc00000000ull;   // 1008 GiB, exclusive
constexpr uint64_t kHugeBandLo = 0x2000000000ull;   // 128 GiB: floor of a huge hinted reserve
constexpr uint64_t kArenaHint = 0x1000000000ull;   // UE4's live hint (DQ VII, Kena)
constexpr uint64_t kArenaLen = 0x8000000000ull;   // 512 GiB: the live MallocBinned3 arena
constexpr uint64_t kArenaAlign = 0x200000ull;
// Every 512 GiB span inside [128 GiB, 1008 GiB) contains [496 GiB, 640 GiB). One occupant anywhere
// in it makes the arena unplaceable; this address is its midpoint.
constexpr uint64_t kMandatoryMid = 0x8e00000000ull;   // 568 GiB
constexpr uint64_t kEnomem = 0x8002000cull;

std::string describe(const host::GuestRangeOccupancy& o) {
    char line[200];
    std::snprintf(
        line, sizeof line, "%d occupant(s), first base=0x%llx size=0x%llx state=0x%lx type=0x%lx",
        o.occupants, (unsigned long long)o.shown[0].base, (unsigned long long)o.shown[0].size,
        (unsigned long)o.shown[0].state, (unsigned long)o.shown[0].type);
    return line;
}

// DllCharacteristics of a PE32+ image, from its bytes. Returns false when they are not one.
bool pe_dll_characteristics(const uint8_t* image, size_t size, uint16_t* out) {
    if (size < sizeof(IMAGE_DOS_HEADER)) return false;
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(image);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew < 0) return false;
    if (static_cast<size_t>(dos->e_lfanew) + sizeof(IMAGE_NT_HEADERS64) > size) return false;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(image + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) return false;
    if (nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return false;
    *out = nt->OptionalHeader.DllCharacteristics;
    return true;
}

std::vector<std::string> split_paths(const char* joined) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = joined; *p; ++p) {
        if (*p == '|') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += *p;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

}   // namespace

// The link-time fact, for this executable and for every one that boots a guest. Deterministic in
// both directions: red on every run for an executable that has the flag. The list comes from
// CMake (the end of prosper/CMakeLists.txt) and names targets, not prosper_hosts_a_guest() calls,
// so an executable that DROPS its call is still read here.
TEST(WindowsGuestWindow, GuestHostingExecutablesOptOutOfHighEntropyVa) {
    uint16_t own = 0;
    ASSERT_TRUE(pe_dll_characteristics(reinterpret_cast<const uint8_t*>(GetModuleHandleW(nullptr)),
                                       0x1000, &own));
    EXPECT_EQ(own & IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA, 0)
        << "this test is not linked the way a guest-hosting executable is; the two address-space "
           "cases below would then measure a layout no title runs in";

    const std::vector<std::string> hosts = split_paths(PROSPER_GUEST_HOST_FILES);
    ASSERT_GE(hosts.size(), 1u) << "no guest-hosting executable was named: an empty list passes "
                                   "nothing, and must not read as every executable passing";
    for (const std::string& path : hosts) {
        std::ifstream f(path, std::ios::binary);
        ASSERT_TRUE(f.good()) << "cannot open " << path;
        std::vector<uint8_t> head(0x1000);
        f.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
        uint16_t dc = 0;
        ASSERT_TRUE(pe_dll_characteristics(head.data(), static_cast<size_t>(f.gcount()), &dc))
            << path << " is not a PE32+ image";
        EXPECT_EQ(dc & IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA, 0)
            << path
            << " boots a guest and is linked with HIGH_ENTROPY_VA: Windows will scatter "
               "its stacks and heaps across the guest's address range. Link it through "
               "prosper_hosts_a_guest() (prosper/CMakeLists.txt, #4426)";
    }
}

// The property the link flag exists to buy, measured on the address space itself rather than read
// back from the header: nothing the host process owns sits in the guest's range. This is the
// independent arm — it would stay red if some other mechanism put host memory there with the flag
// correctly cleared. With the flag set it is red on most runs, not all (26 of 30 measured): the
// draw sometimes lands the host below 16 GiB, which is why the header test above exists too.
TEST(WindowsGuestWindow, GuestRangeStartsFree) {
    const host::GuestRangeOccupancy o =
        host::query_guest_range_occupancy(kGuestRangeLo, kGuestRangeHi);
    ASSERT_TRUE(o.available && o.complete) << "the walk did not cover the range";
    EXPECT_EQ(o.occupants, 0) << "host allocations sit inside the guest range [16 GiB, 1008 GiB): "
                              << describe(o);
}

// Clearing the flag moves the host's bottom-up allocations to the very bottom of the address
// space, below 4 GiB — and code that takes guest addresses tells an address from an immediate by
// "above 4 GiB" in places (a DMA_DATA source, for one). A guest thread's stack is a host
// allocation, so left alone every guest stack pointer would have become a value those consumers
// read as an immediate. confine_host_allocations_above_4gib() is what boot_program() runs first;
// after it, a fresh allocation and a new thread's stack both land in [4 GiB, 16 GiB).
//
// The "before" arm is not decoration: it shows this process really is in the low regime the call
// exists for, so the "after" arm cannot pass by the allocations having been high all along.
TEST(WindowsGuestWindow, HostAllocationsAreConfinedAbove4GiB) {
    void* before = VirtualAlloc(nullptr, 0x100000, MEM_RESERVE, PAGE_NOACCESS);
    ASSERT_NE(before, nullptr);
    ASSERT_LT(reinterpret_cast<uintptr_t>(before), kFourGiB)
        << "precondition: without the confinement this process allocates below 4 GiB";

    EXPECT_GT(host::confine_host_allocations_above_4gib(), 0u);
    EXPECT_EQ(host::confine_host_allocations_above_4gib(), 0u) << "the call must be idempotent";

    void* after = VirtualAlloc(nullptr, 0x100000, MEM_RESERVE, PAGE_NOACCESS);
    ASSERT_NE(after, nullptr);
    const uintptr_t after_addr = reinterpret_cast<uintptr_t>(after);
    EXPECT_GE(after_addr, kFourGiB) << "a host allocation still landed below 4 GiB";
    EXPECT_LT(after_addr, kGuestRangeLo) << "a host allocation landed inside the guest range";

    uintptr_t stack_addr = 0;
    std::thread([&] {
        volatile int local = 0;
        stack_addr = reinterpret_cast<uintptr_t>(&local);
    }).join();
    EXPECT_GE(stack_addr, kFourGiB) << "a new thread's stack is below 4 GiB: a guest thread's "
                                       "stack pointers would read as immediates";
    EXPECT_LT(stack_addr, kGuestRangeLo) << "a new thread's stack is inside the guest range";

    const host::GuestRangeOccupancy o =
        host::query_guest_range_occupancy(kGuestRangeLo, kGuestRangeHi);
    EXPECT_TRUE(o.complete);
    EXPECT_EQ(o.occupants, 0) << "the confinement pushed host memory into the guest range: "
                              << describe(o);
    VirtualFree(before, 0, MEM_RELEASE);
    VirtualFree(after, 0, MEM_RELEASE);
}

// The product path, both ways. With one 64 KiB host allocation in the span every placement must
// cover, the live 512 GiB arena is refused with ENOMEM and the refusal names that allocation —
// this is the arm that was silent. With the allocation released, the IDENTICAL request succeeds,
// which is what shows the occupant, not the request, decided the first result.
//
// Occupied arm first, on a pristine range: sceKernelMunmap leaves a released span OS-reserved as a
// recyclable placeholder, so after any successful reserve the midpoint could no longer be handed
// to a foreign allocation and the occupied arm would not be constructible at all.
TEST(WindowsGuestWindow, UnplaceableHugeReserveNamesItsOccupant) {
    register_builtin_hle();
    auto reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
    auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    ASSERT_TRUE(reserve && unmap);
    const host::GuestRangeOccupancy before =
        host::query_guest_range_occupancy(kHugeBandLo, kGuestRangeHi);
    ASSERT_TRUE(before.complete && before.occupants == 0)
        << "precondition: the huge band must start free -- " << describe(before);

    void* blocker = VirtualAlloc(reinterpret_cast<void*>(static_cast<uintptr_t>(kMandatoryMid)),
                                 0x10000, MEM_RESERVE, PAGE_NOACCESS);
    ASSERT_NE(blocker, nullptr) << "could not place the blocking allocation, GetLastError="
                                << GetLastError();
    uint64_t refused = kArenaHint;
    testing::internal::CaptureStderr();
    const uint64_t rc = reserve((uint64_t)&refused, kArenaLen, 0, kArenaAlign, 0, 0);
    const std::string err = testing::internal::GetCapturedStderr();
    EXPECT_EQ(rc, kEnomem)
        << "a reservation with an occupant in its mandatory span must be refused";
    EXPECT_NE(err.find("[memhle] reserve FAILED"), std::string::npos) << err;
    char want[64];
    std::snprintf(want, sizeof want, "occupant base=0x%llx", (unsigned long long)kMandatoryMid);
    EXPECT_NE(err.find(want), std::string::npos)
        << "the refusal must name the allocation that blocks it; got:\n"
        << err;

    ASSERT_TRUE(VirtualFree(blocker, 0, MEM_RELEASE));
    uint64_t base = kArenaHint;
    ASSERT_EQ(reserve((uint64_t)&base, kArenaLen, 0, kArenaAlign, 0, 0), 0u)
        << "control: the same request must succeed once the occupant is gone";
    EXPECT_GE(base, kHugeBandLo);
    EXPECT_LE(base + kArenaLen, kGuestRangeHi);
    EXPECT_EQ(base % kArenaAlign, 0u);
    EXPECT_EQ(unmap(base, kArenaLen, 0, 0, 0, 0), 0u);
}
