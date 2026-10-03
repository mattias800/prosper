// #4349: a mapping lease alone excludes neither Windows lazy commitment nor private fallback.
// Real registered HLE maps must authenticate their section backing before CPU readers copy it.
// No game, Vulkan or source touch is needed to obtain a positive fault-safety observation.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "host/memory/committed_section.hpp"

#include <windows.h>
#include <gtest/gtest.h>

namespace prosper {
namespace {

class WindowsDirectFaultSafety : public ::testing::Test {
protected:
    static constexpr uint64_t page = 0x4000;
    HleFn allocate = nullptr, map = nullptr, reserve = nullptr, flexible = nullptr;
    HleFn protect = nullptr, unmap = nullptr, release = nullptr;
    uint64_t physical = 0, source = 0, alias = 0, reserved = 0, flex = 0;

    void SetUp() override {
        register_builtin_hle();
        allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
        map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
        reserve = Hle::lookup(nid_hash("sceKernelReserveVirtualRange"));
        flexible = Hle::lookup(nid_hash("sceKernelMapNamedFlexibleMemory"));
        protect = Hle::lookup(nid_hash("sceKernelMprotect"));
        unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
        release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
        ASSERT_TRUE(allocate && map && reserve && flexible && protect && unmap && release);
        ASSERT_EQ(allocate(0, 0x200000000ull, 4 * page, 4 * page, 0,
                           reinterpret_cast<uint64_t>(&physical)),
                  0u);
        ASSERT_NE(physical, 0u);
        // Nonzero physical offset exercises the exact view equation, including MapViewOfFileEx's
        // native allocation-base delta on hosts using that fallback rather than placeholders.
        ASSERT_EQ(map(reinterpret_cast<uint64_t>(&source), 2 * page, 3, 0, physical + page, page),
                  0u);
        ASSERT_NE(source, 0u);
    }

    void TearDown() override {
        if (source && unmap) EXPECT_EQ(unmap(source, 2 * page, 0, 0, 0, 0), 0u);
        if (alias && unmap) EXPECT_EQ(unmap(alias, 2 * page, 0, 0, 0, 0), 0u);
        if (reserved && unmap) EXPECT_EQ(unmap(reserved, 2 * page, 0, 0, 0, 0), 0u);
        if (flex && unmap) EXPECT_EQ(unmap(flex, page, 0, 0, 0, 0), 0u);
        if (physical && release) EXPECT_EQ(release(physical, 4 * page, 0, 0, 0, 0), 0u);
    }

    static bool safe(uint64_t address, uint64_t bytes) {
        GuestMappingLease lease;
        return guest_memory_direct_range_fault_safe(lease, address, bytes);
    }
};

TEST_F(WindowsDirectFaultSafety, AlreadyCommittedUntouchedViewAndProtectionSplit) {
    ASSERT_TRUE(safe(source, 2 * page)) << "admission must not require first-touch commitment";
    EXPECT_TRUE(safe(source + 7u, 32u));
    EXPECT_TRUE(safe(source + 2 * page - 1u, 1u));
    ASSERT_EQ(protect(source + page, page, 1, 0, 0, 0), 0u);
    EXPECT_TRUE(safe(source, 2 * page))
        << "one view can span compatible tracking/protection splits";
    EXPECT_EQ(*reinterpret_cast<const uint8_t*>(source), 0u);
    EXPECT_EQ(*reinterpret_cast<const uint8_t*>(source + page), 0u);
}

TEST_F(WindowsDirectFaultSafety, NullEmptyOverflowAndViewBoundaryRefuse) {
    EXPECT_FALSE(safe(0, 1));
    EXPECT_FALSE(safe(source, 0));
    EXPECT_FALSE(safe(source, UINT64_MAX));
    EXPECT_FALSE(safe(UINT64_MAX - 1u, 4u));
    EXPECT_FALSE(safe(source + 2 * page, 1));
    EXPECT_FALSE(safe(source + 2 * page - 1u, 2));
    EXPECT_TRUE(safe(source, 2 * page));
}

TEST_F(WindowsDirectFaultSafety, ReservationAndFlexibleBackingRefuse) {
    ASSERT_EQ(reserve(reinterpret_cast<uint64_t>(&reserved), 2 * page, 0, page, 0, 0), 0u);
    ASSERT_EQ(flexible(reinterpret_cast<uint64_t>(&flex), page, 3, 0, 0, 0), 0u);
    EXPECT_FALSE(safe(reserved, page));
    EXPECT_FALSE(safe(source + 2 * page - 1u, 2u));
    EXPECT_FALSE(safe(flex, page));
    EXPECT_TRUE(safe(source, page));
}

TEST_F(WindowsDirectFaultSafety, UntrackedNativeCommitDoesNotAuthenticateDirectBacking) {
    void* private_memory = VirtualAlloc(nullptr, page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ASSERT_NE(private_memory, nullptr);
    EXPECT_FALSE(safe(reinterpret_cast<uintptr_t>(private_memory), page));
    EXPECT_TRUE(VirtualFree(private_memory, 0, MEM_RELEASE));

    HANDLE section =
        CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, page, nullptr);
    ASSERT_NE(section, nullptr);
    void* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, page);
    if (view) {
        EXPECT_TRUE(host::committed_section_range(reinterpret_cast<uintptr_t>(view), page,
                                                  reinterpret_cast<uintptr_t>(view)));
        EXPECT_FALSE(safe(reinterpret_cast<uintptr_t>(view), page));
        EXPECT_FALSE(host::committed_section_range(reinterpret_cast<uintptr_t>(view), page,
                                                   reinterpret_cast<uintptr_t>(view) + page));
        EXPECT_TRUE(UnmapViewOfFile(view));
    }
    EXPECT_NE(view, nullptr);
    EXPECT_TRUE(CloseHandle(section));
}

TEST_F(WindowsDirectFaultSafety, GuardAndNoAccessRefuseWithoutConsumingOrRepairingProtection) {
    DWORD old = 0;
    ASSERT_TRUE(VirtualProtect(reinterpret_cast<void*>(source + page), page,
                               PAGE_READWRITE | PAGE_GUARD, &old));
    EXPECT_FALSE(safe(source + page, 1u));
    EXPECT_FALSE(safe(source, 2 * page))
        << "the full range must be checked, not just its first region";
    EXPECT_TRUE(safe(source, page));
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(VirtualQuery(reinterpret_cast<void*>(source + page), &info, sizeof(info)), 0u);
    EXPECT_NE(info.Protect & PAGE_GUARD, 0u) << "observation must not touch/consume the guard";
    DWORD ignored = 0;
    EXPECT_TRUE(VirtualProtect(reinterpret_cast<void*>(source + page), page, old, &ignored));
    ASSERT_EQ(protect(source + page, page, 0, 0, 0, 0), 0u);
    EXPECT_FALSE(safe(source, 2 * page));
    ASSERT_EQ(protect(source + page, page, 3, 0, 0, 0), 0u);
    EXPECT_TRUE(safe(source, 2 * page));
}

TEST_F(WindowsDirectFaultSafety, CopyOnWriteNativeProtectionCannotBorrowSharedAuthority) {
    DWORD old = 0;
    ASSERT_TRUE(VirtualProtect(reinterpret_cast<void*>(source), page, PAGE_WRITECOPY, &old));
    EXPECT_FALSE(safe(source, page));
    MEMORY_BASIC_INFORMATION info{};
    ASSERT_NE(VirtualQuery(reinterpret_cast<void*>(source), &info, sizeof(info)), 0u);
    EXPECT_EQ(info.Type, static_cast<DWORD>(MEM_MAPPED));
    EXPECT_EQ(info.Protect & 0xffu, static_cast<DWORD>(PAGE_WRITECOPY));
    DWORD ignored = 0;
    EXPECT_TRUE(VirtualProtect(reinterpret_cast<void*>(source), page, old, &ignored));
    EXPECT_TRUE(safe(source, page)) << "this fixture never touches or privatizes the COW page";
}

TEST_F(WindowsDirectFaultSafety, ReleasedPhysicalOriginRevokesStillMappedView) {
    ASSERT_TRUE(safe(source, page));
    ASSERT_EQ(release(physical, 4 * page, 0, 0, 0, 0), 0u);
    physical = 0;
    EXPECT_FALSE(safe(source, page));
}

TEST_F(WindowsDirectFaultSafety, FaultSafetyDoesNotGrantPhysicalAliasIsolation) {
    ASSERT_EQ(map(reinterpret_cast<uint64_t>(&alias), 2 * page, 3, 0, physical + page, page), 0u);
    ASSERT_NE(alias, source);
    ASSERT_TRUE(safe(source, page));
    ASSERT_TRUE(safe(alias, page));
    EXPECT_EQ(guest_memory_topology_relation(source, page, alias, page),
              GuestMemoryTopologyRelation::Overlap);
}

}   // namespace
}   // namespace prosper
