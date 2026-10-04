// A diagnostic raw load used to crash the host on an unmapped fence label. Unreadable
// and partial replacements must also retire the old sample, not report zero or stale data.
#include "gpu/diagnostics/fence_build_journal.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <thread>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

using namespace prosper::gpu;
namespace {
uint64_t address(const void* p) {
    return reinterpret_cast<uintptr_t>(p);
}
constexpr uint64_t packet = 0x210000;
FenceBuildRecord sentinel() {
    return {9, 8, 7, 6, 5};
}
void expect_missing(uint64_t pkt) {
    auto out = sentinel();
    EXPECT_FALSE(fence_build_journal_lookup(pkt, out));
    EXPECT_EQ(out.pkt, 9u);
    EXPECT_EQ(out.addr, 8u);
    EXPECT_EQ(out.pre, 7u);
    EXPECT_EQ(out.t_ms, 6u);
    EXPECT_EQ(out.fold, 5u) << "missing samples leave all outputs untouched";
}
class FenceBuildJournal : public testing::Test {
protected:
    void SetUp() override {
#ifdef _WIN32
        SYSTEM_INFO info{};
        GetSystemInfo(&info);
        page = info.dwPageSize;
        base = static_cast<uint8_t*>(
            VirtualAlloc(nullptr, 2 * page, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
        ASSERT_NE(base, nullptr);
        DWORD previous = 0;
        ASSERT_TRUE(VirtualProtect(base + page, page, PAGE_NOACCESS, &previous));
#else
        page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        void* p =
            mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT_NE(p, MAP_FAILED);
        base = static_cast<uint8_t*>(p);
        ASSERT_EQ(mprotect(base + page, page, PROT_NONE), 0);
#endif
        std::memset(base, 0x51, page);
    }
    void release() {
        if (!base) return;
#ifdef _WIN32
        EXPECT_TRUE(VirtualFree(base, 0, MEM_RELEASE));
#else
        EXPECT_EQ(munmap(base, 2 * page), 0);
#endif
        base = nullptr;
    }
    void TearDown() override { release(); }
    uint8_t* base = nullptr;
    size_t page = 0;
};
}   // namespace
TEST_F(FenceBuildJournal, ReadableValueAndMetadataAreObservedWithoutWrites) {
    constexpr uint64_t value = 0x123456789abcdef0;
    std::memcpy(base, &value, sizeof value);
    std::array<uint32_t, 4> words{1, 2, 3, 4};
    const auto original = words;
    fence_build_journal_record(address(words.data()), address(base), 1234, 56);
    FenceBuildRecord out;
    ASSERT_TRUE(fence_build_journal_lookup(address(words.data()), out));
    EXPECT_EQ(out.pkt, address(words.data()));
    EXPECT_EQ(out.addr, address(base));
    EXPECT_EQ(out.pre, value);
    EXPECT_EQ(out.t_ms, 1234u);
    EXPECT_EQ(out.fold, 56u);
    uint64_t unchanged = 0;
    std::memcpy(&unchanged, base, sizeof unchanged);
    EXPECT_EQ(unchanged, value) << "the observer cannot write the label";
    EXPECT_EQ(words, original) << "packet bytes remain the guest's own";
}
TEST_F(FenceBuildJournal, ReadableZeroIsAvailable) {
    std::memset(base, 0, 8);
    fence_build_journal_record(packet, address(base), 12, 3);
    FenceBuildRecord out;
    ASSERT_TRUE(fence_build_journal_lookup(packet, out));
    EXPECT_EQ(out.pre, 0u) << "a real zero is distinct from an unavailable sample";
}
TEST_F(FenceBuildJournal, UnmappedLabelDoesNotFault) {
    const auto unmapped = address(base);
#ifdef _WIN32
    // Releasing would let the next allocation in this process (the CRT heap, a lazily initialised
    // lock) be placed at this address, which then reads as zero-filled live memory: the Windows CI
    // runner read pre=0 here after MEM_RELEASE. Decommitting leaves the range reserved, so it has no
    // backing AND cannot be handed out again; TearDown still releases the reservation.
    ASSERT_TRUE(VirtualFree(base, 2 * page, MEM_DECOMMIT));
#else
    release();
#endif
    fence_build_journal_record(packet, unmapped, 12, 3);
    expect_missing(packet);
}
TEST_F(FenceBuildJournal, ProtectedLabelDoesNotFault) {
    fence_build_journal_record(packet, address(base + page), 12, 3);
    expect_missing(packet);
}
TEST_F(FenceBuildJournal, PartialSampleCannotBecomeObservedValue) {
    fence_build_journal_record(packet, address(base + page - 4), 12, 3);
    expect_missing(packet);
}
TEST_F(FenceBuildJournal, FailedReplacementRetiresPriorSample) {
    fence_build_journal_record(packet, address(base), 12, 3);
    FenceBuildRecord out;
    ASSERT_TRUE(fence_build_journal_lookup(packet, out));
    fence_build_journal_record(packet, address(base + page), 13, 4);
    expect_missing(packet);
}
TEST_F(FenceBuildJournal, CollisionReplacesOnlyThatPacket) {
    constexpr uint64_t colliding = packet + (65536ull << 2);
    fence_build_journal_record(packet, address(base), 12, 3);
    fence_build_journal_record(colliding, address(base), 13, 4);
    expect_missing(packet);
    FenceBuildRecord out;
    ASSERT_TRUE(fence_build_journal_lookup(colliding, out));
    EXPECT_EQ(out.pkt, colliding);
    EXPECT_EQ(out.t_ms, 13u);
    fence_build_journal_record(packet, address(base + page), 14, 5);
    expect_missing(colliding);
    expect_missing(packet);
}
TEST_F(FenceBuildJournal, InvalidAddressesDoNotInventZeroSamples) {
    for (uint64_t addr : std::array<uint64_t, 5>{0, 4, 0xffff, address(base) + 1, UINT64_MAX - 3}) {
        fence_build_journal_record(packet, address(base), 12, 3);
        fence_build_journal_record(packet, addr, 13, 4);
        expect_missing(packet);
    }
    fence_build_journal_record(0, address(base), 12, 3);
    expect_missing(0);
}
TEST_F(FenceBuildJournal, ConcurrentReplacementCannotTearMetadata) {
    const std::array<uint64_t, 2> values{0xaaaaaaaa12345678, 0xbbbbbbbb87654321};
    fence_build_journal_record(packet, address(&values[0]), 1, 11);
    std::thread writer([&] {
        for (unsigned i = 0; i < 1000; ++i) {
            const unsigned index = i & 1;
            fence_build_journal_record(packet, address(&values[index]), index + 1,
                                       (index + 1) * 11);
        }
    });
    for (unsigned i = 0; i < 1000; ++i) {
        FenceBuildRecord out;
        EXPECT_TRUE(fence_build_journal_lookup(packet, out));
        if (out.t_ms != 1 && out.t_ms != 2) {
            ADD_FAILURE() << "torn timestamp";
            break;
        }
        const auto index = static_cast<size_t>(out.t_ms - 1);
        EXPECT_EQ(out.addr, address(&values[index]));
        EXPECT_EQ(out.pre, values[index]);
        EXPECT_EQ(out.fold, (index + 1) * 11);
    }
    writer.join();
}
