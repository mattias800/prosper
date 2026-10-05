// test_gameupdate — the libSceGameUpdate request lifecycle: init/term, create/check/abort/delete
// and the addcont-latest-version query, pinned to the shipped libSceGameUpdate.sprx: its
// argument-check order, the 0x30-byte param/result structs, the 8 request slots and ids from
// 0x20000000, and a result write-back that touches only `found` (+8) and the byte at +9.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

static uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

// Zero-extended, matching what hle_service.cpp returns.
static constexpr uint64_t kNotInitialized = 0x80412801ull;
static constexpr uint64_t kInvalidArg = 0x80412803ull;
static constexpr uint64_t kInvalidSize = 0x80412804ull;
static constexpr uint64_t kRequestNotFound = 0x80412805ull;
static constexpr uint64_t kAlreadyInitialized = 0x80412802ull;
static constexpr uint64_t kTooManyRequests = 0x80412806ull;
static constexpr uint64_t kAborted = 0x80412807ull;

static uint64_t call_nid(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                         uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0);

// Each TEST starts from "not initialized"; the library state is process-global.
static void reset() {
    register_builtin_hle();
    call_nid("sceGameUpdateTerminate");
}

// The 0x30-byte param/result structs: a u64 size, then fields. Param +8 is the check type.
struct alignas(8) GuStruct { uint8_t b[0x30]; };
static GuStruct make_param(uint32_t check_type = 0) {
    GuStruct p{};
    const uint64_t size = 0x30;
    std::memcpy(p.b, &size, 8);
    std::memcpy(p.b + 8, &check_type, 4);
    return p;
}
static GuStruct make_result() {
    GuStruct r;
    std::memset(r.b, 0xAB, sizeof(r.b));
    const uint64_t size = 0x30;
    std::memcpy(r.b, &size, 8);
    return r;
}

static uint64_t call_nid(const char* nid, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3,
                         uint64_t a4, uint64_t a5) {
    HleFn fn = Hle::lookup(nid_hash(nid));
    EXPECT_NE(fn, nullptr) << nid << " is not registered";
    if (!fn) return ~0ull;
    return fn(a0, a1, a2, a3, a4, a5);
}

TEST(GameUpdate, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceGameUpdateCreateRequest", "sceGameUpdateCheck", "sceGameUpdateAbortRequest",
        "sceGameUpdateDeleteRequest", "sceGameUpdateGetAddcontLatestVersion",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 5, "the 5 lifecycle exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(GameUpdate, NidsResolveToStubValues) {
    // All five NIDs reproduce the stub table verbatim.
    EXPECT_EQ(nid_hash("sceGameUpdateCreateRequest"), "UvcvKaFvupA");
    EXPECT_EQ(nid_hash("sceGameUpdateCheck"), "LYVV9z8+owM");
    EXPECT_EQ(nid_hash("sceGameUpdateAbortRequest"), "d1CNGEOaK28");
    EXPECT_EQ(nid_hash("sceGameUpdateDeleteRequest"), "bcCyjHN5sn0");
    EXPECT_EQ(nid_hash("sceGameUpdateGetAddcontLatestVersion"), "0g0+Oq9xcI0");
    EXPECT_NE(nid_hash("sceGameUpdateCheck"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(GameUpdate, InitializeAndTerminateAreStateful) {
    reset();
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), kNotInitialized);
    EXPECT_EQ(call_nid("sceGameUpdateInitialize"), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateInitialize"), kAlreadyInitialized);
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
}

TEST(GameUpdate, CreateMintsFromTheFirmwareRangeIntoEightSlots) {
    reset();
    EXPECT_EQ(call_nid("sceGameUpdateCreateRequest"), kNotInitialized);
    ASSERT_EQ(call_nid("sceGameUpdateInitialize"), 0u);
    int32_t ids[8];
    for (int32_t& id : ids) {
        id = (int32_t)call_nid("sceGameUpdateCreateRequest");
        EXPECT_GE(id, 0x20000000);
        EXPECT_LE(id, 0x2fffffff);
    }
    EXPECT_NE(ids[0], ids[1]);
    EXPECT_EQ(call_nid("sceGameUpdateCreateRequest"), kTooManyRequests) << "only 8 slots";
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", (uint64_t)ids[3]), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", (uint64_t)ids[3]), kRequestNotFound);
    EXPECT_GE((int32_t)call_nid("sceGameUpdateCreateRequest"), 0x20000000) << "a freed slot is reused";
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
}

TEST(GameUpdate, DeleteAndAbortCheckTheIdBeforeInit) {
    reset();
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", 0), kRequestNotFound);
    EXPECT_EQ(call_nid("sceGameUpdateAbortRequest", (uint64_t)(uint32_t)-1), kRequestNotFound);
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", 0x20000000), kNotInitialized);
    EXPECT_EQ(call_nid("sceGameUpdateAbortRequest", 0x20000000), kNotInitialized);
}

TEST(GameUpdate, CheckValidatesInTheFirmwareOrder) {
    reset();
    GuStruct param = make_param(), result = make_result();
    // Argument checks come before the init check.
    EXPECT_EQ(call_nid("sceGameUpdateCheck", 0, addr(&param), addr(&result)), kRequestNotFound);
    EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x20000000, 0, addr(&result)), kInvalidArg);
    EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x20000000, addr(&param), 0), kInvalidArg);
    for (uint64_t bad : {0x28ull, 0x80ull, 0ull}) {
        GuStruct sized = make_param();
        std::memcpy(sized.b, &bad, 8);
        EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x20000000, addr(&sized), addr(&result)),
                  kInvalidSize)
            << "param size " << bad;
        GuStruct rsized = make_result();
        std::memcpy(rsized.b, &bad, 8);
        EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x20000000, addr(&param), addr(&rsized)),
                  kInvalidSize)
            << "result size " << bad;
    }
    for (uint32_t type : {1u, 2u}) {   // type 1 is PS4-process only
        GuStruct typed = make_param(type);
        EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x20000000, addr(&typed), addr(&result)),
                  kInvalidArg)
            << "check type " << type;
    }
    GuStruct reserved = make_param();
    reserved.b[0x2c] = 1;
    EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x20000000, addr(&reserved), addr(&result)),
              kInvalidArg);
    EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x20000000, addr(&param), addr(&result)),
              kNotInitialized);
    ASSERT_EQ(call_nid("sceGameUpdateInitialize"), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateCheck", 0x2fffffff, addr(&param), addr(&result)),
              kRequestNotFound);
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
}

TEST(GameUpdate, CheckReportsNoUpdateAndTouchesOnlyFound) {
    reset();
    ASSERT_EQ(call_nid("sceGameUpdateInitialize"), 0u);
    const int32_t id = (int32_t)call_nid("sceGameUpdateCreateRequest");
    {
        GuStruct param = make_param(0), result = make_result();
        ASSERT_EQ(call_nid("sceGameUpdateCheck", (uint64_t)id, addr(&param), addr(&result)), 0u);
        uint64_t size = 0;
        std::memcpy(&size, result.b, 8);
        EXPECT_EQ(size, 0x30u) << "the size field is left alone";
        EXPECT_EQ(result.b[8], 0u) << "found = false";
        EXPECT_EQ(result.b[9], 0u);
        for (size_t i = 10; i < sizeof(result.b); ++i)
            EXPECT_EQ(result.b[i], 0xABu) << "byte " << i << " is not written when nothing is found";
    }
    EXPECT_EQ(call_nid("sceGameUpdateAbortRequest", (uint64_t)id), 0u);
    GuStruct param = make_param(), result = make_result();
    EXPECT_EQ(call_nid("sceGameUpdateCheck", (uint64_t)id, addr(&param), addr(&result)), kAborted);
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", (uint64_t)id), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
}

TEST(GameUpdate, AddcontLatestVersionReportsNone) {
    reset();
    uint8_t label[16] = {};
    uint8_t info[0x20];
    std::memset(info, 0xAB, sizeof(info));
    EXPECT_EQ(call_nid("sceGameUpdateGetAddcontLatestVersion", 0, 0, addr(info)), kInvalidArg);
    EXPECT_EQ(call_nid("sceGameUpdateGetAddcontLatestVersion", 0, addr(label), 0), kInvalidArg);
    EXPECT_EQ(call_nid("sceGameUpdateGetAddcontLatestVersion", 0, addr(label), addr(info)),
              kNotInitialized);
    ASSERT_EQ(call_nid("sceGameUpdateInitialize"), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateGetAddcontLatestVersion", 0, addr(label), addr(info)), 0u);
    EXPECT_EQ(info[8], 0u) << "found = false";
    for (size_t i = 0; i < sizeof(info); ++i)
        if (i != 8) EXPECT_EQ(info[i], 0xABu) << "byte " << i << " is not written";
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
}
