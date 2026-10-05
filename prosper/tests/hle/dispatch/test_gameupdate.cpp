// test_gameupdate — the libSceGameUpdate request lifecycle: create/check/abort/delete and
// the addcont-latest-version query.
//
// These five exports were unregistered, so the dispatcher answered `0`: a create that minted
// no id, a check that reported over an untouched result buffer, deletes that deleted nothing.
// Every TEST drives the real NIDs: uninitialized use fails, foreign ids fail, and a check
// fills its caller-sized result with the offline answer (nothing available) while echoing the
// size field back. A no-op acknowledgement would succeed everywhere below, which is exactly
// what the error arms forbid.
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

static uint64_t call_nid(const char* nid, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0,
                         uint64_t a3 = 0, uint64_t a4 = 0, uint64_t a5 = 0) {
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

TEST(GameUpdate, CreateCheckDeleteLifecycle) {
    register_builtin_hle();
    EXPECT_NE(Hle::lookup(nid_hash("sceGameUpdateInitialize")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("sceGameUpdateTerminate")), nullptr);

    // Forced-uninitialized first: terminate clears any state so this arm depends on nothing.
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateCreateRequest"), kNotInitialized)
        << "create before init fails instead of minting an id";
    EXPECT_EQ(call_nid("sceGameUpdateInitialize"), 0u);

    const uint64_t id = call_nid("sceGameUpdateCreateRequest");
    EXPECT_GT(id, 0u) << "create mints a positive request id";
    EXPECT_NE(call_nid("sceGameUpdateCheck", 0xDEADu, 0, 0), 0u)
        << "check on a foreign id fails";
    EXPECT_EQ(call_nid("sceGameUpdateCheck", 0xDEADu, 0, 0), kRequestNotFound)
        << "check on a foreign id reports request-not-found";

    EXPECT_EQ(call_nid("sceGameUpdateAbortRequest", id), 0u) << "abort on a live id succeeds";
    EXPECT_EQ(call_nid("sceGameUpdateAbortRequest", 0xDEADu), kRequestNotFound)
        << "abort on a foreign id fails";
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", id), 0u) << "delete frees the id";
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", id), kRequestNotFound)
        << "delete frees exactly once";
    EXPECT_EQ(call_nid("sceGameUpdateCheck", id, 0, 0), kRequestNotFound)
        << "a deleted id stays deleted";
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
}

TEST(GameUpdate, CheckReportsNoUpdateOverCallerSizedResult) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("sceGameUpdateInitialize"), 0u);
    const uint64_t id = call_nid("sceGameUpdateCreateRequest");
    ASSERT_GT(id, 0u);

    // Caller-led size with a sentinel tail: the offline answer zeroes the body and echoes
    // the size back. Untouched tail bytes would mean "update found" wherever `found` lives.
    uint8_t result[128];
    std::memset(result, 0xAB, sizeof(result));
    uint32_t size = sizeof(result);
    std::memcpy(result, &size, sizeof(size));
    uint8_t param[48]{};
    uint32_t param_size = sizeof(param);
    std::memcpy(param, &param_size, sizeof(param_size));
    EXPECT_EQ(call_nid("sceGameUpdateCheck", id, addr(param), addr(result)), 0u);
    uint32_t echoed = 0;
    std::memcpy(&echoed, result, sizeof(echoed));
    EXPECT_EQ(echoed, (uint32_t)sizeof(result)) << "the size field is echoed back";
    for (size_t i = sizeof(echoed); i < sizeof(result); ++i) {
        EXPECT_EQ(result[i], 0u) << "result body is zeroed (nothing available) at byte " << i;
    }

    EXPECT_EQ(call_nid("sceGameUpdateCheck", id, 0, addr(result)), kInvalidArg)
        << "null param is refused";
    EXPECT_EQ(call_nid("sceGameUpdateCheck", id, addr(param), 0), kInvalidArg)
        << "null result is refused";

    // A garbage size must fail BEFORE writing, not memset the world: sentinel proves it.
    uint8_t small[16];
    std::memset(small, 0xCD, sizeof(small));
    uint32_t huge = 1024u * 1024u;
    std::memcpy(small, &huge, sizeof(huge));
    EXPECT_EQ(call_nid("sceGameUpdateCheck", id, addr(param), addr(small)), kInvalidSize)
        << "garbage sizes fail instead of zeroing a megabyte";
    for (size_t i = sizeof(huge); i < sizeof(small); ++i) {
        EXPECT_EQ(small[i], 0xCDu) << "refused check leaves the buffer untouched at byte " << i;
    }
    EXPECT_EQ(call_nid("sceGameUpdateDeleteRequest", id), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
}

TEST(GameUpdate, AddcontLatestVersionReportsNone) {
    register_builtin_hle();
    EXPECT_EQ(call_nid("sceGameUpdateInitialize"), 0u);

    uint8_t info[48];
    std::memset(info, 0xAB, sizeof(info));
    uint32_t size = sizeof(info);
    std::memcpy(info, &size, sizeof(size));
    EXPECT_EQ(call_nid("sceGameUpdateGetAddcontLatestVersion", 0, 0, addr(info)), 0u);
    uint32_t echoed = 0;
    std::memcpy(&echoed, info, sizeof(echoed));
    EXPECT_EQ(echoed, (uint32_t)sizeof(info)) << "the size field is echoed back";
    for (size_t i = sizeof(echoed); i < sizeof(info); ++i) {
        EXPECT_EQ(info[i], 0u) << "no update known, so the version block is zeroed at byte " << i;
    }
    EXPECT_EQ(call_nid("sceGameUpdateGetAddcontLatestVersion", 0, 0, 0), kInvalidArg)
        << "null info is refused";
    EXPECT_EQ(call_nid("sceGameUpdateTerminate"), 0u);
    EXPECT_EQ(call_nid("sceGameUpdateGetAddcontLatestVersion", 0, 0, addr(info)), kNotInitialized)
        << "terminated state fails instead of answering";
}
