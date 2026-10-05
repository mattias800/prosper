// test_np_manager — libSceNpManager's offline remainder: local request ids, SIGNED_OUT
// account/premium answers, acknowledged callbacks/setters.
//
// The 14 exports below were unregistered, so the dispatcher answered `0`. For the account and
// premium getters 0 is SCE_OK over untouched out-structs — a title reads a blank age/language
// or a successful premium check as valid account state and takes online branches that dead-end
// (the #306 wedge class). Every TEST therefore asserts SIGNED_OUT with sentinels intact, local
// ids where the console always allocates them, and bare acknowledgement where nothing is
// observable. PollAsync/CreateAsyncRequest stay deliberately unregistered (their completion
// flow needs a live capture — see the in-file note).
//
// NID provenance: all names hash cleanly with nid_hash, triple-checked against already
// registered siblings (CheckNpAvailability, GetAccountIdA, UnregisterStateCallbackA).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static uint64_t ptr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

static constexpr uint64_t kSignedOut = 0x80550006ull;  // SCE_NP_ERROR_SIGNED_OUT

TEST(NpManager, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceNpCreateRequest",
        "sceNpAbortRequest",
        "sceNpDeleteRequest",
        "sceNpGetAccountAge",
        "sceNpGetAccountLanguage2",
        "sceNpCheckPremium",
        "sceNpNotifyPremiumFeature",
        "sceNpRegisterGamePresenceCallback",
        "sceNpRegisterNpReachabilityStateCallback",
        "sceNpRegisterPlusEventCallback",
        "sceNpRegisterPremiumEventCallback",
        "sceNpSetContentRestriction",
        "sceNpSetNpTitleId",
        "sceNpUnregisterStateCallback",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 14, "the 14 remainder exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(NpManager, NidsResolveToKnownValues) {
    EXPECT_EQ(nid_hash("sceNpCheckNpAvailability"), "2rsFmlGWleQ");
    EXPECT_EQ(nid_hash("sceNpGetAccountIdA"), "rbknaUjpqWo");
    EXPECT_EQ(nid_hash("sceNpUnregisterStateCallbackA"), "M3wFXbYQtAA");
    EXPECT_EQ(nid_hash("sceNpCreateRequest"), "GpLQDNKICac");
    EXPECT_NE(nid_hash("sceNpCreateRequest"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
}

TEST(NpManager, RequestLifecycle) {
    register_builtin_hle();
    HleFn create = Hle::lookup("GpLQDNKICac");
    HleFn abort = Hle::lookup("OzKvTvg3ZYU");
    HleFn del = Hle::lookup("S7QTn72PrDw");
    ASSERT_NE(create, nullptr);
    ASSERT_NE(abort, nullptr);
    ASSERT_NE(del, nullptr);
    const uint64_t first = create(0, 0, 0, 0, 0, 0);
    EXPECT_GT((int64_t)first, 0) << "CreateRequest hands a positive local id";
    EXPECT_NE(create(0, 0, 0, 0, 0, 0), first) << "request ids are distinct";
    EXPECT_EQ(abort(first, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(del(first, 0, 0, 0, 0, 0), 0u);
}

TEST(NpManager, AccountAndPremiumSignedOut) {
    register_builtin_hle();
    HleFn age = Hle::lookup("+4DegjBqV1g");
    HleFn language = Hle::lookup("3Tcz5bNCfZQ");
    HleFn premium = Hle::lookup("O80NrhUOPGY");
    HleFn notify = Hle::lookup("P6piso307SE");
    for (HleFn f : {age, language, premium, notify}) ASSERT_NE(f, nullptr);
    // Offline and signed out: premium status is unverifiable, so the premium answers take the
    // same signed-out branch as every other account query — never a manufactured "premium".
    uint8_t out[32];
    std::memset(out, 0xAA, sizeof out);
    EXPECT_EQ(age(1, 1, ptr(out), 0, 0, 0), kSignedOut);
    EXPECT_EQ(out[0], 0xAA) << "age untouched";
    std::memset(out, 0xAA, sizeof out);
    EXPECT_EQ(language(1, 1, ptr(out), 0, 0, 0), kSignedOut);
    EXPECT_EQ(out[0], 0xAA) << "language untouched";
    std::memset(out, 0xAA, sizeof out);
    EXPECT_EQ(premium(1, 0, ptr(out), 0, 0, 0), kSignedOut);
    EXPECT_EQ(out[0], 0xAA) << "premium result untouched";
    EXPECT_EQ(notify(0, 0, 0, 0, 0, 0), kSignedOut) << "nowhere to notify offline";
}

TEST(NpManager, CallbacksAndSettersAcknowledged) {
    register_builtin_hle();
    HleFn presence = Hle::lookup("uFJpaKNBAj4");
    HleFn reachability = Hle::lookup("hw5KNqAAels");
    HleFn plus = Hle::lookup("GImICnh+boA");
    HleFn premium_cb = Hle::lookup("+yqjab2fUJA");
    HleFn restriction = Hle::lookup("A2CQ3kgSopQ");
    HleFn title_id = Hle::lookup("Ec63y59l9tw");
    HleFn unregister = Hle::lookup("mjjTXh+NHWY");
    for (HleFn f : {presence, reachability, plus, premium_cb, restriction, title_id, unregister})
        ASSERT_NE(f, nullptr);
    // Callbacks register nowhere (never fired offline); setters acknowledge input-only args.
    EXPECT_EQ(presence(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(reachability(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(plus(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(premium_cb(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(restriction(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(title_id(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unregister(0, 0, 0, 0, 0, 0), 0u);
}
