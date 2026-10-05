// test_np_webapi — libSceNpWebApi2's offline contract: handles are created locally, every call
// that would reach PSN answers SCE_NP_ERROR_SIGNED_OUT (the answer the rest of NP gives).
//
// The 22 exports added here were unregistered, so the dispatcher answered `0` — SCE_OK — while
// writing nothing. Request ids, response buffers and header lengths are exactly the divisor
// and sized-buffer shapes the audio/NP batches kept finding: a title proceeds on garbage and
// dies in its own arithmetic far from the cause. Every TEST therefore asserts what was WRITTEN
// on success, and that refused calls write NOTHING while answering SIGNED_OUT.
//
// Coverage: the 24 exports main and this change register, of the library's 39 in the PS5 3.20
// table. NID provenance: `NidsResolveToKnownValues` pins nid_hash against NIDs copied from the
// 3.20 table, so a wrong hash cannot hide behind lookups that use the same hash.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

using HleFn = uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t);
static uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }
static bool is_error(uint64_t r) { return (int64_t)r < 0; }

static constexpr uint64_t kInvalidArgument = 0x80553402ull;
static constexpr uint64_t kSignedOut = 0x80550006ull;   // SCE_NP_ERROR_SIGNED_OUT

TEST(NpWebApi2, AllNidsBound) {
    register_builtin_hle();
    static const char* table[] = {
        "sceNpWebApi2Initialize",
        "sceNpWebApi2CreateUserContext",
        "sceNpWebApi2CreateRequest",
        "sceNpWebApi2AbortRequest",
        "sceNpWebApi2AddHttpRequestHeader",
        "sceNpWebApi2DeleteRequest",
        "sceNpWebApi2DeleteUserContext",
        "sceNpWebApi2SendRequest",
        "sceNpWebApi2ReadData",
        "sceNpWebApi2GetHttpResponseHeaderValue",
        "sceNpWebApi2GetHttpResponseHeaderValueLength",
        "sceNpWebApi2CheckTimeout",
        "sceNpWebApi2Terminate",
        "sceNpWebApi2PushEventCreateFilter",
        "sceNpWebApi2PushEventCreateHandle",
        "sceNpWebApi2PushEventCreatePushContext",
        "sceNpWebApi2PushEventDeleteFilter",
        "sceNpWebApi2PushEventDeleteHandle",
        "sceNpWebApi2PushEventDeletePushContext",
        "sceNpWebApi2PushEventRegisterCallback",
        "sceNpWebApi2PushEventRegisterPushContextCallback",
        "sceNpWebApi2PushEventStartPushContextCallback",
        "sceNpWebApi2PushEventUnregisterCallback",
        "sceNpWebApi2PushEventUnregisterPushContextCallback",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 24,
                  "the 24 exports main and this change register (of the library's 39)");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(NpWebApi2, NidsResolveToKnownValues) {
    EXPECT_EQ(nid_hash("sceNpWebApi2Initialize"), "+o9816YQhqQ");
    EXPECT_EQ(nid_hash("sceNpWebApi2CreateUserContext"), "sk54bi6FtYM");
    EXPECT_EQ(nid_hash("sceNpWebApi2CheckTimeout"), "3Tt9zL3tkoc");
    // Copied from the PS5 3.20 libSceNpWebApi2 table, not from the registration code.
    EXPECT_EQ(nid_hash("sceNpWebApi2SendRequest"), "lQOCF84lvzw");
    EXPECT_EQ(nid_hash("sceNpWebApi2ReadData"), "OOY9+ObfKec");
    EXPECT_EQ(nid_hash("sceNpWebApi2PushEventCreatePushContext"), "NNVf18SlbT8");
}

TEST(NpWebApi2, RequestLifecycle) {
    register_builtin_hle();
    HleFn create = Hle::lookup("3EI-OSJ65Xc");
    HleFn abort = Hle::lookup("zpiPsH7dbFQ");
    HleFn add_header = Hle::lookup("egOOvrnF6mI");
    HleFn del = Hle::lookup("vvzWO-DvG1s");
    HleFn del_ctx = Hle::lookup("9X9+cneTGUU");
    HleFn term = Hle::lookup("bEvXpcEk200");
    for (HleFn f : {create, abort, add_header, del, del_ctx, term}) ASSERT_NE(f, nullptr);

    int64_t request = 0;
    ASSERT_EQ(create(1, ptr("v2"), ptr("/path"), ptr("GET"), 0, ptr(&request)), 0u);
    EXPECT_GT(request, 0) << "CreateRequest writes its local request id";
    int64_t second = 0;
    ASSERT_EQ(create(1, ptr("v2"), ptr("/path"), ptr("GET"), 0, ptr(&second)), 0u);
    EXPECT_NE(second, request) << "request ids are distinct";
    EXPECT_EQ(create(1, ptr("v2"), ptr("/path"), ptr("GET"), 0, 0), kInvalidArgument)
        << "null request-id pointer is refused";
    EXPECT_EQ(add_header(request, ptr("X-Foo"), ptr("bar"), 0, 0, 0), 0u);
    EXPECT_EQ(abort(request, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(del(request, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(del_ctx(1, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(term(1, 0, 0, 0, 0, 0), 0u);
}

// Every call that would reach PSN, driven through nid_hash(name) with whole sentinel buffers on
// each pointer argument. Each must answer SIGNED_OUT -- not 0, which the guest would read as a
// completed response -- and leave every buffer untouched.
TEST(NpWebApi2, NetworkCallsAnswerSignedOutAndWriteNothing) {
    register_builtin_hle();
    uint8_t b1[64], b2[64], b3[64];
    struct Row {
        const char* name;
        uint64_t a0, a1, a2, a3;
    };
    const Row rows[] = {
        // SendRequest(libCtx, requestId, pData, pRespInfo): a3 is the response-info out pointer.
        {"sceNpWebApi2SendRequest", 1, 1, ptr(b1), ptr(b2)},
        {"sceNpWebApi2ReadData", 1, ptr(b1), sizeof b1, 0},
        {"sceNpWebApi2GetHttpResponseHeaderValue", 1, ptr("Content-Type"), ptr(b1), sizeof b1},
        {"sceNpWebApi2GetHttpResponseHeaderValueLength", 1, ptr("Content-Type"), ptr(b1), 0},
        // CreatePushContext(userCtx, pPushCtxId): the out struct (>= 37 bytes) stays untouched.
        {"sceNpWebApi2PushEventCreatePushContext", 1, ptr(b1), 0, 0},
        {"sceNpWebApi2PushEventStartPushContextCallback", 1, ptr(b1), 0, 0},
    };
    for (const Row& row : rows) {
        SCOPED_TRACE(row.name);
        HleFn fn = Hle::lookup(nid_hash(row.name));
        ASSERT_NE(fn, nullptr);
        std::memset(b1, 0xAA, sizeof b1);
        std::memset(b2, 0xBB, sizeof b2);
        std::memset(b3, 0xCC, sizeof b3);
        const uint64_t r = fn(row.a0, row.a1, row.a2, row.a3, ptr(b3), 0);
        EXPECT_NE(r, 0u) << "a network call answering 0 reads as a completed response";
        EXPECT_EQ(r, kSignedOut);
        for (size_t i = 0; i < sizeof b1; ++i) {
            ASSERT_EQ(b1[i], 0xAA) << "byte " << i << " of the first buffer was written";
            ASSERT_EQ(b2[i], 0xBB) << "byte " << i << " of the second buffer was written";
            ASSERT_EQ(b3[i], 0xCC) << "byte " << i << " of the third buffer was written";
        }
    }
    // CheckTimeout is void: BEAST polls it 921 times a boot, and there is nothing to report.
    HleFn check_timeout = Hle::lookup(nid_hash("sceNpWebApi2CheckTimeout"));
    ASSERT_NE(check_timeout, nullptr);
    EXPECT_EQ(check_timeout(0, 0, 0, 0, 0, 0), 0u);
}

TEST(NpWebApi2, PushEventLifecycle) {
    register_builtin_hle();
    auto fn = [](const char* name) { return Hle::lookup(nid_hash(name)); };
    HleFn create_filter = fn("sceNpWebApi2PushEventCreateFilter");
    HleFn create_handle = fn("sceNpWebApi2PushEventCreateHandle");
    HleFn delete_filter = fn("sceNpWebApi2PushEventDeleteFilter");
    HleFn delete_handle = fn("sceNpWebApi2PushEventDeleteHandle");
    HleFn delete_pushctx = fn("sceNpWebApi2PushEventDeletePushContext");
    HleFn reg_cb = fn("sceNpWebApi2PushEventRegisterCallback");
    HleFn reg_pushctx_cb = fn("sceNpWebApi2PushEventRegisterPushContextCallback");
    HleFn unreg_cb = fn("sceNpWebApi2PushEventUnregisterCallback");
    HleFn unreg_pushctx_cb = fn("sceNpWebApi2PushEventUnregisterPushContextCallback");
    for (HleFn f : {create_filter, create_handle, delete_filter, delete_handle, delete_pushctx,
                    reg_cb, reg_pushctx_cb, unreg_cb, unreg_pushctx_cb})
        ASSERT_NE(f, nullptr);

    // The constructors RETURN positive ids (Worms keeps eax and tests it with js).
    const int64_t filter = (int64_t)create_filter(1, 1, ptr("svc"), 0, 0, 0);
    const int64_t handle = (int64_t)create_handle(1, 0, 0, 0, 0, 0);
    EXPECT_GT(filter, 0) << "filter id is positive";
    EXPECT_GT(handle, 0) << "event handle is positive";
    EXPECT_NE(filter, handle) << "ids are distinct";
    static int dummy_callback_target = 0;
    const uint64_t cb = ptr(&dummy_callback_target);
    EXPECT_GT((int64_t)reg_cb(1, filter, cb, 0, 0, 0), 0) << "callback registration returns an id";
    EXPECT_GT((int64_t)reg_pushctx_cb(1, filter, cb, 0, 0, 0), 0);
    // A null callback is refused rather than registered.
    EXPECT_EQ(reg_cb(1, filter, 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(reg_pushctx_cb(1, filter, 0, 0, 0, 0), kInvalidArgument);
    // Teardown succeeds: there is nothing to release offline.
    EXPECT_EQ(delete_filter(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(delete_handle(1, 1, 0, 0, 0, 0), 0u);
    EXPECT_EQ(delete_pushctx(1, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unreg_cb(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unreg_pushctx_cb(0, 0, 0, 0, 0, 0), 0u);
}
