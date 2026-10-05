// test_np_webapi — libSceNpWebApi2's offline contract: handles are created, every network
// round-trip fails fast instead of blocking.
//
// The 22 exports below were unregistered, so the dispatcher answered `0` — SCE_OK — while
// writing nothing. Request ids, response buffers and header lengths are exactly the divisor
// and sized-buffer shapes the audio/NP batches kept finding: a title proceeds on garbage and
// dies in its own arithmetic far from the cause. Every TEST therefore asserts what was WRITTEN
// on success, and that refused calls write NOTHING while answering UNAVAILABLE.
//
// NID provenance: all 24 names hash cleanly with nid_hash; three reproduce independently known
// values — `+o9816YQhqQ`/`sk54bi6FtYM` (already registered upstream) and `3Tt9zL3tkoc` (the
// BEAST live-boot table's `sceNpWebApi2CheckTimeout`, called 921 times as a poll).
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
static constexpr uint64_t kUnavailable = 0x80553406ull;

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
                  "the table must cover all 24 libSceNpWebApi2 exports");
    for (const char* name : table) {
        EXPECT_NE(Hle::lookup(nid_hash(name)), nullptr) << name << " is not registered";
    }
}

TEST(NpWebApi2, NidsResolveToKnownValues) {
    EXPECT_EQ(nid_hash("sceNpWebApi2Initialize"), "+o9816YQhqQ");
    EXPECT_EQ(nid_hash("sceNpWebApi2CreateUserContext"), "sk54bi6FtYM");
    EXPECT_EQ(nid_hash("sceNpWebApi2CheckTimeout"), "3Tt9zL3tkoc");
    EXPECT_NE(nid_hash("sceNpWebApi2CheckTimeout"), "AAAAAAAAAAA")
        << "positive control: the discriminator rejects a wrong NID";
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

TEST(NpWebApi2, DataPathUnavailable) {
    register_builtin_hle();
    HleFn send = Hle::lookup("lQOCF84lvzw");
    HleFn read = Hle::lookup("OOY9+ObfKec");
    HleFn get_value = Hle::lookup("hksbskNToEA");
    HleFn get_length = Hle::lookup("HwP3aM+c85c");
    HleFn check_timeout = Hle::lookup("3Tt9zL3tkoc");
    for (HleFn f : {send, read, get_value, get_length, check_timeout}) ASSERT_NE(f, nullptr);

    // No service behind the API: every round-trip fails fast WITHOUT touching its outputs,
    // which the guest would otherwise read as a completed response.
    uint8_t buf[64];
    std::memset(buf, 0xAA, sizeof buf);
    EXPECT_EQ(send(1, 0, 0, 0, 0, 0), kUnavailable);
    EXPECT_EQ(read(1, ptr(buf), sizeof buf, 0, 0, 0), kUnavailable);
    EXPECT_EQ(buf[0], 0xAA) << "response buffer untouched";
    std::memset(buf, 0xAA, sizeof buf);
    EXPECT_EQ(get_value(1, ptr("Content-Type"), ptr(buf), sizeof buf, 0, 0), kUnavailable);
    EXPECT_EQ(buf[0], 0xAA) << "header value untouched";
    size_t length = 0xDEADu;
    EXPECT_EQ(get_length(1, ptr("Content-Type"), ptr(&length), 0, 0, 0), kUnavailable);
    EXPECT_EQ(length, 0xDEADu) << "header length untouched: the guest must not size by it";
    // CheckTimeout is void: BEAST polls it 921 times a boot, and there is nothing to report.
    EXPECT_EQ(check_timeout(0, 0, 0, 0, 0, 0), 0u);
}

TEST(NpWebApi2, PushEventLifecycle) {
    register_builtin_hle();
    HleFn create_filter = Hle::lookup("MsaFhR+lPE4");
    HleFn create_handle = Hle::lookup("WV1GwM32NgY");
    HleFn create_pushctx = Hle::lookup("NNVf18SlbT8");
    HleFn delete_filter = Hle::lookup("KJdPcOGmK58");
    HleFn delete_handle = Hle::lookup("fIATVMo4Y1w");
    HleFn delete_pushctx = Hle::lookup("QafxeZM3WK4");
    HleFn reg_cb = Hle::lookup("fY3QqeNkF8k");
    HleFn reg_pushctx_cb = Hle::lookup("lxtHJMwBsaU");
    HleFn start_pushctx_cb = Hle::lookup("AAj9X+4aGYA");
    HleFn unreg_cb = Hle::lookup("hOnIlcGrO6g");
    HleFn unreg_pushctx_cb = Hle::lookup("PmyrbbJSFz0");
    for (HleFn f : {create_filter, create_handle, create_pushctx, delete_filter, delete_handle,
                    delete_pushctx, reg_cb, reg_pushctx_cb, start_pushctx_cb, unreg_cb,
                    unreg_pushctx_cb})
        ASSERT_NE(f, nullptr);

    EXPECT_GT(create_filter(1, 1, ptr("svc"), 0, 0, 0), 0u) << "filter handle is positive";
    EXPECT_GT(create_handle(1, 0, 0, 0, 0, 0), 0u) << "event handle is positive";
    EXPECT_GT(reg_cb(1, 1, 0, 0, 0, 0), 0u) << "callback registration returns a handle";
    EXPECT_GT(reg_pushctx_cb(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(delete_filter(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(delete_handle(1, 1, 0, 0, 0, 0), 0u);
    EXPECT_EQ(delete_pushctx(1, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unreg_cb(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(unreg_pushctx_cb(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(start_pushctx_cb(0, 0, 0, 0, 0, 0), 0u);
    // No push service exists to host a push context: creation fails, teardown succeeds.
    EXPECT_EQ(create_pushctx(0, 0, 0, 0, 0, 0), kUnavailable);
}
