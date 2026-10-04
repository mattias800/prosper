// test_np_sublibs — the batch-2 NP sub-libraries on the honest signed-out console.
//
// The NP sub-libraries of this batch — libSceNpTrophy (v1), libSceNpTus, libSceNpScore,
// libSceNpCommerce, libSceNpAuth, libSceNpUtility, libSceNpMatching2 and libSceNpSns — were
// unregistered, so every call reached the dispatcher's `return 0` — SCE_OK: trophy content
// "queried" into unwritten memory (the #213 class), TUS/Score data "fetched" from a PSN the
// console is not signed in to, a PS Store "transaction" that never happened, and OAuth/SNS
// tokens "issued" to a console with no user. This pins the batch: valid local ids where a real
// console always allocates them, SIGNED_OUT where the PSN path is needed, the v1 trophy
// facility's no-user answer for trophy content, and the headless dialog lifecycle where there
// is no interactive UI.
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using namespace prosper;

namespace {

constexpr uint64_t kSignedOut = 0x80550006ull;      // SCE_NP_ERROR_SIGNED_OUT
constexpr uint64_t kTrophyNoUser = 0x8055161Dull;   // ORBIS_NP_TROPHY_ERROR_USER_NOT_LOGGED_IN

uint64_t ptr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

// Every export of the four libraries in the PS5 3.20 stub table must be bound: an unregistered
// NID answers SCE_OK with nothing written, which is the failure this batch removes.
void expect_all_bound() {
    const char* const table[] = {
        // libSceNpTrophy (v1)
        "HbkjbobZlCY",
        "K7U6tEAQf7c",
        "OdPIOFpEAvU",
        "DJCAxto9SEU",
        "KTnHs7W-9Uk",
        "O9plkqa2e0k",
        "NV18n8OcheI",
        "FiPpytPUPMA",
        "BBkBINKo6gw",
        "Hit5yM0Teo0",
        "I5R5Ogfbk68",
        "MNbxxwNdlHY",
        "IYP3f2W09og",
        "ATUwGfspKic",
        "KqUVGDgQBm0",
        "A4uMPmErD4I",
        "OBL+l6HG9xk",
        "BvdThnVvwdY",
        "G8xmRUFao68",
        "N9jpdPz5f-8",
        "N3CQzag7-zs",
        "EF9zjnlAzIA",
        "EXiyfabxFNQ",
        "MaLlLHKP+No",
        "GrV7Y4IhWkc",
        "AJD0VSnMfW0",
        "CSh2GsVQTzs",
        "PtaF5aJl7k0",
        "BMmIU5R9IHY",
        // libSceNpTus (incl. the TSS small-storage family)
        "Bhy8+oecGac",
        "Fn-dGukBgnY",
        "MRVb2Cf0GHg",
        "FBtrk+7lk14",
        "Hbh2aBvvmvM",
        "Geq1bMwgZYo",
        "KGKDdRCFx8c",
        "MrVmNrJDbG8",
        "APFah4-5Xec",
        "GdB427dT3Iw",
        "A2UmHdK04c8",
        "Okr6FBSrkJw",
        "EDT5bP6YzBo",
        "CXzUOM9sXU0",
        "K-+Yqc-NppQ",
        "ButwCvsydkk",
        "GYhbiRtkE1Y",
        "JwnE9Oa1uF8",
        "HOzszO4ONWU",
        "CWEHUFkY1qI",
        "BzG8mG9YlKY",
        "CaH+Sxlw32k",
        "OoFvgzwawAY",
        "OHtKS5V1T5k",
        "FTE3OvH61qo",
        "PLxFGYCJwww",
        "JR6kI-8f+Hk",
        "Cixh7HDKWfk",
        "My+pAALkHp8",
        "ArImtTqUSGM",
        "IFYWOwYI6DY",
        "JgcNwFHoOL4",
        "M33Y2TnyonE",
        "LcPB2rnhQqo",
        "OFxVYJEkcmc",
        "FxNDPDnWfMc",
        "KG9+4eIb+cY",
        "IRje5yEXS0U",
        "DB0vaHTzA6g",
        "N7b6dmpQNiI",
        "INrufkNCkiE",
        "FzxN3tOouj8",
        "Iu58d6g6uwU",
        "EbWqOt3QjKU",
        "ORhzSuuXwxo",
        "M6aYoa47YgI",
        "Mf-WMA0jYCc",
        "JJ9GGMludxY",
        "FCz0hTJFyh4",
        "OkC55HsotJ4",
        "Eup4MP1wNtc",
        "LGTjTkHPHTE",
        "IGIcxlUabSA",
        "BQfR51i4kck",
        "JbitD262GhY",
        "BYPJFWzFPjA",
        "PSUR+UoLS6c",
        "FL+Z3zCKNTs",
        "P2Pe4LGS2II",
        "E5NZIzggbuk",
        // libSceNpScore
        "KW9M0bQ-Zx0",
        "AW8qyjYrUbk",
        "Fi7kmKbX6hk",
        "NK8-SgYf6r4",
        "CxK68584JAU",
        "C3xZj35v8Z8",
        "Gb3TI0mDYiI",
        "IeOvDyN-aZc",
        "A0Avi9kebsY",
        "MkuIzUw6utQ",
        "AMbOn+-6eXA",
        "DkoVok6FFEI",
        "NmZEgoiEq6Y",
        "G1DfNRstkSQ",
        "LcoVwcBjQ9E",
        "FgL5PwYzrrw",
        "DT0XBtgtOSI",
        "L4oAo9in0TA",
        "Pqk8SC63p1U",
        // libSceNpCommerce
        "0aR2aWmQal4",
        "DfSCDRA3EjY",
        "IXmfUaze9So",
        "CCbC+lqqvF0",
        "LR5cwFMMCVE",
        "r42bWcQbtZY",
        "NU3ckGHMFXo",
        "m-I92Ab50W8",
        "DHmwsa6S8Tc",
        "dsqCVsNM0Zg",
        "uKTDW8hk-ts",
        // libSceNpAuth
        "6bwFkosYRQg",
        "N+mr7GjTvr8",
        "cE7wIsqXdZ8",
        "H8wG9Bk-nPc",
        "PM3IZCw-7m0",
        "gjSyfzSsDcE",
        "SK-S7daqJSE",
        "KAUXQ9GdWp8",
        "KI4dHLlTNl0",
        "IDX0S5EsEh4",
        "OaB-LoJqHis",
        "RdsFVsgSpZY",
        // libSceNpUtility (bandwidth, lookup, word-filter, app-info)
        "G6iWw8aUQtA",
        "F6Dl+2zlua0",
        "M9+zoKE8cBA",
        "pLr1fEQS1z8",
        "kvdMF48mB3Y",
        "hqzi1IHdQQQ",
        "mA0zsbqm+kA",
        "BYIZGKm6bO4",
        "DVZE+fAhgFY",
        "JXlTj9RRCFo",
        "CQr9UxPHUFs",
        "M533Q+LU7EQ",
        "ALaxchvEEnk",
        "GtqDK9zkoIE",
        "OYz4v5Uek9U",
        "EMV72WO7V34",
        "GnEVmFiV6OI",
        "D6tnM1Uti4g",
        "F4EVrruHuy8",
        "IX9dAus6baE",
        "Or5SShyG0dk",
        "M7ivWj5yKzg",
        // libSceNpMatching2
        "HHZpTF30wto",
        "AupHEf8WOhM",
        "ODxEHb9f7B8",
        "Kxlf9+pa0GY",
        "Hddl5xnQQEY",
        "FagjVl+bHFI",
        "DMxxNNLh6ms",
        "EcYuZkNhHI8",
        "GkvclTMjNdI",
        // libSceNpSns (Facebook, Twitch, YouTube, shared Int family)
        "D+F4GKuY3oE",
        "E2nPI+P0g8o",
        "Ho3XDEydBjM",
        "KLqzbBxATrU",
        "OPj-CXHNEFE",
        "PLJbF9b-Who",
        "JTvAKV1iQkE",
        "E-sppToVlnc",
        "DXUeCRu7DLE",
        "GGfyLTvE+LI",
        "FtyS8XLBqNE",
        "JzCW8dx4mKk",
        "OINq9QxFYqU",
        "O6K3LE8qXsc",
        "Gp+C91igTkE",
        "ExrhvJ8QANU",
        "A+mSQ2U6wWY",
        "AW7NonbfeFk",
        "Pz-00XG-VNU",
        "GHLVam6hZZ4",
        "FElwHkpLvmw",
    };
    static_assert(sizeof(table) / sizeof(table[0]) == 183,
                  "the table must cover all 183 non-module exports of the eight libraries");
    for (const char* nid : table) { EXPECT_NE(Hle::lookup(nid), nullptr) << "unbound NID " << nid; }
}

}  // namespace

TEST(NpSublibs, AllNidsBound) {
    register_builtin_hle();
    expect_all_bound();
}

TEST(NpTrophyV1, LifecycleIdsAndContentFailure) {
    register_builtin_hle();
    HleFn createctx = Hle::lookup("HbkjbobZlCY");
    HleFn createhandle = Hle::lookup("K7U6tEAQf7c");
    HleFn regctx = Hle::lookup("DJCAxto9SEU");
    HleFn abort_handle = Hle::lookup("KTnHs7W-9Uk");
    HleFn getgameinfo = Hle::lookup("IYP3f2W09og");
    HleFn total = Hle::lookup("BvdThnVvwdY");
    HleFn unlock = Hle::lookup("G8xmRUFao68");
    ASSERT_NE(createctx, nullptr);
    ASSERT_NE(createhandle, nullptr);
    ASSERT_NE(regctx, nullptr);
    ASSERT_NE(abort_handle, nullptr);
    ASSERT_NE(getgameinfo, nullptr);
    ASSERT_NE(total, nullptr);
    ASSERT_NE(unlock, nullptr);

    uint32_t ctx = 0xDEADu, handle = 0xDEADu;
    ASSERT_EQ(createctx(ptr(&ctx), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(ctx, 1u) << "CreateContext writes its local id";
    ASSERT_EQ(createhandle(ptr(&handle), ctx, ptr("PPSA00000"), 0, 0, 0), 0u);
    EXPECT_EQ(handle, 1u) << "CreateHandle writes its local id";
    EXPECT_EQ(regctx(ctx, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(abort_handle(ctx, handle, 0, 0, 0, 0), 0u);

    // The no-user answer, with the out-structs left exactly as the caller left them.
    uint8_t game_info[0x100];
    std::memset(game_info, 0xAA, sizeof(game_info));
    EXPECT_EQ(getgameinfo(ctx, handle, ptr(game_info), 0, 0, 0), kTrophyNoUser);
    EXPECT_EQ(game_info[0], 0xAA) << "a refused query must not write its out struct";
    int32_t total_count = -7;
    EXPECT_EQ(total(ctx, handle, ptr(&total_count), 0, 0, 0), kTrophyNoUser);
    EXPECT_EQ(total_count, -7) << "NumInfoGetTotal leaves the count untouched";
    EXPECT_EQ(unlock(ctx, handle, 1, 100, 0, 0), kTrophyNoUser);
}

TEST(NpTus, LocalIdsAndNoData) {
    register_builtin_hle();
    HleFn ctx = Hle::lookup("Bhy8+oecGac");
    HleFn req = Hle::lookup("Hbh2aBvvmvM");
    HleFn thread_param = Hle::lookup("KGKDdRCFx8c");
    HleFn abort_req = Hle::lookup("Geq1bMwgZYo");
    HleFn set_data = Hle::lookup("FzxN3tOouj8");
    HleFn get_data = Hle::lookup("CWEHUFkY1qI");
    HleFn wait_async = Hle::lookup("BYPJFWzFPjA");
    HleFn poll_async = Hle::lookup("N7b6dmpQNiI");
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(req, nullptr);
    ASSERT_NE(thread_param, nullptr);
    ASSERT_NE(abort_req, nullptr);
    ASSERT_NE(set_data, nullptr);
    ASSERT_NE(get_data, nullptr);
    ASSERT_NE(wait_async, nullptr);
    ASSERT_NE(poll_async, nullptr);

    uint32_t context = 0xDEADu, request = 0xDEADu;
    ASSERT_EQ(ctx(ptr(&context), ptr("PPSA00000"), 0, 0, 0, 0), 0u);
    EXPECT_EQ(context, 1u) << "CreateTitleCtx writes its local id";
    ASSERT_EQ(req(ptr(&request), context, 0, 0, 0, 0), 0u);
    EXPECT_EQ(request, 1u) << "CreateRequest writes its local id";
    EXPECT_EQ(thread_param(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(abort_req(request, 0, 0, 0, 0, 0), 0u);

    uint8_t data[0x100];
    std::memset(data, 0x55, sizeof(data));
    EXPECT_EQ(set_data(request, context, ptr(data), sizeof(data), 0, 0), kSignedOut);
    int32_t status = -3;
    EXPECT_EQ(get_data(request, context, ptr(&status), 0, 0, 0), kSignedOut);
    EXPECT_EQ(status, -3) << "a refused read leaves the status untouched";
    EXPECT_EQ(wait_async(request, 0, 0, 0, 0, 0), kSignedOut);
    EXPECT_EQ(poll_async(request, 0, 0, 0, 0, 0), kSignedOut);
}

TEST(NpScore, LocalIdsAndNoData) {
    register_builtin_hle();
    HleFn ctx = Hle::lookup("KW9M0bQ-Zx0");
    HleFn req = Hle::lookup("AW8qyjYrUbk");
    HleFn record = Hle::lookup("DT0XBtgtOSI");
    HleFn ranking = Hle::lookup("MkuIzUw6utQ");
    HleFn del_req = Hle::lookup("NK8-SgYf6r4");
    ASSERT_NE(ctx, nullptr);
    ASSERT_NE(req, nullptr);
    ASSERT_NE(record, nullptr);
    ASSERT_NE(ranking, nullptr);
    ASSERT_NE(del_req, nullptr);

    uint32_t context = 0xDEADu, request = 0xDEADu;
    ASSERT_EQ(ctx(ptr(&context), ptr("PPSA00000"), 0, 0, 0, 0), 0u);
    EXPECT_EQ(context, 1u);
    ASSERT_EQ(req(ptr(&request), context, 0, 0, 0, 0), 0u);
    EXPECT_EQ(request, 1u);
    EXPECT_EQ(del_req(request, 0, 0, 0, 0, 0), 0u);

    uint8_t out[0x80];
    std::memset(out, 0x77, sizeof(out));
    EXPECT_EQ(record(request, context, ptr(out), 0, 0, 0), kSignedOut);
    EXPECT_EQ(out[0], 0x77) << "a refused record leaves its out untouched";
    EXPECT_EQ(ranking(request, context, 0, 0, 0, 0), kSignedOut);
}

TEST(NpCommerce, HeadlessDialogLifecycle) {
    register_builtin_hle();
    HleFn init = Hle::lookup("0aR2aWmQal4");
    HleFn open = Hle::lookup("DfSCDRA3EjY");
    HleFn status = Hle::lookup("CCbC+lqqvF0");
    HleFn get_result = Hle::lookup("r42bWcQbtZY");
    HleFn close = Hle::lookup("NU3ckGHMFXo");
    HleFn term = Hle::lookup("m-I92Ab50W8");
    HleFn show_icon = Hle::lookup("DHmwsa6S8Tc");
    ASSERT_NE(init, nullptr);
    ASSERT_NE(open, nullptr);
    ASSERT_NE(status, nullptr);
    ASSERT_NE(get_result, nullptr);
    ASSERT_NE(close, nullptr);
    ASSERT_NE(term, nullptr);
    ASSERT_NE(show_icon, nullptr);

    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 0u) << "NONE before Initialize";
    ASSERT_EQ(init(ptr("PPSA00000"), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 1u) << "INITIALIZED after Initialize";
    // Headless: Open auto-dismisses, so the game's "wait until FINISHED" loop still exits.
    ASSERT_EQ(open(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 3u) << "FINISHED after the headless Open";
    // No signed-in user can complete a store transaction; the result stays unwritten.
    uint8_t result[0x100];
    std::memset(result, 0x33, sizeof(result));
    EXPECT_EQ(get_result(0, ptr(result), 0, 0, 0, 0), kSignedOut);
    EXPECT_EQ(result[0], 0x33) << "a refused result must not write its out struct";
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 3u) << "the refused GetResult does not end the dialog";
    ASSERT_EQ(close(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 3u);
    ASSERT_EQ(term(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 0u) << "NONE after Terminate";
    EXPECT_EQ(show_icon(0, 0, 0, 0, 0, 0), 0u);
}

TEST(NpAuth, RequestLifecycleNoCodes) {
    register_builtin_hle();
    HleFn create = Hle::lookup("6bwFkosYRQg");
    HleFn del = Hle::lookup("H8wG9Bk-nPc");
    HleFn timeout = Hle::lookup("PM3IZCw-7m0");
    HleFn get_code = Hle::lookup("KAUXQ9GdWp8");
    HleFn wait_async = Hle::lookup("SK-S7daqJSE");
    ASSERT_NE(create, nullptr);
    ASSERT_NE(del, nullptr);
    ASSERT_NE(timeout, nullptr);
    ASSERT_NE(get_code, nullptr);
    ASSERT_NE(wait_async, nullptr);

    uint32_t request = 0xDEADu;
    ASSERT_EQ(create(ptr(&request), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(request, 1u) << "CreateRequest writes its local id";
    EXPECT_EQ(timeout(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(del(request, 0, 0, 0, 0, 0), 0u);

    uint8_t code[0x80];
    std::memset(code, 0x11, sizeof(code));
    EXPECT_EQ(get_code(request, ptr(code), 0, 0, 0, 0), kSignedOut);
    EXPECT_EQ(code[0], 0x11) << "no authorization code is ever written";
    EXPECT_EQ(wait_async(request, 0, 0, 0, 0, 0), kSignedOut);
}

TEST(NpUtility, InitAndOfflinePaths) {
    register_builtin_hle();
    HleFn init = Hle::lookup("G6iWw8aUQtA");
    HleFn net_is_init = Hle::lookup("JXlTj9RRCFo");
    HleFn net_init = Hle::lookup("DVZE+fAhgFY");
    HleFn lookup_create = Hle::lookup("CQr9UxPHUFs");
    HleFn lookup_np_id = Hle::lookup("D6tnM1Uti4g");
    HleFn bw_start = Hle::lookup("hqzi1IHdQQQ");
    HleFn wordfilter_wait = Hle::lookup("M7ivWj5yKzg");
    ASSERT_NE(init, nullptr);
    ASSERT_NE(net_is_init, nullptr);
    ASSERT_NE(net_init, nullptr);
    ASSERT_NE(lookup_create, nullptr);
    ASSERT_NE(lookup_np_id, nullptr);
    ASSERT_NE(bw_start, nullptr);
    ASSERT_NE(wordfilter_wait, nullptr);

    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(net_is_init(0, 0, 0, 0, 0, 0), 0u) << "the lookup service is not initialized: say so";
    EXPECT_EQ(net_init(ptr("PPSA00000"), 0, 0, 0, 0, 0), kSignedOut);
    uint32_t request = 0xDEADu;
    ASSERT_EQ(lookup_create(ptr(&request), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(request, 1u);
    EXPECT_EQ(lookup_np_id(request, 0, 0, 0, 0, 0), kSignedOut);
    EXPECT_EQ(bw_start(0, 0, 0, 0, 0, 0), kSignedOut);
    EXPECT_EQ(wordfilter_wait(0, 0, 0, 0, 0, 0), kSignedOut);
}

TEST(NpMatching2, NoRoomState) {
    register_builtin_hle();
    HleFn set_port = Hle::lookup("AupHEf8WOhM");
    HleFn establish = Hle::lookup("EcYuZkNhHI8");
    HleFn create_ctx = Hle::lookup("Kxlf9+pa0GY");
    HleFn worlds = Hle::lookup("FagjVl+bHFI");
    ASSERT_NE(set_port, nullptr);
    ASSERT_NE(establish, nullptr);
    ASSERT_NE(create_ctx, nullptr);
    ASSERT_NE(worlds, nullptr);

    EXPECT_EQ(set_port(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(establish(0, 0, 0, 0, 0, 0), kSignedOut) << "signaling is PSN-backed";
    // The context out-width is unverified, so the constructor refuses rather than write it.
    uint8_t ctx_bytes[0x10];
    std::memset(ctx_bytes, 0x22, sizeof(ctx_bytes));
    EXPECT_EQ(create_ctx(ptr(ctx_bytes), 0, 0, 0, 0, 0), kSignedOut);
    EXPECT_EQ(ctx_bytes[0], 0x22) << "the unverified out-width is never written";
    EXPECT_EQ(worlds(0, 0, 0, 0, 0, 0), kSignedOut);
}

TEST(NpSns, NoLinkedAccounts) {
    register_builtin_hle();
    HleFn fb_create = Hle::lookup("D+F4GKuY3oE");
    HleFn fb_token = Hle::lookup("FtyS8XLBqNE");
    HleFn linked = Hle::lookup("AW7NonbfeFk");
    HleFn fb_abort = Hle::lookup("PLJbF9b-Who");
    HleFn yt_create = Hle::lookup("Ho3XDEydBjM");
    HleFn yt_token = Hle::lookup("FElwHkpLvmw");
    ASSERT_NE(fb_create, nullptr);
    ASSERT_NE(fb_token, nullptr);
    ASSERT_NE(linked, nullptr);
    ASSERT_NE(fb_abort, nullptr);
    ASSERT_NE(yt_create, nullptr);
    ASSERT_NE(yt_token, nullptr);

    uint32_t request = 0xDEADu;
    ASSERT_EQ(fb_create(ptr(&request), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(request, 1u);
    EXPECT_EQ(fb_abort(request, 0, 0, 0, 0, 0), 0u);

    uint8_t token[0x40];
    std::memset(token, 0x44, sizeof(token));
    EXPECT_EQ(fb_token(request, ptr(token), 0, 0, 0, 0), kSignedOut);
    EXPECT_EQ(token[0], 0x44) << "no SNS token is ever written";
    EXPECT_EQ(linked(0, 0, 0, 0, 0, 0), kSignedOut);
    uint32_t yt = 0;
    ASSERT_EQ(yt_create(ptr(&yt), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(yt_token(yt, ptr(token), 0, 0, 0, 0), kSignedOut);
}
