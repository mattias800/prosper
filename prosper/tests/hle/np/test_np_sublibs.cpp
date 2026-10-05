// test_np_sublibs — the batch-2 NP sub-libraries on the honest signed-out console.
//
// The NP sub-libraries of this batch — libSceNpTrophy (v1), libSceNpTus, libSceNpScore,
// libSceNpCommerce, libSceNpAuth, libSceNpUtility, libSceNpMatching2 and libSceNpSns — were
// unregistered, so every call reached the dispatcher's `return 0` — SCE_OK: trophy content
// "queried" into unwritten memory (the #213 class), TUS/Score data "fetched" from a PSN the
// console is not signed in to, a PS Store "transaction" that never happened, and OAuth/SNS
// tokens "issued" to a console with no user. This pins the batch: request/context constructors
// that RETURN a local id, SIGNED_OUT where the PSN path is needed, the Trophy2 answer for v1
// trophy content, and the headless dialog lifecycle where there is no interactive UI.
//
// Every lookup goes through nid_hash(name), never a hand-typed NID: an earlier revision registered
// 157 mistyped NIDs and its tests looked them up with the same literals, so nothing could notice.
// A few known-answer NIDs copied from the PS5 3.20 reference pin nid_hash itself from outside.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <string>

using namespace prosper;

namespace {

constexpr uint64_t kSignedOut = 0x80550006ull;           // SCE_NP_ERROR_SIGNED_OUT (NpManager)
constexpr uint64_t kTrophyUnavailable = 0x80551500ull;   // what NpTrophy2 answers for content

enum Kind {
    kOkKind,                  // succeeds, writes nothing
    kSignedOutKind,           // SIGNED_OUT, writes nothing
    kTrophyUnavailableKind,   // the Trophy2 content answer, writes nothing
    kReturnedIdKind,          // returns a positive id; arg0 is NOT an out-pointer
    kWrittenIdKind,           // PS4 trophy ABI: writes an s32 id through arg0, returns SCE_OK
};

struct Entry {
    const char* name;
    Kind kind;
};

// Every non-dialog export this batch registers, with the contract it must keep.
const Entry kTable[] = {
    {"sceNpTrophyCreateContext", kWrittenIdKind},
    {"sceNpTrophyCreateHandle", kWrittenIdKind},
    {"sceNpTrophyIntCreateHandle", kWrittenIdKind},
    {"sceNpTrophyRegisterContext", kOkKind},
    {"sceNpTrophyDestroyContext", kOkKind},
    {"sceNpTrophyDestroyHandle", kOkKind},
    {"sceNpTrophyAbortHandle", kOkKind},
    {"sceNpTrophyIntAbortHandle", kOkKind},
    {"sceNpTrophySystemRemoveAll", kOkKind},
    {"sceNpTrophySystemUnregisterTitleSyncedCallback", kOkKind},
    {"sceNpTrophySystemUnregisterTitleUpdateCallback", kOkKind},
    {"sceNpTrophySystemWrapDebugLockTrophy", kOkKind},
    {"sceNpTrophySystemWrapDebugUnlockTrophy", kOkKind},
    {"sceNpTrophySystemWrapRemoveUserData", kOkKind},
    {"sceNpTrophyGetGameInfo", kTrophyUnavailableKind},
    {"sceNpTrophyGetTrophyUnlockState", kTrophyUnavailableKind},
    {"sceNpTrophyGetGroupInfo", kTrophyUnavailableKind},
    {"sceNpTrophyGetTrophyInfo", kTrophyUnavailableKind},
    {"sceNpTrophyGetGroupIcon", kTrophyUnavailableKind},
    {"sceNpTrophyGetTrophyIcon", kTrophyUnavailableKind},
    {"sceNpTrophyNumInfoGetTotal", kTrophyUnavailableKind},
    {"sceNpTrophyUnlockTrophy", kTrophyUnavailableKind},
    {"sceNpTrophyShowTrophyList", kTrophyUnavailableKind},
    {"sceNpTrophyIntGetProgress", kTrophyUnavailableKind},
    {"sceNpTrophyIntNetSyncTitle", kTrophyUnavailableKind},
    {"sceNpTrophyIntNetSyncTitles", kTrophyUnavailableKind},
    {"sceNpTrophySystemDbgCtl", kTrophyUnavailableKind},
    {"sceNpTrophySystemWrapGetGroupDetails", kTrophyUnavailableKind},
    {"sceNpTrophySystemWrapGetPlayedTrophyTitles", kTrophyUnavailableKind},
    {"sceNpTrophySystemWrapGetTitleDetails", kTrophyUnavailableKind},
    {"sceNpTrophySystemWrapGetTrophyDetailsArray", kTrophyUnavailableKind},
    {"sceNpTrophySystemWrapGetTrophyTitleIdsByNpTitleId", kTrophyUnavailableKind},
    {"sceNpTusCreateTitleCtx", kReturnedIdKind},
    {"sceNpTusCreateNpTitleCtxA", kReturnedIdKind},
    {"sceNpTssCreateNpTitleCtx", kReturnedIdKind},
    {"sceNpTssCreateNpTitleCtxA", kReturnedIdKind},
    {"sceNpTusCreateRequest", kReturnedIdKind},
    {"sceNpScoreCreateTitleCtx", kReturnedIdKind},
    {"sceNpScoreCreateRequest", kReturnedIdKind},
    {"sceNpTusAbortRequest", kOkKind},
    {"sceNpTusSetThreadParam", kOkKind},
    {"sceNpScoreAbortRequest", kOkKind},
    {"sceNpScoreDeleteRequest", kOkKind},
    {"sceNpScoreSetThreadParam", kOkKind},
    {"sceNpScoreSetTimeout", kOkKind},
    {"sceNpTusAddAndGetVariable", kSignedOutKind},
    {"sceNpTusAddAndGetVariableA", kSignedOutKind},
    {"sceNpTusAddAndGetVariableAAsync", kSignedOutKind},
    {"sceNpTusAddAndGetVariableAsync", kSignedOutKind},
    {"sceNpTusAddAndGetVariableVUser", kSignedOutKind},
    {"sceNpTusDeleteMultiSlotData", kSignedOutKind},
    {"sceNpTusDeleteMultiSlotDataA", kSignedOutKind},
    {"sceNpTusDeleteMultiSlotDataAAsync", kSignedOutKind},
    {"sceNpTusDeleteMultiSlotDataVUser", kSignedOutKind},
    {"sceNpTusDeleteMultiSlotVariable", kSignedOutKind},
    {"sceNpTusDeleteMultiSlotVariableA", kSignedOutKind},
    {"sceNpTusGetData", kSignedOutKind},
    {"sceNpTusGetDataA", kSignedOutKind},
    {"sceNpTusGetDataAAsync", kSignedOutKind},
    {"sceNpTusGetDataAVUser", kSignedOutKind},
    {"sceNpTusGetDataAVUserAsync", kSignedOutKind},
    {"sceNpTusGetDataAsync", kSignedOutKind},
    {"sceNpTusGetDataForCrossSave", kSignedOutKind},
    {"sceNpTusGetDataForCrossSaveVUser", kSignedOutKind},
    {"sceNpTusGetDataVUserAsync", kSignedOutKind},
    {"sceNpTusGetFriendsDataStatusA", kSignedOutKind},
    {"sceNpTusGetFriendsVariable", kSignedOutKind},
    {"sceNpTusGetFriendsVariableAAsync", kSignedOutKind},
    {"sceNpTusGetFriendsVariableAsync", kSignedOutKind},
    {"sceNpTusGetMultiSlotDataStatus", kSignedOutKind},
    {"sceNpTusGetMultiSlotDataStatusA", kSignedOutKind},
    {"sceNpTusGetMultiSlotVariableAsync", kSignedOutKind},
    {"sceNpTusGetMultiSlotVariableVUser", kSignedOutKind},
    {"sceNpTusGetMultiUserDataStatusA", kSignedOutKind},
    {"sceNpTusGetMultiUserVariable", kSignedOutKind},
    {"sceNpTusGetMultiUserVariableAsync", kSignedOutKind},
    {"sceNpTusGetMultiUserVariableVUser", kSignedOutKind},
    {"sceNpTusPollAsync", kSignedOutKind},
    {"sceNpTusSetData", kSignedOutKind},
    {"sceNpTusSetDataA", kSignedOutKind},
    {"sceNpTusSetDataAAsync", kSignedOutKind},
    {"sceNpTusSetDataAVUser", kSignedOutKind},
    {"sceNpTusSetDataVUser", kSignedOutKind},
    {"sceNpTusSetMultiSlotVariable", kSignedOutKind},
    {"sceNpTusSetMultiSlotVariableA", kSignedOutKind},
    {"sceNpTusSetMultiSlotVariableAsync", kSignedOutKind},
    {"sceNpTusSetMultiSlotVariableVUser", kSignedOutKind},
    {"sceNpTusTryAndSetVariable", kSignedOutKind},
    {"sceNpTusTryAndSetVariableA", kSignedOutKind},
    {"sceNpTusTryAndSetVariableAAsync", kSignedOutKind},
    {"sceNpTusTryAndSetVariableAVUser", kSignedOutKind},
    {"sceNpTusTryAndSetVariableAsync", kSignedOutKind},
    {"sceNpTusTryAndSetVariableVUser", kSignedOutKind},
    {"sceNpTusWaitAsync", kSignedOutKind},
    {"sceNpTssGetData", kSignedOutKind},
    {"sceNpTssGetSmallStorage", kSignedOutKind},
    {"sceNpTssGetSmallStorageAsync", kSignedOutKind},
    {"sceNpTssGetStorageAsync", kSignedOutKind},
    {"sceNpScoreCensorComment", kSignedOutKind},
    {"sceNpScoreCensorCommentAsync", kSignedOutKind},
    {"sceNpScoreGetBoardInfoAsync", kSignedOutKind},
    {"sceNpScoreGetFriendsRanking", kSignedOutKind},
    {"sceNpScoreGetFriendsRankingA", kSignedOutKind},
    {"sceNpScoreGetGameData", kSignedOutKind},
    {"sceNpScoreGetRankingByNpId", kSignedOutKind},
    {"sceNpScorePollAsync", kSignedOutKind},
    {"sceNpScoreRecordGameData", kSignedOutKind},
    {"sceNpScoreRecordGameDataAsync", kSignedOutKind},
    {"sceNpScoreRecordScore", kSignedOutKind},
    {"sceNpScoreSanitizeComment", kSignedOutKind},
    {"sceNpScoreWaitAsync", kSignedOutKind},
    {"sceNpCommerceShowPsStoreIcon", kOkKind},
    {"sceNpCommerceHidePsStoreIcon", kOkKind},
    {"sceNpCommerceSetPsStoreIconLayout", kOkKind},
    {"sceNpAuthCreateRequest", kReturnedIdKind},
    {"sceNpAuthCreateAsyncRequest", kReturnedIdKind},
    {"sceNpAuthAbortRequest", kOkKind},
    {"sceNpAuthDeleteRequest", kOkKind},
    {"sceNpAuthSetTimeout", kOkKind},
    {"sceNpAuthPollAsync", kSignedOutKind},
    {"sceNpAuthWaitAsync", kSignedOutKind},
    {"sceNpAuthGetAuthorizationCodeA", kSignedOutKind},
    {"sceNpAuthGetAuthorizationCodeV3", kSignedOutKind},
    {"sceNpAuthGetIdToken", kSignedOutKind},
    {"sceNpAuthGetIdTokenV3", kSignedOutKind},
    {"sceNpUtilityInit", kOkKind},
    {"sceNpAppInfoIntInitialize", kOkKind},
    {"sceNpAppInfoIntFinalize", kOkKind},
    {"sceNpBandwidthTestShutdown", kOkKind},
    {"sceNpBandwidthTestAbort", kOkKind},
    {"sceNpBandwidthTestInitStartDownload", kSignedOutKind},
    {"sceNpBandwidthTestInitStartUpload", kSignedOutKind},
    {"sceNpBandwidthTestGetStatus", kSignedOutKind},
    {"sceNpLookupNetInit", kSignedOutKind},
    {"sceNpLookupNetIsInit", kOkKind},
    {"sceNpLookupCreateRequest", kReturnedIdKind},
    {"sceNpLookupCreateTitleCtx", kReturnedIdKind},
    {"sceNpLookupDeleteRequest", kOkKind},
    {"sceNpLookupDeleteTitleCtx", kOkKind},
    {"sceNpLookupAbortRequest", kOkKind},
    {"sceNpLookupSetTimeout", kOkKind},
    {"sceNpLookupNetNpId", kSignedOutKind},
    {"sceNpLookupNpId", kSignedOutKind},
    {"sceNpLookupPollAsync", kSignedOutKind},
    {"sceNpLookupWaitAsync", kSignedOutKind},
    {"sceNpWordFilterPollAsync", kSignedOutKind},
    {"sceNpWordFilterWaitAsync", kSignedOutKind},
    {"sceNpMatching2SetExtraInitParam", kOkKind},
    {"sceNpMatching2SignalingSetPort", kOkKind},
    {"sceNpMatching2SignalingAbortConnection", kOkKind},
    {"sceNpMatching2CreateContextInternal", kSignedOutKind},
    {"sceNpMatching2GetRoomJoinedSlotMaskLocal", kSignedOutKind},
    {"sceNpMatching2GetWorldIdArrayForAllServers", kSignedOutKind},
    {"sceNpMatching2SetRoomDataInternalExt", kSignedOutKind},
    {"sceNpMatching2SignalingEstablishConnection", kSignedOutKind},
    {"sceNpMatching2SignalingGetPort", kSignedOutKind},
    {"sceNpSnsIntCreateRequest", kReturnedIdKind},
    {"sceNpSnsTwitchCreateRequest", kReturnedIdKind},
    {"sceNpSnsYouTubeCreateRequest", kReturnedIdKind},
    {"sceNpSnsIntDeleteRequest", kOkKind},
    {"sceNpSnsIntAbortRequest", kOkKind},
    {"sceNpSnsFacebookAbortRequest", kOkKind},
    {"sceNpSnsFacebookDeleteRequest", kOkKind},
    {"sceNpSnsTwitchAbortRequest", kOkKind},
    {"sceNpSnsYouTubeAbortRequest", kOkKind},
    {"sceNpSnsYouTubeDeleteRequest", kOkKind},
    {"sceNpSnsFacebookGetAccessToken", kSignedOutKind},
    {"sceNpSnsIntFbGetGameAccessToken", kSignedOutKind},
    {"sceNpSnsIntFbGetGameAccessTokenAllowed", kSignedOutKind},
    {"sceNpSnsIntFbGetSystemAccessToken", kSignedOutKind},
    {"sceNpSnsIntTwGetSystemAccessToken", kSignedOutKind},
    {"sceNpSnsIntYtGetAccessToken", kSignedOutKind},
    {"sceNpSnsIntYtRefreshMasterToken", kSignedOutKind},
    {"sceNpSnsIntLinkedStatus", kSignedOutKind},
    {"sceNpSnsIntUnlink", kSignedOutKind},
    {"sceNpSnsTwitchGetAccessToken", kSignedOutKind},
    {"sceNpSnsYouTubeGetAccessToken", kSignedOutKind},
};

uint64_t ptr(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

HleFn fn(const char* name) {
    return Hle::lookup(nid_hash(name));
}

struct Sentinels {
    uint8_t buf[6][0x80];
    Sentinels() { std::memset(buf, 0xA5, sizeof(buf)); }
    bool intact(int from = 0) const {
        for (int i = from; i < 6; i++)
            for (uint8_t b : buf[i])
                if (b != 0xA5) return false;
        return true;
    }
    uint64_t arg(int i) const { return ptr(buf[i]); }
};

}  // namespace

TEST(NpSublibs, NidHashMatchesThePs5Reference) {
    // Known answers copied from the PS5 3.20 library reference, i.e. from outside nid_hash: if the
    // hash drifted, the name-keyed table below would still agree with itself.
    EXPECT_EQ(nid_hash("sceNpTrophyAbortHandle"), "aTnHs7W-9Uk");
    EXPECT_EQ(nid_hash("sceNpCommerceDialogInitialize"), "0aR2aWmQal4");
    EXPECT_EQ(nid_hash("sceNpTusCreateTitleCtx"), "hhy8+oecGac");
    EXPECT_EQ(nid_hash("sceNpAuthCreateAsyncRequest"), "N+mr7GjTvr8");
    register_builtin_hle();
    EXPECT_NE(Hle::lookup("aTnHs7W-9Uk"), nullptr) << "the literal 3.20 NID itself is bound";
    EXPECT_NE(Hle::lookup("hhy8+oecGac"), nullptr) << "the literal 3.20 NID itself is bound";
}

TEST(NpSublibs, EveryExportKeepsItsContract) {
    register_builtin_hle();
    std::set<std::string> seen;
    for (const Entry& e : kTable) {
        SCOPED_TRACE(e.name);
        EXPECT_TRUE(seen.insert(e.name).second) << "listed once";
        HleFn f = fn(e.name);
        ASSERT_NE(f, nullptr) << "bound under nid_hash(name)";
        EXPECT_STREQ(Hle::name_of(nid_hash(e.name)), e.name) << "name table agrees";
        Sentinels s;
        const uint64_t r = f(s.arg(0), s.arg(1), s.arg(2), s.arg(3), s.arg(4), s.arg(5));
        switch (e.kind) {
            case kOkKind:
                EXPECT_EQ(r, 0u);
                EXPECT_TRUE(s.intact()) << "a lifecycle success writes nothing";
                break;
            case kSignedOutKind:
                EXPECT_EQ(r, kSignedOut);
                EXPECT_TRUE(s.intact()) << "a refused call leaves every out-parameter untouched";
                break;
            case kTrophyUnavailableKind:
                EXPECT_EQ(r, kTrophyUnavailable) << "same answer as NpTrophy2";
                EXPECT_TRUE(s.intact()) << "a refused query must not write its out struct";
                break;
            case kReturnedIdKind:
                EXPECT_GT((int64_t)(int32_t)r, 0) << "the id comes back in eax";
                EXPECT_TRUE(s.intact()) << "arg0 is not an out-pointer (kills: writing the id "
                                           "through arg0 over a live guest object)";
                break;
            case kWrittenIdKind:
                EXPECT_EQ(r, 0u);
                EXPECT_EQ(*reinterpret_cast<const int32_t*>(s.buf[0]), 1) << "id written via arg0";
                EXPECT_TRUE(s.intact(1)) << "nothing beyond arg0 is written";
                break;
        }
    }
}

TEST(NpSublibs, ConstructorsReturnDistinctIdsAndNeverDereferenceArg0) {
    register_builtin_hle();
    // TUS/Score/Lookup take VALUES in arg0 (serviceLabel, titleCtxId); dereferencing it would
    // fault at guest address 1. Uncharted calls the argument-less sceNpAuthCreateRequest with rdi
    // still pointing at a live object.
    const uint64_t a = fn("sceNpScoreCreateRequest")(1, 0, 0, 0, 0, 0);
    const uint64_t b = fn("sceNpScoreCreateRequest")(1, 0, 0, 0, 0, 0);
    EXPECT_GT((int32_t)a, 0);
    EXPECT_GT((int32_t)b, 0);
    EXPECT_NE(a, b) << "each request is a distinct id (kills: constant-1 stub)";
    uint64_t live_object = 0x1122334455667788ull;
    const uint64_t id = fn("sceNpAuthCreateRequest")(ptr(&live_object), 0, 0, 0, 0, 0);
    EXPECT_GT((int32_t)id, 0);
    EXPECT_EQ(live_object, 0x1122334455667788ull) << "the caller's object is untouched";
}

TEST(NpTrophyV1, LifecycleIdsMatchTrophy2) {
    register_builtin_hle();
    uint32_t ctx = 0xDEADu, handle = 0xDEADu;
    ASSERT_EQ(fn("sceNpTrophyCreateContext")(ptr(&ctx), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(ctx, 1u) << "CreateContext writes its local id (PS4 ABI, as Trophy2)";
    ASSERT_EQ(fn("sceNpTrophyCreateHandle")(ptr(&handle), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(handle, 1u);
    // A value that is not a plausible guest pointer is never dereferenced.
    EXPECT_EQ(fn("sceNpTrophyCreateHandle")(8, 0, 0, 0, 0, 0), 0u);
    uint8_t unlock_state[0x80];
    std::memset(unlock_state, 0xAA, sizeof(unlock_state));
    EXPECT_EQ(fn("sceNpTrophyGetTrophyUnlockState")(ctx, handle, ptr(unlock_state), 0, 0, 0),
              kTrophyUnavailable)
        << "unlock state is refused, not SCE_OK with an unwritten flag array (#213)";
    EXPECT_EQ(unlock_state[0], 0xAA);
    EXPECT_EQ(fn("sceNpTrophyDestroyHandle")(handle, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(fn("sceNpTrophyDestroyContext")(ctx, 0, 0, 0, 0, 0), 0u);
}

TEST(NpCommerce, HeadlessDialogLifecycle) {
    register_builtin_hle();
    HleFn status = fn("sceNpCommerceDialogGetStatus");
    HleFn update = fn("sceNpCommerceDialogUpdateStatus");
    ASSERT_NE(status, nullptr);
    ASSERT_NE(update, nullptr);
    ASSERT_EQ(fn("sceNpCommerceDialogTerminate")(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 0u) << "NONE before Initialize";
    ASSERT_EQ(fn("sceNpCommerceDialogInitialize")(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 1u) << "INITIALIZED after Initialize";
    // Headless: Open auto-dismisses, so the game's "wait until FINISHED" loop (Dead Cells polls
    // UpdateStatus for 3) still exits.
    ASSERT_EQ(fn("sceNpCommerceDialogOpen")(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(update(0, 0, 0, 0, 0, 0), 3u) << "FINISHED after the headless Open";
    ASSERT_EQ(fn("sceNpCommerceDialogOpen2")(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 3u);
    // GetResult writes "nothing completed": titles read the authorized byte at +4 without
    // checking the return (PPSA20052), so it must not be left as caller stack.
    uint8_t result[0x20];
    std::memset(result, 0x33, sizeof(result));
    EXPECT_EQ(fn("sceNpCommerceDialogGetResult")(ptr(result), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(*reinterpret_cast<int32_t*>(result), 1) << "result: USER_CANCELED";
    EXPECT_EQ(result[4], 0u) << "authorized = false (kills: leaving the caller's byte)";
    EXPECT_EQ(result[5], 0x33u) << "nothing past the authorized byte is written";
    EXPECT_EQ(fn("sceNpCommerceDialogGetResult")(0, 0, 0, 0, 0, 0), 0x80550003ull)
        << "a null result pointer is refused, not dereferenced";
    ASSERT_EQ(fn("sceNpCommerceDialogClose")(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 3u);
    ASSERT_EQ(fn("sceNpCommerceDialogTerminate")(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 0u) << "NONE after Terminate";
}
