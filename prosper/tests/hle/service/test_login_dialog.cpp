// test_login_dialog — the remaining dialog entry points run their headless lifecycles.
//
// libSceLoginDialog (beyond Initialize), the three MsgDialog progress-bar setters and the whole
// two-call libSceWebBrowserDialog were unregistered, so the dispatcher answered `0`. For the
// lifecycles 0 reads as NONE forever — a title polling GetStatus for FINISHED waits forever —
// so each lifecycle below pins INITIALIZED -> FINISHED -> NONE through the real NIDs, mirroring
// the MsgDialog/SigninDialog contracts. The progress bars are display-only setters with no
// out-parameters: acknowledged, with refusal arms proving nothing observable changes either way.
//
// NID provenance: all names hash cleanly with nid_hash; LoginDialogInitialize reproduces the
// already-registered qP-EvQRl2Hc. LoginDialog/MediaDialog GetResult entry points stay
// UNREGISTERED on purpose (unpinned result layouts — same rationale as SigninDialog GetResult).
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

TEST(LoginDialog, Contract) {
    register_service_hle();
    HleFn init = Hle::lookup(nid_hash("sceLoginDialogInitialize"));
    HleFn open = Hle::lookup(nid_hash("sceLoginDialogOpen"));
    HleFn update = Hle::lookup(nid_hash("sceLoginDialogUpdateStatus"));
    HleFn status = Hle::lookup(nid_hash("sceLoginDialogGetStatus"));
    HleFn close = Hle::lookup(nid_hash("sceLoginDialogClose"));
    HleFn term = Hle::lookup(nid_hash("sceLoginDialogTerminate"));
    for (HleFn f : {init, open, update, status, close, term}) ASSERT_NE(f, nullptr);
    EXPECT_EQ(nid_hash("sceLoginDialogInitialize"), "qP-EvQRl2Hc")
        << "Initialize keeps its firmware NID";

    EXPECT_EQ(update(0, 0, 0, 0, 0, 0), 0u) << "no dialog before Initialize reads as NONE";
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(update(0, 0, 0, 0, 0, 0), 1u) << "INITIALIZED after Initialize";
    EXPECT_EQ(open(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(update(0, 0, 0, 0, 0, 0), 3u) << "headless Open auto-dismisses to FINISHED";
    EXPECT_EQ(status(0, 0, 0, 0, 0, 0), 3u) << "GetStatus agrees with UpdateStatus";
    EXPECT_EQ(close(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(update(0, 0, 0, 0, 0, 0), 0u) << "Close clears to NONE";
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(term(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(update(0, 0, 0, 0, 0, 0), 0u) << "Terminate clears to NONE";
    EXPECT_EQ(Hle::lookup(nid_hash("sceLoginDialogGetResult")), nullptr)
        << "GetResult stays unregistered: unpinned layout";
}

TEST(MsgDialog, ProgressBarsAcknowledged) {
    register_service_hle();
    HleFn inc = Hle::lookup(nid_hash("sceMsgDialogProgressBarInc"));
    HleFn set_msg = Hle::lookup(nid_hash("sceMsgDialogProgressBarSetMsg"));
    HleFn set_value = Hle::lookup(nid_hash("sceMsgDialogProgressBarSetValue"));
    for (HleFn f : {inc, set_msg, set_value}) ASSERT_NE(f, nullptr);
    // Display-only: no out-parameters exist, so acknowledgement is the whole contract.
    EXPECT_EQ(inc(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(set_msg(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(set_value(0, 50, 0, 0, 0, 0), 0u);
}

TEST(WebBrowserDialog, Lifecycle) {
    register_service_hle();
    HleFn init = Hle::lookup(nid_hash("sceWebBrowserDialogInitialize"));
    HleFn term = Hle::lookup(nid_hash("sceWebBrowserDialogTerminate"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(term, nullptr);
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(term(0, 0, 0, 0, 0, 0), 0u) << "the two-call firmware surface opens and closes";
}
