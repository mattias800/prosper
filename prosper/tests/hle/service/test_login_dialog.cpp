// test_login_dialog — the remaining dialog entry points run their headless lifecycles.
//
// libSceLoginDialog (beyond Initialize), the three MsgDialog progress-bar setters and
// libSceWebBrowserDialog's Initialize/Terminate were unregistered, so the dispatcher answered `0`.
// For LoginDialog that 0 read as NONE forever; once Open finishes, the importing titles call
// GetResult and treat a first result word of 0 as a successful login, so GetResult must report
// "not completed". The lifecycle and every error code below are the shipped libSceLoginDialog
// module's (see the handler block in hle_service.cpp).
//
// NID provenance: all names hash cleanly with nid_hash; LoginDialogInitialize reproduces the
// already-registered qP-EvQRl2Hc.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

using namespace prosper;

namespace {
uint64_t addr(const void* p) { return (uint64_t)(uintptr_t)p; }

struct Login {
    HleFn init, open, update, status, result, close, term;
};
Login login() {
    register_service_hle();
    Login l{Hle::lookup(nid_hash("sceLoginDialogInitialize")),
            Hle::lookup(nid_hash("sceLoginDialogOpen")),
            Hle::lookup(nid_hash("sceLoginDialogUpdateStatus")),
            Hle::lookup(nid_hash("sceLoginDialogGetStatus")),
            Hle::lookup(nid_hash("sceLoginDialogGetResult")),
            Hle::lookup(nid_hash("sceLoginDialogClose")),
            Hle::lookup(nid_hash("sceLoginDialogTerminate"))};
    // The dialog status is process-wide; start every case from NONE whatever ran before.
    if (l.term) l.term(0, 0, 0, 0, 0, 0);
    return l;
}
// A valid Open param: size 0x40, selector 0, reserved words zero.
std::array<uint32_t, 16> open_param() {
    std::array<uint32_t, 16> p{};
    p[0] = 0x40u;
    return p;
}
constexpr uint64_t kNotInitialized = 0x81340001ull;
constexpr uint64_t kAlreadyInitialized = 0x81340002ull;
constexpr uint64_t kParamInvalid = 0x81340003ull;
constexpr uint64_t kBusy = 0x81340005ull;
}  // namespace

TEST(LoginDialog, LifecycleReachesFinishedAndCloseKeepsIt) {
    const Login l = login();
    for (HleFn f : {l.init, l.open, l.update, l.status, l.result, l.close, l.term})
        ASSERT_NE(f, nullptr);
    EXPECT_EQ(nid_hash("sceLoginDialogInitialize"), "qP-EvQRl2Hc")
        << "Initialize keeps its firmware NID";
    auto param = open_param();

    EXPECT_EQ(l.update(0, 0, 0, 0, 0, 0), 0u) << "no dialog before Initialize reads as NONE";
    EXPECT_EQ(l.init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(l.update(0, 0, 0, 0, 0, 0), 1u) << "INITIALIZED after Initialize";
    EXPECT_EQ(l.open(addr(param.data()), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(l.update(0, 0, 0, 0, 0, 0), 3u) << "headless Open auto-dismisses to FINISHED";
    EXPECT_EQ(l.status(0, 0, 0, 0, 0, 0), 3u) << "GetStatus agrees with UpdateStatus";
    EXPECT_EQ(l.close(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(l.update(0, 0, 0, 0, 0, 0), 3u)
        << "Close stores FINISHED (firmware 0x1e3b), so close-then-wait loops exit";
    EXPECT_EQ(l.term(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(l.update(0, 0, 0, 0, 0, 0), 0u) << "Terminate clears to NONE";
}

TEST(LoginDialog, GetResultReportsThatNoLoginHappened) {
    const Login l = login();
    ASSERT_NE(l.result, nullptr) << "GetResult must be registered: titles read it after FINISHED";
    std::array<uint32_t, 4> r{};
    EXPECT_EQ(l.result(0, 0, 0, 0, 0, 0), kParamInvalid) << "NULL result";
    EXPECT_EQ(l.result(addr(r.data()), 0, 0, 0, 0, 0), kNotInitialized) << "before Initialize";
    EXPECT_EQ(r[0], 0u) << "nothing written before Initialize";

    ASSERT_EQ(l.init(0, 0, 0, 0, 0, 0), 0u);
    r = {};
    EXPECT_EQ(l.result(addr(r.data()), 0, 0, 0, 0, 0), kBusy) << "not finished yet";
    EXPECT_EQ(r[0], 1u) << "the default is written before the finished check";

    auto param = open_param();
    ASSERT_EQ(l.open(addr(param.data()), 0, 0, 0, 0, 0), 0u);
    r = {};   // the titles zero the struct before calling GetResult
    EXPECT_EQ(l.result(addr(r.data()), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(r[0], 1u) << "first word 1 = not completed: a zeroed struct must not read as success";
    EXPECT_EQ(r[1], 0x81340007u) << "the module's no-service default";
    EXPECT_EQ(r[2], 0u) << "only 8 bytes are written";
    EXPECT_EQ(l.term(0, 0, 0, 0, 0, 0), 0u);
}

TEST(LoginDialog, OutOfOrderCallsAndBadParamsReturnTheModulesCodes) {
    const Login l = login();
    auto param = open_param();
    EXPECT_EQ(l.term(0, 0, 0, 0, 0, 0), kNotInitialized) << "Terminate from NONE";
    EXPECT_EQ(l.close(0, 0, 0, 0, 0, 0), kNotInitialized) << "Close from NONE";
    EXPECT_EQ(l.open(addr(param.data()), 0, 0, 0, 0, 0), kNotInitialized) << "Open from NONE";
    EXPECT_EQ(l.update(0, 0, 0, 0, 0, 0), 0u) << "a refused Open leaves NONE";

    ASSERT_EQ(l.init(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(l.init(0, 0, 0, 0, 0, 0), kAlreadyInitialized) << "Initialize twice";
    EXPECT_EQ(l.open(0, 0, 0, 0, 0, 0), kParamInvalid) << "NULL param";
    auto bad = param;
    bad[0] = 0x3cu;
    EXPECT_EQ(l.open(addr(bad.data()), 0, 0, 0, 0, 0), kParamInvalid) << "size != 0x40";
    bad = param;
    bad[1] = 2u;
    EXPECT_EQ(l.open(addr(bad.data()), 0, 0, 0, 0, 0), kParamInvalid) << "selector >= 2";
    for (int w = 0x2c / 4; w <= 0x3c / 4; ++w) {
        bad = param;
        bad[w] = 1u;
        EXPECT_EQ(l.open(addr(bad.data()), 0, 0, 0, 0, 0), kParamInvalid)
            << "non-zero reserved word at +" << std::hex << w * 4;
    }
    EXPECT_EQ(l.update(0, 0, 0, 0, 0, 0), 1u) << "every refused Open left the dialog INITIALIZED";
    EXPECT_EQ(l.term(0, 0, 0, 0, 0, 0), 0u);
}

TEST(MsgDialog, ProgressBarsKeepTheDispatcherDefault) {
    register_service_hle();
    HleFn inc = Hle::lookup(nid_hash("sceMsgDialogProgressBarInc"));
    HleFn set_msg = Hle::lookup(nid_hash("sceMsgDialogProgressBarSetMsg"));
    HleFn set_value = Hle::lookup(nid_hash("sceMsgDialogProgressBarSetValue"));
    for (HleFn f : {inc, set_msg, set_value}) ASSERT_NE(f, nullptr);
    // Guest-identical to the unregistered dispatcher; see the handler comment for why.
    EXPECT_EQ(inc(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(set_msg(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(set_value(0, 50, 0, 0, 0, 0), 0u);
}

TEST(WebBrowserDialog, OnlyInitializeAndTerminateAreRegistered) {
    register_service_hle();
    EXPECT_NE(Hle::lookup(nid_hash("sceWebBrowserDialogInitialize")), nullptr);
    EXPECT_NE(Hle::lookup(nid_hash("sceWebBrowserDialogTerminate")), nullptr);
    // The real lifecycle is #4463; until then Open must stay visibly unimplemented.
    EXPECT_EQ(Hle::lookup(nid_hash("sceWebBrowserDialogOpen")), nullptr);
}
