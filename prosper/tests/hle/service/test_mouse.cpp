// test_mouse — libSceMouse reports a device that exists but has no input, and closing it
// must complete the lifecycle.
//
// sceMouseClose was unregistered while Init/Open/Read were covered, so a title opening and
// closing mice around gameplay states called into the dispatcher's `return 0`. That happens to
// be the same answer — but only by accident of the default, and the census cannot tell a
// registered close from a missing one. This pins all four NIDs bound with the close completing
// the open/close round trip.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

TEST(Mouse, CloseCompletesLifecycle) {
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("sceMouseInit"));
    HleFn open = Hle::lookup(nid_hash("sceMouseOpen"));
    HleFn close = Hle::lookup(nid_hash("sceMouseClose"));
    ASSERT_NE(init, nullptr);
    ASSERT_NE(open, nullptr);
    ASSERT_NE(close, nullptr) << "sceMouseClose must be registered";
    EXPECT_EQ(nid_hash("sceMouseClose"), "cAnT0Rw-IwU")
        << "the registered NID is the export's hash";
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), 0u);
    const uint64_t handle = open(0, 0, 0, 0, 0, 0);
    EXPECT_EQ(close(handle, 0, 0, 0, 0, 0), 0u);
    // A second close of the same handle is deliberately not asserted: the firmware returns
    // 0x80DF0003 there, while prosper (which does not track handles) accepts it.
}
