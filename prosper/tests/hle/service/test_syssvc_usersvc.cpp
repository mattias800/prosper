// test_syssvc_usersvc — the remaining SystemService/UserService getters answer their
// contracts instead of the dispatcher's default.
//
// sceSystemService{Set,DisableAutoSet}NoticeScreenSkipFlag, sceSystemServicePowerTick,
// sceUserServiceInitialize2 and sceUserServiceGetPlatformPrivacyWs1 were unregistered. The
// getters among them return SCE_OK over untouched out-words — a title reads a garbage flag,
// privacy value or tick result as real console state. Every TEST drives the real NIDs: flags
// round-trip what was set, the tick acknowledges, and privacy reads a deterministic default.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

TEST(SysService, NoticeSkipRoundTrip) {
    register_service_hle();
    HleFn get = Hle::lookup("3RQ5aQfnstU");
    HleFn set = Hle::lookup("Q3utJvma4Mo");
    HleFn noauto = Hle::lookup("8Lo6Zv94aho");
    HleFn tick = Hle::lookup("XbbJC3E+L5M");
    ASSERT_NE(get, nullptr);
    ASSERT_NE(set, nullptr);
    ASSERT_NE(noauto, nullptr);
    ASSERT_NE(tick, nullptr);
    EXPECT_EQ(nid_hash("sceSystemServiceSetNoticeScreenSkipFlag"), "Q3utJvma4Mo");
    EXPECT_EQ(nid_hash("sceSystemServicePowerTick"), "XbbJC3E+L5M");
    uint8_t flag = 0xAA;
    ASSERT_EQ(get((uint64_t)(uintptr_t)&flag, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(flag, 0u) << "no-skip default reads back";
    EXPECT_EQ(set(0, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(noauto(0, 0, 0, 0, 0, 0), 0u);
    flag = 0xAA;
    ASSERT_EQ(get((uint64_t)(uintptr_t)&flag, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(flag, 1u) << "a set skip reads back: read-your-writes";
    EXPECT_EQ(tick(0, 0, 0, 0, 0, 0), 0u) << "keep-awake hint acknowledged";
    EXPECT_EQ(get(0, 0, 0, 0, 0, 0), 0u) << "null out-pointer still succeeds";
}

TEST(UserService, PrivacyDefaultAndInit2) {
    register_service_hle();
    HleFn privacy = Hle::lookup("D-CzAxQL0XI");
    HleFn init2 = Hle::lookup(nid_hash("sceUserServiceInitialize2"));
    ASSERT_NE(privacy, nullptr);
    ASSERT_NE(init2, nullptr) << "sceUserServiceInitialize2 must be registered";
    EXPECT_EQ(nid_hash("sceUserServiceGetPlatformPrivacyWs1"), "D-CzAxQL0XI");
    EXPECT_EQ(init2(0, 0, 0, 0, 0, 0), 0u);
    int32_t value = 0x5A5A5A5A;
    ASSERT_EQ(privacy(1, (uint64_t)(uintptr_t)&value, 0, 0, 0, 0), 0u);
    EXPECT_EQ(value, 0) << "deterministic default instead of stack garbage";
    EXPECT_EQ(Hle::lookup("AAAAAAAAAAA"), nullptr)
        << "positive control: the lookup rejects an unknown NID";
}
