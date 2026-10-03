// #4330: a separately unarmed process must not publish or time a native host wait scope.
#include "diagnostics/native_host_wait.hpp"
#include <gtest/gtest.h>

using namespace prosper::diagnostics;
TEST(NativeHostWaitDisabled, ScopeDoesNotEnterTheObserver) {
    ASSERT_FALSE(native_host_wait_observation_enabled())
        << "CTest must unset the process-start lever";
    const auto before = native_host_wait_registry().snapshot(1, {});
    {
        NativeHostWaitScope scope(NativeHostWaitSite::Equeue, 0xa11,
                                  NativeHostWaitMode::RelativeMicroseconds, 9000);
        EXPECT_EQ(native_host_wait_registry().snapshot(1, {}).entered_run, before.entered_run);
    }
    EXPECT_EQ(native_host_wait_registry().snapshot(1, {}).entered_run, before.entered_run);
    EXPECT_EQ(native_host_wait_registry().snapshot(1, {}).dropped_run, before.dropped_run);
}
