// #2168: scePthreadMutexDestroy / pthread_mutex_destroy must refuse a mutex that is still held.
//
// A console returns EBUSY for a held normal mutex and leaves it alive (measured; see
// tests/data/console_oracle/kernel.golden.tsv, mutex_destroy_while_held). prosper used to answer 0
// and retire the slot. The two spellings differ in ENCODING only, the same split as the condvar's
// (test_cond_destroy_busy.cpp): the POSIX name returns the bare FreeBSD errno 16, the Sony name the
// libkernel form 0x80020010. The oracle replay measures the Sony spelling only, so the POSIX half is
// asserted here, through the real dispatch table.

#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using prosper::Hle;
using prosper::HleFn;

namespace {

uint64_t call(HleFn fn, uint64_t a0 = 0) { return fn(a0, 0, 0, 0, 0, 0); }

HleFn by_name(const char* name) { return Hle::lookup(prosper::nid_hash(name)); }

}   // namespace

TEST(MutexDestroyBusy, HeldMutexIsRefusedInBothEncodings) {
    prosper::register_kernel_hle();
    HleFn init = by_name("pthread_mutex_init");
    HleFn lock = by_name("pthread_mutex_lock");
    HleFn unlock = by_name("pthread_mutex_unlock");
    HleFn posix_destroy = by_name("pthread_mutex_destroy");
    HleFn sce_destroy = by_name("scePthreadMutexDestroy");
    ASSERT_TRUE(init && lock && unlock && posix_destroy && sce_destroy);

    // Control first: an unheld mutex still destroys, so "always EBUSY" cannot pass.
    uint64_t spare = 0;
    ASSERT_EQ(call(init, (uint64_t)(uintptr_t)&spare), 0u);
    EXPECT_EQ(call(posix_destroy, (uint64_t)(uintptr_t)&spare), 0u);

    uint64_t slot = 0;
    ASSERT_EQ(call(init, (uint64_t)(uintptr_t)&slot), 0u);
    ASSERT_EQ(call(lock, (uint64_t)(uintptr_t)&slot), 0u);

    // 16 is FreeBSD EBUSY: the guest's value, not the host's.
    EXPECT_EQ(call(posix_destroy, (uint64_t)(uintptr_t)&slot), 16u)
        << "pthread_mutex_destroy on a held mutex returns the bare FreeBSD EBUSY";
    EXPECT_EQ(call(sce_destroy, (uint64_t)(uintptr_t)&slot), 0x80020010ull)
        << "scePthreadMutexDestroy returns the encoded EBUSY";

    // Two refusals later the mutex is still alive: it unlocks, then destroys cleanly.
    EXPECT_NE(slot, 0u) << "a refused destroy leaves the slot intact";
    EXPECT_EQ(call(unlock, (uint64_t)(uintptr_t)&slot), 0u);
    EXPECT_EQ(call(sce_destroy, (uint64_t)(uintptr_t)&slot), 0u);
}
