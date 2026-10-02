// test_attr_getstack -- scePthreadAttrGetstack(attr, void** addr, size_t* size) is registered and
// reports exactly what the two single-field getters report.
//
// WHY. It was unregistered, so the "unimplemented, returning 0" stub left both outputs untouched and
// a caller read whatever its stack held. Assassin's Creed Black Flag Resynced calls it from a worker
// thread's start-up path.
//
// WHAT EACH ARM KILLS:
//   M1  Getstack is unregistered again (stub)               -> the sentinel-overwrite arm
//   M2  Getstack writes only one of the two outputs          -> the paired sentinel arms
//   M3  Getstack disagrees with Getstackaddr/Getstacksize    -> the equality arms
//   M4  Getstack dereferences a NULL output                  -> the NULL-tolerance arm (crash)
#include "hle/dispatch/dispatch.hpp"
#include <gtest/gtest.h>
#include "hle/dispatch/nid.hpp"

#include <cstdint>
#include <cstdio>

using namespace prosper;

#define CHECK(c, m) EXPECT_TRUE(c) << (m)

TEST(AttrGetstack, Contract) {
    printf("== test_attr_getstack ==\n");
    register_builtin_hle();
    HleFn init = Hle::lookup(nid_hash("scePthreadAttrInit"));
    HleFn getstack = Hle::lookup(nid_hash("scePthreadAttrGetstack"));
    HleFn getaddr = Hle::lookup(nid_hash("scePthreadAttrGetstackaddr"));
    HleFn getsize = Hle::lookup(nid_hash("scePthreadAttrGetstacksize"));
    CHECK(getstack != nullptr, "scePthreadAttrGetstack is registered (M1)");
    if (!init || !getstack || !getaddr || !getsize) FAIL() << "legacy early exit";

    uint64_t attr = 0;
    CHECK(init((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0) == 0 && attr != 0, "attr initialised");

    void* sentinel_addr = (void*)(uintptr_t)0xdeadbeefcafef00dull;
    size_t sentinel_size = (size_t)0xfeedfacefeedfaceull;
    void* a = sentinel_addr; size_t s = sentinel_size;
    void* ea = sentinel_addr; size_t es = sentinel_size;
    CHECK(getstack((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&a, (uint64_t)(uintptr_t)&s, 0, 0, 0) == 0,
          "Getstack returns 0");
    getaddr((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&ea, 0, 0, 0, 0);
    getsize((uint64_t)(uintptr_t)&attr, (uint64_t)(uintptr_t)&es, 0, 0, 0, 0);
    CHECK(a != sentinel_addr, "the address output was written, not left as sentinel (M1/M2)");
    CHECK(s != sentinel_size, "the size output was written, not left as sentinel (M1/M2)");
    CHECK(a == ea && s == es, "Getstack agrees with Getstackaddr + Getstacksize (M3)");
    CHECK(getstack((uint64_t)(uintptr_t)&attr, 0, 0, 0, 0, 0) == 0 &&
          getstack(0, (uint64_t)(uintptr_t)&a, (uint64_t)(uintptr_t)&s, 0, 0, 0) == 0,
          "NULL outputs and a NULL attr are tolerated (M4)");

}
