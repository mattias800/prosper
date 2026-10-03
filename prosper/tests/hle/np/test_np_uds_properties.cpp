// test_np_uds_properties (#4275) -- standalone NP Universal Data System property objects and arrays.
//
// The six calls were unregistered, so each returned SCE_OK without doing anything. Create never
// wrote its out-handle, and Hollow Knight: Silksong went on to set, attach and destroy whatever stale
// value that slot already held. Every arm goes through the real NIDs.
//
// WHAT EACH ARM KILLS:
//   CreateWritesDistinctHandles      Create returning SCE_OK without writing *out (the old stub)
//   SettersValidateTheirHandles      setters that accept anything, or reject what Create returned
//   DestroyFreesExactlyOnce          a Destroy that accepts unknown or already-destroyed handles
//   SessionSignalingInitialize       the init staying unregistered, or accepting a null block
#include "hle/dispatch/dispatch.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace prosper;

namespace {

constexpr uint64_t kInvalidArgument = 0x80550003ull;

struct Uds {
    HleFn create_object = Hle::lookup("s6W4Zl4Slgk");
    HleFn create_array = Hle::lookup("Hm7qubT3b70");
    HleFn array_set_string = Hle::lookup("4llLk7YJRTE");
    HleFn object_set_array = Hle::lookup("Wxbg5x3pTXA");
    HleFn destroy_object = Hle::lookup("kKUH0Viib3c");
    HleFn destroy_array = Hle::lookup("W-0xwY0ZMjw");
    bool all() const {
        return create_object && create_array && array_set_string && object_set_array &&
               destroy_object && destroy_array;
    }
};

uint64_t ptr(const void* p) { return (uint64_t)(uintptr_t)p; }

Uds registered() {
    register_builtin_hle();
    return Uds{};
}

}  // namespace

TEST(NpUdsProperties, CreateWritesDistinctHandles) {
    const Uds u = registered();
    ASSERT_TRUE(u.all());
    uint64_t object = 0xDEAD0000DEAD0000ull, array = 0xDEAD0000DEAD0000ull, second = 0;
    ASSERT_EQ(u.create_object(ptr(&object), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.create_array(ptr(&array), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.create_object(ptr(&second), 0, 0, 0, 0, 0), 0u);
    EXPECT_NE(object, 0xDEAD0000DEAD0000ull) << "CreateEventPropertyObject wrote its out-handle";
    EXPECT_NE(array, 0xDEAD0000DEAD0000ull) << "CreateEventPropertyArray wrote its out-handle";
    EXPECT_NE(object, 0u);
    EXPECT_NE(object, second);
    EXPECT_NE(object, array);
    EXPECT_EQ(u.create_object(0, 0, 0, 0, 0, 0), kInvalidArgument) << "null out-pointer is refused";
    EXPECT_EQ(u.create_array(0, 0, 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(u.destroy_object(object, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_object(second, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_array(array, 0, 0, 0, 0, 0), 0u);
}

TEST(NpUdsProperties, SettersValidateTheirHandles) {
    const Uds u = registered();
    ASSERT_TRUE(u.all());
    uint64_t object = 0, array = 0;
    ASSERT_EQ(u.create_object(ptr(&object), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.create_array(ptr(&array), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.array_set_string(array, ptr("PlayStation"), 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.object_set_array(object, ptr("platforms"), array, 0, 0, 0), 0u);
    // The Silksong failure mode: stale values that were never handed out.
    EXPECT_EQ(u.array_set_string(0x7c0000000ull, ptr("x"), 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(u.object_set_array(0x400410d000ull, ptr("k"), array, 0, 0, 0), kInvalidArgument);
    // Kinds are not interchangeable, and a null string or key is refused.
    EXPECT_EQ(u.array_set_string(object, ptr("x"), 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(u.object_set_array(object, ptr("k"), object, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(u.array_set_string(array, 0, 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(u.object_set_array(object, 0, array, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(u.destroy_array(array, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_object(object, 0, 0, 0, 0, 0), 0u);
}

TEST(NpUdsProperties, DestroyFreesExactlyOnce) {
    const Uds u = registered();
    ASSERT_TRUE(u.all());
    uint64_t object = 0, array = 0;
    ASSERT_EQ(u.create_object(ptr(&object), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.create_array(ptr(&array), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_array(object, 0, 0, 0, 0, 0), kInvalidArgument) << "wrong kind";
    EXPECT_EQ(u.destroy_object(object, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_object(object, 0, 0, 0, 0, 0), kInvalidArgument) << "double destroy";
    EXPECT_EQ(u.array_set_string(array, ptr("x"), 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_array(array, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.array_set_string(array, ptr("x"), 0, 0, 0, 0), kInvalidArgument)
        << "a destroyed handle is no longer accepted";
}

TEST(NpUdsProperties, SessionSignalingInitialize) {
    register_builtin_hle();
    HleFn init = Hle::lookup("ysmw6J-P8Ak");
    ASSERT_NE(init, nullptr);
    uint8_t param[0x40] = {1};
    EXPECT_EQ(init(ptr(param), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), kInvalidArgument);
}
