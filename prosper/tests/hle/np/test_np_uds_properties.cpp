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
#include <cstring>
#include <string>

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

// Scalar setters across both containers: a live handle of the right kind plus a pointer-like
// key (objects) succeed; stale handles, swapped kinds and null keys/buffers are refused. The
// value itself travels by register, so there is nothing further to validate about it.
TEST(NpUdsProperties, ScalarSettersValidateHandles) {
    const Uds u = registered();
    ASSERT_TRUE(u.all());
    register_builtin_hle();
    HleFn arr_i32 = Hle::lookup("BypQuF113-k");
    HleFn arr_bool = Hle::lookup("0+l4QSWCM4E");
    HleFn arr_bin = Hle::lookup("IEdUCV9j2Cw");
    HleFn arr_obj = Hle::lookup("XY14n3jNIpE");
    HleFn arr_arr = Hle::lookup("rdi9BAfDLq8");
    HleFn obj_bool = Hle::lookup("Fidd8vWgyVE");
    HleFn obj_i64 = Hle::lookup("56QLTqx911s");
    HleFn obj_bin = Hle::lookup("wAcxBDLHj1M");
    HleFn obj_obj = Hle::lookup("74ASEqxSnkM");
    for (HleFn f :
         {arr_i32, arr_bool, arr_bin, arr_obj, arr_arr, obj_bool, obj_i64, obj_bin, obj_obj})
        ASSERT_NE(f, nullptr) << "every scalar setter must be registered";

    uint64_t object = 0, array = 0, second = 0;
    ASSERT_EQ(u.create_object(ptr(&object), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.create_array(ptr(&array), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.create_object(ptr(&second), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(arr_i32(array, 7, 0, 0, 0, 0), 0u);
    EXPECT_EQ(arr_bool(array, 1, 0, 0, 0, 0), 0u);
    EXPECT_EQ(arr_bin(array, ptr("blob"), 4, 0, 0, 0), 0u);
    EXPECT_EQ(obj_bool(object, ptr("k"), 1, 0, 0, 0), 0u);
    EXPECT_EQ(obj_i64(object, ptr("k"), 0x123456789ull, 0, 0, 0), 0u);
    EXPECT_EQ(obj_bin(object, ptr("k"), ptr("blob"), 4, 0, 0), 0u);
    // Stale handles, swapped kinds, null keys and null buffers are all refused.
    EXPECT_EQ(arr_i32(0x7c0000000ull, 7, 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(arr_i32(object, 7, 0, 0, 0, 0), kInvalidArgument) << "object is not an array";
    EXPECT_EQ(obj_bool(array, ptr("k"), 1, 0, 0, 0), kInvalidArgument) << "array is not an object";
    EXPECT_EQ(obj_bool(object, 0, 1, 0, 0, 0), kInvalidArgument) << "null key";
    EXPECT_EQ(arr_bin(array, 0, 4, 0, 0, 0), kInvalidArgument) << "null buffer";
    EXPECT_EQ(obj_bin(object, ptr("k"), 0, 4, 0, 0), kInvalidArgument);
    // Object/array attach: value of the wrong kind refused. *out is the INSERTED element, never
    // the caller's own value, so it survives the caller destroying its value (module +0x51c7).
    EXPECT_EQ(arr_obj(array, array, 0, 0, 0, 0), kInvalidArgument)
        << "an array value is not an object";
    uint64_t inserted = 0;
    EXPECT_EQ(arr_obj(array, second, ptr(&inserted), 0, 0, 0), 0u);
    EXPECT_NE(inserted, 0u);
    EXPECT_NE(inserted, second) << "*out is the inserted copy, not the caller's value";
    uint64_t fresh = 0;
    EXPECT_EQ(arr_obj(array, 0, ptr(&fresh), 0, 0, 0), 0u);
    EXPECT_NE(fresh, 0u) << "null value allocates";
    EXPECT_EQ(u.destroy_object(fresh, 0, 0, 0, 0, 0), 0u);
    uint64_t inserted2 = 0;
    EXPECT_EQ(obj_obj(object, ptr("k"), second, ptr(&inserted2), 0, 0), 0u);
    EXPECT_NE(inserted2, second) << "*out is the inserted copy, not the caller's value";
    uint64_t inserted_arr = 0;
    EXPECT_EQ(arr_arr(array, array, ptr(&inserted_arr), 0, 0, 0), 0u);
    EXPECT_NE(inserted_arr, array) << "*out is the inserted copy, not the caller's value";
    EXPECT_EQ(obj_obj(object, ptr("k"), array, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(arr_arr(array, object, 0, 0, 0, 0), kInvalidArgument);
    // The caller destroys its own value; sets through the inserted handles still succeed.
    EXPECT_EQ(u.destroy_object(second, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(obj_bool(inserted, ptr("k"), 1, 0, 0, 0), 0u)
        << "the inserted element outlives the caller's value";
    EXPECT_EQ(obj_bool(inserted2, ptr("k"), 1, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_object(inserted, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_object(inserted2, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_array(inserted_arr, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_object(object, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_array(array, 0, 0, 0, 0, 0), 0u);
}

// Handle/context teardown validates against the only id in circulation; Terminate always
// succeeds; stats/strings report the empty offline state without touching caller memory
// they have no data for.
TEST(NpUdsProperties, HandleContextLifecycle) {
    register_builtin_hle();
    HleFn abort = Hle::lookup("jZCqWFgMehE");
    HleFn destroy_ctx = Hle::lookup("wB7IWzGp2v0");
    HleFn term = Hle::lookup("47UAEuQl+iI");
    ASSERT_NE(abort, nullptr);
    ASSERT_NE(destroy_ctx, nullptr);
    ASSERT_NE(term, nullptr);
    EXPECT_EQ(abort(1, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(abort(2, 0, 0, 0, 0, 0), kInvalidArgument) << "foreign id refused";
    EXPECT_EQ(abort(0, 0, 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(destroy_ctx(1, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(destroy_ctx(0xDEADu, 0, 0, 0, 0, 0), kInvalidArgument);
    EXPECT_EQ(term(0, 0, 0, 0, 0, 0), 0u);
}

TEST(NpUdsProperties, StatsAndStringsReportEmpty) {
    register_builtin_hle();
    HleFn estimate = Hle::lookup("+s14jq-KGYw");
    HleFn to_string = Hle::lookup("vj6CQGWtEBg");
    HleFn mem_stat = Hle::lookup("su7jW3VDDb4");
    HleFn storage_stat = Hle::lookup("KmN62tT4U8A");
    for (HleFn f : {estimate, to_string, mem_stat, storage_stat}) ASSERT_NE(f, nullptr);
    EXPECT_EQ(estimate(0, 0, 0, 0, 0, 0), kInvalidArgument) << "null event";
    uint64_t size = 0xDEADu;
    // Events are counter ids, not pointers: any nonzero id estimates.
    EXPECT_EQ(estimate(1, ptr(&size), 0, 0, 0, 0), 0u);
    EXPECT_EQ(size, 3u) << "an empty event serializes to 3 bytes";
    EXPECT_EQ(to_string(0, 0, 0, 0, 0, 0), kInvalidArgument) << "null event";
    char buf[16];
    std::memset(buf, 0xAA, sizeof buf);
    uint64_t str_size = 0xDEADu;
    EXPECT_EQ(to_string(1, ptr(buf), sizeof buf, ptr(&str_size), 0, 0), 0u);
    EXPECT_EQ(str_size, 3u);
    EXPECT_EQ(std::string(buf), "{}") << "an empty event stringifies to {}";
    EXPECT_EQ(to_string(1, 0, 16, 0, 0, 0), kInvalidArgument)
        << "neither a buffer nor a size pointer: nowhere to answer (module +0x599d)";
    uint8_t mem[24];
    std::memset(mem, 0xAA, sizeof mem);
    EXPECT_EQ(mem_stat(ptr(mem), 0, 0, 0, 0, 0), 0u);
    for (uint8_t b : mem) EXPECT_EQ(b, 0u) << "memory stats zeroed: nothing stored";
    EXPECT_EQ(mem_stat(0, 0, 0, 0, 0, 0), kInvalidArgument);
    uint8_t storage[56];
    std::memset(storage, 0xAA, sizeof storage);
    EXPECT_EQ(storage_stat(1, ptr(storage), 0, 0, 0, 0), 0u);
    for (uint8_t b : storage) EXPECT_EQ(b, 0u) << "storage stats zeroed";
    EXPECT_EQ(storage_stat(1, 0, 0, 0, 0, 0), kInvalidArgument);
    std::memset(storage, 0xAA, sizeof storage);
    EXPECT_EQ(storage_stat(2, ptr(storage), 0, 0, 0, 0), kInvalidArgument)
        << "a context no CreateContext produced is refused";
    EXPECT_EQ(storage[0], 0xAA) << "and the buffer is untouched";
}

TEST(NpUdsProperties, SessionSignalingInitialize) {
    register_builtin_hle();
    HleFn init = Hle::lookup("ysmw6J-P8Ak");
    ASSERT_NE(init, nullptr);
    uint8_t param[0x40] = {1};
    EXPECT_EQ(init(ptr(param), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(init(0, 0, 0, 0, 0, 0), kInvalidArgument);
}

// Review of #4290: CreateEvent's property-object out must be accepted by ObjectSetArray. Metaphor
// attaches arrays only to objects from CreateEvent, and a validator that only knew
// CreateEventPropertyObject's handles refused all of them. The event owns the object, so
// DestroyEventPropertyObject must refuse it rather than free a handle it never allocated.
TEST(NpUdsProperties, CreateEventObjectsAcceptArrays) {
    const Uds u = registered();
    ASSERT_TRUE(u.all());
    HleFn create_event = Hle::lookup("p+GcLqwpL9M");
    ASSERT_NE(create_event, nullptr);
    uint64_t array = 0;
    ASSERT_EQ(u.create_array(ptr(&array), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.array_set_string(array, ptr("PlayStation"), 0, 0, 0, 0), 0u);
    // Dead Cells' shape: properties out in a3.
    uint64_t event = 0, properties = 0;
    ASSERT_EQ(create_event(ptr("activityStart"), 0, ptr(&event), ptr(&properties), 0, 0), 0u);
    ASSERT_NE(properties, 0u);
    EXPECT_EQ(u.object_set_array(properties, ptr("platforms"), array, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_object(properties, 0, 0, 0, 0, 0), kInvalidArgument)
        << "the event owns its property object";
    // Alex Kidd's shape: properties out in a4, with a3 and a5 zero.
    uint64_t event2 = 0, properties2 = 0;
    ASSERT_EQ(create_event(ptr("activityTerminate"), 0, ptr(&event2), 0, ptr(&properties2), 0), 0u);
    ASSERT_NE(properties2, 0u);
    EXPECT_EQ(u.object_set_array(properties2, ptr("platforms"), array, 0, 0, 0), 0u);
    EXPECT_EQ(u.destroy_array(array, 0, 0, 0, 0, 0), 0u);
}

// DestroyHandle validates against the only handle id in circulation (#3630 bucket D forwards
// this result, and Dreaming Sarah imports it). Unregistered, any handle read as success.
TEST(NpUdsProperties, DestroyHandleValidatesTheHandle) {
    register_builtin_hle();
    HleFn create = Hle::lookup("hT0IAEvN+M0");
    HleFn destroy = Hle::lookup("AUIHb7jUX3I");
    ASSERT_NE(create, nullptr) << "CreateHandle must stay registered";
    ASSERT_NE(destroy, nullptr) << "DestroyHandle must be registered";
    int32_t handle = 0;
    ASSERT_EQ(create(ptr(&handle), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(handle, 1) << "CreateHandle hands out id 1";
    EXPECT_EQ(destroy(1, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(destroy(0, 0, 0, 0, 0, 0), kInvalidArgument) << "id 0 was never handed out";
    EXPECT_EQ(destroy(2, 0, 0, 0, 0, 0), kInvalidArgument) << "foreign id is refused";
    EXPECT_EQ(destroy(0xDEADu, 0, 0, 0, 0, 0), kInvalidArgument) << "garbage is refused";
}

// EventPropertyObjectSetInt32 (YE4dbtbz6OE resolved via nid_hash; Dreaming Sarah imports it):
// same validation as the SetArray sibling — a live object and a pointer-like key.
TEST(NpUdsProperties, ObjectSetInt32ValidatesItsObject) {
    const Uds u = registered();
    ASSERT_TRUE(u.all());
    HleFn set_int32 = Hle::lookup("YE4dbtbz6OE");
    ASSERT_NE(set_int32, nullptr) << "SetInt32 must be registered";
    uint64_t object = 0, array = 0;
    ASSERT_EQ(u.create_object(ptr(&object), 0, 0, 0, 0, 0), 0u);
    ASSERT_EQ(u.create_array(ptr(&array), 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(set_int32(object, ptr("score"), 100, 0, 0, 0), 0u);
    EXPECT_EQ(set_int32(0x7c0000000ull, ptr("score"), 100, 0, 0, 0), kInvalidArgument)
        << "stale handle refused";
    EXPECT_EQ(set_int32(array, ptr("score"), 100, 0, 0, 0), kInvalidArgument)
        << "an array is not an object";
    EXPECT_EQ(set_int32(object, 0, 100, 0, 0, 0), kInvalidArgument) << "null key refused";
    EXPECT_EQ(u.destroy_object(object, 0, 0, 0, 0, 0), 0u);
    EXPECT_EQ(set_int32(object, ptr("score"), 100, 0, 0, 0), kInvalidArgument)
        << "a destroyed handle is no longer accepted";
    EXPECT_EQ(u.destroy_array(array, 0, 0, 0, 0, 0), 0u);
}
