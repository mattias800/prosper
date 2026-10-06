// test_np_state_callbacks — the guest state callbacks of libSceNetCtl and libSceNpManager.
//
// sceNetCtlRegisterCallback, sceNetCtlCheckCallback and sceNpRegisterStateCallbackA were
// registered only on POSIX hosts, so on Windows the guest's boot-time registration fell to the
// dispatcher's 0 over an untouched callback-id out-parameter and its per-frame check delivered
// nothing. The table is also exercised without a guest, and the HLE arms drive the real NIDs with
// host functions standing in for the guest callbacks.
//
// The capacities, ids, duplicate rules and error codes pinned here are read from the shipped
// libSceNetCtl.sprx (sceNetCtlRegisterCallback RVA 0x2010, sceNetCtlUnregisterCallback RVA 0x2450)
// and libSceNpManager.sprx (sceNpRegisterStateCallbackA RVA 0x15580, sceNpUnregisterStateCallbackA
// RVA 0x15b10). The literal 8s are deliberate: they pin the modules' slot count, not the constant.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/np/state_callbacks.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <vector>

using namespace prosper;

namespace {

constexpr uint64_t err(uint32_t c) {
    return (uint64_t)(int64_t)(int32_t)c;
}
constexpr uint64_t kNetCtlInvalidAddr = err(0x80412107u);
constexpr uint64_t kNetCtlCallbackMax = err(0x80412103u);
constexpr uint64_t kNetCtlIdNotFound = err(0x80412104u);
constexpr uint64_t kNetCtlInvalidId = err(0x80412105u);
constexpr uint64_t kNpInvalidArgument = err(0x80550003u);
constexpr uint64_t kNpAlreadyRegistered = err(0x80550008u);
constexpr uint64_t kNpNotRegistered = err(0x80550009u);
constexpr uint64_t kNpInvalidCallbackId = err(0x80550015u);
constexpr uint64_t kNpCallbackMax = err(0x8055001Du);
constexpr int kModuleSlots = 8;   // both modules: 8 slots of 0x18 bytes

uint64_t u64(const void* p) {
    return (uint64_t)(uintptr_t)p;
}

struct NetEvent {
    int32_t type;
    void* arg;
};
struct NpEvent {
    int32_t user, state;
    void* arg;
};
std::vector<NetEvent> g_net_events;
std::vector<NpEvent> g_np_events;

void net_cb(int32_t type, void* arg) {
    g_net_events.push_back({type, arg});
}
void net_cb_other(int32_t type, void* arg) {
    g_net_events.push_back({type + 100, arg});
}
void np_cb(int32_t user, int32_t state, void* arg) {
    g_np_events.push_back({user, state, arg});
}
// Distinct functions, so a test can fill NpManager's table without tripping its duplicate rule.
template <int N>
void np_cb_n(int32_t user, int32_t state, void* arg) {
    g_np_events.push_back({user, state + 100 * N, arg});
}
template <int... N>
std::vector<uint64_t> np_cb_family(std::integer_sequence<int, N...>) {
    return {u64((void*)&np_cb_n<N>)...};
}
const std::vector<uint64_t>& np_cbs() {
    static const std::vector<uint64_t> v = np_cb_family(std::make_integer_sequence<int, 9>{});
    return v;
}

uint64_t call(const char* name, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0) {
    HleFn fn = Hle::lookup(nid_hash(name));
    EXPECT_NE(fn, nullptr) << name << " is not registered";
    return fn ? fn(a0, a1, a2, 0, 0, 0) : 0;
}

class StateCallbacks : public ::testing::Test {
protected:
    void SetUp() override {
        register_builtin_hle();
        np::reset_state_callbacks_for_test();
        g_net_events.clear();
        g_np_events.clear();
    }
};

}  // namespace

TEST(StateCallbackTable, SlotsAreIndependentAndBounded) {
    np::StateCallbackTable t;
    for (int i = 0; i < np::StateCallbackTable::kCapacity; ++i) EXPECT_EQ(t.add(0x1000 + i, i), i);
    EXPECT_EQ(t.add(0x2000, 0), -1) << "a full table refuses instead of overwriting";
    EXPECT_TRUE(t.remove(1));
    EXPECT_FALSE(t.remove(1)) << "double removal";
    EXPECT_FALSE(t.remove(-1));
    EXPECT_FALSE(t.remove(np::StateCallbackTable::kCapacity));
    EXPECT_EQ(t.add(0x3000, 9), 1) << "the freed slot is reused";
    EXPECT_EQ(t.size(), np::StateCallbackTable::kCapacity);
}

TEST(StateCallbackTable, HoldsEightSlotsLikeBothModules) {
    EXPECT_EQ(np::StateCallbackTable::kCapacity, kModuleSlots) << "both modules hold 8";
    np::StateCallbackTable t;
    for (int i = 0; i < kModuleSlots; ++i) EXPECT_EQ(t.add(0x1000, i), i) << "repeats allowed";
    EXPECT_EQ(t.add(0x1000, 0), np::StateCallbackTable::kFull);
}

TEST(StateCallbackTable, RefusingDuplicatesIsDecidedBeforeFullness) {
    np::StateCallbackTable t;
    EXPECT_EQ(t.add(0xA, 0, true), 0);
    EXPECT_EQ(t.add(0xA, 1, true), np::StateCallbackTable::kDuplicate);
    EXPECT_EQ(t.size(), 1) << "a refused duplicate takes no slot";
    for (int i = 1; i < kModuleSlots; ++i) EXPECT_EQ(t.add(0x100 + i, 0, true), i);
    EXPECT_EQ(t.add(0xA, 0, true), np::StateCallbackTable::kDuplicate)
        << "a full table holding the function still answers duplicate, not full";
    EXPECT_EQ(t.add(0xB, 0, true), np::StateCallbackTable::kFull);
    EXPECT_TRUE(t.remove(0));
    EXPECT_EQ(t.add(0xA, 0, true), 0) << "after removal the function may register again";
}

TEST(StateCallbackTable, EachRegistrationIsOwedExactlyOneDelivery) {
    np::StateCallbackTable t;
    t.add(0xA, 1);
    std::vector<uint64_t> seen;
    EXPECT_EQ(t.pump([&](uint64_t fn, uint64_t) { seen.push_back(fn); }), 1);
    EXPECT_EQ(t.pump([&](uint64_t fn, uint64_t) { seen.push_back(fn); }), 0);
    t.add(0xB, 2);  // a late registration is owed its own delivery, not the first one's
    EXPECT_EQ(t.pump([&](uint64_t fn, uint64_t) { seen.push_back(fn); }), 1);
    EXPECT_EQ(seen, (std::vector<uint64_t>{0xA, 0xB}));
}

TEST(StateCallbackTable, DeliveryRunsWithoutTheLockHeld) {
    np::StateCallbackTable t;
    t.add(0xA, 0);
    int registered_from_callback = -2;
    const int n = t.pump([&](uint64_t fn, uint64_t) {
        if (fn == 0xA) registered_from_callback = t.add(0xB, 0);
    });
    EXPECT_EQ(registered_from_callback, 1) << "a callback may register another without deadlock";
    EXPECT_EQ(n, 2) << "the registration made mid-pump is delivered in the same pass";
}

TEST_F(StateCallbacks, NetCtlRegisterWritesIdAndCheckDeliversDisconnected) {
    int32_t cid = 0x7777;
    int marker = 0;
    EXPECT_EQ(call("sceNetCtlRegisterCallback", u64((void*)&net_cb), u64(&marker), u64(&cid)), 0u);
    EXPECT_EQ(cid, 0) << "the first id the module hands out is slot 0 (ids are 0-based)";
    EXPECT_TRUE(g_net_events.empty()) << "registration alone delivers nothing";

    EXPECT_EQ(call("sceNetCtlCheckCallback"), 0u);
    ASSERT_EQ(g_net_events.size(), 1u);
    EXPECT_EQ(g_net_events[0].type, 1) << "an offline console reports DISCONNECTED";
    EXPECT_EQ(g_net_events[0].arg, &marker) << "the registered user argument is passed back";

    call("sceNetCtlCheckCallback");
    EXPECT_EQ(g_net_events.size(), 1u) << "no state change, no further event";
}

TEST_F(StateCallbacks, NetCtlStateMatchesTheEventItDelivers) {
    int32_t state = 0x7777;
    EXPECT_EQ(call("sceNetCtlGetState", u64(&state)), 0u);
    EXPECT_EQ(state, 0) << "DISCONNECTED: the same state the callback event reports";
    EXPECT_EQ(call("sceNetCtlGetState", 0), kNetCtlInvalidAddr);
}

TEST_F(StateCallbacks, NetCtlRegistersEveryCallbackAndRefusesPastCapacity) {
    int32_t cids[kModuleSlots] = {};
    for (int i = 0; i < kModuleSlots; ++i) {
        EXPECT_EQ(call("sceNetCtlRegisterCallback", u64((void*)&net_cb), 0, u64(&cids[i])), 0u);
        EXPECT_EQ(cids[i], i) << "ids are the 0-based slot indices";
    }
    int32_t extra = 0x7777;
    EXPECT_EQ(call("sceNetCtlRegisterCallback", u64((void*)&net_cb_other), 0, u64(&extra)),
              kNetCtlCallbackMax);
    EXPECT_EQ(extra, 0x7777) << "a refused registration writes nothing";
    call("sceNetCtlCheckCallback");
    EXPECT_EQ(g_net_events.size(), (size_t)kModuleSlots)
        << "every registration is delivered, none overwritten";
}

TEST_F(StateCallbacks, NetCtlUnregisterFreesTheSlot) {
    int32_t cids[kModuleSlots] = {};
    for (int i = 0; i < kModuleSlots; ++i)
        call("sceNetCtlRegisterCallback", u64((void*)&net_cb), 0, u64(&cids[i]));
    EXPECT_EQ(call("sceNetCtlUnregisterCallback", 3), 0u);
    int32_t cid = 0x7777;
    EXPECT_EQ(call("sceNetCtlRegisterCallback", u64((void*)&net_cb_other), 0, u64(&cid)), 0u)
        << "the freed slot makes room on a full table";
    EXPECT_EQ(cid, 3) << "and it is the lowest free slot";
    EXPECT_EQ(call("sceNetCtlUnregisterCallback", 3), 0u);
    EXPECT_EQ(call("sceNetCtlUnregisterCallback", 3), kNetCtlIdNotFound) << "double unregister";
    EXPECT_EQ(call("sceNetCtlUnregisterCallback", 8), kNetCtlInvalidId) << "one past the table";
    EXPECT_EQ(call("sceNetCtlUnregisterCallback", 0xFFFFFFFFull), kNetCtlInvalidId)
        << "-1 is out of range by the module's unsigned compare";
}

TEST_F(StateCallbacks, NetCtlReRegistrationCycleNeverRunsOutOfSlots) {
    // A title that registers on each entry to an online menu and unregisters on exit: on hardware
    // every registration succeeds, because unregister frees the slot.
    for (int round = 0; round < 3 * kModuleSlots; ++round) {
        int32_t cid = 0x7777;
        ASSERT_EQ(call("sceNetCtlRegisterCallback", u64((void*)&net_cb), 0, u64(&cid)), 0u)
            << "round " << round;
        EXPECT_EQ(cid, 0);
        ASSERT_EQ(call("sceNetCtlUnregisterCallback", (uint64_t)cid), 0u) << "round " << round;
    }
}

TEST_F(StateCallbacks, NetCtlUnregisteredCallbackIsNotDelivered) {
    int32_t cid = 0x7777;
    call("sceNetCtlRegisterCallback", u64((void*)&net_cb), 0, u64(&cid));
    EXPECT_EQ(call("sceNetCtlUnregisterCallback", (uint64_t)cid), 0u);
    call("sceNetCtlCheckCallback");
    EXPECT_TRUE(g_net_events.empty());
}

TEST_F(StateCallbacks, NetCtlRegisterRejectsNullArguments) {
    int32_t cid = 0x7777;
    EXPECT_EQ(call("sceNetCtlRegisterCallback", 0, 0, u64(&cid)), kNetCtlInvalidAddr);
    EXPECT_EQ(cid, 0x7777);
    EXPECT_EQ(call("sceNetCtlRegisterCallback", u64((void*)&net_cb), 0, 0), kNetCtlInvalidAddr);
    call("sceNetCtlCheckCallback");
    EXPECT_TRUE(g_net_events.empty()) << "a rejected registration must not be delivered";
}

TEST_F(StateCallbacks, NetCtlCheckWithNothingRegisteredIsSuccess) {
    EXPECT_EQ(call("sceNetCtlCheckCallback"), 0u);
    EXPECT_TRUE(g_net_events.empty());
}

TEST_F(StateCallbacks, NpRegisterReturnsPositiveIdsAndCheckDeliversSignedOut) {
    int marker = 0;
    const uint64_t id1 = call("sceNpRegisterStateCallbackA", u64((void*)&np_cb), u64(&marker));
    const uint64_t id2 = call("sceNpRegisterStateCallbackA", np_cbs()[1], u64(&marker));
    EXPECT_EQ(id1, 1u) << "ids are slot+1";
    EXPECT_EQ(id2, 2u);
    EXPECT_TRUE(g_np_events.empty());

    EXPECT_EQ(call("sceNpCheckCallback"), 0u);
    ASSERT_EQ(g_np_events.size(), 2u);
    for (const NpEvent& e : g_np_events) {
        EXPECT_EQ(e.user, 1) << "the initial user";
        EXPECT_EQ(e.state % 100, 1) << "SIGNED_OUT -- never SIGNED_IN (2) on an offline console";
        EXPECT_EQ(e.arg, &marker);
    }
    call("sceNpCheckCallback");
    EXPECT_EQ(g_np_events.size(), 2u);
}

TEST_F(StateCallbacks, NpRegisterRefusesADuplicateFunction) {
    int marker = 0;
    EXPECT_EQ(call("sceNpRegisterStateCallbackA", u64((void*)&np_cb), u64(&marker)), 1u);
    EXPECT_EQ(call("sceNpRegisterStateCallbackA", u64((void*)&np_cb), u64(&marker)),
              kNpAlreadyRegistered);
    call("sceNpCheckCallback");
    EXPECT_EQ(g_np_events.size(), 1u) << "the refused duplicate is not delivered a second time";
}

TEST_F(StateCallbacks, NpRegisterRejectsNullAndRefusesPastCapacity) {
    EXPECT_EQ(call("sceNpRegisterStateCallbackA", 0, 0), kNpInvalidArgument);
    for (int i = 0; i < kModuleSlots; ++i)
        EXPECT_EQ(call("sceNpRegisterStateCallbackA", np_cbs()[i], 0), (uint64_t)(i + 1));
    EXPECT_EQ(call("sceNpRegisterStateCallbackA", np_cbs()[kModuleSlots], 0), kNpCallbackMax)
        << "a ninth distinct function";
    EXPECT_EQ(call("sceNpRegisterStateCallbackA", np_cbs()[0], 0), kNpAlreadyRegistered)
        << "a full table holding the function answers ALREADY_REGISTERED, checked first";
}

TEST_F(StateCallbacks, NpUnregisterFreesTheSlot) {
    for (int i = 0; i < kModuleSlots; ++i) call("sceNpRegisterStateCallbackA", np_cbs()[i], 0);
    EXPECT_EQ(call("sceNpUnregisterStateCallbackA", 3), 0u);
    EXPECT_EQ(call("sceNpUnregisterStateCallbackA", 3), kNpNotRegistered) << "double unregister";
    EXPECT_EQ(call("sceNpRegisterStateCallbackA", np_cbs()[kModuleSlots], 0), 3u)
        << "the freed slot is reused on a full table";
    EXPECT_EQ(call("sceNpUnregisterStateCallbackA", 1), 0u);
    EXPECT_EQ(call("sceNpRegisterStateCallbackA", np_cbs()[0], 0), 1u)
        << "an unregistered function is no longer a duplicate";
    EXPECT_EQ(call("sceNpUnregisterStateCallbackA", 0), kNpInvalidCallbackId) << "ids start at 1";
    EXPECT_EQ(call("sceNpUnregisterStateCallbackA", 9), kNpInvalidCallbackId) << "past the table";
    EXPECT_EQ(call("sceNpUnregisterStateCallbackA", 0xFFFFFFFFull), kNpInvalidCallbackId);
}

TEST_F(StateCallbacks, NpUnregisteredCallbackIsNotDelivered) {
    const uint64_t id = call("sceNpRegisterStateCallbackA", u64((void*)&np_cb), 0);
    EXPECT_EQ(call("sceNpUnregisterStateCallbackA", id), 0u);
    call("sceNpCheckCallback");
    EXPECT_TRUE(g_np_events.empty());
}

TEST_F(StateCallbacks, NetCtlAndNpTablesAreIndependent) {
    int32_t cid = 0;
    call("sceNetCtlRegisterCallback", u64((void*)&net_cb), 0, u64(&cid));
    call("sceNpRegisterStateCallbackA", u64((void*)&np_cb), 0);
    call("sceNpCheckCallback");
    EXPECT_EQ(g_np_events.size(), 1u);
    EXPECT_TRUE(g_net_events.empty()) << "NpCheckCallback does not pump NetCtl";
}
