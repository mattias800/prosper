// #3554: AMPR registration stores userdata, upgrades an existing zero, and preserves the first
// nonzero value. Losing it hands a completion consumer the wrong opaque context pointer.
// This pins Prosper's existing compatibility rule through three registered callers and real
// completion events, rather than claiming a universal Sony re-registration contract.
// POSIX only: the legacy sizing call registers (eq,id,0) here; Windows only returns its size.
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/kernel/kernel_event_filters.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace prosper { uint64_t prosper_eq_apr_udata(uint64_t eq, int64_t id); }
using namespace prosper;

namespace {
int fails = 0;

bool check(bool condition, const char* scope, const char* label) {
    std::printf("  [%s] %s: %s\n", condition ? "ok" : "FAIL", scope, label);
    if (!condition) ++fails;
    return condition;
}

int finish() {
    if (fails) std::printf("== FAIL: %d ==\n", fails);
    else std::printf("== PASS ==\n");
    return fails ? 1 : 0;
}

uint64_t pointer(const void* value) { return reinterpret_cast<uintptr_t>(value); }

// Public SceKernelEvent layout, also used by the existing equeue tests.
struct Event {
    int64_t ident;
    int16_t filter;
    uint16_t flags;
    uint32_t fflags;
    int64_t data;
    uint64_t udata;
};
static_assert(sizeof(Event) == 0x20 && offsetof(Event, data) == 0x10 &&
              offsetof(Event, udata) == 0x18);

struct Api {
    HleFn create, destroy, add, bind, measure, submit, wait, count, userdata, destroy_cb;
};
struct Channel {
    uint64_t eq = 0;
    int64_t id = 1;
    alignas(16) std::array<uint8_t, 256> cb{};
};

void add(const Api& api, const Channel& channel, uint64_t value, const char* scope) {
    check(api.add(channel.eq, static_cast<uint64_t>(channel.id), value, 0, 0, 0) == 0,
          scope, "registered AddAmprEvent returns success");
}

void bind(const Api& api, const Channel& channel, uint64_t tag, const char* scope) {
    check(api.bind(pointer(channel.cb.data()), channel.eq, static_cast<uint64_t>(channel.id),
                   tag, 0, 0) == 0, scope, "registered legacy binding returns success");
}

void measure(const Api& api, const Channel& channel, const char* scope) {
    check(api.measure(channel.eq, static_cast<uint64_t>(channel.id), 0, 0, 0, 0) == 20,
          scope, "registered legacy sizing returns twenty bytes");
}

// An accepted Submit owns an asynchronous delivery obligation. If its matching event cannot be
// consumed, returning from main could destroy g_eqs while the producer still consults it. Keep
// all storage live and bypass static teardown on that unexpected failure, never as a valid red.
[[noreturn]] void undelivered() {
    finish();
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(1);
}

Event complete(const Api& api, const Channel& channel, uint64_t tag, const char* scope) {
    if (!check(api.submit(pointer(channel.cb.data()), 1, 0, 0, 0, 0) == 0,
               scope, "registered plain Submit returns success")) undelivered();
    Event event{};
    bool received = false, protocol = true;
    uint32_t timeout_us = 0; // A nonnull zero timeout is a poll, never an infinite wait.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (unsigned polls = 0; polls < 2000 && std::chrono::steady_clock::now() < deadline; ++polls) {
        int32_t count = -1;
        const uint64_t rc = api.wait(channel.eq, pointer(&event), 1, pointer(&count),
                                     pointer(&timeout_us), 0);
        if (rc == 0 && count == 1) { received = true; break; }
        if (rc != 0x8002003CULL || count != 0) { protocol = false; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (!check(received && protocol, scope, "bounded polling consumes one actual completion"))
        undelivered();
    if (!check(event.ident == channel.id && event.filter == EVFILT_AMPR &&
               static_cast<uint64_t>(event.data) == tag, scope,
               "the event carries the expected APR id, filter and completion tag")) undelivered();
    if (!check(api.count(channel.eq, 0, 0, 0, 0, 0) == 0, scope,
               "the completion leaves its live queue empty")) undelivered();
    // Delivery does not join the immortal pointer worker; it proves this accepted post arrived.
    return event;
}

void userdata(const Api& api, const Event& event, uint64_t expected,
              const char* scope, const char* label) {
    const uint64_t accessed = api.userdata(pointer(&event), 0, 0, 0, 0, 0);
    check(event.udata == expected && accessed == expected, scope, label);
}
} // namespace

int main() {
    std::printf("== test_ampr_udata ==\n");
    // These cached diagnostics can inspect absent guest code or the host caller's stack.
    constexpr const char* diagnostics[] = {
        "PROSPER_DUMPCODE", "PROSPER_WAITCALLER", "PROSPER_WAITCAP", "PROSPER_AMPRLOG", "PROSPER_EVLOG"
    };
    bool quiet = true;
    for (const char* key : diagnostics) quiet = (::unsetenv(key) == 0) && quiet;
    if (!check(quiet, "setup", "guest-memory diagnostics are disabled before registered calls"))
        return finish();
    register_builtin_hle();
    const Api api{
        Hle::lookup(nid_hash("sceKernelCreateEqueue")),
        Hle::lookup(nid_hash("sceKernelDeleteEqueue")),
        Hle::lookup("bBfz7kMF2Ho"), Hle::lookup("H896Pt-yB4I"), Hle::lookup("sSAUCCU1dv4"),
        Hle::lookup("eE4Szl8sil8"), Hle::lookup(nid_hash("sceKernelWaitEqueue")),
        Hle::lookup(nid_hash("sceKernelGetEventCount")),
        Hle::lookup(nid_hash("sceKernelGetEventUserData")), Hle::lookup("GuchCTefuZw")
    };
    if (!check(api.create && api.destroy && api.add && api.bind && api.measure && api.submit &&
               api.wait && api.count && api.userdata && api.destroy_cb, "setup",
               "all three registration routes and completion APIs are registered")) return finish();
    check(api.add == Hle::lookup(nid_hash("sceKernelAddAmprEvent")), "setup",
          "the actual AddAmprEvent NID resolves to the named caller");

    std::array<Channel, 4> channels{};
    std::array<uint64_t, 4> contexts{};
    uint64_t completion_context = 0;
    channels[0].id = 0;
    bool created = true;
    for (auto& channel : channels)
        created = check(api.create(pointer(&channel.eq), 0, 0, 0, 0, 0) == 0 && channel.eq,
                        "setup", "create a live queue with owned command-buffer storage") && created;
    bool distinct = created;
    for (size_t i = 0; i < channels.size(); ++i)
        for (size_t j = 0; j < i; ++j) distinct = (channels[i].eq != channels[j].eq) && distinct;
    check(distinct, "setup", "all four queue lifetimes remain distinct until final cleanup");
    const uint64_t tag = pointer(&completion_context);
    bool opaque_distinct = tag != 0;
    for (size_t i = 0; i < contexts.size(); ++i) {
        opaque_distinct = (pointer(&contexts[i]) != 0 && pointer(&contexts[i]) != tag) && opaque_distinct;
        for (size_t j = 0; j < i; ++j)
            opaque_distinct = (pointer(&contexts[i]) != pointer(&contexts[j])) && opaque_distinct;
    }
    check(opaque_distinct, "setup", "opaque userdata contexts and the pointer completion tag are distinct");
    if (created && distinct && opaque_distinct) {
        auto& a = channels[0];
        const uint64_t first = pointer(&contexts[0]), later = pointer(&contexts[1]);
        add(api, a, first, "A first-nonzero");
        check(prosper_eq_apr_udata(a.eq, a.id) == first, "A first-nonzero",
              "first nonzero registration stores the opaque userdata");
        bind(api, a, tag, "A first-nonzero");
        check(prosper_eq_apr_udata(a.eq, a.id) == first, "A first-nonzero",
              "legacy zero binding preserves existing nonzero userdata");
        measure(api, a, "A first-nonzero");
        check(prosper_eq_apr_udata(a.eq, a.id) == first, "A first-nonzero",
              "legacy zero sizing preserves existing nonzero userdata");
        add(api, a, 0, "A first-nonzero");
        check(prosper_eq_apr_udata(a.eq, a.id) == first, "A first-nonzero",
              "explicit zero Add preserves existing nonzero userdata");
        add(api, a, later, "A first-nonzero");
        check(prosper_eq_apr_udata(a.eq, a.id) == first, "A first-nonzero",
              "a later nonzero Add preserves the first nonzero userdata");
        userdata(api, complete(api, a, tag, "A first-nonzero"), first, "A first-nonzero",
                 "pointer completion and GetEventUserData return the first nonzero userdata");

        auto& b = channels[1];
        bind(api, b, 0, "B binding-first");
        check(prosper_eq_apr_udata(b.eq, b.id) == 0, "B binding-first",
              "a zero-only binding has zero userdata before Add");
        userdata(api, complete(api, b, 0, "B zero control"), 0, "B zero control",
                 "a real zero-userdata completion survives the registered accessor");
        add(api, b, pointer(&contexts[1]), "B binding-first");
        check(prosper_eq_apr_udata(b.eq, b.id) == pointer(&contexts[1]), "B binding-first",
              "binding-first zero registration upgrades to nonzero userdata");
        // Legacy binding has no fallback/deduplication worker, so this second plain Submit reuses
        // the same binding without a new zero registration after the upgrade.
        userdata(api, complete(api, b, 0, "B binding-first"), pointer(&contexts[1]), "B binding-first",
                 "binding-first completion and GetEventUserData return upgraded userdata");

        auto& c = channels[2];
        measure(api, c, "C sizing-first");
        check(prosper_eq_apr_udata(c.eq, c.id) == 0, "C sizing-first",
              "zero-only sizing has zero userdata before Add");
        add(api, c, pointer(&contexts[2]), "C sizing-first");
        check(prosper_eq_apr_udata(c.eq, c.id) == pointer(&contexts[2]), "C sizing-first",
              "sizing-first zero registration upgrades before binding");
        bind(api, c, 0, "C sizing-first");
        userdata(api, complete(api, c, 0, "C sizing-first"), pointer(&contexts[2]), "C sizing-first",
                 "sizing-first completion and GetEventUserData return upgraded userdata");

        auto& d = channels[3];
        add(api, d, 0, "D explicit-zero");
        check(prosper_eq_apr_udata(d.eq, d.id) == 0, "D explicit-zero",
              "explicit zero Add has zero userdata before upgrade");
        add(api, d, pointer(&contexts[3]), "D explicit-zero");
        check(prosper_eq_apr_udata(d.eq, d.id) == pointer(&contexts[3]), "D explicit-zero",
              "explicit zero registration upgrades to nonzero userdata");
        bind(api, d, 0, "D explicit-zero");
        userdata(api, complete(api, d, 0, "D explicit-zero"), pointer(&contexts[3]), "D explicit-zero",
                 "explicit-zero completion and GetEventUserData return upgraded userdata");
    }
    // Keeping every queue live until here isolates userdata from the separate reuse issue #4103.
    for (const auto& channel : channels) {
        check(api.destroy_cb(pointer(channel.cb.data()), 0, 0, 0, 0, 0) == 0, "cleanup",
              "the registered command-buffer destructor returns success");
        if (channel.eq) check(api.destroy(channel.eq, 0, 0, 0, 0, 0) == 0, "cleanup",
                              "registered DeleteEqueue returns success for the drained live queue");
    }
    return finish();
}
