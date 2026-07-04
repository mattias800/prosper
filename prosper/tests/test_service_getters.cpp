// test_service_getters — guards the UserService/libkernel startup stubs reduced in the HLE audit.
// The UserService getters are (userId, out*) queries that must write a deterministic default to the
// output pointer (accessibility off = 0, age level = adult) rather than leave it uninitialized; the
// libkernel registration/hook calls must resolve to a benign OK-returning no-op. Registered by raw
// NID, so this looks them up by NID and exercises the output contract.
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Call an (userId, int* out) getter and return the value it wrote (sentinel-initialized to catch a
// no-write).
static int32_t call_int_getter(HleFn fn, int32_t sentinel) {
    int32_t out = sentinel;
    fn(1 /*userId*/, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
    return out;
}

int main() {
    printf("== test_service_getters ==\n");
    register_builtin_hle();

    // UserService getters: age level -> adult (18), accessibility -> off (0). Must WRITE the output.
    struct { const char* nid; int32_t want; const char* what; } getters[] = {
        {"woNpu+45RLk", 18, "GetAgeLevel -> adult (18)"},
        {"rnEhHqG-4xo",  0, "GetAccessibilityChatTranscription -> 0"},
        {"O6IW1-Dwm-w",  0, "GetAccessibilityZoomFollowFocus -> 0"},
        {"-3Y5GO+-i78",  0, "GetAccessibilityTriggerEffect -> 0"},
    };
    for (auto& g : getters) {
        HleFn fn = Hle::lookup(g.nid);
        CHECK(fn != nullptr, g.what);
        if (fn) CHECK(call_int_getter(fn, (int32_t)0xDEAD) == g.want, g.what);
    }

    // libkernel registration/no-op calls: resolve + return OK (0), no crash.
    const char* noops[] = {"rNhWz+lvOMU", "pB-yGZ2nQ9o", "WhCc1w3EhSI",
                           "p5EcQeEeJAE", "bnZxYgAFeA0", "DGMG3JshrZU"};
    for (const char* nid : noops) {
        HleFn fn = Hle::lookup(nid);
        CHECK(fn != nullptr, nid);
        if (fn) CHECK(fn(0, 0, 0, 0, 0, 0) == 0, nid);
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
