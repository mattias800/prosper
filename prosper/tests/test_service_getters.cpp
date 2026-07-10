// test_service_getters — guards the UserService/libkernel startup stubs reduced in the HLE audit.
// The UserService getters are (userId, out*) queries that must write a deterministic default to the
// output pointer (accessibility off = 0, age level = adult) rather than leave it uninitialized; the
// libkernel registration/hook calls must resolve to a benign OK-returning no-op. Registered by raw
// NID, so this looks them up by NID and exercises the output contract.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

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

    // Message-dialog lifecycle (#144): GetStatus must report NONE before an Open (the old handler
    // returned FINISHED unconditionally, so a guest guarding on GetStatus saw "done" at the wrong
    // stage). Transitions: Initialize -> INITIALIZED(1); Open -> FINISHED(3, auto-dismiss); Close ->
    // NONE(0). Registered by name; look up the NIDs.
    {
        auto init   = Hle::lookup(nid_hash("sceMsgDialogInitialize"));
        auto open   = Hle::lookup(nid_hash("sceMsgDialogOpen"));
        auto close  = Hle::lookup(nid_hash("sceMsgDialogClose"));
        auto status = Hle::lookup(nid_hash("sceMsgDialogGetStatus"));
        auto upd    = Hle::lookup(nid_hash("sceMsgDialogUpdateStatus"));
        CHECK(init && open && close && status && upd, "MsgDialog lifecycle functions registered");
        if (init && open && close && status && upd) {
            close(0,0,0,0,0,0);   // reset to a known idle state
            CHECK(status(0,0,0,0,0,0) == 0, "GetStatus before Initialize -> NONE(0) (was FINISHED)");
            init(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 1, "after Initialize -> INITIALIZED(1)");
            CHECK(upd(0,0,0,0,0,0) == 1, "UpdateStatus before Open also reports INITIALIZED, not FINISHED");
            open(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 3, "after Open -> FINISHED(3) (auto-dismiss)");
            close(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 0, "after Close -> back to NONE(0)");
        }
    }

    // SystemService / AppContent / NP getters: each must WRITE its out-param with the CORRECT value —
    // returning success with an unfilled (or wrong-valued) out is the harmful-stub class whose downstream
    // effects are hard to trace (a 0.0 safe-area ratio collapses the viewport; lang 0 localizes to
    // Japanese; NP state SIGNED_IN would send an offline title into network flows). These handlers carry
    // careful fixes but had no test locking them; this does.
    {
        // sceSystemServiceParamGetInt(paramId=LANG(1), int* out) -> en-US(1), not Japanese(0).
        if (HleFn f = Hle::lookup(nid_hash("sceSystemServiceParamGetInt"))) {
            int32_t out = (int32_t)0xDEAD; f(1 /*LANG*/, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
            CHECK(out == 1, "ParamGetInt(LANG) -> en-US(1), not the uninitialized/Japanese default");
        } else CHECK(false, "sceSystemServiceParamGetInt registered");

        // sceSystemServiceGetStatus(status*) -> fills 12 bytes, byte[6] (isCpuMode7CpuNormal)=1, and
        // MUST NOT write past 12 (the oversized-write class — cf. the pad-overflow crash).
        if (HleFn f = Hle::lookup(nid_hash("sceSystemServiceGetStatus"))) {
            uint8_t buf[16]; memset(buf, 0xAB, sizeof buf);
            f((uint64_t)(uintptr_t)buf, 0, 0, 0, 0, 0);
            bool body_ok = true; for (int i = 0; i < 12; i++) if (buf[i] != (i == 6 ? 1 : 0)) body_ok = false;
            CHECK(body_ok, "GetStatus fills 12 bytes (byte6=isCpuMode7CpuNormal=1, rest 0)");
            CHECK(buf[12] == 0xAB && buf[15] == 0xAB, "GetStatus does NOT write past its 12-byte struct");
        } else CHECK(false, "sceSystemServiceGetStatus registered");

        // sceSystemServiceGetDisplaySafeAreaInfo(info*) -> ratio=1.0 (a 0.0 ratio collapses the viewport).
        if (HleFn f = Hle::lookup("1n37q1Bvc5Y")) {
            uint8_t buf[132]; memset(buf, 0xAB, sizeof buf);
            f((uint64_t)(uintptr_t)buf, 0, 0, 0, 0, 0);
            float ratio; memcpy(&ratio, buf, 4);
            CHECK(ratio == 1.0f, "GetDisplaySafeAreaInfo -> ratio 1.0 (full display, no viewport collapse)");
        } else CHECK(false, "sceSystemServiceGetDisplaySafeAreaInfo registered");

        // sceAppContentTemporaryDataMount2(opt, char mp[16]) -> exactly "/temp0\0" (7 bytes), never 16.
        if (HleFn f = Hle::lookup("buYbeLOGWmA")) {
            char mp[16]; memset(mp, 0xAB, sizeof mp);
            f(0, (uint64_t)(uintptr_t)mp, 0, 0, 0, 0);
            CHECK(memcmp(mp, "/temp0\0", 7) == 0, "TemporaryDataMount2 writes \"/temp0\\0\" (game builds paths from it)");
            CHECK((uint8_t)mp[7] == 0xAB, "TemporaryDataMount2 writes exactly 7 bytes, not the full 16");
        } else CHECK(false, "sceAppContentTemporaryDataMount2 registered");

        // sceAppContentTemporaryDataGetAvailableSpaceKb(mp, uint64_t* kb) -> nonzero free (1 GiB).
        if (HleFn f = Hle::lookup("SaKib2Ug0yI")) {
            uint64_t kb = 0xDEAD; f(0, (uint64_t)(uintptr_t)&kb, 0, 0, 0, 0);
            CHECK(kb == 1048576ull, "TemporaryDataGetAvailableSpaceKb -> 1 GiB free (not uninitialized)");
        } else CHECK(false, "sceAppContentTemporaryDataGetAvailableSpaceKb registered");

        // sceNpGetState(userId, SceNpState* out) -> SIGNED_OUT(1). SIGNED_IN would push an offline title
        // into network flows that never complete.
        if (HleFn f = Hle::lookup(nid_hash("sceNpGetState"))) {
            int32_t st = (int32_t)0xDEAD; f(1, (uint64_t)(uintptr_t)&st, 0, 0, 0, 0);
            CHECK(st == 1, "sceNpGetState -> SIGNED_OUT(1), written (offline)");
        } else CHECK(false, "sceNpGetState registered");

        // sceNpGetAccountIdA(userId, uint64_t* out) -> signed-out error AND a written 0 (no stale id).
        if (HleFn f = Hle::lookup(nid_hash("sceNpGetAccountIdA"))) {
            uint64_t id = 0xDEAD; uint64_t r = f(1, (uint64_t)(uintptr_t)&id, 0, 0, 0, 0);
            CHECK(r != 0, "sceNpGetAccountIdA -> signed-out error (not spurious success)");
            CHECK(id == 0, "sceNpGetAccountIdA writes 0 to the account-id out (no uninitialized id)");
        } else CHECK(false, "sceNpGetAccountIdA registered");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
