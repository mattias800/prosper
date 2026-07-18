// test_service_getters — guards the UserService/libkernel startup stubs reduced in the HLE audit.
// The UserService getters are (userId, out*) queries that must write a deterministic default to the
// output pointer (accessibility off = 0, age level = adult) rather than leave it uninitialized; the
// libkernel registration/hook calls must resolve to a benign OK-returning no-op. Registered by raw
// NID, so this looks them up by NID and exercises the output contract.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/callback_fs.hpp"
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

#ifndef _WIN32
    // Guest callback shims must recover saved r11 from the import stub's real +0x28 slot. +0x18 is
    // forwarded arg9 and can legitimately contain an eboot pointer; treating it as %fs crashes the
    // next host-libc call on executable memory.
    {
        struct alignas(16) FakeTcb { uint8_t bytes[0x110]; } tcb{};
        uint64_t tp = (uint64_t)(uintptr_t)&tcb;
        *(uint64_t*)(tcb.bytes + 0x00) = tp;
        *(uint32_t*)(tcb.bytes + 0x108) = 0x50524F53u;
        uint64_t frame[7] = {};
        frame[0] = 0x600001000ull;    // return into an import swap stub
        frame[3] = 0x400000000ull;    // arg9 decoy: the old +0x18 bug selected this as %fs
        frame[5] = tp;                // +0x28: saved r11 / real guest %fs
        CHECK(callback_guest_fs_from_entry_stack((uint64_t)(uintptr_t)frame) == tp,
              "callback shim recovers guest fs from saved-r11 +0x28, not arg9 +0x18");
        *(uint32_t*)(tcb.bytes + 0x108) = 0;
        CHECK(callback_guest_fs_from_entry_stack((uint64_t)(uintptr_t)frame) == 0,
              "callback shim rejects a value without the guest-TCB magic");
        frame[0] = 0x400001000ull;
        CHECK(callback_guest_fs_from_entry_stack((uint64_t)(uintptr_t)frame) == 0,
              "host-context callback entry does not invent a guest fs frame");
    }
#endif

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

    // Sanitizer malloc replacement query returns a real empty table, not nullptr. Runtimes inspect
    // callback slots even when no sanitizer is active (Dead Cells reads slot +0x18 during startup).
    {
        HleFn fn = Hle::lookup("py6L8jiVAN8");
        CHECK(fn != nullptr, "sceKernelGetSanitizerMallocReplaceExternal registered");
        if (fn) {
            uint64_t addr = fn(0, 0, 0, 0, 0, 0);
            CHECK(addr != 0, "sanitizer malloc replacement table is non-null");
            if (addr) {
                const uint64_t* table = (const uint64_t*)(uintptr_t)addr;
                CHECK(table[0] == 0x70, "sanitizer malloc replacement table reports size 0x70");
                bool empty = true;
                for (int i = 1; i < 14; ++i) empty &= table[i] == 0;
                CHECK(empty, "sanitizer malloc replacement callbacks default to null");
            }
        }
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

    // SaveDataDialog lifecycle (#768): returning generic success (0 == NONE) from UpdateStatus makes
    // a title poll forever. Headless Open auto-dismisses to FINISHED and preserves the proven result
    // prefix without touching the reserved tail.
    {
        auto init   = Hle::lookup("s9e3+YpRnzw");
        auto open   = Hle::lookup("4tPhsP6FpDI");
        auto close  = Hle::lookup("fH46Lag88XY");
        auto status = Hle::lookup("ERKzksauAJA");
        auto update = Hle::lookup("KK3Bdg1RWK0");
        auto result = Hle::lookup("yEiJ-qqr6Cg");
        auto term   = Hle::lookup("YuH2FA7azqQ");
        CHECK(init && open && close && status && update && result && term,
              "SaveDataDialog lifecycle functions registered");
        if (init && open && close && status && update && result && term) {
            term(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 0, "SaveDataDialog before Initialize -> NONE(0)");
            init(0,0,0,0,0,0);
            CHECK(update(0,0,0,0,0,0) == 1, "SaveDataDialog Initialize -> INITIALIZED(1)");
            uint8_t param[0x98]{};
            *(uint32_t*)(param + 0x34) = 3;
            *(uint64_t*)(param + 0x70) = 0x123456789abcdef0ull;
            open((uint64_t)(uintptr_t)param,0,0,0,0,0);
            CHECK(update(0,0,0,0,0,0) == 3, "SaveDataDialog Open auto-dismisses -> FINISHED(3)");
            uint8_t out[0x50]; memset(out, 0xAB, sizeof out);
            result((uint64_t)(uintptr_t)out,0,0,0,0,0);
            CHECK(*(uint32_t*)(out + 0x00) == 3 && *(uint32_t*)(out + 0x04) == 0 &&
                  *(uint32_t*)(out + 0x08) == 0,
                  "SaveDataDialog result reports mode and neutral OK/INVALID outcome");
            CHECK(*(uint64_t*)(out + 0x20) == 0x123456789abcdef0ull,
                  "SaveDataDialog result preserves caller userData");
            CHECK(out[0x28] == 0xAB && out[0x4f] == 0xAB,
                  "SaveDataDialog result does not overwrite its reserved tail");
            close(0,0,0,0,0,0);
            CHECK(status(0,0,0,0,0,0) == 0, "SaveDataDialog Close -> NONE(0)");
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

            out = (int32_t)0xDEAD;
            f(2 /*DATE_FORMAT*/, (uint64_t)(uintptr_t)&out, 0, 0, 0, 0);
            CHECK(out == 2,
                  "ParamGetInt(DATE_FORMAT) -> MM/DD/YYYY(2), matching the en-US default");
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

    // Dead Cells late-startup services (#552): opaque UDS objects must be non-null, and the
    // synchronous SaveData backend must expose one completion event per unmount instead of
    // reporting success with an untouched type-0 event.
    {
        auto create_event = Hle::lookup("p+GcLqwpL9M");
        auto set_string   = Hle::lookup("MfDb+4Nln64");
        auto post_event   = Hle::lookup("CzkKf7ahIyU");
        auto destroy_event= Hle::lookup("wG+84pnNIuo");
        CHECK(create_event && set_string && post_event && destroy_event,
              "NpUniversalDataSystem event lifecycle registered");
        if (create_event && set_string && post_event && destroy_event) {
            uint64_t event = 0, properties = 0;
            uint64_t r = create_event((uint64_t)(uintptr_t)"ActivityStart", 0,
                                      (uint64_t)(uintptr_t)&event,
                                      (uint64_t)(uintptr_t)&properties, 0, 0);
            CHECK(r == 0, "CreateEvent succeeds for valid output pointers");
            CHECK(event != 0 && properties != 0 && event != properties,
                  "CreateEvent writes distinct non-null opaque objects");
            CHECK(set_string(properties, (uint64_t)(uintptr_t)"activityId",
                             (uint64_t)(uintptr_t)"MainQuest", 0, 0, 0) == 0,
                  "EventPropertyObjectSetString accepts the returned object");
            CHECK(post_event(1, 1, event, 1, 0, 0) == 0,
                  "PostEvent accepts the returned event");
            CHECK(destroy_event(event, 0, 0, 0, 0, 0) == 0,
                  "DestroyEvent accepts the returned event");
        }

        auto umount = Hle::lookup("uW4vfTwMQVo");
        auto get_event = Hle::lookup("j8xKtiFj0SY");
        CHECK(umount && get_event, "SaveData unmount event lifecycle registered");
        if (umount && get_event) {
            uint8_t event[112]; memset(event, 0xAB, sizeof event);
            CHECK(get_event(0, (uint64_t)(uintptr_t)event, 0, 0, 0, 0) == 0x809F0018ull,
                  "GetEventResult with no completion -> NO_EVENT");
            umount(0, 0, 0, 0, 0, 0);
            CHECK(get_event(0, (uint64_t)(uintptr_t)event, 0, 0, 0, 0) == 0,
                  "GetEventResult consumes the queued unmount completion");
            CHECK(*(uint32_t*)(event + 0) == 1, "SaveData completion type -> UMOUNT_BACKUP(1)");
            CHECK(*(int32_t*)(event + 4) == 0 && *(int32_t*)(event + 8) == 1,
                  "SaveData completion reports success for initial user");
            bool empty_tail = true; for (size_t i = 12; i < 104; ++i) empty_tail &= event[i] == 0;
            CHECK(empty_tail, "GetEventResult zeroes the empty title, dir name, and reserved tail");
            CHECK(event[104] == 0xAB && event[111] == 0xAB,
                  "GetEventResult writes exactly the 104-byte event struct");
            CHECK(get_event(0, (uint64_t)(uintptr_t)event, 0, 0, 0, 0) == 0x809F0018ull,
                  "GetEventResult completion is one-shot");
        }

        if (HleFn entitlement = Hle::lookup("xddD23+8TfQ")) {
            uint8_t info[32]; memset(info, 0xAB, sizeof info);
            CHECK(entitlement(0, (uint64_t)(uintptr_t)"DEADCELLSBASESEED",
                              (uint64_t)(uintptr_t)info, 0, 0, 0) != 0,
                  "unknown addcont entitlement fails cleanly while signed out");
            bool untouched = true; for (uint8_t b : info) untouched &= b == 0xAB;
            CHECK(untouched, "failed entitlement query leaves output untouched");
        } else CHECK(false, "sceNpEntitlementAccessGetAddcontEntitlementInfo registered");

        if (HleFn game_intent = Hle::lookup("m87BHxt-H60")) {
            uint64_t init[5] = {0x28, 0, 0, 0, 0};
            CHECK(game_intent((uint64_t)(uintptr_t)init, 0, 0, 0, 0, 0) == 0,
                  "sceNpGameIntentInitialize accepts the 0x28-byte inert init struct");
        } else CHECK(false, "sceNpGameIntentInitialize registered");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
