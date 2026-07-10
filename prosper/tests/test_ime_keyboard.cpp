// test_ime_keyboard (#186) — the libSceIme keyboard API PPSA02664 polls every frame. No physical
// keyboard, so it reports a consistent "none connected" state (empty resource array, disconnected
// info) rather than leaving the caller's structs uninitialized. Struct layouts mirror shadPS4
// src/core/libraries/ime/ime_common.h (offsets asserted below).
#include "../src/hle/dispatch.hpp"
#include "../src/hle/platform_ui.hpp"
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>

using namespace prosper;

// A stand-in frontend that reports one connected keyboard (id 42) via the PlatformUi hook (#347).
struct KbdUi : PlatformUi {
    int keyboardResourceIds(int32_t /*userId*/, uint32_t* out, int max) override {
        if (max >= 1 && out) { out[0] = 42; return 1; }
        return 0;
    }
};

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

struct KbdResIds { int32_t user_id; uint32_t resource_id[5]; };
struct KbdInfo   { int32_t user_id; uint32_t device; uint32_t type; uint32_t repeat_delay;
                   uint32_t repeat_rate; uint32_t status; int8_t reserved[12]; };
static_assert(sizeof(KbdResIds) == 24, "ResourceIdArray size");
static_assert(sizeof(KbdInfo) == 36 && offsetof(KbdInfo, status) == 20, "KeyboardInfo layout");

int main() {
    printf("== test_ime_keyboard ==\n");
    register_builtin_hle();

    HleFn open  = Hle::lookup("eaFXjfJv3xs"), update = Hle::lookup("-4GCfYdNF1s"),
          info  = Hle::lookup("VkqLPArfFdc"), resid  = Hle::lookup("dKadqZFgKKQ");
    CHECK(open && update && info && resid, "Ime keyboard functions registered");
    if (!(open && update && info && resid)) { printf("== FAIL ==\n"); return 1; }

    CHECK(open(1, 0, 0, 0, 0, 0) == 0, "sceImeKeyboardOpen -> OK (Error, not a handle)");
    CHECK(update(0, 0, 0, 0, 0, 0) == 0, "sceImeUpdate -> OK (no events to pump)");

    // GetResourceId(userId=1, out): echoes userId, reports no connected keyboards (all-zero ids).
    KbdResIds ids; memset(&ids, 0xAB, sizeof ids);
    CHECK(resid(1, (uint64_t)(uintptr_t)&ids, 0, 0, 0, 0) == 0, "GetResourceId -> OK");
    CHECK(ids.user_id == 1, "GetResourceId echoes the queried userId");
    bool empty = true; for (int i = 0; i < 5; i++) if (ids.resource_id[i] != 0) empty = false;
    CHECK(empty, "GetResourceId reports an empty id array (no keyboards connected)");

    // GetInfo(resourceId, info): a disconnected device — all zero except user_id — and no write past 36.
    uint8_t buf[40]; memset(buf, 0xAB, sizeof buf);
    CHECK(info(1, (uint64_t)(uintptr_t)buf, 0, 0, 0, 0) == 0, "GetInfo -> OK");
    KbdInfo* ki = (KbdInfo*)buf;
    CHECK(ki->user_id == 1, "GetInfo user_id = default user 1");
    CHECK(ki->device == 0 && ki->type == 0 && ki->status == 0, "GetInfo reports a disconnected/no-device keyboard");
    CHECK(ki->repeat_delay == 0 && ki->repeat_rate == 0, "GetInfo repeat fields zeroed");
    CHECK(buf[36] == 0xAB && buf[39] == 0xAB, "GetInfo does NOT write past its 36-byte struct");

    // Null out-pointer -> INVALID_ADDRESS, never a wild write.
    CHECK(resid(1, 0, 0, 0, 0, 0) == 0x80BC0031ull, "GetResourceId(NULL) -> INVALID_ADDRESS");
    CHECK(info(1, 0, 0, 0, 0, 0) == 0x80BC0031ull, "GetInfo(NULL) -> INVALID_ADDRESS");

    // --- PlatformUi hook (#347): a registered frontend that reports a keyboard flips presence on. ---
    KbdUi kbd; set_platform_ui(&kbd);
    KbdResIds ids2; memset(&ids2, 0xAB, sizeof ids2);
    resid(1, (uint64_t)(uintptr_t)&ids2, 0, 0, 0, 0);
    CHECK(ids2.user_id == 1 && ids2.resource_id[0] == 42 && ids2.resource_id[1] == 0,
          "backend: GetResourceId reports the frontend's keyboard (id 42) then zeros");
    uint8_t buf2[40]; memset(buf2, 0xAB, sizeof buf2);
    info(42 /*the reported id*/, (uint64_t)(uintptr_t)buf2, 0, 0, 0, 0);
    KbdInfo* ki2 = (KbdInfo*)buf2;
    CHECK(ki2->device == 1 && ki2->status == 1, "backend: GetInfo reports a CONNECTED keyboard (device/status = 1)");
    CHECK(buf2[36] == 0xAB, "backend: GetInfo still bounded to 36 bytes");
    set_platform_ui(nullptr);
    // Back to headless: presence off again.
    KbdResIds ids3; memset(&ids3, 0xAB, sizeof ids3);
    resid(1, (uint64_t)(uintptr_t)&ids3, 0, 0, 0, 0);
    CHECK(ids3.resource_id[0] == 0, "after unregister: GetResourceId reports no keyboards again");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
