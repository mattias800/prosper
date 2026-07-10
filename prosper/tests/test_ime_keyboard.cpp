// test_ime_keyboard (#186) — the libSceIme keyboard API PPSA02664 polls every frame. No physical
// keyboard, so it reports a consistent "none connected" state (empty resource array, disconnected
// info) rather than leaving the caller's structs uninitialized. Struct layouts mirror shadPS4
// src/core/libraries/ime/ime_common.h (offsets asserted below).
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <cstring>

using namespace prosper;

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

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
