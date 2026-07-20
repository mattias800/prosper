// test_ime_input — sceImeUpdate must invoke the guest's registered handler once per queued
// keyboard event, building the SceImeEvent shape the guest handler reads (issue #1093, ABI derived
// from PPSA02664's handler at eboot+0xf2c540):
//   handler(rdi=arg, rsi=SceImeEvent*); event +0x00 u32 id (0x101=down, 0x102=up), +0x08 u16 keycode.
// A no-event stub (the old behavior) delivered no input, so titles reading input through the IME
// keyboard path never saw a keypress. Drives the real HLE via the NID registry.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include "../src/hle/ime_input.hpp"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Fake guest handler: records each delivered event's id + keycode.
struct Rec { uint32_t id; uint16_t keycode; };
static std::vector<Rec> g_rec;
static void fake_handler(uint64_t /*arg*/, void* ev) {
    auto* e = (uint8_t*)ev;
    g_rec.push_back({*(uint32_t*)(e + 0x00), *(uint16_t*)(e + 0x08)});
}

int main() {
    printf("== test_ime_input ==\n");
    register_builtin_hle();
    auto update = reinterpret_cast<uint64_t (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t)>(
        Hle::lookup(nid_hash("sceImeUpdate")));
    CHECK(update != nullptr, "sceImeUpdate registered");
    if (!update) { printf("fails=%d\n", fails); return 1; }
    const uint64_t H = (uint64_t)(uintptr_t)&fake_handler;

    // No queued events -> handler is never called (a title with no input this frame).
    g_rec.clear();
    update(H, 0, 0, 0, 0, 0, 0);
    CHECK(g_rec.empty(), "no events -> handler not invoked");

    // Enter down then up -> two deliveries, correct ids and keycode (HID Enter = 0x28).
    g_rec.clear();
    ime_push_key(0x28, true);
    ime_push_key(0x28, false);
    update(H, 0, 0, 0, 0, 0, 0);
    CHECK(g_rec.size() == 2, "both queued key events delivered in one update");
    if (g_rec.size() == 2) {
        CHECK(g_rec[0].id == 0x101 && g_rec[0].keycode == 0x28, "key DOWN -> id 0x101, keycode 0x28");
        CHECK(g_rec[1].id == 0x102 && g_rec[1].keycode == 0x28, "key UP   -> id 0x102, keycode 0x28");
    }

    // FIFO order across keys, and the queue drains fully (empty on the next update).
    g_rec.clear();
    ime_push_key(0x2c, true);   // Space
    ime_push_key(0x1d, true);   // Z
    update(H, 0, 0, 0, 0, 0, 0);
    CHECK(g_rec.size() == 2 && g_rec[0].keycode == 0x2c && g_rec[1].keycode == 0x1d,
          "events delivered in FIFO order");
    g_rec.clear();
    update(H, 0, 0, 0, 0, 0, 0);
    CHECK(g_rec.empty(), "queue fully drained (no re-delivery)");

    // A null handler must not crash (a title may pump before registering).
    ime_push_key(0x28, true);
    update(0, 0, 0, 0, 0, 0, 0);
    CHECK(true, "null handler tolerated");

    printf("fails=%d\n", fails);
    return fails ? 1 : 0;
}
