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
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

// Fake guest handler: records each delivered event's id + keycode.
struct Rec { uint32_t id; uint16_t keycode; };
static std::vector<Rec> g_rec;
using UpdateFn = uint64_t (*)(uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t);
static UpdateFn g_update = nullptr;
static uint64_t g_handler_addr = 0;
static bool g_reenter_on_space_up = false;
static void fake_handler(uint64_t /*arg*/, void* ev) {
    auto* e = (uint8_t*)ev;
    const Rec rec{*(uint32_t*)(e + 0x00), *(uint16_t*)(e + 0x08)};
    g_rec.push_back(rec);
    if (g_reenter_on_space_up && rec.id == 0x102 && rec.keycode == 0x2c) {
        g_reenter_on_space_up = false;
        g_update(g_handler_addr, 0, 0, 0, 0, 0, 0);
    }
}

int main() {
    printf("== test_ime_input ==\n");
    const std::filesystem::path script_path =
        std::filesystem::temp_directory_path() / "prosper_test_ime_input.route";
    { std::ofstream route(script_path); route << "# headless keyboard route\nf5-6:0x2c\nf7-7:0x28\n"; }
    const std::string script_source = "@" + script_path.string();
#ifdef _WIN32
    _putenv_s("PROSPER_IME_SCRIPT", script_source.c_str());
#else
    setenv("PROSPER_IME_SCRIPT", script_source.c_str(), 1);
#endif
    register_builtin_hle();
    auto update = reinterpret_cast<UpdateFn>(Hle::lookup(nid_hash("sceImeUpdate")));
    CHECK(update != nullptr, "sceImeUpdate registered");
    if (!update) { printf("fails=%d\n", fails); return 1; }
    const uint64_t H = (uint64_t)(uintptr_t)&fake_handler;
    g_update = update;
    g_handler_addr = H;

    std::vector<ImeScriptWindow> parsed;
    std::string parse_error;
    CHECK(!parse_ime_script_route("f-1:0x28", parsed, &parse_error),
          "IME script rejects a negative frame anchor");
    CHECK(!parse_ime_script_route("f18446744073709551616:0x28", parsed, &parse_error),
          "IME script rejects an overflowing frame anchor");
    CHECK(!parse_ime_script_route("f1-18446744073709551616:0x28", parsed, &parse_error),
          "IME script rejects an overflowing range endpoint");

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

    // PROSPER_IME_SCRIPT is anchored to non-null sceImeUpdate calls (the null pump above does not
    // consume a frame). The f5-6 range emits one down transition, remains held, then emits one up.
    g_rec.clear();
    update(H, 0, 0, 0, 0, 0, 0); // f4
    CHECK(g_rec.empty(), "IME script stays idle before its frame window");
    update(H, 0, 0, 0, 0, 0, 0); // f5
    CHECK(g_rec.size() == 1 && g_rec[0].id == 0x101 && g_rec[0].keycode == 0x2c,
          "IME script emits key down at the range start");
    g_rec.clear();
    update(H, 0, 0, 0, 0, 0, 0); // f6
    CHECK(g_rec.empty(), "IME script holds a key without repeating it inside the range");
    g_reenter_on_space_up = true;
    update(H, 0, 0, 0, 0, 0, 0); // f7; callback re-enters at f8
    CHECK(g_rec.size() == 3 &&
              g_rec[0].id == 0x102 && g_rec[0].keycode == 0x2c &&
              g_rec[1].id == 0x101 && g_rec[1].keycode == 0x28 &&
              g_rec[2].id == 0x102 && g_rec[2].keycode == 0x28,
          "reentrant IME update preserves release/press/release transition order");

    std::error_code remove_error;
    std::filesystem::remove(script_path, remove_error);
    printf("fails=%d\n", fails);
    return fails ? 1 : 0;
}
