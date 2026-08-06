// test_agc_dcb_callback_abi — the AGC "command buffer full" callback is a HOST -> GUEST call, and
// its arguments must arrive in the guest's registers, not the host's (#2194).
//
// The guest registers this callback in its DCB and prosper invokes it from allocate_dw when a packet
// does not fit. guest -> host conversion happens in the emitted import stub; nothing converts a call
// going the OTHER way, so on Windows a direct call through the pointer passes arguments in MS-x64
// registers (rcx/rdx/r8) into guest code that reads SysV (rdi/rsi/rdx). They arrive shifted by one
// position, and the guest dereferences a dword count as a pointer.
//
// The callback below is declared __attribute__((sysv_abi)) precisely so it reads its arguments the
// way real guest code does. A plain host callback would be given the host convention on both sides
// and would pass whether or not the trampoline is used -- i.e. it could not fail, which is the trap
// this test exists to avoid.
#include "../src/hle/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); fails++; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

#ifdef _WIN32

// The DCB layout prosper models (hle_agc.cpp AgcDcb). Built as raw storage rather than reusing the
// struct, so the test also pins the OFFSETS the guest writes to -- if a field moves, this fails.
struct alignas(16) TestDcb {
    uint32_t* bottom;       // +0x00
    uint32_t* top;          // +0x08
    uint32_t* cursor_up;    // +0x10
    uint32_t* cursor_down;  // +0x18
    void*     callback;     // +0x20
    void*     user_data;    // +0x28
    uint32_t  reserved_dw;  // +0x30
    uint32_t  pad;
};
static_assert(offsetof(TestDcb, callback) == 0x20, "callback must sit at +0x20");
static_assert(offsetof(TestDcb, user_data) == 0x28, "user_data must sit at +0x28");
static_assert(offsetof(TestDcb, reserved_dw) == 0x30, "reserved_dw must sit at +0x30");

static volatile uint64_t g_seen_dcb;
static volatile uint64_t g_seen_need;
static volatile uint64_t g_seen_user;
static volatile int      g_calls;

// A stand-in for the guest's buffer-full handler, reading its arguments as guest code does.
// Returns false: "I could not grow the buffer", which makes allocate_dw give up rather than retry.
extern "C" __attribute__((sysv_abi)) bool test_dcb_full(void* dcb, uint32_t need, void* user) {
    g_seen_dcb = (uint64_t)(uintptr_t)dcb;
    g_seen_need = need;
    g_seen_user = (uint64_t)(uintptr_t)user;
    ++g_calls;
    return false;
}

#endif  // _WIN32

int main() {
#ifndef _WIN32
    std::printf("  [skip] host->guest ABI conversion is a Windows-only concern\n");
    std::printf("== PASS ==\n");
    return 0;
#else
    register_builtin_hle();   // populates the NID table, as every other AGC test does
    HleFn set_sh_range = Hle::lookup("n2fD4A+pb+g");   // sceAgcCbSetShRegisterRangeDirect
    CHECK(set_sh_range != nullptr, "sceAgcCbSetShRegisterRangeDirect is registered");
    if (!set_sh_range) { std::printf("== FAIL: %d ==\n", fails); return 1; }

    static uint32_t ring[64];
    TestDcb dcb{};
    dcb.bottom = ring;
    dcb.top = ring + 64;
    dcb.cursor_up = ring;
    // Deliberately tiny window: 2 dwords available, so the very first packet overflows and the
    // buffer-full callback is the path under test rather than an incidental one.
    dcb.cursor_down = ring + 2;
    dcb.callback = (void*)&test_dcb_full;
    dcb.user_data = (void*)(uintptr_t)0xD00DFEEDCAFEB00Dull;   // must arrive intact, not truncated
    dcb.reserved_dw = 0;

    g_seen_dcb = g_seen_need = g_seen_user = 0;
    g_calls = 0;

    static const uint32_t values[4] = {1, 2, 3, 4};
    // num=4 -> the packet needs 4+2 dwords against a 2-dword window, so allocate_dw must consult the
    // callback. It answers false, so the HLE returns 0 -- the point is the arguments, not the result.
    set_sh_range((uint64_t)(uintptr_t)&dcb, 0x100, (uint64_t)(uintptr_t)values, 4, 0, 0);

    CHECK(g_calls == 1, "the buffer-full callback was invoked exactly once");
    CHECK(g_seen_dcb == (uint64_t)(uintptr_t)&dcb,
          "argument 1 is the DCB, in the guest's first register");
    CHECK(g_seen_need == 6,
          "argument 2 is the dword count needed (4 values + 2 header), not something else");
    // The one that actually failed in #2194: the guest received `need` here instead of user_data,
    // dereferenced it, and took the title's primary thread down.
    CHECK(g_seen_user == 0xD00DFEEDCAFEB00Dull,
          "argument 3 is user_data, delivered whole and in the guest's THIRD register");

    if (fails) { std::printf("== FAIL: %d ==\n", fails); return 1; }
    std::printf("== PASS ==\n");
    return 0;
#endif
}
