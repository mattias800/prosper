// test_agc_epilogue_window — #1748 guard: a prosper AGC builder must never emit more dwords than the
// hardware packet it stands for, because the guest reserves its command buffer from the REAL sizes.
//
// The failure this pins down (Asterix & Obelix - Babylon Mission, PPSA30490): the title's submit
// epilogue is a Dcb window of exactly 16 dwords holding sceAgcDcbAcquireMem (8, RDNA2 ACQUIRE_MEM)
// followed by sceAgcCbReleaseMem / the end-of-pipe action (8, RDNA2 RELEASE_MEM). prosper's
// ReleaseMem builder emitted 9 dwords, so the append did not fit and AgcDcb::allocate_dw invoked the
// guest's "buffer full" callback. That callback hands over a fresh ~600 KiB command-buffer chunk, so
// every single submit burned one — ~210 MiB/s of direct memory that is never reclaimed (the chunk's
// own completion fence lands in the NEXT chunk, which is never submitted), emptying prosper's
// direct-memory pool in ~125 s and killing a guest thread on the guest allocator's unchecked NULL.
//
// The assertions are behavioural, not a magic constant: build the two packets into a 16-dword window
// with a callback attached and require that the callback is NEVER reached. With the 9-dword builder
// the second append overflows and the callback fires, so this test fails.
//
// test_agc_getsize (#1143) is a different guard: it checks GetSize == builder, i.e. that the two
// agree. It passes for ANY common size and therefore cannot see this defect.
#include "../src/hle/dispatch.hpp"
#include "../src/gpu/pm4_decode.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace prosper;

// ABI mirror of AgcDcb (hle_agc.cpp): the builders cast a0 to this, advance cursor_up, and call
// `callback` when the request does not fit in [cursor_up, cursor_down).
struct Dcb {
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    bool (*callback)(Dcb*, uint32_t, void*); void* user_data; uint32_t reserved_dw; uint32_t pad;
};

// sceAgcCbReleaseMem is a 9-argument HLE (data_sel and the fence value arrive past the register
// args); Hle::lookup hands back the 6-argument HleFn, so call it through its real signature.
using HleFn9 = PROSPER_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                             uint64_t, uint64_t, uint64_t, uint64_t);

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("  [FAIL] "); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } \
                              else { std::printf("  [ok]   "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static int g_callback_hits = 0;
static bool grow_callback(Dcb*, uint32_t, void*) { ++g_callback_hits; return false; }

// The guest's epilogue window, reproduced exactly: 16 dwords, no reserve, growth callback attached.
static uint32_t g_window[16];
static Dcb epilogue_window() {
    std::memset(g_window, 0, sizeof g_window);
    Dcb d{};
    d.bottom = g_window;
    d.top = g_window + 16;
    d.cursor_up = g_window;
    d.cursor_down = g_window + 16;
    d.callback = grow_callback;
    d.reserved_dw = 0;
    return d;
}

int main() {
    std::printf("== test_agc_epilogue_window (#1748 packet-size overrun guard) ==\n");
    register_builtin_hle();

    HleFn acquire = Hle::lookup("57labkp+rSQ");   // sceAgcDcbAcquireMem
    HleFn release = Hle::lookup("wr23dPKyWc0");   // sceAgcCbReleaseMem (end-of-pipe action)
    CHECK(acquire && release, "sceAgcDcbAcquireMem + sceAgcCbReleaseMem registered");
    if (!acquire || !release) { std::printf("FAILED (%d)\n", ++fails); return 1; }

    // 1. Each builder emits exactly the dword count of the RDNA2 packet it stands for.
    {
        Dcb d = epilogue_window();
        g_callback_hits = 0;
        acquire((uint64_t)(uintptr_t)&d, 0, 0, 0, 0, 0);
        const uint64_t acquire_dw = (uint64_t)(d.cursor_up - g_window);
        CHECK(acquire_dw == 8, "sceAgcDcbAcquireMem emits %llu dwords (RDNA2 ACQUIRE_MEM = 8)",
              (unsigned long long)acquire_dw);
    }
    {
        Dcb d = epilogue_window();
        g_callback_hits = 0;
        release((uint64_t)(uintptr_t)&d, 0, 0, 0, 0, 0);
        const uint64_t release_dw = (uint64_t)(d.cursor_up - g_window);
        CHECK(release_dw == 8, "sceAgcCbReleaseMem emits %llu dwords (RDNA2 RELEASE_MEM = 8)",
              (unsigned long long)release_dw);
    }

    // 2. The real shape: both packets must fit the guest's 16-dword epilogue with no growth request.
    {
        Dcb d = epilogue_window();
        g_callback_hits = 0;
        const uint64_t acq = acquire((uint64_t)(uintptr_t)&d, 0, 0, 0, 0, 0);
        const uint64_t rel = ((HleFn9)release)((uint64_t)(uintptr_t)&d, 0x28, 0, 1, 0,
                                               0x1122334455667788ull, 2, 1, 0);
        CHECK(acq != 0 && rel != 0, "both builders appended a packet (acquire=%s release=%s)",
              acq ? "ok" : "REFUSED", rel ? "ok" : "REFUSED");
        CHECK(g_callback_hits == 0,
              "buffer-full callback never invoked (hits=%d) — the epilogue fits its 16-dword window",
              g_callback_hits);
        CHECK((uint64_t)(d.cursor_up - g_window) == 16,
              "epilogue consumed %llu of 16 dwords",
              (unsigned long long)(d.cursor_up - g_window));
    }

    // 3. The 8-dword packet still decodes, and the historical 9-dword form recorded in pre-#1748
    //    captures decodes with its full build snapshot.
    {
        Dcb d = epilogue_window();
        g_callback_hits = 0;
        // data_sel=2 (64-bit write), value 1 — the DCB chunk-completion fence PPSA30490 builds.
        ((HleFn9)release)((uint64_t)(uintptr_t)&d, 0x28, 0, 1, 0, 0x1122334455667788ull, 2, 1, 0);
        std::vector<gpu::Pm4Command> ops;
        gpu::decode_pm4(g_window, 8, ops);
        CHECK(ops.size() == 1 && ops[0].kind == gpu::Pm4Command::Kind::ReleaseMem,
              "8-dword packet decodes as one ReleaseMem (%zu ops)", ops.size());
        if (ops.size() == 1) {
            CHECK(ops[0].rel_addr == 0x1122334455667788ull, "decoded label address 0x%llx",
                  (unsigned long long)ops[0].rel_addr);
            CHECK(ops[0].rel_data_sel == 2, "decoded data_sel %u", ops[0].rel_data_sel);
            CHECK(ops[0].rel_value_valid && ops[0].rel_value == 1, "decoded fence value 0x%llx",
                  (unsigned long long)ops[0].rel_value);
            // The build snapshot survives as its HIGH half in the single spare payload dword. Assert
            // the value, not merely that the branch ran: a wrong shift decodes to something and would
            // silently change what stale_release_generation compares against. `label_build_pre` reads
            // the destination qword, and 0x1122334455667788 is not mapped, so the snapshot is 0 —
            // check the shift with a hand-built packet instead.
            uint32_t pkt[8] = {};
            std::memcpy(pkt, g_window, sizeof pkt);
            pkt[7] = 0xDEADBEEFu;                       // build_pre high dword
            std::vector<gpu::Pm4Command> hi;
            gpu::decode_pm4(pkt, 8, hi);
            CHECK(hi.size() == 1 && hi[0].rel_build_pre_valid &&
                  hi[0].rel_build_pre == 0xDEADBEEF00000000ull,
                  "8-dword build snapshot decodes to 0x%llx (high half, low half zero)",
                  (unsigned long long)(hi.empty() ? 0 : hi[0].rel_build_pre));
        }
    }
    {
        // Historical 9-dword packet: header + [addr lo/hi, data_sel, val lo/hi, action, pre lo/hi].
        uint32_t old_pkt[9] = {};
        old_pkt[0] = 0xC0000000u | ((9u - 2u) << 16) | (0x10u << 8) | (0x18u << 2);  // PM4(9, IT_NOP, R_RELEASE_MEM)
        old_pkt[1] = 0x55667788u; old_pkt[2] = 0x11223344u;
        old_pkt[3] = 1;
        old_pkt[4] = 1; old_pkt[5] = 0;
        old_pkt[6] = 0x28;                                     // event action
        old_pkt[7] = 0xAAAAAAAAu; old_pkt[8] = 0xBBBBBBBBu;    // build_pre lo/hi (full qword form)
        std::vector<gpu::Pm4Command> ops;
        gpu::decode_pm4(old_pkt, 9, ops);
        CHECK(ops.size() == 1 && ops[0].kind == gpu::Pm4Command::Kind::ReleaseMem,
              "9-dword historical packet decodes as one ReleaseMem (%zu ops)", ops.size());
        if (ops.size() == 1) {
            CHECK(ops[0].rel_addr == 0x1122334455667788ull, "historical label address 0x%llx",
                  (unsigned long long)ops[0].rel_addr);
            CHECK(ops[0].rel_build_pre_valid && ops[0].rel_build_pre == 0xBBBBBBBBAAAAAAAAull,
                  "historical build snapshot 0x%llx (full qword retained)",
                  (unsigned long long)ops[0].rel_build_pre);
        }
    }

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
