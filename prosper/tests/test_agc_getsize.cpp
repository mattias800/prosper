// test_agc_getsize — #1143 drift guard for the AGC GetSize cluster.
//
// The guest reserves command-buffer space with sceAgc*GetSize BEFORE calling the matching builder,
// so GetSize must equal the exact dword count the builder emits. Today the constants (Jump=5, AcquireMem
// =8, ReleaseMem/EOP=8, Rewind=2, Nop=n dwords) are correct but were untested — a future change to a
// builder's `begin_packet(a0, N, ...)` size would silently desync the reservation and corrupt the stream.
//
// This test MEASURES each builder's actual cursor advance (the real emitted size) and asserts its GetSize
// == that × 4, so any drift fails here. Rewind has no builder (reserve-only); its GetSize is asserted
// against the documented 2-dword contract that a future REWIND builder must honor.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace prosper;

// ABI mirror of AgcDcb (hle_agc.cpp) — the builders cast a0 to this and advance cursor_up.
struct Dcb {
    uint32_t* bottom; uint32_t* top; uint32_t* cursor_up; uint32_t* cursor_down;
    void* callback; void* user_data; uint32_t reserved_dw; uint32_t pad;
};

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("  [FAIL] "); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } \
                              else { std::printf("  [ok]   "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static uint32_t g_buf[256];
static Dcb fresh() {
    std::memset(g_buf, 0, sizeof g_buf);
    Dcb d{}; d.bottom = g_buf; d.top = g_buf + 256; d.cursor_up = g_buf; d.cursor_down = g_buf + 256;
    return d;
}

// Some builders are 9-argument HLEs (DmaData reads a8, ReleaseMem reads a6/a7). Calling them through
// the 6-argument HleFn leaves those reading whatever the stack held, which does not change the
// emitted size for any of them but is still uninitialised-read UB and would trip a sanitiser. Call
// every builder through the wide signature with explicit zeros instead.
using HleFn9 = PROSPER_SYSV_ABI uint64_t (*)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                                             uint64_t, uint64_t, uint64_t, uint64_t);

// Call a builder, return the dwords it emitted (cursor advance).
static uint64_t emitted(HleFn builder, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    Dcb d = fresh();
    ((HleFn9)builder)((uint64_t)(uintptr_t)&d, a1, a2, a3, a4, a5, 0, 0, 0);
    return (uint64_t)(d.cursor_up - g_buf);
}

int main() {
    std::printf("== test_agc_getsize (#1143 builder/GetSize drift guard) ==\n");
    register_builtin_hle();

    // (getsize NID, builder NID, EXPECTED dwords, builder args).
    //
    // `exp_dw` is the point of this table, not decoration. Asserting only "GetSize == builder" is a
    // DRIFT guard: both sides read the same `kDw*` constant, so it stays green at any value and
    // cannot see a builder whose size is simply wrong. #1756 shipped exactly that failure — a commit
    // whose doc claimed DMA_DATA was 7 while the code emitted 9, with the suite green — and a full
    // ctest is still green today with `kDwDmaData` reverted to 9 unless this column exists. Pinning
    // the absolute count makes it a VALUE guard: changing a builder's size now fails here and must be
    // changed deliberately, in the one place, with the reason.
    struct Case { const char* name; const char* gs_nid; const char* build_nid; uint64_t exp_dw;
                  uint64_t a1,a2,a3,a4,a5; };
    const Case cases[] = {
        { "Jump",           "VEGu4dixjUg", "xSAR0LTcRKM", 5, 0, 0, 0x1000, 0, 0 },  // sceAgcDcbJump -> 5 dw
        { "AcquireMem/Dcb", "-vnlTPPXPrw", "57labkp+rSQ", 8, 0, 0, 0,      0, 0 },  // sceAgcDcbAcquireMem -> 8 dw
        { "AcquireMem/Acb", "ewobAQeMo5k", "KT-hTp-Ch14", 8, 0, 0, 0,      0, 0 },  // sceAgcAcbAcquireMem -> 8 dw
        { "ReleaseMem/EOP", "hL7C0IRpWZI", "wr23dPKyWc0", 8, 0, 0, 0,      0, 0 },  // sceAgcCbReleaseMem -> 8 dw
        // #1756: the rest of the family. libSceAgc 3.20 exports 65 GetSize functions and prosper
        // answered 6; the other 59 fell through to the unimplemented path and returned 0, which
        // makes a guest that sizes its buffer from GetSize reserve NOTHING. Every pair below is
        // asserted the same way — the GetSize must equal what the builder actually writes — so the
        // fix cannot drift back apart. (The three size-carrying builders are excluded: their
        // GetSize argument position is unknown. See #1756.)
        { "DrawIndex",          "6ee9Hd3EWXQ", "q88lQ+GP5Yk", 7, 0, 0, 0, 0, 0 },
        { "DrawIndexAuto",      "WrdP9Zxx3lQ", "Yw0jKSqop+E", 7, 0, 0, 0, 0, 0 },
        { "DrawIndexOffset",    "qMlfB1ZhMDc", "B+aG9DUnTKA", 3, 0, 0, 0, 0, 0 },
        { "DrawIndexIndirect",  "mStuvI0zOtc", "t1vNu082-jM", 4, 0, 0, 0, 0, 0 },
        { "Dispatch",           "Abendgtz+3o", "k3GhuSNmBLU", 6, 0, 0, 0, 0, 0 },
        { "DispatchIndirect/D", "w8HVkEeXPv8", "CtB+A9-VxO0", 4, 0, 0, 0, 0, 0 },
        { "DispatchIndirect/A", "PxKWV2fVAps", "j3EtxFkSIhQ", 4, 0, 0, 0, 0, 0 },
        { "DmaData/Dcb",        "2ccJz9LQI+w", "WmAc2MEj6Io", 7, 0, 0, 0, 0, 0 },
        { "DmaData/Acb",        "M0ttm8h7SKA", "-RnpfpxIhec", 7, 0, 0, 0, 0, 0 },
        { "EventWrite/Dcb",     "C4l9fB17t8w", "aJf+j5yntiU", 4, 0, 0, 0, 0, 0 },
        { "EventWrite/Acb",     "Y-5vneiBtzk", "cFazmnXpJOE", 4, 0, 0, 0, 0, 0 },
        { "SetIndexBuffer",     "j4emHHndCPY", "l4fM9K-Lyks", 3, 0, 0, 0, 0, 0 },
        { "SetIndexCount",      "mljzuGDZRQ4", "8N2tmT3jmC8", 2, 0, 0, 0, 0, 0 },
        { "SetIndexSize",       "ca4KPvp0qLQ", "GIIW2J37e70", 2, 0, 0, 0, 0, 0 },
        { "SetNumInstances",    "6DFuRKT4C9w", "tSBxhAPyytQ", 2, 0, 0, 0, 0, 0 },
        { "StallCbParser",      "+u6dKSLWM2o", "u2T2DiA5hRI", 2, 0, 0, 0, 0, 0 },
        { "SetShRegDirect",     "QhPDD513V0w", "pFLArOT53+w", 3, 0, 0, 0, 0, 0 },
        { "SetCxRegDirect",     "1DeUNpRIDDA", "LHFXRrlTPD8", 3, 0, 0, 0, 0, 0 },
        { "SetUcRegDirect",     "aP1Ki9G3++4", "w4-d0n60hdo", 3, 0, 0, 0, 0, 0 },
        { "SetShRegsIndirect",  "nNlUtdDDvZ0", "-HOOCn0JY48", 4, 0, 0, 0, 0, 0 },
        { "SetCxRegsIndirect",  "GBCh3zCihoU", "ZvwO9euwYzc", 4, 0, 0, 0, 0, 0 },
        { "SetUcRegsIndirect",  "UQGTw4xRlcM", "hvUfkUIQcOE", 4, 0, 0, 0, 0, 0 },
    };
    for (const auto& c : cases) {
        HleFn gs = Hle::lookup(c.gs_nid), build = Hle::lookup(c.build_nid);
        CHECK(gs && build, "%s: GetSize + builder registered", c.name);
        if (!gs || !build) continue;
        uint64_t dw = emitted(build, c.a1, c.a2, c.a3, c.a4, c.a5);
        uint64_t sz = gs(0, 0, 0, 0, 0, 0);
        CHECK(dw == c.exp_dw && sz == c.exp_dw * 4,
              "%s: builder emits %llu dwords and GetSize says %llu bytes; both must be the pinned %llu",
              c.name, (unsigned long long)dw, (unsigned long long)sz, (unsigned long long)c.exp_dw);
    }

    // Nop: GetSize(n) must equal the builder's n-dword emission for several counts (the builder floors at 1).
    {
        HleFn nop_gs = Hle::lookup("t7PlZ9nt5Lc"), nop = Hle::lookup("LtTouSCZjHM");
        CHECK(nop_gs && nop, "Nop: GetSize + builder registered");
        if (nop_gs && nop) {
            for (uint64_t n : { (uint64_t)1, (uint64_t)5, (uint64_t)13 }) {
                uint64_t dw = emitted(nop, n, 0, 0, 0, 0);   // builder count is a1
                uint64_t sz = nop_gs(n, 0, 0, 0, 0, 0);      // GetSize count is a0
                CHECK(dw == n && sz == dw * 4,
                      "Nop(%llu): GetSize=%llu == builder %llu dwords * 4",
                      (unsigned long long)n, (unsigned long long)sz, (unsigned long long)dw);
            }
        }
    }

    // Rewind: reserve-only (no builder today). Lockstep contract (#1143 note 2): a future REWIND builder
    // must emit <= 2 dwords, or this GetSize must change with it.
    {
        HleFn rewind_gs = Hle::lookup("QIXCsbipds0");
        CHECK(rewind_gs != nullptr, "Rewind: GetSize registered");
        if (rewind_gs)
            CHECK(rewind_gs(0, 0, 0, 0, 0, 0) == 2u * 4u,
                  "RewindGetSize == 2 dwords * 4 (reserve-only; a future builder must emit <= 2 dw)");
    }

    std::printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
