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

// Call a builder, return the dwords it emitted (cursor advance).
template <class Fn> static uint64_t emitted(Fn builder, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    Dcb d = fresh();
    builder((uint64_t)(uintptr_t)&d, a1, a2, a3, a4, a5);
    return (uint64_t)(d.cursor_up - g_buf);
}

int main() {
    std::printf("== test_agc_getsize (#1143 builder/GetSize drift guard) ==\n");
    register_builtin_hle();

    // (getsize NID, builder NID, builder args) — GetSize must equal builder-emitted-dwords * 4.
    struct Case { const char* name; const char* gs_nid; const char* build_nid; uint64_t a1,a2,a3,a4,a5; };
    const Case cases[] = {
        { "Jump",           "VEGu4dixjUg", "xSAR0LTcRKM", 0, 0, 0x1000, 0, 0 },  // sceAgcDcbJump -> 5 dw
        { "AcquireMem/Dcb", "-vnlTPPXPrw", "57labkp+rSQ", 0, 0, 0,      0, 0 },  // sceAgcDcbAcquireMem -> 8 dw
        { "AcquireMem/Acb", "ewobAQeMo5k", "KT-hTp-Ch14", 0, 0, 0,      0, 0 },  // sceAgcAcbAcquireMem -> 8 dw
        { "ReleaseMem/EOP", "hL7C0IRpWZI", "wr23dPKyWc0", 0, 0, 0,      0, 0 },  // sceAgcCbReleaseMem -> 8 dw
        // #1756: the rest of the family. libSceAgc 3.20 exports 65 GetSize functions and prosper
        // answered 6; the other 59 fell through to the unimplemented path and returned 0, which
        // makes a guest that sizes its buffer from GetSize reserve NOTHING. Every pair below is
        // asserted the same way — the GetSize must equal what the builder actually writes — so the
        // fix cannot drift back apart. (The three size-carrying builders are excluded: their
        // GetSize argument position is unknown. See #1756.)
        { "DrawIndex",          "6ee9Hd3EWXQ", "q88lQ+GP5Yk", 0, 0, 0, 0, 0 },
        { "DrawIndexAuto",      "WrdP9Zxx3lQ", "Yw0jKSqop+E", 0, 0, 0, 0, 0 },
        { "DrawIndexOffset",    "qMlfB1ZhMDc", "B+aG9DUnTKA", 0, 0, 0, 0, 0 },
        { "DrawIndexIndirect",  "mStuvI0zOtc", "t1vNu082-jM", 0, 0, 0, 0, 0 },
        { "Dispatch",           "Abendgtz+3o", "k3GhuSNmBLU", 0, 0, 0, 0, 0 },
        { "DispatchIndirect/D", "w8HVkEeXPv8", "CtB+A9-VxO0", 0, 0, 0, 0, 0 },
        { "DispatchIndirect/A", "PxKWV2fVAps", "j3EtxFkSIhQ", 0, 0, 0, 0, 0 },
        { "DmaData/Dcb",        "2ccJz9LQI+w", "WmAc2MEj6Io", 0, 0, 0, 0, 0 },
        { "DmaData/Acb",        "M0ttm8h7SKA", "-RnpfpxIhec", 0, 0, 0, 0, 0 },
        { "EventWrite/Dcb",     "C4l9fB17t8w", "aJf+j5yntiU", 0, 0, 0, 0, 0 },
        { "EventWrite/Acb",     "Y-5vneiBtzk", "cFazmnXpJOE", 0, 0, 0, 0, 0 },
        { "SetIndexBuffer",     "j4emHHndCPY", "l4fM9K-Lyks", 0, 0, 0, 0, 0 },
        { "SetIndexCount",      "mljzuGDZRQ4", "8N2tmT3jmC8", 0, 0, 0, 0, 0 },
        { "SetIndexSize",       "ca4KPvp0qLQ", "GIIW2J37e70", 0, 0, 0, 0, 0 },
        { "SetNumInstances",    "6DFuRKT4C9w", "tSBxhAPyytQ", 0, 0, 0, 0, 0 },
        { "StallCbParser",      "+u6dKSLWM2o", "u2T2DiA5hRI", 0, 0, 0, 0, 0 },
        { "SetShRegDirect",     "QhPDD513V0w", "pFLArOT53+w", 0, 0, 0, 0, 0 },
        { "SetCxRegDirect",     "1DeUNpRIDDA", "LHFXRrlTPD8", 0, 0, 0, 0, 0 },
        { "SetUcRegDirect",     "aP1Ki9G3++4", "w4-d0n60hdo", 0, 0, 0, 0, 0 },
        { "SetShRegsIndirect",  "nNlUtdDDvZ0", "-HOOCn0JY48", 0, 0, 0, 0, 0 },
        { "SetCxRegsIndirect",  "GBCh3zCihoU", "ZvwO9euwYzc", 0, 0, 0, 0, 0 },
        { "SetUcRegsIndirect",  "UQGTw4xRlcM", "hvUfkUIQcOE", 0, 0, 0, 0, 0 },
    };
    for (const auto& c : cases) {
        HleFn gs = Hle::lookup(c.gs_nid), build = Hle::lookup(c.build_nid);
        CHECK(gs && build, "%s: GetSize + builder registered", c.name);
        if (!gs || !build) continue;
        uint64_t dw = emitted(build, c.a1, c.a2, c.a3, c.a4, c.a5);
        uint64_t sz = gs(0, 0, 0, 0, 0, 0);
        CHECK(dw > 0 && sz == dw * 4,
              "%s: GetSize=%llu == builder %llu dwords * 4", c.name,
              (unsigned long long)sz, (unsigned long long)dw);
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
