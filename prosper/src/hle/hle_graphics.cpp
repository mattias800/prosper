// hle_graphics.cpp — HEADLESS graphics bring-up.
//
// The PS5 GPU stack — libSceAgc (low-level GPU command building) and libSceVideoOut (display /
// flip) — is not yet translated to the host GPU (the eventual AGC->Vulkan + RDNA2 shader
// recompiler work, milestones M4/M5). Until then, these HLE handlers provide *valid but empty*
// graphics objects so the game's own logic (init, main loop, simulation) executes headlessly.
// This lets us map the real graphics call sequence to translate, and prove the runtime runs.
//
// This is NOT rendering and NOT faking visible output: object getters return zeroed buffers the
// guest can read/write without faulting; action/submit calls are no-ops. Every entry here is a
// placeholder to be replaced by a real translation. Graphics functions live in sparsely-documented
// Sony libs, so they're registered by raw NID with a note on the observed role.
#include "dispatch.hpp"
#include "nid.hpp"
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <utility>

namespace prosper {

#define HLE(name) static uint64_t name(uint64_t a0, uint64_t a1, uint64_t a2, \
                                       uint64_t a3, uint64_t a4, uint64_t a5)

namespace {
// Optional fork-safe counter bumped when the guest calls into the graphics libs (libSceAgc /
// libSceVideoOut). Tests use it to assert the boot advanced all the way into GPU/display init — a
// regression guard for the entire runtime bring-up (loader → IL2CPP → C# startup → services →
// graphics). Forward-compatible: these calls keep happening as we implement more, so the guard
// stays valid even once deeper blockers (e.g. the locale-facet fault) are fixed.
volatile int* g_gfx_counter = nullptr;
inline void gfx_tick() { if (g_gfx_counter) *g_gfx_counter = *g_gfx_counter + 1; }  // avoid -Wvolatile
}

void set_gfx_call_counter(volatile int* counter) { g_gfx_counter = counter; }

// sceAgcGraphicsGetRegisterDefaults2 / ...Internal (NIDs 2JtWUUiYBXs / wRbq6ZjNop4). Per Kyty
// (Graphics.cpp:766, 1307), these return a RegisterDefaults table the game reads to seed GPU register
// state. Layout: tbl0/1/2/3 (ShaderRegister** arrays), unknown[2], index table, count@0x38. Returning
// an empty (count=0) table gave the game nothing to build its internal register-offset table from, so
// Unity's Gen5 init left that table null and a later register lookup null-derefed (eboot+0x3b5ea6).
// We now return the REAL default tables (127 cx+sh+uc registers), vendored from Kyty in
// agc_reg_defaults.cpp, so the game can build a populated register-offset table.
extern "C" void* prosper_agc_reg_defaults(unsigned int ver);           // -> g_reg_defaults1 (127 regs)
extern "C" void* prosper_agc_reg_defaults_internal(unsigned int ver);  // -> g_reg_defaults2 (21 regs)
HLE(g_agc_regdefs)     { gfx_tick(); return (uint64_t)(uintptr_t)prosper_agc_reg_defaults((unsigned)a0); }
HLE(g_agc_regdefs_int) { gfx_tick(); return (uint64_t)(uintptr_t)prosper_agc_reg_defaults_internal((unsigned)a0); }

// +kSrjIVxKFE (kAgcNids[14]): the AGC register-context constructor. The game calls this on its
// register context (embedded at device+0x48) as the first operation, expecting it to install a
// "register classify table" at [context+0x08]. That table is consumed by the eboot-internal
// classifier at eboot+0x3b5ea0: classify(table, sel, key) = (key < table.limit16[sel]) ?
// table.subarray[sel][key] : 0x7fff, where limit16[] is a u16[4] at table+0x2e and subarray*[] is a
// void*[4] at table+0x08. With no real impl the context stayed zeroed and the register-set loop
// null-derefed the table (boot blocker, see docs/GRAPHICS.md).
//
// The register banks are an array of 0x70-byte sub-objects at context+0x38 (index 0..2 = the cx/sh/uc
// register sets): the setter thunks (eboot+0x3a7aa0/0x3a7b20/0x3a7b60) do `add $0x38,%rdi` then
// `sub = (context+0x38) + sel*0x70` before running the register-set loop. Each sub-object holds its
// own classify table at [sub+0x08] and register banks at [sub+0x10]/[sub+0x18]. So the constructor
// must install a table into every sub-object, not into the context base.
//
// STAGE 1 (this commit): install a zeroed table in each sub-object -> every per-selector limit is 0
// -> classify (eboot+0x3b5ea0) returns 0x7fff for all keys -> the register-set loops skip every
// register (their `cmp $0x7fff; je skip`) without ever touching the (still-null) register banks. This
// is a structurally-valid empty init: it unblocks the boot so the NEXT blocker is observable. STAGE 2
// will populate the tables from the real register offsets (agc_reg_defaults.cpp) and allocate the
// register banks so registers are actually stored.
namespace {
alignas(64) uint8_t g_agc_ctx_regmap[0x40] = {0};   // all four per-selector limits (u16 @ +0x2e) = 0
constexpr int   kAgcCtxSubCount  = 3;               // cx / sh / uc register-set sub-objects
constexpr size_t kAgcCtxSubBase  = 0x38;            // first sub-object offset inside the context
constexpr size_t kAgcCtxSubStride = 0x70;           // per sub-object (matches eboot+0x3b0210: *0x70)
}
HLE(g_agc_ctx_init) {   // a0 = context pointer (device+0x48)
    gfx_tick();
    if (a0) {
        uint8_t* ctx = (uint8_t*)(uintptr_t)a0;
        for (int i = 0; i < kAgcCtxSubCount; i++)
            *(void**)(ctx + kAgcCtxSubBase + i * kAgcCtxSubStride + 0x08) = g_agc_ctx_regmap;
    }
    return 0;
}

// --- libSceVideoOut (display / frame presentation). Headless: accept opens/flips and simulate
// flip completion so the game's render loop advances (submit -> wait completion -> submit next).
namespace {
    int g_vo_handle = 0;
    uint64_t g_flip_count = 0;   // incremented per SubmitFlip so GetFlipStatus shows progress
}
HLE(g_vo_open)        { gfx_tick(); return (uint64_t)(int64_t)(++g_vo_handle + 0x1000); }  // positive handle
HLE(g_vo_close)       { return 0; }
HLE(g_vo_submitflip)  { g_flip_count++; return 0; }                            // accept the flip
HLE(g_vo_flippending) { return 0; }                                            // never pending -> can submit next
HLE(g_vo_flipstatus)  { // (handle, SceVideoOutFlipStatus* status): report our simulated flip count.
    // SceVideoOutFlipStatus is exactly 0x40 bytes — writing more smashes the caller's stack canary!
    if (a1) { uint8_t* s = (uint8_t*)(uintptr_t)a1; memset(s, 0, 0x40);
              *(uint64_t*)(s + 0x00) = g_flip_count;          // count
              *(int64_t*) (s + 0x18) = (int64_t)g_flip_count; // flipArg
              *(int32_t*) (s + 0x38) = 0; }                   // currentBuffer
    return 0;
}
HLE(g_vo_resstatus)   { if (a1) memset((void*)(uintptr_t)a1, 0, 0x20); return 0; }  // SceVideoOutResolutionStatus ~0x20

// Diagnostic tracer for the (undocumented) libSceAgc / libSceAgcDriver calls: logs the NID, the guest
// callsite, and all six args (gated on PROSPER_GFXLOG). Behaviour is identical to the unimplemented
// stub — returns 0, changes no control flow — it is purely observable. This is the RE bootstrap for
// the real AGC->Vulkan work (M4/M5), NOT faked output.
//
// Per-NID identification: each NID is bound to a distinct template thunk `glog_thunk<I>` that knows
// its own name (kAgcNids[I]). The guest reaches the thunk via our stub's tail-jump, so at thunk entry
// [rsp] is the guest return address into eboot (base 0x400000000) — __builtin_return_address(0) gives
// the exact callsite. Result: `PROSPER_GFXLOG=1 boot_trace <dump>` emits a self-describing dataset
// (NID + callsite + args per call) — everything needed to RE the AGC object model.
namespace {
// The libSceAgc/AgcDriver NIDs the game calls (observed via boot_trace's unimplemented log). The
// first four are AgcDriver; the rest are libSceAgc functions used during GfxDevicePS5SharedData init
// whose null results currently leave GPU objects zeroed — the graphics blocker.
const char* const kAgcNids[] = {
    "MM4IZSEYytQ", "XlNp7jzGiPo", "Zw7uUVPulbw", "w2rJhmD+dsE",
    "23LRUSvYu1M", "BfBDZGbti7A", "TRO721eVt4g", "MWiElSNE8j8", "-KRzWekV120", "wr23dPKyWc0",
    "VmW0Tdpy420", "ZvwO9euwYzc", "vcmNN+AAXnY", "d-6uF9sZDIU", "+kSrjIVxKFE", "H7uZqCoNuWk",
    "f3dg2CSgRKY", "hvUfkUIQcOE", "6lNcCp+fxi4", "vRoArM9zaIk", "0fWWK5uG9rQ", "3KDcnM3lrcU",
    "57labkp+rSQ", "LtTouSCZjHM", "V++UgBtQhn0", "aJf+j5yntiU", "fPSCdQxgpSw", "i1jyy49AjXU",
    "tSBxhAPyytQ",   // fires once after the Sh patch pair during CreateWorkload; not in Kyty — RE via args
    "k3GhuSNmBLU",   // fires once just before "unity default resources" load; not in Kyty — RE via args
};
constexpr size_t kAgcNidCount = sizeof(kAgcNids) / sizeof(kAgcNids[0]);

uint64_t glog_impl(const char* nid, void* ra,
                   uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4, uint64_t a5) {
    gfx_tick();
    if (getenv("PROSPER_GFXLOG"))
        fprintf(stderr, "[gfx] libSceAgc::%s  from eboot+0x%llx  a0=0x%llx a1=0x%llx a2=0x%llx a3=0x%llx a4=0x%llx a5=0x%llx\n",
                nid, (unsigned long long)((uint64_t)ra - 0x400000000ull),
                (unsigned long long)a0, (unsigned long long)a1, (unsigned long long)a2,
                (unsigned long long)a3, (unsigned long long)a4, (unsigned long long)a5);
    // Shader-recompiler prep: dump the blob args of sceAgcCreateShader (f3dg2CSgRKY) to identify the
    // RDNA2 shader container format. Args are game-heap pointers (already mapped), so the reads are
    // safe; we only touch plausible pointers. Gated on PROSPER_AGCSHADER.
    if (getenv("PROSPER_AGCSHADER") && !strcmp(nid, "f3dg2CSgRKY")) {
        const uint64_t args[4] = {a0, a1, a2, a3};
        for (int ai = 0; ai < 4; ai++) {
            if (args[ai] < 0x100000ull) { fprintf(stderr, "  a%d=0x%llx (immediate)\n", ai, (unsigned long long)args[ai]); continue; }
            const uint8_t* p = (const uint8_t*)(uintptr_t)args[ai];
            fprintf(stderr, "  a%d=0x%llx: ", ai, (unsigned long long)args[ai]);
            for (int b = 0; b < 32; b++) fprintf(stderr, "%02x ", p[b]);
            fprintf(stderr, " | ");
            for (int b = 0; b < 32; b++) { char c = (char)p[b]; fprintf(stderr, "%c", (c >= 32 && c < 127) ? c : '.'); }
            fprintf(stderr, "\n");
        }
        // Dump the first shader's code blob (a2) to a file so it can be disassembled as RDNA2.
        static bool dumped = false;
        if (!dumped && a2 > 0x100000ull) {
            dumped = true;
            uint64_t sz = (a4 && a4 <= 0x10000) ? a4 : 0x200;   // a4 looks like a size; cap defensively
            if (FILE* f = fopen("/mnt/c/Users/matti/repos/ps5ys/prosper/build-linux/shader0.bin", "wb")) {
                fwrite((const void*)(uintptr_t)a2, 1, sz, f); fclose(f);
                fprintf(stderr, "  [wrote shader0.bin: %llu bytes from a2]\n", (unsigned long long)sz);
            }
        }
    }
    return 0;
}
template <size_t I>
__attribute__((noinline)) uint64_t glog_thunk(uint64_t a0, uint64_t a1, uint64_t a2,
                                              uint64_t a3, uint64_t a4, uint64_t a5) {
    return glog_impl(kAgcNids[I], __builtin_return_address(0), a0, a1, a2, a3, a4, a5);
}
template <size_t... Is>
void register_agc_tracers(std::index_sequence<Is...>) {
    (Hle::register_fn(kAgcNids[Is], (HleFn)&glog_thunk<Is>, kAgcNids[Is]), ...);
}
}

void register_graphics_hle() {
    #define RN(nid, fn) Hle::register_fn(nid, (HleFn)(fn), nid)   // raw NID (graphics libs undocumented)
    #define R(str, fn)  Hle::register_fn(nid_hash(str), (HleFn)(fn), str)
    // libSceAgc getters whose results the guest dereferences → return stable zeroed objects.
    RN("2JtWUUiYBXs", g_agc_regdefs);       // GraphicsGetRegisterDefaults2       -> g_reg_defaults1
    RN("wRbq6ZjNop4", g_agc_regdefs_int);   // GraphicsGetRegisterDefaults2Internal -> g_reg_defaults2
    // All traced libSceAgc/AgcDriver NIDs → per-NID logging thunks (still return 0; observable only).
    register_agc_tracers(std::make_index_sequence<kAgcNidCount>{});
    // Override the +kSrjIVxKFE tracer with the real register-context constructor (must come AFTER the
    // tracer registration above so it wins; registry is last-write-wins per NID).
    RN("+kSrjIVxKFE", g_agc_ctx_init);      // AGC register-context init (installs classify table)
    // libSceVideoOut display / flip
    R("sceVideoOutOpen", g_vo_open);            R("sceVideoOutClose", g_vo_close);
    R("sceVideoOutSubmitFlip", g_vo_submitflip);R("sceVideoOutIsFlipPending", g_vo_flippending);
    R("sceVideoOutSetFlipRate", g_vo_close);    R("sceVideoOutAddFlipEvent", g_vo_close);
    R("sceVideoOutGetFlipStatus", g_vo_flipstatus);
    R("sceVideoOutGetResolutionStatus", g_vo_resstatus);
    R("sceVideoOutRegisterBuffers", g_vo_close);R("sceVideoOutSetBufferAttribute", g_vo_close);
    #undef R
    #undef RN
}

} // namespace prosper
