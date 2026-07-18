// test_ajm (#187) — libSceAjm headless decode lifecycle. prosper doesn't decode compressed audio, but
// the handle lifecycle must be REAL: Initialize/InstanceCreate/BatchStartBuffer fill their handle
// out-param with a valid value (never the caller's garbage — the bug the generic stub caused), inputs
// are validated, and a batch reports started->complete so the guest's audio pipeline proceeds. Names
// hash to the game's import NIDs, so R()-registration reaches the guest.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static constexpr uint64_t kInvalidContext   = 0xffffffff80930002ull;
static constexpr uint64_t kInvalidInstance  = 0xffffffff80930003ull;
static constexpr uint64_t kInvalidParameter = 0xffffffff80930005ull;

int main() {
    printf("== test_ajm ==\n");
    register_builtin_hle();

    // The names must hash to the game's import NIDs (#187 table), else R() registers under the wrong key.
    CHECK(nid_hash("sceAjmInitialize") == "dl+4eHSzUu4", "sceAjmInitialize hashes to its import NID");
    CHECK(nid_hash("sceAjmModuleRegister") == "Q3dyFuwGn64", "sceAjmModuleRegister hashes to its import NID");

    HleFn init = Hle::lookup(nid_hash("sceAjmInitialize"));
    HleFn modreg = Hle::lookup(nid_hash("sceAjmModuleRegister"));
    HleFn icreate = Hle::lookup(nid_hash("sceAjmInstanceCreate"));
    HleFn idestroy = Hle::lookup(nid_hash("sceAjmInstanceDestroy"));
    HleFn bstart = Hle::lookup(nid_hash("sceAjmBatchStartBuffer"));
    HleFn bwait = Hle::lookup(nid_hash("sceAjmBatchWait"));
    HleFn fin = Hle::lookup(nid_hash("sceAjmFinalize"));
    CHECK(init && modreg && icreate && idestroy && bstart && bwait && fin, "AJM functions registered");
    if (!(init && modreg && icreate && idestroy && bstart && bwait && fin)) { printf("== FAIL ==\n"); return 1; }

    // Initialize fills a valid (non-zero) context handle.
    uint32_t ctx = 0xDEAD;
    CHECK(init(0, (uint64_t)(uintptr_t)&ctx, 0, 0, 0, 0) == 0, "sceAjmInitialize -> OK");
    CHECK(ctx != 0 && ctx != 0xDEAD, "Initialize WROTE a valid non-zero context (not left garbage)");
    CHECK(init(0, 0, 0, 0, 0, 0) == kInvalidParameter, "Initialize(NULL out) -> INVALID_PARAMETER");
    CHECK(init(0, 1, 0, 0, 0, 0) == kInvalidParameter,
          "Initialize(inaccessible out) -> INVALID_PARAMETER without host fault");
    uint32_t rejected_ctx = 0xCAFE;
    CHECK(init(1, (uint64_t)(uintptr_t)&rejected_ctx, 0, 0, 0, 0) == kInvalidParameter,
          "Initialize(nonzero reserved) -> INVALID_PARAMETER");
    CHECK(rejected_ctx == 0xCAFE, "invalid Initialize leaves the context output untouched");

    // ModuleRegister needs a context.
    CHECK(modreg(ctx, 1 /*At9Dec*/, 0, 0, 0, 0) == 0, "sceAjmModuleRegister(ctx) -> OK");
    CHECK(modreg(0, 1, 0, 0, 0, 0) == kInvalidContext, "ModuleRegister(context 0) -> INVALID_CONTEXT");

    // InstanceCreate fills a valid instance handle, distinct from the context.
    uint32_t inst = 0xBEEF;
    CHECK(icreate(ctx, 1 /*At9Dec*/, 0 /*flags*/, (uint64_t)(uintptr_t)&inst, 0, 0) == 0, "sceAjmInstanceCreate -> OK");
    CHECK(inst != 0 && inst != 0xBEEF && inst != ctx, "InstanceCreate WROTE a distinct non-zero instance");
    CHECK(icreate(0, 1, 0, (uint64_t)(uintptr_t)&inst, 0, 0) == kInvalidContext, "InstanceCreate(ctx 0) -> INVALID_CONTEXT");
    CHECK(icreate(ctx, 1, 0, 1, 0, 0) == kInvalidParameter,
          "InstanceCreate(inaccessible out) -> INVALID_PARAMETER without host fault");

    // Batch: StartBuffer fills a batch id; Wait completes it.
    uint32_t batch = 0xF00D;
    CHECK(bstart(ctx, 0 /*batch buf*/, 0 /*size*/, 0 /*prio*/, 0 /*err*/, (uint64_t)(uintptr_t)&batch) == 0,
          "sceAjmBatchStartBuffer -> OK");
    CHECK(batch != 0 && batch != 0xF00D, "BatchStartBuffer WROTE a valid non-zero batch id");
    CHECK(bstart(ctx, 0, 0, 0, 0, 1) == kInvalidParameter,
          "BatchStartBuffer(inaccessible out) -> INVALID_PARAMETER without host fault");
    CHECK(bwait(ctx, batch, 0, 0, 0, 0) == 0, "sceAjmBatchWait -> OK (batch completed)");

    // Teardown.
    CHECK(idestroy(ctx, inst, 0, 0, 0, 0) == 0, "sceAjmInstanceDestroy -> OK");
    CHECK(idestroy(0, inst, 0, 0, 0, 0) == kInvalidContext, "InstanceDestroy(ctx 0) -> INVALID_CONTEXT");
    CHECK(idestroy(ctx, 0, 0, 0, 0, 0) == kInvalidInstance, "InstanceDestroy(instance 0) -> INVALID_INSTANCE");
    CHECK(fin(0, 0, 0, 0, 0, 0) == 0, "sceAjmFinalize -> OK");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
