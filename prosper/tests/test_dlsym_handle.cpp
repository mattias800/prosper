// test_dlsym_handle — sceKernelDlsym must honor its MODULE HANDLE (#147). Two modules exporting
// the same NID: a lookup against the second module's handle must return the SECOND's address —
// the old global first-definition-wins table aliased it to the first (wrong plugin initialized).
// LoadStartModule resolves a linked-module path (basename match) to a real handle; an unknown path
// returns ENOENT (#146) — dlsym against a non-module handle still falls back to the global table.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <cstdint>
#include <unordered_map>
#include <string>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_dlsym_handle ==\n");
    register_builtin_hle();
    auto load  = Hle::lookup(nid_hash("sceKernelLoadStartModule"));
    auto dlsym = Hle::lookup(nid_hash("sceKernelDlsym"));
    CHECK(load && dlsym, "LoadStartModule + Dlsym registered");
    if (fails) { printf("== FAIL ==\n"); return 1; }

    // Two synthetic modules that BOTH export PSN_PrxInitialize, at different addresses; the
    // global table holds the first definition (what the linker's first-wins pass produces).
    static const std::unordered_map<std::string, uint64_t> modA{ { nid_hash("PSN_PrxInitialize"), 0x1111 },
                                                                 { nid_hash("OnlyInA"),           0xAAAA } };
    static const std::unordered_map<std::string, uint64_t> modB{ { nid_hash("PSN_PrxInitialize"), 0x2222 } };
    static const std::unordered_map<std::string, uint64_t> global{ { nid_hash("PSN_PrxInitialize"), 0x1111 },
                                                                   { nid_hash("OnlyInA"),           0xAAAA } };
    set_module_export_tables({ { "/host/dump/sce_module/A.prx", &modA },
                               { "/host/dump/sce_module/B.prx", &modB } });
    set_module_exports(&global);

    auto U = [](const void* p) { return (uint64_t)(uintptr_t)p; };
    uint64_t hA = load(U("/app0/sce_module/A.prx"), 0, 0, 0, 0, 0);
    uint64_t hB = load(U("/app0/sce_module/B.prx"), 0, 0, 0, 0, 0);
    CHECK(hA >= 0x10000 && hB >= 0x10000 && hA != hB,
          "linked-module paths resolve to real, distinct handles (basename match)");
    // Case-INSENSITIVE basename match (#146): the guest's runtime path can differ in case from the
    // preload path (the Messenger loads "Il2CppUserAssemblies.prx" vs preloaded "Il2cpp..."). A
    // different-case load of A.prx must resolve to the SAME real handle, not ENOENT.
    CHECK(load(U("/app0/other/a.PRX"), 0, 0, 0, 0, 0) == hA,
          "different-case basename resolves to the same module handle (case-insensitive)");

    uint64_t addr = 0;
    CHECK(dlsym(hA, U("PSN_PrxInitialize"), U(&addr), 0, 0, 0) == 0 && addr == 0x1111,
          "dlsym(handle A) returns A's export");
    addr = 0;
    CHECK(dlsym(hB, U("PSN_PrxInitialize"), U(&addr), 0, 0, 0) == 0 && addr == 0x2222,
          "dlsym(handle B) returns B's export (NOT the global first definition)");

    // A symbol absent from the handle's module falls back to the global table.
    addr = 0;
    CHECK(dlsym(hB, U("OnlyInA"), U(&addr), 0, 0, 0) == 0 && addr == 0xAAAA,
          "dlsym(handle B, symbol only in A) falls back to the global table");

    // Unknown path -> ENOENT (#146): a PRX not in the linked set isn't present, so LoadStartModule
    // reports module-not-found instead of a fake success handle that loads nothing.
    uint64_t hX = load(U("/app0/sce_module/NotLinked.prx"), 0, 0, 0, 0, 0);
    CHECK(hX == 0x80020002ull, "unknown path -> SCE_KERNEL_ERROR_ENOENT (not a fake success handle)");
    // A dlsym against a non-module handle (0, or the ENOENT value) still falls back to the global
    // export table — the PSN.prx / UnityPluginLoad by-name path is unaffected.
    addr = 0;
    CHECK(dlsym(0, U("PSN_PrxInitialize"), U(&addr), 0, 0, 0) == 0 && addr == 0x1111,
          "dlsym(non-module handle) resolves via the global table");

    // Unknown symbol: ESRCH, out param untouched (callers pre-seed a fallback).
    addr = 0xFEED;
    CHECK(dlsym(hA, U("NoSuchExport"), U(&addr), 0, 0, 0) == 0x80020003 && addr == 0xFEED,
          "unknown symbol -> ESRCH, *funcAddr untouched");

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
