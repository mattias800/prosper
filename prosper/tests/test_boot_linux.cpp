// test_boot_linux — M3/M4 integration test. Links the main executable together with
// the game's own modules (Il2cppUserAssemblies = compiled C#, PS5Util) and boots.
// Cross-module imports now resolve to real code; only true system calls hit HLE/stubs.
// The guest is multithreaded and will eventually fault in a stubbed path during
// bring-up, so we run it in a forked child and measure depth via shared memory.
#include "../src/self/module.hpp"
#include "../src/loader/linker.hpp"
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <ctime>

using namespace prosper;

int main(int argc, char** argv) {
    std::string dump = (argc >= 2) ? argv[1] : "../../PPSA24651-app0";
    printf("== test_boot_linux (multi-module): %s ==\n", dump.c_str());

    // Main executable first, then the game's own PRX modules.
    std::vector<LinkInput> inputs = {
        { dump + "/eboot.bin",                    0x400000000ull },
        { dump + "/Media/Modules/Il2cppUserAssemblies.prx", 0x440000000ull },
        { dump + "/Media/Modules/PS5Util.prx",    0x4c0000000ull },
    };
    const uint64_t STUB_BASE = 0x600000000ull;

    Program prog;
    std::string err;
    if (!link_program(inputs, STUB_BASE, prog, &err)) { printf("  [FAIL] link: %s\n", err.c_str()); return 1; }
    printf("  linked %zu modules: %zu imports (%zu cross-module, %zu stubbed / %zu slots)\n",
           prog.mods.size(), prog.total_imports, prog.resolved_cross_module, prog.stubbed, prog.slots.size());

    // Register the linked module set so sceKernelLoadStartModule resolves the game's OWN runtime
    // PRX loads to real handles (#146/#147) — the guest LoadStartModule's its preloaded Il2Cpp/
    // PS5Util modules, which must succeed for IL2CPP to bootstrap. Mirrors boot_program.cpp.
    set_module_exports(&prog.exports);
    { std::vector<ModuleExportTable> mt;
      for (const auto& me : prog.mod_exports) mt.push_back({ me.path, &me.nids });
      set_module_export_tables(std::move(mt)); }

    register_builtin_hle();
    set_app0_root(dump);                        // guest "/app0" -> the game dump directory
    for (auto& img : prog.imgs)
        if (!map_image(img, &err)) { printf("  [FAIL] map: %s\n", err.c_str()); return 1; }
    if (!install_stubs(prog.slots, prog.stub_base, prog.stub_size, &err)) { printf("  [FAIL] stubs: %s\n", err.c_str()); return 1; }
    install_trap_handler();

    volatile int* p = (volatile int*)mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *p = 0; dispatch_set_progress(p);
    // Fork-safe counter for GC stop-the-world exception deliveries. This is only reached deep in
    // il2cpp_init, *after* the 15-thread GC handshake that used to deadlock — so a non-zero count
    // proves the deadlock fix + the exception-based thread suspension + stack scanning all work.
    volatile int* ex = (volatile int*)mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *ex = 0; set_exc_raise_counter(ex);
    // Fork-safe counter for graphics-lib calls (libSceAgc / libSceVideoOut). A non-zero count proves
    // the boot ran the *entire* runtime — loader -> IL2CPP init -> C# startup -> PS5 services -> and
    // into GPU/display initialization. Regression guard for the whole bring-up pipeline.
    volatile int* gfx = (volatile int*)mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    *gfx = 0; set_gfx_call_counter(gfx);

    printf("  entry=0x%llx  %zu dependent-module init fns  running guest in a child...\n",
           (unsigned long long)prog.entry, prog.init_fns.size());
    pid_t pid = fork();
    if (pid == 0) { run_guest_inits(prog.init_fns); run_entry(prog.imgs[0]); _exit(0); }
    // Wait for the depth milestones, not a fixed wall-clock window. The guest reads the ~GB game
    // dump over WSL's /mnt/c 9p mount, which on a COLD page cache is far slower than warm — the old
    // fixed 200×50 ms = 10 s budget made this test flaky (issue #89): a cold first-of-session run
    // could stall before graphics while a warm run reached gfx in ~1 s. Poll to a generous ceiling
    // but BREAK as soon as the milestones (GC stop-the-world + a graphics-lib call) are met, so a
    // warm run still finishes fast. PROSPER_BOOTTEST_TIMEOUT_MS overrides the ceiling for slow CI.
    long ceil_ms = 60000;
    if (const char* e = getenv("PROSPER_BOOTTEST_TIMEOUT_MS")) { long v = atol(e); if (v > 0) ceil_ms = v; }
    const long tick_ms = 50;
    for (long waited = 0; waited < ceil_ms; waited += tick_ms) {
        int st; if (waitpid(pid, &st, WNOHANG) == pid) break;           // child exited on its own
        if (*ex >= 1 && *gfx >= 1) break;                               // milestones reached — done early
        struct timespec ts{ 0, tick_ms * 1000 * 1000 }; nanosleep(&ts, nullptr);
    }
    int st; if (waitpid(pid, &st, WNOHANG) == 0) { kill(pid, SIGKILL); waitpid(pid, &st, 0); }

    int reached = *p;
    int raises  = *ex;
    int gfxcalls = *gfx;
    printf("  guest reached %d distinct unimplemented system calls; %d GC stop-the-world exception(s); %d graphics-lib call(s)\n",
           reached, raises, gfxcalls);
    // Pass gates on the two DEPTH milestones, which are forward-compatible (they stay true as deeper
    // blockers are fixed):
    // (1) >=1 GC stop-the-world exception proves the boot got *through* the IL2CPP GC thread-
    //     suspension handshake (the deadlock we fixed) and ran the exception-based stop-the-world.
    // (2) >=1 graphics-lib call proves the boot ran the whole runtime into GPU/display init.
    // The distinct-unimplemented-syscall count is now INFORMATIONAL only, not a gate: it legitimately
    // DROPS toward 0 as functions get implemented (the boot then advances to deeper, already-handled
    // calls), so the old `>=1` floor produced false failures independent of real boot depth.
    bool gc_stw   = raises  >= 1;
    bool graphics = gfxcalls >= 1;
    if (gc_stw && graphics) {
        printf("\n== PASS: booted through IL2CPP + GC stop-the-world into graphics init (%d unimpl [informational], %d>=1 raise, %d>=1 gfx) ==\n",
               reached, raises, gfxcalls);
        return 0;
    }
    if (!gc_stw)   printf("\n== FAIL: GC stop-the-world never ran (%d raises) — deadlock/GC regression? ==\n", raises);
    if (!graphics) printf("\n== FAIL: never reached graphics init (%d gfx calls) in %ld ms — boot regressed before GPU/display (or raise PROSPER_BOOTTEST_TIMEOUT_MS) ==\n", gfxcalls, ceil_ms);
    return 2;
}
