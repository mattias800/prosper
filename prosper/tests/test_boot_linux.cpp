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
    for (int i = 0; i < 200; i++) { int st; if (waitpid(pid, &st, WNOHANG) == pid) break; struct timespec ts{0, 50 * 1000 * 1000}; nanosleep(&ts, nullptr); }
    int st; if (waitpid(pid, &st, WNOHANG) == 0) { kill(pid, SIGKILL); waitpid(pid, &st, 0); }

    int reached = *p;
    int raises  = *ex;
    int gfxcalls = *gfx;
    printf("  guest reached %d distinct unimplemented system calls; %d GC stop-the-world exception(s); %d graphics-lib call(s)\n",
           reached, raises, gfxcalls);
    // (1) unimpl count *drops* as we implement more functions (boot then advances to new, deeper
    // calls). >=3 robustly proves the pipeline: link -> map -> stubs -> crt -> heap -> vmem -> game.
    const int THRESHOLD = 3;
    // (2) >=1 exception raise proves the boot got *through* the IL2CPP GC thread-suspension
    // handshake (the deadlock we fixed) and ran the exception-based stop-the-world + stack scan.
    // (3) >=1 graphics-lib call proves the boot ran the whole runtime into GPU/display init — the
    // deepest reproducible milestone. Forward-compatible: stays true as deeper blockers are fixed.
    bool pipeline = reached >= THRESHOLD;
    bool gc_stw   = raises  >= 1;
    bool graphics = gfxcalls >= 1;
    if (pipeline && gc_stw && graphics) {
        printf("\n== PASS: booted through IL2CPP + GC stop-the-world into graphics init (%d>=%d unimpl, %d>=1 raise, %d>=1 gfx) ==\n",
               reached, THRESHOLD, raises, gfxcalls);
        return 0;
    }
    if (!pipeline) printf("\n== FAIL: stalled early (%d < %d unimpl) ==\n", reached, THRESHOLD);
    if (!gc_stw)   printf("\n== FAIL: GC stop-the-world never ran (%d raises) — deadlock/GC regression? ==\n", raises);
    if (!graphics) printf("\n== FAIL: never reached graphics init (%d gfx calls) — boot regressed before GPU/display ==\n", gfxcalls);
    return 2;
}
