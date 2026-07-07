// test_trap_linux — verify the link + map + stub + dispatch chain. Links just the main
// executable (no HLE registered, so every import becomes an unimplemented stub slot),
// then "calls" representative slots and checks each is identified by lib::nid and returns 0.
#include "../src/self/module.hpp"
#include "../src/loader/linker.hpp"
#include "../src/host/exec_image.hpp"
#include "../src/hle/dispatch.hpp"
#include <cstdio>
#include <string>

using namespace prosper;
static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; \
    printf("  [FAIL] %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char** argv) {
    std::string dump = (argc >= 2) ? argv[1] : "../../PPSA24651-app0";
    printf("== test_trap_linux: %s ==\n", dump.c_str());

    Program prog;
    std::string err;
    // No HLE registered -> all imports become stub slots we can trap on.
    if (!link_program({ { dump + "/eboot.bin", 0x400000000ull } }, 0x600000000ull, prog, &err)) {
        printf("  [FAIL] link: %s\n", err.c_str()); return 1;
    }
    CHECK(prog.slots.size() > 500, "expected many stub slots, got %zu", prog.slots.size());
    for (auto& img : prog.imgs) CHECK(map_image(img, &err), "map: %s", err.c_str());
    CHECK(install_stubs(prog.slots, prog.stub_base, prog.stub_size, &err), "install_stubs: %s", err.c_str());
    install_trap_handler();

    for (const char* lib : {"libkernel", "libSceAgc", "libc", "libSceVideoOut"}) {
        long slot = -1;
        for (size_t i = 0; i < prog.slots.size(); i++) if (prog.slots[i].lib == lib) { slot = (long)i; break; }
        CHECK(slot >= 0, "no stub slot for %s", lib);
        if (slot < 0) continue;
        reset_call_log();
        uint64_t ret = invoke_stub((uint64_t)slot);
        CHECK(ret == 0, "%s stub should return 0, got %llu", lib, (unsigned long long)ret);
        CHECK(call_order().size() == 1 && call_order()[0] == (uint32_t)slot,
              "%s: dispatch did not record slot %ld", lib, slot);
        printf("  dispatched %s::%s -> 0\n", lib, prog.slots[slot].nid.c_str());
    }

    printf("\n== %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail;
}
