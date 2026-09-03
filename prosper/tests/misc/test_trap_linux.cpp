// test_trap_linux — verify the link + map + stub + dispatch chain. Links just the main
// executable (no HLE registered, so every import becomes an unimplemented stub slot),
// then "calls" representative slots and checks each is identified by lib::nid and returns 0.
#include "self/module.hpp"
#include "loader/linker.hpp"
#include "host/image/exec_image.hpp"
#include "hle/dispatch/dispatch.hpp"
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
    // NOT `> 500` (#2997). That threshold was PPSA24651's slot count, and this test is not about how
    // many imports a title has -- it is about the link -> map -> stub -> dispatch chain, which is
    // title-agnostic. Measured across all 55 local dumps: slot counts run from 179 (PPSA16901) to
    // over 500, so the threshold FAILED four titles on a property none of them was doing anything
    // wrong about -- including PPSA15552 (Dead Cells, rung 6) and PPSA03839 (Tactics Ogre, rung 3).
    // A red case is worse than a skip: it forces a bring-up agent to prove a failure is pre-existing
    // before trusting anything else in the run.
    //
    // What replaces it is NOT `!slots.empty()`. Review showed that would be strictly SUBSUMED by the
    // four `slot >= 0` probes below -- no state fails it while those pass -- so it would read as an
    // assertion while carrying no weight. These are the linker's own accounting invariants instead,
    // and they are title-agnostic in the way the count only pretended to be:
    //   * the module imports something at all;
    //   * every import is accounted for as either cross-module-resolved or stubbed. This cannot fail
    //     on any input: pass 2's body increments total_imports and then exactly one of the other two
    //     (linker.cpp:115 / :119 / :131), with no third path. It is an EDIT guard, not a detector --
    //     it fires if a future filter or early-out counts an import without accounting for it.
    //   * every stubbed import got its OWN slot. This is the one that carries weight: deduplicating
    //     on the wrong key (lib instead of NID) leaves the accounting identity intact -- 612 == 0 +
    //     612 on PPSA24651 -- while collapsing 612 slots to 35, which is what a slot-count floor
    //     could not tell apart from a title that simply imports less. It is an equality only because
    //     no eboot imports the same NID twice: MEASURED at 0 duplicates across all 55 dumps, not
    //     derived (module.cpp:221 does not dedupe imports, and linker.cpp:122 keys the slot table on
    //     the NID alone).
    CHECK(prog.total_imports > 0, "link produced no imports at all");
    CHECK(prog.total_imports == prog.resolved_cross_module + prog.stubbed,
          "import accounting does not close: %zu total != %zu cross-module + %zu stubbed",
          prog.total_imports, prog.resolved_cross_module, prog.stubbed);
    CHECK(prog.stubbed == prog.slots.size(),
          "%zu imports stubbed but %zu slots emitted", prog.stubbed, prog.slots.size());
    for (auto& img : prog.imgs) CHECK(map_image(img, &err), "map: %s", err.c_str());
    CHECK(install_stubs(prog.slots, prog.stub_base, prog.stub_size, &err), "install_stubs: %s", err.c_str());
    install_trap_handler();

    // These four are the real coverage, and the four names are MEASURED rather than assumed: all 55
    // local dumps import all four and dispatch all four correctly (#2997). If a future title does
    // not import one of them, this loop fails with "no stub slot for <lib>" -- which names the cause
    // directly, so it stays a fixed list rather than becoming adaptive on a case that has never
    // occurred.
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
