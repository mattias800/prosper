// test_trap_linux — verify the link + map + stub + dispatch chain. Links just the main
// executable (no HLE registered, so every import becomes an unimplemented stub slot),
// then "calls" representative slots and checks each is identified by lib::nid and returns 0.
#include "self/module.hpp"
#include "loader/linker.hpp"
#include "host/image/exec_image.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/import_data_seed.hpp"
#include "hle/dispatch/nid.hpp"
#include <cstdint>
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
    if (!link_program({ { dump + "/eboot.bin", 0x400000000ull } }, 0x600000000ull, 0x7c0000000ull,
                      prog, &err)) {
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
    //   * every import is accounted for as cross-module-resolved, stubbed, or bound to the DATA
    //     aperture. This cannot fail on any input: pass 2's body increments total_imports and then
    //     exactly one of the other three, with no fourth path. It is an EDIT guard, not a detector --
    //     it fires if a future filter or early-out counts an import without accounting for it. The
    //     third bucket arrived with #3529; before it, a DATA import was counted in `stubbed` and
    //     bound to an executable trampoline.
    //   * every stubbed import got its OWN slot. This is the one that carries weight: deduplicating
    //     on the wrong key (lib instead of NID) leaves the accounting identity intact -- 612 == 0 +
    //     612 on PPSA24651 -- while collapsing 612 slots to 35, which is what a slot-count floor
    //     could not tell apart from a title that simply imports less. It is an equality only because
    //     no eboot imports the same NID twice: MEASURED at 0 duplicates across all 55 dumps, not
    //     derived (module.cpp does not dedupe imports, and linker.cpp keys the slot table on the
    //     NID alone).
    CHECK(prog.total_imports > 0, "link produced no imports at all");
    CHECK(prog.total_imports == prog.resolved_cross_module + prog.stubbed + prog.bound_data,
          "import accounting does not close: %zu total != %zu cross-module + %zu stubbed + %zu data",
          prog.total_imports, prog.resolved_cross_module, prog.stubbed, prog.bound_data);
    CHECK(prog.stubbed == prog.slots.size(),
          "%zu imports stubbed but %zu slots emitted", prog.stubbed, prog.slots.size());
    CHECK(prog.bound_data == prog.data_slots.size(),
          "%zu data imports bound but %zu data slots emitted", prog.bound_data,
          prog.data_slots.size());

    // #3529. A DATA import (ELF STT_OBJECT) names a VARIABLE, so it must NOT be bound into the
    // executable stub aperture: the guest reads it as a value and may store through it, and a store
    // into that aperture rewrites a trampoline. Two halves, and the first is what makes the second
    // mean anything:
    //
    //   1. The eboot really imports at least one STT_OBJECT symbol. Without this the case below
    //      passes vacuously on any dump -- the exact failure mode the charter's positive-control
    //      rule names. PPSA24651's eboot declares four STT_OBJECT imports, one of which is
    //      `__stack_chk_guard`; the NID-keyed case further down names that one explicitly.
    //   2. Every data-slot address lies in the data aperture and outside the whole stub aperture.
    //      Checked against the aperture, not against the slots emitted so far, because the stub
    //      table can still grow at runtime (append_stubs, #639) and an address that is merely past
    //      today's last stub is not safe.
    //
    // The mutation that fails this: bind data imports at `stub_base` again (delete the STT_OBJECT
    // branch in linker.cpp pass 2) -- then data_slots is empty and (1) fails; keep the branch but
    // spell the address `out.stub_base + slot * out.stub_size` and (2) fails.
    CHECK(!prog.data_slots.empty(),
          "this eboot imports no STT_OBJECT symbol, so the data-binding case below proves nothing");
    for (size_t i = 0; i < prog.data_slots.size(); i++) {
        const uint64_t a = prog.data_base + i * prog.data_stride;
        CHECK(a >= prog.data_base && a < prog.data_base + kImportDataApertureBytes,
              "data slot %zu at 0x%llx is outside the import-data aperture", i,
              (unsigned long long)a);
        CHECK(a < prog.stub_base || a >= prog.stub_base + kStubApertureBytes,
              "data slot %zu (%s::%s) at 0x%llx is inside the executable stub aperture", i,
              prog.data_slots[i].lib.c_str(), prog.data_slots[i].nid.c_str(),
              (unsigned long long)a);
    }
    // ...and the guest image really points at those addresses. The loop above checks the slot table;
    // this checks what apply_relocations actually baked into an import_addr entry, which is what the
    // guest dereferences. Find the eboot's own STT_OBJECT imports and read their bound addresses.
    {
        size_t checked = 0;
        for (const auto& imp : prog.mods[0]->imports) {
            if (imp.elf_type != STT_OBJECT) continue;
            auto it = prog.imgs[0].import_addr.find(imp.sym_index);
            CHECK(it != prog.imgs[0].import_addr.end(), "data import %s was never bound",
                  imp.nid.c_str());
            CHECK(it->second < prog.stub_base || it->second >= prog.stub_base + kStubApertureBytes,
                  "data import %s::%s is bound to 0x%llx, inside the executable stub aperture",
                  imp.lib_name.c_str(), imp.nid.c_str(), (unsigned long long)it->second);
            checked++;
        }
        CHECK(checked > 0, "no STT_OBJECT import found in the eboot's own import table");
    }

    for (auto& img : prog.imgs) CHECK(map_image(img, &err), "map: %s", err.c_str());
    CHECK(install_stubs(prog.slots, prog.stub_base, prog.stub_size, &err), "install_stubs: %s", err.c_str());
    CHECK(install_import_data(prog.data_slots, prog.data_base, prog.data_stride, &err),
          "install_import_data: %s", err.c_str());
    // The aperture is real memory: readable, WRITABLE, and holding what install_import_data put
    // there. The writability half is the one #3529 is really about -- before it, this same store
    // landed on prosper's executable stub bytes. Non-executability is a property of the
    // PROT_READ|PROT_WRITE mapping and is deliberately not asserted by jumping into it here, which
    // would abort the test binary rather than report.
    //
    // Slots are addressed BY NID, never by index: which import gets slot 0 depends on symbol order
    // in the dump, so an index-keyed assertion would silently start measuring a different variable
    // on a different title.
    {
        const std::string guard_nid = nid_hash("__stack_chk_guard");
        size_t guard_slot = SIZE_MAX, plain_slot = SIZE_MAX;
        for (size_t i = 0; i < prog.data_slots.size(); i++) {
            if (prog.data_slots[i].nid == guard_nid) guard_slot = i;
            else if (plain_slot == SIZE_MAX)         plain_slot = i;
        }
        // Every dump in the local corpus imports __stack_chk_guard (60 of 60), so its absence here
        // means the symbol type or the NID hash changed, not that this title is unusual.
        CHECK(guard_slot != SIZE_MAX, "__stack_chk_guard is not among this eboot's data imports");
        if (guard_slot != SIZE_MAX) {
            volatile uint64_t* g = (volatile uint64_t*)(uintptr_t)import_data_addr(guard_slot);
            // The seeded value, not merely "non-zero": a slot that happened to hold anything else
            // would pass a non-zero test while meaning the seed never ran.
            CHECK(g && *g == kStackGuardValue,
                  "__stack_chk_guard slot holds 0x%llx, expected the seeded 0x%llx",
                  (unsigned long long)(g ? *g : 0), (unsigned long long)kStackGuardValue);
        }
        // An UNSEEDED slot must be zero. This is the arm that keeps the one above honest: if
        // install_import_data wrote the canary into every slot (or the mapping were not zero-filled
        // at all), this fails while the seed assertion still passes.
        if (plain_slot != SIZE_MAX) {
            volatile uint64_t* q = (volatile uint64_t*)(uintptr_t)import_data_addr(plain_slot);
            CHECK(q != nullptr, "import_data_addr(%zu) is 0 after install_import_data", plain_slot);
            if (q) {
                CHECK(*q == 0, "unseeded import-data slot %zu is not zero (0x%llx)", plain_slot,
                      (unsigned long long)*q);
                *q = 0xfeedfacecafebeefull;
                CHECK(*q == 0xfeedfacecafebeefull,
                      "import-data slot %zu did not retain a guest store", plain_slot);
                *q = 0;
            }
        }
    }
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
