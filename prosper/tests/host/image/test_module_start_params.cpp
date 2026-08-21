// test_module_start_params.cpp — which modules are started with a real SCE module-param descriptor.
//
// The defect this pins: prosper starts every preloaded PRX through the PS5 module-entry ABI
// `module_start(size_t argc, const void *argp)`, defaulting to (0, NULL). A native Unity plugin that
// performs a version handshake stores both arguments at module_start and validates them later, so
// (0, NULL) makes it take its mismatch branch and then dereference the NULL argp.
//
// Until PGA TOUR 2K25 (PPSA17952) the handshake had only ever been seen in a separate PSN.prx, so
// only those ranges were registered. That title links the Unity PSN package's native half straight
// into Media/Modules/Il2cppUserAssemblies.prx, whose module_start then does the handshake — and the
// title died at 1.2 s on every boot with SIGSEGV addr=0x4 reading argp->version off NULL, after
// printing "PSN is an old version that cannot be used by the current player runtime".
//
// Both directions are asserted and the EXCLUSION arm is the load-bearing one: a predicate that
// answered true for everything would satisfy the inclusion arm perfectly while silently changing
// the entry ABI of every module prosper links.

#include <cstdio>
#include <cstdint>

#include "host/image/boot_program.hpp"
#include "host/image/module_start_params.hpp"

using prosper::module_start_param_ranges;
using prosper::module_start_wants_param_descriptor;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

// Every module prosper links has DT_INIT at image+0x10 and an empty DT_INIT_ARRAY (measured with
// PROSPER_INITLOG=1 across all ten of PPSA17952's modules), so a module's single init function is
// its base + 0x10. Query that address rather than the bare base: it is what run_guest_inits sees.
static constexpr uint64_t kInit = 0x10;

int main() {
    printf("module_start param ranges\n");

    // --- INCLUSION: modules whose module_start performs the handshake. ---
    CHECK(module_start_wants_param_descriptor(prosper::BOOT_IL2CPP + kInit),
          "Il2cppUserAssemblies.prx module_start gets the descriptor (PPSA17952)");
    CHECK(module_start_wants_param_descriptor(prosper::BOOT_PSN + kInit),
          "PSN.prx module_start gets the descriptor");
    CHECK(module_start_wants_param_descriptor(prosper::BOOT_SAVEDATA + kInit),
          "SaveData.prx module_start gets the descriptor");
    CHECK(module_start_wants_param_descriptor(prosper::BOOT_PSNCORE + kInit),
          "PSNCore.prx module_start gets the descriptor (PPSA02664)");
    CHECK(module_start_wants_param_descriptor(prosper::BOOT_PSNCOMMON + kInit),
          "PSNCommon.prx module_start gets the descriptor (PPSA02664)");

    // --- EXCLUSION: everything else keeps the (0, NULL) entry ABI it has always had. ---
    // Without these a blanket-true predicate passes the whole inclusion block above.
    CHECK(!module_start_wants_param_descriptor(prosper::BOOT_EBOOT + kInit),
          "the main executable is NOT given the descriptor");
    CHECK(!module_start_wants_param_descriptor(prosper::BOOT_LIBC + kInit),
          "libc.prx is NOT given the descriptor");
    CHECK(!module_start_wants_param_descriptor(prosper::BOOT_PS5UTIL + kInit),
          "PS5Util.prx is NOT given the descriptor");
    CHECK(!module_start_wants_param_descriptor(prosper::BOOT_NPCPPWEBAPI + kInit),
          "libSceNpCppWebApi.prx is NOT given the descriptor");
    CHECK(!module_start_wants_param_descriptor(prosper::BOOT_COMMONDIALOG + kInit),
          "CommonDialog.prx is NOT given the descriptor");
    CHECK(!module_start_wants_param_descriptor(prosper::BOOT_PLUGIN_AUTO_BASE + kInit),
          "an auto-discovered Unity plugin is NOT given the descriptor");

    // --- STRUCTURE: a range whose end is below its start silently matches nothing, which would
    // make an inclusion arm above fail for a reason that reads as policy rather than as a typo. ---
    const auto ranges = module_start_param_ranges();
    CHECK(!ranges.empty(), "the range list is non-empty");
    bool ordered = true;
    for (const auto& r : ranges) if (r.first >= r.second) ordered = false;
    CHECK(ordered, "every range is a non-empty [begin, end)");

    // The Il2cpp range must stop at the next module base rather than swallowing its neighbours:
    // boot_program.hpp's own address classifier bounds the Il2cpp image at BOOT_PSNCORE.
    CHECK(!module_start_wants_param_descriptor(prosper::BOOT_IL2CPP - 1),
          "the Il2cpp range does not extend below BOOT_IL2CPP");

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
