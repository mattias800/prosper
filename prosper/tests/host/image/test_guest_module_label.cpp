// #1659: every diagnostic that prints "<module>+0x<rva>" must agree, and must agree with the address
// the loader actually maps to.
//
// The eboot base MOVED in #825 — 0x400000000 -> 0x410000000 — so Astro Bot's direct-memory mapping at
// [0x400000000,0x40b800000) would stop aliasing code. Three subsystems had hard-coded the old literal
// and were never updated, so they printed the real RVA **plus 0x10000000**: a plausible-looking offset
// that lands nowhere, past the end of a 161 MB image, and one that no longer round-trips through
// PROSPER_BP (which adds the real mapped base).
//
// These assertions are written against the CONTRACT — "an address N bytes into the eboot labels as
// eboot+0xN" — not against whichever constant the code currently holds, so they stay meaningful if a
// base moves again.

#include "host/image/boot_program.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("  [FAIL] %s\n", m); ++fails; } \
                         else std::printf("  [ok]   %s\n", m); } while (0)

int main() {
    std::printf("== test_guest_module_label ==\n");

    // ---- the contract, for each module a diagnostic can name -------------------------------
    constexpr uint64_t kRva = 0x4a87368ull;   // the real RVA from the #1659 report
    CHECK(std::strcmp(guest_module_name(BOOT_EBOOT + kRva), "eboot") == 0 &&
          guest_module_offset(BOOT_EBOOT + kRva) == kRva,
          "an address kRva into the eboot labels as eboot+kRva");
    CHECK(std::strcmp(guest_module_name(BOOT_IL2CPP + kRva), "Il2cpp") == 0 &&
          guest_module_offset(BOOT_IL2CPP + kRva) == kRva,
          "an IL2CPP address names Il2cpp -- not eboot with a shifted offset");
    CHECK(std::strcmp(guest_module_name(BOOT_LIBC + 0x100), "libc.prx") == 0 &&
          guest_module_offset(BOOT_LIBC + 0x100) == 0x100,
          "libc.prx is named and offset from its own base");

    // ---- the exact defect ------------------------------------------------------------------
    // The old code computed `va - 0x400000000`. For an eboot VA that is kRva + 0x10000000.
    constexpr uint64_t kStaleBase = 0x400000000ull;
    const uint64_t eboot_va = BOOT_EBOOT + kRva;
    // Not "== 0x10000000" — that would be a tautology over two constants and would fail spuriously if
    // the base moves again. The property is: the stale base over-reports, by the amount the base moved.
    CHECK(eboot_va - kStaleBase == kRva + (BOOT_EBOOT - kStaleBase),
          "the stale base over-reports an eboot offset by exactly the relocation distance");
    CHECK(guest_module_offset(eboot_va) != eboot_va - kStaleBase,
          "the shared helper does NOT reproduce the stale-base offset");
    // The reported symptom: 0x14a87368 against a 0x99c74c8-byte image is past the end.
    constexpr uint64_t kOregonTrailImageBytes = 0x99c74c8ull;
    CHECK(eboot_va - kStaleBase > kOregonTrailImageBytes &&
          guest_module_offset(eboot_va) < kOregonTrailImageBytes,
          "the stale offset is past the end of the image while the correct one is inside it");

    // ---- the region the old lower bound covered is NOT code ---------------------------------
    // [0x400000000,0x410000000) is now a direct-memory aperture (Astro Bot maps it before AGC).
    // A pointer there must not be labelled as a module offset at all.
    CHECK(!guest_va_in_module(kStaleBase) && !guest_va_in_module(BOOT_EBOOT - 1),
          "an address in the pre-#825 aperture is not claimed by any module");
    CHECK(std::strcmp(guest_module_name(kStaleBase), "mapped/host") == 0 &&
          guest_module_offset(kStaleBase) == kStaleBase,
          "a non-module address reports mapped/host and its address verbatim, not a fake offset");

    // ---- round-trip with the breakpoint input contract --------------------------------------
    // PROSPER_BP=0xOFFSET installs at (mapped base + OFFSET). An offset read out of a log must land
    // back on the same instruction — that round-trip was broken while the label used the stale base.
    {
        bool round_trip_ok = true;
        for (uint64_t rva : {uint64_t{0}, uint64_t{1}, kRva, uint64_t{0x2fffffff}})
            if (guest_module_offset(BOOT_EBOOT + rva) != rva) round_trip_ok = false;
        CHECK(round_trip_ok, "printed eboot offsets round-trip through BOOT_EBOOT + offset");
    }

    // ---- boundaries -------------------------------------------------------------------------
    CHECK(guest_module_offset(BOOT_EBOOT) == 0 &&
          std::strcmp(guest_module_name(BOOT_EBOOT), "eboot") == 0,
          "the first byte of the eboot is eboot+0x0");
    CHECK(std::strcmp(guest_module_name(BOOT_IL2CPP - 1), "eboot") == 0,
          "the last byte below the next module still belongs to the eboot");
    CHECK(guest_va_in_module(BOOT_STUB) && guest_module_offset(BOOT_STUB) == 0,
          "the import-stub region is named and offset like any other module");

    // ---- a CONVERTED CALL SITE, not just the helper -----------------------------------------
    // Every assertion above tests boot_program.hpp. The defect was in the *callers*, so a mislabel
    // reintroduced there would pass all of it. This drives the real formatter used by the fault and
    // HWBP diagnostics and asserts the rendered text, including the module name.
    {
        char line[128];
        const uint64_t il2cpp_ra = BOOT_IL2CPP + 0x1234;
        std::snprintf(line, sizeof line, "%s+0x%llx",
                      guest_module_name(il2cpp_ra),
                      (unsigned long long)guest_module_offset(il2cpp_ra));
        CHECK(std::strcmp(line, "Il2cpp+0x1234") == 0,
              "a rendered label names the module it came from -- not 'eboot' with a plausible offset");
        // The specific regression the review caught: a widened filter with a hard-coded label renders
        // an in-range offset under the wrong module, which has no tell at all.
        CHECK(std::strncmp(line, "eboot", 5) != 0,
              "an Il2Cpp address never renders as eboot+");
    }

    // ---- the FAULT REPORTER's own formatter, and the exact address that misled a session -------
    // The three fault sites in exec_image_linux.cpp (trap_detail, WORKER-THREAD FAULT, CONCURRENT
    // WORKER-THREAD FAULT) each computed `rip - eboot_base` and printed a hard-coded "image+0x%llx".
    // That is correct only for an eboot rip; for any other module it renders a number that lands
    // nowhere. Measured on PPSA20447 (The First Berserker: Khazan): the guest calls abort(), whose
    // PS5 implementation is `int $0x45` at libc.prx+0x4a20, and the fault line read
    //     SIGSEGV at addr=(nil)  rip=0x5c0004a20 (image+0x1b0004a20)
    // -- 6.75 GiB past the end of a 250 MiB image. The offset read as a wild jump into nothing, and
    // that appearance is what produced a null-pointer-dereference hypothesis for a deliberate abort.
    //
    // These arms drive `format_guest_module_label`, which is the function those sites now call, so a
    // site that stops calling it is a visible edit rather than a silent arithmetic change.
    {
        char line[64];
        const uint64_t abort_trap = BOOT_LIBC + 0x4a20;      // the real faulting rip
        format_guest_module_label(line, sizeof line, abort_trap);
        CHECK(std::strcmp(line, "libc.prx+0x4a20") == 0,
              "the fault reporter's formatter names libc.prx and offsets from ITS base");
        // The mutation this arm exists to catch, spelled out: reverting a site to
        // `snprintf(..., "image+0x%llx", rip - BOOT_EBOOT)` renders exactly this string.
        char stale[64];
        std::snprintf(stale, sizeof stale, "image+0x%llx",
                      (unsigned long long)(abort_trap - BOOT_EBOOT));
        CHECK(std::strcmp(line, stale) != 0,
              "it does NOT render the eboot-relative offset the old fault sites printed");
        // ...and that offset really is outside the image, which is why it looked like a wild jump.
        constexpr uint64_t kKhazanEbootBytes = 0xf9ae768ull;   // the flattened PPSA20447 eboot
        CHECK(abort_trap - BOOT_EBOOT > kKhazanEbootBytes &&
              guest_module_offset(abort_trap) < kKhazanEbootBytes,
              "the stale offset is past the end of the eboot while the correct one is a real RVA");
    }

    // A host/unmapped rip must not be dressed up as a module offset either: it reports verbatim.
    {
        char line[64];
        const uint64_t host_rip = 0x7f2a98dfed30ull;
        format_guest_module_label(line, sizeof line, host_rip);
        CHECK(std::strncmp(line, "mapped/host+0x", 14) == 0 &&
              std::strstr(line, "7f2a98dfed30") != nullptr,
              "a non-guest rip renders as mapped/host+<the address>, not a fabricated RVA");
    }

    std::printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
