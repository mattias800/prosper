// test_loader_synth_link — link_program's structural contract, on synthesized modules (#2567).
//
// WHAT THIS CLOSES
// ----------------
// `plugin_autolink` already covers `discover_extra_plugin_modules` and the collision *predicate* with
// no dump. Its remaining 14 assertions — the ones that put the predicate through `link_program` and
// check the export table, the skip record and the #1635 alias record — sat behind `if (argc >= 2)`
// and a real `Media/Plugins/PSN.prx`, so they never ran in CI and never ran for any agent without
// that title installed. This case registers unconditionally and runs everywhere.
//
// It is not a re-skin of the dump-backed block: the dump version can only link the SAME file twice,
// so its `winner_path == loser_path` and `owner_path == path` assertions are satisfied trivially.
// Here the colliding pair is two DIFFERENT modules — the shape Evergate (PPSA01885) actually ships
// with libfmod.prx and libfmodL.prx — so those records have to name the right module, not merely a
// non-empty one. The dump-backed case stays as it is: it exercises real Sony-built modules and is
// not redundant with this.
//
// WHY THE NEGATIVE AND HAND-BUILT ARMS ARE NOT OPTIONAL
// ----------------------------------------------------
// A fixture the loader accepts proves that the generator and the loader agree. Two things make the
// claim stronger than that, and both live in this file or its sibling:
//   * `test_loader_synth_reject.cpp` holds the corrupted variants the loader must refuse, each with a
//     control differing in exactly one field;
//   * arm 6 below links a module built BY HAND (`handmade_prx.h`) whose file offsets and vaddrs
//     diverge — a geometry `synth_prx.h` cannot express. A positive control drawn from the same
//     source as the null it validates tests the discriminator, never the domain.
#include "hle/dispatch/nid.hpp"
#include "loader/linker.hpp"
#include "self/module.hpp"
#include "fixtures/handmade_prx.h"
#include "fixtures/synth_prx.h"
#include "fixtures/test_scratch.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace prosper;
using prosper_test::SynthModuleSpec;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

// Module bases, far enough apart that an address can be attributed to exactly one image. Typed
// constants rather than an enum: these are added to the fixtures' own layout enums, and enum + enum
// arithmetic is deprecated in C++20.
static constexpr uint64_t kBase0    = 0x400000000ull;
static constexpr uint64_t kBase1    = 0x500000000ull;
static constexpr uint64_t kBase2    = 0x600000000ull;
static constexpr uint64_t kStubBase = 0x700000000ull;

// The 8 bytes at `rel_va` inside linked module `mod_index`, or a sentinel the tests cannot mistake
// for a real address when the read is out of range.
static uint64_t image_u64(const Program& p, size_t mod_index, uint64_t rel_va) {
    if (mod_index >= p.imgs.size()) return 0xdeadbeefdeadbeefull;
    const uint8_t* q = p.imgs[mod_index].at(p.imgs[mod_index].base + rel_va);
    if (!q || q + 8 > p.imgs[mod_index].mem.data() + p.imgs[mod_index].mem.size())
        return 0xdeadbeefdeadbeefull;
    uint64_t v; memcpy(&v, q, 8); return v;
}

static uint64_t export_of(const Program& p, const std::string& nid) {
    const auto it = p.exports.find(nid);
    return it == p.exports.end() ? 0 : it->second;
}

int main() {
    printf("== test_loader_synth_link ==\n");

    // Distinct synthetic symbol names, hashed with prosper's own NID function so the fixtures carry
    // real 11-character Sony NIDs rather than invented strings.
    const std::string kShared   = nid_hash("prosperSynthSharedExport");
    const std::string kOnlyA    = nid_hash("prosperSynthReleaseOnlyExport");
    const std::string kOnlyB    = nid_hash("prosperSynthLoggingOnlyExport");
    const std::string kMainOnly = nid_hash("prosperSynthMainOnlyExport");
    const std::string kMissing  = nid_hash("prosperSynthUnsatisfiedImport");
    const std::string kAltB     = nid_hash("prosperSynthLoggingAlternateExport");
    const std::string kHandA    = nid_hash("prosperHandmadeExportA");
    const std::string kHandB    = nid_hash("prosperHandmadeExportB");
    const std::string kHandImp  = nid_hash("prosperHandmadeImport");

    const std::string dir = prosper_test::test_scratch_dir().string();
    auto emit = [&](const char* name, const SynthModuleSpec& spec) {
        const std::string path = dir + "/" + name;
        std::string err;
        if (!prosper_test::write_synth_prx(path, spec, &err)) {
            printf("  [FAIL] cannot write fixture %s: %s\n", name, err.c_str());
            fails++;
        }
        return path;
    };

    SynthModuleSpec main_spec;    main_spec.exports = { kMainOnly };  main_spec.imports = { kShared };
    SynthModuleSpec release_spec; release_spec.exports = { kShared, kOnlyA };
    SynthModuleSpec logging_spec; logging_spec.exports = { kShared, kOnlyB };
                                  logging_spec.imports = { kOnlyA, kMissing };
    // The mutation arm: identical to `logging_spec` except that its first export is a DIFFERENT NID.
    SynthModuleSpec alt_spec = logging_spec; alt_spec.exports = { kAltB, kOnlyB };
    // A module whose only "export" is defined but valueless. linker.cpp has the export predicate
    // written twice (is_exported_symbol, and the inline test where out.exports is built); if those
    // ever drift, this module is accepted by one and rejected by the other.
    // A DUPLICATE BUILD: every export it has, an earlier module already provides. This is the shape
    // the collision guard exists for -- Evergate ships libfmod.prx beside libfmodL.prx with the same
    // export set, and linking both would run two init_arrays and make dlsym answer differently per
    // handle. Distinct from the `logging` module above, which shares ONE export and provides one of
    // its own; that is two different libraries, not two builds of one (#3497).
    SynthModuleSpec subsumed_spec; subsumed_spec.exports = { kShared, kOnlyA };
    // An import nothing satisfies, so "a skipped module contributes no stub slot" is a claim
    // with something to be wrong about. Without it that assertion held vacuously.
                                   subsumed_spec.imports = { kMissing };

    SynthModuleSpec valueless_spec; valueless_spec.exports = { kShared };
                                    valueless_spec.zero_value_exports = true;

    const std::string main_path      = emit("synth_main.prx",      main_spec);
    const std::string release_path   = emit("synth_release.prx",   release_spec);
    const std::string logging_path   = emit("synth_logging.prx",   logging_spec);
    const std::string alt_path       = emit("synth_alt.prx",       alt_spec);
    const std::string subsumed_path  = emit("synth_subsumed.prx",  subsumed_spec);
    const std::string valueless_path = emit("synth_valueless.prx", valueless_spec);
    if (fails) { printf("FAILED (%d) -- fixtures were not written\n", fails); return 1; }

    const uint64_t kExp0 = prosper_test::synth_export_va(0);
    const uint64_t kExp1 = prosper_test::synth_export_va(1);
    const uint64_t kGot0 = prosper_test::synth_got_va(0);
    const uint64_t kGot1 = prosper_test::synth_got_va(1);

    // ---- Arm 1: control — two different modules export the same NID, neither flagged ------------
    // This is the real-world shape (#1635): both are linked, first-wins keeps the earlier definition,
    // and the discarded one must be RECORDED rather than dropped silently.
    {
        const std::vector<LinkInput> inputs = {
            { main_path, kBase0 }, { release_path, kBase1 }, { logging_path, kBase2 },
        };
        Program p; std::string err;
        const bool ok = link_program(inputs, kStubBase, p, &err);
        CHECK(ok, "control: link_program succeeds on three synthetic modules");
        if (!ok) { printf("  link error: %s\n", err.c_str()); printf("FAILED\n"); return 1; }

        CHECK(p.mods.size() == 3, "control: three modules linked");
        CHECK(p.skipped_modules.empty(), "control: an unflagged duplicate is still linked");

        CHECK(export_of(p, kMainOnly) == kBase0 + kExp0, "export table: the main module's own export");
        CHECK(export_of(p, kShared) == kBase1 + kExp0,
              "export table: first definition wins for the shared NID");
        CHECK(export_of(p, kOnlyA) == kBase1 + kExp1, "export table: the first module's private export");
        CHECK(export_of(p, kOnlyB) == kBase2 + kExp1, "export table: the second module's private export");

        CHECK(p.mod_exports.size() == 3 && p.mod_exports[1].path == release_path &&
              p.mod_exports[2].path == logging_path,
              "per-module export tables are parallel to the linked modules");

        CHECK(p.aliased_exports.size() == 1,
              "#1635: exactly the one duplicated NID is recorded as an alias");
        if (p.aliased_exports.size() == 1) {
            const auto& a = p.aliased_exports.front();
            CHECK(a.nid == kShared, "#1635: the alias record names the colliding NID");
            // The dump-backed arm links one file twice, so winner_path == loser_path there whatever
            // the loader does. Two distinct modules make this assertion able to fail.
            CHECK(a.winner_path == release_path, "#1635: the alias names the module that won");
            CHECK(a.loser_path == logging_path, "#1635: the alias names the module that lost");
            CHECK(a.winner != a.loser, "#1635: the discarded definition is a different address");
            CHECK(a.winner >= kBase1 && a.winner < kBase1 + prosper_test::kSynthImageSize,
                  "#1635: first-wins -- the winner lies in the earlier module's image");
            CHECK(a.loser >= kBase2 && a.loser < kBase2 + prosper_test::kSynthImageSize,
                  "#1635: the loser lies in the later module's image");
        }

        // The alias's consequence, which the dump-backed arm cannot observe: an importer of the
        // aliased NID binds to the WINNER. That is what makes a silent alias a hazard at all.
        CHECK(image_u64(p, 0, kGot0) == kBase1 + kExp0,
              "an import of the aliased NID binds to the winning definition");
        // Ordinary cross-module resolution: module 2 imports module 1's private export.
        CHECK(image_u64(p, 2, kGot0) == kBase1 + kExp1,
              "a cross-module import is bound to the exporting module's address");
        // ...and an import nothing exports falls to a stub slot.
        CHECK(p.slots.size() == 1 && p.slots[0].nid == kMissing,
              "the one unsatisfied import produced exactly one stub slot");
        CHECK(image_u64(p, 2, kGot1) == kStubBase,
              "the unsatisfied import's GOT entry addresses its stub slot");
        CHECK(p.resolved_cross_module == 2 && p.stubbed == 1 && p.total_imports == 3,
              "import census: two resolved cross-module, one stubbed");
    }

    // ---- Arm 0: the threshold itself -------------------------------------------------------------
    // The link fixtures below are 100% and 50% subsumed, so ANY threshold in (0.5, 1.0] leaves them
    // both green -- they exercise the two sides but pin no boundary. Raised in review of #3508.
    // These do, directly, and they are the only thing that would notice 0.90 being edited.
    {
        // A helper rather than a braced literal: `CHECK(ExportSubsumption{100, 90}...)` splits on the
        // comma at preprocessor level and is read as three macro arguments. Function-call parens
        // protect it.
        auto sub = [](size_t exported, size_t claimed) {
            ExportSubsumption s; s.exported = exported; s.already_claimed = claimed;
            return s.subsumed();
        };
        CHECK(sub(100, 90), "threshold: exactly 90% counts as subsumed");
        CHECK(!sub(100, 89), "threshold: 89% does not");
        // The corpus this was set from: duplicate builds at 99.7-100%, distinct libraries at
        // 0.4-19.2%. Both ends must land on the correct side, with the real measured numbers.
        CHECK(sub(357, 357),   "corpus: Evergate libfmodstudioL (100.0%) is subsumed");
        CHECK(sub(1093, 1090), "corpus: Evergate libfmodL (99.7%) is subsumed");
        CHECK(!sub(1199, 230), "corpus: Darksiders steam_api_ps5 (19.2%) is NOT");
        CHECK(!sub(527, 2),    "corpus: BlazBlue cri_lips_unity (0.4%) is NOT");
        // A module with no exports cannot be subsumed by anything -- guards a divide-by-zero and the
        // reading that "0 of 0 claimed" is 100%.
        CHECK(!sub(0, 0), "an exportless module is never subsumed");
    }

    // ---- Arm 2a: the guard fires on a DUPLICATE BUILD -------------------------------------------
    // Every export of `subsumed` is already provided, so linking it would duplicate a library
    // wholesale. This is the case the flag was written for and it must keep being skipped.
    {
        std::vector<LinkInput> inputs = {
            { main_path, kBase0 }, { release_path, kBase1 }, { subsumed_path, kBase2 },
        };
        inputs[2].skip_on_export_collision = true;
        Program p; std::string err;
        const bool ok = link_program(inputs, kStubBase, p, &err);
        CHECK(ok, "duplicate build: link_program succeeds");
        CHECK(p.skipped_modules.size() == 1, "duplicate build: the subsumed module is skipped");
        CHECK(p.mods.size() == 2, "duplicate build: only the two other modules are linked");
        if (p.skipped_modules.size() == 1) {
            const auto& sk = p.skipped_modules[0];
            CHECK(sk.path == subsumed_path, "duplicate build: the skip record names the dropped module");
            CHECK(sk.owner_path == release_path,
                  "duplicate build: the skip record names the module that already owns the NID");
            CHECK(sk.nid == kShared, "duplicate build: the skip record carries the exact colliding NID");
        }
        CHECK(p.aliased_exports.empty(), "duplicate build: a skipped module contributes no aliases");
        // `subsumed_spec` imports kMissing, which nothing exports. A LINKED module would produce a
        // stub slot for it; a skipped one contributes nothing at all. That is the difference this
        // asserts. (The first version of this arm checked kOnlyA's address instead, which a fully
        // subsumed module can never win either way -- toothless. Re-review of #3508.)
        CHECK(p.slots.empty(),
              "duplicate build: a skipped module's unsatisfied import creates no stub slot");
    }

    // ---- Arm 2b: the guard does NOT fire on two DIFFERENT libraries ------------------------------
    // `logging` shares one export with `release` and provides one of its own. Under the old
    // any-collision rule the whole module was dropped and kOnlyB was lost with it -- which is how
    // Darksiders II lost steam_api_ps5.prx (1199 exports, 19.2% claimed) over a single NID and
    // BlazBlue lost cri_lips_unity.prx (527 exports, 0.4% claimed) over two (#3497).
    //
    // Kills: reverting to "skip on any collision". That makes `mods.size()` 2 and `kOnlyB` absent,
    // so both assertions below redden -- which is exactly what this arm existed to catch and could
    // not, when its fixture shared 50% of a two-symbol module and was called a duplicate build.
    {
        std::vector<LinkInput> inputs = {
            { main_path, kBase0 }, { release_path, kBase1 }, { logging_path, kBase2 },
        };
        inputs[2].skip_on_export_collision = true;
        Program p; std::string err;
        const bool ok = link_program(inputs, kStubBase, p, &err);
        CHECK(ok, "distinct library: link_program succeeds");
        CHECK(p.skipped_modules.empty(), "distinct library: the module is NOT skipped");
        CHECK(p.mods.size() == 3, "distinct library: all three modules are linked");
        CHECK(export_of(p, kOnlyB) == kBase2 + kExp1,
              "distinct library: its OWN export survives, rather than being lost with it");
        // First-definition-wins is unchanged, and the alias is recorded rather than silent.
        CHECK(export_of(p, kShared) == kBase1 + kExp0,
              "distinct library: the shared NID still resolves to whoever claimed it first");
        CHECK(p.aliased_exports.size() == 1,
              "distinct library: the duplicate is REPORTED as an alias, not hidden");
    }

    // ---- Arm 3: the mutation that must NOT fire the guard ---------------------------------------
    // Identical to arm 2 in every respect except one export NID. Without this arm, a guard that
    // skipped every flagged module would pass arm 2 exactly as well as a correct one.
    {
        std::vector<LinkInput> inputs = {
            { main_path, kBase0 }, { release_path, kBase1 }, { alt_path, kBase2 },
        };
        inputs[2].skip_on_export_collision = true;
        Program p; std::string err;
        const bool ok = link_program(inputs, kStubBase, p, &err);
        CHECK(ok, "mutation: link_program succeeds");
        CHECK(p.skipped_modules.empty(),
              "mutation: a flagged module whose exports are all distinct is NOT skipped");
        CHECK(p.mods.size() == 3, "mutation: all three modules are linked");
        CHECK(p.aliased_exports.empty(), "mutation: no NID is aliased");
        CHECK(export_of(p, kAltB) == kBase2 + kExp0, "mutation: the distinct export is in the table");
    }

    // ---- Arm 4: a defined-but-valueless symbol is not an export ---------------------------------
    {
        std::vector<LinkInput> inputs = {
            { main_path, kBase0 }, { release_path, kBase1 }, { valueless_path, kBase2 },
        };
        inputs[2].skip_on_export_collision = true;
        Program p; std::string err;
        const bool ok = link_program(inputs, kStubBase, p, &err);
        CHECK(ok, "valueless: link_program succeeds");
        CHECK(p.skipped_modules.empty(),
              "valueless: a defined symbol with st_value 0 does not collide (it is not an export)");
        CHECK(p.mods.size() == 3, "valueless: the module is linked");
        CHECK(export_of(p, kShared) == kBase1 + kExp0,
              "valueless: the valueless definition did not claim the NID either");
    }

    // ---- Arm 5: the same imports, stubbed instead of bound, when no module exports them ----------
    // `alt_spec` imports exactly what `logging_spec` does. In arm 3 its first import was bound
    // cross-module to the release module; dropping that module — changing nothing about the importer —
    // must move the same import to a stub slot. This is what proves resolution keys on the NID rather
    // than on the import's position, and it is the "wrong NID" case seen from the import side.
    {
        const std::vector<LinkInput> inputs = { { main_path, kBase0 }, { alt_path, kBase2 } };
        Program p; std::string err;
        const bool ok = link_program(inputs, kStubBase, p, &err);
        CHECK(ok, "no-exporter: link_program succeeds");
        CHECK(p.slots.size() == 3, "no-exporter: all three imports now need stub slots");
        CHECK(p.slots.size() == 3 && p.slots[0].nid == kShared && p.slots[1].nid == kOnlyA &&
              p.slots[2].nid == kMissing,
              "no-exporter: stub slots are allocated in first-encounter order");
        CHECK(image_u64(p, 1, kGot0) == kStubBase + 1 * p.stub_size,
              "no-exporter: the previously cross-module import now addresses its own stub slot");
        CHECK(p.resolved_cross_module == 0 && p.stubbed == 3, "no-exporter: nothing resolved cross-module");
    }

    // ---- Arm 6: a module built BY HAND, outside the generator ------------------------------------
    // handmade_prx.h lays out two PT_LOADs whose file offsets and vaddrs diverge, so every dynamic
    // table is reached through Module::va2foff's translation. synth_prx.h cannot express that shape:
    // a loader that confused a vaddr for a file offset would pass every arm above and fail here.
    {
        const std::string hand_path = dir + "/handmade.prx";
        std::string err;
        if (!prosper_test::write_module_bytes(
                hand_path, prosper_test::handmade_prx_bytes(kHandA, kHandB, kHandImp), &err)) {
            printf("  [FAIL] cannot write the hand-built fixture: %s\n", err.c_str());
            fails++;
        } else {
            SynthModuleSpec importer_spec; importer_spec.imports = { kHandA };
            const std::string importer_path = emit("synth_hand_importer.prx", importer_spec);

            const std::vector<LinkInput> inputs = { { importer_path, kBase0 }, { hand_path, kBase1 } };
            Program p;
            const bool ok = link_program(inputs, kStubBase, p, &err);
            CHECK(ok, "handmade: link_program accepts a module laid out by hand");
            if (!ok) printf("  link error: %s\n", err.c_str());

            CHECK(export_of(p, kHandA) == kBase1 + prosper_test::kHandExportAVa,
                  "handmade: its first export is in the table at the right address");
            CHECK(export_of(p, kHandB) == kBase1 + prosper_test::kHandExportBVa,
                  "handmade: its second export is in the table at the right address");
            CHECK(image_u64(p, 0, kGot0) == kBase1 + prosper_test::kHandExportAVa,
                  "handmade: a generated module's import binds to the hand-built definition");
            // The relocation lands in the SECOND load segment, whose file offset is not its vaddr.
            CHECK(image_u64(p, 1, prosper_test::kHandInitArrayVa) == kBase1 + prosper_test::kHandExportAVa,
                  "handmade: a relocation is applied through a non-identity vaddr->file-offset map");
            bool init_seen = false;
            for (uint64_t fn : p.init_fns)
                if (fn == kBase1 + prosper_test::kHandExportAVa) init_seen = true;
            CHECK(init_seen, "handmade: DT_INIT_ARRAY is collected from the relocated data segment");

            // The collision guard's domain includes hand-built modules: a generated module flagged
            // optional must be dropped when the hand-built one already owns its NID.
            SynthModuleSpec clash_spec; clash_spec.exports = { kHandA };
            const std::string clash_path = emit("synth_hand_clash.prx", clash_spec);
            std::vector<LinkInput> guarded = { { hand_path, kBase1 }, { clash_path, kBase2 } };
            guarded[1].skip_on_export_collision = true;
            Program g;
            const bool okg = link_program(guarded, kStubBase, g, &err);
            CHECK(okg, "handmade: the guarded link succeeds");
            CHECK(g.skipped_modules.size() == 1 && g.skipped_modules[0].owner_path == hand_path &&
                  g.skipped_modules[0].nid == kHandA,
                  "handmade: the collision guard fires with the hand-built module as the owner");
        }
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
