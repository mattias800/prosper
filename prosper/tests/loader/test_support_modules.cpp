// test_support_modules.cpp — a bundled support PRX is preloaded only when something imports it.
//
// The defect: prosper preloads a fixed list of optional modules because it has no runtime PRX
// loading. Preloading runs the module's `module_start` — guest code — so a module added for one
// title can wedge an unrelated title that merely ships the same file.
// `sce_module/libSceNpCppWebApi.prx` was added for *Sonic Origins*, which imports it;
// *Sniper Ghost Warrior Contracts 2* ships it, imports `libSceNpWebApi2` instead, and deadlocks in
// that module's `module_start` 81 ms into the boot.
//
// Pure policy cases need no dump or SELF parser. The discovery case uses only synthetic modules
// to exercise the real candidate list and parsed dependency veto without executing constructors.
// Both directions matter and the KEEP arm is the load-bearing one: a filter that drops everything
// would satisfy the drop arm perfectly and silently break Sonic Origins.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "loader/support_modules.hpp"
#include "host/image/boot_program.hpp"
#include "fixtures/handmade_prx.h"
#include "fixtures/synth_prx.h"
#include "fixtures/test_scratch.h"

using prosper::LinkInput;
using prosper::support_module_lib_name;
using prosper::unimported_support_module_indices;

#define CHECK(cond, msg) EXPECT_TRUE(cond) << (msg)

static LinkInput candidate(const char* path) {
    LinkInput e; e.path = path; e.base = 0; e.only_if_imported = true; return e;
}
static LinkInput ordinary(const char* path) {
    LinkInput e; e.path = path; e.base = 0; return e;
}
static bool dropped(const std::vector<size_t>& d, size_t i) {
    for (size_t x : d) if (x == i) return true;
    return false;
}

TEST(SupportModules, Contract) {
    printf("== test_support_modules ==\n");

    // --- library name derivation -------------------------------------------------------------
    CHECK(support_module_lib_name("/dump/sce_module/libSceNpCppWebApi.prx") == "libSceNpCppWebApi",
          "a POSIX path yields the bare library name");
    CHECK(support_module_lib_name("C:\\dump\\sce_module\\libSceNpCppWebApi.prx") == "libSceNpCppWebApi",
          "a Windows-spelled path yields the same name");
    CHECK(support_module_lib_name("libc.prx") == "libc", "a bare filename works");
    CHECK(support_module_lib_name("/d/libfmodstudio.sprx") == "libfmodstudio", "the .sprx extension is stripped");

    // --- the basename rule is NOT general, and it fails CLOSED --------------------------------
    // Pinned so the limitation is visible rather than latent. These three are real spellings from
    // PPSA03130's own import table: two of them the basename rule cannot produce. Nothing is broken
    // today (the only module using the flag is spelled consistently), but the next person to set
    // `only_if_imported` on a `.native.prx` would get a SILENT wrong drop, because a name that does
    // not match reads exactly like a module nobody imports.
    CHECK(support_module_lib_name("/d/libSceMsgDialog.native.prx") == "libSceMsgDialog.native",
          "NATIVE: the basename rule keeps a '.native' stem (this one matches the real import)");
    CHECK(support_module_lib_name("/d/libSceAjm.native.prx") != "libSceAjm",
          "NATIVE (fails closed): the rule CANNOT derive 'libSceAjm', which is how the real "
          "libSceAjm.native.prx is imported -- do not set only_if_imported on it without a mapping");
    CHECK(support_module_lib_name("/d/libSceSaveData.native.prx") != "libSceSaveData_native",
          "NATIVE (fails closed): nor 'libSceSaveData_native', the real spelling for that one");

    // --- the DROP direction: nobody imports it -------------------------------------------------
    {
        std::vector<LinkInput> in = { ordinary("/d/eboot.bin"), candidate("/d/sce_module/libSceNpCppWebApi.prx") };
        // The real PPSA03130 case: the eboot imports a DIFFERENT NP library whose name is a near
        // miss, which is exactly the confusion the byte-search version of this check would make.
        std::vector<std::vector<std::string>> imports = { { "libSceNpWebApi2", "libSceAgcCore", "libc" }, {} };
        const auto d = unimported_support_module_indices(in, imports);
        CHECK(d.size() == 1 && dropped(d, 1),
              "DROP: a candidate no module imports is dropped (libSceNpWebApi2 is not it)");
    }

    // --- the KEEP direction: something imports it ----------------------------------------------
    {
        std::vector<LinkInput> in = { ordinary("/d/eboot.bin"), candidate("/d/sce_module/libSceNpCppWebApi.prx") };
        std::vector<std::vector<std::string>> imports = { { "libSceNpCppWebApi", "libc" }, {} };
        const auto d = unimported_support_module_indices(in, imports);
        CHECK(d.empty(), "KEEP: a candidate the eboot imports is preloaded (the Sonic Origins case)");
    }

    // --- a candidate may not vouch for itself ---------------------------------------------------
    {
        std::vector<LinkInput> in = { ordinary("/d/eboot.bin"), candidate("/d/sce_module/libSceNpCppWebApi.prx") };
        // The candidate's OWN entry names the library; the eboot's does not.
        std::vector<std::vector<std::string>> imports = { { "libc" }, { "libSceNpCppWebApi" } };
        const auto d = unimported_support_module_indices(in, imports);
        CHECK(d.size() == 1 && dropped(d, 1),
              "SELF: a candidate importing its own library does not justify its own preload");
    }

    // --- two candidates that import each other keep neither -------------------------------------
    {
        std::vector<LinkInput> in = { ordinary("/d/eboot.bin"), candidate("/d/a.prx"), candidate("/d/b.prx") };
        std::vector<std::vector<std::string>> imports = { { "libc" }, { "b" }, { "a" } };
        const auto d = unimported_support_module_indices(in, imports);
        CHECK(d.size() == 2 && dropped(d, 1) && dropped(d, 2),
              "MUTUAL: two candidates importing each other cannot keep each other alive");
    }

    // --- non-candidates are never touched --------------------------------------------------------
    {
        std::vector<LinkInput> in = { ordinary("/d/eboot.bin"), ordinary("/d/Media/Plugins/AkSoundEngine.prx") };
        // The Wwise/FMOD/PSN plugins are preloaded BECAUSE nothing imports them statically — they
        // are reached through sceKernelDlsym at runtime. Applying the rule to them would drop every
        // one, which is the way this change could plausibly break other titles.
        std::vector<std::vector<std::string>> imports = { { "libc" }, {} };
        const auto d = unimported_support_module_indices(in, imports);
        CHECK(d.empty(), "RUNTIME PLUGINS: an input that did not opt in is never dropped, imported or not");
    }

    // --- no candidates at all: no work, no drops ---------------------------------------------------
    {
        std::vector<LinkInput> in = { ordinary("/d/eboot.bin"), ordinary("/d/sce_module/libc.prx") };
        std::vector<std::vector<std::string>> imports = { {}, {} };
        CHECK(unimported_support_module_indices(in, imports).empty(),
              "NO CANDIDATES: a title that ships none of these files is unaffected");
    }

    // --- indices come back descending, so the caller's erase loop stays valid ------------------------
    {
        std::vector<LinkInput> in = { ordinary("/d/eboot.bin"), candidate("/d/a.prx"),
                                      ordinary("/d/x.prx"),    candidate("/d/b.prx") };
        std::vector<std::vector<std::string>> imports = { {}, {}, {}, {} };
        const auto d = unimported_support_module_indices(in, imports);
        bool desc = true;
        for (size_t i = 1; i < d.size(); ++i) if (d[i - 1] <= d[i]) desc = false;
        CHECK(d.size() == 2 && desc, "ORDER: dropped indices are descending (erase-safe)");
    }

}

using prosper::ModuleLinkFacts;

TEST(SupportModules, InitOnLoadVetoedWhenImported) {
    LinkInput eboot = ordinary("/d/eboot.bin");
    LinkInput imported = ordinary("/d/libA.prx"); imported.init_on_load = true;
    LinkInput lone = ordinary("/d/libB.prx");     lone.init_on_load = true;
    std::vector<LinkInput> in = {eboot, imported, lone};
    // eboot imports libA; libB's own import of itself must not vouch for it.
    std::vector<ModuleLinkFacts> facts(3);
    facts[0].imported_libs = {"libA"};
    facts[2].imported_libs = {"libB"};
    const auto cleared = prosper::veto_imported_init_on_load(in, facts);
    EXPECT_EQ(cleared.size(), 1u);
    EXPECT_FALSE(in[1].init_on_load) << "an imported module must init eagerly";
    EXPECT_TRUE(in[2].init_on_load) << "an unimported module keeps deferral";
}

TEST(SupportModules, InitOnLoadVetoedByDeferredImporter) {
    LinkInput a = ordinary("/d/libA.prx"); a.init_on_load = true;
    LinkInput b = ordinary("/d/libB.prx"); b.init_on_load = true;
    std::vector<LinkInput> in = {ordinary("/d/eboot.bin"), a, b};
    std::vector<ModuleLinkFacts> facts(3);
    facts[1].imported_libs = {"libB"};
    prosper::veto_imported_init_on_load(in, facts);
    EXPECT_TRUE(in[1].init_on_load);
    EXPECT_FALSE(in[2].init_on_load) << "B is a dependency of A, so it must init at boot";
}

// The linker binds an import to an export by NID, so the library name an importer uses need not
// match the candidate's filename. `middleware.prx` exporting what `RuntimeApi` importers ask for.
TEST(SupportModules, InitOnLoadVetoedByNidDependencyWithMismatchedLibraryName) {
    LinkInput mw = ordinary("/d/middleware.prx"); mw.init_on_load = true;
    std::vector<LinkInput> in = {ordinary("/d/eboot.bin"), mw};
    std::vector<ModuleLinkFacts> facts(2);
    facts[0].imported_libs = {"RuntimeApi"};          // does NOT match "middleware"
    facts[0].imported_nids = {"nidA", "nidB"};
    facts[1].exported_nids = {"nidB", "nidC"};
    prosper::veto_imported_init_on_load(in, facts);
    EXPECT_FALSE(in[1].init_on_load) << "the eboot calls a NID this module defines";
}

TEST(SupportModules, InitOnLoadKeptWhenNoNidOrNameDependency) {
    LinkInput mw = ordinary("/d/middleware.prx"); mw.init_on_load = true;
    std::vector<LinkInput> in = {ordinary("/d/eboot.bin"), mw};
    std::vector<ModuleLinkFacts> facts(2);
    facts[0].imported_libs = {"RuntimeApi"};
    facts[0].imported_nids = {"nidA"};
    facts[1].exported_nids = {"nidC"};
    prosper::veto_imported_init_on_load(in, facts);
    EXPECT_TRUE(in[1].init_on_load);
}

TEST(SupportModules, InitOnLoadVetoUsesActualSectionZeroExport) {
    prosper::Module provider;
    prosper::Symbol actual;
    actual.nid = "definedNid";
    actual.value = 0x1000u;
    actual.shndx = 0;
    actual.is_import = false;
    provider.symbols = {actual};
    auto zero_value = actual;
    zero_value.nid = "zeroNid";
    zero_value.value = 0;
    zero_value.shndx = 1;
    provider.symbols.push_back(zero_value);
    auto imported = actual;
    imported.nid = "importedNid";
    imported.is_import = true;
    provider.symbols.push_back(imported);
    auto unnamed = actual;
    unnamed.nid.clear();
    provider.symbols.push_back(unnamed);
    const auto provider_facts = prosper::module_link_facts(provider);
    EXPECT_EQ(provider_facts.exported_nids, std::vector<std::string>{"definedNid"});

    LinkInput middleware = ordinary("/d/middleware.prx");
    middleware.init_on_load = true;
    std::vector<LinkInput> inputs = {ordinary("/d/eboot.bin"), middleware};
    std::vector<ModuleLinkFacts> facts(2);
    facts[0].imported_libs = {"RuntimeApi"};
    facts[0].imported_nids = {"definedNid"};
    facts[1] = provider_facts;
    prosper::veto_imported_init_on_load(inputs, facts);
    EXPECT_FALSE(inputs[1].init_on_load);

    inputs[1].init_on_load = true;
    facts[0].imported_nids = {"zeroNid", "importedNid"};
    prosper::veto_imported_init_on_load(inputs, facts);
    EXPECT_TRUE(inputs[1].init_on_load) << "non-exports do not manufacture a dependency";
}

TEST(SupportModules, DeferredInitRequiresAffirmativeRootModuleContract) {
    EXPECT_TRUE(prosper::has_deferred_module_init_contract("/d", "/d/libaegir_f.prx"));
    EXPECT_TRUE(prosper::has_deferred_module_init_contract("/d/", "/d/LIBAegir_F.PRX"));
    EXPECT_TRUE(prosper::has_deferred_module_init_contract("C:\\d", "C:\\d\\libaegir_f.prx"));
    EXPECT_TRUE(prosper::has_deferred_module_init_contract(".", "./libaegir_f.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("/d", "/d/libmemorywrapper_f.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("/d", "/d/unknown.prx"));
    EXPECT_FALSE(
        prosper::has_deferred_module_init_contract("/d", "/d/Media/Plugins/libaegir_f.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("/d", "/d/Media/Plugins/PSN.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("/d", "/d/Media/Plugins/libfmodL.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("/d", "/other/libaegir_f.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("/d", "/d/../other/libaegir_f.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("", "libaegir_f.prx"));
    EXPECT_FALSE(prosper::has_deferred_module_init_contract("/d", ""));
}

// Exercise the real discovery/candidate/veto path, not just the filename helper. Unknown root
// modules and runtime plugins must retain eager constructors even without static importers.
TEST(SupportModules, BootDiscoveryKeepsUnknownAndPluginInitEager) {
    ASSERT_EQ(std::getenv("PROSPER_NO_PLUGIN_AUTOLINK"), nullptr);
    namespace fs = std::filesystem;
    const auto root = prosper_test::test_scratch_path("deferred-init-root");
    fs::create_directories(root / "Media/Plugins");
    std::string error;
    const auto eboot = root / "eboot.bin";
    ASSERT_TRUE(prosper_test::write_module_bytes(
        eboot.string(), prosper_test::handmade_prx_bytes("entryA", "entryB", "unusedNid"), &error))
        << error;
    prosper_test::SynthModuleSpec engine;
    engine.exports = {"engineEntry"};
    const auto engine_path = root / "libaegir_f.prx";
    ASSERT_TRUE(prosper_test::write_synth_prx(engine_path.string(), engine, &error)) << error;
    ASSERT_TRUE(prosper_test::write_module_bytes(
        (root / "unknown.prx").string(),
        prosper_test::handmade_prx_bytes("unknownA", "unknownB", "unusedNid"), &error))
        << error;
    prosper_test::SynthModuleSpec plugin;
    plugin.exports = {"pluginEntry"};
    ASSERT_TRUE(prosper_test::write_synth_prx((root / "Media/Plugins/native_plugin.prx").string(),
                                              plugin, &error))
        << error;
    prosper_test::SynthModuleSpec wrapper;
    wrapper.exports = {"wrapperEntry"};
    ASSERT_TRUE(
        prosper_test::write_synth_prx((root / "libmemorywrapper_f.prx").string(), wrapper, &error))
        << error;
    const auto parsed_engine = prosper::Module::load(engine_path.string(), &error);
    ASSERT_TRUE(parsed_engine.has_value()) << error;
    ASSERT_NE(parsed_engine->init_va, 0u);

    auto inputs = prosper::boot_link_inputs(root.string(), false);
    auto find_input = [&](const fs::path& path) {
        return std::find_if(inputs.begin(), inputs.end(),
                            [&](const LinkInput& input) { return fs::path(input.path) == path; });
    };
    auto selected = find_input(engine_path);
    ASSERT_NE(selected, inputs.end());
    EXPECT_TRUE(selected->init_on_load);
    for (const auto& path : {eboot, root / "unknown.prx", root / "libmemorywrapper_f.prx",
                             root / "Media/Plugins/native_plugin.prx"}) {
        const auto eager = find_input(path);
        ASSERT_NE(eager, inputs.end()) << path.string();
        EXPECT_FALSE(eager->init_on_load) << path.string();
    }

    // A genuine parsed NID dependency vetoes the known candidate even when the importer library
    // name differs.
    prosper_test::SynthModuleSpec importer;
    importer.imports = {"engineEntry"};
    ASSERT_TRUE(prosper_test::write_synth_prx(eboot.string(), importer, &error)) << error;
    inputs = prosper::boot_link_inputs(root.string(), false);
    selected = find_input(engine_path);
    ASSERT_NE(selected, inputs.end());
    EXPECT_FALSE(selected->init_on_load);

    // An unreadable importer is unknown, not proof that no constructor depends on the engine.
    ASSERT_TRUE(prosper_test::write_module_bytes(
        eboot.string(), prosper_test::handmade_prx_bytes("entryA", "entryB", "unusedNid"), &error))
        << error;
    auto malformed = plugin;
    malformed.e_machine = 0;
    ASSERT_TRUE(prosper_test::write_synth_prx((root / "unknown.prx").string(), malformed, &error))
        << error;
    ASSERT_FALSE(prosper::Module::load((root / "unknown.prx").string(), &error).has_value());
    inputs = prosper::boot_link_inputs(root.string(), false);
    selected = find_input(engine_path);
    ASSERT_NE(selected, inputs.end());
    EXPECT_FALSE(selected->init_on_load);
}
