// test_support_modules.cpp — a bundled support PRX is preloaded only when something imports it.
//
// The defect: prosper preloads a fixed list of optional modules because it has no runtime PRX
// loading. Preloading runs the module's `module_start` — guest code — so a module added for one
// title can wedge an unrelated title that merely ships the same file.
// `sce_module/libSceNpCppWebApi.prx` was added for *Sonic Origins*, which imports it;
// *Sniper Ghost Warrior Contracts 2* ships it, imports `libSceNpWebApi2` instead, and deadlocks in
// that module's `module_start` 81 ms into the boot.
//
// The policy is a pure function so both directions are testable with no dump and no SELF parser.
// Both directions matter and the KEEP arm is the load-bearing one: a filter that drops everything
// would satisfy the drop arm perfectly and silently break Sonic Origins.

#include <cstdio>
#include <string>
#include <vector>

#include "loader/support_modules.hpp"

using prosper::LinkInput;
using prosper::support_module_lib_name;
using prosper::unimported_support_module_indices;

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  [FAIL] %s\n", msg); fails++; } \
                              else        { printf("  [ok]   %s\n", msg); } } while (0)

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

int main() {
    printf("== test_support_modules ==\n");

    // --- library name derivation -------------------------------------------------------------
    CHECK(support_module_lib_name("/dump/sce_module/libSceNpCppWebApi.prx") == "libSceNpCppWebApi",
          "a POSIX path yields the bare library name");
    CHECK(support_module_lib_name("C:\\dump\\sce_module\\libSceNpCppWebApi.prx") == "libSceNpCppWebApi",
          "a Windows-spelled path yields the same name");
    CHECK(support_module_lib_name("libc.prx") == "libc", "a bare filename works");
    CHECK(support_module_lib_name("/d/libfmodstudio.sprx") == "libfmodstudio", "the .sprx extension is stripped");

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

    printf(fails ? "\ntest_support_modules: %d FAILURE(S)\n" : "\ntest_support_modules: all ok\n", fails);
    return fails ? 1 : 0;
}
