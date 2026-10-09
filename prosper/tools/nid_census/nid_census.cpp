// nid_census — which of a title's imports fall to the dispatcher's return-0 default?
//
// An import prosper does not register never reaches a handler: `prosper_on_unimpl` logs it once and
// returns 0 (src/hle/dispatch/dispatch.cpp). For a contract whose success code is 0 — i.e. most `SCE_OK`
// functions — that 0 IS "success", so the guest is told an operation completed that never ran, and
// any out-parameter the call was supposed to fill is left holding whatever was already there. That
// is strictly worse than an error return: the failure is silent at the call site and surfaces
// arbitrarily far away. Issue #2081 calls this the FALSE SUCCESS class.
//
// The boot log already reports each such NID ("[prosper] unimplemented: ... -> returning 0"), but
// only for the code paths a particular run happened to execute, and only after booting the title.
// This answers the same question STATICALLY and exhaustively: every import in the module, whether
// or not a run reaches it, across as many titles as you point it at.
//
// It is deliberately built on prosper's OWN sources of truth rather than a source grep:
//   * imports come from `Module::load` — the same parser the loader uses, so the census sees the
//     exact NID set the loader will try to bind;
//   * cross-module exports come from `module_export_nids`, the loader's own definition of what a
//     module contributes to the global export table;
//   * registration comes from `Hle::registered` after `register_builtin_hle()` — the real runtime
//     registry, so a NID registered from a table, a loop, or a raw literal is seen identically.
// A grep over `register_fn(...)` call sites would miss every non-literal registration and would
// report those as false-success candidates. Asking the registry cannot.
//
// The cross-module half is not cosmetic. `linker.cpp` pass 2 resolves an import against the global
// export table FIRST — "Cross-module export beats a stub slot" — so an import that another of the
// title's own modules defines never reaches the dispatcher at all, no matter what the HLE registry
// says. A census that only differenced imports against the registry would report every one of those
// as a false-success candidate. Reachability here therefore means "unregistered AND undefined by
// any module shipped with this title", per title, which is exactly the condition under which
// `prosper_on_unimpl` runs.
//
// --data-only answers a different question with the same machinery (#3529): which imports are
// DATA (ELF STT_OBJECT) rather than functions, and how many of those no sibling module defines.
// Those are exactly the bindings the linker sends to the writable import-data aperture, where the
// guest reads a zero it was never promised. It is the static counterpart of the runtime
// `[import-data]` listing PROSPER_STUBDUMP prints. Registration is meaningless for a variable, so
// that column is not a filter in this mode.
//
// Usage:
//   nid_census <app0-dir|module> [more...] [--names <PS5-3.20_Libs-dir>]
//              [--registered] [--tsv] [--lib <substr>] [--self-check] [--data-only]
//
// `<app0-dir>` selects the loader's link set; an explicit module path selects that module alone.
// Each input's selection and parse counts are reported beside the aggregate census. Passing several
// dump roots ranks each NID by how many of them import it, which is the signal #2081 asks for.
//
// `--names` points at the PS5 stub dump (a directory of `sprx_dlsym(...)` libraries), which is
// AUTHORITATIVE, OR at a flat `NID name` database file (aerolib.csv / ps5rs), which is a SECONDARY
// fallback. Both carry `<NID> <-> <funcName>` pairs directly. A flat source's names are verified
// against prosper's `nid_hash` unconditionally and only proven preimages are shown (the rest are
// dropped and counted), because a community string is only trustworthy once it hashes to the NID.
// For the authoritative dump, `--self-check` re-derives each pair and reports any disagreement:
// there a mismatch points at prosper's own `nid_hash`, so the dump name is kept and flagged rather
// than dropped.
#include "host/image/boot_program.hpp"
#include "../common/nid_stub_names.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "../../src/loader/linker.hpp"
#include "../../src/self/module.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace prosper;
namespace fs = std::filesystem;

namespace {

struct Row {
    std::string nid;
    std::string name;   // "" when the stub dump does not name it
    std::set<std::string> libs;   // import library names as the module declares them
    std::set<std::string> titles;   // legacy aggregate labels: distinct input basenames
    size_t modules = 0;   // how many modules import it
    // ELF64_ST_TYPE values this NID was imported with. A set, not a scalar: nothing stops two
    // modules from declaring the same NID with different types, and collapsing that to one value
    // would hide it. STT_OBJECT here means the linker binds it to the import-data aperture (#3529).
    std::set<unsigned> elf_types;
    bool object() const { return elf_types.count(STT_OBJECT) != 0; }
};

struct ModuleSelection {
    std::vector<fs::path> paths;
    bool single_module = false;
};

struct InputScope {
    std::string input;
    bool single_module = false;
    size_t selected = 0, read = 0, failed = 0, data_bindings = 0;
};

// Metadata stays on one LF-delimited record even when a POSIX path contains control bytes.
// Backslashes are escaped too, so literal escape-looking text remains distinguishable. Preserve
// ordinary UTF-8 bytes and the existing aggregate TSV fields.
std::string input_label(const std::string& input) {
    constexpr char hex[] = "0123456789abcdef";
    std::string label;
    for (unsigned char c : input) {
        if (c == '\\')
            label += "\\\\";
        else if (c < 0x20 || c == 0x7f) {
            label += "\\x";
            label += hex[c >> 4];
            label += hex[c & 0xf];
        } else
            label += static_cast<char>(c);
    }
    return label;
}

// The ELF symbol types this corpus actually carries, spelled for a report.
const char* sym_type_name(unsigned t) {
    switch (t) {
        case STT_NOTYPE: return "NOTYPE";
        case STT_OBJECT: return "OBJECT";
        case STT_FUNC: return "FUNC";
        case STT_SECTION: return "SECTION";
        case STT_FILE: return "FILE";
        case STT_COMMON: return "COMMON";
        case STT_TLS: return "TLS";
        default: return "?";
    }
}
std::string sym_types_of(const Row& r) {
    std::string out;
    for (unsigned t : r.elf_types) {
        if (!out.empty()) out += "+";
        out += sym_type_name(t);
    }
    return out.empty() ? "-" : out;
}

// ---- the name table: authoritative firmware dump, or a secondary flat database ----------------
// The authoritative source is the PS5 stub dump; each generated library has one loader line per
// export:
//     if(sprx_dlsym(__handle, "PI7jIZj4pcE", &__ptr_sceRandomGetRandomNumber)) return;
// so the pair is read off directly. No hashing is required to build it, which is what makes
// --self-check meaningful: the hash is checked AGAINST the dump rather than used to produce it. A
// flat `NID name` database (aerolib.csv / ps5rs) is a secondary fallback whose names are instead
// verified by nid_hash before they are shown (see load_names). `StubNames::source` records which.
//
// The reading of the dump itself lives in tools/common/nid_stub_names.hpp so that self_dump
// --import-slots names its imports from the identical parse; only the --self-check control and its
// mismatch accounting are this tool's.
struct NameTable {
    std::map<std::string, std::string>
        by_nid;   // nid -> function name (verified, for a flat source)
    std::map<std::string, std::string> lib_of;   // nid -> library file stem
    size_t pairs = 0, mismatches = 0, dropped = 0, rejected = 0, conflicts = 0;
    prosper_tools::NameSource source = prosper_tools::NameSource::None;
    bool dir_ok = false;
};

NameTable load_names(const std::string& path, bool self_check) {
    NameTable t;
    // A DIRECTORY is the authoritative per-library sprx_dlsym firmware dump; a FILE is a flat
    // `NID name` community database (aerolib.csv / ps5rs), which is SECONDARY. load_nid_names
    // dispatches on that. No on_pair here: a NID is the hash of its name, so every pair is
    // independently checkable, and that check is run below.
    auto stub = prosper_tools::load_nid_names(path);
    t.source = stub.source;
    t.pairs = stub.pairs;
    t.rejected = stub.rejected;
    t.conflicts = stub.conflicts;
    t.dir_ok = stub.dir_ok;
    const bool flat = (stub.source == prosper_tools::NameSource::FlatDb);
    for (const auto& [nid, name] : stub.by_nid) {
        std::string chosen = name;
        bool verified = (nid_hash(name) == nid);
        if (!verified && flat) {
            // A flat database may list several names for one NID; the first need not be the real
            // preimage, so take whichever candidate hashes back to the NID (N1).
            if (auto ci = stub.candidates.find(nid); ci != stub.candidates.end())
                for (const auto& cand : ci->second)
                    if (nid_hash(cand) == nid) {
                        chosen = cand;
                        verified = true;
                        break;
                    }
        }
        if (!verified) {
            // A flat (secondary) name that is not a proven preimage of the NID is dropped: it would
            // otherwise present an unverifiable community string as if it were the real symbol. An
            // authoritative dump name is kept, and --self-check reports the disagreement instead,
            // since there it means prosper's own nid_hash is what to look at.
            if (flat) {
                t.dropped++;
                continue;
            }
            if (self_check) {
                t.mismatches++;
                // stderr, not stdout: under --tsv these land above the header and corrupt the
                // stream a consumer parses. The COUNT is reported in the scope block below.
                fprintf(stderr, "  [name-mismatch] %s: dump says %s, nid_hash() says %s\n",
                        name.c_str(), nid.c_str(), nid_hash(name).c_str());
            }
        }
        t.by_nid.emplace(nid, chosen);
        if (auto li = stub.lib_of.find(nid); li != stub.lib_of.end())
            t.lib_of.emplace(nid, li->second);
    }
    return t;
}

// ---- module discovery -------------------------------------------------------------------------
bool is_module_file(const fs::path& p) {
    if (p.filename() == "eboot.bin") return true;
    const std::string ext = p.extension().string();
    return ext == ".prx" || ext == ".sprx";
}

// The module set is now the LOADER'S link set, not a tree scan (#2199).
//
// This tool's central inference is that an import satisfied by a SIBLING module's export never
// reaches the dispatcher, so it is excluded from the census. That is sound only if the sibling set
// matches what the loader actually links. A tree scan is strictly larger -- it picks up .sprx (never
// auto-linked), everything under sce_module/ (the loader takes exactly two named files), plugin
// directories outside Media/Plugins/ and the dump root, and modules whose file the
// loader would drop or refuse. Every one of those made the tool exclude a binding that DOES fall to
// the dispatcher's `return 0` at runtime.
//
// The direction of that error is what made it worth fixing: a FALSE ABSENCE. The row is missing
// rather than wrong, so nothing in the output looks suspicious, in a report whose entire purpose is
// to enumerate what reaches the dispatcher.
//
// A single regular file still means "just this module", which is how --lib and single-module runs
// work; only a dump ROOT goes through the loader's set.
ModuleSelection collect_modules(const std::string& input) {
    ModuleSelection out;
    std::error_code ec;
    const fs::path root(input);
    if (fs::is_regular_file(root, ec)) {
        out.single_module = true;
        out.paths.push_back(root);
        return out;
    }
    // verbose=false: boot_link_inputs prints the loader's auto-link and case-correction lines, and
    // --tsv writes machine-readable rows to the same stdout.
    for (const auto& li : prosper::boot_link_inputs(input, /*verbose=*/false))
        out.paths.push_back(fs::path(li.path));
    std::sort(out.paths.begin(), out.paths.end());
    return out;
}

// Report the population before filtering and the registration state of the shown subset.
// A module that failed to parse contributes no imports, so its absence must also be explicit.
// `prefix` is "# " for --tsv (comment lines a consumer skips) and "" for the human report.
void print_scope(const char* prefix, size_t total, size_t modules_read, size_t modules_failed,
                 size_t unregistered, size_t shown, size_t shown_unregistered,
                 size_t satisfied_cross_module, const std::vector<InputScope>& input_scopes,
                 size_t data_satisfied_cross_module, size_t mismatches,
                 const std::string& lib_filter, bool self_check, bool data_only) {
    for (const auto& input : input_scopes) {
        printf("%sinput: %s -> ", prefix, input_label(input.input).c_str());
        if (input.single_module)
            printf("single module");
        else
            printf("link set, %zu module%s", input.selected, input.selected == 1 ? "" : "s");
        printf(" (%zu read, %zu unreadable)\n", input.read, input.failed);
    }
    printf("%sscope: %zu distinct imported NIDs over %zu module(s) read, %zu unreadable\n", prefix,
           total, modules_read, modules_failed);
    printf("%sscope: %zu unregistered before filtering, %zu shown (%zu unregistered)%s%s\n", prefix,
           unregistered, shown, shown_unregistered,
           lib_filter.empty() ? "" : ", --lib filter=", lib_filter.c_str());
    printf("%sscope: %zu binding(s) excluded as satisfied by a sibling module's export\n", prefix,
           satisfied_cross_module);
    {
        // #3529: what reaches the writable import-data aperture. A data import a sibling module
        // DEFINES is bound to that definition and never comes here, which is why the exclusion
        // count is reported beside it rather than left implicit.
        size_t data_total = 0, inputs_with_data = 0;
        for (const auto& input : input_scopes) {
            data_total += input.data_bindings;
            if (input.data_bindings) inputs_with_data++;
        }
        printf("%sscope: %zu DATA binding(s) (ELF STT_OBJECT) unresolved by any sibling module, "
               "over %zu of %zu input(s); a further %zu were satisfied cross-module\n",
               prefix, data_total, inputs_with_data, input_scopes.size(),
               data_satisfied_cross_module);
        // Keep each argument, including zero-count and same-basename inputs. An explicit module
        // is not a title, and its imports are classified without any sibling from another input.
        if (data_only)
            for (const auto& input : input_scopes)
                printf("%sdata input: %s -> %zu unresolved DATA binding(s)\n", prefix,
                       input_label(input.input).c_str(), input.data_bindings);
    }
    if (self_check) printf("%sscope: name-table self-check %zu mismatch(es)\n", prefix, mismatches);
    if (modules_failed)
        printf("%sWARNING: %zu module(s) did not parse -- their imports are ABSENT from this "
               "census, so a NID missing below may be unmeasured rather than unimported\n",
               prefix, modules_failed);
    // The distinction an absence in this report is most likely to be misread as. Recorded because it
    // already produced a wrong lead: ArcRunner statically imports all three sceAgc*GetSize gaps from
    // #1756 plus sceKernelWaitCommandBufferCompletion — a ready-made explanation for its fault — and
    // calls NONE of them. The runtime unimplemented-call census over a full faulting run was 12 NIDs,
    // none in libSceAgc/libSceAgcDriver/libkernel (#1226).
    printf(
        "%sNOTE: this is a STATIC import census -- what the selected modules MAY call, not what "
        "they did. A NID "
        "listed here may never execute, and a fault is not explained by its presence. For what a "
        "run actually called, use prosper_on_unimpl's first-seen census from a live boot, or "
        "hle_calls (#1980), and bound it to the window the behaviour occurs in.\n",
        prefix);
    printf("%sNOTE: dump roots use the loader's link set (boot_program.cpp); explicit module "
           "inputs inspect that module alone. Cross-module exclusions use the modules "
           "selected for each input.\n",
           prefix);
    printf("%sNOTE: aggregate #lbl (TSV titles/title_list) groups inputs by basename; "
           "the per-input lines identify each argument.\n",
           prefix);
}

void usage(const char* argv0) {
    fprintf(
        stderr,
        "usage: %s <app0-dir|module> [more...] [--names <dump-dir|nid-csv>]\n"
        "         [--registered] [--tsv] [--lib <substr>] [--self-check]\n\n"
        "  --names PATH   name NIDs from the PS5 stub dump (a directory, authoritative) or a\n"
        "                 flat `NID name` database file (aerolib.csv / ps5rs, secondary: its\n"
        "                 names are shown only when they hash back to the NID)\n"
        "  --registered   also list imports that DO have a handler (default: only unregistered)\n"
        "  --tsv          machine-readable output\n"
        "  --lib SUBSTR   only report NIDs whose import library contains SUBSTR\n"
        "  --self-check   verify every dump NID against prosper's nid_hash()\n"
        "  --data-only    only DATA imports (ELF STT_OBJECT) -- what the linker binds to the\n"
        "                 writable import-data aperture rather than to a code stub (#3529)\n",
        argv0);
}

}   // namespace

int main(int argc, char** argv) {
    std::vector<std::string> inputs;
    std::string names_dir, lib_filter;
    bool show_registered = false, tsv = false, self_check = false, data_only = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--names" && i + 1 < argc)
            names_dir = argv[++i];
        else if (a == "--lib" && i + 1 < argc)
            lib_filter = argv[++i];
        else if (a == "--registered")
            show_registered = true;
        else if (a == "--tsv")
            tsv = true;
        else if (a == "--self-check")
            self_check = true;
        else if (a == "--data-only")
            data_only = true;
        else if (a == "-h" || a == "--help") {
            usage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            usage(argv[0]);
            return 2;
        } else
            inputs.push_back(a);
    }
    if (inputs.empty()) {
        usage(argv[0]);
        return 2;
    }

    NameTable names;
    if (!names_dir.empty()) {
        names = load_names(names_dir, self_check);
        const bool flat = (names.source == prosper_tools::NameSource::FlatDb);
        if (!names.dir_ok)
            fprintf(stderr,
                    "[names] WARNING: %s is not a readable dump directory or database "
                    "file -- NO names loaded, every NID below reads as '?'\n",
                    names_dir.c_str());
        if (!tsv) {
            printf("[names] %zu name(s) from %s (%s)\n", names.by_nid.size(), names_dir.c_str(),
                   prosper_tools::name_source_str(names.source));
            if (flat)
                printf(
                    "[names] secondary source: %zu verified by nid_hash, %zu dropped unverified, "
                    "%zu malformed line(s), %zu conflict(s)\n",
                    names.by_nid.size(), names.dropped, names.rejected, names.conflicts);
            else if (names.source != prosper_tools::NameSource::FirmwareDump)
                printf("[names] no names loaded\n");
            else if (self_check)
                printf("[names] self-check: %zu mismatch(es) against nid_hash()\n",
                       names.mismatches);
            else
                printf(
                    "[names] authoritative dump (pass --self-check to verify against nid_hash)\n");
        }
    }

    // The registry is the ground truth for "is there a handler?". Populate it exactly the way a
    // boot does, then query it per NID.
    register_builtin_hle();

    std::map<std::string, Row> rows;
    size_t modules_read = 0, modules_failed = 0, satisfied_cross_module = 0;
    // Per-input DATA accounting (#3529), even outside --data-only. Selection context is kept in
    // argument order rather than keyed by a basename that may identify several different inputs.
    std::vector<InputScope> input_scopes;
    size_t data_satisfied_cross_module = 0;

    for (const auto& input : inputs) {
        const std::string title = fs::path(input).filename().string();
        const auto mods = collect_modules(input);
        InputScope scope{input, mods.single_module, mods.paths.size()};
        if (mods.paths.empty()) fprintf(stderr, "[warn] no modules under %s\n", input.c_str());

        // A title's modules are linked together, so its own exports are the first resolver. Parse
        // every module once, keep them, and take the union of their exports before classifying any
        // import — an import defined by a sibling module is bound to that definition and never
        // reaches the dispatcher, so it is not a candidate here at all.
        std::vector<Module> loaded;
        std::set<std::string> title_exports;
        for (const auto& mp : mods.paths) {
            std::string err;
            auto m = Module::load(mp.string(), &err);
            if (!m) {
                modules_failed++;
                scope.failed++;
                fprintf(stderr, "[warn] %s: %s\n", mp.string().c_str(), err.c_str());
                continue;
            }
            modules_read++;
            scope.read++;
            for (const auto& nid : module_export_nids(*m)) title_exports.insert(nid);
            loaded.push_back(std::move(*m));
        }

        for (const auto& m : loaded) {
            for (const auto& im : m.imports) {
                if (title_exports.count(im.nid)) {
                    satisfied_cross_module++;
                    if (im.elf_type == STT_OBJECT) data_satisfied_cross_module++;
                    continue;
                }
                Row& r = rows[im.nid];
                r.nid = im.nid;
                if (!im.lib_name.empty()) r.libs.insert(im.lib_name);
                r.titles.insert(title);
                r.modules++;
                r.elf_types.insert(im.elf_type);
                // One binding per IMPORT, not per NID: two modules importing the same variable are
                // two bindings that both land on the (one, deduped) data slot.
                if (im.elf_type == STT_OBJECT) scope.data_bindings++;
            }
        }
        input_scopes.push_back(std::move(scope));
    }

    // Classify against the live registry.
    std::vector<Row*> selected;
    size_t total = 0, unregistered = 0, shown_unregistered = 0;
    for (auto& [nid, r] : rows) {
        total++;
        const bool registered = Hle::registered(nid);
        if (!registered) unregistered++;
        // A variable has no handler to register, so the registration filter would silently drop
        // every row in this mode. --data-only selects on the symbol type instead.
        if (data_only) {
            if (!r.object()) continue;
        } else if (registered && !show_registered)
            continue;
        if (auto it = names.by_nid.find(nid); it != names.by_nid.end()) r.name = it->second;
        if (!lib_filter.empty()) {
            bool hit = false;
            for (const auto& l : r.libs)
                if (l.find(lib_filter) != std::string::npos) hit = true;
            if (auto it = names.lib_of.find(nid);
                it != names.lib_of.end() && it->second.find(lib_filter) != std::string::npos)
                hit = true;
            if (!hit) continue;
        }
        selected.push_back(&r);
        if (!registered) shown_unregistered++;
    }

    // Preserve the aggregate's existing rank by distinct input basenames.
    std::sort(selected.begin(), selected.end(), [](const Row* a, const Row* b) {
        if (a->titles.size() != b->titles.size()) return a->titles.size() > b->titles.size();
        if (a->name != b->name) return a->name < b->name;
        return a->nid < b->nid;
    });

    // Where a shown name came from: a flat database is secondary, so a consumer can tell its names
    // apart from authoritative dump names on the row itself (not only from the [names] header).
    const char* name_src_tag = names.source == prosper_tools::NameSource::FlatDb ? "flat_db"
                               : names.source == prosper_tools::NameSource::FirmwareDump
                                   ? "firmware_dump"
                                   : "-";

    if (tsv) {
        if (!names_dir.empty())
            printf("# names: source=%s path=%s named=%zu dropped=%zu malformed=%zu conflicts=%zu\n",
                   prosper_tools::name_source_str(names.source), names_dir.c_str(),
                   names.by_nid.size(), names.dropped, names.rejected, names.conflicts);
        // name_src is appended as the LAST column so the legacy 0-7 column positions a consumer
        // depends on stay put; a flat-sourced name is still distinguishable per row.
        printf("nid\tname\tsym_type\tregistered\ttitles\tmodules\tlibs\ttitle_list\tname_src\n");
        for (const Row* r : selected) {
            std::string libs, tl;
            for (const auto& l : r->libs) {
                if (!libs.empty()) libs += ",";
                libs += l;
            }
            for (const auto& t : r->titles) {
                if (!tl.empty()) tl += ",";
                tl += t;
            }
            printf("%s\t%s\t%s\t%d\t%zu\t%zu\t%s\t%s\t%s\n", r->nid.c_str(),
                   r->name.empty() ? "?" : r->name.c_str(), sym_types_of(*r).c_str(),
                   Hle::registered(r->nid) ? 1 : 0, r->titles.size(), r->modules, libs.c_str(),
                   tl.c_str(), r->name.empty() ? "-" : name_src_tag);
        }
        print_scope("# ", total, modules_read, modules_failed, unregistered, selected.size(),
                    shown_unregistered, satisfied_cross_module, input_scopes,
                    data_satisfied_cross_module, names.mismatches, lib_filter, self_check,
                    data_only);
        return 0;
    }

    printf("%s", data_only ? "\n== DATA imports (STT_OBJECT) no sibling module defines -> "
                             "import-data aperture ==\n"
                 : show_registered
                     ? "\n== imports (registered and unregistered) ==\n"
                     : "\n== imports with NO registered handler -> dispatcher returns 0 ==\n");
    printf("%-13s %-52s %-8s %10s %5s  %s\n", "NID", "name", "sym type", "registered", "#lbl",
           "import library");
    for (const Row* r : selected) {
        std::string libs;
        for (const auto& l : r->libs) {
            if (!libs.empty()) libs += ",";
            libs += l;
        }
        printf("%-13s %-52s %-8s %10s %5zu  %s\n", r->nid.c_str(),
               r->name.empty() ? "?" : r->name.c_str(), sym_types_of(*r).c_str(),
               Hle::registered(r->nid) ? "yes" : "no", r->titles.size(), libs.c_str());
    }
    printf("\n");
    print_scope("", total, modules_read, modules_failed, unregistered, selected.size(),
                shown_unregistered, satisfied_cross_module, input_scopes,
                data_satisfied_cross_module, names.mismatches, lib_filter, self_check, data_only);
    return 0;
}
