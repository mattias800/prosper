// test_nid — validate the NID hash and report import name coverage.
// The algorithm is locked by a golden vector verified against the game's real
// libc.prx exports: memcpy -> "Q3VBxCXhUHs". A dictionary match count against the
// loaded module guards the whole pipeline (hash + symbol extraction).
#include "../src/self/module.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <set>
#include <string>

using namespace prosper;
static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { if (cond) g_pass++; else { g_fail++; \
    printf("  [FAIL] %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

int main(int argc, char** argv) {
    const char* path = (argc >= 2) ? argv[1] : "../../PPSA24651-app0/eboot.bin";
    printf("== test_nid: %s ==\n", path);

    // Golden vectors (validated against real libc.prx export NIDs).
    CHECK(nid_hash("memcpy") == "Q3VBxCXhUHs", "memcpy NID = '%s' expected 'Q3VBxCXhUHs'", nid_hash("memcpy").c_str());
    CHECK(nid_hash("memcpy").size() == 11, "nid length != 11");

    std::string err;
    auto mo = Module::load(path, &err);
    CHECK(mo.has_value(), "load: %s", err.c_str());
    if (!mo) return 1;
    Module& m = *mo;

    // All NIDs this module references (imports + exports).
    std::set<std::string> nids;
    for (auto& im : m.imports) nids.insert(im.nid);
    for (auto& s : m.symbols) if (!s.is_import && !s.nid.empty()) nids.insert(s.nid);

    NidDb db;
    int matched = 0;
    for (auto& name : builtin_symbol_names())
        if (nids.count(nid_hash(name))) matched++;
    printf("  built-in dictionary names present in this module: %d / %zu\n",
           matched, builtin_symbol_names().size());
    CHECK(matched >= 10, "only %d dictionary names matched (pipeline suspect)", matched);

    // Coverage: how many imports can we now show by name?
    int named = 0;
    for (auto& im : m.imports)
        if (!db.resolve(im.nid).empty()) named++;
    printf("  imports resolvable to names via built-in DB: %d / %zu\n", named, m.imports.size());

    // Show a few resolved imports as a sanity sample.
    int shown = 0;
    for (auto& im : m.imports) {
        const std::string& n = db.resolve(im.nid);
        if (!n.empty() && shown < 12) { printf("    %s::%s  ->  %s\n", im.lib_name.c_str(), im.nid.c_str(), n.c_str()); shown++; }
    }

    printf("\n== %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail;
}
