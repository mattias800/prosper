// test_dtpmod_tls — DTPMOD64 must resolve a cross-module TLS reference to the DEFINING module's
// TLS module id, not the relocating module's (#136). A module-local TLS symbol keeps this module's
// id. Builds synthetic Module + LoadedImage (no game dump) and drives apply_relocations directly.
#include "../src/self/module.hpp"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unordered_map>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static uint64_t rd64(const LoadedImage& img, uint64_t off) {
    uint64_t v = 0; std::memcpy(&v, img.mem.data() + off, 8); return v;
}

int main() {
    printf("== test_dtpmod_tls ==\n");

    // Module with TLS assigned id 5; symbol[1] is a cross-module TLS IMPORT (nid "TLSX"),
    // symbol[2] is a module-local TLS def. Two DTPMOD64 relocs, one per symbol.
    Module m;
    m.symbols.resize(3);
    m.symbols[1].nid = "TLSX"; m.symbols[1].is_import = true;  m.symbols[1].value = 0;
    m.symbols[2].nid = "local"; m.symbols[2].is_import = false; m.symbols[2].value = 0x40;
    m.relocs.push_back({ /*offset*/ 0x00, R_X86_64_DTPMOD64, /*sym*/ 1, /*addend*/ 0, false });
    m.relocs.push_back({ /*offset*/ 0x08, R_X86_64_DTPMOD64, /*sym*/ 2, /*addend*/ 0, false });

    auto make_img = [&]() {
        LoadedImage img;
        img.base = 0; img.min_vaddr = 0; img.max_vaddr = 0x100;
        img.mem.assign(0x100, 0);
        img.tls_modid = 5;   // THIS module's TLS id
        return img;
    };

    // 1. WITH the resolver map: the import "TLSX" is defined by module id 9.
    {
        std::unordered_map<std::string, uint32_t> tls_by_nid{ {"TLSX", 9} };
        LoadedImage img = make_img();
        size_t applied = apply_relocations(m, img, &tls_by_nid);
        CHECK(applied == 2, "both DTPMOD64 relocs applied");
        CHECK(rd64(img, 0x00) == 9, "cross-module TLS import resolves to the DEFINING module id (9), not 5");
        CHECK(rd64(img, 0x08) == 5, "module-local TLS resolves to this module's id (5)");
    }

    // 2. WITHOUT a resolver (the old behavior / a table-less caller): the import can't be resolved,
    //    so it falls back to the local id AND is counted unhandled — no longer a silent mis-resolve.
    {
        LoadedImage img = make_img();
        size_t applied = apply_relocations(m, img, /*tls_modid_by_nid*/ nullptr);
        CHECK(rd64(img, 0x00) == 5, "no resolver: cross-module import falls back to local id (documented)");
        CHECK(rd64(img, 0x08) == 5, "no resolver: module-local TLS still resolves to the local id");
        (void)applied;
    }

    // 3. An unknown import (not in the map) also falls back + logs, never mis-resolves silently.
    {
        std::unordered_map<std::string, uint32_t> tls_by_nid{ {"OTHER", 7} };
        LoadedImage img = make_img();
        apply_relocations(m, img, &tls_by_nid);
        CHECK(rd64(img, 0x00) == 5, "import absent from the map falls back to the local id");
    }

    if (fails) { printf("== FAIL: %d ==\n", fails); return 1; }
    printf("== PASS ==\n");
    return 0;
}
