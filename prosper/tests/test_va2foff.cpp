// test_va2foff — regression guard for issue #113 (PPSA02664 booted 0 imports).
//
// The bug: Module::va2foff resolves a guest VA to a file offset by scanning
// `loads`, which carries PT_LOAD *and* the non-LOAD file-backed phdrs
// (DYNAMIC/PROCPARAM/TLS). The dynamic tables physically live inside a PT_LOAD,
// but a DYNAMIC phdr's own logical p_offset points at UNMAPPED bytes. When the
// DYNAMIC entry precedes the covering LOAD in program-header order (as in
// PPSA02664, but not PPSA24651), a single-pass scan matched DYNAMIC first and
// returned its garbage self-offset — so the loader parsed 0 dynamic tags / 0
// imports and jumped to garbage. The fix: resolve PT_LOAD first, non-LOAD only
// as a fallback. This test needs no game dump — it is a pure va2foff unit check.
#include "../src/self/module.hpp"
#include <cstdio>

using namespace prosper;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  [FAIL] %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static Segment seg(uint32_t type, uint64_t vaddr, uint64_t filesz, uint64_t file_off) {
    Segment s; s.type = type; s.vaddr = vaddr; s.filesz = filesz; s.memsz = filesz; s.file_off = file_off;
    return s;
}

int main() {
    printf("== test_va2foff (issue #113 regression) ==\n");

    // Reproduce PPSA02664's ordering: DYNAMIC (vaddr inside the LOAD's range) is
    // listed BEFORE the PT_LOAD that actually contains its bytes. DYNAMIC's own
    // file_off is the wrong (logical/unmapped) offset; the LOAD's file_off is the
    // real one. Numbers mirror the real dump: LOAD [0x1aeabb0, +0xb42c0) @file
    // 0x19bfe60; DYNAMIC vaddr 0x1b9e350 -> correct file 0x1a73600.
    {
        Module m;
        m.loads.push_back(seg(PT_DYNAMIC, 0x1b9e350, 0xb20, 0x1a724f0)); // wrong self-offset, listed first
        m.loads.push_back(seg(PT_LOAD,    0x1aeabb0, 0xb42c0, 0x19bfe60)); // covering LOAD, listed second
        int64_t f = m.va2foff(0x1b9e350);
        CHECK(f == 0x1a73600, "DYNAMIC-before-LOAD: va2foff=0x%llx expected 0x1a73600 (LOAD-resolved, not self-offset)",
              (unsigned long long)f);
    }

    // Order-independence: same result when the LOAD is listed first (PPSA24651's
    // ordering, which worked before the fix — must still work).
    {
        Module m;
        m.loads.push_back(seg(PT_LOAD,    0x1aeabb0, 0xb42c0, 0x19bfe60));
        m.loads.push_back(seg(PT_DYNAMIC, 0x1b9e350, 0xb20, 0x1a724f0));
        int64_t f = m.va2foff(0x1b9e350);
        CHECK(f == 0x1a73600, "LOAD-before-DYNAMIC: va2foff=0x%llx expected 0x1a73600", (unsigned long long)f);
    }

    // Fallback: a VA that no PT_LOAD covers must still resolve via a non-LOAD
    // file-backed segment (e.g. PROCPARAM sitting outside every LOAD).
    {
        Module m;
        m.loads.push_back(seg(PT_LOAD,        0x1000, 0x1000, 0x8000));
        m.loads.push_back(seg(PT_SCE_PROCPARAM, 0x50000, 0x60, 0x40000));
        int64_t f = m.va2foff(0x50010);
        CHECK(f == 0x40010, "non-LOAD fallback: va2foff=0x%llx expected 0x40010", (unsigned long long)f);
    }

    // Unmapped VA returns -1.
    {
        Module m;
        m.loads.push_back(seg(PT_LOAD, 0x1000, 0x1000, 0x8000));
        CHECK(m.va2foff(0x99999) == -1, "unmapped VA should return -1");
    }

    printf("\n== %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail;
}
