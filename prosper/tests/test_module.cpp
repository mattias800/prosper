// test_module — agentic-first, self-checking loader test against the real dump.
// Exit code 0 = all pass; nonzero = failure count. No human observation needed.
//
// Usage: test_module <path-to-eboot.bin>
#include "../src/self/module.hpp"
#include <cstdio>
#include <cstring>
#include <set>

using namespace prosper;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond, ...) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; printf("  [FAIL] %s:%d  ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } \
} while (0)

static uint64_t read64(const LoadedImage& img, uint64_t va) {
    const uint8_t* p = img.at(va); uint64_t v = 0; if (p) memcpy(&v, p, 8); return v;
}

int main(int argc, char** argv) {
    const char* path = (argc >= 2) ? argv[1] : "../../PPSA24651-app0/eboot.bin";
    printf("== test_module: %s ==\n", path);

    std::string err;
    auto mo = Module::load(path, &err);
    CHECK(mo.has_value(), "Module::load failed: %s", err.c_str());
    if (!mo) { printf("FATAL: cannot load\n"); return 1; }
    Module& m = *mo;

    // --- ELF identity ---
    CHECK(m.e_machine == 0x3e, "machine=0x%x expected x86-64", m.e_machine);
    CHECK(m.e_type == ET_SCE_DYNEXEC, "e_type=0x%x expected DYNEXEC", m.e_type);

    // --- segments ---
    int nload = 0; for (auto& s : m.segments) if (s.type == PT_LOAD) nload++;
    CHECK(nload == 5, "PT_LOAD count=%d expected 5", nload);

    // --- imports (the HLE surface we measured with self_dump) ---
    CHECK(m.imports.size() == 612, "imports=%zu expected 612", m.imports.size());
    std::set<std::string> libs;
    for (auto& im : m.imports) libs.insert(im.lib_name);
    CHECK(libs.size() == 35, "distinct import libs=%zu expected 35", libs.size());
    CHECK(libs.count("libkernel"),   "missing libkernel in imports");
    CHECK(libs.count("libSceAgc"),   "missing libSceAgc in imports");
    CHECK(libs.count("libc"),        "missing libc in imports");
    CHECK(libs.count("libSceVideoOut"), "missing libSceVideoOut in imports");

    // --- relocations parsed ---
    size_t exp_rela = m.rela_sz / 24, exp_plt = m.jmprel_sz / 24;
    CHECK(m.relocs.size() == exp_rela + exp_plt,
          "relocs=%zu expected %zu (%zu rela + %zu plt)", m.relocs.size(), exp_rela + exp_plt, exp_rela, exp_plt);
    CHECK(exp_rela > 10000, "rela count=%zu unexpectedly small", exp_rela);
    CHECK(exp_plt >= 500 && exp_plt <= 700, "plt reloc count=%zu out of expected range", exp_plt);

    // --- build image ---
    const uint64_t BASE = 0x400000;
    LoadedImage img;
    CHECK(build_image(m, BASE, img), "build image");
    CHECK(img.mem.size() >= 0x2000000, "image size=0x%zx smaller than expected", img.mem.size());
    CHECK(img.entry == BASE + m.e_entry, "entry mismatch");

    // entry must land inside an executable segment, and code bytes must be present
    bool entry_in_x = false;
    for (auto& p : img.prot)
        if (p.x && img.entry >= BASE + p.vaddr && img.entry < BASE + p.vaddr + p.size) entry_in_x = true;
    CHECK(entry_in_x, "entry 0x%llx not inside an executable segment", (unsigned long long)img.entry);
    const uint8_t* ecode = img.at(img.entry);
    CHECK(ecode != nullptr, "entry not mapped");
    if (ecode) {
        bool nonzero = false; for (int i = 0; i < 16; i++) if (ecode[i]) nonzero = true;
        CHECK(nonzero, "16 bytes at entry are all zero (code not copied?)");
    }

    // --- bind imports to stub slots, then relocate ---
    const uint64_t STUB_BASE = BASE + 0x40000000, STUB_SZ = 16;
    bind_imports_to_stubs(m, img, STUB_BASE, STUB_SZ);
    CHECK(img.import_addr.size() == m.imports.size(), "import_addr size mismatch");

    size_t applied = apply_relocations(m, img);
    CHECK(applied >= m.relocs.size() * 9 / 10, "applied=%zu of %zu relocs (too few)", applied, m.relocs.size());

    // Spot-check a JUMP_SLOT: its GOT entry must now hold the assigned stub address.
    bool checked_js = false;
    for (auto& r : m.relocs) {
        if (r.type == R_X86_64_JUMP_SLOT && img.import_addr.count(r.sym)) {
            uint64_t got = read64(img, BASE + r.offset);
            CHECK(got == img.import_addr[r.sym],
                  "JUMP_SLOT@0x%llx got=0x%llx expected stub 0x%llx",
                  (unsigned long long)(BASE + r.offset), (unsigned long long)got,
                  (unsigned long long)img.import_addr[r.sym]);
            checked_js = true; break;
        }
    }
    CHECK(checked_js, "no JUMP_SLOT import reloc found to spot-check");

    // Spot-check a RELATIVE: target must hold base + addend.
    bool checked_rel = false;
    for (auto& r : m.relocs) {
        if (r.type == R_X86_64_RELATIVE) {
            uint64_t got = read64(img, BASE + r.offset);
            CHECK(got == BASE + (uint64_t)r.addend,
                  "RELATIVE@0x%llx got=0x%llx expected 0x%llx",
                  (unsigned long long)(BASE + r.offset), (unsigned long long)got,
                  (unsigned long long)(BASE + (uint64_t)r.addend));
            checked_rel = true; break;
        }
    }
    CHECK(checked_rel, "no RELATIVE reloc found to spot-check");

    printf("\n== %d passed, %d failed ==\n", g_pass, g_fail);
    return g_fail;
}
