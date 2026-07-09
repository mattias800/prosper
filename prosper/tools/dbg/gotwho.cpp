// gotwho — map a GOT slot vaddr (eboot-relative) to the import symbol (NID + lib) it binds.
// Usage: gotwho <eboot.bin> <got_vaddr_hex>...
#include "../../src/self/module.hpp"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <eboot> <got_va_hex>...\n", argv[0]); return 1; }
    std::string err;
    auto m = prosper::Module::load(argv[1], &err);
    if (!m) { fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    for (int i = 2; i < argc; i++) {
        uint64_t va = strtoull(argv[i], nullptr, 16);
        bool found = false;
        for (const auto& r : m->relocs) {
            if (r.offset != va) continue;
            found = true;
            if (r.sym && r.sym < m->symbols.size()) {
                const auto& s = m->symbols[r.sym];
                printf("0x%llx: type=%u plt=%d sym=%u nid=%s lib=%s\n",
                       (unsigned long long)va, r.type, (int)r.is_plt, r.sym,
                       s.nid.c_str(), s.lib_name.c_str());
            } else {
                printf("0x%llx: type=%u plt=%d sym=%u (no symbol)\n",
                       (unsigned long long)va, r.type, (int)r.is_plt, r.sym);
            }
        }
        if (!found) printf("0x%llx: no reloc found\n", (unsigned long long)va);
    }
    return 0;
}
