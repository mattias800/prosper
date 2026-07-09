// nid2got — map an import NID to its GOT slot vaddr(s) so callers can be found in the text.
// Usage: nid2got <eboot.bin> <nid>...
#include "../../src/self/module.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <eboot> <nid>...\n", argv[0]); return 1; }
    std::string err;
    auto m = prosper::Module::load(argv[1], &err);
    if (!m) { fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    for (int i = 2; i < argc; i++) {
        bool found = false;
        for (const auto& r : m->relocs) {
            if (!r.sym || r.sym >= m->symbols.size()) continue;
            const auto& s = m->symbols[r.sym];
            if (s.nid != argv[i]) continue;
            printf("%s (%s): got=0x%llx type=%u plt=%d sym=%u\n", argv[i], s.lib_name.c_str(),
                   (unsigned long long)r.offset, r.type, (int)r.is_plt, r.sym);
            found = true;
        }
        if (!found) printf("%s: not found\n", argv[i]);
    }
    return 0;
}
