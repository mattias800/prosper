// imgdump — dump a module's built (unrelocated-base-0) flat image to a file for offline
// disassembly: objdump -D -b binary -m i386:x86-64 <out.img> then grep by plain VA offset.
// Usage: imgdump <eboot.bin|*.prx> <out.img>
#include "self/module.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <module> <out.img>\n", argv[0]); return 1; }
    std::string err;
    auto m = prosper::Module::load(argv[1], &err);
    if (!m) { fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    prosper::LoadedImage img;
    if (!prosper::build_image(*m, 0, img, &err)) {
        fprintf(stderr, "build image failed: %s\n", err.c_str());
        return 1;
    }
    FILE* f = fopen(argv[2], "wb");
    if (!f) { perror("fopen"); return 1; }
    fwrite(img.mem.data(), 1, img.mem.size(), f);
    fclose(f);
    fprintf(stderr, "wrote %zu bytes (min_vaddr=0x%llx max_vaddr=0x%llx entry=0x%llx)\n",
            img.mem.size(), (unsigned long long)img.min_vaddr,
            (unsigned long long)img.max_vaddr, (unsigned long long)img.entry);
    return 0;
}
