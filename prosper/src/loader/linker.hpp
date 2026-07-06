// linker.hpp — multi-module dynamic linker. Loads the main executable plus its
// dependent PRX modules into one guest address space, builds a global export table,
// and resolves every import to either another module's export (real cross-module call)
// or an HLE stub slot (implemented handler or unimplemented logger). Host-agnostic.
#pragma once
#include "../self/module.hpp"
#include "../hle/dispatch.hpp"   // ImportSlot
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper {

struct LinkInput { std::string path; uint64_t base; };

// Per-module TLS template for the general-dynamic model (__tls_get_addr). init_va points at the
// module's mapped tdata (guest==host address, since we map at guest base); a per-thread block of
// `memsz` is allocated lazily, `filesz` bytes copied from init_va, the rest zeroed (tbss).
struct TlsTemplate { uint64_t init_va = 0, filesz = 0, memsz = 0, align = 0; };

struct Program {
    std::vector<std::unique_ptr<Module>> mods;   // unique_ptr: stable addresses for imports[]
    std::vector<LoadedImage>             imgs;    // parallel to mods
    std::vector<ImportSlot>              slots;   // unresolved imports -> stub slots
    std::vector<uint64_t>                init_fns; // dependent-module init fns, in call order
    std::vector<TlsTemplate>             tls_templates; // indexed by module TLS id (0 = unused)
    uint64_t entry = 0;                            // main module entry
    uint64_t stub_base = 0, stub_size = 96;   // 96 leaves room for the gated guest-%fs swap stub (~74 bytes)

    // Stats for reporting.
    size_t total_imports = 0, resolved_cross_module = 0, stubbed = 0;
};

// Link the given modules (the first is the main executable). Applies relocations.
// Returns false with *err on failure.
bool link_program(const std::vector<LinkInput>& inputs, uint64_t stub_base,
                  Program& out, std::string* err);

} // namespace prosper
