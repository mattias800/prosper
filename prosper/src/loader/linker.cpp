#include "linker.hpp"
#include <unordered_map>
#include <cstring>

namespace prosper {

bool link_program(const std::vector<LinkInput>& inputs, uint64_t stub_base,
                  Program& out, std::string* err) {
    auto fail = [&](const std::string& s) { if (err) *err = s; return false; };
    if (inputs.empty()) return fail("no modules to link");
    out.stub_base = stub_base;

    // --- Pass 1: load every module and build its image at its base. ---
    for (auto& in : inputs) {
        std::string e;
        auto mo = Module::load(in.path, &e);
        if (!mo) return fail("load " + in.path + ": " + e);
        auto mod = std::make_unique<Module>(std::move(*mo));
        LoadedImage img = build_image(*mod, in.base);
        out.mods.push_back(std::move(mod));
        out.imgs.push_back(std::move(img));
    }
    out.entry = out.imgs[0].entry;

    // Assign each module with a PT_TLS segment a TLS module id (>=1; 0 = "no TLS"), and record its
    // per-thread template. init_va is the module's mapped tdata (guest base + tls_vaddr). Used by
    // the general-dynamic TLS relocs (DTPMOD64) + __tls_get_addr (see hle_kernel.cpp).
    out.tls_templates.push_back({});   // index 0 reserved / invalid
    for (size_t i = 0; i < out.mods.size(); i++) {
        Module& m = *out.mods[i];
        if (m.tls_memsz == 0) continue;
        out.imgs[i].tls_modid = (uint32_t)out.tls_templates.size();
        out.tls_templates.push_back({ out.imgs[i].base + m.tls_vaddr,
                                      m.tls_filesz, m.tls_memsz, m.tls_align });
    }

    // --- Global export table: NID -> guest address (first definition wins). ---
    std::unordered_map<std::string, uint64_t> exports;
    for (size_t i = 0; i < out.mods.size(); i++) {
        const Module& m = *out.mods[i];
        uint64_t base = out.imgs[i].base;
        for (auto& s : m.symbols)
            if (!s.is_import && !s.nid.empty() && s.value != 0)
                exports.emplace(s.nid, base + s.value);
    }

    // --- Pass 2: resolve every import. Cross-module export beats a stub slot. ---
    std::unordered_map<std::string, uint32_t> nid_to_slot;   // dedupe stub slots by NID
    for (size_t i = 0; i < out.mods.size(); i++) {
        const Module& m = *out.mods[i];
        LoadedImage& img = out.imgs[i];
        for (auto& imp : m.imports) {
            out.total_imports++;
            auto ex = exports.find(imp.nid);
            if (ex != exports.end()) {
                img.import_addr[imp.sym_index] = ex->second;     // real cross-module target
                out.resolved_cross_module++;
                continue;
            }
            auto it = nid_to_slot.find(imp.nid);
            uint32_t slot;
            if (it != nid_to_slot.end()) slot = it->second;
            else {
                slot = (uint32_t)out.slots.size();
                out.slots.push_back({ imp.lib_name, imp.nid });
                nid_to_slot.emplace(imp.nid, slot);
            }
            img.import_addr[imp.sym_index] = out.stub_base + (uint64_t)slot * out.stub_size;
            out.stubbed++;
        }
    }

    // --- Pass 3: apply relocations now that all import addresses are known. ---
    for (size_t i = 0; i < out.mods.size(); i++)
        apply_relocations(*out.mods[i], out.imgs[i]);

    // --- Collect init functions for dependent modules (module 0 = main exe runs its
    // own ctors via _start; PRX modules need their init_array run by the loader). We run
    // them in reverse load order (deepest dependency first). init_array entries were just
    // relocated to absolute addresses, so read them straight from the image. ---
    for (size_t i = out.mods.size(); i-- > 1; ) {
        const Module& m = *out.mods[i];
        LoadedImage& img = out.imgs[i];
        if (m.init_va) out.init_fns.push_back(img.base + m.init_va);
        for (uint64_t off = 0; off + 8 <= m.init_array_sz; off += 8) {
            const uint8_t* p = img.at(img.base + m.init_array_va + off);
            if (!p) break;
            uint64_t fn; memcpy(&fn, p, 8);
            if (fn) out.init_fns.push_back(fn);
        }
    }
    return true;
}

} // namespace prosper
