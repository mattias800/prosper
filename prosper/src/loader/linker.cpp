#include "linker.hpp"
#include "tls_layout.hpp"
#include <unordered_map>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace prosper {

// The one definition of "what this module contributes to the export table". The predicate must stay
// identical to the one used when `out.exports` is built below.
static bool is_exported_symbol(const Symbol& s) {
    return !s.is_import && !s.nid.empty() && s.value != 0;
}

std::vector<std::string> module_export_nids(const Module& m) {
    std::vector<std::string> nids;
    for (const auto& s : m.symbols) if (is_exported_symbol(s)) nids.push_back(s.nid);
    return nids;
}

ExportCollision find_export_collision(
    const Module& m, const std::unordered_map<std::string, std::string>& claimed) {
    for (const auto& s : m.symbols) {
        if (!is_exported_symbol(s)) continue;
        const auto it = claimed.find(s.nid);
        if (it != claimed.end()) return { s.nid, it->second };
    }
    return {};
}

bool link_program(const std::vector<LinkInput>& inputs, uint64_t stub_base,
                  Program& out, std::string* err) {
    auto fail = [&](const std::string& s) { if (err) *err = s; return false; };
    if (inputs.empty()) return fail("no modules to link");
    out.stub_base = stub_base;

    // --- Pass 1: load every module and build its image at its base. ---
    // NID -> path of the accepted module that exports it, in list order. Mirrors the
    // first-definition-wins global export table built in the pass below, so an optional input can be
    // rejected BEFORE its image is built rather than silently aliasing an earlier module's exports.
    std::unordered_map<std::string, std::string> nid_owner;
    for (auto& in : inputs) {
        std::string e;
        auto mo = Module::load(in.path, &e);
        if (!mo) return fail("load " + in.path + ": " + e);
        if (in.skip_on_export_collision) {
            const ExportCollision hit = find_export_collision(*mo, nid_owner);
            if (!hit.nid.empty()) {
                out.skipped_modules.push_back({ in.path, hit.nid, hit.owner_path });
                continue;
            }
        }
        for (const auto& s : mo->symbols)
            if (is_exported_symbol(s)) nid_owner.emplace(s.nid, in.path);
        auto mod = std::make_unique<Module>(std::move(*mo));
        LoadedImage img;
        if (!build_image(*mod, in.base, img, &e))
            return fail("load " + in.path + ": " + e);
        out.mods.push_back(std::move(mod));
        out.imgs.push_back(std::move(img));
    }
    // Only optional inputs can be dropped above, and the main executable is never optional — but do
    // not index into an empty vector if a caller ever flags every input.
    if (out.imgs.empty()) return fail("every module was skipped on an export collision");
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

    // --- Global export table: NID -> guest address (first definition wins). Retained in
    // out.exports so sceKernelDlsym can resolve exported symbols by name post-link. ---
    std::unordered_map<std::string, uint64_t>& exports = out.exports;
    for (size_t i = 0; i < out.mods.size(); i++) {
        const Module& m = *out.mods[i];
        uint64_t base = out.imgs[i].base;
        Program::ModuleExports me; me.path = m.path;
        for (auto& s : m.symbols)
            if (!s.is_import && !s.nid.empty() && s.value != 0) {
                // emplace() is a NO-OP on an existing key, so first-wins — intended, but it was also
                // silent (#1635). Record every alias: which NID, who won, who lost, and both
                // addresses. Behaviour is unchanged; only the reporting is new.
                const auto [it, inserted] = exports.emplace(s.nid, base + s.value);
                if (!inserted && it->second != base + s.value) {
                    const char* winner = "?";
                    for (const auto& prior : out.mod_exports) {
                        const auto f = prior.nids.find(s.nid);
                        if (f != prior.nids.end() && f->second == it->second) {
                            winner = prior.path.c_str(); break;
                        }
                    }
                    out.aliased_exports.push_back({ s.nid, winner, m.path, it->second,
                                                    base + s.value });
                }
                me.nids.emplace(s.nid, base + s.value);   // per-module view for handle-first dlsym (#147)
            }
        out.mod_exports.push_back(std::move(me));
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

    // --- TLS identity by exported symbol NID (#136/#338): a cross-module tls_index needs both the
    // DEFINING module id (DTPMOD64) and the symbol's offset within that module's TLS block (DTPOFF64).
    // Keeping them in one record prevents the pair from resolving through different definitions.
    // Only TLS-bearing modules contribute defined exports. First definition wins, matching exports. ---
    TlsSymbolMap tls_symbols_by_nid;
    for (size_t i = 0; i < out.mods.size(); i++) {
        uint32_t mid = out.imgs[i].tls_modid;
        if (!mid) continue;
        for (auto& s : out.mods[i]->symbols)
            if (!s.is_import && !s.nid.empty())
                tls_symbols_by_nid.emplace(s.nid, TlsSymbolLocation{ mid, s.value });
    }

    // Initial-exec TLS relocations must use the exact Variant-II layout later used to allocate each
    // thread's static TLS. Computing it through the shared helper prevents linker/allocator drift.
    const StaticTlsLayout static_tls =
        make_static_tls_layout(out.tls_templates.data(), out.tls_templates.size());

    // --- Pass 3: apply relocations now that all import addresses are known. ---
    for (size_t i = 0; i < out.mods.size(); i++)
        apply_relocations(*out.mods[i], out.imgs[i], &tls_symbols_by_nid,
                          &static_tls.module_below);

    // --- Collect init functions for dependent modules (module 0 = main exe runs its
    // own ctors via _start; PRX modules need their init_array run by the loader). We run
    // them in reverse load order (deepest dependency first). init_array entries were just
    // relocated to absolute addresses, so read them straight from the image. ---
    if (getenv("PROSPER_INITLOG")) {
        for (size_t i = 0; i < out.mods.size(); i++) {
            const Module& m = *out.mods[i];
            fprintf(stderr, "[initlog] module %zu base=0x%llx init_va=0x%llx init_array_va=0x%llx entries=%llu path=%s\n",
                    i, (unsigned long long)out.imgs[i].base, (unsigned long long)m.init_va,
                    (unsigned long long)m.init_array_va, (unsigned long long)(m.init_array_sz / 8),
                    m.path.c_str());
        }
    }
    for (size_t i = out.mods.size(); i-- > 1; ) {
        const Module& m = *out.mods[i];
        LoadedImage& img = out.imgs[i];
        if (m.init_va) out.init_fns.push_back(img.base + m.init_va);
        for (uint64_t off = 0; off + 8 <= m.init_array_sz; off += 8) {
            const uint8_t* p = img.at(img.base + m.init_array_va + off);
            // at() only checks the start va is in-image; guarantee all 8 bytes are too (the sibling
            // write64 in apply_relocations has the same p+8 guard) so a DT_INIT_ARRAY landing in the
            // final <8 bytes of a malformed image cannot read past mem.
            if (!p || p + 8 > img.mem.data() + img.mem.size()) break;
            uint64_t fn; memcpy(&fn, p, 8);
            if (fn) out.init_fns.push_back(fn);
        }
    }
    return true;
}

} // namespace prosper
