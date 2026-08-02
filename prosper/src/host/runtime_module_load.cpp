// runtime_module_load.cpp — see runtime_module_load.hpp (#639).
#include "host/runtime_module_load.hpp"

#if defined(__linux__) || defined(__APPLE__) || defined(_WIN32)

#include "host/boot_program.hpp"
#include "host/exec_image.hpp"
#include "loader/linker.hpp"
#include "hle/dispatch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper {
namespace {

constexpr uint64_t kEnoent  = 0x80020002ull;   // SCE_KERNEL_ERROR_ENOENT
constexpr uint64_t kEnoexec = 0x80020008ull;   // SCE_KERNEL_ERROR_ENOEXEC
constexpr uint64_t kEnomem  = 0x8002000cull;   // SCE_KERNEL_ERROR_ENOMEM
constexpr uint64_t kEinval  = 0x80020016ull;   // SCE_KERNEL_ERROR_EINVAL

// One runtime-loaded module. Held in a deque and never erased: its export map and its name are
// published to the HLE (sceKernelDlsym / the unwinder) as raw pointers that must stay valid for
// the rest of the run, and unload is not modelled.
struct RuntimeModule {
    std::string host_path, guest_path, name;
    std::unique_ptr<Module> mod;
    uint64_t base = 0, lo = 0, hi = 0, handle = 0;
    std::unordered_map<std::string, uint64_t> nids;   // this module's own exports (#147)
};

// RECURSIVE, and that is load-bearing: the lock is held across the module's own module_start /
// init_array, which is guest code, and a module's module_start calling sceKernelLoadStartModule for
// a dependency is both legal on PS5 and exactly what a "shell eboot + per-stage PRX" title invites.
// A plain std::mutex would self-deadlock there — silently, which is strictly worse than the ENOENT
// this replaced. Re-entrancy is safe because every published pointer stays valid across a nested
// load: g_loaded is a std::deque (push_back never invalidates a reference to an existing element),
// so the outer frame's `rm` survives, and a nested load can only ever pop its OWN element.
std::recursive_mutex g_mx;       // serialises the whole load; also guards everything below
Program* g_prog = nullptr;
std::deque<RuntimeModule> g_loaded;
std::unordered_map<std::string, uint32_t> g_nid_to_slot;   // NID -> import stub-slot index
TlsSymbolMap g_tls_symbols;      // exported TLS symbol NID -> {defining module id, in-block offset}
unsigned g_next_base_slot = 0;

bool modlog() { static const int v = getenv("PROSPER_MODLOG") ? 1 : 0; return v != 0; }

const char* basename_of(const std::string& p) {
    const char* b = p.c_str();
    for (const char* c = b; *c; c++) if (*c == '/' || *c == '\\') b = c + 1;
    return b;
}

// #639 runs guest code (module_start / init_array) from inside an HLE handler. Under the guest-%fs
// gate the handler is on the HOST %fs, so the module's own initialisation would read host TLS as
// guest TLS. Restore the guest thread pointer for exactly the duration of the guest call, the same
// way the NetCtl/Np callback delivery does (hle_service.cpp). No-op when guest_fs is 0.
#if !defined(_WIN32) && !defined(__APPLE__)
inline uint64_t rd_fsbase() { uint64_t v; __asm__ volatile("rdfsbase %0" : "=r"(v)); return v; }
inline void     wr_fsbase(uint64_t v) { __asm__ volatile("wrfsbase %0" : : "r"(v)); }
struct GuestFsScope {
    uint64_t saved = 0, active = 0;
    explicit GuestFsScope(uint64_t guest_fs) {
        if (guest_fs) { saved = rd_fsbase(); wr_fsbase(guest_fs); active = guest_fs; }
    }
    ~GuestFsScope() { if (active) wr_fsbase(saved); }
};
#else
struct GuestFsScope { explicit GuestFsScope(uint64_t) {} };
#endif

// Call one module entry with the PS5 module-entry ABI, module_start(size_t argc, const void* argp).
// Plain init_array ctors take no arguments and ignore rdi/rsi; a real module_start reads both, so
// passing the guest's own (args, argp) through is what the module was asked to start with.
uint64_t call_module_entry(uint64_t fn, uint64_t args, uint64_t argp, uint64_t guest_fs) {
    GuestFsScope fs(guest_fs);
    return ((uint64_t (*)(uint64_t, uint64_t))(uintptr_t)fn)(args, argp);
}

} // namespace

void runtime_module_loader_init(Program* p) {
    std::lock_guard<std::recursive_mutex> lk(g_mx);
    // Reset EVERY piece of loader state, not just the tables rebuilt below: a second boot_program in
    // one process would otherwise inherit the previous program's loaded modules and hand out their
    // stale handles. (No caller does that today; making the reset total is cheaper than the note.)
    g_prog = p;
    g_loaded.clear();
    g_next_base_slot = 0;
    g_nid_to_slot.clear();
    g_tls_symbols.clear();
    if (!p) return;
    // The linker deduped stub slots by NID but kept that map local to link_program. Rebuild it so a
    // runtime module's import of an already-stubbed NID reuses the SAME slot: a second slot for the
    // same NID would double-count it in the unimplemented-call census and emit a duplicate stub.
    for (size_t i = 0; i < p->slots.size(); i++) g_nid_to_slot.emplace(p->slots[i].nid, (uint32_t)i);
    // Cross-module general-dynamic TLS: same construction as link_program's, so a runtime module's
    // DTPMOD64/DTPOFF64 pair against a pre-linked module resolves through one record.
    for (size_t i = 0; i < p->mods.size() && i < p->imgs.size(); i++) {
        const uint32_t mid = p->imgs[i].tls_modid;
        if (!mid) continue;
        for (const auto& s : p->mods[i]->symbols)
            if (!s.is_import && !s.nid.empty())
                g_tls_symbols.emplace(s.nid, TlsSymbolLocation{ mid, s.value });
    }
}

size_t runtime_loaded_module_count() {
    std::lock_guard<std::recursive_mutex> lk(g_mx);
    return g_loaded.size();
}

uint64_t runtime_load_start_module(const char* guest_path, uint64_t args, uint64_t argp,
                                   uint64_t guest_fs, int32_t* out_res, uint64_t* out_handle) {
    if (!guest_path || !*guest_path) return kEinval;
    std::lock_guard<std::recursive_mutex> lk(g_mx);
    if (!g_prog) {
        // No booted program to load against (a unit test, or a caller that never ran
        // boot_program). Keep #146's honest answer for a path that is not in the linked set rather
        // than inventing a new error the guest has never seen from this call.
        static bool warned = false;
        if (!warned) { warned = true;
            fprintf(stderr, "[loadmod] runtime module loading is not initialised; '%s' -> ENOENT\n",
                    guest_path); }
        return kEnoent;
    }

    // Repeat load of a path this loader already served: hand back the same handle. (The caller's
    // module_handle_for_path check normally catches this first — it matches on basename, which is
    // the PS5's own module identity — but a differently spelled path for the same file must not
    // map the module twice.)
    for (const auto& m : g_loaded)
        if (m.guest_path == guest_path) {
            if (out_handle) *out_handle = m.handle;
            if (out_res) *out_res = 0;
            return 0;
        }

    const std::string host_path = resolve_guest_path(guest_path);
    if (host_path.empty() || host_path.compare(0, 16, "/prosper-denied/") == 0) return kEnoent;
    if (FILE* f = fopen(host_path.c_str(), "rb")) fclose(f);
    else {
        if (modlog())
            fprintf(stderr, "[loadmod] '%s' -> '%s' does not exist -> ENOENT\n",
                    guest_path, host_path.c_str());
        return kEnoent;
    }

    std::string e;
    auto parsed = Module::load(host_path, &e);
    if (!parsed) {
        fprintf(stderr, "[loadmod] '%s': cannot parse module (%s) -> ENOEXEC\n",
                guest_path, e.c_str());
        return kEnoexec;
    }

    if (g_next_base_slot >= BOOT_RUNTIME_MODULE_SLOTS) {
        fprintf(stderr, "[loadmod] '%s': no free runtime module base slot (max %u, #639) -> ENOMEM\n",
                guest_path, BOOT_RUNTIME_MODULE_SLOTS);
        return kEnomem;
    }
    // Claim the base slot NOW, before anything below can fail. A failure path that left the slot
    // free would let the next module map at the same base while this one's rolled-back-but-not-
    // reverted state (a TLS template whose init_va points into that memory) still named it.
    // Burning a slot on a failed load is the cheap, safe side of that trade.
    const uint64_t base =
        BOOT_RUNTIME_MODULE_BASE + (uint64_t)g_next_base_slot * BOOT_RUNTIME_MODULE_STRIDE;
    g_next_base_slot++;

    g_loaded.emplace_back();
    RuntimeModule& rm = g_loaded.back();
    // Publish nothing until the module is fully live. On any failure below, undo in reverse order:
    // pop the module, drop any TLS template it appended (re-publishing the shorter list is safe —
    // nothing was relocated against that module id), and forget any NID it claimed a stub slot for.
    const size_t tls_templates_before = g_prog->tls_templates.size();
    std::vector<std::string> claimed_nids;
    auto abandon = [&](uint64_t err) {
        g_loaded.pop_back();
        if (g_prog->tls_templates.size() > tls_templates_before) {
            g_prog->tls_templates.resize(tls_templates_before);
            set_tls_modules(g_prog->tls_templates.data(), g_prog->tls_templates.size());
        }
        for (const auto& nid : claimed_nids) g_nid_to_slot.erase(nid);
        return err;
    };

    rm.host_path = host_path;
    rm.guest_path = guest_path;
    rm.name = basename_of(host_path);
    rm.mod = std::make_unique<Module>(std::move(*parsed));
    rm.base = base;

    LoadedImage img;
    if (!build_image(*rm.mod, base, img, &e)) {
        fprintf(stderr, "[loadmod] '%s': %s -> ENOEXEC\n", guest_path, e.c_str());
        return abandon(kEnoexec);
    }
    // The module occupies [base + min_vaddr, base + max_vaddr), so max_vaddr — not the image byte
    // count — is what has to fit the slot: a nonzero min_vaddr would otherwise pass this and still
    // run into the next slot.
    if (img.max_vaddr > BOOT_RUNTIME_MODULE_STRIDE) {
        fprintf(stderr,
                "[loadmod] '%s': image spans 0x%llx bytes of VA, larger than the 0x%llx runtime "
                "module slot (#639) -> ENOMEM\n", guest_path, (unsigned long long)img.max_vaddr,
                (unsigned long long)BOOT_RUNTIME_MODULE_STRIDE);
        return abandon(kEnomem);
    }

    // --- TLS. A module with a real PT_TLS gets a NEW general-dynamic module id; static (initial-
    // exec) TLS cannot be extended once threads exist, so TPOFF64 is reported and left unapplied
    // by apply_relocations rather than baked to a wrong offset. ---
    if (rm.mod->tls_memsz) {
        img.tls_modid = (uint32_t)g_prog->tls_templates.size();
        g_prog->tls_templates.push_back({ base + rm.mod->tls_vaddr, rm.mod->tls_filesz,
                                          rm.mod->tls_memsz, rm.mod->tls_align });
        set_tls_modules(g_prog->tls_templates.data(), g_prog->tls_templates.size());
        for (const auto& s : rm.mod->symbols)
            if (!s.is_import && !s.nid.empty())
                g_tls_symbols.emplace(s.nid, TlsSymbolLocation{ img.tls_modid, s.value });
    }
    size_t tpoff = 0;
    for (const auto& r : rm.mod->relocs) if (r.type == R_X86_64_TPOFF64) tpoff++;
    if (tpoff)
        fprintf(stderr,
                "[loadmod] '%s': %zu initial-exec TLS relocations (R_X86_64_TPOFF64) CANNOT be "
                "applied to a module loaded after the guest's threads exist; they are left "
                "unapplied (#639). Report this module.\n", guest_path, tpoff);

    // --- Imports: an export of an already-loaded module beats a stub slot, exactly as at boot. ---
    const size_t first_new_slot = g_prog->slots.size();
    std::vector<ImportSlot> new_slots;
    size_t cross = 0, stubbed = 0;
    for (const auto& imp : rm.mod->imports) {
        auto ex = g_prog->exports.find(imp.nid);
        if (ex != g_prog->exports.end()) { img.import_addr[imp.sym_index] = ex->second; cross++; continue; }
        bool from_runtime = false;
        for (const auto& other : g_loaded) {
            if (&other == &rm) continue;
            auto it = other.nids.find(imp.nid);
            if (it != other.nids.end()) {
                img.import_addr[imp.sym_index] = it->second; cross++; from_runtime = true; break;
            }
        }
        if (from_runtime) continue;
        auto slot = g_nid_to_slot.find(imp.nid);
        if (slot == g_nid_to_slot.end()) {
            // Slot indices are assigned now and the vector is grown in ONE synchronised append
            // below, so a concurrent unimplemented-import call never sees a moving buffer.
            const uint32_t idx = (uint32_t)(first_new_slot + new_slots.size());
            new_slots.push_back({ imp.lib_name, imp.nid });
            slot = g_nid_to_slot.emplace(imp.nid, idx).first;
            claimed_nids.push_back(imp.nid);   // rolled back by abandon() if the load fails
        }
        img.import_addr[imp.sym_index] = g_prog->stub_base + (uint64_t)slot->second * g_prog->stub_size;
        stubbed++;
    }
    const size_t appended_at = dispatch_append_slots(&g_prog->slots, new_slots);
    (void)appended_at;   // == first_new_slot; append_stubs re-checks it

    // Emit the stubs BEFORE the relocations point guest code at them. A failure here leaves the
    // appended slots in place: they are unreferenced (nothing has been relocated to them yet) and
    // removing them would need the same lock the dispatcher reads them under.
    if (!append_stubs(g_prog->slots, first_new_slot, &e)) {
        fprintf(stderr, "[loadmod] '%s': %s -> ENOMEM\n", guest_path, e.c_str());
        return abandon(kEnomem);
    }

    apply_relocations(*rm.mod, img, &g_tls_symbols, nullptr);

    if (!map_image(img, &e)) {
        fprintf(stderr, "[loadmod] '%s': %s -> ENOMEM\n", guest_path, e.c_str());
        return abandon(kEnomem);
    }
    rm.lo = base + img.min_vaddr;
    rm.hi = base + img.max_vaddr;

    // --- Exports: this module's own NID -> address table, for handle-first sceKernelDlsym (#147).
    // It is NOT merged into the program's global first-definition-wins table: that table is what
    // the ALREADY-RELOCATED modules were bound against, and adding to it now cannot change any of
    // those bindings while it could change what a later name-only dlsym resolves to. ---
    for (const auto& s : rm.mod->symbols)
        if (!s.is_import && !s.nid.empty() && s.value != 0) rm.nids.emplace(s.nid, base + s.value);

    // Unwind descriptor and export table must land on the SAME index: the handle
    // sceKernelGetModuleInfoFromAddr derives from the unwind order is consumed by sceKernelDlsym
    // against the export order (hle_kernel.cpp).
    UnwindModuleDesc ud;
    ud.lo = rm.lo; ud.hi = rm.hi;
    for (const auto& s : rm.mod->segments)
        if (s.type == 0x6474e550u) { ud.ehframe_hdr = base + s.vaddr; ud.ehframe_hdr_sz = s.memsz; }
    if (!rm.mod->loads.empty()) { ud.seg0 = base + rm.mod->loads[0].vaddr; ud.seg0_sz = rm.mod->loads[0].memsz; }
    ud.name = rm.name.c_str();
    const size_t unwind_index = add_unwind_module(ud);
    rm.handle = add_module_export_table(rm.host_path, &rm.nids);
    if (rm.handle - kSceModuleHandleBase != unwind_index) {
        // Cannot happen while both appends are serialised by g_mx and nothing else appends after
        // boot — but if it ever did, every address-derived handle from here on would resolve
        // against a different module than it names. Refuse the load rather than continue with one.
        fprintf(stderr,
                "[loadmod] '%s': INTERNAL — unwind index %zu != export index %llu; an "
                "address-derived module handle would name a different module (#639) -> ENOEXEC\n",
                guest_path, unwind_index, (unsigned long long)(rm.handle - kSceModuleHandleBase));
        return abandon(kEnoexec);
    }

    fprintf(stderr,
            "[loadmod] loaded '%s' -> %s @ 0x%llx (%zu exports, %zu imports: %zu cross-module, "
            "%zu stubbed, %zu new slots) handle=0x%llx\n",
            guest_path, rm.name.c_str(), (unsigned long long)base, rm.nids.size(),
            rm.mod->imports.size(), cross, stubbed, g_prog->slots.size() - first_new_slot,
            (unsigned long long)rm.handle);

    // --- Start it. DT_INIT (module_start) first, then DT_INIT_ARRAY, matching the order
    // link_program + run_guest_inits use for a pre-linked dependent module. The init_array entries
    // were relocated to absolute addresses above, so read them from the mapped image. ---
    uint64_t res = 0;
    const std::string guest_path_copy = rm.guest_path;   // rm may be referenced across the calls below
    const uint64_t init_va = rm.mod->init_va, ia_va = rm.mod->init_array_va, ia_sz = rm.mod->init_array_sz;
    if (init_va) res = call_module_entry(base + init_va, args, argp, guest_fs);
    for (uint64_t off = 0; off + 8 <= ia_sz; off += 8) {
        const uint64_t slot_va = base + ia_va + off;
        if (slot_va < rm.lo || slot_va + 8 > rm.hi) break;
        uint64_t fn = 0; memcpy(&fn, (const void*)(uintptr_t)slot_va, 8);
        if (fn) call_module_entry(fn, 0, 0, guest_fs);
    }
    if (modlog())
        fprintf(stderr, "[loadmod] started '%s' module_start(0x%llx, 0x%llx) -> 0x%llx\n",
                guest_path_copy.c_str(), (unsigned long long)args, (unsigned long long)argp,
                (unsigned long long)res);

    if (out_res) *out_res = (int32_t)res;
    if (out_handle) *out_handle = rm.handle;
    return 0;
}

} // namespace prosper

#else   // no guest-execution substrate on this platform

namespace prosper {
void runtime_module_loader_init(Program*) {}
uint64_t runtime_load_start_module(const char*, uint64_t, uint64_t, uint64_t, int32_t*, uint64_t*) {
    return 0x80020002ull;   // SCE_KERNEL_ERROR_ENOENT
}
size_t runtime_loaded_module_count() { return 0; }
} // namespace prosper

#endif
