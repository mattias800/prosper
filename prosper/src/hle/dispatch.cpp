#include "dispatch.hpp"
#include <unordered_map>
#include <mutex>

namespace prosper {

namespace {
    // `placeholder` marks a deliberately-overridable registration (e.g. hle_graphics' per-NID glog
    // tracing thunks, which a real handler in hle_agc is EXPECTED to replace later). Overwriting a
    // placeholder is intended and not flagged; overwriting a REAL handler with a different fn is the
    // accidental-shadow class (#330) that register_fn records.
    struct Reg { HleFn fn; std::string name; bool placeholder = false; };
    std::unordered_map<std::string, Reg>& registry() {
        static std::unordered_map<std::string, Reg> r; return r;
    }
    const std::vector<ImportSlot>* g_slots = nullptr;
    NidDb*        g_db  = nullptr;

    // unimplemented-call bookkeeping
    std::vector<uint32_t>            g_order;      // first-seen import indices
    // Call count per import index. A pre-sized VECTOR (not an unordered_map): indexed directly by idx, it
    // never hashes or rehash-allocates on the hot path. The map form crashed inside prosper_on_unimpl on a
    // worker thread (a rehash allocation racing under heavy concurrent unimpl traffic during live render);
    // a fixed vector eliminates that whole class. Sized once from the import table (dispatch_init).
    std::vector<uint64_t>            g_count;      // index -> call count (index-addressed, no alloc after init)
    volatile int*                    g_progress = nullptr;
    // Guards g_order/g_count/g_progress against concurrent worker-thread unimpl calls (many under
    // -force-gfx-direct/guest-fs). The lock is cheap; the state changes are tiny.
    std::mutex                       g_unimpl_mx;
}

void dispatch_set_progress(volatile int* counter) { g_progress = counter; }

void (*g_hwwatch_hook)(uint64_t) = nullptr;   // set by the Linux exec harness (perf HW watchpoints)

namespace { std::vector<ShadowedReg>& shadow_list() { static std::vector<ShadowedReg> s; return s; } }

void Hle::register_fn(const std::string& nid, HleFn fn, const char* name) {
    auto& reg = registry();
    auto it = reg.find(nid);
    // A second registration of the same NID with a DIFFERENT handler silently shadows the first
    // (last-write-wins). Record it so a test can flag the real-impl-shadowed-by-stub class (#330) —
    // UNLESS the existing entry is a deliberately-overridable placeholder (that override is intended).
    // A re-registration with the SAME fn (harmless dedup) is not recorded.
    if (it != reg.end() && it->second.fn != fn && !it->second.placeholder)
        shadow_list().push_back({ nid, it->second.name, name ? name : "" });
    reg[nid] = { fn, name ? name : "", false };
}
void Hle::register_placeholder(const std::string& nid, HleFn fn, const char* name) {
    // Same as register_fn but marks the entry overridable, so a later real handler replacing it is
    // not flagged as an accidental shadow. Does NOT itself overwrite a real (non-placeholder) entry
    // silently: if one already exists with a different fn, that's still recorded.
    auto& reg = registry();
    auto it = reg.find(nid);
    if (it != reg.end() && it->second.fn != fn && !it->second.placeholder)
        shadow_list().push_back({ nid, it->second.name, name ? name : "" });
    reg[nid] = { fn, name ? name : "", true };
}
const std::vector<ShadowedReg>& Hle::shadowed_registrations() { return shadow_list(); }
HleFn Hle::lookup(const std::string& nid) {
    auto it = registry().find(nid);
    return it == registry().end() ? nullptr : it->second.fn;
}
const char* Hle::name_of(const std::string& nid) {
    auto it = registry().find(nid);
    return it == registry().end() ? "" : it->second.name.c_str();
}

void dispatch_init(const std::vector<ImportSlot>* slots, NidDb* db) {
    g_slots = slots; g_db = db;
    // Pre-size the call-count vector to the import table so prosper_on_unimpl never allocates.
    if (slots) { std::lock_guard<std::mutex> lk(g_unimpl_mx); g_count.assign(slots->size(), 0); }
}
void reset_call_log() { std::lock_guard<std::mutex> lk(g_unimpl_mx); g_order.clear();
                        for (auto& c : g_count) c = 0; }
const std::vector<uint32_t>& call_order() { return g_order; }

extern "C" uint64_t prosper_on_unimpl(uint64_t import_index) {
    uint32_t idx = (uint32_t)import_index;
    std::lock_guard<std::mutex> lk(g_unimpl_mx);   // serialize concurrent worker-thread unimpl calls
    if (idx >= g_count.size()) return 0;           // out-of-range import index: ignore (no alloc, no fault)
    uint64_t c = ++g_count[idx];
    if (c == 1) {
        g_order.push_back(idx);
        if (g_progress) (*g_progress)++;
        if (g_slots && idx < g_slots->size()) {
            const auto& im = (*g_slots)[idx];
            std::string nm = g_db ? g_db->resolve(im.nid) : std::string();
            fprintf(stderr, "[prosper] unimplemented: %s::%s%s%s%s  -> returning 0\n",
                    im.lib.c_str(), im.nid.c_str(),
                    nm.empty() ? "" : " [", nm.c_str(), nm.empty() ? "" : "]");
        }
    }
    return 0; // let the guest proceed so we can discover the next call
}

void dump_call_log(FILE* f) {
    fprintf(f, "\n=== unimplemented Sony calls (first-seen order) ===\n");
    for (uint32_t idx : g_order) {
        const char* lib = "?"; std::string nid, nm;
        if (g_slots && idx < g_slots->size()) {
            lib = (*g_slots)[idx].lib.c_str();
            nid = (*g_slots)[idx].nid;
            if (g_db) nm = g_db->resolve(nid);
        }
        fprintf(f, "  %4u x  %-24s %s%s%s%s\n", (unsigned)g_count[idx], lib,
                nid.c_str(), nm.empty() ? "" : "  [", nm.c_str(), nm.empty() ? "" : "]");
    }
    fprintf(f, "  (%zu distinct unimplemented functions)\n", g_order.size());
}

} // namespace prosper
