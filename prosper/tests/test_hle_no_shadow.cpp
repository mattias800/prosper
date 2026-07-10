// test_hle_no_shadow — guards the whole double-registration-shadow class (#330).
//
// Hle::register_fn is last-write-wins with no warning: register a NID in two files and whichever
// register_*() runs LAST silently takes it. When the loser was a real implementation and the winner
// a naive stub, the guest gets the stub (e.g. scePthreadGetprio returning OK without writing its
// out-param). register_fn now records every collision where the handler CHANGED; this test asserts
// there are none left. If an intentional override is ever added, list its NID in kAllowed with a
// justification — silence here is the healthy state.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdio>
#include <string>
#include <unordered_map>

using namespace prosper;

static uint64_t dummy_a(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) { return 1; }
static uint64_t dummy_b(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) { return 2; }

int main() {
    printf("== test_hle_no_shadow ==\n");
    // Reverse map: NID -> readable name, from the built-in symbol list (for diagnostics only).
    std::unordered_map<std::string, std::string> nid2name;
    for (const auto& n : builtin_symbol_names()) nid2name.emplace(nid_hash(n), n);
    auto name_of = [&](const std::string& nid) -> std::string {
        auto it = nid2name.find(nid); return it == nid2name.end() ? std::string("?") : it->second;
    };
    register_builtin_hle();

    const auto& shadows = Hle::shadowed_registrations();
    // Known-intentional overrides (none today). Format: NID that a later registration deliberately
    // replaces with a better handler. Keep this empty unless there's a documented reason.
    static const char* kAllowed[] = { nullptr };

    int unexpected = 0;
    for (const auto& s : shadows) {
        bool allowed = false;
        for (const char** a = kAllowed; *a; ++a) if (s.nid == *a) { allowed = true; break; }
        printf("  %s NID=%s (%s) : \"%s\" was overwritten by \"%s\"\n",
               allowed ? "[allowed]" : "[SHADOW]", s.nid.c_str(), name_of(s.nid).c_str(),
               s.prev_name.c_str(), s.new_name.c_str());
        if (!allowed) unexpected++;
    }
    if (shadows.empty()) printf("  (no NID registered twice with differing handlers)\n");

    if (unexpected) {
        printf("== FAIL: %d unexpected shadowed registration(s) — a later register_*() silently\n"
               "   replaced a handler. If the WINNER is a naive stub shadowing a real impl, that is\n"
               "   the #330 bug (getter returns OK without writing its out-param). Remove the duplicate\n"
               "   (leave the function in one file), or allowlist it in kAllowed with a reason. ==\n",
               unexpected);
        return 1;
    }

    // Self-validation: prove the detector actually fires, so a clean list above means "no shadow",
    // not "detector broken". A real->real overwrite IS recorded; a placeholder->real override is NOT.
    size_t before = Hle::shadowed_registrations().size();
    Hle::register_fn("test.dummy.shadow", (HleFn)dummy_a, "dummy_a");
    Hle::register_fn("test.dummy.shadow", (HleFn)dummy_b, "dummy_b");   // real->real, different fn
    bool caught = Hle::shadowed_registrations().size() == before + 1;
    printf("  [%s] mechanism: overwriting a real handler with a different fn is recorded\n", caught ? "ok" : "FAIL");
    Hle::register_placeholder("test.dummy.ph", (HleFn)dummy_a, "ph");
    size_t mid = Hle::shadowed_registrations().size();
    Hle::register_fn("test.dummy.ph", (HleFn)dummy_b, "real");          // placeholder->real override
    bool ignored = Hle::shadowed_registrations().size() == mid;
    printf("  [%s] mechanism: overriding a placeholder is NOT recorded\n", ignored ? "ok" : "FAIL");
    if (!caught || !ignored) { printf("== FAIL: shadow detector is not working ==\n"); return 1; }

    printf("== PASS ==\n");
    return 0;
}
