// test_sysmodule_state — libSceSysmodule answers its state query from prosper's own load history.
//
// The defect this guards (#2002) is a SKIPPED INITIALIZATION, not a rejection: every module id
// answered SCE_OK ("loaded"), so a guest running the standard
//     if (sceSysmoduleIsLoaded(id) == SCE_SYSMODULE_ERROR_UNLOADED) { LoadModule(id); Initialize(); }
// idiom concluded the setup was already done and silently did less. Grand Theft Auto V loses its
// sceAppContentInitialize + userDefinedParam1 read that way (eboot+0x197a22a, `cmp eax,0x805a1001`).
//
// Every check below is named and fails independently, and each one pins a different part of the
// derivation rather than the observable alone:
//   unqueried-id-is-unloaded   dies if IsLoaded goes back to a constant SCE_OK (the #2002 bug)
//   loaded-id-is-loaded        dies if IsLoaded becomes a constant UNLOADED, or if LoadModule
//                              stops recording the id
//   unloaded-id-is-unloaded    dies if UnloadModule stops clearing the id
//   distinct-ids-independent   dies if the state is a single global flag rather than per-id
//   two-ids-loaded-at-once     dies if the state is a single *slot* — one id at a time
//   unload-leaves-other-id     dies if unloading one id clears another
//   id-key-is-not-byte-wide    dies if the key aliases ids that share a low byte (`id & 0xFF`)
//   load-id-truncated-to-uint16  dies if only ONE of Load/IsLoaded masks the id to its ABI width
//   load-still-succeeds        dies if the honest query answer leaks into the load result
// A constant in EITHER direction therefore fails a named check, which is what separates this from a
// test that would also pass on unfixed master.
//
// `two-ids-loaded-at-once` and `unload-leaves-other-id` exist because of a review finding: without
// them a "remember only the most recently loaded id" implementation (`uint16_t g_last; bool g_have;`)
// satisfies every other check here, INCLUDING `distinct-ids-independent` — which never held two ids
// at the same time and so did not test what its name claims. Holding two simultaneously is the only
// arrangement that separates a per-id map from a single slot.
#include "../src/hle/dispatch.hpp"
#include "../src/hle/nid.hpp"
#include <cstdint>
#include <cstdio>

using namespace prosper;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

static const uint64_t kUnloaded = 0x805A1001;   // SCE_SYSMODULE_ERROR_UNLOADED (GTA V's own compare)

int main() {
    printf("== test_sysmodule_state ==\n");
    register_builtin_hle();

    HleFn is_loaded = Hle::lookup(nid_hash("sceSysmoduleIsLoaded"));
    HleFn load      = Hle::lookup(nid_hash("sceSysmoduleLoadModule"));
    HleFn unload    = Hle::lookup(nid_hash("sceSysmoduleUnloadModule"));
    CHECK(is_loaded && load && unload, "sysmodule-handlers-registered");
    if (!is_loaded || !load || !unload) return 1;

    // 0xB4 is the id GTA V asks about before sceAppContentInitialize; 0x19 is another id it queries.
    const uint64_t kAppContent = 0xB4, kOther = 0x19;

    CHECK(is_loaded(kAppContent, 0, 0, 0, 0, 0) == kUnloaded, "unqueried-id-is-unloaded");

    CHECK(load(kAppContent, 0, 0, 0, 0, 0) == 0, "load-still-succeeds");
    CHECK(is_loaded(kAppContent, 0, 0, 0, 0, 0) == 0, "loaded-id-is-loaded");
    CHECK(is_loaded(kOther, 0, 0, 0, 0, 0) == kUnloaded, "distinct-ids-independent");

    CHECK(unload(kAppContent, 0, 0, 0, 0, 0) == 0, "unload-still-succeeds");
    CHECK(is_loaded(kAppContent, 0, 0, 0, 0, 0) == kUnloaded, "unloaded-id-is-unloaded");

    // Two ids held at the same time — the only arrangement a single-slot implementation fails.
    CHECK(load(kAppContent, 0, 0, 0, 0, 0) == 0 && load(kOther, 0, 0, 0, 0, 0) == 0,
          "second-load-succeeds");
    CHECK(is_loaded(kAppContent, 0, 0, 0, 0, 0) == 0 && is_loaded(kOther, 0, 0, 0, 0, 0) == 0,
          "two-ids-loaded-at-once");
    CHECK(unload(kOther, 0, 0, 0, 0, 0) == 0 && is_loaded(kAppContent, 0, 0, 0, 0, 0) == 0,
          "unload-leaves-other-id");

    // The id is a uint16_t in the Sony prototype; upper bits of the register must not create a
    // second, distinct entry for the same module — on EITHER side. Masking only in IsLoaded, or
    // only in LoadModule, leaves the two disagreeing about which entry an id names.
    CHECK(is_loaded(0xFFFF0000u | kAppContent, 0, 0, 0, 0, 0) == 0, "query-id-truncated-to-uint16");
    CHECK(unload(kAppContent, 0, 0, 0, 0, 0) == 0 &&
          load(0xFFFF0000u | kAppContent, 0, 0, 0, 0, 0) == 0 &&
          is_loaded(kAppContent, 0, 0, 0, 0, 0) == 0, "load-id-truncated-to-uint16");

    // A byte-wide or otherwise aliasing key satisfies every check above, because 0xB4 and 0x19
    // differ in their low byte — so `id & 0xFF` never has to distinguish them. Two ids that SHARE
    // a low byte are the discriminator, and 0x01B4 is a real id width (the corpus queries 0x130).
    const uint64_t kLowByteTwin = 0x01B4;
    CHECK(unload(kAppContent, 0, 0, 0, 0, 0) == 0 && load(kLowByteTwin, 0, 0, 0, 0, 0) == 0,
          "load-low-byte-twin");
    CHECK(is_loaded(kAppContent, 0, 0, 0, 0, 0) == kUnloaded, "id-key-is-not-byte-wide");

    // ===== the *Internal entry points share the same state (#2128) ============================
    // Unregistered, all eight fell to prosper_on_unimpl's `return 0` -- SUCCESS for this contract,
    // recorded nowhere. A title that loaded through one of them and then called IsLoaded was told
    // UNLOADED about a load prosper had just reported as succeeding: #2002's defect on a parallel
    // path, and invisible to every assertion above because those drive only the plain spellings.
    {
        HleFn load_i    = Hle::lookup(nid_hash("sceSysmoduleLoadModuleInternal"));
        HleFn load_arg  = Hle::lookup(nid_hash("sceSysmoduleLoadModuleInternalWithArg"));
        HleFn unload_i  = Hle::lookup(nid_hash("sceSysmoduleUnloadModuleInternal"));
        HleFn unload_a  = Hle::lookup(nid_hash("sceSysmoduleUnloadModuleInternalWithArg"));
        HleFn isload_i  = Hle::lookup(nid_hash("sceSysmoduleIsLoadedInternal"));
        HleFn by_name_l = Hle::lookup(nid_hash("sceSysmoduleLoadModuleByNameInternal"));
        HleFn by_name_u = Hle::lookup(nid_hash("sceSysmoduleUnloadModuleByNameInternal"));
        HleFn get_hand  = Hle::lookup(nid_hash("sceSysmoduleGetModuleHandleInternal"));
        CHECK(load_i && load_arg && unload_i && unload_a && isload_i && by_name_l && by_name_u &&
              get_hand, "all-eight-internal-entry-points-registered");

        if (load_i && isload_i && unload_i && load_arg && unload_a) {
            const uint64_t kInt = 0x0222;
            // CROSS-SPELLING, which is the whole point: load through *Internal, query through the
            // PLAIN one. A per-spelling map would satisfy a same-spelling round trip and still leave
            // the contradiction this issue is about.
            CHECK(load_i(kInt, 0, 0, 0, 0, 0) == 0, "internal-load-succeeds");
            CHECK(is_loaded(kInt, 0, 0, 0, 0, 0) == 0, "internal-load-is-visible-to-plain-IsLoaded");
            CHECK(isload_i(kInt, 0, 0, 0, 0, 0) == 0, "internal-load-is-visible-to-IsLoadedInternal");
            CHECK(unload_i(kInt, 0, 0, 0, 0, 0) == 0 && is_loaded(kInt, 0, 0, 0, 0, 0) == kUnloaded,
                  "internal-unload-is-visible-to-plain-IsLoaded");
            // The WithArg forms carry (args, argp, pRes) after the id; the id is still a0, so they
            // must record identically. Passing junk in the tail is the point of this arm.
            CHECK(load_arg(kInt, 7, 0xDEAD, 0xBEEF, 0, 0) == 0 &&
                  is_loaded(kInt, 0, 0, 0, 0, 0) == 0, "withArg-load-records-on-the-id");
            CHECK(unload_a(kInt, 7, 0xDEAD, 0xBEEF, 0, 0) == 0 &&
                  is_loaded(kInt, 0, 0, 0, 0, 0) == kUnloaded, "withArg-unload-records-on-the-id");
        }

        // The three that CANNOT be served honestly must not answer 0. For ByName the id-keyed map
        // has no key to write; for GetModuleHandle a 0 is a valid handle shape the caller will
        // dereference -- worse than the missing implementation it stands in for.
        if (by_name_l && by_name_u && get_hand) {
            CHECK(by_name_l(0, 0, 0, 0, 0, 0) != 0, "LoadModuleByName-refuses-rather-than-succeeds");
            CHECK(by_name_u(0, 0, 0, 0, 0, 0) != 0, "UnloadModuleByName-refuses-rather-than-succeeds");
            CHECK(get_hand(kAppContent, 0, 0, 0, 0, 0) != 0, "GetModuleHandleInternal-never-answers-0");
        }
    }

    printf(fails ? "== FAILURES: %d ==\n" : "== all checks passed ==\n", fails);
    return fails ? 1 : 0;
}
