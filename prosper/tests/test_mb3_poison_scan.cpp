// test_mb3_poison_scan — the interior free-chain poison scan (#1945).
//
// Two properties, both of which this family has been burned by the absence of:
//
//   1. A `found == 0` must be distinguishable from a scan that walked nothing. The instrument this
//      one replaces (`PROSPER_MB3WATCH`) armed on a stale address window and therefore printed
//      nothing on every current title, which reads exactly like "armed and saw no corruption".
//      So the census is asserted, not just the verdict.
//   2. The poison predicate is STRUCTURAL, not a value shape. A next-link that is neither 0 nor a
//      0x20-aligned pointer into mapped guest memory must be found whatever it looks like — the
//      live values on PPSA07809 were 0x30016000, 0xff000000ff000000 and 0x0002400100024001, and a
//      byte-shift test sees only the first.
//
// The counter-arm is the same chain with the poison removed: it must report zero while the census
// shows the same nodes were walked, so a pass cannot come from a scan that silently did nothing.

#include "../src/gpu/mb3_freelist.hpp"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <sys/mman.h>

using namespace prosper::gpu;

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", what); failures++; }
    else       printf("ok: %s\n", what);
}

// The scan only considers blocks inside the guest window [0x20_0000_0000, 0x40_0000_0000), so the
// fake pool has to live there. A fixed mapping in that range is what the emulator itself uses for
// guest memory; if it is unavailable the test cannot run and says so rather than passing vacuously.
static void* map_fixed(uint64_t at, size_t len) {
#ifdef MAP_FIXED_NOREPLACE
    void* p = mmap((void*)(uintptr_t)at, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
#else
    void* p = mmap((void*)(uintptr_t)at, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
    if (p == MAP_FAILED || (uint64_t)(uintptr_t)p != at) return nullptr;
    return p;
}

int main() {
    // Property 1: an empty registry must announce that it walked nothing.
    mb3_reset_pool_candidates_for_test();
    char census[256] = {};
    int found = mb3_poison_scan(nullptr, 0, 64, census, sizeof census);
    check(found == 0, "empty registry finds no poison");
    check(strstr(census, "WALKED NOTHING") != nullptr,
          "empty registry census says the zero is void, not negative");

    const uint64_t kPool = 0x2ab0000000ull;      // 64 KiB-aligned, inside the guest window
    const size_t   kLen  = 0x10000;
    void* mem = map_fixed(kPool, kLen);
    if (!mem) {
        fprintf(stderr, "SKIP: cannot map a fake pool at 0x%llx\n", (unsigned long long)kPool);
        return failures ? 1 : 0;
    }
    memset(mem, 0, kLen);

    // Lay out a size-class-1 bundle chain: bin head at pool+0x20, then three 0x20-byte nodes. The
    // nodes sit above the 0x400 bin region so they are never mistaken for bins.
    auto at = [&](uint64_t off) { return (uint64_t*)((uint8_t*)mem + off); };
    const uint64_t n0 = kPool + 0x1000, n1 = kPool + 0x1020, n2 = kPool + 0x1040;
    *at(0x20)   = n0;        // bin[1].partial_head
    *at(0x1000) = n1;        // n0.next
    *at(0x1020) = n2;        // n1.next
    *at(0x1040) = 0;         // n2.next = end of chain
    mb3_reset_pool_candidates_for_test();
    mb3_note_tls_pool_candidate(kPool);

    // Counter-arm FIRST: a clean chain must report zero, with a census proving it walked the nodes.
    Mb3PoisonHit hits[4] = {};
    found = mb3_poison_scan(hits, 4, 64, census, sizeof census);
    check(found == 0, "clean chain reports no poison");
    check(strstr(census, "WALKED NOTHING") == nullptr && strstr(census, "nodes=") != nullptr,
          "clean-chain census shows nodes were walked");
    const std::string clean_census = census;

    // Now poison the middle node's link with each of the three shapes observed live. None of them
    // is 0x20-aligned-in-window, so all three must be found, at the node that holds them.
    const uint64_t shapes[] = {0x30016000ull, 0xff000000ff000000ull, 0x0002400100024001ull};
    for (uint64_t bad : shapes) {
        *at(0x1020) = bad;
        memset(hits, 0, sizeof hits);
        found = mb3_poison_scan(hits, 4, 64, census, sizeof census);
        char what[128];
        snprintf(what, sizeof what, "poison 0x%llx is found", (unsigned long long)bad);
        check(found == 1, what);
        snprintf(what, sizeof what, "poison 0x%llx is attributed to the node that holds it",
                 (unsigned long long)bad);
        check(hits[0].node == n1 && hits[0].bad_next == bad, what);
        check(hits[0].pool_base == kPool && hits[0].class_off == 0x20 && hits[0].hops == 2,
              "hit carries pool, size class and chain distance");
    }

    // Restoring the link must restore the clean verdict — the scan is reporting the chain's state,
    // not latching.
    *at(0x1020) = n2;
    found = mb3_poison_scan(hits, 4, 64, census, sizeof census);
    check(found == 0, "repaired chain reports no poison again");
    check(clean_census == census, "repaired chain walks exactly what the clean chain walked");

    munmap(mem, kLen);
    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("all mb3 poison-scan checks passed\n");
    return 0;
}
