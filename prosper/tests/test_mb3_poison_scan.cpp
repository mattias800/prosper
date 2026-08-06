// test_mb3_poison_scan — the interior free-chain poison scan (#1945).
//
// Two properties, both of which this family has been burned by the absence of:
//
//   1. A `found == 0` must be distinguishable from a scan that walked nothing. The instrument this
//      one replaces (`PROSPER_MB3WATCH`) armed on a stale address window and therefore printed
//      nothing on every current title, which reads exactly like "armed and saw no corruption".
//      So the census is asserted, not just the verdict.
//   2a. The scan stays on size class idx=1. Every class shares the bin shape but not the node
//      alignment, so applying the idx=1 rule to all of them reports healthy 16-byte-class heads as
//      poison — measured live before this was narrowed. A healthy chain hung off a DIFFERENT class
//      with 0x10-aligned nodes must therefore report zero.
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
#if !defined(_WIN32)
#include <sys/mman.h>
#endif

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
    if (p == MAP_FAILED) return nullptr;
    if ((uint64_t)(uintptr_t)p != at) { munmap(p, len); return nullptr; }   // wrong address: give it back
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
        // 77 is ctest's SKIP_RETURN_CODE. Returning 0 here would let the whole structural half of
        // this test silently not run while the job goes green — the exact failure mode the rest of
        // this file exists to prevent.
        fprintf(stderr, "SKIP: cannot map a fake pool at 0x%llx\n", (unsigned long long)kPool);
        return failures ? 1 : 77;
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
    // not latching. (Repair BEFORE the next case: the loop above leaves the last poison in place.)
    *at(0x1020) = n2;
    found = mb3_poison_scan(hits, 4, 64, census, sizeof census);
    check(found == 0, "repaired chain reports no poison again");
    check(clean_census == census, "repaired chain walks exactly what the clean chain walked");

    // The MAPPEDNESS leg: correctly aligned, but the target is not mapped at all. (This case was
    // first written as the window-leg case and it is not one — review measured that it is rejected
    // by safe_read before the window test is ever reached, so deleting the window line still left
    // the file green. Both legs are pinned separately below.)
    *at(0x1020) = 0x1000000020ull;              // 0x20-aligned, unmapped
    found = mb3_poison_scan(hits, 4, 64, census, sizeof census);
    check(found == 1 && hits[0].bad_next == 0x1000000020ull,
          "an aligned link to unmapped memory is poison");
    *at(0x1020) = n2;

    // The WINDOW leg, discriminated: an ordinary anonymous mapping is page-aligned (so it passes the
    // alignment test) and readable (so it passes safe_read). ONLY the window line can reject it, so
    // deleting that line fails exactly this check and nothing else.
    void* ow_map = mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ow_map == MAP_FAILED) {
        fprintf(stderr, "FAIL: could not map an out-of-window page for the window-leg case\n");
        failures++;
    } else {
        const uint64_t ow = (uint64_t)(uintptr_t)ow_map;
        check((ow & 0x1full) == 0, "the out-of-window probe address is 0x20-aligned (else it is void)");
        check(ow < 0x2000000000ull || ow >= 0x4000000000ull,
              "the out-of-window probe address really is outside the window (else it is void)");
        *at(0x1020) = ow;
        found = mb3_poison_scan(hits, 4, 64, census, sizeof census);
        check(found == 1 && hits[0].bad_next == ow,
              "a MAPPED, aligned link outside the guest window is poison too");
        *at(0x1020) = n2;
        munmap(ow_map, 0x1000);
    }

    // A healthy 16-byte-class chain (0x10-aligned nodes, class offset 0x00) must NOT be reported:
    // it is legal for its own class and the scan has no business judging it. Before the scan was
    // narrowed to idx=1 this reported every such head as poison, dozens per submit on a live title.
    const uint64_t m0 = kPool + 0x2000, m1 = kPool + 0x2010;
    *at(0x00)   = m0;
    *at(0x2000) = m1;
    *at(0x2010) = 0;
    found = mb3_poison_scan(hits, 4, 64, census, sizeof census);
    check(found == 0, "a healthy 0x10-aligned chain in another size class is not called poison");

    munmap(mem, kLen);
    if (failures) { fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    printf("all mb3 poison-scan checks passed\n");
    return 0;
}
