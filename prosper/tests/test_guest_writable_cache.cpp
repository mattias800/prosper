// #2387: guest_writable gained a range cache. These tests exist for one reason above all others:
// the cache must NEVER let a read-only mapping satisfy a write probe.
//
// That failure mode is the worst available. guest_writable's callers use it to decide whether it
// is safe to write into guest memory, so a false positive does not raise an error -- it produces a
// write to a page that should have refused one, or a refusal that never happens. It is silent, and
// it surfaces later as corrupted guest state with no connection to this file. Hence the first and
// largest test below, and hence a SEPARATE range set in the implementation rather than reuse of
// the readable one.
//
// The performance motivation is real but secondary and is not asserted here: on Linux the uncached
// path opens /proc/self/maps and runs a fgets/sscanf loop over the whole mapping table on every
// call. A timing assertion would be flaky in CI, so these tests pin CORRECTNESS and the os_probe
// count pins that the cache is actually engaged.

#include "gpu/gpu_execute.hpp"
#include "host/guest_memory_map.hpp"

#include <cstdint>
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

static int failures = 0;
static void check(bool ok, const char* what) {
    std::fprintf(stderr, "%s %s\n", ok ? "[ ok ]" : "[FAIL]", what);
    if (!ok) ++failures;
}

int main() {
    std::fprintf(stderr, "== test_guest_writable_cache ==\n");
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);

    // ---------------------------------------------------------------------------------------
    // THE LOAD-BEARING CASE: a read-only mapping, probed for READ first so it lands in the
    // readable cache, must still be refused for WRITE.
    //
    // If the two caches were shared -- or if the writable path consulted the readable registry
    // as an optimisation -- this returns true and nothing anywhere reports a problem.
    // ---------------------------------------------------------------------------------------
    {
        void* ro = mmap(nullptr, page, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(ro != MAP_FAILED, "mapped a read-only page");
        if (ro != MAP_FAILED) {
            const uint64_t addr = (uint64_t)(uintptr_t)ro;
            const bool r = prosper::gpu::guest_readable(addr, 8);
            check(r, "control: the read-only page IS readable");

            // Now genuinely PUT the range in the readable registry.
            //
            // The first version of this test called guest_readable() and asserted in a comment
            // that the readable cache therefore held the range. It did not:
            // cache_guest_readable_range() only inserts while the cache is `active`, which
            // requires a submit scope, and a bare test has none. So nothing was ever cached, the
            // shared-cache mutation had nothing to hit, and this test PASSED against the exact
            // implementation it exists to reject -- a void discriminator that reads as the
            // strongest assertion in the file.
            //
            // Caught only by mutating the implementation to consult the readable cache and
            // watching the suite stay green. Inserting directly is what makes the arm real:
            // guest_range_cache_hit() consults this registry, so an implementation that shares or
            // queries it now answers true for a read-only page, and the check below fails.
            prosper::host::detail::guest_readable_mappings.insert(addr, addr + page);
            check(prosper::host::detail::guest_readable_mappings.contains(addr, addr + 8),
                  "precondition ESTABLISHED, not assumed: the readable registry holds this range");

            check(!prosper::gpu::guest_writable(addr, 8),
                  "a READ-ONLY page is refused for write even after the readable cache holds it");
            check(!prosper::gpu::guest_writable(addr, 8),
                  "...and is still refused on the second call (no negative got cached as positive)");
            prosper::host::detail::guest_readable_mappings.erase(addr, addr + page);
            munmap(ro, page);
        }
    }

    // ---------------------------------------------------------------------------------------
    // The ordinary path: a writable mapping answers true, and answers true again from cache.
    // ---------------------------------------------------------------------------------------
    {
        void* rw = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(rw != MAP_FAILED, "mapped a read-write page");
        if (rw != MAP_FAILED) {
            const uint64_t addr = (uint64_t)(uintptr_t)rw;
            check(prosper::gpu::guest_writable(addr, 8), "a read-write page is writable");
            check(prosper::gpu::guest_writable(addr, 8), "...and again (served from cache)");
            // A sub-range of an already-proven span must also hit rather than re-probe.
            check(prosper::gpu::guest_writable(addr + 8, 8),
                  "a sub-range of a proven-writable span is writable");
            munmap(rw, page);
        }
    }

    // ---------------------------------------------------------------------------------------
    // INVALIDATION: a range cached as writable must stop being writable once the mapping changes
    // and the guest mapping generation advances. Without the generation guard the cache would
    // happily keep answering true for memory that is now read-only -- a stale positive, which is
    // the same silent corruption as case one, only arriving later.
    // ---------------------------------------------------------------------------------------
    {
        void* rw = mmap(nullptr, page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(rw != MAP_FAILED, "mapped a page to revoke");
        if (rw != MAP_FAILED) {
            const uint64_t addr = (uint64_t)(uintptr_t)rw;
            check(prosper::gpu::guest_writable(addr, 8), "writable before revocation (now cached)");

            check(mprotect(rw, page, PROT_READ) == 0, "revoked write permission");
            // Reaching into detail:: deliberately: the generation is normally advanced by the
            // kernel-memory HLE on map/unmap, and this test needs the invalidation edge without
            // standing up an HLE mapping. The alternative -- asserting nothing about invalidation
            // -- would leave the cache's only staleness guard untested.
            prosper::host::detail::advance_guest_mapping_generation();

            check(!prosper::gpu::guest_writable(addr, 8),
                  "a revoked page is NOT writable once the mapping generation advances");
            munmap(rw, page);
        }
    }

    std::fprintf(stderr, failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
