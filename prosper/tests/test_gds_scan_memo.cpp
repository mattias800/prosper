// Regression test for the per-draw internal-GDS scan memo (#2334).
//
// fragment_spirv_uses_internal_gds walks the entire SPIR-V module and builds two unordered_maps.
// render_runner.h called it once per DRAW at two sites, on a module that changes only when the
// shader does; on Blue Prince's collapsed state (~4,048 draws/submit) it measured 5.64% of the
// saturated render thread. fragment_uses_internal_gds_memoized caches the result on fs_identity.
//
// The modules here are built BY HAND rather than recompiled, so the positive case is constructed
// outside whatever produces the negative one -- a fixture that generated both from one recompile
// could share a defect that makes the GDS case inexpressible, and the test would then pass without
// ever exercising it.

#include "render_runner.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what);
}

// A minimal module the scanner can parse: 5-word header, then OpDecorate pairs.
// The scanner reports true when some id carries DescriptorSet==1 AND Binding==0.
static std::vector<uint32_t> make_module(uint32_t descriptor_set) {
    constexpr uint32_t kOpDecorate = 71, kDecBinding = 33, kDecDescriptorSet = 34;
    const uint32_t decorate = (4u << 16) | kOpDecorate;   // 4-word OpDecorate
    return {
        0x07230203u, 0x00010000u, 0u, 8u, 0u,             // magic, version, generator, bound, schema
        decorate, 1u, kDecDescriptorSet, descriptor_set,
        decorate, 1u, kDecBinding, 0u,
    };
}

int main() {
    const bool memo_disabled = getenv("PROSPER_NO_GDS_SCAN_MEMO") != nullptr;
    const std::vector<uint32_t> gds = make_module(1);      // set=1, binding=0 -> internal GDS
    const std::vector<uint32_t> plain = make_module(2);    // set=2            -> not internal GDS

    // The underlying scanner must disagree about these two modules, or every arm below is vacuous.
    check(prosper::gpu::fragment_spirv_uses_internal_gds(gds),
          "hand-built set=1/binding=0 module scans as internal GDS");
    check(!prosper::gpu::fragment_spirv_uses_internal_gds(plain),
          "hand-built set=2 module does not scan as internal GDS");

    // Distinct identities must not contaminate each other.
    check(prosper::test::fragment_uses_internal_gds_memoized(101, gds), "identity 101 -> gds true");
    check(!prosper::test::fragment_uses_internal_gds_memoized(102, plain), "identity 102 -> plain false");
    check(prosper::test::fragment_uses_internal_gds_memoized(101, gds), "identity 101 repeat -> true");
    check(!prosper::test::fragment_uses_internal_gds_memoized(102, plain), "identity 102 repeat -> false");

    // THE DISCRIMINATOR. Re-ask under an identity already cached, but hand it the OTHER module's
    // words. A memo that is actually consulted returns the cached answer and never looks at the
    // words; a memo that silently does nothing re-walks them and returns the other result. This is
    // the only arm that can tell "cached" from "recomputed the same answer", and it is why the two
    // modules must disagree (asserted above).
    //
    // Reusing one identity for two different modules cannot happen in production -- identities come
    // from a monotonic never-reset counter, so an evicted shader recompiles to a NEW identity. It is
    // done here precisely because it is the observable consequence of caching.
    const bool stale = prosper::test::fragment_uses_internal_gds_memoized(101, plain);
    if (memo_disabled)
        check(!stale, "PROSPER_NO_GDS_SCAN_MEMO: identity 101 + plain words re-scans -> false");
    else
        check(stale, "memo hit: identity 101 + plain words returns the CACHED true, not a re-scan");

    // Identity 0 is never minted by the recompile cache and must bypass the memo entirely, so it
    // can never collide with a real module. Both answers must track the words handed in.
    check(prosper::test::fragment_uses_internal_gds_memoized(0, gds), "identity 0 + gds -> true");
    check(!prosper::test::fragment_uses_internal_gds_memoized(0, plain), "identity 0 + plain -> false");
    check(prosper::test::fragment_uses_internal_gds_memoized(0, gds), "identity 0 + gds again -> true");

    std::fprintf(stderr, "%s (%s)\n", failures ? "FAILURES" : "all arms passed",
                 memo_disabled ? "memo disabled" : "memo enabled");
    return failures ? 1 : 0;
}
