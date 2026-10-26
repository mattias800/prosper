#include "gpu/execute/draw_index_width.hpp"
#include <algorithm>
#include <cstring>
#include <stdint.h>

namespace prosper::gpu {
// clang-format off: preserve the byte-identical moved bodies.

// Detect an UNANNOUNCED 32-bit index buffer (#304). DOLL's UE4 Slate/UMG quad index buffers are
// 32-bit, but the title never calls sceAgcDcbSetIndexSize and programs no VGT_INDEX_TYPE register,
// so index_type defaults to 16-bit — which misreads each 32-bit index as TWO 16-bit ones (the low
// half = the real index, the high half = 0), collapsing a quad's [0,1,2,2,1,3] to a degenerate
// [0,0,1,0,2,0]. The fingerprint is unmistakable and cheap: read the same buffer as 16-bit and as
// 32-bit for `n` entries; a real 32-bit buffer has EVERY odd 16-bit word (the zero high halves) == 0
// AND every 32-bit value small (< 0x10000) and not all-zero. Genuine 16-bit index buffers (DOLL's
// scene/text meshes; every Messenger quad, e.g. [0,1,2,2,3,0]) have a non-zero odd word and are
// rejected. A genuine 16-bit list with every odd word zero remains byte-ambiguous. The legacy
// offset path reads p16 at the 16-bit address and p32 at the recomputed 32-bit address. Additional
// same-byte recovery checks the original draw region first; see resolve_draw_index_source.
// CONFIDENCE: MED (a byte fingerprint, not an explicit guest index-width declaration).
bool index_buffer_is_unannounced_32bit(const uint16_t* p16, const uint32_t* p32, uint32_t n) {
    if (n < 2) return false;                         // need at least one odd word to test
    bool odd_zero = true, all_small = true, any_nonzero = false, has_odd = false;
    // memcpy loads, NOT typed derefs: p16/p32 view the SAME guest bytes, and reading one object
    // through both element types is strict-aliasing UB — Apple Clang 21 at -O2 proved it and
    // compiled the caller into ud2 (found by the macOS port; Linux GCC happened to tolerate it).
    // Fixed-size memcpy compiles to the same single loads without the aliasing assumption.
    for (uint32_t i = 0; i < n; i++) {
        if (i & 1) {
            has_odd = true;
            // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): preserve original uint32_t byte-offset arithmetic.
            uint16_t w; memcpy(&w, (const char*)p16 + 2u * i, 2);
            if (w != 0) odd_zero = false;
        }
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): preserve original uint32_t byte-offset arithmetic.
        uint32_t d; memcpy(&d, (const char*)p32 + 4u * i, 4);
        if (d >= 0x10000u) all_small = false;
        if (d != 0) any_nonzero = true;
    }
    return has_odd && odd_zero && all_small && any_nonzero;
}

// #304, part two: the SAME unannounced 32-bit index buffer, but with indices at or above 0x10000.
//
// The detector above requires every high half to be ZERO, which is only true while a title's indices
// stay under 65536. Tomb Raider I-III Remastered (PPSA16901) draws its level geometry out of one
// shared ~775,000-vertex pool, so a draw's indices sit in a 64 KiB window well above zero and every
// high half is the same NON-ZERO constant. The zero-fingerprint cannot match that, the buffer is read
// as 16-bit, and every triangle becomes a degenerate (N, K, N) sliver -- which is what shattered that
// title's world while its character meshes, whose index buffers are genuinely 16-bit, rendered
// correctly in the same frame.
//
// Measured on a live boot to Croft Manor (2026-08-26): 508,688 indexed draws, every one reporting
// index_type=0, i.e. the title announces an index size exactly never. Its 32-bit readings look like
// 428289..428292 and 774895..774898 -- consecutive, tightly clustered, and inside the pool's record
// count. (Only one decomposition of those draws is quoted here, under `kMinSamples` below. An earlier
// pass bucketed them by different criteria and the two totals do not reconcile; publishing both
// invited exactly that arithmetic to be checked and fail, so the superseded one is not repeated.)
//
// THE BYTE PATTERN ALONE IS NOT SUFFICIENT, and this is the part that matters. When the guest supplies
// no DrawIndexOffset the two addresses are THE SAME BYTES (see the caller: `addr32 == index_addr`
// unless `from_offset`), so a 16-bit buffer with a period-2 pattern -- a fan or cone encoded as a
// triangle strip `[rim, apex, rim, apex, ...]`, or a line list radiating from one hub -- is
// byte-identical to a clustered 32-bit list. No further test on these two pointers can tell them
// apart. Verified, not assumed: a 64-spoke line list to hub vertex 7 and a triangle-strip cone with
// apex 12 both satisfied every byte-pattern clause of an earlier version of this function.
//
// So the deciding evidence comes from OUTSIDE the buffer: an index must address a vertex that
// actually exists. `vertex_upper_bound` is the bound vertex buffer's UNCLAMPED record count
// (size/stride). The two cases separate immediately -- the cone's 32-bit reading demands 786,640
// vertices from a mesh that has some tens, while Tomb Raider's demands 775,111 from a pool holding
// exactly 775,111. A caller that cannot supply a bound passes 0 and this declines, because a
// discriminator that cannot see is not a licence to guess.
//
// Fingerprint, every clause required:
//   * a bound vertex-buffer record count, and every 32-bit index below it;
//   * at least `kMinSamples` entries, so "one parity is constant" is a real constraint. At most 64
//     entries are examined -- note that means the first 64 WORDS (dwords 0..31) for the parity loop
//     and 64 DWORDS for the range loop, which are different extents. The cap is deliberate rather
//     than an optimisation: a real run straddling a 64 KiB boundary carries TWO high halves and a
//     full scan would reject it on the parity clause. **It only rescues LATE crossings**, though --
//     swept and measured, a crossing at dword 5, 22 or 31 is still rejected, and only crossings at
//     dword 32 or beyond survive. So the cap is a partial mitigation, and it costs precision in the
//     other direction: only the first 32 dwords of a long buffer need the pattern. That is why the
//     vertex-range bound above, and not the pattern, has to carry the decision;
//   * one PARITY of the 16-bit reading -- either one -- holding the same NON-ZERO value. Which parity
//     depends on how the 16-bit address aligns against the 32-bit grid: a DrawIndexOffset scaled by 2
//     instead of 4 lands 2 mod 4 as often as not, and on the Croft Manor frame the even parity carried
//     it for 55,677 draws against 21,871 for the odd one, so checking only one misses most of it;
//   * the 32-bit reading spanning less than one 64 KiB window, not all zero, and at least
//     `kMinDistinct` distinct values.
// Zero high halves are left to the detector above so every buffer it already classifies keeps its
// existing verdict; this one only ever sees what that one rejected.
// CONFIDENCE: MED, and the residual is smaller than "a pool like Tomb Raider's" would suggest -- do
// not read this bound as narrow. A period-2 buffer's implied index is `constant << 16 | varying`, so
// the constant only has to be 1 for the implied indices to start just above 2^16. The general form is
// `(constant << 16) + max_sampled_varying + 1`, so the threshold moves with the construction: 65,602
// records for apex 1 / rim 2..65, 65,600 for rim 0..63, and as low as 65,544 for an n=8 draw -- do
// not quote any single one of these as a safe line. All of them are just above 2^16, not ~775,000,
// which is the point. The executor's
// own clamp constant three lines above `vb_records_unclamped` is 65,536, i.e. vertex buffers of about
// that size are routine, so a title with a modest pool AND fan/cone geometry whose apex index is small
// can still satisfy every clause here. That residual is now bounded on the OTHER side by
// index_size_detection_permitted() (#3009): a title that announced an index size never reaches this
// function at all, so the exposure is titles that announce nothing -- which is still the exposure for
// the two known cases, and the vertex-range bound remains their only guard.
bool index_buffer_is_unannounced_32bit_high(const uint16_t* p16, const uint32_t* p32,
                                                   uint32_t n, uint32_t vertex_upper_bound) {
    constexpr uint32_t kMinSamples = 8;     // >= 4 words per parity before "all equal" means anything
    constexpr uint32_t kMinDistinct = 4;    // a real index list is not two repeated values
    // An absolute ceiling ON TOP of the caller's bound. The bound is a decoded V#'s size/stride and is
    // therefore only as trustworthy as that descriptor: the line that clamps `vb_entries` to 65536
    // three lines above its capture exists precisely because over-sized vertex buffers do occur. With
    // no ceiling, a period-2 buffer whose constant is 400 implies indices near 26.2M and is accepted
    // the moment a bound exceeds them. 16.7M records is far above any real mesh and cheap to keep.
    constexpr uint32_t kMaxPlausibleIndex = 1u << 24;
    // The `vertex_upper_bound == 0` half is REDUNDANT and kept only to state intent, in the same
    // spirit as compute_address_range_contains() above: with no bound, `ceiling` folds to
    // min(0, 1<<24) = 0 and the very first `d >= ceiling` rejects anyway. Deleting it alone changes
    // no behaviour and no mutation can catch it -- which is exactly why it is annotated rather than
    // defended with an arm. Do not read it as load-bearing.
    if (n < kMinSamples || vertex_upper_bound == 0) return false;
    // Folding the two ceilings is exact -- `d >= min(A,B)` is `d >= A || d >= B` -- and the bound-0
    // early-return above means `ceiling` is never 0. What it hides is that the absolute ceiling also
    // lowers a LEGITIMATE large bound: a genuine pool of more than 16.7M vertices is declined here,
    // silently and without a distinguishable reason. No such pool has been measured; if one turns up,
    // this is the line to split.
    const uint32_t ceiling = std::min(vertex_upper_bound, kMaxPlausibleIndex);
    const uint32_t m = std::min(n, 64u);

    // Outside evidence first: every index must address a vertex the bound buffer actually holds.
    uint32_t lo = UINT32_MAX, hi = 0;
    // `any_nonzero` is likewise dead: an all-zero reading has exactly one distinct value, so the
    // `distinct < kMinDistinct` test below already rejects it. Kept for symmetry with the detector
    // above, whose version is load-bearing because it has no distinct-count clause.
    bool any_nonzero = false;
    uint32_t distinct = 0, seen[kMinDistinct] = {};
    for (uint32_t i = 0; i < m; i++) {
        // memcpy loads, NOT typed derefs: p16/p32 may view the SAME guest bytes and reading one object
        // through both element types is strict-aliasing UB (see the note on the detector above).
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): preserve original uint32_t byte-offset arithmetic.
        uint32_t d; memcpy(&d, (const char*)p32 + 4u * i, 4);
        if (d >= ceiling) return false;
        if (d != 0) any_nonzero = true;
        lo = std::min(lo, d); hi = std::max(hi, d);
        if (distinct < kMinDistinct) {
            bool dup = false;
            for (uint32_t k = 0; k < distinct; k++) if (seen[k] == d) { dup = true; break; }
            if (!dup) seen[distinct++] = d;
        }
    }
    if (!any_nonzero || distinct < kMinDistinct) return false;
    if ((hi - lo) >= 0x10000u) return false;            // must fit one 64 KiB index window

    // Then the byte fingerprint: one parity holds the repeated high half.
    bool even_const = true, odd_const = true;
    uint16_t even0 = 0, odd0 = 0;
    bool have_even = false, have_odd = false;
    for (uint32_t i = 0; i < m; i++) {
        // NOLINTNEXTLINE(bugprone-implicit-widening-of-multiplication-result): preserve original uint32_t byte-offset arithmetic.
        uint16_t w; memcpy(&w, (const char*)p16 + 2u * i, 2);
        if (i & 1) { if (!have_odd)  { odd0  = w; have_odd  = true; } else if (w != odd0)  odd_const  = false; }
        else       { if (!have_even) { even0 = w; have_even = true; } else if (w != even0) even_const = false; }
    }
    return (have_even && even_const && even0 != 0) || (have_odd && odd_const && odd0 != 0);
}
// clang-format on

// #4637, the third shape: a draw into a pool of MORE than 65,536 vertices whose indices cross a
// 64 KiB window inside the draw. Tomb Raider I's Lara's Home draws 133,353 indices from the start of
// a 489,947-vertex pool: the first ones sit below 65,536 (high half zero, so the detector above that
// wants a non-zero constant declines) and later ones above it (so the zero-high-half detector, which
// wants every index under 65,536, declines too). Read as 16-bit, the level is long triangles
// radiating from vertex 0.
//
// A bound alone cannot distinguish a small genuine 16-bit mesh in a large shared V# from
// 32-bit indices. Require local consecutive readings too: at least three quarters of the n-1
// adjacent deltas must be below 0x8000. Changing the odd 16-bit index normally moves a misread dword
// by 0x10000, whereas crossing 65535 -> 65536 moves a real index by one. A minority of jumps allows
// separate mesh regions in one draw. The fraction and distance are conservative heuristic policy,
// not an architectural index-width contract. CONFIDENCE: MED.
//
// KNOWN LIMIT: period-2 16-bit strips still alias 32-bit lists. So can non-periodic buffers with
// sufficiently rare odd-word changes or compensating low halves: this is not limited to fans.
// The supplied bound is the largest V# size/stride, potentially unrelated or aperture-sized;
// it is an upper bound only, not proof that these vertices belong to this mesh.
bool index_buffer_is_unannounced_32bit_in_pool(const uint32_t* p32, uint32_t n,
                                               uint32_t vertex_upper_bound) {
    constexpr uint32_t kMinSamples = 8;
    constexpr uint32_t kMinDistinct = 4;
    constexpr uint32_t kMaxPlausibleIndex = 1u << 24;
    if (n < kMinSamples || vertex_upper_bound <= 0x10000u) return false;
    const uint32_t ceiling = std::min(vertex_upper_bound, kMaxPlausibleIndex);
    uint32_t distinct = 0, seen[kMinDistinct] = {};
    uint32_t previous = 0, nonlocal = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t d;
        std::memcpy(&d, reinterpret_cast<const char*>(p32) + uint64_t{4} * i, 4);
        if (d >= ceiling) return false;
        if (i != 0) {
            const uint32_t delta = d > previous ? d - previous : previous - d;
            if (delta >= 0x8000u && ++nonlocal > (n - 1u) / 4u) return false;
        }
        previous = d;
        if (distinct < kMinDistinct) {
            bool dup = false;
            for (uint32_t k = 0; k < distinct; k++)
                if (seen[k] == d) {
                    dup = true;
                    break;
                }
            if (!dup) seen[distinct++] = d;
        }
    }
    return distinct >= kMinDistinct;
}

} // namespace prosper::gpu
