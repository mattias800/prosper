// The depth/stencil layer-stride registry, and the identity a stride observation belongs to.
//
// THE DEFECT. The registry was keyed by guest base alone. A base is not a stable identity: the guest
// frees an allocation and maps another at the same address, and the entry then answers for a surface
// that no longer exists. That is not bookkeeping -- the stride sizes the range a guest write
// invalidates, so a stride carried across a reuse either leaves stale pixels resident
// (under-invalidation) or evicts a neighbouring slice that was still good (over-invalidation). The
// previous code handled reuse with a heuristic in the place an identity belongs: take whichever
// observation came last, and complain once. That cannot distinguish "the allocation was recycled"
// from "two live views disagree", so it reported the first as loudly as the second and resolved
// neither correctly.
//
// THE SECOND DEFECT, same family. The producer's stride rode on `PersistentDsKey` as a member
// excluded from operator== and from the hash. The cache is keyed by that struct and the invalidation
// loop iterates the STORED keys, so the stride it read was whatever the attachment that CREATED the
// entry happened to see. A producer that programmed DB_DEPTH_SLICE on a later attachment never
// reached the invalidation, and one that reprogrammed it was ignored in favour of the frozen value.
// An excluded field cannot make a lookup miss, so nothing ever surfaced it. The stride now lives in
// the registry, refreshed on every attachment; arm 7 is that case.
//
// The arms below are built so that the KEY CHANGE is what they detect, not the plumbing. Restoring
// base-alone keying fails arms 1-3 outright, and re-freezing the stride fails arm 7; they are
// constructed by hand at this header's own production functions, which are the ones the renderer
// calls.
#include "render_runner.h"

#include <cstdio>
#include <cstdint>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what);
}

using prosper::test::ds_layer_stride_for;
using prosper::test::ds_slice_bytes_from_db_depth_slice;
using prosper::test::ds_stride_identity_base;
using prosper::test::note_ds_layer_stride;
using prosper::test::note_ds_producer_slice_bytes;
using prosper::test::PersistentDsKey;
using prosper::test::PersistentDsKeyHash;

int main() {
    // GTA V's cube-depth shape: one base, six layers, a consumer descriptor that carries the exact
    // layer stride. The second surface is a DIFFERENT allocation that the guest later maps at the
    // same address -- the case base-alone keying cannot express.
    constexpr uint64_t kBase = 0x2052ac0000ull;
    constexpr uint64_t kStrideA = 4u * 1024u * 1024u;
    constexpr uint64_t kStrideB = 512u * 1024u;

    // ---- 1. Two surfaces at one recycled base keep their OWN strides ----------------------------
    // Under base-alone keying one of these two answers is necessarily wrong: whichever was published
    // second overwrites the other. Both must hold simultaneously.
    note_ds_layer_stride(kBase, 1024, 1024, kStrideA);
    note_ds_layer_stride(kBase, 512, 512, kStrideB);
    check(ds_layer_stride_for(kBase, 1024, 1024) == kStrideA,
          "the first surface keeps its stride after a different surface is mapped at its base");
    check(ds_layer_stride_for(kBase, 512, 512) == kStrideB,
          "the recycled surface reports its own stride, not its predecessor's");

    // ---- 2. Reuse does not INHERIT a stride -----------------------------------------------------
    // The invalidation consequence directly: a surface that never published a stride must read as
    // unknown (0 -> whole-allocation fallback), never as the previous tenant's value. Inheriting
    // kStrideA here would offset every slice by 4 MiB into memory this surface does not own.
    check(ds_layer_stride_for(kBase, 256, 256) == 0,
          "a surface at a recycled base with no published stride is unknown, not inherited");

    // ---- 3. Height is part of the identity, not just width --------------------------------------
    // Same base, same width, different height. These are different surfaces and must not share.
    note_ds_layer_stride(kBase, 1024, 512, 777u);
    check(ds_layer_stride_for(kBase, 1024, 512) == 777u &&
              ds_layer_stride_for(kBase, 1024, 1024) == kStrideA,
          "surfaces differing only in height are separate identities");

    // ---- 4. A disagreement that SURVIVES the identity is still a disagreement -------------------
    // Same base, same dimensions, two different strides: reuse is already excluded by the identity,
    // so this is a genuine decode error. The later value is taken (arbitrary between two claims we
    // cannot adjudicate -- the report on stderr is the mitigation, not the pick).
    note_ds_layer_stride(kBase, 1024, 1024, kStrideA + 64u);
    check(ds_layer_stride_for(kBase, 1024, 1024) == kStrideA + 64u,
          "a same-identity disagreement takes the later value");

    // ---- 5. Degenerate observations record nothing ----------------------------------------------
    note_ds_layer_stride(0, 1024, 1024, kStrideA);
    note_ds_layer_stride(0x2060000000ull, 64, 64, 0);
    check(ds_layer_stride_for(0, 1024, 1024) == 0, "a zero base never resolves");
    check(ds_layer_stride_for(0x2060000000ull, 64, 64) == 0,
          "a zero stride is not recorded, so the identity stays unknown");

    // ---- 6. Precedence: the producer's register outranks a consumer descriptor ------------------
    constexpr uint64_t kProducerBase = 0x2070000000ull;
    note_ds_layer_stride(kProducerBase, 128, 128, 0x1000u);
    check(ds_layer_stride_for(kProducerBase, 128, 128) == 0x1000u,
          "with only a consumer stride, the consumer's answers");
    note_ds_producer_slice_bytes(kProducerBase, 128, 128, 0x4000u);
    check(ds_layer_stride_for(kProducerBase, 128, 128) == 0x4000u,
          "a producer stride outranks a consumer one for the same identity");

    // ---- 7. A producer stride published LATE still reaches the lookup --------------------------
    // This is the defect that moved slice_bytes off PersistentDsKey. The cache is keyed by that
    // struct and the invalidation loop iterates the STORED keys, so a stride carried on the key was
    // frozen at entry creation: a producer that programmed DB_DEPTH_SLICE only on a later attachment
    // never reached the invalidation, and one that reprogrammed it was ignored in favour of the
    // frozen first value. Here the surface is known to the registry BEFORE the producer speaks.
    constexpr uint64_t kLateBase = 0x2080000000ull;
    note_ds_layer_stride(kLateBase, 2048, 2048, 0x2000u);          // entry exists, consumer only
    note_ds_producer_slice_bytes(kLateBase, 2048, 2048, 0x9000u);  // producer programs it later
    check(ds_layer_stride_for(kLateBase, 2048, 2048) == 0x9000u,
          "a producer stride published after the entry exists still reaches the lookup");
    note_ds_producer_slice_bytes(kLateBase, 2048, 2048, 0xa000u);  // and reprogrammed later still
    check(ds_layer_stride_for(kLateBase, 2048, 2048) == 0xa000u,
          "a REPROGRAMMED producer stride replaces the earlier one rather than being frozen");
    note_ds_producer_slice_bytes(kLateBase, 2048, 2048, 0);        // "never programmed" on this pass
    check(ds_layer_stride_for(kLateBase, 2048, 2048) == 0xa000u,
          "an unprogrammed (zero) producer register does not erase a known stride");

    // ---- 7b. Recorder and lookup agree on WHICH base identifies a surface -----------------------
    // A depth surface that is written but never read has depth_read_base == 0. If the recorder keyed
    // on the read base and the lookup on the write base, they would never meet and per-slice
    // invalidation would silently never engage for exactly those surfaces -- a divergence invisible
    // from either site alone. One expression serves both.
    check(ds_stride_identity_base(0x2052ac0000ull, 0x2052ac0000ull) == 0x2052ac0000ull,
          "a read+write surface identifies by its read base");
    check(ds_stride_identity_base(0, 0x2090000000ull) == 0x2090000000ull,
          "a WRITE-ONLY depth surface identifies by its write base, not by 0");
    check(ds_stride_identity_base(0x2091000000ull, 0x2092000000ull) == 0x2091000000ull,
          "when both are present the read base wins, deterministically");
    check(ds_stride_identity_base(0, 0) == 0, "a surface with neither base has no identity");
    {
        // End to end through that selection: publish under the write-only surface's identity and
        // resolve it the way the invalidation loop does.
        constexpr uint64_t kWriteOnly = 0x20a0000000ull;
        note_ds_layer_stride(ds_stride_identity_base(0, kWriteOnly), 640, 480, 0x3000u);
        check(ds_layer_stride_for(ds_stride_identity_base(0, kWriteOnly), 640, 480) == 0x3000u,
              "a write-only surface's stride round-trips through the shared identity");
    }

    // ---- 8. DB_DEPTH_SLICE decode ---------------------------------------------------------------
    // SLICE_TILE_MAX gives the per-slice size as (SLICE_TILE_MAX + 1) * 64 bytes; a never-programmed
    // register reads 0 and must surface as "unknown", not as a zero stride.
    check(ds_slice_bytes_from_db_depth_slice(0) == 0,
          "an unprogrammed DB_DEPTH_SLICE is unknown, not a zero stride");
    {
        const uint32_t tile_max = 1023u;
        const uint32_t reg = tile_max << prosper::agc::Pm4::DB_DEPTH_SLICE_SLICE_TILE_MAX_SHIFT;
        check(ds_slice_bytes_from_db_depth_slice(reg) == (uint64_t)(tile_max + 1u) * 64ull,
              "SLICE_TILE_MAX decodes as (tile_max + 1) * 64 bytes");
    }

    // ---- 9. PersistentDsKey is a pure identity --------------------------------------------------
    // Every member takes part in equality and in the hash. This is the falsifiable half: dropping
    // any field from operator== must fail here. It is also what keeps the struct honest -- the
    // stride used to live here as a member excluded from both, and that exclusion is exactly what
    // let a stale value ride in the cache key unnoticed.
    {
        const PersistentDsKey base_key{0x2052ac0000ull, 0x2052ac0000ull, 0x2054aa0000ull,
                                       0x2054aa0000ull, 0x2055310000ull, 1920, 1080, 7, 3};
        const PersistentDsKey same_key{0x2052ac0000ull, 0x2052ac0000ull, 0x2054aa0000ull,
                                       0x2054aa0000ull, 0x2055310000ull, 1920, 1080, 7, 3};
        check(base_key == same_key,
              "two independently built keys with the same fields are one identity");
        check(PersistentDsKeyHash{}(base_key) == PersistentDsKeyHash{}(same_key),
              "...and hash to the same bucket, as unordered_map requires");

        struct Field { const char* name; PersistentDsKey key; };
        auto with = [&](void (*mutate)(PersistentDsKey&)) {
            PersistentDsKey k = base_key; mutate(k); return k;
        };
        const Field fields[] = {
            {"dr",    with([](PersistentDsKey& k) { k.dr = 0x2099990000ull; })},
            {"dw",    with([](PersistentDsKey& k) { k.dw = 0x2099990000ull; })},
            {"sr",    with([](PersistentDsKey& k) { k.sr = 0x2099990000ull; })},
            {"sw",    with([](PersistentDsKey& k) { k.sw = 0x2099990000ull; })},
            {"htile", with([](PersistentDsKey& k) { k.htile = 0x2099990000ull; })},
            {"w",     with([](PersistentDsKey& k) { k.w = 1280; })},
            {"h",     with([](PersistentDsKey& k) { k.h = 720; })},
            {"fmt",   with([](PersistentDsKey& k) { k.fmt = 9; })},
            {"slice", with([](PersistentDsKey& k) { k.slice = 5; })},
        };
        for (const Field& f : fields) {
            char msg[128];
            std::snprintf(msg, sizeof msg, "%s is part of PersistentDsKey identity", f.name);
            check(!(base_key == f.key), msg);
        }
    }

    // ---- 10. END TO END through BOTH production seams -------------------------------------------
    // Arm 7b tests `ds_stride_identity_base` in isolation, which cannot detect a CALL SITE that
    // fails to use it -- the divergence would live in the gap between the recorder and the lookup,
    // and a test of the shared helper stays green through it. This arm closes that gap: the stride
    // is published by `persistent_ds_key_for` (the shipping construction seam) and resolved by
    // `invalidate_persistent_ds_guest_write` (the shipping consumer), with nothing hand-fed between
    // them. The surface is WRITE-ONLY, which is exactly the case a read-base-keyed recorder loses.
    //
    // If the two sites disagree about which base identifies the surface, the stride resolves as
    // unknown, per-slice precision collapses to whole-allocation, and the sibling slice below is
    // evicted along with the target -- so the second check is the detector.
    {
        constexpr uint64_t kBase = 0x20b0000000ull;
        constexpr uint32_t kW = 256, kH = 256;
        constexpr uint32_t kTileMax = 255u;              // (255 + 1) * 64 = 16384 bytes per slice
        constexpr uint64_t kSliceBytes = 16384ull;

        auto key_for_slice = [&](uint32_t slice_index) {
            prosper::gpu::ResolvedPipelineState ps;
            ps.depth_read_base = 0;                      // write-only: the divergence case
            ps.depth_write_base = kBase;
            ps.db_depth_slice = kTileMax
                                << prosper::agc::Pm4::DB_DEPTH_SLICE_SLICE_TILE_MAX_SHIFT;
            ps.db_depth_view = slice_index;
            return prosper::test::persistent_ds_key_for(ps, /*htile=*/0, kW, kH, /*format=*/7);
        };

        auto& cache = prosper::test::persistent_ds_cache();
        const PersistentDsKey k0 = key_for_slice(0);
        const PersistentDsKey k3 = key_for_slice(3);
        check(!(k0 == k3), "two slices of one write-only allocation are distinct identities");
        cache[k0].depth_valid = true;
        cache[k3].depth_valid = true;

        // A guest write landing inside slice 3 and nowhere near slice 0.
        prosper::test::invalidate_persistent_ds_guest_write(kBase + 3ull * kSliceBytes + 64ull, 128);
        check(!cache[k3].depth_valid,
              "a guest write inside slice 3 invalidates slice 3");
        check(cache[k0].depth_valid,
              "...and leaves slice 0 resident -- per-slice precision survived both seams");
    }

    if (failures) { std::fprintf(stderr, "== FAIL: %d ==\n", failures); return 1; }
    std::fprintf(stderr, "== PASS ==\n");
    return 0;
}
