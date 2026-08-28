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
// THE SECOND DEFECT, same family. The stride rode on `PersistentDsKey` as a member excluded from
// operator== and from the hash. The cache is keyed by that struct and the invalidation loop iterates
// the STORED keys, so the stride it read was frozen at whatever the attachment that CREATED the
// entry saw. An excluded field cannot make a lookup miss, so nothing ever surfaced it. The stride
// now lives in the registry, keyed by identity.
//
// The arms below are built so that the KEY CHANGE is what they detect, not the plumbing. Restoring
// base-alone keying fails arms 1-3 outright, and collapsing the recorder/lookup identity fails arm 8; they are constructed by hand at this header's own production functions, which are
// the ones the renderer calls.
#include "fixtures/render_runner.h"

#include <cstdio>
#include <cstdint>

static int failures = 0;
static void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++failures; }
    else std::fprintf(stderr, "ok: %s\n", what);
}

using prosper::test::ds_layer_stride_for;
using prosper::test::ds_stride_identity_base;
using prosper::test::note_ds_layer_stride;
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

    // ---- 6. Recorder and lookup agree on WHICH base identifies a surface -----------------------
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

    // ---- 7. PersistentDsKey is a pure identity --------------------------------------------------
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

    // ---- 8. END TO END through BOTH production seams -------------------------------------------
    // Arm 6 tests `ds_stride_identity_base` in isolation, which cannot detect a CALL SITE that fails
    // to use it -- the divergence would live in the gap between publisher and lookup, and a test of
    // the shared helper stays green through it. This arm closes that gap: the identity is built by
    // `persistent_ds_key_for` (the shipping construction seam) and consumed by
    // `invalidate_persistent_ds_guest_write` (the shipping consumer), with nothing hand-fed between
    // them. The surface is WRITE-ONLY, the case a read-base-keyed lookup loses.
    //
    // The fixture is PHYSICAL: a 256x256 D32 slice is 256*256*4 = 262144 bytes, so the layer stride
    // is 262144 (tightly packed). An earlier revision used 16384, which is smaller than one slice
    // and therefore describes a layout that cannot exist -- the slices would overlap.
    {
        constexpr uint64_t kBase = 0x20b0000000ull;
        constexpr uint32_t kW = 256, kH = 256;
        constexpr uint64_t kSliceBytes = 262144ull;      // == kW * kH * 4, tightly packed

        auto key_for_slice = [&](uint32_t slice_index) {
            prosper::gpu::ResolvedPipelineState ps;
            ps.depth_read_base = 0;                      // write-only: the divergence case
            ps.depth_write_base = kBase;
            ps.db_depth_view = slice_index;
            return prosper::test::persistent_ds_key_for(ps, /*htile=*/0, kW, kH, /*format=*/7);
        };

        auto& cache = prosper::test::persistent_ds_cache();
        const PersistentDsKey k0 = key_for_slice(0);
        const PersistentDsKey k3 = key_for_slice(3);
        check(!(k0 == k3), "two slices of one write-only allocation are distinct identities");

        // Publish the stride under the identity the lookup will use.
        note_ds_layer_stride(ds_stride_identity_base(0, kBase), kW, kH, kSliceBytes);
        cache[k0].depth_valid = true;
        cache[k3].depth_valid = true;

        // A guest write landing inside slice 3 and nowhere near slice 0.
        prosper::test::invalidate_persistent_ds_guest_write(kBase + 3ull * kSliceBytes + 64ull, 128);
        check(!cache[k3].depth_valid,
              "a guest write inside slice 3 invalidates slice 3");
        check(cache[k0].depth_valid,
              "...and leaves slice 0 resident -- per-slice precision survived both seams");
        // Note which check is the detector, because it is not the one you would guess: with a
        // physical stride, losing it collapses the window to ONE slice at the base, and the slice-3
        // address then falls outside that window entirely -- so a recorder/lookup divergence makes
        // the FIRST check fail (slice 3 never invalidated), not the second.
    }

    // ---- 9. The stencil plane is NOT strided by the DEPTH plane's layer stride -------------------
    // `learned` is the depth layer stride. Stencil is a separate allocation with its own layout --
    // 1 byte/pixel against depth's 4 -- so striding into it by depth's value lands about 4x too far
    // out. That is the unsafe direction twice: a real write to stencil slice N is missed (stale
    // stencil resident) and a write that belongs to no slice of this surface can be misattributed
    // to it. This arm pins the second half, which is the observable one.
    {
        constexpr uint64_t kDepth = 0x20c0000000ull;
        constexpr uint64_t kStencil = 0x20d0000000ull;   // far from depth, so ranges cannot alias
        constexpr uint32_t kW = 256, kH = 256;
        constexpr uint64_t kDepthSlice = 262144ull;      // depth layer stride
        constexpr uint64_t kStencilBytes = 65536ull;     // kW * kH, one byte per pixel

        prosper::gpu::ResolvedPipelineState ps;
        ps.depth_read_base = kDepth;
        ps.depth_write_base = kDepth;
        ps.stencil_read_base = kStencil;
        ps.stencil_write_base = kStencil;
        ps.db_depth_view = 3;
        const PersistentDsKey k = prosper::test::persistent_ds_key_for(ps, 0, kW, kH, 7);
        note_ds_layer_stride(ds_stride_identity_base(kDepth, kDepth), kW, kH, kDepthSlice);

        auto& cache = prosper::test::persistent_ds_cache();
        cache[k].depth_valid = true;
        cache[k].stencil_valid = true;

        // Where the DEPTH-strided bug would look for slice 3's stencil: kStencil + 3 * 262144.
        // The whole stencil allocation is only 6 * 65536 = 393216 bytes, so this address is past the
        // end of it and belongs to no slice of this surface. Invalidating on it is the misattribution.
        prosper::test::invalidate_persistent_ds_guest_write(kStencil + 3ull * kDepthSlice + 64ull, 128);
        check(cache[k].stencil_valid,
              "a write beyond the stencil allocation, where a depth-strided offset would look, does "
              "not invalidate stencil");

        // A write inside the stencil window IS caught -- so the arm above is not passing merely
        // because stencil invalidation stopped working altogether.
        prosper::test::invalidate_persistent_ds_guest_write(kStencil + 64ull, 128);
        check(!cache[k].stencil_valid, "a write inside the stencil plane still invalidates stencil");
        check(cache[k].depth_valid,
              "...and a stencil-only write leaves the depth aspect valid");
    }

    // ---- 10. A byte-preserving HTILE rewrite STILL discards retained depth (#3089) -----------
    // This case previously asserted the opposite. The premise it rested on -- "byte-identical
    // metadata describes the same logical surface" -- does not hold, because prosper never writes
    // rendered HiZ back into the guest HTILE plane. The guest copy is therefore a constant that the
    // guest's own writes keep reproducing, so the comparator reports changed=0 for a fast CLEAR
    // exactly as readily as for the decompress the exception was written for. Byte equality cannot
    // tell them apart, and the invalidation is the conservative half.
    //
    // Blue Prince (PPSA25009) is the measured counterexample: all three of its 196,608-byte HTILE
    // planes are rewritten with all-zero words (49,152/49,152 equal, zero transitions), so every
    // write after the first compares equal, every DS invalidation is suppressed (agree=1,
    // suppressed=59,999) and the title renders a pure black frame -- 0.00% non-black over 16,500+
    // colour readbacks. Restoring the invalidation restores its fade-in to ~21%.
    //
    // The contract of notify_guest_gpu_write_preserving_bytes already draws this line: byte
    // preservation licenses skipping guest-memory watches and the submit journal, never the
    // renderer-alias invalidation, whose owners "may differ from the exact guest bytes even when a
    // compute result does not".
    {
        constexpr uint64_t kDepth = 0x20e0000000ull;
        constexpr uint64_t kStencil = 0x20f0000000ull;
        constexpr uint64_t kHtile = 0x2100000000ull;
        constexpr uint32_t kW = 256, kH = 256;

        prosper::gpu::ResolvedPipelineState ps;
        ps.depth_read_base = ps.depth_write_base = kDepth;
        ps.stencil_read_base = ps.stencil_write_base = kStencil;
        const PersistentDsKey key =
            prosper::test::persistent_ds_key_for(ps, kHtile, kW, kH, /*format=*/7);
        auto& image = prosper::test::persistent_ds_cache()[key];
        image.depth_valid = true;
        image.stencil_valid = true;

        prosper::gpu::set_guest_gpu_write_origin("gpu-preserving");
        const size_t preserved =
            prosper::test::invalidate_persistent_ds_guest_write(kHtile, 4096);
        prosper::gpu::set_guest_gpu_write_origin(nullptr);
        check(preserved == 1 && !image.depth_valid && !image.stencil_valid,
              "a byte-preserving HTILE rewrite still invalidates both retained aspects (#3089)");

        // The changed-origin arm is kept beside it so the assertion above cannot pass merely
        // because HTILE invalidation stopped depending on the origin in the wrong direction.
        image.depth_valid = true;
        image.stencil_valid = true;
        prosper::gpu::set_guest_gpu_write_origin("compute-writeback(buffer-full)");
        const size_t changed =
            prosper::test::invalidate_persistent_ds_guest_write(kHtile, 4096);
        prosper::gpu::set_guest_gpu_write_origin(nullptr);
        check(changed == 1 && !image.depth_valid && !image.stencil_valid,
              "a changed HTILE write still invalidates both aspects");

        // ...and the origin must not have become load-bearing anywhere else: a write that touches
        // NO plane of this surface stays inert under the preserving origin, so the arms above
        // measure the HTILE decision rather than a blanket invalidate-everything.
        image.depth_valid = true;
        image.stencil_valid = true;
        prosper::gpu::set_guest_gpu_write_origin("gpu-preserving");
        const size_t unrelated =
            prosper::test::invalidate_persistent_ds_guest_write(kHtile - 0x100000ull, 4096);
        prosper::gpu::set_guest_gpu_write_origin(nullptr);
        check(unrelated == 0 && image.depth_valid && image.stencil_valid,
              "a preserving write outside every plane of this surface invalidates nothing");
    }

    if (failures) { std::fprintf(stderr, "== FAIL: %d ==\n", failures); return 1; }
    std::fprintf(stderr, "== PASS ==\n");
    return 0;
}
