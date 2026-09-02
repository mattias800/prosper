// #3195: the LOGICAL/PHYSICAL extent split for an R32_UINT StorageImage lowered to a linear atomic
// SSBO, extracted from execute_item so it can be tested at all.
//
// #3195 asked whether a ternary selecting between two byte-identical calls
// (`atomic_image ? resource_bytes_for(r, guest_bytes) : resource_bytes_for(r, guest_bytes)`) was a
// dead condition or an atomic arm that had lost its `atomic_slice_bytes` extent. It is dead: the
// arms converged when the NON-atomic one adopted the bounded call, and the atomic arm never
// changed. What made the condition redundant is the invariant below -- `guest_bytes` already
// carries the PHYSICAL padded footprint by the time the source pointer is taken.
//
// That invariant is what these arms pin. Every one has a counter-arm: a helper that returned the
// LOGICAL extent (the shape the "dropped extent" reading imagined), or that ignored the layer
// count, fails at least one of them rather than silently under-bounding the readability probe by
// exactly the tile padding.
#include "gpu/resources/atomic_image_staging.hpp"
#include <cstdio>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); ++fails; } \
                         else printf("  [ok]   %s\n", m); } while (0)

// Sonic Racing: CrossWorlds' full-screen atomic dispatch (#2265), the surface the padding was
// measured on.
static ShaderResource crossworlds_surface() {
    ShaderResource r{};
    r.cls = ResourceClass::StorageImage;
    r.format = DataFormat::Uint32;
    r.num_components = 1;
    r.img_dim = 1;              // plain 2D
    r.depth = 1;
    r.width = 3840;
    r.height = 2160;
    r.tile_mode = 27;
    r.gpu_addr = 0x2046ce0000ull;
    return r;
}

int main() {
    printf("== atomic-image staging extents (#3195 / #2265) ==\n");

    // --- the discriminator: physical is NOT logical for a tiled surface -----------------------
    {
        const ShaderResource r = crossworlds_surface();
        const AtomicImageStagingExtents e = atomic_image_staging_extents(r);
        CHECK(shader_resource_supports_atomic_image_buffer(r),
              "the measured surface really is an atomic-image candidate");
        CHECK(e.valid, "CrossWorlds' 3840x2160 tile-27 surface is an expressible layout");
        CHECK(e.layers == 1u, "a plain 2D view stages exactly one layer");
        CHECK(e.linear_bytes == 33177600ull,
              "logical extent is the tight w*h*4 the shader indexes");
        CHECK(e.slice_bytes == 33423360u,
              "physical slice is padded to whole micro-tiles (2160 rows -> 2176)");
        CHECK(e.guest_bytes == e.slice_bytes,
              "one layer: the guest footprint is one slice");
        // THE ARM THE "dropped extent" READING WOULD FAIL. A helper handing back the logical size
        // here under-bounds the readability probe by 245,760 bytes, and the detile then walks past
        // the region proven readable -- which is precisely what asking `resource_bytes_for` with
        // the wrong extent would do.
        CHECK(e.guest_bytes > e.linear_bytes,
              "the guest bound STRICTLY exceeds the logical extent (245,760 bytes of padding)");
        CHECK(e.guest_bytes - static_cast<size_t>(e.linear_bytes) == 245760u,
              "and by exactly the measured padding");
    }

    // --- layering: the guest bound must cover the LAST layer the detile loop reads -------------
    {
        ShaderResource r = crossworlds_surface();
        r.img_dim = 5;          // SQ 2D_ARRAY
        r.depth = 3;            // layer COUNT for an arrayed view
        const AtomicImageStagingExtents e = atomic_image_staging_extents(r);
        CHECK(e.valid, "a three-layer array of that surface is expressible");
        CHECK(e.layers == 3u, "an arrayed view stages `depth` layers");
        CHECK(e.slice_bytes == 33423360u, "the slice stride is per-layer and unchanged by layering");
        // Counter-arm for a helper that forgot the layer count: it would report one slice here.
        CHECK(e.guest_bytes == 3u * 33423360u, "the guest footprint is layers * slice stride");
        CHECK(e.linear_bytes == 3ull * 33177600ull, "the staging buffer packs all layers tightly");
        // The upload steps `slice_bytes` per layer and reads a full slice at the last one, so the
        // bound must cover exactly that -- no more, and critically no less.
        const size_t last_layer_end = (e.layers - 1u) * e.slice_bytes + e.slice_bytes;
        CHECK(last_layer_end == e.guest_bytes,
              "the bound ends exactly where the last layer's detile source ends");
    }

    // --- linear surfaces: a declared row pitch is padding too ---------------------------------
    {
        ShaderResource r = crossworlds_surface();
        r.tile_mode = 0;
        r.width = 100; r.height = 50;
        r.linear_row_pitch_bytes = 512;      // tight row is 400
        const AtomicImageStagingExtents e = atomic_image_staging_extents(r);
        CHECK(e.valid, "a linear surface with a padded row pitch is expressible");
        CHECK(e.slice_bytes == 512u * 50u, "the physical slice walks the DECLARED row pitch");
        CHECK(e.linear_bytes == 100ull * 50ull * 4ull, "the logical extent stays tight");
        CHECK(e.guest_bytes > e.linear_bytes, "padded linear is physically larger too");
    }
    {
        ShaderResource r = crossworlds_surface();
        r.tile_mode = 0;
        r.width = 100; r.height = 50;
        r.linear_row_pitch_bytes = 0;        // undeclared -> tight
        const AtomicImageStagingExtents e = atomic_image_staging_extents(r);
        CHECK(e.valid, "an unpadded linear surface is expressible");
        CHECK(e.slice_bytes == 100u * 4u * 50u, "an undeclared pitch falls back to the tight row");
        CHECK(e.guest_bytes == static_cast<size_t>(e.linear_bytes),
              "with no padding anywhere the two extents coincide -- the ONLY shape where they may");
    }

    // --- layouts the transform cannot express -------------------------------------------------
    {
        ShaderResource r = crossworlds_surface();
        r.tile_mode = 0;
        r.width = 100; r.height = 50;
        r.linear_row_pitch_bytes = 320;      // narrower than the 400-byte tight row
        CHECK(!atomic_image_staging_extents(r).valid,
              "a row pitch narrower than one tight row is rejected, not read skewed");
    }
    {
        ShaderResource r = crossworlds_surface();
        r.img_dim = 5;
        r.depth = 200;                       // 200 * 33 MB overflows the 32-bit probe
        CHECK(!atomic_image_staging_extents(r).valid,
              "an extent that will not fit the 32-bit guest-readability probe is rejected");
    }
    {
        ShaderResource r = crossworlds_surface();
        r.height = 0;                        // empty footprint
        CHECK(!atomic_image_staging_extents(r).valid, "an empty footprint is rejected");
    }

    printf(fails ? "FAILED (%d)\n" : "PASSED\n", fails);
    return fails ? 1 : 0;
}
