// #3205: mip-chain provenance is part of an image's IDENTITY only when a chain exists.
//
// Why this file exists. #3048 added five `mip_chain_*` fields describing an allocation-wide mip
// chain, and appended them to the predicate deciding whether two image descriptors alias ONE
// Vulkan image. The fields are not populated uniformly: `image_base_level_view` fills them in,
// `unmapped_format_image_view` deliberately leaves them zero ("zero element extent = not
// modelled"). Two IDENTICAL single-level descriptors for the same guest address therefore compared
// unequal purely by which construction path built them.
//
// The consequence is not a cache miss. Descriptors that stop comparing equal stop aliasing one
// image, and GTA V's 4K output shader writes the four 8x8 quadrants of each 16x16 block through
// four bindings at a single address -- so three of four stores were dropped and a checkerboard
// covered the frame. Measured on PPSA04263 before the guard: 33,615 aliasing rejections across 46
// distinct addresses in under a minute, every one with `declared_mip_levels == 1` on both sides.
#include "gpu/agc/agc_shader_layout.hpp"
#include <cstdio>

using prosper::gpu::ShaderResource;
using prosper::gpu::shader_resource_mip_chain_provenance_matches;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); ++fails; } \
                         else printf("  [ok]   %s\n", m); } while (0)

// A descriptor as `image_base_level_view` leaves it: provenance filled in.
static ShaderResource with_provenance(uint32_t declared_levels) {
    ShaderResource r{};
    r.declared_mip_levels = declared_levels;
    r.mip_chain_element_width = 2560;
    r.mip_chain_element_height = 1440;
    r.mip_chain_bytes_per_block = 4;
    r.mip_chain_max_level = 0;
    r.mip_chain_base_level = 0;
    return r;
}

// The same descriptor as `unmapped_format_image_view` leaves it: provenance unset.
static ShaderResource without_provenance(uint32_t declared_levels) {
    ShaderResource r{};
    r.declared_mip_levels = declared_levels;
    return r;
}

int main() {
    printf("== mip provenance identity (#3205) ==\n");

    // THE REGRESSION. Both single-level, provenance set on one side only. These describe the same
    // image and must alias; the observed failure had exactly this shape (ew=0/2560, eh=0/1440,
    // bpb=0/4, decl=1/1).
    CHECK(shader_resource_mip_chain_provenance_matches(without_provenance(1), with_provenance(1)),
          "single-level descriptors match when provenance is set on ONE side only");
    CHECK(shader_resource_mip_chain_provenance_matches(with_provenance(1), without_provenance(1)),
          "...and the comparison is symmetric");

    // Control: the arm above must not pass merely because the predicate returns true for
    // everything. With a chain declared, differing provenance really does mean different images.
    CHECK(!shader_resource_mip_chain_provenance_matches(without_provenance(4), with_provenance(4)),
          "MULTI-level descriptors with differing provenance do NOT match");

    // #3048's intent, preserved: identical provenance on a real chain still matches.
    CHECK(shader_resource_mip_chain_provenance_matches(with_provenance(4), with_provenance(4)),
          "multi-level descriptors with identical provenance match");

    // The boundary. One side declares a chain, the other does not: the provenance is meaningful to
    // at least one of them, so it must be compared rather than waved through.
    CHECK(!shader_resource_mip_chain_provenance_matches(without_provenance(1), with_provenance(4)),
          "a chain on ONE side still compares provenance (not waved through)");

    // Each field must count on its own -- a predicate that only looked at, say, element width
    // would pass every arm above.
    for (int field = 0; field < 5; ++field) {
        ShaderResource a = with_provenance(4), b = with_provenance(4);
        const char* name = "";
        switch (field) {
            case 0: b.mip_chain_element_width += 1;  name = "element_width";  break;
            case 1: b.mip_chain_element_height += 1; name = "element_height"; break;
            case 2: b.mip_chain_bytes_per_block += 1;name = "bytes_per_block";break;
            case 3: b.mip_chain_max_level += 1;      name = "max_level";      break;
            case 4: b.mip_chain_base_level += 1;     name = "base_level";     break;
        }
        char msg[128];
        snprintf(msg, sizeof msg, "a difference in %s alone prevents a match", name);
        CHECK(!shader_resource_mip_chain_provenance_matches(a, b), msg);
    }

    printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
