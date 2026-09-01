// #3204: the image-identity predicates, extracted from execute_item so they can be tested at all.
//
// These decide whether two bindings alias ONE Vulkan image. Callers depend on that aliasing: GTA V's
// 4K output shader writes the four 8x8 quadrants of each 16x16 block through four bindings at one
// guest address, so a binding that stops aliasing drops three of four stores and renders a
// checkerboard. #3205 was exactly that, introduced by a field added to an unnamed local expression
// no test could reach.
//
// Each arm below states a rule the aliasing depends on, and each has a counter-arm: a predicate that
// returned a constant would fail the pair.
#include "gpu/resources/image_identity.hpp"
#include <cstdio>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); ++fails; } \
                         else printf("  [ok]   %s\n", m); } while (0)

static ShaderResource base_descriptor() {
    ShaderResource r{};
    r.gpu_addr = 0x2046ce0000ull;
    r.size = 2560 * 1440 * 4;
    r.width = 2560; r.height = 1440; r.depth = 1;
    r.num_components = 4;
    r.declared_mip_levels = 1;
    r.host_data_size = 2560 * 1440 * 4;
    return r;
}

int main() {
    printf("== image identity (#3204) ==\n");
    const ComputeImageViewShape shape{false, 1, 1};

    // --- host backing -------------------------------------------------------------------------
    {
        ShaderResource a = base_descriptor(), b = base_descriptor();
        // THE ARM GTA V DEPENDS ON. A capture materializes each descriptor's bytes into its own
        // blob, so two bindings for one guest range have DIFFERENT host_data pointers. They must
        // still be the same backing, via address + size.
        static uint8_t blob_a[4], blob_b[4];
        a.host_data = blob_a; b.host_data = blob_b;
        CHECK(shader_resource_same_host_backing(a, b),
              "different host_data pointers at one gpu_addr are still the same backing");

        // Counter-arm: a genuinely different address is NOT the same backing, so the rule above is
        // not just "always true".
        b.gpu_addr = a.gpu_addr + 0x1000;
        CHECK(!shader_resource_same_host_backing(a, b),
              "...but a different gpu_addr is not");

        // And a null gpu_addr cannot vouch for identity -- only the pointer can.
        ShaderResource c = base_descriptor(), d = base_descriptor();
        c.gpu_addr = d.gpu_addr = 0; c.host_data = blob_a; d.host_data = blob_b;
        CHECK(!shader_resource_same_host_backing(c, d),
              "a zero gpu_addr does not make two distinct pointers the same backing");
    }

    // --- view identity ------------------------------------------------------------------------
    {
        ShaderResource a = base_descriptor(), b = base_descriptor();
        CHECK(shader_resource_same_view(a, b, shape, shape, true),
              "two identical descriptors are the same view");
        CHECK(!shader_resource_same_view(a, b, shape, shape, false),
              "...unless the caller's backing representation differs");

        const ComputeImageViewShape storage_shape{true, 1, 1};
        CHECK(!shader_resource_same_view(a, b, shape, storage_shape, true),
              "a storage binding and a sampled binding are never the same view");

        const ComputeImageViewShape deeper{false, 2, 1};
        CHECK(!shader_resource_same_view(a, b, shape, deeper, true),
              "a different realized texel depth is a different view");
        const ComputeImageViewShape layered{false, 1, 6};
        CHECK(!shader_resource_same_view(a, b, shape, layered, true),
              "a different realized layer count is a different view");

        b.width += 1;
        CHECK(!shader_resource_same_view(a, b, shape, shape, true),
              "a different extent is a different view");
    }

    // --- #3205 regression, at the level that actually decides aliasing --------------------------
    {
        ShaderResource a = base_descriptor(), b = base_descriptor();
        // One construction path fills provenance in, the other leaves it unset. Both single-level:
        // the provenance describes nothing, and must not split the image.
        b.mip_chain_element_width = 2560;
        b.mip_chain_element_height = 1440;
        b.mip_chain_bytes_per_block = 4;
        CHECK(shader_resource_same_view(a, b, shape, shape, true),
              "#3205: single-level descriptors alias when provenance is set on one side only");

        // Counter-arm: with a chain declared, that same difference DOES mean different images.
        a.declared_mip_levels = b.declared_mip_levels = 4;
        CHECK(!shader_resource_same_view(a, b, shape, shape, true),
              "...but with a chain declared, differing provenance is a different view");
    }

    // --- dcc identity -------------------------------------------------------------------------
    {
        ShaderResource a = base_descriptor(), b = base_descriptor();
        CHECK(shader_resource_same_dcc_identity(a, b), "identical dcc state matches");
        b.compression_enabled = !b.compression_enabled;
        CHECK(!shader_resource_same_dcc_identity(a, b),
              "a different compression state is a different image");
        CHECK(!shader_resource_same_view(a, b, shape, shape, true),
              "...and that difference reaches the view predicate");
    }

    // --- sampler identity ----------------------------------------------------------------------
    {
        ShaderResource a = base_descriptor(), b = base_descriptor();
        CHECK(shader_resource_same_sampler(a, b), "identical sampler state matches");
        b.max_aniso_ratio += 1;
        CHECK(!shader_resource_same_sampler(a, b), "a different anisotropy is a different sampler");
        ShaderResource c = base_descriptor(), d = base_descriptor();
        d.swizzle[2] = d.swizzle[2] + 1;
        CHECK(!shader_resource_same_sampler(c, d), "a different swizzle is a different sampler");
    }

    printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
