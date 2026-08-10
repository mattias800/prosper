// test_descriptor_array_render — a runtime-selected descriptor ARRAY actually binds (#2412 stage 5).
//
// The backend now sizes the descriptor pool, the layout binding, the write, and the
// VkDescriptorBufferInfo run table from a resource's arity instead of hardcoding 1. Nothing in the
// guest path produces an arity yet, so without this test the whole N>1 half of that change is
// structurally inexpressible in the suite: 230 green tests would say nothing about it, because none
// of them can build an array. That is the trap this file exists to close.
//
// The shader does NOT need to index the array for this to be a real test. A SPIR-V module declaring
// an ordinary single binding is a declared array size of 1, which Vulkan permits against a layout
// whose descriptorCount is larger — the shader simply reads element 0. So binding a 3-entry array to
// the existing vertex-fetch VS exercises every one of the four sites and lets a PIXEL assert the
// result:
//
//   Case A  arity 3, entries[0] = the fullscreen quad          -> GREEN
//           Proves the layout, the write and the run table agree: a layout declaring 3 against a
//           write supplying 1 is a validation error, and a mis-sized run table binds the wrong
//           buffer. It does NOT prove the pool accounting -- measured: reverting the pool fix alone
//           leaves every arm here passing, because RADV over-allocates rather than returning
//           VK_ERROR_OUT_OF_POOL_MEMORY. The pool change is spec-correctness for a strict
//           implementation, and this test cannot see it on this driver. Said explicitly because the
//           first version of this comment claimed otherwise and a mutation arm refuted it.
//   Case B  arity 3, entries[0] = an off-screen decoy          -> NOT green
//           The discriminator that makes Case A mean something: element 0 really is entries[0], not
//           whatever happened to land there. Note what B alone cannot show: with ONE resource in the
//           draw the run-table offset is trivially 0, so the offset machinery is untested by A and B.
//           Case D covers that.
//   Case D  two bindings, the FIRST with arity 2                -> GREEN
//           The second resource's run starts at offset 2, so an implementation indexing the run table
//           by resource index instead of by offset reads the first binding's second entry and the
//           quad never arrives. This is the only arm that exercises a non-zero offset.
//   Case C  arity 1 baseline (no table_entries)                -> byte-identical to Case A
//           Proves routing element 0 through the array path delivers exactly what the
//           single-descriptor path always delivered, rather than something merely also green.
#include "../src/gpu/rdna2_to_spirv.hpp"
#include "../src/gpu/shader_resources.hpp"
#include "../src/gpu/render_state.hpp"
#include "render_runner.h"
#include <cstdio>
#include <cstdint>
#include <vector>

using namespace prosper::gpu;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); fails++; } \
                         else       { printf("  [ok]   %s\n", m); } } while (0)

int main() {
    printf("== test_descriptor_array_render ==\n");
    const uint32_t W = 64, H = 64;

    // Same blobs as test_indexed_render: a VS that fetches (x,y) from the storage buffer at binding 3
    // indexed by vid, and a green PS.
    const uint32_t vs[] = {
        0x7e060280u, 0x7e0802f2u, 0xe0042000u, 0x80020100u, 0xf80008cfu, 0x04030201u, 0xbf810000u,
    };
    const uint32_t ps[] = { 0x7E000280u, 0x7E0202F2u, 0x7E040280u, 0x7E0602F2u, 0xF800180Fu,
                            0x03020100u, 0xBF810000u };

    ShaderResourceTable rt;
    ShaderResource vb{};
    vb.cls = ResourceClass::VertexBuffer; vb.format = DataFormat::Float32; vb.num_components = 2;
    vb.binding = 3; vb.stride = 8; vb.sgpr_base = 8;
    rt.resources.push_back(vb);

    std::vector<uint32_t> vert = recompile_vertex(vs, sizeof(vs)/sizeof(vs[0]), &rt);
    std::vector<uint32_t> frag = recompile_fragment(ps, sizeof(ps)/sizeof(ps[0]));
    CHECK(!vert.empty() && !frag.empty(), "recompiled vertex-fetch VS + green PS");
    if (vert.empty() || frag.empty()) { printf("== FAIL ==\n"); return 1; }

    auto f = [](float v) { union { float f; uint32_t u; } c; c.f = v; return c.u; };
    auto isGreen = [&](const std::vector<uint8_t>& px, uint32_t x, uint32_t y) {
        const uint8_t* p = &px[((size_t)y * W + x) * 4];
        return p[1] > 0x80 && p[0] < 0x40 && p[2] < 0x40;
    };
    auto greenAt9 = [&](const std::vector<uint8_t>& px) {
        const uint32_t xs[] = {0, W/2, W-1}, ys[] = {0, H/2, H-1};
        uint32_t g = 0; for (uint32_t y : ys) for (uint32_t x : xs) if (isGreen(px, x, y)) g++;
        return g == 9u;
    };

    // A full-viewport quad in perimeter order, and an entirely off-screen decoy of the same shape.
    const std::vector<uint32_t> quad = { f(-1.f), f(-1.f),  f(-1.f), f(1.f),
                                         f( 1.f), f( 1.f),  f( 1.f), f(-1.f) };
    const std::vector<uint32_t> decoy = { f(9.f), f(9.f),  f(9.f), f(11.f),
                                          f(11.f), f(11.f), f(11.f), f(9.f) };
    ResolvedPipelineState list_ps; list_ps.topology = 3;   // VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST

    // `entries` empty -> the single-descriptor path (arity 1). Otherwise an arity-N array.
    auto draw_with = [&](const std::vector<uint32_t>& single,
                         std::vector<std::vector<uint32_t>> entries) {
        prosper::test::BackendDraw d;
        d.vs = vert; d.fs = frag; d.ps = &list_ps; d.vcount = 4; d.indices = {0,1,2, 2,3,0};
        prosper::test::FrameResource r; r.binding = 3; r.set = 0;
        if (entries.empty()) r.dwords = single; else r.table_entries = std::move(entries);
        d.R.push_back(std::move(r));
        return d;
    };

    // --- Case C first, so it is the baseline the others are compared against -----------------------
    std::vector<uint8_t> px_single = prosper::test::render_draws_rgba({ draw_with(quad, {}) }, W, H);
    CHECK(px_single.size() == (size_t)W*H*4, "arity-1 baseline produced a frame");
    if (px_single.size() != (size_t)W*H*4) { printf("== FAIL ==\n"); return 1; }
    CHECK(greenAt9(px_single), "arity-1 baseline fills the viewport GREEN");

    // --- Case A: a 3-entry array whose element 0 is the quad --------------------------------------
    std::vector<uint8_t> px_arr =
        prosper::test::render_draws_rgba({ draw_with({}, { quad, decoy, decoy }) }, W, H);
    CHECK(px_arr.size() == (size_t)W*H*4,
          "arity-3 array produced a frame at all (layout, write and run table agree)");
    if (px_arr.size() != (size_t)W*H*4) { printf("== FAIL ==\n"); return 1; }
    CHECK(greenAt9(px_arr), "a 3-entry descriptor array binds, and element 0 draws the GREEN quad");
    CHECK(px_arr == px_single,
          "routing element 0 through the array path is byte-identical to the single-descriptor path");
    // The graphics-pipeline cache must NOT hand Case A the pipeline Case C created (#2471). The two
    // draws share every byte of shader and every item of fixed state, so a key that omits descriptor
    // arity collides them — and the collision is invisible to every assertion above, because the
    // stale pipeline still draws a correct green quad. What it is not is spec-valid: the pipeline was
    // created under an arity-1 VkPipelineLayout and the sets are bound under an arity-3 one, which is
    // VUID-vkCmdDrawIndexed-None-08600, "Set 0 binding 0 descriptorCount 3 doesn't match 1". A
    // validation layer sees it; pixels never will. So the arity axis is asserted through the cache
    // decision instead.
    const prosper::test::BackendPipelineCacheStats after_a =
        prosper::test::backend_pipeline_cache_stats();
    CHECK(after_a.hits == 0 && after_a.misses == 1,
          "an arity-3 draw does not reuse the arity-1 baseline's pipeline (hits=0, misses=1)");

    // --- Case B: the discriminator — element 0 really is entries[0] -------------------------------
    std::vector<uint8_t> px_swapped =
        prosper::test::render_draws_rgba({ draw_with({}, { decoy, quad, quad }) }, W, H);
    CHECK(px_swapped.size() == (size_t)W*H*4, "arity-3 array with a decoy at element 0 produced a frame");
    if (px_swapped.size() != (size_t)W*H*4) { printf("== FAIL ==\n"); return 1; }
    CHECK(!greenAt9(px_swapped),
          "a decoy at element 0 is NOT green — element 0 really is entries[0]");
    CHECK(px_swapped != px_arr,
          "which entry sits at element 0 changes the picture (entries are not collapsed into one)");
    // The other half of the arity assertion, and the one that makes it mean something. Case B has
    // Case A's shaders AND Case A's arity, so it MUST hit — keying arity is meant to separate
    // arity-1 from arity-3, not to defeat the cache. Without this arm the check above is satisfied
    // by any change that makes every draw miss (disabling the cache, keying a pointer, keying a
    // counter), none of which fixes the layout pairing.
    const prosper::test::BackendPipelineCacheStats after_b =
        prosper::test::backend_pipeline_cache_stats();
    CHECK(after_b.hits == 1 && after_b.misses == 0,
          "a second arity-3 draw with the same shaders DOES reuse Case A's pipeline (hits=1)");

    // --- Case D: a non-zero run-table offset -------------------------------------------------------
    // Binding 2 comes FIRST and occupies two descriptors, so binding 3's run starts at offset 2. The
    // shader fetches only binding 3; binding 2 is a layout entry it does not declare, which Vulkan
    // permits. An implementation that wrote the second resource's info at dbi[resource_index] instead
    // of dbi[offset] would hand binding 3 the tail of binding 2's array and lose the quad.
    {
        prosper::test::BackendDraw d;
        d.vs = vert; d.fs = frag; d.ps = &list_ps; d.vcount = 4; d.indices = {0,1,2, 2,3,0};
        prosper::test::FrameResource pad; pad.binding = 2; pad.set = 0;
        pad.table_entries = { decoy, decoy };
        prosper::test::FrameResource quad_res; quad_res.binding = 3; quad_res.set = 0;
        quad_res.dwords = quad;
        d.R.push_back(std::move(pad));
        d.R.push_back(std::move(quad_res));
        std::vector<uint8_t> px_off = prosper::test::render_draws_rgba({ std::move(d) }, W, H);
        CHECK(px_off.size() == (size_t)W*H*4, "two bindings with a leading arity-2 array produced a frame");
        if (px_off.size() == (size_t)W*H*4)
            CHECK(greenAt9(px_off),
                  "binding 3 at run-table offset 2 still receives its own buffer (offsets are applied)");
    }

    // --- Arm 5: the layout must declare what the WRITE supplies, per class (#2477) ----------------
    // Arrays are implemented for storage buffers only. The texture / storage-image write and the
    // internal-GDS write each supply exactly ONE descriptor, so a layout that took `descriptor_arity()`
    // for those classes declared N against a write of 1 -- elements 1..N-1 never written, undefined
    // descriptors bound.
    //
    // Asserted on `written_descriptor_count()` rather than on a frame, and that is not a shortcut:
    // NO pixel assertion can see this class. The shader reads element 0, element 0 IS written, so the
    // quad is green whether the layout declared 1 or 3 -- exactly the blind spot that let #2471's
    // spec-invalid binding pass every arm it had. The only thing a rendered frame proves here is that
    // the guard did not break the working path, which the render below does check.
    {
        prosper::test::FrameResource buf{};
        buf.binding = 3; buf.set = 0;
        buf.table_entries = { quad, decoy, decoy };
        CHECK(buf.descriptor_arity() == 3 && buf.written_descriptor_count() == 3,
              "storage buffer: the array path writes N, so the layout declares N");

        // A texture carrying table_entries -- unreachable from any current producer, constructed BY
        // HAND here precisely because nothing else can construct it (charter: build one positive
        // instance of the class outside whatever produced the null).
        prosper::test::FrameResource tex{};
        tex.binding = 4; tex.set = 0;
        tex.has_uniform_color = true;          // makes is_texture() true without a pixel upload
        tex.table_entries = { quad, decoy, decoy };
        CHECK(tex.is_texture(), "control: the hand-built resource really is a texture class");
        CHECK(tex.descriptor_arity() == 3,
              "control: it really declares 3 entries, so the two counts CAN disagree here");
        CHECK(tex.written_descriptor_count() == 1,
              "texture: its write supplies one, so the layout must declare one -- not its arity");

        prosper::test::FrameResource gds{};
        gds.binding = 5; gds.set = 0;
        gds.is_internal_gds = true;
        gds.table_entries = { quad, decoy };
        CHECK(gds.written_descriptor_count() == 1,
              "internal GDS: its write supplies one, so the layout must declare one");

        // And the guard must not have disturbed the path that does work.
        std::vector<uint8_t> px_guard =
            prosper::test::render_draws_rgba({ draw_with({}, { quad, decoy, decoy }) }, W, H);
        CHECK(px_guard.size() == (size_t)W*H*4 && greenAt9(px_guard),
              "the storage-buffer array still renders GREEN with the per-class guard in place");

        // The assertions above cover the ACCESSOR. They do not cover the LAYOUT SITE that consumes
        // it, and that distinction is the whole defect: reverting only
        // `lb[i].descriptorCount = r.written_descriptor_count()` to `descriptor_arity()` -- i.e.
        // reverting the fix while leaving the accessor intact -- leaves every assertion above green.
        // Measured, not assumed.
        //
        // So drive the site. This draw carries the texture-with-`table_entries` alongside the buffer
        // the shader reads, which is the FIRST time the layout loop sees a class whose declared arity
        // and written count disagree. With the fix it declares 1 against a write of 1; without it, 3
        // against 1, and elements 1..2 are never written.
        //
        // And this arm IS ctest-visible, which is worth stating precisely because the obvious
        // expectation -- "no pixel assertion can see it, so only the validation layer can" -- is
        // wrong here, and I had written that before measuring it.
        //
        // Reverting the call site alone makes the POOL and the LAYOUT disagree: the pool reserves
        // `written_descriptor_count()` (1 sampler) while the layout declares `descriptor_arity()` (3),
        // so `vkAllocateDescriptorSets` requests 3 COMBINED_IMAGE_SAMPLER from a pool holding 1. On
        // this adapter that returns no set, the following `vkUpdateDescriptorSets` gets
        // `dstSet == VK_NULL_HANDLE`, and the test dies with **exit 139**. Measured both ways.
        //
        // Note this is the mirror image of #2471, where the pool arm was unverifiable because RADV
        // over-allocates rather than returning OUT_OF_POOL_MEMORY. A pool can be over-allocated
        // into; it cannot invent capacity for a descriptor TYPE it was never sized for.
        //
        // The validation layer adds the reason rather than the symptom -- `pool only has a total of 1`
        // and `dstSet is VK_NULL_HANDLE` -- so `vk_validation_scan.py` is still the instrument that
        // explains a failure here, and it is clean with the fix in place.
        //
        // WHAT THIS ARM'S DISCRIMINATION RESTS ON, because it is not self-contained (raised in review):
        // it discriminates via POOL EXHAUSTION, not via the frame. Trace the mutated path if the
        // allocation ever succeeds -- a more generously sized pool, a different adapter, or a driver
        // that over-allocates samplers the way RADV over-allocates buffers:
        //
        //     layout declares 3, write supplies 1  -> set allocates
        //     elements 1..2 undefined              -> shader reads element 0
        //     element 0 is written                 -> quad is GREEN -> this arm PASSES
        //
        // That is not hypothetical hand-waving: over-allocation is measured behaviour on this very
        // adapter for buffers (#2471), which is why the pool arm there was unverifiable. So if pool
        // sizing for COMBINED_IMAGE_SAMPLER ever becomes generous, this arm stops discriminating and
        // goes quiet rather than failing. If that happens, assert the emitted `descriptorCount`
        // directly instead of relying on the allocator to refuse.
        {
            prosper::test::BackendDraw d = draw_with(quad, {});
            prosper::test::FrameResource tex_bound{};
            tex_bound.binding = 4; tex_bound.set = 0;
            tex_bound.has_uniform_color = true;
            tex_bound.uniform_color = {0.25f, 0.5f, 0.75f, 1.0f};
            tex_bound.tw = 64; tex_bound.th = 32;
            tex_bound.table_entries = { quad, decoy, decoy };   // arity 3 on a class that writes 1
            d.R.push_back(std::move(tex_bound));
            std::vector<uint8_t> px_mixed = prosper::test::render_draws_rgba({ std::move(d) }, W, H);
            CHECK(px_mixed.size() == (size_t)W*H*4,
                  "a draw binding a texture whose declared arity exceeds its written count produced "
                  "a frame (the layout site saw the disagreement)");
            if (px_mixed.size() == (size_t)W*H*4)
                CHECK(greenAt9(px_mixed),
                      "and it still renders GREEN -- the guard keeps layout and write consistent "
                      "instead of declaring descriptors nothing writes");
        }
    }

    printf(fails ? "== FAIL ==\n" : "== PASS ==\n");
    return fails ? 1 : 0;
}
