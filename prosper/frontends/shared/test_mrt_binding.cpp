// One active-colour-binding policy, exercised through a real DrawItem.
//
// Pass grouping and same-pass feedback detection must agree about what an active binding is. They
// did not: the feedback path carried a second, looser copy that fell back to the named slot-0/1
// write mask whenever the ARRAY mask read zero, and never required a defined format. That
// classifies stale named state as a live binding, which denies the authoritative direct-GPU path --
// and where no CPU snapshot exists, degrades to guest bytes rather than to a slower correct source.
#include <cstdio>
#include <cstdint>

#include "mrt_binding.hpp"

static int failures = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); ++failures; } } while (0)

namespace {

// Production's predicate is `backend_color_format(raw) != VK_FORMAT_UNDEFINED`, and
// backend_color_format's default branch maps EVERY unrecognised value -- including zero -- to
// R8G8B8A8_UNORM. So the production predicate is total: it returns true for every input, and the
// format term in the active-binding rule never rejects anything.
//
// That is deliberate, not an oversight. Synthetic callers and legacy captures omit CB_COLOR_INFO and
// rely on the backend's established RGBA8 fallback; requiring a decoded format would discard those
// otherwise-valid draws (the same reasoning the executor's `color_effect` check records). A live
// measurement agrees: PROSPER_MRT_CENSUS reported "format known" for every slot in all 16,384 pass
// groups of a routed GTA V boot -- the column is constant because the predicate is.
//
// Modelled here as the constant it is. An earlier version of this test modelled `raw != 0` and
// asserted a zero format made a slot inactive, which is a behaviour production does not have -- the
// central policy test proving the opposite of the policy. `mrt_format_total_mapping` in
// test_recompiled_fragment pins the backend mapping this constant depends on, so a change there
// fails rather than silently invalidating this file.
bool format_defined(uint32_t raw) { (void)raw; return true; }

prosper::gpu::DrawItem make_draw() {
    prosper::gpu::DrawItem draw{};
    return draw;
}

}  // namespace

int main() {
    using prosper::frontend::mrt_active_color;
    using prosper::frontend::mrt_active_color_count;
    using prosper::frontend::mrt_draw_binds_target;

    constexpr uint64_t kMrt0 = 0x2050000000ull;
    constexpr uint64_t kMrt2 = 0x2083e00000ull;

    // A slot in the ARRAY representation is active with base + format + mask.
    {
        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 37u;
        draw.ps.color_targets[2].write_mask = 0xfu;
        CHECK(mrt_active_color(draw, 2, format_defined) == kMrt2);
        CHECK(mrt_draw_binds_target(draw, kMrt2, format_defined));
        CHECK(mrt_active_color_count(draw, format_defined) == 3u);
        // The case the MRT0-only feedback rule missed entirely.
        CHECK(!mrt_draw_binds_target(draw, 0x2099cc0000ull, format_defined));
    }

    // A base with a ZERO write mask is not a target. Stale bases are sticky in the guest's register
    // file, so treating one as a binding both over-counts attachments and manufactures feedback.
    {
        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 37u;
        draw.ps.color_targets[2].write_mask = 0u;
        CHECK(mrt_active_color(draw, 2, format_defined) == 0u);
        CHECK(!mrt_draw_binds_target(draw, kMrt2, format_defined));
    }

    // A base whose format was never seen IS still active: zero means "use the backend's RGBA8
    // fallback", not "undefined". Asserted explicitly because the opposite reads as the safer
    // expectation and is wrong -- rejecting it would drop attachments for every synthetic caller and
    // every legacy capture that omits CB_COLOR_INFO.
    {
        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 0u;
        draw.ps.color_targets[2].write_mask = 0xfu;
        CHECK(mrt_active_color(draw, 2, format_defined) == kMrt2);
        CHECK(mrt_draw_binds_target(draw, kMrt2, format_defined));
    }

    // Legacy shape: DrawItem predates the complete array, and captures through v33 carry the first
    // two attachments in the named fields. With the array ABSENT the named state is authoritative.
    {
        auto draw = make_draw();
        draw.color0_base = kMrt0;
        draw.ps.color0_format = 37u;
        draw.ps.color_write_mask = 0xfu;
        CHECK(mrt_active_color(draw, 0, format_defined) == kMrt0);
        CHECK(mrt_draw_binds_target(draw, kMrt0, format_defined));
    }

    // THE defect this policy exists to prevent. The array representation IS present for slot 0 and
    // says the slot is masked off, while stale named state still carries a non-zero mask. The looser
    // rule fell back to the named mask whenever the array mask read zero and called this a binding.
    {
        auto draw = make_draw();
        draw.color_targets[0].base = kMrt0;
        draw.ps.color_targets[0].format = 37u;
        draw.ps.color_targets[0].write_mask = 0u;   // the array says: masked off
        draw.color0_base = kMrt0;
        draw.ps.color0_format = 37u;
        draw.ps.color_write_mask = 0xfu;            // stale named state says otherwise
        CHECK(mrt_active_color(draw, 0, format_defined) == 0u);
        CHECK(!mrt_draw_binds_target(draw, kMrt0, format_defined));
    }

    // A zero address never binds, and a draw with nothing bound binds nothing.
    {
        auto draw = make_draw();
        CHECK(!mrt_draw_binds_target(draw, 0, format_defined));
        CHECK(!mrt_draw_binds_target(draw, kMrt0, format_defined));
        CHECK(mrt_active_color_count(draw, format_defined) == 1u);
    }

    // The two materialization decisions, at the seams that own their gate. Written against the
    // decisions rather than the helper because the previous round's tests called
    // mrt_draw_binds_target() directly and stayed green when either production site was reverted to
    // `sampled != draw.color0_base`.
    {
        using prosper::frontend::mrt_direct_serves;
        using prosper::frontend::mrt_uniform_live_serves;

        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 37u;
        draw.ps.color_targets[2].write_mask = 0xfu;

        // Sampling an unrelated surface: the direct image serves.
        CHECK(mrt_direct_serves(draw, 0x2099cc0000ull, /*is_storage_image=*/false, /*img_dim=*/1u,
                                /*extent_compatible=*/true, /*has_persistent_target=*/true,
                                format_defined));
        // Sampling THIS pass's MRT2: it must not, or the descriptor and the colour attachment
        // become the same image.
        CHECK(!mrt_direct_serves(draw, kMrt2, false, 1u, true, true, format_defined));
        // The non-feedback preconditions still gate it.
        CHECK(!mrt_direct_serves(draw, 0x2099cc0000ull, /*is_storage_image=*/true, 1u, true, true,
                                 format_defined));
        CHECK(!mrt_direct_serves(draw, 0x2099cc0000ull, false, /*img_dim=*/5u, true, true,
                                 format_defined));
        CHECK(!mrt_direct_serves(draw, 0x2099cc0000ull, false, 1u, /*extent_compatible=*/false,
                                 true, format_defined));
        CHECK(!mrt_direct_serves(draw, 0x2099cc0000ull, false, 1u, true,
                                 /*has_persistent_target=*/false, format_defined));

        // The uniform fast path carries the same gate. It and mrt_direct_serves must agree: if this
        // one said yes on a collision the CPU materialisation would be skipped, and the direct bind
        // would then be refused with no snapshot left to fall back to.
        CHECK(mrt_uniform_live_serves(draw, 0x2099cc0000ull, /*preconditions=*/true,
                                      format_defined));
        CHECK(!mrt_uniform_live_serves(draw, kMrt2, true, format_defined));
        CHECK(!mrt_uniform_live_serves(draw, 0x2099cc0000ull, /*preconditions=*/false,
                                       format_defined));

        // A slot that is bound but MASKED OFF is not a target, so sampling it is not feedback and
        // the direct path must remain available -- denying it would push a live surface onto a
        // slower path, or onto guest bytes when no snapshot exists.
        auto masked = make_draw();
        masked.color_targets[2].base = kMrt2;
        masked.ps.color_targets[2].format = 37u;
        masked.ps.color_targets[2].write_mask = 0u;
        CHECK(mrt_direct_serves(masked, kMrt2, false, 1u, true, true, format_defined));
    }

    if (failures == 0) std::printf("mrt_binding: OK\n");
    return failures == 0 ? 0 : 1;
}
