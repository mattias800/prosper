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

// The backend maps essentially every raw guest format onto a real VkFormat, with 0 the only
// "undefined" value. Modelled directly so this test needs no backend.
bool format_defined(uint32_t raw) { return raw != 0u; }

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

    // ...and neither is a base whose format was never seen.
    {
        auto draw = make_draw();
        draw.color_targets[2].base = kMrt2;
        draw.ps.color_targets[2].format = 0u;
        draw.ps.color_targets[2].write_mask = 0xfu;
        CHECK(mrt_active_color(draw, 2, format_defined) == 0u);
        CHECK(!mrt_draw_binds_target(draw, kMrt2, format_defined));
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

    if (failures == 0) std::printf("mrt_binding: OK\n");
    return failures == 0 ? 0 : 1;
}
