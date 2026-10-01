// Execute the production ProvenVotes path at all four quad positions, with three measured
// helper invocations. Strict replay/source bytes and effective pipeline variants stay separate.
#include "fixtures/render_runner.h"
#include "fixtures/spirv_fragment_vote_execution.hpp"
#include "fixtures/spirv_fragment_vote_fixtures.hpp"
#include "fixtures/spirv_fragment_neutral_fixtures.hpp"
#include <cstdio>
#include <string_view>

namespace {
using prosper::gpu::FragmentWavePolicy;
namespace f = prosper::test::fragment_vote_execution;
int failures = 0;
void check(bool ok, const char* name) {
    std::printf("[%s] %s\n", ok ? "ok" : "FAIL", name);
    failures += !ok;
}
struct Frame {
    std::vector<uint8_t> pixels;
    prosper::test::BackendPipelineCacheStats pipelines;
};
Frame render(const std::vector<uint32_t>& vs, const prosper::gpu::SharedShaderWords& fs,
             uint64_t identity, FragmentWavePolicy policy, bool legacy_native = false) {
    prosper::test::BackendDraw draw;
    draw.vs = vs;
    draw.fs_shared = fs;
    draw.fs_identity = identity;
    draw.vcount = 3;
    draw.fragment_wave_policy = policy;
    draw.allow_partial_wave_fragment = legacy_native; // explicit diagnostic control, never live policy
    for (uint32_t set = 0; set < 2; ++set) {
        prosper::test::FrameResource cb; cb.binding = 2; cb.set = set; draw.R.push_back(cb);
        prosper::test::FrameResource vb; vb.binding = 3; vb.set = set; draw.R.push_back(vb);
    }
    auto pixels = prosper::test::render_draws_rgba({draw}, f::width, f::height);
    return {std::move(pixels), prosper::test::backend_pipeline_cache_stats()};
}
bool blue(const uint8_t* p) { return p[0] == 0 && p[1] == 0 && p[2] == 255 && p[3] == 255; }
void verify_clear(const Frame& frame, const char* name = "Strict refuses source Wave64 without inheriting a live pipeline") {
    bool clear = frame.pixels.size() == f::width * f::height * 4;
    if (clear) for (size_t i = 0; i < frame.pixels.size(); i += 4) clear &= blue(&frame.pixels[i]);
    check(clear, name);
}
void verify_derivatives(const Frame& frame, uint32_t x, uint32_t y, uint32_t expected_blue = 0) {
    check(frame.pixels.size() == f::width * f::height * 4, "proven module renders a target");
    if (frame.pixels.size() != f::width * f::height * 4) return;
    size_t covered = 0;
    for (size_t i = 0; i < frame.pixels.size(); i += 4) covered += !blue(&frame.pixels[i]);
    const auto* p = &frame.pixels[(y * f::width + x) * 4];
    std::printf("quad=(%u,%u) rgba=%u,%u,%u,%u covered=%zu\n", x & 1, y & 1,
                p[0], p[1], p[2], p[3], covered);
    check(covered == 1 && p[2] == expected_blue && p[3] == 255,
          "exactly the selected primitive-edge pixel renders with the expected final induction state");
    check(p[0] >= 31 && p[0] <= 32 && p[1] >= 31 && p[1] <= 32,
          "analytic dFdx(x)=dFdy(y)=1 includes helper values through the uniform vote branch");
}
void verify_buffer_authority(bool deterministic) {
    namespace inputs = prosper::test::fragment_votes;
    const auto source = std::make_shared<const std::vector<uint32_t>>(inputs::storage_predicate());
    const auto captured = *source;
    const auto vertex = f::edge_vertex(2, 2);
    const auto draw = [&](uint64_t identity) {
        prosper::test::BackendDraw d;
        d.vs = vertex; d.fs_shared = source; d.fs_identity = identity; d.vcount = 3;
        d.fragment_wave_policy = FragmentWavePolicy::ProvenVotes;
        for (uint32_t set = 0; set < 2; ++set) {
            prosper::test::FrameResource cb; cb.binding = 2; cb.set = set; d.R.push_back(cb);
            prosper::test::FrameResource vb; vb.binding = 3; vb.set = set; d.R.push_back(vb);
        }
        prosper::test::FrameResource values; values.binding = 0; values.dwords = {0, 0x3f800000u};
        d.R.push_back(values);
        prosper::test::FrameResource predicate; predicate.binding = 5; predicate.dwords = {1};
        d.R.push_back(predicate);
        return d;
    };
    const auto verify = [&](const std::vector<uint8_t>& pixels, bool admitted) {
        check(pixels.size() == f::width * f::height * 4, "buffer predicate renders a target");
        if (pixels.size() != f::width * f::height * 4) return;
        // Scalar Location0 defines red only. Do not assert unspecified G/B/A output components.
        const auto red = pixels[(2 * f::width + 2) * 4];
        check(red == (admitted ? 255 : 0), "buffer-vote admission follows actual device and whole-pass authority");
    };
    for (const bool readonly_first : {false, true}) {
        const uint64_t id = readonly_first ? 0xf3995001u : 0xf3995000u;
        for (const bool readonly : {readonly_first, !readonly_first, readonly_first}) {
            std::vector<prosper::test::BackendDraw> draws{draw(id)};
            if (!readonly) {
                // A later writer invalidates the WHOLE pass certificate. Its own Wave64 shader
                // refuses before recording; this is an authority control, not a raced GPU write.
                auto writer = draw(id + 0x100);
                writer.fs_shared = std::make_shared<const std::vector<uint32_t>>(
                    inputs::storage_predicate(false, false, true));
                draws.push_back(std::move(writer));
            }
            verify(prosper::test::render_draws_rgba(draws, f::width, f::height), readonly && deterministic);
        }
    }
    check(*source == captured, "buffer certificate/cache changes never alter shared captured shader bytes");
}
void verify_loop_votes(uint32_t x, uint32_t y, bool poison) {
    namespace loops = prosper::test::fragment_loop_votes;
    const auto vertex = f::edge_vertex(x, y);
    for (uint32_t bound = 0; bound < 4; ++bound) {
        const auto source = std::make_shared<const std::vector<uint32_t>>(
            loops::make_module(loops::Shape::Counter, bound, poison));
        const auto captured = *source;
        const auto lowered = prosper::gpu::lower_fragment_votes(*source);
        check(lowered.refusal == prosper::gpu::FragmentVoteRefusal::None && lowered.uniform_votes == 1,
              "initialized finite induction supplies one independent guest-wave vote");
        const uint64_t identity = 0xf4011000u + y * 256 + x * 16 + bound;
        verify_clear(render(vertex, source, identity, FragmentWavePolicy::Strict));
        for (const bool warm : {false, true}) {
            const auto frame = render(vertex, source, identity, FragmentWavePolicy::ProvenVotes);
            if (bound) verify_derivatives(frame, x, y, bound);
            else verify_clear(frame, "zero-trip effective loop leaves its defined blue initial output");
            check(frame.pipelines.hits == (warm ? 1u : 0u) && frame.pipelines.misses == (warm ? 0u : 1u),
                  "loop module uses the production cold/warm effective pipeline variant");
        }
        verify_clear(render(vertex, source, identity, FragmentWavePolicy::Strict));
        check(*source == captured, "loop lowering and pipeline reuse preserve shared captured bytes");
    }
    for (const auto shape : {loops::Shape::Nested, loops::Shape::CrossCarried, loops::Shape::BoolToggle,
                             loops::Shape::CounterWithDeadVote}) {
        const auto source = std::make_shared<const std::vector<uint32_t>>(loops::make_module(shape, 2, poison));
        verify_derivatives(render(vertex, source, 0, FragmentWavePolicy::ProvenVotes), x, y,
                           shape == loops::Shape::CrossCarried ? 10 : shape == loops::Shape::BoolToggle ? 9 : 2);
    }
    // An even number of toggles also ends true if the Boolean carry is accidentally frozen.
    // Odd trips finish false, so blue = 4*count observes the changing Boolean independently.
    for (const uint32_t bound : {1u, 3u}) {
        const auto source = std::make_shared<const std::vector<uint32_t>>(
            loops::make_module(loops::Shape::BoolToggle, bound, poison));
        verify_derivatives(render(vertex, source, 0, FragmentWavePolicy::ProvenVotes), x, y, 4 * bound);
    }
    for (const auto shape : {loops::Shape::VaryingTrip, loops::Shape::VaryingInit, loops::Shape::VaryingBoolUpdate,
                             loops::Shape::SecondUnsafeLoop}) {
        const auto source = std::make_shared<const std::vector<uint32_t>>(loops::make_module(shape));
        verify_clear(render(vertex, source, 0, FragmentWavePolicy::ProvenVotes),
                     "helper-dependent recurrence refuses rather than executing a narrower vote");
    }
}

void verify_per_draw_loop_bounds(bool deterministic) {
    namespace loops = prosper::test::fragment_loop_votes;
    const auto source = std::make_shared<const std::vector<uint32_t>>(loops::make_module(loops::Shape::BufferBound));
    for (const bool reverse : {false, true}) {
        std::vector<prosper::test::BackendDraw> draws;
        for (uint32_t i = 0; i < 2; ++i) {
            const uint32_t bound = reverse ? 1 - i : i;
            prosper::test::BackendDraw draw;
            draw.vs = f::edge_vertex(2 + bound, 2); draw.fs_shared = source;
            draw.fs_identity = 0xf4011b00u; draw.vcount = 3;
            draw.fragment_wave_policy = FragmentWavePolicy::ProvenVotes;
            for (uint32_t set = 0; set < 2; ++set) {
                prosper::test::FrameResource cb; cb.binding = 2; cb.set = set; draw.R.push_back(cb);
                prosper::test::FrameResource vb; vb.binding = 3; vb.set = set; draw.R.push_back(vb);
            }
            prosper::test::FrameResource input; input.binding = 5; input.dwords = {bound}; draw.R.push_back(input);
            draws.push_back(std::move(draw));
        }
        const auto pixels = prosper::test::render_draws_rgba(draws, f::width, f::height);
        // Guest per-draw expectations, NOT an oracle for native Vulkan Any: the host may pack
        // commands into one subgroup. Effective copies must keep the distinct descriptor values.
        const Frame frame{pixels, {}};
        if (deterministic) verify_derivatives(frame, 3, 2, 1);
        else verify_clear(frame, "ordinary robustness refuses both buffer-derived loop certificates");
    }
}

void verify_neutral_buffer_variants(bool deterministic) {
    namespace neutral = prosper::test::fragment_neutral;
    const auto source = std::make_shared<const std::vector<uint32_t>>(
        neutral::make_module(neutral::Shape::BufferPredicate));
    const auto captured = *source;
    const auto make_draw = [&](uint64_t identity) {
        prosper::test::BackendDraw draw;
        draw.vs = f::edge_vertex(2, 2); draw.fs_shared = source; draw.fs_identity = identity;
        draw.vcount = 3; draw.fragment_wave_policy = FragmentWavePolicy::ProvenVotes;
        prosper::test::FrameResource predicate; predicate.binding = 5; predicate.dwords = {0};
        draw.R.push_back(predicate);
        return draw;
    };
    for (const bool readonly_first : {false, true}) {
        const uint64_t identity = readonly_first ? 0xf4014501u : 0xf4014500u;
        uint32_t visit = 0;
        for (const bool readonly : {readonly_first, !readonly_first, readonly_first}) {
            std::vector<prosper::test::BackendDraw> draws{make_draw(identity)};
            if (!readonly) {
                // Invalidate the whole-pass certificate without actually racing the predicate:
                // this independent writer's Wave64 module refuses before command recording.
                auto writer = make_draw(identity + 0x100);
                writer.fs_shared = std::make_shared<const std::vector<uint32_t>>(
                    prosper::test::fragment_votes::storage_predicate(false, false, true));
                prosper::test::FrameResource values; values.binding = 0; values.dwords = {0, 0x3f800000u};
                writer.R.push_back(values); draws.push_back(std::move(writer));
            }
            const auto pixels = prosper::test::render_draws_rgba(draws, f::width, f::height);
            const Frame frame{pixels, prosper::test::backend_pipeline_cache_stats()};
            verify_derivatives(frame, 2, 2);
            // With robust2 these are distinct Copy(P)/Copy(TRUE) modules. Without it both passes
            // select the identical neutral module and should share a warm pipeline.
            const bool hit = visit >= (deterministic ? 2u : 1u);
            check(frame.pipelines.hits == (hit ? 1u : 0u) &&
                  frame.pipelines.misses == (hit ? 0u : 1u),
                  "pipeline cache names each admitted buffer-certificate effective module");
            ++visit;
        }
    }
    check(*source == captured, "dual-admission cache variants preserve captured source words");
}

void verify_neutral_votes(uint32_t x, uint32_t y, bool poison) {
    namespace neutral = prosper::test::fragment_neutral;
    const auto vertex = f::edge_vertex(x, y);
    for (const auto predicate : {neutral::Predicate::Helpers, neutral::Predicate::Visible,
                                neutral::Predicate::AllFalse, neutral::Predicate::AllTrue}) {
        const auto source = std::make_shared<const std::vector<uint32_t>>(
            neutral::make_module(neutral::Shape::Masked, predicate, poison));
        const auto captured = *source;
        const auto lowered = prosper::gpu::lower_fragment_votes(*source);
        check(lowered.neutral_votes == 1 && lowered.uniform_votes == 0,
              "neutral certificate does not promote source predicate into uniform facts");
        const bool visible = predicate == neutral::Predicate::Visible || predicate == neutral::Predicate::AllTrue;
        const uint64_t identity = 0xf4013000u + y * 256 + x * 16 + static_cast<uint32_t>(predicate);
        verify_clear(render(vertex, source, identity, FragmentWavePolicy::Strict));
        for (const bool warm : {false, true}) {
            const auto frame = render(vertex, source, identity, FragmentWavePolicy::ProvenVotes);
            verify_derivatives(frame, x, y, visible ? 64 : 0);
            check(frame.pipelines.hits == (warm ? 1u : 0u) && frame.pipelines.misses == (warm ? 0u : 1u),
                  "neutral TRUE module has its own production cold/warm pipeline variant");
        }
        verify_clear(render(vertex, source, identity, FragmentWavePolicy::Strict));
        check(*source == captured, "neutral pipeline/replay calls preserve immutable source words");
    }
    for (const auto shape : {neutral::Shape::Termination, neutral::Shape::UndefinedConjunction,
                             neutral::Shape::MaskedFloat, neutral::Shape::MaskedBitcastPoison}) {
        // All-false is the newly executed domain. No earlier Kill removes helpers before the
        // analytic derivative, and unselected poison/undefined values must not affect state.
        const auto source = std::make_shared<const std::vector<uint32_t>>(
            neutral::make_module(shape, neutral::Predicate::AllFalse, poison));
        verify_derivatives(render(vertex, source, 0, FragmentWavePolicy::ProvenVotes), x, y);
    }
    const auto terminated = std::make_shared<const std::vector<uint32_t>>(
        neutral::make_module(neutral::Shape::Termination, neutral::Predicate::AllTrue));
    verify_clear(render(vertex, terminated, 0, FragmentWavePolicy::ProvenVotes),
                 "neutral control preserves actual live Kill decisions for true predicates");
    for (const auto shape : {neutral::Shape::ScalarExport, neutral::Shape::TerminationExport,
                             neutral::Shape::LiveUnequalPhi, neutral::Shape::SecondConsumer,
                             neutral::Shape::PoisonConjunction, neutral::Shape::BitcastPoisonConjunction,
                             neutral::Shape::SecondUnsafeVote}) {
        const auto source = std::make_shared<const std::vector<uint32_t>>(neutral::make_module(shape));
        verify_clear(render(vertex, source, 0, FragmentWavePolicy::ProvenVotes),
                     "unsafe neutral export/controller refuses rather than executing a host-width approximation");
    }
}
} // namespace

int main(int argc, char** argv) {
    const bool poison = argc == 2 && std::string_view(argv[1]) == "--poison-helpers";
    const bool expect_no_robust2 = argc == 2 && std::string_view(argv[1]) == "--expect-no-robust2";
    if (argc != 1 && !poison && !expect_no_robust2) return 2;
    const auto& context = prosper::test::render_vk_ctx();
    if (!context.ok || !(context.subgroup_stages & VK_SHADER_STAGE_FRAGMENT_BIT) ||
        !(context.subgroup_operations & VK_SUBGROUP_FEATURE_QUAD_BIT) ||
        !(context.subgroup_operations & VK_SUBGROUP_FEATURE_VOTE_BIT)) {
        std::puts("SKIP: fragment subgroup helper witness is unavailable");
        return 77;
    }
    if (context.subgroup_size_control && context.min_subgroup_size <= 64 &&
        context.max_subgroup_size >= 64 &&
        (context.required_subgroup_size_stages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
        std::puts("SKIP: this device supplies native Wave64; the unsupported-width lowering would not run");
        return 77;
    }
    if (expect_no_robust2)
        check(!context.deterministic_storage_reads,
              "requested robust2 opt-out actually removes deterministic storage-read authority");
    const auto fragment = std::make_shared<const std::vector<uint32_t>>(f::derivative_fragment(poison));
    const auto captured = *fragment;
    const auto witness = std::make_shared<const std::vector<uint32_t>>(f::helper_witness());
    const auto effective = prosper::gpu::lower_fragment_votes(*fragment);
    check(effective.refusal == prosper::gpu::FragmentVoteRefusal::None && effective.uniform_votes == 1,
          "source has one independently lowerable uniform vote");
    check(prosper::gpu::fragment_spirv_required_subgroup_size(*fragment) == 64 &&
          prosper::gpu::fragment_spirv_required_subgroup_size(effective.words) == 0,
          "source and effective modules have distinct width contracts");
    for (uint32_t y = 2; y < 4; ++y) for (uint32_t x = 2; x < 4; ++x) {
        const auto vertex = f::edge_vertex(x, y);
        const auto edge = render(vertex, witness, 0, FragmentWavePolicy::Strict, true);
        check(edge.pixels.size() == f::width * f::height * 4, "quad witness renders");
        if (edge.pixels.size() != f::width * f::height * 4) continue;
        size_t covered = 0;
        for (size_t i = 0; i < edge.pixels.size(); i += 4) covered += !blue(&edge.pixels[i]);
        const auto* p = &edge.pixels[(y * f::width + x) * 4];
        check(covered == 1 && p[0] == ((y & 1) * 2 + (x & 1)) && p[1] == 3,
              "the chosen quad position has three actual helper neighbors");
        const uint64_t id = 0xf3990000u + y * 16 + x;
        verify_clear(render(vertex, fragment, id, FragmentWavePolicy::Strict));
        const auto live = render(vertex, fragment, id, FragmentWavePolicy::ProvenVotes);
        verify_derivatives(live, x, y);
        check(live.pipelines.misses == 1 && live.pipelines.hits == 0, "first effective variant creates its pipeline");
        const auto warm = render(vertex, fragment, id, FragmentWavePolicy::ProvenVotes);
        verify_derivatives(warm, x, y);
        check(warm.pipelines.hits == 1 && warm.pipelines.misses == 0, "unchanged effective variant reuses its pipeline");
        verify_clear(render(vertex, fragment, id, FragmentWavePolicy::Strict));

        // Both routes omit required-size pNext but create DIFFERENT module bytes for one source
        // identity. Exercise both cache orders; the variant bit must prevent either false hit.
        for (const bool live_first : {false, true}) {
            const uint64_t alternate = id + (live_first ? 0x200u : 0x100u);
            const auto first = render(vertex, fragment, alternate,
                live_first ? FragmentWavePolicy::ProvenVotes : FragmentWavePolicy::Strict, !live_first);
            check(first.pipelines.misses == 1, "first width-independent cache variant is cold");
            const auto second = render(vertex, fragment, alternate,
                live_first ? FragmentWavePolicy::Strict : FragmentWavePolicy::ProvenVotes, live_first);
            check(second.pipelines.misses == 1 && second.pipelines.hits == 0,
                  "effective bytes and native diagnostic bytes do not collide at required width zero");
            if (live_first) verify_derivatives(first, x, y);
            else verify_derivatives(second, x, y);
            verify_clear(render(vertex, fragment, alternate, FragmentWavePolicy::Strict));
        }
        // No source identity bypasses both transform memos and pipeline cache ownership.
        verify_derivatives(render(vertex, fragment, 0, FragmentWavePolicy::ProvenVotes), x, y);
        verify_clear(render(vertex, fragment, 0, FragmentWavePolicy::Strict));
        check(*fragment == captured, "live/cache/replay calls leave shared captured source words byte-identical");
        verify_loop_votes(x, y, poison);
        verify_neutral_votes(x, y, poison);
    }
    verify_buffer_authority(context.deterministic_storage_reads);
    verify_neutral_buffer_variants(context.deterministic_storage_reads);
    verify_per_draw_loop_bounds(context.deterministic_storage_reads);
    std::printf("== %s: %d failures ==\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
