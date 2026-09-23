#include "shared/live/world_depth_ab.hpp"

#include <array>
#include <cstdio>
#include <initializer_list>

using prosper::frontend::WorldDepthAbConfig;
using prosper::frontend::WorldDepthDrawFact;
using prosper::frontend::WorldDepthPassKind;

static bool check(bool condition, const char* reason) {
    if (!condition) std::fprintf(stderr, "world-depth-ab: %s\n", reason);
    return condition;
}

int main() {
    WorldDepthAbConfig config;
    if (!check(prosper::frontend::parse_world_depth_ab(
                   "4:3930:0x2053960000:0x2049a00000:3840:2160", config),
               "valid config declined") ||
        !check(config.mode == 4 && config.capture_pad_flip == 3930 &&
                   config.depth_base == 0x2053960000ull &&
                   config.color_base == 0x2049a00000ull,
               "valid config parsed incorrectly"))
        return 1;
    for (const char* malformed : {
             "4:3930:0x2053960000:0x2049a00000:3840",
             "1:3930:0x2053960000:0x2049a00000:3840:2160",
             "4:3930:2053960000:0x2049a00000:3840:2160",
             "4:-1:0x2053960000:0x2049a00000:3840:2160",
             "4:3930:0x2053960000:0x2049a00000:16384:2160",
             "4:3930:0x2053960000:0x2049a00000:3840:2160:extra",
         })
        if (!check(!prosper::frontend::parse_world_depth_ab(malformed, config),
                   "malformed config admitted"))
            return 1;

    WorldDepthAbConfig c2_config = config;
    c2_config.mode = 0;
    const auto c2_arm = prosper::frontend::world_c2_arm_allowed;
    if (!check(c2_arm(true, true, c2_config, "1"),
               "exact Sonic c2 diagnostic failed to arm") ||
        !check(!c2_arm(false, true, c2_config, "1"),
               "unarmed title/output directory entered c2 diagnostic") ||
        !check(!c2_arm(true, false, c2_config, "1"),
               "non-metadata run entered c2 diagnostic") ||
        !check(!c2_arm(true, true, c2_config, "0"),
               "wrong explicit c2 value entered diagnostic"))
        return 1;
    c2_config.mode = 4;
    if (!check(!c2_arm(true, true, c2_config, "1"),
               "depth override policy entered c2 content diagnostic"))
        return 1;
    c2_config = config;
    c2_config.mode = 0;
    --c2_config.capture_pad_flip;
    if (!check(!c2_arm(true, true, c2_config, "1"),
               "different pad entered c2 content diagnostic"))
        return 1;
    const auto c2_attempt = prosper::frontend::world_c2_should_attempt;
    if (!check(c2_attempt(true, false, true),
               "new exact callback did not authorize one c2 attempt") ||
        !check(!c2_attempt(true, true, true),
               "second c2 readback attempt authorized") ||
        !check(!c2_attempt(true, false, false),
               "wrong callback shape authorized c2 readback") ||
        !check(!c2_attempt(false, false, true),
               "default-off c2 readback authorized"))
        return 1;
    const auto metadata_range = prosper::frontend::world_c2_metadata_range_valid;
    if (!check(metadata_range(0, 0) && metadata_range(0x1000, 0x100),
               "valid absent/present c2 metadata declined") ||
        !check(!metadata_range(0, 0x100) && !metadata_range(0x1000, 0),
               "partial c2 metadata identity accepted") ||
        !check(!metadata_range(UINT64_MAX - 3, 8),
               "overflowing c2 metadata range accepted"))
        return 1;

    using prosper::frontend::WorldC2AlphaPassFact;
    const auto alpha_allowed = prosper::frontend::world_c2_alpha_pass_allowed;
    const WorldC2AlphaPassFact exact_alpha{
        true, true, false, 3930, 3930, 3930, 3930, 1, 1,
        0x204f9a0000ull, 0x8u, 0x20059c8b00ull, 1, true};
    const auto funnel_allowed = prosper::frontend::world_c2_funnel_allowed;
    if (!check(alpha_allowed(exact_alpha) && funnel_allowed(exact_alpha),
               "exact c2 alpha pass or scoped funnel declined")) return 1;
    const auto rejects_alpha_mutation = [&](auto mutate, const char* reason) {
        WorldC2AlphaPassFact variant = exact_alpha;
        mutate(variant);
        return check(!alpha_allowed(variant), reason);
    };
    if (!rejects_alpha_mutation([](auto& f) { f.armed = false; },
                                "default-off alpha audit admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.before_complete = false; },
                                "alpha audit admitted without pre-version") ||
        !rejects_alpha_mutation([](auto& f) { f.attempted = true; },
                                "second alpha attempt admitted") ||
        !rejects_alpha_mutation([](auto& f) { --f.callback_pad; },
                                "other callback pad admitted") ||
        !rejects_alpha_mutation([](auto& f) { ++f.guest_flip; },
                                "drifted guest flip admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.draws = 2; },
                                "multi-draw alpha pass admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.color_count = 3; },
                                "MRT alpha pass admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.color_base ^= 0x1000; },
                                "other target admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.color_mask = 0xf; },
                                "RGB-writing pass admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.fragment_program ^= 0x100; },
                                "other fragment program admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.c1_binding_count = 0; },
                                "missing c1 input admitted") ||
        !rejects_alpha_mutation([](auto& f) { f.c1_binding_count = 2; },
                                "ambiguous c1 inputs admitted"))
        return 1;
    WorldC2AlphaPassFact nonterminal_alpha = exact_alpha;
    nonterminal_alpha.terminal_pass = false;
    if (!check(alpha_allowed(nonterminal_alpha) && !funnel_allowed(nonterminal_alpha),
               "non-terminal pass retains alpha audit but declines scoped funnel"))
        return 1;

    WorldDepthDrawFact clear;
    clear.depth_read_base = clear.depth_write_base = 0x2053960000ull;
    clear.depth_width = 3840; clear.depth_height = 2160;
    clear.depth_test = clear.depth_write = clear.clear_enable = true;
    clear.has_scissor = true; clear.scissor_right = 3840; clear.scissor_bottom = 2160;
    clear.compare_op = 7; clear.topology = 4; clear.vertex_count = 4;
    std::array<WorldDepthDrawFact, 1> one{clear};
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Clear,
               "world clear did not arm"))
        return 1;

    one[0].depth_width = one[0].depth_height = 1024;
    one[0].depth_read_base = one[0].depth_write_base = 0x2048a00000ull;
    one[0].scissor_right = one[0].scissor_bottom = 1024;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Unrelated,
               "shadow clear armed"))
        return 1;
    one[0] = clear;
    one[0].scissor_right = 3000;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Unrelated,
               "wrong scissor armed"))
        return 1;
    one[0] = clear;
    one[0].depth_read_base = 0x2048a00000ull;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Unrelated,
               "aliasing read identity armed"))
        return 1;

    WorldDepthDrawFact geometry = clear;
    geometry.compare_op = 6;
    geometry.topology = 3;
    geometry.vertex_count = 270;
    geometry.index_count = 270;
    geometry.color_base = config.color_base;
    geometry.color_write_mask = 0xf;
    geometry.scissor_right = 2880;
    geometry.scissor_bottom = 1620;
    one[0] = geometry;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::Geometry,
               "world geometry did not arm"))
        return 1;
    one[0].color_base ^= 0x1000;
    if (!check(prosper::frontend::classify_world_depth_pass(config, one) ==
                   WorldDepthPassKind::OtherWorld,
               "different world color mislabeled as target geometry"))
        return 1;
    const std::array<WorldDepthDrawFact, 2> mixed{clear, WorldDepthDrawFact{}};
    if (!check(prosper::frontend::classify_world_depth_pass(config, mixed) ==
                   WorldDepthPassKind::Ambiguous,
               "mixed target pass armed"))
        return 1;
    const std::array<WorldDepthDrawFact, 2> mixed_scissors{clear, geometry};
    if (!check(prosper::frontend::classify_world_depth_pass(config, mixed_scissors) ==
                   WorldDepthPassKind::Ambiguous,
               "mixed clear and scene scissor armed as one pass"))
        return 1;
    if (!check(prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Clear),
               "selected world clear lost its override") ||
        !check(prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Geometry),
               "selected world geometry lost its override") ||
        !check(!prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::OtherWorld),
               "unproven same-depth pass gained an override") ||
        !check(!prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Ambiguous),
               "mixed pass gained an override") ||
        !check(!prosper::frontend::world_depth_scope_allowed(WorldDepthPassKind::Unrelated),
               "shadow or unrelated pass gained an override"))
        return 1;
    const auto override_enabled = prosper::frontend::world_depth_override_enabled;
    if (!check(override_enabled(true, false, WorldDepthPassKind::Geometry),
               "selected A/B geometry cannot override depth policy") ||
        !check(!override_enabled(true, true, WorldDepthPassKind::Geometry),
               "metadata-only geometry still overrides depth policy") ||
        !check(!override_enabled(true, true, WorldDepthPassKind::Clear),
               "metadata-only clear still overrides depth policy") ||
        !check(!override_enabled(false, false, WorldDepthPassKind::Geometry),
               "unarmed geometry overrides depth policy"))
        return 1;
    const auto at_pad = prosper::frontend::world_census_observation_allowed;
    if (!check(at_pad(true, true, 3930, 3930), "exact-pad census declined") ||
        !check(!at_pad(true, true, 3930, 3929),
               "prior pad was treated as requested observation time") ||
        !check(!at_pad(true, false, 3930, 3930),
               "ordinary A/B capture entered metadata-only census"))
        return 1;
    // The live caller supplies the renderer's active address for each slot. A c0/c1-only
    // implementation passes the old detail-line check but misses these c3/c7 bindings.
    const std::array<uint64_t, 8> active_targets{
        config.color_base, 0, 0, 0x204f9a0000ull, 0, 0, 0, config.color_base};
    int slot_reads = 0;
    const auto target_slots = prosper::frontend::world_census_target_slots(
        static_cast<uint32_t>(active_targets.size()), config.color_base, 0x204f9a0000ull,
        [&](uint32_t slot) { ++slot_reads; return active_targets[slot]; });
    if (!check(target_slots.world == 0x81u && target_slots.composite == 0x8u &&
                   slot_reads == 8,
               "world/composite slot scan missed an active c2..c7 binding"))
        return 1;
    const auto world_c2_shape = [] {
        prosper::frontend::WorldC2CallbackShape shape;
        shape.observe(62, 0x1, 0x4, 62, 62, 62, 62, WorldDepthPassKind::Geometry, false);
        shape.observe(1, 0x1, 0, 1, 0, 0, 0, WorldDepthPassKind::Geometry, false);
        return shape;
    };
    auto c2_shape = world_c2_shape();
    if (!check(c2_shape.ready(), "exact 62+1 c0-world/c2 callback shape refused"))
        return 1;
    c2_shape = {};
    c2_shape.observe(62, 0x1, 0x4, 62, 62, 62, 62,
                     WorldDepthPassKind::Geometry, false);
    c2_shape.observe(1, 0x1, 0, 1, 1, 0, 0,
                     WorldDepthPassKind::Geometry, false);
    if (!check(!c2_shape.ready(),
               "terminal world-only draw unexpectedly active on c1 was accepted"))
        return 1;
    c2_shape = world_c2_shape();
    c2_shape.observe(1, 0, 0x1, 0, 0, 1, 0, WorldDepthPassKind::Unrelated, false);
    if (!check(!c2_shape.ready(), "later same-address c0 writer retained c2 capture gate"))
        return 1;
    c2_shape = {};
    c2_shape.observe(1, 0x1, 0, 1, 0, 0, 0, WorldDepthPassKind::Geometry, false);
    c2_shape.observe(62, 0x1, 0x4, 62, 62, 62, 62, WorldDepthPassKind::Geometry, false);
    if (!check(!c2_shape.ready(), "reversed c2 callback shape authorized readback"))
        return 1;
    c2_shape = {};
    c2_shape.observe(62, 0x1, 0x4, 62, 62, 62, 61, WorldDepthPassKind::Geometry, false);
    c2_shape.observe(1, 0x1, 0, 1, 0, 0, 0, WorldDepthPassKind::Geometry, false);
    if (!check(!c2_shape.ready(), "one draw missing c2 still authorized readback"))
        return 1;
    c2_shape = {};
    c2_shape.observe(62, 0x1, 0x4, 62, 62, 62, 62, WorldDepthPassKind::Geometry, true);
    c2_shape.observe(1, 0x1, 0, 1, 0, 0, 0, WorldDepthPassKind::Geometry, false);
    if (!check(!c2_shape.ready(), "resolve group authorized c2 readback"))
        return 1;
    c2_shape = {};
    c2_shape.observe(62, 0x1, 0x4, 62, 61, 62, 62, WorldDepthPassKind::Geometry, false);
    c2_shape.observe(1, 0x1, 0, 1, 0, 0, 0, WorldDepthPassKind::Geometry, false);
    if (!check(!c2_shape.ready(), "changed world c1 MRT binding authorized c2 readback"))
        return 1;
    const auto tail_ok = prosper::frontend::world_c2_tail_identity_allowed;
    if (!check(tail_ok(3930, 3930, 3930, true, 3930, 3930, 0),
               "exact callback tail refused c2 content probe") ||
        !check(!tail_ok(3930, 3930, 3931, true, 3930, 3930, 0),
               "pad drift authorized c2 content probe") ||
        !check(!tail_ok(3930, 3930, 3930, true, 3929, 3930, 0),
               "earlier guest version authorized c2 content probe") ||
        !check(!tail_ok(3930, 3930, 3930, true, 3930, 3930, 1),
               "guest-flip instability authorized c2 content probe") ||
        !check(!tail_ok(3930, 3930, 3930, false, 3930, 3930, 0),
               "missing guest-flip evidence authorized c2 content probe"))
        return 1;

    const auto later_inputs = prosper::frontend::world_c2_later_input_group;
    if (!check(later_inputs(true, true, true, false, false),
               "later c0 writer did not enter targeted descriptor census") ||
        !check(!later_inputs(true, false, true, false, false),
               "c2-only writer without exact producer shape entered later-input census") ||
        !check(!later_inputs(true, true, true, true, false),
               "62-draw world+c2 producer entered later-input census") ||
        !check(!later_inputs(true, true, false, true, false),
               "world-only group entered later-input census") ||
        !check(!later_inputs(true, true, true, false, true),
               "resolve group entered later-input census") ||
        !check(!later_inputs(false, true, true, false, false),
               "default-off callback entered later-input census"))
        return 1;

    prosper::frontend::WorldC2DescriptorCensus targets;
    const uint64_t c2_base = 0x204f9a0000ull;
    if (!check(decltype(targets)::target(config.color_base, config.color_base,
                                         0x204b9e0000ull, c2_base) &&
                   decltype(targets)::target(0x204b9e0000ull, config.color_base,
                                             0x204b9e0000ull, c2_base) &&
                   decltype(targets)::target(c2_base, config.color_base,
                                             0x204b9e0000ull, c2_base) &&
                   !decltype(targets)::target(0x1234, config.color_base,
                                              0x204b9e0000ull, c2_base),
               "three-address descriptor filter accepted or lost a target"))
        return 1;
    for (uint32_t binding = 0;
         binding < static_cast<uint32_t>(targets.kMaximumFacts); ++binding)
        targets.add({1, 0, c2_base, binding, 3840, 2160, 5});
    targets.add({1, 0, c2_base, 0, 3840, 2160, 5}); // a duplicate is not overflow
    if (!check(targets.complete() && targets.facts.size() == targets.kMaximumFacts &&
                   targets.matches == targets.kMaximumFacts + 1,
               "complete targeted descriptor census did not pass positive control"))
        return 1;
    targets.add({1, 0, c2_base, 99, 3840, 2160, 5});
    if (!check(!targets.complete() && targets.overflow == 1 &&
                   targets.matches == targets.kMaximumFacts + 2,
               "thirteenth target descriptor was silently truncated"))
        return 1;
    int pad_reads = 0;
    const auto read_pad = [&] { ++pad_reads; return int64_t{3930}; };
    if (!check(prosper::frontend::world_census_callback_pad(false, false, read_pad) == -1 &&
                   pad_reads == 0,
               "default-off callback sampled the pad and acquired the flip lock") ||
        !check(prosper::frontend::world_census_callback_pad(true, false, read_pad) == -1 &&
                   pad_reads == 0,
               "ordinary depth A/B callback paid for metadata census") ||
        !check(prosper::frontend::world_census_callback_pad(true, true, read_pad) == 3930 &&
                   pad_reads == 1,
               "armed metadata census failed to sample its callback pad"))
        return 1;
    const auto ambiguous = prosper::frontend::world_depth_capture_pass_ambiguous;
    if (!check(!ambiguous(false, WorldDepthPassKind::Geometry, true, true),
               "first matching world geometry declined") ||
        !check(ambiguous(true, WorldDepthPassKind::Geometry, true, true),
               "duplicate matching world geometry accepted") ||
        !check(ambiguous(true, WorldDepthPassKind::OtherWorld, true, false),
               "later depth target overwrite accepted") ||
        !check(ambiguous(true, WorldDepthPassKind::Unrelated, true, false),
               "later color target overwrite accepted") ||
        !check(ambiguous(false, WorldDepthPassKind::Ambiguous, true, false),
               "ambiguous target pass accepted") ||
        !check(!ambiguous(true, WorldDepthPassKind::Unrelated, false, false),
               "independent later pass invalidated world capture"))
        return 1;

    // The retained pad-3930 A/B selected an eight-draw character version of this address. An
    // earlier unrelated frame had a 162-draw world version. Only a large producer followed by a
    // composite at this SAME pad can qualify for a later, separately reviewed readback probe.
    prosper::frontend::WorldProducerCensus census;
    if (!check(!census.observe_at(true, true, 3930, 3929, true, true, false,
                                  WorldDepthPassKind::Geometry, 162) &&
                   census.relevant_groups == 0,
               "prior-pad producer entered the exact-pad census"))
        return 1;
    if (!check(census.observe_at(true, true, 3930, 3930, true, true, false,
                                 WorldDepthPassKind::Geometry, 8) &&
                   census.relevant_groups == 1,
               "matching callback observation did not activate census"))
        return 1;
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 2);
    if (!check(!census.has_single_ordered_candidate() && census.max_world_draws == 8,
               "small character-only producer authorized follow-up"))
        return 1;
    census = {};
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 2);
    census.observe(true, true, false, WorldDepthPassKind::Geometry, 162);
    if (!check(!census.has_single_ordered_candidate(),
               "composite before world producer counted as downstream"))
        return 1;
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 2);
    if (!check(census.has_single_ordered_candidate(),
               "ordered large world producer failed positive control"))
        return 1;
    ++census.unstable_callbacks;
    if (!check(!census.has_single_ordered_candidate(),
               "pad shift inside callback retained a false candidate"))
        return 1;
    --census.unstable_callbacks;
    census.observe(true, true, false, WorldDepthPassKind::Geometry, 120);
    if (!check(!census.has_single_ordered_candidate(),
               "two large world versions called a single producer"))
        return 1;
    census = {};
    census.observe(true, true, false, WorldDepthPassKind::Geometry, 162);
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 1);
    if (!check(census.has_single_ordered_candidate(),
               "second-world-group negative control did not first establish a candidate"))
        return 1;
    census.observe(true, true, false, WorldDepthPassKind::OtherWorld, 1);
    if (!check(census.world_groups == 2 && census.large_world_groups == 1 &&
                   !census.has_single_ordered_candidate(),
               "a later one-draw world write still passed the single-version gate"))
        return 1;
    census = {};
    census.observe(true, true, false, WorldDepthPassKind::OtherWorld, 1);
    census.observe(true, true, false, WorldDepthPassKind::Geometry, 162);
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 1);
    if (!check(census.world_groups == 2 && census.large_world_groups == 1 &&
                   !census.has_single_ordered_candidate(),
               "an earlier one-draw world seed still passed the single-version gate"))
        return 1;
    census = {};
    census.observe(true, true, true, WorldDepthPassKind::Geometry, 162);
    if (!check(!census.has_single_ordered_candidate(),
               "world/composite bound in one unordered group counted as sequential"))
        return 1;
    census = {};
    census.observe_guest_flip(3930);
    census.observe(true, true, false, WorldDepthPassKind::Geometry, 162);
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 1);
    if (!check(census.has_single_ordered_candidate(),
               "overlap negative control did not first establish an ordered candidate"))
        return 1;
    census = {};
    census.observe_guest_flip(3930);
    census.observe(true, true, true, WorldDepthPassKind::Geometry, 162);
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 1);
    if (!check(census.world_groups == 1 && census.large_world_groups == 1 &&
                   census.composite_after_large_world == 1 &&
                   census.world_composite_overlap_groups == 1 &&
                   !census.has_single_ordered_candidate(),
               "MRT world/composite overlap still authorized a single version join"))
        return 1;
    census = {};
    census.observe_guest_flip(3930);
    census.observe(true, true, false, WorldDepthPassKind::Geometry, 162);
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 1);
    if (!check(census.has_single_ordered_candidate(),
               "guest-flip negative control did not first establish an ordered candidate"))
        return 1;
    census.observe_guest_flip(3931);
    if (!check(census.unstable_guest_flips == 1 &&
                   !census.has_single_ordered_candidate(),
               "changed guest flip still authorized a single version join"))
        return 1;
    census = {};
    census.observe(true, true, false, WorldDepthPassKind::Geometry, 162);
    census.observe(false, false, true, WorldDepthPassKind::Unrelated, 1);
    if (!check(census.has_single_ordered_candidate(),
               "descriptor-overflow negative control did not first establish a candidate"))
        return 1;
    ++census.descriptor_overflow;
    if (!check(!census.has_single_ordered_candidate(),
               "truncated composite descriptors still authorized a version join"))
        return 1;
    census = {};
    for (uint64_t i = 0; i < prosper::frontend::WorldProducerCensus::kMaximumLines - 1; ++i)
        if (!check(census.reserve_detail_line(), "detail line under cap refused")) return 1;
    census.observe(true, true, false, WorldDepthPassKind::OtherWorld, 1);
    if (!check(!census.reserve_detail_line() && census.reported_lines == 63 &&
                   census.omitted_lines == 1 && !census.has_single_ordered_candidate(),
               "shared group/callback detail cap failed closed with one summary row reserved"))
        return 1;
    return 0;
}
