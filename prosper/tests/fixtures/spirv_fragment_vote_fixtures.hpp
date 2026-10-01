#pragma once

// Project-owned rewrite certificates and counterexamples. Reuse the independent address/effect
// corpus, not the recompiler: the previous output-width proof is deliberately NOT the oracle.
#include "spirv_wave_width_fixtures.hpp"
#include "spirv_fragment_loop_vote_fixtures.hpp"
#include <algorithm>

namespace prosper::test::fragment_votes {

struct Fixture {
    std::string name;
    std::vector<uint32_t> words;
    bool immutable_storage;
    bool admitted;
    uint32_t uniform_votes;
    uint32_t dead_votes;
    bool inventory_refusal = false;
};

inline void marker(std::vector<uint32_t>& words, const std::string& text) {
    const size_t at = words.size();
    words.resize(at + 1 + (text.size() + 4) / 4, 0);
    words[at] = (static_cast<uint32_t>(words.size() - at) << 16) | 330;
    for (size_t i = 0; i < text.size(); ++i)
        words[at + 1 + i / 4] |= static_cast<uint32_t>(static_cast<unsigned char>(text[i])) << (8 * (i % 4));
}

inline size_t find(const std::vector<uint32_t>& words, uint32_t opcode) {
    for (size_t at = 5; at < words.size(); at += words[at] >> 16)
        if ((words[at] & 0xffffu) == opcode) return at;
    return words.size();
}

inline void insert(std::vector<uint32_t>& words, size_t at, const std::vector<uint32_t>& extra) {
    words.insert(words.begin() + at, extra.begin(), extra.end());
}

inline std::vector<uint32_t> contract(std::vector<uint32_t> words) {
    std::vector<uint32_t> metadata;
    marker(metadata, "Prosper.FragmentSubgroupSize=64");
    marker(metadata, "Prosper.FragmentSubgroupWhy=2");
    insert(words, find(words, 71), metadata);
    return words;
}

inline std::vector<uint32_t> storage_predicate(bool floating = false, bool preserve = false,
                                               bool writer = false, bool ext_abs = false) {
    // Writer arm uses distinct per-pixel destinations from the existing corpus, not raced
    // same-address non-atomic writes. Negative write proof is a whole-pass/alias contract.
    auto words = wave_width::make_module(writer ? wave_width::Shape::ExternalStore :
                                                wave_width::Shape::BufferRead, true);
    // The new buffer predicate replaces the corpus's varying scalar input completely.
    // Remove that unused interface and its load/compare, so the execution fixture links
    // against a Position-only vertex shader without an unmatched Location0 input.
    const auto entry = find(words, 15);
    words[entry] -= 1u << 16;
    words.erase(words.begin() + entry + 5); // %22, after the two-word "main" string
    for (size_t at = 5; at < words.size();) {
        const uint32_t count = words[at] >> 16, opcode = words[at] & 0xffffu;
        if ((opcode == 71 && words[at + 1] == 22) ||
            (opcode == 59 && words[at + 2] == 22) ||
            (opcode == 61 && words[at + 2] == 50) ||
            (opcode == 186 && words[at + 2] == 51))
            words.erase(words.begin() + at, words.begin() + at + count);
        else at += count;
    }
    // A separate runtime word buffer supplies the predicate. Keep the original corpus's
    // float buffer/output and local fixed array untouched. Integer bits use an actual Int32
    // load, not an unpreserved float load followed by a Bitcast.
    const bool float_word = floating || ext_abs;
    words[3] = 101;
    insert(words, find(words, 54), {
        (3u << 16) | 29u, 96, float_word ? 4u : 3u,
        (3u << 16) | 30u, 97, 96,
        (4u << 16) | 32u, 98, 12, 97,
        (4u << 16) | 32u, 99, 12, float_word ? 4u : 3u,
        (4u << 16) | 59u, 98, 100, 12});
    insert(words, find(words, 71), {
        (4u << 16) | 71u, 96, 6, 4,
        (3u << 16) | 71u, 97, 2,
        (5u << 16) | 72u, 97, 0, 35, 0,
        (4u << 16) | 71u, 100, 34, 0,
        (4u << 16) | 71u, 100, 33, 5});
    std::vector<uint32_t> extra;
    const auto op = [&](uint32_t code, std::initializer_list<uint32_t> args) {
        wave_width::instruction(extra, code, args);
    };
    op(65, {99, 90, 100, 8, 8}); // fixed buffer[0], shared by all invocations including helpers
    op(61, {float_word ? 4u : 3u, 91, 90});
    if (floating) op(186, {2, 93, 91, 12});
    else if (ext_abs) {
        op(12, {4, 92, 94, 4, 91}); // arbitrary-domain GLSL FAbs is NOT a certificate
        op(186, {2, 93, 92, 12});
    } else {
        op(171, {2, 93, 91, 8}); // integer bits != 0: no floating-point assumptions
    }
    size_t any = find(words, 335);
    words[any + 4] = 93;
    insert(words, any, extra);
    if (ext_abs) {
        // ExtInstImport %94 "GLSL.std.450" before MemoryModel.
        insert(words, find(words, 14), {(6u << 16) | 11u, 94, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0});
    }
    if (preserve) {
        insert(words, 5, {(2u << 16) | 17u, 4466});
        insert(words, 5, {(2u << 16) | 17u, 4464}); // DenormPreserve
        std::vector<uint32_t> extension;
        wave_width::extension(extension, "SPV_KHR_float_controls");
        const auto imports = find(words, 11);
        insert(words, imports == words.size() ? find(words, 14) : imports, extension);
        insert(words, find(words, 71), {(4u << 16) | 16u, 40, 4461, 32});
        insert(words, find(words, 71), {(4u << 16) | 16u, 40, 4459, 32});
    }
    return contract(std::move(words));
}

inline std::vector<uint32_t> derivative_branch(bool uniform) {
    auto words = wave_width::make_module(wave_width::Shape::EarlyReturn, false, uniform);
    // Replace the constant branch by Any and its early return by an unused derivative.
    words[find(words, 250) + 1] = 52;
    const auto ret = find(words, 253);
    words.erase(words.begin() + ret);
    insert(words, ret, {(4u << 16) | 207u, 4, 90, 50, (2u << 16) | 249u, 62});
    return contract(std::move(words));
}

inline std::vector<Fixture> fixtures() {
    std::vector<Fixture> out;
    const auto add = [&](const char* name, wave_width::Shape shape, bool dependent, bool uniform,
                         bool admitted) {
        out.push_back({name, contract(wave_width::make_module(shape, dependent, uniform)), false,
                       admitted, admitted && uniform ? 1u : 0u, admitted && !uniform ? 1u : 0u});
    };
    add("uniform_true_observable_index", wave_width::Shape::BufferRead, true, true, true);
    auto false_vote = out.back().words;
    // ConstantTrue %14 becomes ConstantFalse; every other word stays identical.
    false_vote[find(false_vote, 41)] = (3u << 16) | 42u;
    out.push_back({"uniform_false_observable_index", std::move(false_vote), false, true, 1, 0});
    add("varying_observable_index", wave_width::Shape::BufferRead, true, false, false);
    add("dead_select_index", wave_width::Shape::BufferRead, false, false, true);
    add("varying_local_store", wave_width::Shape::LocalStore, true, false, false);
    add("dead_select_local_store", wave_width::Shape::LocalStore, false, false, true);
    add("varying_external_store", wave_width::Shape::ExternalStore, true, false, false);
    add("dead_select_external_store", wave_width::Shape::ExternalStore, false, false, true);
    add("varying_copy_source", wave_width::Shape::CopySourceExternal, true, false, false);
    add("dead_select_copy_source", wave_width::Shape::CopySourceExternal, false, false, true);
    add("varying_atomic_address", wave_width::Shape::AtomicAddress, true, false, false);
    add("dead_select_atomic_address", wave_width::Shape::AtomicAddress, false, false, true);
    add("varying_image_identity", wave_width::Shape::ImageIdentity, true, false, false);
    add("dead_select_image_identity", wave_width::Shape::ImageIdentity, false, false, true);
    add("varying_dref", wave_width::Shape::DrefSample, true, false, false);
    add("dead_select_implicit_dref", wave_width::Shape::DrefSample, false, false, true);
    add("varying_effect_call", wave_width::Shape::OpaqueCall, true, false, false);
    add("uniform_effect_call", wave_width::Shape::OpaqueCall, true, true, true);
    add("varying_loop_count", wave_width::Shape::LoopCounter, true, false, false);
    add("uniform_loop_count", wave_width::Shape::LoopCounter, true, true, true);
    add("output_dead_loop_is_not_rewrite_safe", wave_width::Shape::DeadLoopCounter, true, false, false);
    out.push_back({"varying_guard_unused_derivative", derivative_branch(false), false, false, 0, 0});
    out.push_back({"uniform_guard_derivative", derivative_branch(true), false, true, 1, 0});
    out.push_back({"immutable_storage_integer", storage_predicate(), true, true, 1, 0});
    out.push_back({"uncertified_storage_integer", storage_predicate(), false, false, 0, 0});
    auto fixed_array = storage_predicate();
    const auto runtime_array = find(fixed_array, 29);
    fixed_array[runtime_array] = (4u << 16) | 28u;
    insert(fixed_array, runtime_array + 3, {10}); // same word buffer, declared fixed length2
    out.push_back({"fixed_array_is_not_robust_descriptor_bounds", std::move(fixed_array), true, false, 0, 0});
    out.push_back({"storage_writer_invalidates_certificate", storage_predicate(false, false, true), true, false, 0, 0});
    out.push_back({"preserved_storage_f32_compare", storage_predicate(true, true), true, true, 1, 0});
    auto flush_denorm = storage_predicate(true, true);
    for (size_t at = 5; at < flush_denorm.size(); at += flush_denorm[at] >> 16) {
        const uint32_t op = flush_denorm[at] & 0xffffu;
        if (op == 17 && flush_denorm[at + 1] == 4464) flush_denorm[at + 1] = 4465;
        if (op == 16 && flush_denorm[at + 2] == 4459) flush_denorm[at + 2] = 4460;
    }
    out.push_back({"flush_mode_does_not_define_compare_operands", std::move(flush_denorm), true, false, 0, 0});
    auto unspecified_denorm = storage_predicate(true, true);
    for (size_t at = 5; at < unspecified_denorm.size(); at += unspecified_denorm[at] >> 16) {
        if ((unspecified_denorm[at] & 0xffffu) == 16 && unspecified_denorm[at + 2] == 4459) {
            unspecified_denorm.erase(unspecified_denorm.begin() + at, unspecified_denorm.begin() + at + 4);
            break;
        }
    }
    out.push_back({"unspecified_subnormal_compare", std::move(unspecified_denorm), true, false, 0, 0});
    out.push_back({"unpreserved_storage_f32_compare", storage_predicate(true), true, false, 0, 0});
    out.push_back({"arbitrary_glsl_domain", storage_predicate(false, true, false, true), true, false, 0, 0});
    auto float_bits = storage_predicate(true, true);
    size_t compare = 5;
    while ((float_bits[compare] & 0xffffu) != 186 || float_bits[compare + 2] != 93)
        compare += float_bits[compare] >> 16;
    float_bits.erase(float_bits.begin() + compare, float_bits.begin() + compare + 5);
    insert(float_bits, compare, {(4u << 16) | 124u, 3, 92, 91,
                                (5u << 16) | 171u, 2, 93, 92, 8});
    out.push_back({"float_bits_are_not_defined_by_value_preserve", std::move(float_bits), true, false, 0, 0});
    auto integer_float = storage_predicate(false, true);
    const auto integer_compare = find(integer_float, 171);
    integer_float.erase(integer_float.begin() + integer_compare, integer_float.begin() + integer_compare + 5);
    insert(integer_float, integer_compare, {(4u << 16) | 124u, 4, 92, 91,
                                          (5u << 16) | 186u, 2, 93, 92, 12});
    out.push_back({"arbitrary_integer_bits_are_not_a_float_domain", std::move(integer_float), true, false, 0, 0});
    for (const uint32_t index : {0x3fffffffu, 0x40000000u}) {
        auto indexed = storage_predicate();
        indexed[3] = 102;
        insert(indexed, find(indexed, 54), {(4u << 16) | 43u, 3, 101, index});
        for (size_t at = 5; at < indexed.size(); at += indexed[at] >> 16)
            if ((indexed[at] & 0xffffu) == 65 && indexed[at + 2] == 90) indexed[at + 5] = 101;
        out.push_back({index == 0x3fffffffu ? "bounded_oob_word_index" : "wrapping_word_index",
                       std::move(indexed), true, index == 0x3fffffffu, index == 0x3fffffffu ? 1u : 0u, 0});
    }
    for (const bool masked : {false, true}) {
        auto indexed = storage_predicate();
        indexed[3] = 105;
        insert(indexed, find(indexed, 54), {(4u << 16) | 43u, 3, 101, 0x3fffffffu});
        std::vector<uint32_t> extra;
        if (masked) wave_width::instruction(extra, 199, {3, 102, 91, 101});
        wave_width::instruction(extra, 65, {99, 103, 100, 8, masked ? 102u : 91u});
        wave_width::instruction(extra, 61, {3, 104, 103});
        const auto predicate = find(indexed, 171);
        indexed[predicate + 3] = 104;
        insert(indexed, predicate, extra);
        out.push_back({masked ? "masked_uniform_dynamic_word_index" : "unbounded_uniform_word_index",
                       std::move(indexed), true, masked, masked ? 1u : 0u, 0});
    }
    for (const bool second_safe : {true, false}) {
        auto two = out.front().words;
        two[3] = 98;
        std::vector<uint32_t> extra;
        wave_width::instruction(extra, 335, {2, 96, 11, second_safe ? 14u : 51u});
        if (!second_safe) {
            wave_width::instruction(extra, 169, {4, 97, 96, 13, 12});
            wave_width::instruction(extra, 62, {23, 97});
        }
        insert(two, find(two, 253), extra);
        out.push_back({second_safe ? "two_uniform_votes" : "safe_then_unsafe_vote_rolls_back",
                       std::move(two), false, second_safe, second_safe ? 2u : 0u, 0});
    }
    auto subgroup_builtin = out.front().words;
    subgroup_builtin[3] = 98;
    insert(subgroup_builtin, find(subgroup_builtin, 71), {
        (4u << 16) | 71u, 97, 11, 36, (3u << 16) | 71u, 97, 14}); // SubgroupSize, Flat
    insert(subgroup_builtin, find(subgroup_builtin, 54), {
        (4u << 16) | 32u, 96, 1, 3, (4u << 16) | 59u, 96, 97, 1});
    const auto entry = find(subgroup_builtin, 15);
    const auto entry_count = subgroup_builtin[entry] >> 16;
    subgroup_builtin[entry] += 1u << 16;
    insert(subgroup_builtin, entry + entry_count, {97});
    out.push_back({"valid_subgroup_builtin_inventory", std::move(subgroup_builtin), false, false, 0, 0, true});
    auto no_wrap = out.front().words;
    no_wrap[3] = 97;
    std::vector<uint32_t> extension;
    wave_width::extension(extension, "SPV_KHR_no_integer_wrap_decoration");
    insert(no_wrap, find(no_wrap, 14), extension);
    insert(no_wrap, find(no_wrap, 71), {(3u << 16) | 71u, 96, 4470});
    insert(no_wrap, find(no_wrap, 335), {(5u << 16) | 128u, 3, 96, 8, 9});
    out.push_back({"valid_no_wrap_arithmetic_inventory", std::move(no_wrap), false, false, 0, 0, true});
    // Memory effects can make the collective's execution rendezvous observable even when
    // the Boolean is dead or constant. These are inventory refusals, not dead-SSA proofs.
    for (auto& fixture : out) {
        if (fixture.name == "varying_external_store" || fixture.name == "dead_select_external_store" ||
            fixture.name == "varying_atomic_address" || fixture.name == "dead_select_atomic_address" ||
            fixture.name == "varying_image_identity" || fixture.name == "dead_select_image_identity" ||
            fixture.name == "varying_effect_call" || fixture.name == "uniform_effect_call" ||
            fixture.name == "storage_writer_invalidates_certificate") {
            fixture.admitted = false;
            fixture.uniform_votes = fixture.dead_votes = 0;
            fixture.inventory_refusal = true;
        }
    }
    namespace loops = fragment_loop_votes;
    const struct { const char* name; loops::Shape shape; bool admitted; } loop_cases[] = {
        {"initialized_counter_self_bool", loops::Shape::Counter, true},
        {"coupled_nested_counter_reset", loops::Shape::Nested, true},
        {"cross_carried_initialized_phis", loops::Shape::CrossCarried, true},
        {"one_undef_seed_in_cross_carried_component", loops::Shape::CrossUndefined, false},
        {"initialized_changing_boolean_recurrence", loops::Shape::BoolToggle, true},
        {"helper_dependent_boolean_carry_update", loops::Shape::VaryingBoolUpdate, false},
        {"immutable_constant_index_loop_bound", loops::Shape::BufferBound, true},
        {"helper_varying_trip_direct_header_export", loops::Shape::VaryingTrip, false},
        {"undef_counter_initializer", loops::Shape::UndefinedInit, false},
        {"helper_varying_counter_initializer", loops::Shape::VaryingInit, false},
        {"vote_is_not_an_initializer_root", loops::Shape::VoteInit, false},
        {"unsupported_counter_update", loops::Shape::UnsupportedUpdate, false},
        {"uniform_diamond_different_phi_values", loops::Shape::UniformDiamond, true},
        {"varying_diamond_different_phi_values", loops::Shape::VaryingDiamond, false},
        {"uniform_switch_different_phi_values", loops::Shape::UniformSwitch, true},
        {"varying_switch_different_phi_values", loops::Shape::VaryingSwitch, false},
        {"uniform_induction_is_not_address_authority", loops::Shape::InductionAddress, false},
        {"bounded_masked_induction_does_not_expand_load_authority", loops::Shape::MaskedInductionAddress, false},
        {"safe_counter_then_unsafe_vote_rollback", loops::Shape::SecondUnsafeVote, false},
        {"safe_loop_then_varying_loop_rollback", loops::Shape::SecondUnsafeLoop, false},
        {"uniform_loop_combines_with_genuinely_dead_vote", loops::Shape::CounterWithDeadVote, true},
        {"committed_trace_facts_do_not_recertify_dead_masked_load", loops::Shape::CounterWithDeadMaskedLoad, true},
        {"old_constant_vote_with_varying_control", loops::Shape::ConstantWithVaryingControl, true},
        {"old_dead_vote_with_varying_control", loops::Shape::DeadWithVaryingControl, true},
    };
    for (const auto& item : loop_cases) {
        const bool dead = item.shape == loops::Shape::DeadWithVaryingControl;
        out.push_back({item.name, loops::make_module(item.shape), true, item.admitted,
                       item.admitted && !dead ? 1u : 0u,
                       dead || item.shape == loops::Shape::CounterWithDeadVote ||
                           item.shape == loops::Shape::CounterWithDeadMaskedLoad ? 1u : 0u});
    }
    for (uint32_t bound = 0; bound < 4; ++bound)
        out.push_back({"initialized_loop_bound_" + std::to_string(bound),
                       loops::make_module(loops::Shape::Counter, bound), false, true, 1, 0});
    for (const uint32_t bound : {1u, 3u})
        out.push_back({"changing_boolean_odd_bound_" + std::to_string(bound),
                       loops::make_module(loops::Shape::BoolToggle, bound), false, true, 1, 0});
    return out;
}

} // namespace prosper::test::fragment_votes
