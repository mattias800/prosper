#pragma once
#include "spirv_fragment_vote_fixtures.hpp"

namespace prosper::test::fragment_neutral {
enum class Shape {
    Masked, Termination, ScalarExport, TerminationExport, DeadUnequalPhi, LiveUnequalPhi,
    SecondConsumer, PoisonConjunction, UndefinedConjunction, MaskedFloat, UnmaskedFloat,
    DeadLoad, DeadDivide, DeadDerivative, BodyStore, BodyEscape, SecondUnsafeVote,
    BitcastPoisonConjunction, MaskedBitcastPoison,
    UndefinedReconsume, NonEntry, Nested, Reentered, BuiltinReconsume, LocationReconsume,
    BufferPredicate, BitcastPoisonSelection, FloatPoisonConjunction, FloatPoisonSelection,
};
enum class Predicate { Helpers, Visible, AllFalse, AllTrue };

// Independent project-owned SSA, not title bytes or the recompiler's own emission. All-false
// waves may skip the original region. Mixed waves execute it for every guest invocation while
// Select preserves each lane's integer state; the live Boolean Phi also governs termination.
inline std::vector<uint32_t> make_module(Shape shape = Shape::Masked,
                                        Predicate predicate = Predicate::Helpers,
                                        bool poison_helpers = false) {
    namespace base = wave_width;
    std::vector<uint32_t> w{0x07230203u, 0x00010300u, 0, 180, 0};
    const auto op = [&](uint32_t code, std::initializer_list<uint32_t> args) {
        base::instruction(w, code, args);
    };
    const bool floating = shape == Shape::MaskedFloat || shape == Shape::UnmaskedFloat ||
        shape == Shape::BitcastPoisonConjunction || shape == Shape::BitcastPoisonSelection ||
        shape == Shape::MaskedBitcastPoison || shape == Shape::FloatPoisonConjunction ||
        shape == Shape::FloatPoisonSelection ||
        shape == Shape::UndefinedReconsume || shape == Shape::BuiltinReconsume ||
        shape == Shape::LocationReconsume;
    const bool reconsume = shape == Shape::UndefinedReconsume || shape == Shape::BuiltinReconsume ||
        shape == Shape::LocationReconsume;
    op(17, {1}); op(17, {61}); op(17, {62});
    if (floating) {
        op(17, {4466}); base::extension(w, "SPV_KHR_float_controls");
        op(11, {90, 0x4c534c47u, 0x6474732eu, 0x3035342eu, 0});
    }
    op(14, {0, 1});
    if (shape == Shape::LocationReconsume) op(15, {4, 40, 0x6e69616du, 0, 22, 23, 24, 123});
    else op(15, {4, 40, 0x6e69616du, 0, 22, 23, 24});
    op(16, {40, 7});
    if (floating) op(16, {40, 4461, 32});
    std::vector<uint32_t> metadata;
    fragment_votes::marker(metadata, "Prosper.FragmentSubgroupSize=64");
    fragment_votes::marker(metadata, "Prosper.FragmentSubgroupWhy=2");
    w.insert(w.end(), metadata.begin(), metadata.end());
    op(71, {22, 11, 15}); op(71, {23, 30, 0}); op(71, {24, 11, 23});
    if (shape == Shape::LocationReconsume) op(71, {123, 30, 1});
    if (shape == Shape::BufferPredicate) {
        op(71, {126, 6, 4}); op(71, {127, 2}); op(72, {127, 0, 35, 0});
        op(71, {131, 34, 0}); op(71, {131, 33, 5});
    }
    op(19, {1}); op(20, {2}); op(21, {3, 32, 0}); op(22, {4, 32});
    op(33, {5, 1}); op(23, {6, 4, 4}); op(32, {7, 1, 6}); op(32, {18, 3, 6});
    op(32, {19, 1, 2}); op(32, {26, 7, 2});
    op(43, {3, 8, 0}); op(43, {3, 9, 64}); op(43, {3, 11, 3});
    op(43, {4, 12, 0}); op(43, {4, 13, 0x3f800000u});
    op(41, {2, 14}); op(42, {2, 15}); op(43, {4, 16, 0x3e000000u});
    op(43, {4, 17, 0x3b808081u}); // 1/255, exact integer-state observation
    if (shape == Shape::BufferPredicate) {
        op(29, {126, 3}); op(30, {127, 126}); op(32, {128, 12, 127}); op(32, {129, 12, 3});
    }
    if (shape == Shape::BitcastPoisonConjunction || shape == Shape::BitcastPoisonSelection ||
        shape == Shape::MaskedBitcastPoison || shape == Shape::FloatPoisonConjunction ||
        shape == Shape::FloatPoisonSelection)
        op(43, {4, 89, 0x7f800000u}); // +Inf, NOT finite bitcast authority
    op(59, {7, 22, 1}); op(59, {18, 23, 3}); op(59, {19, 24, 1});
    if (shape == Shape::LocationReconsume) op(59, {7, 123, 1});
    if (shape == Shape::BufferPredicate) op(59, {128, 131, 12});
    if (shape == Shape::UndefinedConjunction) op(1, {2, 54});
    if (shape == Shape::UndefinedReconsume) op(1, {3, 117});
    op(54, {1, 40, 0, 5});
    if (shape == Shape::NonEntry || shape == Shape::Reentered) {
        op(248, {44}); op(249, {45});
    }
    if (shape == Shape::Nested) {
        op(248, {44}); op(61, {2, 117, 24});
        op(247, {118, 0}); op(250, {117, 45, 118});
    }
    op(248, {45});
    if (shape == Shape::Reentered) {
        op(246, {118, 119, 0}); op(249, {46}); op(248, {46});
    }
    if (shape == Shape::PoisonConjunction || shape == Shape::DeadLoad) op(59, {26, 25, 7});
    op(61, {6, 50, 22}); op(81, {4, 51, 50, 0}); op(81, {4, 52, 50, 1});
    op(61, {2, 53, 24});
    if (shape == Shape::PoisonConjunction) op(61, {2, 54, 25});
    uint32_t p = 53;
    if (shape == Shape::BufferPredicate) {
        op(65, {129, 132, 131, 8, 8}); op(61, {3, 133, 132}); op(171, {2, 55, 133, 8}); p = 55;
    }
    if (shape == Shape::BuiltinReconsume) op(124, {3, 117, 51});
    if (shape == Shape::LocationReconsume) {
        op(61, {6, 124, 123}); op(81, {4, 125, 124, 0}); op(124, {3, 117, 125});
    }
    if (reconsume) {
        op(124, {4, 118, 117}); op(180, {2, 119, 118, 12});
        op(167, {2, 55, 15, 119}); p = 55;
    }
    if (predicate != Predicate::Helpers) {
        if (predicate == Predicate::Visible) op(168, {2, 55, 53});
        if (predicate == Predicate::AllFalse) op(167, {2, 55, 53, 15});
        if (predicate == Predicate::AllTrue) op(166, {2, 55, 53, 14});
        p = 55;
    }
    op(335, {2, 60, 11, p}); op(247, {70, 0}); op(250, {60, 65, 70}); op(248, {65});
    uint32_t new_state = 9;
    if (shape == Shape::MaskedFloat || shape == Shape::UnmaskedFloat) {
        // A result that can be poison is harmless only in the unselected state arm. F32 numerical
        // results are never the oracle for raw integer export identity.
        op(12, {4, 75, 90, 8, 51}); op(12, {4, 76, 90, 14, 75});
        op(129, {4, 77, 76, 13}); op(133, {4, 78, 77, 13});
        op(12, {4, 79, 90, 10, 78}); op(124, {3, 85, 79}); new_state = 85;
    }
    if (shape == Shape::BitcastPoisonConjunction || shape == Shape::BitcastPoisonSelection ||
        shape == Shape::MaskedBitcastPoison) {
        op(124, {3, 85, 89});
        if (shape == Shape::MaskedBitcastPoison) new_state = 85;
    }
    if (shape == Shape::FloatPoisonConjunction || shape == Shape::FloatPoisonSelection)
        op(12, {4, 85, 90, 4, 89}); // FAbs(+Inf): implicit NotInf remains under SZI preserve
    if (reconsume) {
        op(124, {4, 120, 117}); op(180, {2, 121, 120, 12});
        op(167, {2, 81, p, 121});
    }
    if (shape == Shape::ScalarExport) op(83, {3, 80, 9});
    else op(169, {3, 80, p, new_state, 8});
    if (shape == Shape::Termination) op(168, {2, 82, p});
    else if (shape == Shape::TerminationExport) op(83, {2, 82, 15});
    else if (shape == Shape::PoisonConjunction || shape == Shape::UndefinedConjunction) {
        op(167, {2, 81, p, 54}); op(168, {2, 82, 81});
    } else if (shape == Shape::UnmaskedFloat) {
        op(186, {2, 81, 79, 12}); op(167, {2, 84, p, 81}); op(168, {2, 82, 84});
    } else if (shape == Shape::BitcastPoisonConjunction || shape == Shape::BitcastPoisonSelection ||
               shape == Shape::FloatPoisonConjunction || shape == Shape::FloatPoisonSelection) {
        if (shape == Shape::FloatPoisonConjunction || shape == Shape::FloatPoisonSelection)
            op(186, {2, 81, 85, 12});
        else op(171, {2, 81, 85, 8});
        // The paired modules differ only in this architectural mask instruction. Strict AND
        // propagates comparison poison into live EXEC; Select suppresses it when EXEC is false.
        if (shape == Shape::BitcastPoisonSelection || shape == Shape::FloatPoisonSelection)
            op(169, {2, 84, p, 81, 15});
        else op(167, {2, 84, p, 81});
        op(168, {2, 82, 84});
    } else if (reconsume) op(168, {2, 82, 81});
    else op(83, {2, 82, 14});
    if (shape == Shape::DeadLoad) op(61, {2, 86, 25});
    if (shape == Shape::DeadDivide) op(134, {3, 86, 9, 8});
    if (shape == Shape::DeadDerivative) op(207, {4, 86, 51});
    if (shape == Shape::BodyStore) { op(80, {6, 87, 12, 12, 12, 13}); op(62, {23, 87}); }
    op(249, {70}); op(248, {70});
    const uint32_t header = shape == Shape::Reentered ? 46 : 45;
    op(245, {3, 100, 8, header, 80, 65}); op(245, {2, 101, 14, header, 82, 65});
    if (shape == Shape::DeadUnequalPhi || shape == Shape::LiveUnequalPhi)
        op(245, {3, 102, 8, 45, 9, 65});
    uint32_t keep = 101;
    if (shape == Shape::LiveUnequalPhi) { op(170, {2, 103, 102, 8}); keep = 103; }
    if (shape == Shape::SecondConsumer) { op(167, {2, 103, 101, 60}); keep = 103; }
    if (shape == Shape::SecondUnsafeVote) { op(335, {2, 103, 11, 53}); keep = 103; }
    if (shape == Shape::BodyEscape) keep = 82;
    op(247, {105, 0}); op(250, {keep, 105, 104}); op(248, {104}); op(252, {}); op(248, {105});
    uint32_t x = 51, y = 52;
    if (poison_helpers) { op(169, {4, 106, 53, 12, 51}); op(169, {4, 107, 53, 12, 52}); x = 106; y = 107; }
    op(207, {4, 110, x}); op(208, {4, 111, y});
    op(133, {4, 112, 110, 16}); op(133, {4, 113, 111, 16});
    op(112, {4, 114, 100}); op(133, {4, 115, 114, 17});
    op(80, {6, 116, 112, 113, 115, 13}); op(62, {23, 116});
    if (shape == Shape::Nested) { op(249, {118}); op(248, {118}); }
    if (shape == Shape::Reentered) {
        op(249, {119}); op(248, {119}); op(250, {53, 45, 118}); op(248, {118});
    }
    op(253, {}); op(56, {});
    return w;
}

struct Fixture { const char* name; Shape shape; bool admitted; bool strict = true; };
inline std::vector<Fixture> fixtures() {
    return {{"masked_state", Shape::Masked, true}, {"live_termination", Shape::Termination, true},
        {"unmasked_scalar", Shape::ScalarExport, false}, {"unmasked_termination", Shape::TerminationExport, false},
        {"dead_unequal_phi", Shape::DeadUnequalPhi, true}, {"live_unequal_phi", Shape::LiveUnequalPhi, false},
        {"second_consumer", Shape::SecondConsumer, false}, {"poison_conjunction", Shape::PoisonConjunction, false},
        {"undefined_conjunction", Shape::UndefinedConjunction, true}, {"masked_float", Shape::MaskedFloat, true},
        {"unmasked_float", Shape::UnmaskedFloat, false}, {"dead_load", Shape::DeadLoad, false},
        {"dead_divide", Shape::DeadDivide, false}, {"dead_derivative", Shape::DeadDerivative, false},
        {"body_store", Shape::BodyStore, false},
        // This graph deliberately violates SSA dominance; it tests the caller's escape guard,
        // not a strict source/effective SPIR-V path or an executable shader.
        {"body_escape_parser_only", Shape::BodyEscape, false, false},
        {"second_unsafe_vote", Shape::SecondUnsafeVote, false},
        {"bitcast_poison_conjunction", Shape::BitcastPoisonConjunction, false},
        {"bitcast_poison_selection", Shape::BitcastPoisonSelection, true},
        {"float_poison_conjunction", Shape::FloatPoisonConjunction, false},
        {"float_poison_selection", Shape::FloatPoisonSelection, true},
        {"masked_bitcast_poison", Shape::MaskedBitcastPoison, true},
        {"undefined_reconsume", Shape::UndefinedReconsume, false},
        {"non_entry_selection", Shape::NonEntry, false}, {"nested_selection", Shape::Nested, false},
        {"reentered_selection", Shape::Reentered, false},
        {"builtin_reconsume", Shape::BuiltinReconsume, true},
        {"location_reconsume", Shape::LocationReconsume, false},
        {"buffer_predicate_uncertified", Shape::BufferPredicate, true}};
}
} // namespace prosper::test::fragment_neutral
