#pragma once
#include "spirv_wave_width_fixtures.hpp"

namespace prosper::test::fragment_loop_votes {
enum class Shape {
    Counter, Nested, CrossCarried, CrossUndefined, BoolToggle, VaryingBoolUpdate,
    BufferBound, VaryingTrip, UndefinedInit, VaryingInit, VoteInit,
    UnsupportedUpdate, UniformDiamond, VaryingDiamond, UniformSwitch, VaryingSwitch,
    InductionAddress, MaskedInductionAddress, SecondUnsafeVote, SecondUnsafeLoop, CounterWithDeadVote,
    CounterWithDeadMaskedLoad,
    ConstantWithVaryingControl, DeadWithVaryingControl,
};

// Independent synthetic SSA. The same loop carries an initialized integer counter and a
// true/self Boolean Phi. Only the bound/initializer/update/control arm differs in the rejects.
// Fragment coordinates supply real varying data; HelperInvocation makes helper-specific rejects
// explicit. The admitted arm exports analytic derivatives under a loop-dependent vote.
inline std::vector<uint32_t> make_module(Shape shape, uint32_t bound = 2, bool poison = false) {
    std::vector<uint32_t> w{0x07230203u, 0x00010300u, 0, 160, 0};
    const auto op = [&](uint32_t code, std::initializer_list<uint32_t> args) {
        wave_width::instruction(w, code, args);
    };
    op(17, {1}); op(17, {61}); op(17, {62});
    op(14, {0, 1});
    op(15, {4, 40, 0x6e69616du, 0, 22, 23, 24});
    op(16, {40, 7});
    for (const std::string text : {"Prosper.FragmentSubgroupSize=64", "Prosper.FragmentSubgroupWhy=2"}) {
        const size_t at = w.size();
        w.resize(at + 1 + (text.size() + 4) / 4, 0);
        w[at] = (static_cast<uint32_t>(w.size() - at) << 16) | 330;
        for (size_t i = 0; i < text.size(); ++i)
            w[at + 1 + i / 4] |= static_cast<uint32_t>(static_cast<unsigned char>(text[i])) << (8 * (i % 4));
    }
    op(71, {22, 11, 15}); op(71, {23, 30, 0}); op(71, {24, 11, 23});
    op(71, {30, 6, 4}); op(71, {31, 2}); op(72, {31, 0, 35, 0});
    op(71, {35, 34, 0}); op(71, {35, 33, 5});
    op(19, {1}); op(20, {2}); op(21, {3, 32, 0}); op(22, {4, 32});
    op(33, {5, 1}); op(23, {6, 4, 4});
    op(32, {7, 1, 6}); op(32, {18, 3, 6}); op(32, {19, 1, 2});
    op(43, {3, 8, 0}); op(43, {3, 9, 1}); op(43, {3, 10, bound}); op(43, {3, 11, 3});
    op(43, {4, 12, 0}); op(43, {4, 13, 0x3f800000u}); op(41, {2, 14});
    op(42, {2, 15}); op(43, {4, 16, 0x3e000000u}); // .125
    op(43, {4, 21, 0x3b808081u}); // 1/255, observable induction state in blue
    op(44, {6, 17, 12, 12, 13, 13}); // clear-blue sentinel
    op(29, {30, 3}); op(30, {31, 30}); op(32, {32, 12, 31}); op(32, {33, 12, 3});
    op(59, {7, 22, 1}); op(59, {18, 23, 3}); op(59, {19, 24, 1}); op(59, {32, 35, 12});
    if (shape == Shape::UndefinedInit || shape == Shape::CrossUndefined) op(1, {3, 55});
    op(54, {1, 40, 0, 5}); op(248, {45});
    op(62, {23, 17});
    op(61, {6, 50, 22}); op(81, {4, 51, 50, 0}); op(81, {4, 52, 50, 1});
    op(61, {2, 53, 24});
    uint32_t limit = 10, initial = 8, preheader = 45;
    if (shape == Shape::BufferBound) {
        op(65, {33, 36, 35, 8, 8}); op(61, {3, 54, 36}); limit = 54;
    }
    if (shape == Shape::VaryingTrip) { op(169, {3, 54, 53, 9, 10}); limit = 54; }
    if (shape == Shape::UndefinedInit) initial = 55;
    if (shape == Shape::VaryingInit) { op(169, {3, 54, 53, 9, 8}); initial = 54; }
    if (shape == Shape::VoteInit) {
        op(335, {2, 56, 11, 53}); op(169, {3, 54, 56, 9, 8}); initial = 54;
    }
    const bool diamond = shape == Shape::UniformDiamond || shape == Shape::VaryingDiamond;
    const bool selection_switch = shape == Shape::UniformSwitch || shape == Shape::VaryingSwitch;
    if (diamond || selection_switch) {
        op(247, {49, 0});
        if (diamond) op(250, {shape == Shape::UniformDiamond ? 14u : 53u, 46, 47});
        else {
            // Place the selector's definition before SelectionMerge (merge must precede terminator).
            w.resize(w.size() - 3);
            op(169, {3, 57, shape == Shape::UniformSwitch ? 14u : 53u, 8, 9});
            op(247, {49, 0}); op(251, {57, 47, 0, 46});
        }
        op(248, {46}); op(249, {49}); op(248, {47}); op(249, {49});
        op(248, {49}); op(245, {3, 54, 9, 46, 10, 47}); limit = 54; preheader = 49;
    }
    const bool legacy_control = shape == Shape::ConstantWithVaryingControl || shape == Shape::DeadWithVaryingControl;
    if (legacy_control) {
        op(335, {2, 67, 11, shape == Shape::ConstantWithVaryingControl ? 14u : 53u});
        op(169, {3, 66, 67, 9, 8}); // dead pure graph in the second arm
        op(247, {69, 0}); op(250, {53, 65, 69});
        op(248, {65}); op(249, {69}); op(248, {69}); op(253, {}); op(56, {});
        return w;
    }
    const bool nested = shape == Shape::Nested;
    if (nested) {
        op(249, {90}); op(248, {90});
        op(245, {3, 91, 8, preheader, 97, 95}); op(245, {2, 92, 14, preheader, 62, 95});
        op(176, {2, 93, 91, 10}); op(246, {96, 95, 0}); op(250, {93, 94, 96});
        op(248, {94}); preheader = 94;
    }
    op(249, {60}); op(248, {60});
    op(245, {3, 61, initial, preheader, 71, 70});
    const bool changing_bool = shape == Shape::BoolToggle || shape == Shape::VaryingBoolUpdate;
    op(245, {2, 62, nested ? 92u : 14u, preheader, changing_bool ? 125u : 62u, 70});
    const bool cross = shape == Shape::CrossCarried || shape == Shape::CrossUndefined;
    if (cross) {
        op(245, {3, 98, 8, preheader, 99, 70});
        op(245, {3, 99, shape == Shape::CrossUndefined ? 55u : 9u, preheader, 98, 70});
    }
    op(176, {2, 63, 61, limit}); op(246, {69, 70, 0}); op(250, {63, 65, 69});
    op(248, {65});
    if (shape == Shape::InductionAddress || shape == Shape::MaskedInductionAddress) {
        if (shape == Shape::MaskedInductionAddress) op(199, {3, 37, 61, 9});
        op(65, {33, 36, 35, 8, shape == Shape::MaskedInductionAddress ? 37u : 61u});
        op(61, {3, 54, 36}); op(171, {2, 86, 54, 8});
    } else op(171, {2, 86, cross ? 98u : 61u, cross ? 99u : 9u});
    op(167, {2, 66, 62, 86}); op(335, {2, 67, 11, 66});
    if (shape == Shape::CounterWithDeadMaskedLoad) {
        op(199, {3, 37, 61, 9}); op(65, {33, 36, 35, 8, 37}); op(61, {3, 54, 36});
        op(171, {2, 122, 54, 8}); op(335, {2, 123, 11, 122}); op(169, {3, 124, 123, 9, 8});
    }
    op(247, {68, 0}); op(250, {67, 72, 68}); op(248, {72});
    uint32_t x = 51, y = 52;
    if (poison) { op(169, {4, 87, 53, 12, 51}); op(169, {4, 88, 53, 12, 52}); x = 87; y = 88; }
    op(207, {4, 73, x}); op(208, {4, 74, y});
    op(133, {4, 75, 73, 16}); op(133, {4, 76, 74, 16});
    op(80, {6, 77, 75, 76, 12, 13}); op(62, {23, 77});
    op(249, {68}); op(248, {68}); op(249, {70}); op(248, {70});
    op(shape == Shape::UnsupportedUpdate ? 134u : 128u, {3, 71, 61, 9});
    if (shape == Shape::BoolToggle) op(168, {2, 125, 62});
    if (shape == Shape::VaryingBoolUpdate) op(167, {2, 125, 62, 53});
    op(249, {60}); op(248, {69});
    if (nested) {
        op(249, {95}); op(248, {95}); op(128, {3, 97, 91, 9}); op(249, {90}); op(248, {96});
    }
    // Preserve the last derivative result while making the actual final induction state visible.
    // Output was initialized before the loop; these values never supply proof/control roots.
    uint32_t encoded = nested ? 91 : 61;
    if (cross) {
        op(128, {3, 109, 61, 61}); op(128, {3, 110, 109, 109});
        op(128, {3, 111, 99, 99}); op(128, {3, 112, 110, 98});
        op(128, {3, 113, 112, 111}); encoded = 113; // 4*count + x + 2*y
    }
    if (changing_bool) {
        op(128, {3, 109, 61, 61}); op(128, {3, 110, 109, 109});
        op(169, {3, 126, 62, 9, 8}); op(128, {3, 113, 110, 126}); encoded = 113;
    }
    op(61, {6, 114, 23}); op(81, {4, 115, 114, 0}); op(81, {4, 116, 114, 1});
    op(112, {4, 117, encoded}); op(133, {4, 118, 117, 21});
    op(171, {2, 119, nested ? 91u : 61u, 8}); op(169, {4, 120, 119, 118, 13});
    op(80, {6, 121, 115, 116, 120, 13}); op(62, {23, 121});
    if (shape == Shape::SecondUnsafeLoop) {
        op(249, {100}); op(248, {100}); op(245, {3, 101, 8, 69, 105, 104});
        op(169, {3, 106, 53, 9, 10}); op(176, {2, 102, 101, 106});
        op(246, {108, 104, 0}); op(250, {102, 103, 108});
        op(248, {103}); op(249, {104}); op(248, {104});
        op(128, {3, 105, 101, 9}); op(249, {100}); op(248, {108});
    }
    if (shape == Shape::VaryingTrip || shape == Shape::SecondUnsafeVote || shape == Shape::SecondUnsafeLoop) {
        // The loop-header counter is consumed DIRECTLY at exit. No merge Phi/local store can
        // be mistaken for the cause of refusing a varying-trip trace.
        op(170, {2, 80, shape == Shape::SecondUnsafeLoop ? 101u : 61u, 10});
        op(335, {2, 81, 11, shape == Shape::SecondUnsafeVote ? 53u : 80u});
        op(169, {4, 82, 81, 13, 12});
        op(80, {6, 83, 82, 12, 12, 13}); op(62, {23, 83});
    }
    if (shape == Shape::CounterWithDeadVote) {
        op(335, {2, 81, 11, 53}); op(169, {3, 82, 81, 9, 8});
    }
    op(253, {}); op(56, {});
    return w;
}
} // namespace prosper::test::fragment_loop_votes
