// #4935: a LOW symbolic reload is not a complete saved mask. Executing production portable
// modules must preserve the other half's current producer; a stale Bool silently disables lanes.
#include "bpermute_spirv_oracle.hpp"
#include "fixtures/portable_mask_slots.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {
namespace f = prosper::test::portable_mask_slots;
namespace base = prosper::test::portable_mask_half;
void expect_values(const std::vector<uint32_t>& code, uint64_t mask, bool dispatcher = false,
                   uint32_t old_input_words = 0) {
    const auto attempt = f::compile(code, dispatcher, old_input_words);
    const auto& module = attempt.module;
    ASSERT_FALSE(module.empty());
    bpermute_oracle::Interpreter oracle(module);
    oracle.supplied_push_constants.resize(old_input_words, UINT32_MAX);
    const auto actual = oracle.run_buffers(64, {{3, std::vector<uint32_t>(64, 0xdeadbeef)}}, 3);
    ASSERT_TRUE(oracle.error.empty()) << oracle.error;
    ASSERT_EQ(actual.size(), 64u);
    for (uint32_t lane = 0; lane < 64; ++lane)
        EXPECT_EQ(actual[lane], ((mask >> lane) & 1u) ? 9u : 0u) << "logical lane " << lane;
    for (size_t pc = 5; pc < module.size();) {
        const auto width = module[pc] >> 16;
        ASSERT_NE(width, 0u);
        EXPECT_NE(module[pc] & 0xffffu, 339u) << "symbolic halves need no late ballot";
        pc += width;
    }
}
void expect_refusal(const std::vector<uint32_t>& code, const char* reason,
                    bool dispatcher = false) {
    const auto attempt = f::compile(code, dispatcher);
    EXPECT_TRUE(attempt.module.empty());
    EXPECT_NE(attempt.refusal.find(reason), std::string::npos) << attempt.refusal;
}
} // namespace

TEST(PortableMaskSlots, ZeroLowAndFullHighKeepOnlyTheHighLanes) {
    expect_values(f::program(false, true), 0xffffffff00000000ull);
}
TEST(PortableMaskSlots, FullLowAndZeroHighKeepOnlyTheLowLanes) {
    expect_values(f::program(true, false), UINT32_MAX);
}
TEST(PortableMaskSlots, UnchangedFullHalvesKeepAllLanes) {
    expect_values(f::program(true, true), UINT64_MAX);
}
TEST(PortableMaskSlots, UnchangedEmptyHalvesKeepNoLanes) {
    expect_values(f::program(false, false), 0);
}
TEST(PortableMaskSlots, HighReloadBeforeLowKeepsTheCurrentHalfRoles) {
    expect_values(f::program(false, true, true), 0xffffffff00000000ull);
    expect_values(f::program(true, false, true), UINT32_MAX);
}
TEST(PortableMaskSlots, NamedSlotNumbersDoNotDetermineHalfIdentity) {
    expect_values(f::program(false, true, false, 41, 7), 0xffffffff00000000ull);
}
TEST(PortableMaskSlots, ScalarCopiesKeepBothHalfGenerations) {
    expect_values(f::program(false, true, false, 0, 1, true), 0xffffffff00000000ull);
}
TEST(PortableMaskSlots, CountedPostludeKeepsBothHalfGenerations) {
    expect_values(f::program(false, true, false, 0, 1, false, true), 0xffffffff00000000ull);
}
TEST(PortableMaskSlots, MissingHighCannotUseTheLowReloadAsAWholeMask) {
    expect_refusal(f::missing_half(true), "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskSlots, MissingLowCannotInventItsWord) {
    expect_refusal(f::missing_half(false), "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskSlots, TwoLowSpellingsCannotAuthenticateAHighHalf) {
    auto code = f::program(false, true);
    code[6] = (code[6] & ~0x1ffu) | 126u;
    expect_refusal(code, "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskSlots, DispatcherComposesTheSamePhysicalHalfRoles) {
    expect_values(f::program(false, true), 0xffffffff00000000ull, true);
}
TEST(PortableMaskSlots, CompleteB64CopyCannotResurrectOldLaunchWords) {
    auto code = f::program(false, true);
    code.resize(11);
    code.push_back(base::mov(24, 20, true));
    base::sink(code, 24);
    expect_values(code, 0xffffffff00000000ull, false, 26);
}
TEST(PortableMaskSlots, CopiedSlotsKeepTheirValueAfterTheOriginalWordChanges) {
    auto code = f::program(false, true);
    code.resize(11);
    f::spill(code, 20, 17);
    f::spill(code, 21, 23);
    code.push_back(base::mov(21, 128));
    f::reload(code, 24, 17);
    f::reload(code, 25, 23);
    base::sink(code, 24);
    expect_values(code, 0xffffffff00000000ull);
}
TEST(PortableMaskSlots, NumericReadCannotTreatASymbolicHalfAsData) {
    auto code = f::program(false, true);
    code.insert(code.begin() + 11, 0x7e080214u);
    expect_refusal(code, "portable-mask-half-data-unavailable");
}
TEST(PortableMaskSlots, MissingHalfOnOneArmCannotBorrowTheOtherArm) {
    expect_refusal(f::joined(false), "portable-mask-half-slot-unavailable");
    expect_values(f::joined(true), 0);
}
TEST(PortableMaskSlots, JoinRequiresTheSamePhysicalHalfRole) {
    expect_refusal(f::joined(true, true), "portable-mask-half-slot-unavailable");
    expect_values(f::joined(true), 0);
}
TEST(PortableMaskSlots, LoopCarriesTheCurrentPredicateAndRole) {
    expect_values(f::loop_slots(false), 0xffffffff00000000ull);
}
TEST(PortableMaskSlots, LoopCannotMergeLowAndHighRoles) {
    expect_refusal(f::loop_slots(true), "portable-mask-half-pair-unavailable");
}
TEST(PortableMaskSlots, EarlyLaneExitsCannotChangeTheSavedHalfProducers) {
    auto code = f::program(false, true);
    code.resize(11);
    code.insert(code.end(), {0x7e080280u, 0x4a080881u, 0x7da80900u, 0xbf89fffdu});
    base::sink(code);
    expect_values(code, 0xffffffff00000000ull);
}
TEST(PortableMaskSlots, OracleDoesNotInventUnsuppliedLaunchWords) {
    const auto attempt = f::compile(f::program(false, true), false, 26);
    ASSERT_FALSE(attempt.module.empty());
    bpermute_oracle::Interpreter oracle(attempt.module);
    EXPECT_TRUE(oracle.run_buffers(64, {{3, std::vector<uint32_t>(64)}}, 3).empty());
    EXPECT_EQ(oracle.error, "missing supplied push constants");
}
TEST(PortableMaskSlots, OracleRefusesTruncatedLaunchWords) {
    const auto attempt = f::compile(f::program(false, true), false, 26);
    ASSERT_FALSE(attempt.module.empty());
    bpermute_oracle::Interpreter oracle(attempt.module);
    oracle.supplied_push_constants.resize(25, UINT32_MAX);
    EXPECT_TRUE(oracle.run_buffers(64, {{3, std::vector<uint32_t>(64)}}, 3).empty());
    EXPECT_EQ(oracle.error, "unsupported push constant ABI");
}
TEST(PortableMaskSlots, OracleRefusesADifferentPushConstantStride) {
    auto attempt = f::compile(f::program(false, true), false, 26);
    ASSERT_FALSE(attempt.module.empty());
    bpermute_oracle::Interpreter declarations(attempt.module);
    uint32_t array_id = 0;
    for (const auto& global : declarations.globals) {
        if (global.a.size() >= 3 && global.a[2] == 9) {
            ASSERT_EQ(array_id, 0u);
            const auto& block = declarations.ty(declarations.pointee(global.a[0]));
            ASSERT_EQ(block.op, 30u);
            ASSERT_EQ(block.a.size(), 1u);
            array_id = block.a[0];
        }
    }
    ASSERT_NE(array_id, 0u);
    bool changed = false;
    for (size_t pc = 5; pc < attempt.module.size(); pc += attempt.module[pc] >> 16) {
        if ((attempt.module[pc] & 0xffffu) == 71 && attempt.module[pc + 1] == array_id &&
            attempt.module[pc + 2] == 6) {
            attempt.module[pc + 3] = 8;
            changed = true;
            break;
        }
    }
    ASSERT_TRUE(changed);
    bpermute_oracle::Interpreter oracle(attempt.module);
    oracle.supplied_push_constants.resize(26, UINT32_MAX);
    EXPECT_TRUE(oracle.run_buffers(64, {{3, std::vector<uint32_t>(64)}}, 3).empty());
    EXPECT_EQ(oracle.error, "unsupported push constant ABI");
}
TEST(PortableMaskSlots, OracleRefusesAnOutOfBoundsPushConstantRead) {
    auto attempt = f::compile(f::program(false, true), false, 26);
    ASSERT_FALSE(attempt.module.empty());
    bpermute_oracle::Interpreter declarations(attempt.module);
    uint32_t variable = 0, outside = 0;
    for (const auto& global : declarations.globals)
        if (global.a.size() >= 3 && global.a[2] == 9) variable = global.a[1];
    for (const auto& [id, constant] : declarations.constants)
        if (constant.words == std::vector<uint32_t>{26} && declarations.uint_type(constant.type))
            outside = id;
    ASSERT_NE(variable, 0u);
    ASSERT_NE(outside, 0u);
    bool changed = false;
    for (size_t pc = 5; pc < attempt.module.size(); pc += attempt.module[pc] >> 16) {
        if ((attempt.module[pc] & 0xffffu) == 65 && attempt.module[pc + 3] == variable) {
            ASSERT_EQ(attempt.module[pc] >> 16, 6u);
            attempt.module[pc + 5] = outside;
            changed = true;
            break;
        }
    }
    ASSERT_TRUE(changed);
    bpermute_oracle::Interpreter oracle(attempt.module);
    oracle.supplied_push_constants.resize(26, UINT32_MAX);
    EXPECT_TRUE(oracle.run_buffers(64, {{3, std::vector<uint32_t>(64)}}, 3).empty());
    EXPECT_EQ(oracle.error, "AccessChain out of bounds");
}
