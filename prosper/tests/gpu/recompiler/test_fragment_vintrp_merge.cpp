// #4933: omitting native interpolation writers leaves an arm-local SSA value alive after
// its first merge. A later selection then emits a phi whose skipped incoming value does not
// dominate that predecessor. Inspect actual public compiler modules; spv_validate checks them
// independently. Storage inventory never establishes genuine per-lane launch input authority.
#include "fixtures/fragment_vintrp_merge.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace prosper::gpu {
namespace {
namespace fixture = prosper::test::fragment_vintrp_merge;
struct Module {
    struct Instruction {
        uint32_t opcode, block;
        std::vector<uint32_t> operands;
    };
    std::vector<Instruction> instructions;
    std::map<uint32_t, uint32_t> constants;
    std::vector<uint32_t> selections;
    explicit Module(const std::vector<uint32_t>& words) {
        uint32_t block = 0;
        for (size_t at = 5; at < words.size();) {
            const size_t width = words[at] >> 16;
            if (!width || width > words.size() - at) ADD_FAILURE() << "malformed SPIR-V";
            if (!width || width > words.size() - at) break;
            const uint32_t op = words[at] & 0xffffu;
            const std::vector<uint32_t> operands(words.begin() + at + 1,
                                                 words.begin() + at + width);
            if (op == 248 && operands.size() == 1) block = operands[0];
            if (op == 247 && operands.size() == 2) selections.push_back(operands[0]);
            if (op == 43 && operands.size() == 3) constants[operands[1]] = operands[2];
            instructions.push_back({op, block, operands});
            at += width;
        }
    }
    std::set<uint32_t> interpolation_words() const {
        std::set<uint32_t> loads, components, words;
        for (const auto& in : instructions) {
            const auto& a = in.operands;
            if (in.opcode == 61 && a.size() == 3) loads.insert(a[1]);
            if (in.opcode == 81 && a.size() == 4 && loads.contains(a[2])) components.insert(a[1]);
            if (in.opcode == 124 && a.size() == 3 && components.contains(a[2])) words.insert(a[1]);
        }
        return words;
    }
    bool first_merge_retains_interpolation_and_old_value() const {
        if (selections.empty()) return false;
        const auto words = interpolation_words();
        for (const auto& in : instructions) {
            const auto& a = in.operands;
            if (in.opcode != 245 || in.block != selections[0] || a.size() != 6) continue;
            for (size_t edge : {2u, 4u}) {
                const size_t other = edge == 2 ? 4 : 2;
                const auto old = constants.find(a[other]);
                if (words.contains(a[edge]) && old != constants.end() && old->second == 7)
                    return true;
            }
        }
        return false;
    }
};
std::vector<uint32_t> compile(const std::vector<uint32_t>& code, uint64_t address,
                              const FragmentInterpolationLayout* layout = nullptr) {
    return recompile_fragment(code.data(), code.size(), nullptr, nullptr, UINT32_MAX, layout, false,
                              {RecompileDiagnosticStage::Fragment, address});
}
std::set<int> writes(uint32_t opcode, bool owned = false) {
    const uint32_t word = fixture::interpolation(opcode);
    std::vector<Rdna2Inst> ins;
    EXPECT_EQ(rdna2_walk(&word, 1, ins), 1u);
    std::set<int> vector, scalar;
    loop_written_regs(ins, 0, 1, vector, scalar, owned);
    EXPECT_TRUE(scalar.empty());
    return vector;
}
void merged(uint32_t opcode, bool one_arm, uint64_t address) {
    const auto words = compile(fixture::program(opcode, one_arm), address);
    ASSERT_FALSE(words.empty()) << last_terminal_reject_reason(address);
    const Module module(words);
    ASSERT_EQ(module.selections.size(), 2u) << "exercise two genuine structured selections";
    ASSERT_EQ(module.interpolation_words().size(), 1u);
    EXPECT_TRUE(module.first_merge_retains_interpolation_and_old_value())
        << "the first join must merge the arm-local interpolation with genuine old v3=7";
}
} // namespace

TEST(FragmentVintrpMerge, NativeP2InventoriesTheNumericDestination) {
    EXPECT_EQ(writes(1), (std::set<int>{3}));
}
TEST(FragmentVintrpMerge, NativeMovInventoriesTheNumericDestination) {
    EXPECT_EQ(writes(2), (std::set<int>{3}));
}
TEST(FragmentVintrpMerge, NativeP1RemainsANoop) {
    EXPECT_TRUE(writes(0).empty());
    const auto words = compile(fixture::program(0), 0xa493301u);
    ASSERT_FALSE(words.empty());
    const Module module(words);
    ASSERT_EQ(module.selections.size(), 2u);
    EXPECT_TRUE(module.interpolation_words().empty());
    EXPECT_FALSE(module.first_merge_retains_interpolation_and_old_value());
}
TEST(FragmentVintrpMerge, OwnedP1KeepsItsPartialDestinationStorage) {
    EXPECT_EQ(writes(0, true), (std::set<int>{3}));
}
TEST(FragmentVintrpMerge, NativeP2ValueCrossesTwoSelections) {
    merged(1, false, 0xa493302u);
}
TEST(FragmentVintrpMerge, NativeMovValueCrossesTwoSelections) {
    merged(2, false, 0xa493303u);
}
TEST(FragmentVintrpMerge, NativeP2OneArmPreservesTheSkippedValue) {
    merged(1, true, 0xa493304u);
}
TEST(FragmentVintrpMerge, NativeMovOneArmPreservesTheSkippedValue) {
    merged(2, true, 0xa493305u);
}
TEST(FragmentVintrpMerge, MissingMovParameterStillRefuses) {
    FragmentInterpolationLayout layout;
    layout.requires_geometry = true; // explicitly absent P10; default derivation allocates it
    const auto words = compile(fixture::program(2, false, 0), 0xa493306u, &layout);
    EXPECT_TRUE(words.empty()) << "P10 is unavailable without geometry parameter ownership";
    EXPECT_NE(last_terminal_reject_reason(0xa493306u).find("pc=6"), std::string::npos);
}
} // namespace prosper::gpu
