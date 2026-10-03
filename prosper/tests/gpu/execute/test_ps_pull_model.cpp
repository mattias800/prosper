// #4230: the real registered draw producer must retain a distinct pull triple and matching
// qualifiers. Device-free SOURCE/packing evidence; pixel arithmetic is tested separately.
#include "fixtures/ps_pull_model_fixture.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>

namespace e = prosper::test::ps_pull;
namespace {
bool has_decoration_at(const std::vector<uint32_t>& module, uint32_t location,
                       uint32_t decoration) {
    std::set<uint32_t> located, decorated;
    for (size_t pc = 5; pc < module.size();) {
        const uint32_t words = module[pc] >> 16, op = module[pc] & 0xffff;
        if (!words || words > module.size() - pc) return false;
        if (op == 71 && words >= 3) { // SPIR-V OpDecorate
            if (module[pc + 2] == 30 && words == 4 && module[pc + 3] == location)
                located.insert(module[pc + 1]); // Location
            if (module[pc + 2] == decoration) decorated.insert(module[pc + 1]);
        }
        pc += words;
    }
    for (uint32_t variable : located)
        if (decorated.contains(variable)) return true;
    return false;
}
void retain(const char* name, const prosper::gpu::DrawItem& draw) {
    // Diagnostic CPU-test-only SOURCE retention. Default cases create no files or guest selectors.
    const char* root = std::getenv("PROSPER_PS_PULL_SPV_DIRECTORY");
    if (!root || !*root) return;
    std::filesystem::create_directories(root);
    for (const auto& [stage, module] :
         std::array{std::pair{"vs", &draw.vs_words()}, std::pair{"gs", &draw.gs},
                    std::pair{"fs", &draw.fs_words()}}) {
        std::ofstream file(std::filesystem::path(root) / (std::string(name) + "-" + stage + ".spv"),
                           std::ios::binary);
        file.write(reinterpret_cast<const char*>(module->data()),
                   static_cast<std::streamsize>(module->size() * sizeof(uint32_t)));
        file.close();
        EXPECT_TRUE(bool(file)) << name << '/' << stage << ": actual SOURCE retention failed";
    }
}
}   // namespace

TEST(PsPullModel, ActualProducingModulesAndPackedGuestReads) {
    for (const auto& c : e::cases) {
        SCOPED_TRACE(c.name);
        const auto raw = e::fragment_words(c);
        std::vector<prosper::gpu::Rdna2Inst> instructions;
        prosper::gpu::rdna2_walk(raw.data(), raw.size(), instructions);
        const size_t first = c.words == e::Words::Pull ? 0 : 1;
        ASSERT_GT(instructions.size(), first + 1);
        EXPECT_EQ(instructions[first].fmt, prosper::gpu::Rdna2Format::VOP1);
        EXPECT_EQ(instructions[first].opcode, 1u);   // original MOV -> actual EXP sink
        EXPECT_EQ(instructions[first].src[0].kind, prosper::gpu::OperandKind::VGPR);
        EXPECT_EQ(instructions[first].src[0].value, c.first_vgpr);
        EXPECT_EQ(instructions[first + 1].src[0].value, c.first_vgpr + 1);
        ASSERT_GT(instructions.size(), first + 3);
        if (c.words != e::Words::LinearAfterReservedPull)
            EXPECT_EQ(instructions[first + 2].src[0].value, c.first_vgpr + 2)
                << "third physical pull word is read, never a synthetic alpha/constant";
        prosper::gpu::DrawItem draw;
        ASSERT_TRUE(e::realize(c, draw)) << "real registration/realization failed";
        ASSERT_FALSE(draw.vs_words().empty());
        ASSERT_FALSE(draw.fs_words().empty());
        ASSERT_FALSE(draw.gs.empty()) << "pull-only original guest code must request its producer";
        EXPECT_EQ(draw.system_inputs.ena, c.ena);
        EXPECT_EQ(draw.system_inputs.addr, c.addr);
        const auto layout = prosper::gpu::fragment_interpolation_layout(
            raw.data(), raw.size(), &draw.system_inputs, &draw.pixel_inputs);
        ASSERT_TRUE(layout.valid && layout.requires_geometry);
        if (c.words == e::Words::Pull) {
            ASSERT_NE(layout.system_locations[3], UINT32_MAX);
            EXPECT_TRUE(has_decoration_at(draw.gs, layout.system_locations[3], 13))
                << "NoPerspective GS pull plane";
            EXPECT_TRUE(has_decoration_at(draw.fs_words(), layout.system_locations[3], 13))
                << "matching FS qualifier";
        } else {
            EXPECT_EQ(layout.system_locations[3], UINT32_MAX)
                << "ADDR reserves disabled pull, never grants its words";
            EXPECT_TRUE(has_decoration_at(draw.gs, layout.system_locations[5], 13));
            EXPECT_TRUE(has_decoration_at(draw.fs_words(), layout.system_locations[5], 13));
            if (c.words == e::Words::CenterAndLinear) {
                EXPECT_FALSE(has_decoration_at(draw.gs, layout.system_locations[1], 13));
                EXPECT_FALSE(has_decoration_at(draw.fs_words(), layout.system_locations[1], 13));
            }
        }
        retain(c.name, draw);
    }
}

TEST(PsPullModel, DisabledOrUnaddressedPullDoesNotInventProducer) {
    const uint32_t raw[]{0x7e0002f2u, 0xf800180fu, 0, 0xbf810000u};
    for (const auto mapping : {prosper::gpu::PixelSystemInputMapping{0, 8},
                               prosper::gpu::PixelSystemInputMapping{8, 0}}) {
        const auto layout =
            prosper::gpu::fragment_interpolation_layout(raw, std::size(raw), &mapping);
        EXPECT_TRUE(layout.valid);
        EXPECT_FALSE(layout.requires_geometry);
        EXPECT_EQ(layout.system_locations[3], UINT32_MAX);
    }
}

TEST(PsPullModel, OracleDistinguishesOldTupleAndThirdWordCorruption) {
    const auto& c = e::cases[2];
    const auto want = e::expected(c, 5, 4);
    const std::array<float, 4> genuine{float(want[0]), float(want[1]), float(want[2]), 1};
    ASSERT_TRUE(e::matches(genuine, want));
    auto wrong = genuine;
    wrong[2] =
        1;   // old producer's denominator, including the formerly constant third physical word
    EXPECT_FALSE(e::matches(wrong, want));
    wrong = {float(want[0] / want[2]), float(want[1] / want[2]), 1, 1};
    EXPECT_FALSE(e::matches(wrong, want)) << "old smooth {I,J,1,1} is not a pull triple";
    wrong = genuine;
    std::swap(wrong[0], wrong[1]);
    EXPECT_FALSE(e::matches(wrong, want)) << "asymmetric sample distinguishes I/J ordering";
}
