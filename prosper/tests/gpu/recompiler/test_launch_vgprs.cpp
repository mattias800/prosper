// test_launch_vgprs: the VGPRs a guest wave starts with (#5146).
//
// The fresh-mask guard (#4943) refuses a predicate formed from a VGPR that no instruction and no
// launch value defined. A hardware launch value can supply DATA only where this backend actually
// binds it; an unsupplied word stays unavailable. The inventory records that same binding contract:
//   - compute: v0..v2 thread ids, as COMPUTE_PGM_RSRC2.TIDIG_COMP_CNT enables them;
//   - pixel: the SPI_PS_INPUT_ENA/ADDR fields, packed in field order, ADDR reserving widths.
// Each accepted arm below has a refused twin that differs only in the launch contract, so neither
// "every VGPR is defined" nor "no launch VGPR is defined" can pass the suite.
#include "gpu/recompiler/rdna2_launch_vgprs.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace prosper::gpu;

namespace {

constexpr const char* kFreshMask = "fabricated-fresh-mask-source";

bool refused_as_fresh_mask(uint64_t address) {
    return last_terminal_reject_reason(address).find(kFreshMask) != std::string::npos;
}

// v_cmp_eq_i32 vcc, 0, v<vgpr>; v8 = vcc ? 0 : 1.0; export (v8, 0, 0, 1.0) to MRT0.
std::vector<uint32_t> fragment_compare(uint32_t vgpr, std::vector<uint32_t> prefix = {}) {
    std::vector<uint32_t> code = std::move(prefix);
    for (const uint32_t word : {
             0x7d040080u | (vgpr << 9),   // v_cmp_eq_i32 vcc, 0, v<vgpr>
             0x7e120280u,                 // v_mov_b32 v9, 0
             0x021012f2u,                 // v_cndmask_b32 v8, 1.0, v9, vcc
             0x7e140280u,                 // v_mov_b32 v10, 0
             0x7e1602f2u,                 // v_mov_b32 v11, 1.0
             0xf800180fu,
             0x0b0a0908u,    // exp mrt0 v8, v9, v10, v11 done vm
             0xbf810000u,                 // s_endpgm
         })
        code.push_back(word);
    return code;
}

std::vector<uint32_t> compile_fragment(const std::vector<uint32_t>& code,
                                       const PixelSystemInputMapping& inputs, uint64_t address) {
    return recompile_fragment(code.data(), code.size(), nullptr, &inputs, UINT32_MAX, nullptr, true,
                              {RecompileDiagnosticStage::Fragment, address});
}

// v_cmp_gt_u32 vcc, 4, v<vgpr>; v3 = vcc ? 1 : 0 (via v4 = 1); s_endpgm.
std::vector<uint32_t> compute_compare(uint32_t vgpr) {
    return {
        0x7d880084u | (vgpr << 9),   // v_cmp_gt_u32 vcc, 4, v<vgpr>
        0x7e080281u,                 // v_mov_b32 v4, 1
        0x02060880u,                 // v_cndmask_b32 v3, 0, v4, vcc
        0xbf810000u,                 // s_endpgm
    };
}

std::vector<uint32_t> compile_compute(uint32_t vgpr, uint32_t tidig, uint64_t address) {
    ComputeShaderConfig config;
    config.local_x = 4;
    config.local_y = tidig >= 1 ? 4 : 1;
    config.local_z = tidig >= 2 ? 4 : 1;
    config.tidig_comp_cnt = tidig;
    const auto code = compute_compare(vgpr);
    return recompile_compute(code.data(), code.size(), nullptr, config,
                             {RecompileDiagnosticStage::Compute, address});
}

}   // namespace

TEST(LaunchVgprInventory, PixelFieldsPackInFieldOrderWithAddrReservingWidths) {
    // PERSP_CENTER (2 words), POS_X..POS_W, FRONT_FACE: the Kena 0x1f02 layout from #5052.
    PixelSystemInputMapping inputs{0x1f02u, 0x1f02u, FragmentFrontFaceEncoding::AllBits};
    const auto launch = pixel_launch_vgprs(inputs, false);
    ASSERT_EQ(launch.size(), 7u);
    const uint32_t fields[] = {1, 1, 8, 9, 10, 11, 12};
    for (uint32_t i = 0; i < launch.size(); ++i) {
        EXPECT_EQ(launch[i].vgpr, i);
        EXPECT_EQ(launch[i].field, fields[i]);
    }
    EXPECT_EQ(launch[1].component, 1u);
    // Native interpolation reads attributes directly; it has no barycentric value to bind.
    EXPECT_EQ(launch[0].state, LaunchVgprState::Unsupplied);
    EXPECT_EQ(pixel_launch_vgprs(inputs, true)[0].state, LaunchVgprState::Supplied);
    for (uint32_t i = 2; i < 7; ++i) EXPECT_EQ(launch[i].state, LaunchVgprState::Supplied);

    // ADDR without ENA reserves the slot and shifts later fields, but loads nothing.
    PixelSystemInputMapping reserved{0x1002u, 0x1102u, FragmentFrontFaceEncoding::SignedFloat};
    const auto shifted = pixel_launch_vgprs(reserved, false);
    ASSERT_EQ(shifted.size(), 4u);
    EXPECT_EQ(shifted[2].field, 8u);
    EXPECT_EQ(shifted[2].state, LaunchVgprState::Reserved);
    EXPECT_EQ(shifted[3].vgpr, 3u);
    EXPECT_EQ(shifted[3].field, 12u);

    // FRONT_FACE needs the observed SPI_BARYC_CNTL encoding; the three unbound fields stay so.
    PixelSystemInputMapping unbound{0xd080u, 0xd080u};
    for (const auto& word : pixel_launch_vgprs(unbound, true))
        EXPECT_EQ(word.state, LaunchVgprState::Unsupplied) << pixel_launch_field_name(word.field);
}

TEST(LaunchVgprInventory, ComputeThreadIdsFollowTidigCompCnt) {
    for (uint32_t tidig = 0; tidig <= 3; ++tidig) {
        const auto launch = compute_launch_vgprs(tidig);
        ASSERT_EQ(launch.size(), (tidig >= 2 ? 2u : tidig) + 1u) << "tidig=" << tidig;
        for (uint32_t i = 0; i < launch.size(); ++i) {
            EXPECT_EQ(launch[i].vgpr, i);
            EXPECT_EQ(launch[i].field, i);
            EXPECT_EQ(launch[i].state, LaunchVgprState::Supplied);
        }
    }
}

TEST(LaunchVgprCompare, ComputeThreadIdCompareIsAcceptedForEachLoadedComponent) {
    for (uint32_t tidig = 0; tidig <= 2; ++tidig) {
        const uint64_t base = 0x51460000u + tidig * 0x100u;
        for (uint32_t vgpr = 0; vgpr <= tidig; ++vgpr)
            EXPECT_FALSE(compile_compute(vgpr, tidig, base + vgpr).empty())
                << "tidig=" << tidig << " v" << vgpr << ": "
                << last_terminal_reject_reason(base + vgpr);
        // Negative control: the next VGPR is not loaded at this TIDIG_COMP_CNT.
        const uint64_t unloaded = base + 0x80u;
        EXPECT_TRUE(compile_compute(tidig + 1, tidig, unloaded).empty()) << "tidig=" << tidig;
        EXPECT_TRUE(refused_as_fresh_mask(unloaded)) << last_terminal_reject_reason(unloaded);
    }
}

TEST(LaunchVgprCompare, PixelPositionCompareIsAcceptedOnlyWhereEnaLoadsIt) {
    // PERSP_CENTER in v0/v1 and POS_X in v2.
    const PixelSystemInputMapping loaded{0x102u, 0x102u};
    EXPECT_FALSE(compile_fragment(fragment_compare(2), loaded, 0x51461000u).empty())
        << last_terminal_reject_reason(0x51461000u);
    // Same VGPR, POS_X reserved by ADDR only: hardware loads nothing into v2.
    const PixelSystemInputMapping reserved{0x2u, 0x102u};
    EXPECT_TRUE(compile_fragment(fragment_compare(2), reserved, 0x51461001u).empty());
    EXPECT_TRUE(refused_as_fresh_mask(0x51461001u));
    // A VGPR past the launch set was never written by anything.
    EXPECT_TRUE(compile_fragment(fragment_compare(3), loaded, 0x51461002u).empty());
    EXPECT_TRUE(refused_as_fresh_mask(0x51461002u));
}

TEST(LaunchVgprCompare, InterpolatedValueCompareIsAcceptedButRawBarycentricIsNot) {
    const PixelSystemInputMapping center{0x2u, 0x2u};
    const std::vector<uint32_t> interpolate = {
        0xbefc0380u,   // s_mov_b32 m0, 0
        0xc8100000u,   // v_interp_p1_f32 v4, v0, attr0.x
        0xc8110001u,   // v_interp_p2_f32 v4, v1, attr0.x
    };
    EXPECT_FALSE(compile_fragment(fragment_compare(4, interpolate), center, 0x51462000u).empty())
        << last_terminal_reject_reason(0x51462000u);
    // The I barycentric itself is loaded by hardware, but native interpolation has no value for
    // it: comparing it must stay refused rather than compare a zero.
    EXPECT_TRUE(compile_fragment(fragment_compare(0), center, 0x51462001u).empty());
    EXPECT_TRUE(refused_as_fresh_mask(0x51462001u));
}

TEST(LaunchVgprInventory, AncillaryReportsPartialWhileKeepingItsExistingBinding) {
    // The shell currently publishes Layer[10:0] in bits16..26 only (#5160).
    // An inventory must not describe the whole hardware ANCILLARY word as supplied.
    const PixelSystemInputMapping ancillary{1u << 13, 1u << 13};
    const auto launch = pixel_launch_vgprs(ancillary, false);
    ASSERT_EQ(launch.size(), 1u);
    EXPECT_EQ(launch[0].vgpr, 0u);
    EXPECT_EQ(launch[0].field, 13u);
    EXPECT_STREQ(launch_vgpr_state_name(launch[0].state), "partial");
    // A reporting refactor must still bind the existing layer value into v0.
    EXPECT_FALSE(compile_fragment(fragment_compare(0), ancillary, 0x51463000u).empty())
        << last_terminal_reject_reason(0x51463000u);
}
