// Synthetic coverage consumers need actual input producers before they can test memory lowering.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace prosper::test::coverage_inputs {
inline constexpr std::array<uint32_t, 7> flat_consumer{
    0xd70f6a01u, 0x00020c0cu, // v_add_co_u32 v1, vcc, s12, v6
    0x50041af9u, 0x86860680u, // v_add_co_ci_u32 v2 (SDWA), s13
    0xdc200000u, 0x007d0001u, // flat_load_ubyte v0, v[1:2]
    0xbf810000u,
};
template <std::size_t N, std::size_t P>
constexpr auto with_inputs(const std::array<uint32_t, N>& consumer,
                           const std::array<uint32_t, P>& producers) {
    std::array<uint32_t, N + P> result{};
    for (std::size_t i = 0; i < P; ++i) result[i] = producers[i];
    for (std::size_t i = 0; i < N; ++i) result[P + i] = consumer[i];
    return result;
}
inline constexpr std::array flat_offset_input{0x7e0c0300u}; // v6 = real local-ID v0
inline constexpr auto flat_with_offset = with_inputs(flat_consumer, flat_offset_input);

inline constexpr auto wave32_half_mask_consumer = std::to_array<uint32_t>({
    0x7C000000u, // v_cmp_f_f32 vcc, s0, v0
    0xBEEA076Au, // s_not_b32 vcc_lo, vcc_lo
    0xBEFE036Au, // s_mov_b32 exec_lo, vcc_lo
    0x7E000281u, // v_mov_b32 v0, 1 (EXEC-predicated)
    0xBEFE03C1u, // s_mov_b32 exec_lo, -1
    0xBF810000u,
});
inline constexpr std::array wave32_half_mask_input{
    0xbe800380u}; // s0 = 0, actual scalar operand of the compare
inline constexpr auto wave32_half_mask =
    with_inputs(wave32_half_mask_consumer, wave32_half_mask_input);

inline constexpr auto wave32_vcc_gt_zero_consumer = std::to_array<uint32_t>({
    0x7C022880u, // v_cmp_eq_f32 vcc, 0, v20
    0xBF08806Au, // s_cmp_gt_u32 vcc_lo, 0
    0xBF810000u,
});
inline constexpr std::array wave32_vcc_gt_zero_input{0x7e280300u}; // v20 = real local-ID v0
inline constexpr auto wave32_vcc_gt_zero =
    with_inputs(wave32_vcc_gt_zero_consumer, wave32_vcc_gt_zero_input);

inline constexpr auto compute_structured_saved_mbcnt_b64_consumer = std::to_array<uint32_t>({
    0xBE84047Eu,                // s_mov_b64 s[4:5], exec
    0x7E060280u,                // v_mov_b32 v3, 0 (merge live-in)
    0x7C020300u,                // v_cmp_lt_f32 vcc, v0, v1 (varying guest-wave branch)
    0xBF860001u,                // s_cbranch_vccz +1 -> merge
    0x7E060281u,                // guarded write keeps the structured branch live
    0xD7650003u,
    0x00010004u,   // v_mbcnt_lo_u32_b32 v3, s4, 0
    0xD7660003u,
    0x00020605u,   // v_mbcnt_hi_u32_b32 v3, s5, v3
    0xBF810000u,
});
inline constexpr std::array compute_structured_saved_mbcnt_b64_input{
    0x7e020281u}; // v1 = 1, second operand of the varying compare
inline constexpr auto compute_structured_saved_mbcnt_b64 = with_inputs(
    compute_structured_saved_mbcnt_b64_consumer, compute_structured_saved_mbcnt_b64_input);

inline constexpr auto mask_nor_consumer = std::to_array<uint32_t>({
    0x7C040CF9u,
    0x06869880u,
    0x8DEA6A18u,
    0xBF810000u,
});
inline constexpr std::array mask_nor_input{0x7e0c0300u}; // v6 = real local-ID v0
inline constexpr auto mask_nor = with_inputs(mask_nor_consumer, mask_nor_input);

inline constexpr auto compute_cfg_dispatch_saved_mask_not_consumer = std::to_array<uint32_t>({
    0xBE94037Eu, // s_mov_b32 s20, exec_lo (complete Wave32 mask)
    0xBEEA0381u, // s_mov_b32 vcc_lo, 1 (older scalar-data lifetime)
    0xBF070000u, // s_cmp_lg_u32 s0, s0 (false)
    0xBF850003u, // s_cbranch_scc1 +3 -> mask merge
    0x856B807Eu, // s_cselect_b32 vcc_hi, exec_lo, 0
    0xBEEA0714u, // s_not_b32 vcc_lo, s20
    0x8D6A6B6Au, // s_nor_b32 vcc_lo, vcc_lo, vcc_hi
});
inline constexpr std::array compute_cfg_dispatch_saved_mask_not_input{
    0xbe800380u}; // s0 = 0, genuine SCC operands
inline constexpr auto compute_cfg_dispatch_saved_mask_not = with_inputs(
    compute_cfg_dispatch_saved_mask_not_consumer, compute_cfg_dispatch_saved_mask_not_input);

inline constexpr auto compute_cfg_dispatch_saved_mask_not_boundary_consumer =
    std::to_array<uint32_t>({
        0xBE94037Eu, // s_mov_b32 s20, exec_lo (complete Wave32 mask)
        0xBE950714u, // s_not_b32 s21, s20
        0xBF060000u, // s_cmp_eq_u32 s0, s0 (true; creates a conditional CFG edge)
        0xBF850001u, // s_cbranch_scc1 +1 -> consumer
        0xBF800000u, // unreachable s_nop 0 fallthrough arm
        0xBEFE0315u, // s_mov_b32 exec_lo, s21 (mask-only consumer)
    });
inline constexpr std::array compute_cfg_dispatch_saved_mask_not_boundary_input{
    0xbe800380u}; // s0 = 0, genuine SCC operands
inline constexpr auto compute_cfg_dispatch_saved_mask_not_boundary =
    with_inputs(compute_cfg_dispatch_saved_mask_not_boundary_consumer,
                compute_cfg_dispatch_saved_mask_not_boundary_input);

inline constexpr uint32_t dispatch_tail[] = {
    0xBE840380u,   // s4=0: explicit input of the tail SCC comparison
    0xBE800380u, 0x7E000280u, 0x7E020300u, 0xD7610013u, 0x00014A7Eu, 0xD7610013u, 0x0001507Fu,
    0xD760000Eu, 0x00014B13u, 0xD760000Fu, 0x00015113u, 0xBEFE040Eu, 0xE00C2000u, 0x80020400u,
    0x7DB900F9u, 0x86050007u, 0x7D020200u, 0xBF860006u, 0xBF0A8204u, 0x360000FDu, 0xBF840001u,
    0x81008100u, 0x81008100u, 0xBF82FFF4u, 0xBF810000u,
};
inline constexpr auto x16_wave_mask_consumer = std::to_array<uint32_t>({
    0xf4100100u,
    0xfa000000u,   // pc0: x16 s[4:19]
    0xf0000108u,
    0x00010704u,   // pc2: image_load through s[4:11]
    0xf0000108u,
    0x00030604u,   // pc4: image_load through s[12:19]
    0xbf840001u,                 // pc6: conditional skip of pc7
    0xbf800000u,                 // pc7: no-op
    0x7d0424f9u,
    0x0606840au,   // pc8: v_cmp_* s[4:5], v10, v18
    0x7e020205u,                 // pc10: v_mov_b32 v1,s5
    0xbf810000u,
});
inline constexpr std::array x16_wave_mask_inputs{
    0x7e140300u, // v10 = genuine local-ID v0
    0x7e240280u, // v18 = 0
    0xbf068080u, // independent SCC: s_cmp_eq_u32 0, 0
};
inline constexpr auto x16_wave_mask_lifetime =
    with_inputs(x16_wave_mask_consumer, x16_wave_mask_inputs);
inline constexpr auto mask_scc_consumer = std::to_array<uint32_t>({
    0xBE9B037Eu,              // s_mov_b32 s27, exec_lo
    0x7D8A0A15u,              // v_cmp_* vcc, s21, v5       (Astro PC1079)
    0x7D8A0AF9u,
    0x0686EB19u, // v_cmp_* s107, s25, v5      (Astro PC1080)
    0x7D8A0AF9u,
    0x06869C1Au, // v_cmp_* s28, s26, v5       (Astro PC1082)
    0x876A6A1Bu,              // s_and_b32 vcc_lo, s27, vcc_lo
    0x876A6B6Au,              // s_and_b32 vcc_lo, vcc_lo, vcc_hi
    0x876A1C6Au,              // s_and_b32 vcc_lo, vcc_lo, s28
    0xBF840001u,              // s_cbranch_scc0 +1
    0xBF800000u,              // fallthrough work
});
inline constexpr std::array mask_scc_inputs{
    0xbe950380u, // s21 = 0
    0xbe990381u, // s25 = 1
    0xbe9a0382u, // s26 = 2
    0x7e0a0300u, // v5 = genuine local-ID v0
};
inline constexpr auto mask_scc_prefix = with_inputs(mask_scc_consumer, mask_scc_inputs);
inline constexpr std::array scalar_offset_input{0x7e880300u}; // v68 = genuine local-ID v0
inline constexpr std::array scalar_offset_compare{0x7d288812u, 0xbf810000u};
} // namespace prosper::test::coverage_inputs
