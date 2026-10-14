#pragma once
// The VGPRs a guest wave starts with (#5146): which ones RDNA2 hardware loads before the first
// instruction, and which of those the recompiler's backend actually supplies a value for.
//
// One inventory per stage, derived only from the producing launch registers. The stage shells bind
// the SUPPLIED and PARTIAL words into RegState::vreg before seed_vector_word_definedness runs.
// PARTIAL reports the existing incomplete ANCILLARY binding; it does not claim a complete
// architectural word or repair that backend gap (#5160). Every other
// VGPR -- never loaded, reserved by ADDR only, or loaded by hardware but not supplied here -- keeps
// its placeholder mark, and a predicate formed from it is refused instead of reading a zero.
//
// Pixel stage (RDNA2 ISA 70648, PS input VGPRs; SPI_PS_INPUT_ENA / SPI_PS_INPUT_ADDR): sixteen
// fields packed in field order. ADDR reserves each field's width even when ENA is clear, so a
// driver can keep later VGPR numbers stable; ENA alone says what hardware loads.
// Compute stage (COMPUTE_PGM_RSRC2.TIDIG_COMP_CNT): v0 = thread id X, v1 = Y when the count is
// at least 1, v2 = Z when it is at least 2. GFX10 loads each component into its own VGPR.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include <cstdint>
#include <vector>

namespace prosper::gpu {

enum class LaunchVgprState : uint8_t {
    // ADDR reserves the slot but ENA is clear: hardware loads nothing into it.
    Reserved,
    // Hardware loads the word, but this backend has no value for it. An ALU read is refused.
    Unsupplied,
    // Hardware loads the word and the stage shell binds the same value.
    Supplied,
    // The shell binds only part of the hardware word (ANCILLARY layer bits, #5160).
    // Preserve that existing binding while reporting its incomplete coverage honestly.
    Partial,
};

struct LaunchVgpr {
    uint32_t vgpr = 0;
    uint32_t field = 0;   // PS input field 0..15, or compute thread-id component 0..2
    uint32_t component = 0;   // word within a multi-word PS field
    LaunchVgprState state = LaunchVgprState::Reserved;
};

inline constexpr uint32_t kPixelLaunchFields = 16;
// VGPR width of each PS input field, in field order (ISA 70648; matches Mesa's ac_shader_args).
inline constexpr uint8_t kPixelLaunchFieldWidths[kPixelLaunchFields] = {2, 2, 2, 3, 2, 2, 2, 1,
                                                                        1, 1, 1, 1, 1, 1, 1, 1};

// `interpolation_geometry` is FragmentInterpolationLayout::requires_geometry: only the generated
// geometry stage publishes barycentric planes (fields 0..6). Without it, native VINTRP lowering
// reads the attribute directly and never needs the barycentric words, so they are Unsupplied.
std::vector<LaunchVgpr> pixel_launch_vgprs(const PixelSystemInputMapping& inputs,
                                           bool interpolation_geometry);
std::vector<LaunchVgpr> compute_launch_vgprs(uint32_t tidig_comp_cnt);

const char* pixel_launch_field_name(uint32_t field);
const char* launch_vgpr_state_name(LaunchVgprState state);

}   // namespace prosper::gpu
