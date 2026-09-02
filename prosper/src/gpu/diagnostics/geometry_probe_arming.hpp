#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// Can the geometry probe (`PROSPER_GEOM_PROBE=N`) answer at all for this draw?
//
// The probe reads a draw's post-transform clip-space positions back through VK_EXT_transform
// feedback. Transform feedback captures nothing unless the LAST PRE-RASTERIZATION stage declares
// `OpExecutionMode <entry> Xfb` and decorates the captured member -- and prosper's recompiler emits
// that decoration only under its own separate gate (`capture_position` / `capture_geometry_position`
// in rdna2_to_spirv_internal.hpp), which is off for a hand-written fixture shader, for a draw that
// falls back to the synthesised geometry stage without the flag, and for anything the probe did not
// itself cause to be recompiled.
//
// Without this check the probe armed regardless, `vkCmdBeginTransformFeedbackEXT` violated
// VUID-vkCmdBeginTransformFeedbackEXT-None-04128, the counter buffer stayed at zero, and the probe
// printed "transform feedback wrote 0 vertices (draw produced no primitives)". That is not a missing
// answer, it is a WRONG one: the draw did produce primitives, and an investigation that believed the
// line went looking for vanished geometry that was never vanished (#3248). This folder's standing
// rule is to prefer instruments that detect their own invalidity; this is that detector.
//
// Deliberately a structural test on the module the backend is about to hand Vulkan, rather than a
// re-read of the environment variable that was supposed to have caused the decoration. The env var
// says what was ASKED for; the words say what is actually there, and the two diverged.
inline bool spirv_declares_xfb_capture(const uint32_t* words, size_t word_count) {
    constexpr uint32_t kSpirvMagic = 0x07230203u;
    constexpr uint32_t kOpExecutionMode = 16u;
    constexpr uint32_t kOpExecutionModeId = 331u;   // SPIR-V 1.2+ spelling; same operand order
    constexpr uint32_t kExecutionModeXfb = 11u;
    constexpr size_t kHeaderWords = 5;
    if (!words || word_count <= kHeaderWords || words[0] != kSpirvMagic) return false;
    for (size_t i = kHeaderWords; i < word_count;) {
        const uint32_t length = words[i] >> 16;
        const uint32_t opcode = words[i] & 0xFFFFu;
        // A zero length would loop forever, and a length running past the end means the module is
        // malformed. Both answer "no" rather than reading out of bounds: this predicate is asked
        // whether an instrument may trust its own output, so an unreadable module is a refusal.
        if (length == 0 || i + length > word_count) return false;
        if ((opcode == kOpExecutionMode || opcode == kOpExecutionModeId) && length >= 3 &&
            words[i + 2] == kExecutionModeXfb)
            return true;
        i += length;
    }
    return false;
}

inline bool spirv_declares_xfb_capture(const std::vector<uint32_t>& words) {
    return spirv_declares_xfb_capture(words.data(), words.size());
}

} // namespace prosper::gpu
