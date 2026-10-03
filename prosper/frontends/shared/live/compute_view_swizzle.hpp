// T# DST_SEL -> Vulkan component swizzle for live compute's sampled image views. Moved out of
// live_compute.cpp's view construction unchanged (#4291), so the selector table and its #3609
// reserved-value diagnostic live in one named place.
#pragma once

#include <cstdint>
#include <cstdio>
#include <mutex>

#include <vulkan/vulkan.h>

namespace prosper::frontend {

// SQ_SEL: 0=0, 1=1, 4=R, 5=G, 6=B, 7=A -- the same mapping the renderer applies on its sampled
// views. A caller composing a host-layout swap (live_target_host_selector) does so before this.
inline VkComponentSwizzle compute_view_component_swizzle(uint32_t s) {
    switch (s) {
        case 0: return VK_COMPONENT_SWIZZLE_ZERO;
        case 1: return VK_COMPONENT_SWIZZLE_ONE;
        case 4: return VK_COMPONENT_SWIZZLE_R;
        case 5: return VK_COMPONENT_SWIZZLE_G;
        case 6: return VK_COMPONENT_SWIZZLE_B;
        case 7: return VK_COMPONENT_SWIZZLE_A;
        default: break;
    }
    // SQ_SEL 2 and 3 are RESERVED, and anything above 7 cannot come out of a
    // three-bit descriptor field at all. Both used to fall into a silent
    // `default: IDENTITY` -- the selector for whichever position the value sat in --
    // so an undecodable routing became a plausible wrong picture with no diagnostic
    // anywhere. The graphics path was made loud for that reason; this is its twin
    // and was missed (#3609).
    //
    // A reserved selector can no longer reach here FROM A DESCRIPTOR, because
    // `image_descriptor_reject_reason` now refuses such a T# before it becomes a
    // ShaderResource. It stays reachable from a capture deserialized verbatim and
    // from a directly-built resource, so this warns rather than dropping: reproducing
    // the frame that was recorded is replay's job, and the drop belongs upstream.
    static std::once_flag warned;
    std::call_once(warned, [&] {
        fprintf(stderr,
                "[compute] T# DST_SEL %u is reserved or unrepresentable; binding it "
                "as IDENTITY, which is a GUESS at the routing (#3609)\n", s);
    });
    return VK_COMPONENT_SWIZZLE_IDENTITY;
}

} // namespace prosper::frontend
