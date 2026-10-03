// storage_dst_sel.hpp -- how a storage image's T# DST_SEL routes image_load and image_store.
//
// A storage view is bound with the identity component mapping (Vulkan allows nothing else), so the
// recompiler applies the descriptor's selector itself (SQ_SEL: 0 = constant 0, 1 = constant 1,
// 4..7 = stored X..W; 2 and 3 are reserved).
//   * LOAD, forward: returned channel k takes the stored component its selector names, or the
//     constant. This is what a sampled view's component mapping does for textures (#4274).
//   * STORE, inverse: stored channel c receives the VDATA component whose selector names c;
//     constant selectors supply nothing (#4120, #3549).
// Evidence for the store direction: Hollow Knight: Silksong uploads dynamic-font glyphs with
// image_load -> image_store between two Alpha8 descriptors, DST_SEL (0,0,0,X). The load puts the
// glyph in .w and the PS5 menu text renders, so the store writes .w to the single stored channel.
#pragma once

#include "gpu/resources/shader_resources.hpp"

#include <cstdint>

namespace prosper::gpu {

inline constexpr int kDstSelNoSource = -1;

// Store routing. On success `source[c]` is the VDATA component stored into channel c, or
// kDstSelNoSource. Refused (false): a reserved selector, two selectors naming one channel, or a
// channel the format stores (c < components) that no selector names.
bool storage_store_sources(const uint32_t (&swizzle)[4], uint32_t components, int (&source)[4]);

// Load routing. On success, for each returned channel k, `channel[k]` is the stored component it
// takes, or kDstSelNoSource with `constant[k]` holding the raw bits to return instead (the format's
// own one for SQ_SEL_1: integer 1 for an integer format, float 1.0 otherwise). Refused (false): a
// reserved selector.
bool storage_load_selects(const uint32_t (&swizzle)[4], DataFormat format, int (&channel)[4],
                          uint32_t (&constant)[4]);

bool data_format_is_integer(DataFormat format);

}  // namespace prosper::gpu
