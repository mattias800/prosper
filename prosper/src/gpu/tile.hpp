// tile.hpp — GPU surface de-swizzle (detiling). PS5 render targets / textures are stored in a tiled
// (swizzled) memory layout; the host upload path needs a LINEAR surface, so a sampled tiled texture must
// be de-swizzled first. This is a required, standard emulation step (the host Vulkan driver then re-tiles
// for ITS hardware) — not a rendering shortcut.
//
// Currently implements the mode observed for The Messenger's 1920x1080 RGBA render target: T# tile_mode=5
// == GFX10 SW_4KB_S. Layout: 32x32-texel micro-tiles (4KB at 32bpp) laid out row-major, and within a tile
// the texels follow a Morton/Z order with the Y bit in the LOW position of each pair (y0,x0,y1,x1,...).
// Empirically derived (raw-tiled-bytes dump + offline swizzle sweep) and pixel-verified against the game.
#pragma once
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// AGC/GFX10 T# tile_mode values we recognize. 0 = linear (no swizzle). 5 = SW_4KB_S (the RGBA render
// target). Others fall through to a linear copy for now (add as the target needs them).
enum class TileMode : uint32_t { Linear = 0, Sw4KbS = 5 };

// True if `tile_mode` denotes a swizzled layout that detile_surface will de-swizzle.
bool tile_mode_is_tiled(uint32_t tile_mode);

// Byte size of the TILED surface for `tile_mode` — for swizzled modes the height is padded up to a whole
// number of 32-texel tile rows (e.g. 1080 -> 1088), so the tiled buffer is larger than width*height*4.
// The caller must read at least this many bytes of tiled source before detiling. Linear -> width*height*4.
size_t tiled_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode, uint32_t pitch = 0);

// De-swizzle a 32-bpp (RGBA8) surface from tiled `src` into linear `dst` (each width*height*4 bytes).
// `tile_mode` selects the swizzle; Linear/unknown modes do a straight copy. `pitch` is the padded row
// pitch in texels (0 -> use `width`).
void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch = 0);

// Convenience wrapper: detile `src` (tiled) into a returned linear vector. Returns a copy of `src` for
// linear/unknown modes.
std::vector<uint8_t> detile_surface(const std::vector<uint8_t>& src, uint32_t width, uint32_t height,
                                    uint32_t tile_mode, uint32_t pitch = 0);

// Inverse of detile_surface (linear -> tiled). Provided for testing the round-trip; the runtime only
// detiles. Same parameters.
void tile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                  uint32_t tile_mode, uint32_t pitch = 0);

} // namespace prosper::gpu
