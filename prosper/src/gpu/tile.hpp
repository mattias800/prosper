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
#include <cstddef>
#include <cstdint>
#include <vector>

namespace prosper::gpu {

// AGC/GFX10 T# tile_mode values we recognize. 0 = linear (no swizzle). 1 = SW_256B_S (small
// standard-swizzled textures), 5 = SW_4KB_S (the RGBA render target). 9 = SW_64KB_S (standard 64KB,
// DOLL's material textures), 24 = SW_64KB_Z_X
// (depth/texture 64KB with pipe XOR, Astro Bot compute surfaces), and 27 = SW_64KB_R_X
// (render-target 64KB with pipe XOR, DOLL's RT/post composites) — #288/#825.
// Others fall through to a linear copy for now.
enum class TileMode : uint32_t {
    Linear = 0,
    Sw256BS = 1,
    Sw4KbS = 5,
    Sw64KbS = 9,
    Sw64KbZX = 24,
    Sw64KbRX = 27,
};

// GFX10 sampled images in linear mode use a 256-byte-aligned row pitch. Buffer/host-data uploads
// remain tightly packed; callers apply this only to guest-backed sampled Texture resources.
size_t linear_sampled_row_pitch(uint32_t width, uint32_t bytes_per_texel);
size_t linear_sampled_surface_bytes(uint32_t width, uint32_t height, uint32_t bytes_per_texel);

// True if `tile_mode` denotes a swizzled layout that detile_surface will de-swizzle.
bool tile_mode_is_tiled(uint32_t tile_mode);

// Byte size of the TILED surface for `tile_mode` — for swizzled modes the dimensions are padded up
// to whole 4KB micro-tiles, whose texel size depends on bytes_per_texel (a tile is a FIXED 4096
// bytes: 32x32 at 4 B, 64x32 at 2 B, 64x64 at 1 B — #119), so the tiled buffer is larger than
// w*h*bpt. The caller must read at least this many bytes of tiled source. Linear -> w*h*bpt.
size_t tiled_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode, uint32_t pitch = 0,
                           uint32_t bytes_per_texel = 4);

// De-swizzle a surface of `bytes_per_texel`-byte texels from tiled `src` into linear `dst` (each
// width*height*bpt bytes). `tile_mode` selects the swizzle; Linear/unknown modes do a straight
// copy. `pitch` is the padded row pitch in texels (0 -> use `width`).
void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch = 0, uint32_t bytes_per_texel = 4);

// Convenience wrapper: detile `src` (tiled) into a returned linear vector. Returns a copy of `src` for
// linear/unknown modes.
std::vector<uint8_t> detile_surface(const std::vector<uint8_t>& src, uint32_t width, uint32_t height,
                                    uint32_t tile_mode, uint32_t pitch = 0, uint32_t bytes_per_texel = 4);

// Inverse of detile_surface (linear -> tiled). Provided for testing the round-trip; the runtime only
// detiles. Same parameters.
void tile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                  uint32_t tile_mode, uint32_t pitch = 0, uint32_t bytes_per_texel = 4);

// General SW_4KB_S de-swizzle for `bpe`-byte ELEMENTS. The 4KB micro-tile holds 4096/bpe elements;
// its dimensions derive from bpe (wide-before-tall: 16 B -> 16x16, 8 B -> 32x16, ...) — previously
// a caller-supplied SQUARE tile_side, which could not represent the non-square 8 B geometry (#119).
// Block-compressed surfaces use this with element = one compressed block (BC3 = 16 bytes). `dst`
// holds ew*eh*bpe linear bytes; `src_bytes` bounds the tiled read (short/OOB elements detile to
// zero). tile_mode!=SW_4KB_S -> straight copy.
void detile_elements(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                     uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode);

// Byte size of the TILED element surface (element grid padded up to whole 4KB tiles). The caller
// must read at least this many bytes of tiled source before detiling.
size_t tiled_elements_bytes(uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode);

// One thin-2D mip's location in a GFX10 allocation. SW_MODE 0 follows AddrLib's reverse,
// 256-byte-pitch-aligned linear chain. Tiled layouts place the shared mip-tail block first, then the
// remaining levels from smallest to largest. Tail levels share the first 4/64 KiB block;
// byte_offset is their independently addressable in-block origin, and tail_x/y are the equivalent
// element coordinates from AddrLib. `ew`/`eh` are the allocation's level-zero element dimensions
// (texels for plain formats, compressed blocks for BCn).
struct TiledMipLevelLayout {
    size_t byte_offset = 0;
    uint32_t tail_x = 0, tail_y = 0;
    uint32_t tail_block_bytes = 0;
    bool in_tail = false;
    bool supported = false;
};
TiledMipLevelLayout tiled_mip_level_layout(uint32_t ew, uint32_t eh, uint32_t bpe,
                                           uint32_t tile_mode, uint32_t max_mip,
                                           uint32_t mip_level);

// Compatibility helper for callers interested only in non-tail placement. Tail levels intentionally
// retain the historical zero result; use tiled_mip_level_layout when a packed-tail view is required.
size_t tiled_mip_level_offset(uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode,
                              uint32_t max_mip, uint32_t mip_level);

// Access one level packed inside a shared mip-tail block. `src`/`dst` name the allocation base and
// `tail_x/y` are TiledMipLevelLayout's element coordinates. Unlike tile_surface, the write helper
// preserves every byte outside this level so sibling mips in the same block are not destroyed.
void detile_surface_level(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                          uint32_t width, uint32_t height, uint32_t tile_mode,
                          uint32_t bytes_per_texel, uint32_t tail_x, uint32_t tail_y);
void tile_surface_level(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                        uint32_t width, uint32_t height, uint32_t tile_mode,
                        uint32_t bytes_per_texel, uint32_t tail_x, uint32_t tail_y);
void detile_elements_level(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                           uint32_t ew, uint32_t eh, uint32_t bpe, uint32_t tile_mode,
                           uint32_t tail_x, uint32_t tail_y);

// GFX10 volume layout. PS5's observed 3D surfaces use SW_64KB_R_X (mode 27), which AddrLib defines
// as a thin/view-as-2D volume: each Z slice owns a padded grid of 2D 64KB blocks, but Z still
// participates in the pipe-XOR bits inside every block. These helpers return false/zero for tiled
// modes whose 3D pattern is not implemented. `src_bytes` bounds detile reads.
bool tile_mode_supports_volume(uint32_t tile_mode);
size_t tiled_volume_bytes(uint32_t width, uint32_t height, uint32_t depth,
                          uint32_t tile_mode, uint32_t bytes_per_texel);
bool detile_volume(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                   uint32_t width, uint32_t height, uint32_t depth,
                   uint32_t tile_mode, uint32_t bytes_per_texel);
bool tile_volume(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                 uint32_t width, uint32_t height, uint32_t depth,
                 uint32_t tile_mode, uint32_t bytes_per_texel);

// GFX10 DCC control-surface size for the single-sample, base-level SW_64KB_R_X images currently
// emitted by PS5 Unreal. The PS5/default 16-pipe layout uses a 4 KiB metadata block. Each byte in
// that block describes one 256-byte compressed block, so the covered texel extent follows AddrLib's
// thin-resource equation. Returns zero for unsupported swizzles/element sizes or overflow.
size_t gfx10_dcc_metadata_bytes(uint32_t width, uint32_t height, uint32_t depth,
                                uint32_t tile_mode, uint32_t bytes_per_texel,
                                bool pipe_aligned);

// Materialize a uniform, self-contained GFX8-GFX10 DCC fast-clear code into the renderer's RGBA8
// upload format. The validated path is deliberately limited to three-component color surfaces,
// four-component color/alpha surfaces, and the embedded 0000/0001/1110/1111 codes; register clears,
// single-color codes, uncompressed (0xff), and actual compressed blocks return false. Three-component
// formats receive the sampled-format default alpha of one. `alpha_is_on_msb` selects the raw component
// that receives the clear alpha on four-component formats before the T# destination swizzle is applied.
bool gfx10_dcc_fast_clear_rgba8(uint8_t* dst, size_t texel_count,
                                const uint8_t* metadata, size_t metadata_bytes,
                                uint32_t num_components, bool alpha_is_on_msb,
                                uint8_t* clear_code = nullptr);

} // namespace prosper::gpu
