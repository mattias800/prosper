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

// libSceVideoOut's own two-value tiling enum, as passed to sceVideoOutSetBufferAttribute(2) and
// recorded with each registered display buffer. It is NOT a GFX10 swizzle index: it only says
// whether the scanout surface is stored in the hardware's render-target layout or row-major.
inline constexpr uint32_t kVideoOutTilingModeTile   = 0;
inline constexpr uint32_t kVideoOutTilingModeLinear = 1;

// Map a registered scanout's VideoOut tiling mode to the GFX10 swizzle its bytes are actually in, so
// the flipped buffer can be read as an image. TILE on Gen5 means the render-target 64 KiB swizzle
// with the pipe XOR (SW_64KB_R_X) — the layout CB writes and the one a compute dispatch writing the
// display buffer produces.
//
// Evidence (#1968): Sonic Frontiers (PPSA03831) registers tiling_mode=0 (TILE) for a 3840x2160
// 32-bpp scanout, and the raw bytes of its flipped buffer de-swizzle under SW_64KB_R_X into exact,
// recognizable frames (the SEGA logo at flip 60 and an intro shot at flip 180); read linearly the
// same bytes are horizontal-band noise, and no other supported mode resolves them. Before this the
// display buffer was memcpy'd out untouched even though the registry had recorded its tiling mode.
// CONFIDENCE: MED — one title, two frames, 32 bpp only.
//
// **The TILE value is load-bearing in one direction, deliberately.** `kVideoOutTilingModeTile` is 0,
// and `DisplayConfig::SetConfig::tiling_mode` also defaults to 0, so "the guest asked for TILE" and
// "no attribute was parsed, or it was parsed at the wrong offset" are the SAME input here. Every
// other value fails closed to the historical straight copy; this one value fails open into a
// de-swizzle.
//
// The mitigation is not in this function, and it is NOT uniform across callers — be precise about
// which one you are reading. Only the live renderer's last-resort branch additionally proves the
// guest wrote the buffer before publishing from it (`guest_scanout_present.hpp`), so there a
// mis-parsed attribute on an untouched buffer cannot reach the screen. `present_snapshot`'s
// RawScanout path and `present_readback`'s fallback consume the same de-swizzled pixels with **no**
// authorship gate — they are the pre-renderer boot path, where the alternative is showing nothing at
// all. If a title ever reads as band noise where it used to read as an image, suspect this default
// first.
//
// Returns TileMode::Linear (0) for LINEAR and for anything unrecognized or not 4 bytes/texel, which
// reproduces the historical straight copy rather than guessing at a geometry nothing has verified.
uint32_t videoout_scanout_tile_mode(uint32_t videoout_tiling_mode, uint32_t bytes_per_texel);

// Byte size of the TILED surface for `tile_mode` — for swizzled modes the dimensions are padded up
// to whole 4KB micro-tiles, whose texel size depends on bytes_per_texel (a tile is a FIXED 4096
// bytes: 32x32 at 4 B, 64x32 at 2 B, 64x64 at 1 B — #119), so the tiled buffer is larger than
// w*h*bpt. The caller must read at least this many bytes of tiled source. Linear -> w*h*bpt.
size_t tiled_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode, uint32_t pitch = 0,
                           uint32_t bytes_per_texel = 4);

// De-swizzle a surface of `bytes_per_texel`-byte texels from tiled `src` into linear `dst` (each
// width*height*bpt bytes). `tile_mode` selects the swizzle; Linear/unknown modes do a straight
// copy. `pitch` is the padded row pitch in texels (0 -> use `width`).
// PROSPER_TILECENSUS attribution: which SUBSYSTEM asked for this tiling work. The profiler cannot
// say -- at -O3 the callers inline away and DWARF collapses through the guest JIT -- and the geometry
// census alone shows a 4K FP16 surface detiled over and over without saying who wants it (#3149).
// Scoped by RAII at the two subsystem entry points; unset reads as "?".
// `who` is stored by pointer and outlives every scope, so it MUST be a string
// literal (or otherwise static-lifetime).  The census also compares and hashes it by
// pointer identity, so two equal-but-distinct spellings would split one call site into
// two rows.  Passing a temporary's c_str() here dangles.
struct TileCensusScope {
    explicit TileCensusScope(const char* who);
    ~TileCensusScope();
    // A user-declared destructor does NOT suppress the copy constructor, and a copy would restore
    // `prev` twice. Deleting it matters more now that this type lives in a widely included header.
    TileCensusScope(const TileCensusScope&) = delete;
    TileCensusScope& operator=(const TileCensusScope&) = delete;
    const char* prev;
};

void detile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                    uint32_t tile_mode, uint32_t pitch = 0, uint32_t bytes_per_texel = 4);

// Do `detile_surface` and `detile_elements` write EVERY byte of their width*height*bytes_per_element
// destination for this shape? True means a caller may hand them an uninitialised buffer; false means
// the caller must zero the destination first, because one copier below stops at the TILED source
// size and would leave the remainder holding whatever was there. Both entry points route to the same
// three copiers with the same guard, so they share one answer. It does NOT extend to
// `detile_volume` or the mip-tail `*_level` variants, whose destinations are not covered here.
//
// This exists so the answer lives beside the implementation it describes rather than in a caller's
// comment. The frontend materializer hands `detile_surface` a POOLED buffer that still holds the
// previous surface (`frontends/shared/live/decode_scratch.hpp`), so "the destination is zero unless
// something writes it" stopped being true there, and a wrong answer here is a silent wrong picture
// rather than a crash. `tests/gpu/texture/test_tile.cpp` fills the destination with poison and
// asserts none survives, for every shape this returns true for.
bool detile_writes_whole_destination(uint32_t tile_mode, uint32_t bytes_per_texel);

// Convenience wrapper: detile `src` (tiled) into a returned linear vector. Returns a copy of `src` for
// linear/unknown modes.
std::vector<uint8_t> detile_surface(const std::vector<uint8_t>& src, uint32_t width, uint32_t height,
                                    uint32_t tile_mode, uint32_t pitch = 0, uint32_t bytes_per_texel = 4);

// Inverse of detile_surface (linear -> tiled). Provided for testing the round-trip; the runtime only
// detiles. Same parameters.
void tile_surface(uint8_t* dst, const uint8_t* src, uint32_t width, uint32_t height,
                  uint32_t tile_mode, uint32_t pitch = 0, uint32_t bytes_per_texel = 4,
                  bool allow_avx2 = true);

// Exact GFX10 2D-MSAA materialization. The guest swizzle includes sample-coordinate bits in every
// 64 KiB block; treating it as an ordinary surface both under-reads the allocation and scrambles
// every texel. The linear representation is Vulkan-2D-array ready: complete sample planes are
// contiguous (`sample * width * height + y * width + x`). This first implementation is deliberately
// fail-closed outside the observed/published SW_64KB_Z_X 4xaa contract; zero/false means unsupported.
size_t tiled_msaa_surface_bytes(uint32_t width, uint32_t height, uint32_t tile_mode,
                                uint32_t bytes_per_texel, uint32_t sample_count);
bool detile_msaa_surface(uint8_t* dst, const uint8_t* src, size_t src_bytes,
                         uint32_t width, uint32_t height, uint32_t tile_mode,
                         uint32_t bytes_per_texel, uint32_t sample_count);
bool tile_msaa_surface(uint8_t* dst, size_t dst_bytes, const uint8_t* src,
                       uint32_t width, uint32_t height, uint32_t tile_mode,
                       uint32_t bytes_per_texel, uint32_t sample_count);

// Exact GFX10 16-pipe HTILE contract for the observed 2D_MSAA SW_64KB_Z_X surface. AMD AddrLib
// computes one 32 KiB metadata block per 1024x512 pixels; PAL documents the only two uniform HTILE
// initialization values that disable Z compression and make ordinary base texels authoritative.
// The helpers stay fail-closed outside 4xaa, pipe-aligned mode 24. The classifier requires the
// complete expected plane so a short or mixed metadata read cannot authorize base decoding.
size_t gfx10_htile_msaa_metadata_bytes(uint32_t width, uint32_t height,
                                       uint32_t tile_mode, uint32_t sample_count,
                                       bool pipe_aligned);
bool gfx10_htile_metadata_is_decompressed(const uint8_t* metadata, size_t metadata_bytes,
                                           size_t expected_bytes,
                                           uint32_t* uniform_value = nullptr);

// A complete, uniform zero HTILE plane is PAL's exact depth-only fast-clear encoding for +0.0:
// ZMin=ZMax=0 and ZMask=0. It does NOT authorize reading the stale base allocation. The source
// classifier and materializer keep that case distinct from the two ordinary-base initial values.
enum class Gfx10HtileMsaaSource : uint8_t {
    Unsupported,
    UncompressedBase,
    DepthZeroFastClear,
};
Gfx10HtileMsaaSource gfx10_htile_msaa_source(
    const uint8_t* metadata, size_t metadata_bytes,
    uint32_t width, uint32_t height, uint32_t tile_mode,
    uint32_t bytes_per_texel, uint32_t sample_count, bool pipe_aligned);
bool materialize_gfx10_htile_msaa_surface(
    uint8_t* dst, size_t dst_bytes,
    const uint8_t* tiled_base, size_t tiled_base_bytes,
    const uint8_t* metadata, size_t metadata_bytes,
    uint32_t width, uint32_t height, uint32_t tile_mode,
    uint32_t bytes_per_texel, uint32_t sample_count, bool pipe_aligned,
    Gfx10HtileMsaaSource* source = nullptr);

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

// Byte stride between array slices that each contain a complete thin-2D mip chain. GFX10 stores
// every slice's tail-first/reverse chain as one independently aligned unit; array/cube views need
// this stride to select the same mip from consecutive layers without treating those levels as a
// tightly packed image array. Returns zero when the mip layout is not modeled.
size_t tiled_mip_chain_bytes(uint32_t ew, uint32_t eh, uint32_t bpe,
                             uint32_t tile_mode, uint32_t max_mip);

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

// GFX10 volume layouts. SW_64KB_S (mode 9) uses AddrLib's true 3D S3 macroblocks, whose XYZ extent
// depends on bytes-per-element; SW_64KB_R_X (mode 27) is a thin/view-as-2D volume where each Z slice
// owns a padded grid of 2D blocks and Z participates in the pipe-XOR bits. These helpers return
// false/zero for tiled modes whose 3D pattern is not implemented. `src_bytes` bounds detile reads.
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
