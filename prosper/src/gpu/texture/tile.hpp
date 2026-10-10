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
#include <array>
#include <cstdint>
#include <vector>

namespace prosper::gpu {

enum class DataFormat : uint32_t;   // gpu/resources/shader_resources.hpp

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

// The existing 2D mode-27 / eight-byte-texel equation, shared with GPU uploads.
// Each byte-offset bit is parity(x & low16) XOR parity(y & high16). Coordinates
// are whole-face coordinates, including the high bits used by pipe XOR. Blocks
// are 128x64 texels / 64 KiB; this is not the volume or mip-tail equation.
std::array<uint32_t, 16> tile27_rgba16f_equation();

// Modes 24/27, one- or two-byte texels; no 8/16-bit storage capability needed.
// One aligned word groups four byte texels or two halfword texels. Only admit
// equations whose low address bits preserve that horizontal group without RMW stores.
bool tile64_packed_equation(uint32_t mode, uint32_t bytes_per_element,
                            std::array<uint32_t, 16>& equation,
                            uint32_t& block_width, uint32_t& block_height);

// Exact 2D 64 KiB equation used by tile_surface, for word-addressable texels.
// Rejects other layouts; callers must separately exclude volume/mip-tail views.
bool tile64_word_equation(uint32_t tile_mode, uint32_t bytes_per_texel,
                          std::array<uint32_t, 16>& equation,
                          uint32_t& block_width, uint32_t& block_height);

// 3D word tiling: each equation packs x/y/z masks in successive bytes, and each address bit is
// the parity of the masked coordinates. Standard 3D (SW_4KB_S/SW_64KB_S) blocks are 3D; the
// 16-pipe SW_64KB_R_X volume layout is a stack of 2D block rows (block depth 1) whose z terms XOR
// into the in-block offset, exactly as tile_volume lays it out. Only 4/8/16-byte texels have
// independently writable 32-bit output words.
bool tile_volume_word_equation(uint32_t tile_mode, uint32_t bytes_per_texel,
                               std::array<uint32_t, 16>& equation,
                               uint32_t& block_width, uint32_t& block_height,
                               uint32_t& block_depth, uint32_t& block_bits);

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
// An UPPER BOUND on the bytes a 2D (thin) swizzled surface occupies, for overlap tests that must be
// conservative rather than for any layout (#4457). It needs only the swizzle mode's block-size class
// from the GFX10 SW_MODE encoding -- 256 B (1-3), 4 KiB (4-7, 20-23) or 64 KiB (8-11, 16-19, 24-27);
// LINEAR, the VAR modes and anything else return 0 -- because a surface is a whole number of blocks
// whatever the in-block pattern. A block holds 2^n elements of bytes_per_texel * sample_count bytes,
// split near-square between X and Y; rather than depend on which split AddrLib picks for a given
// (mode, element size, samples), this takes the worst whole-block count over every power-of-two
// split within four bits of square. So it is never below the real size for any of those splits, but
// may exceed it by edge blocks. Power-of-two element sizes only; zero means no bound.
size_t thin_surface_bytes_upper_bound(uint32_t width, uint32_t height, uint32_t tile_mode,
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

// One level of a THICK-3D (SW_64KB_S) mip chain's shared tail block. GFX10 packs every
// level of a small 3D pyramid into the allocation's first 64 KiB macroblock at the
// element coordinates below (AddrLib ComputeSurfaceInfoMacroTiled's thick branch:
// tailMaxDim is the block with width halved, at most 10 tail levels, per-level origin
// from the alternating-bit mipOffset times the 256-byte-block geometry). Proven against
// live Black Flag bytes: a 32^3 R8 six-level pyramid detiled through these coordinates
// reproduces every level from the level above at correlation 1.0000 down the whole
// chain (ps5debug-NG carve, base 0x4066a90000: L0 (32,0), L1 (0,16), L2 (16,0),
// L3 (8,8), L4 (0,12), L5 (0,8), means 86.0-86.2). Anything outside that proof --
// Sw4KbS, thin modes, a chain whose level zero is not itself in the tail -- stays
// unsupported: the byte pattern of a wrong tail origin is silent wrong texels.
struct TiledVolumeTailLayout {
    uint32_t tail_x = 0, tail_y = 0;
    uint32_t tail_block_bytes = 0;
    size_t byte_offset = 0;
    bool supported = false;
};
TiledVolumeTailLayout tiled_volume_tail_layout(uint32_t width, uint32_t height, uint32_t depth,
                                               uint32_t bytes_per_texel, uint32_t tile_mode,
                                               uint32_t max_mip, uint32_t mip_level);

// Detile one tail-packed level through the coordinates above. `src` names the shared
// block base; bytes outside this level are never read. Mirrors detile_surface_level's
// contract for the volume case.
bool detile_volume_tail_level(uint8_t* dst, const uint8_t* src, size_t src_bytes, uint32_t width,
                              uint32_t height, uint32_t depth, uint32_t tile_mode,
                              uint32_t bytes_per_texel, uint32_t tail_x, uint32_t tail_y);

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
// upload format. The path covers one- to four-component color surfaces and the embedded
// 0000/0001/1110/1111 codes; register clears, single-color codes, uncompressed (0xff), and actual
// compressed blocks return false. Formats without an alpha channel receive the sampled-format
// default alpha of one, and absent colour components the default zero (one component -> (c,0,0,1),
// two -> (c,c,0,1)); on those the codes 0x40/0x80, whose colour and alpha differ, stay refused. One and two components were added when Black Flag's fast-cleared R16F pool
// surfaces were sampled as their stale base bytes instead of the clear colour (#4131). `alpha_is_on_msb` selects the raw component
// that receives the clear alpha on four-component formats before the T# destination swizzle is applied.
// Whether a surface may be handed to gfx10_dcc_fast_clear_rgba8 at all: three or more components,
// or a narrow surface with an RGBA8 decode buffer whose plane is DCC, or is unclassified but cannot be
// HTILE because its format is never a depth view. A plane known to be HTILE is always refused.
// Whether no depth/stencil view ever uses this sampled format, so its metadata plane cannot be
// HTILE. This list is the depth-safety boundary of the narrow fast-clear admission.
bool gfx10_dcc_format_never_depth(DataFormat format);
bool gfx10_dcc_fast_clear_admits(uint32_t num_components, bool decoded_rgba8, bool metadata_is_dcc,
                                 bool metadata_is_htile, bool format_cannot_be_depth);
bool gfx10_dcc_fast_clear_rgba8(uint8_t* dst, size_t texel_count,
                                const uint8_t* metadata, size_t metadata_bytes,
                                uint32_t num_components, bool alpha_is_on_msb,
                                uint8_t* clear_code = nullptr);

} // namespace prosper::gpu
