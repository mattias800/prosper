// Architectural operand shapes for the instruction-effects table (AMD ISA 70648).
// These facts have no production consumer yet (ADR 0038 step 8a).
#pragma once

#include <cstdint>

namespace prosper::gpu {

enum class Rdna2ScalarDestination : uint8_t {
    None,
    Dst,   // decoded dst, even when the decoder gives a lane read a VGPR operand kind
    Sdst,   // the separate VOP3B scalar destination
    Vcc,   // implicit VCC (VOP2 carry-out)
    Exec,   // implicit EXEC, and no VCC/SDST write (gfx10 CMPX)
};
struct Rdna2ScalarResult {
    Rdna2ScalarDestination destination = Rdna2ScalarDestination::None;
    uint8_t dwords = 0;   // Wave64 width; a lane mask writes one word in Wave32
    bool lane_mask = false;
};

enum class Rdna2MemoryAccess : uint8_t {
    None,
    Load,
    Store,
    Atomic,
    CacheControl,
    Export,
    Interpolate,
    CrossLane,
    Gws,
};
// The address fields and their control-dependent widths (ISA 8.1-8.3, 9, 10, 11).
// Fixed uses vsrc_dwords below. All other layouts override that address portion only.
enum class Rdna2MemoryAddress : uint8_t {
    Fixed,
    Buffer,   // src0: IDXEN + OFFEN VGPRs; src1: four SGPRs; src2: SOFFSET when a register
    Image,   // src0: offset, bias, compare, gradients, DIM body, tail; NSA names each word
    ImageMip,   // Image without sampler, with a trailing mip coordinate
    ImageResinfo,   // src0: one mip VGPR, independent of DIM
    ImageMsaa,   // image_msaa_load: coordinates plus packed fragment IDs (ISA 8.2)
    ImageBvh32,   // src0: 11 VGPRs, or 8 with A16; src1: four SGPRs (ISA 8.2.10)
    ImageBvh64,   // src0: 12 VGPRs, or 9 with A16; src1: four SGPRs
    Flat,   // SEG=0: VADDR pair. GLOBAL: pair if SADDR=null, else VADDR word + SADDR pair.
    // SCRATCH: VADDR word if SADDR=null, otherwise SADDR word. SEG=3 is reserved.
    FlatAddtid,   // GLOBAL only: SADDR pair, no VADDR (ISA 9.3)
    Export,   // src[k] when EN[k]; COMPR uses src0 for EN[1:0], src1 for EN[3:2]
};
// The count of data VGPRs, independently of address VGPRs. Fixed uses the explicit counts.
enum class Rdna2MemoryData : uint8_t {
    Fixed,
    ImageDmask,   // popcount(DMASK); with D16 ceil(popcount(DMASK)/2)
    ImagePacked,   // popcount(DMASK) native components, without format conversion; no D16
    ImageGather,   // four components, or two packed VGPRs with D16
    ImageAtomic,   // reads popcount(DMASK); returns that many words only with GLC
    ImageAtomicCompare,   // reads popcount(DMASK); returns half that many words only with GLC
};
enum class Rdna2MemoryHalf : uint8_t { Full, Low, High };

struct Rdna2MemoryEffects {
    bool described = false;   // false is unknown, never a claim of no effect
    Rdna2MemoryAccess access = Rdna2MemoryAccess::None;
    Rdna2MemoryAddress address = Rdna2MemoryAddress::Fixed;
    Rdna2MemoryData data = Rdna2MemoryData::Fixed;
    // Reads through decoded src fields, only when the operand names that register file.
    // DS: ADDR/DATA0/DATA1, VINTRP: VSRC. Variable address layouts override fixed counts.
    uint8_t vsrc_dwords[4] = {};
    uint8_t ssrc_dwords[3] = {};
    // Data read/written through VDST/VDATA. Buffer/image VDATA is decoded dst. DS has a
    // separate VDST. FLAT data comes from raw words[1][15:8], not decoded dst/src, including
    // atomics; its result is raw words[1][31:24]. No decoder change is implied by these facts.
    uint8_t data_read_dwords = 0;
    uint8_t data_write_dwords = 0;
    uint8_t implicit_reads = 0;   // Rdna2ImplicitRegister bits, separate from legacy VALU fields
    // Memory element bits, not expanded register width. Zero: resource/instruction FORMAT
    // determines the size; image atomics select 32/64 through the resource and legal DMASK.
    uint8_t element_bits = 0;
    bool gds_reads_m0 = false;   // DS with GDS=1 reads M0 base/size (ISA 3.7, 10.5)
    bool write_if_glc = false;
    bool lds_redirect = false;   // LDS redirects the data transfer away from VGPRs; M0 is read
    bool status = false;   // TFE appends a status word; image LWE shares this word
    Rdna2MemoryHalf result_half = Rdna2MemoryHalf::Full;   // preserve the other half of VDST
    bool reads_old_vdst = false;   // VINTRP P2 accumulator, independently of partial writes
    bool image_sampler = false;   // src2 is four SGPRs; src1 is 8, or 4 when R128
    bool image_offset = false;   // a full dword even with A16
    bool image_bias = false;
    bool image_compare = false;   // full dword PCF reference even with A16
    bool image_derivatives = false;   // 2 * spatial dimensions, packed by A16 or G16
    bool image_g16 = false;
    bool image_tail = false;   // LOD/clamp after coordinates
    // Which FLAT segments define this opcode: bit 0 FLAT, bit 1 SCRATCH, bit 2 GLOBAL.
    uint8_t flat_segments = 0;
};

}   // namespace prosper::gpu
