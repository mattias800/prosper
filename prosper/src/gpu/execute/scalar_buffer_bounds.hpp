// Shared scalar-buffer bounds used by folding and resource materialization.
#pragma once

#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>

namespace prosper::gpu {
inline uint64_t scalar_buffer_dword_count(const DecodedBufferDescriptor& descriptor) {
    // A scalar address is a BYTE offset into the descriptor's span:
    //
    //     m_size = (m_stride == 0 ? 1 : m_stride) * m_num_records      [bytes]
    //
    // which is exactly decode_buffer_descriptor::size_bytes. A dword access needs its whole four
    // bytes inside that span, so the dword bound truncates. CONFIDENCE: HIGH.
    //
    // Read the RDNA2 guide's own §7.2.1 with care: it prints `m_size = (m_stride == 0) ? 1 :
    // m_num_records`, dropping the stride factor, even though its buffer-descriptor table (§8.1.8)
    // states NUM_RECORDS is in units of stride when strided and bytes otherwise. The RDNA3 guide
    // §8.1.2 and the RDNA4 guide §8.1.2 both print the corrected product, and RDNA3 §8.4.1 defines
    // out-of-range as `offset >= (stride == 0 ? 1 : stride) * num_records` — a byte-address bound.
    // The RDNA2 line is a documentation omission; do not re-derive the bound from it alone (#2528).
    //
    // #2528: reading NUM_RECORDS as a DWORD count (and STRIDE==0 as one dword) makes every ordinary
    // constant buffer a single dword long. Two independent live falsifications, one per branch:
    //   * STRIDE==0 — The Messenger, whose first level is checked against PS5 hardware, rendered
    //     111,118 consecutive byte-identical fully black frames; Dead Cells collapsed the same way.
    //   * STRIDE!=0 — GTA V's 0x413ced900 indexes a STRIDE=120/NUM_RECORDS=5 descriptor table with
    //     `s_mul_i32 vcc_hi, s9, 0x78` + `s_buffer_load_dwordx4 …, vcc_hi offset:0x8` for s9 in
    //     0..4. A NUM_RECORDS-dword bound admits only 20 of those 600 bytes, so four of the five
    //     records load an all-zero V#; 269 routed dispatches resolved every record to its real
    //     vertex buffer before that bound existed.
    //
    // An EMPTY V# keeps eece6f84's separately reviewed stride-zero rule, which is where the
    // one-dword reading came from and the only domain it was ever exercised in: NUM_RECORDS==0 with
    // STRIDE==0 still admits its first dword, so only a strictly positive offset is proven OOB.
    if (descriptor.num_records == 0u && descriptor.size_bytes == 0u)
        return descriptor.stride == 0u ? 1ull : 0ull;
    return static_cast<uint64_t>(descriptor.size_bytes) / sizeof(uint32_t);
}

inline bool sbuffer_access_is_fully_oob(const Rdna2Inst& in,
                                        const DecodedBufferDescriptor& descriptor,
                                        bool exact_byte_offset_known = false,
                                        uint32_t exact_byte_offset = 0u) {
    if (in.fmt != Rdna2Format::SMEM || in.opcode < 0x8u || in.opcode > 0xCu) return false;

    const uint64_t scalar_dwords = scalar_buffer_dword_count(descriptor);
    if (exact_byte_offset_known) {
        // The fold already preserved the 32-bit SOFFSET+OFFSET result. Scalar memory converts that
        // effective byte offset to its first dword index.
        const uint32_t first_dword = exact_byte_offset >> 2u;
        return static_cast<uint64_t>(first_dword) >= scalar_dwords;
    }

    // An unknown SOFFSET can only be specialized by the established empty-descriptor proof. SOFFSET
    // is unsigned, so a non-negative immediate names its minimum possible dword. Do not generalize
    // this minimum argument to a non-empty descriptor: the exact-offset and emitted dynamic paths
    // below handle those without assuming a runtime value.
    if (static_cast<int32_t>(in.literal) < 0 || descriptor.num_records != 0u ||
        descriptor.size_bytes != 0u)
        return false;
    return static_cast<uint64_t>(in.literal >> 2u) >= scalar_dwords;
}

}   // namespace prosper::gpu
