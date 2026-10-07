// fragment_export_formats.hpp -- the per-MRT facts a fragment module's colour exports depend on
// beyond its own machine code: how SPI_SHADER_COL_FORMAT says a compressed export is packed, and
// whether the colour attachment the export lands in is an integer one (#4703).
//
// Both change the generated SPIR-V, so both are part of the fragment compile key. They are kept in
// a NORMALIZED form so the key only diverges when the code would: an FP16 compressed export and a
// 32-bit uncompressed export into a float target compile exactly as they always have, and the
// all-zero value is that historical behaviour (what every caller that predates #4703 passes).
#pragma once
#include <cstdint>

namespace prosper::gpu {

// SPI_SHADER_COL_FORMAT, one 4-bit field per MRT (AMD register enum SPI_SHADER_*; the same codes
// Kena's lighting-channel writer programs, 0x7 for an R16_UINT target -- #4703).
// CONFIDENCE: HIGH for the enumeration, which is the published register field.
enum class ColorExportFormat : uint32_t {
    Zero = 0,        // SPI_SHADER_ZERO
    R32 = 1,         // SPI_SHADER_32_R
    GR32 = 2,        // SPI_SHADER_32_GR
    AR32 = 3,        // SPI_SHADER_32_AR
    Fp16Abgr = 4,    // SPI_SHADER_FP16_ABGR   -- compressed: two f16 per dword
    Unorm16Abgr = 5, // SPI_SHADER_UNORM16_ABGR -- compressed: two unorm16 per dword
    Snorm16Abgr = 6, // SPI_SHADER_SNORM16_ABGR -- compressed: two snorm16 per dword
    Uint16Abgr = 7,  // SPI_SHADER_UINT16_ABGR  -- compressed: two u16 per dword
    Sint16Abgr = 8,  // SPI_SHADER_SINT16_ABGR  -- compressed: two s16 per dword
    Abgr32 = 9,      // SPI_SHADER_32_ABGR
};

// The numeric class of a fragment colour output. Vulkan leaves the stored value undefined unless
// the output's type matches the attachment's numeric format, so an integer attachment needs a
// uvec4 / ivec4 output rather than the vec4 every module used to declare.
enum class FragmentOutputClass : uint8_t { Float = 0, Uint = 1, Sint = 2 };

struct FragmentExportFormats {
    // One nibble per MRT, holding ONLY the compressed formats whose unpack differs from FP16
    // (5..8). Every other code is stored as 0, which is the FP16 / raw-bits path.
    uint32_t compressed_formats = 0;
    uint8_t uint_outputs = 0;   // bit n: MRTn's attachment is an unsigned-integer format
    uint8_t sint_outputs = 0;   // bit n: MRTn's attachment is a signed-integer format

    constexpr ColorExportFormat compressed_format(uint32_t mrt) const {
        if (mrt >= 8u) return ColorExportFormat::Fp16Abgr;
        const uint32_t code = (compressed_formats >> (mrt * 4u)) & 0xFu;
        return code ? static_cast<ColorExportFormat>(code) : ColorExportFormat::Fp16Abgr;
    }
    constexpr FragmentOutputClass output_class(uint32_t mrt) const {
        if (mrt >= 8u) return FragmentOutputClass::Float;
        if (uint_outputs & (1u << mrt)) return FragmentOutputClass::Uint;
        if (sint_outputs & (1u << mrt)) return FragmentOutputClass::Sint;
        return FragmentOutputClass::Float;
    }
    // A module can only be keyed on a value it could have been built from.
    constexpr bool canonical() const {
        if (uint_outputs & sint_outputs) return false;
        for (uint32_t mrt = 0; mrt < 8u; ++mrt) {
            const uint32_t code = (compressed_formats >> (mrt * 4u)) & 0xFu;
            if (code != 0u && (code < 5u || code > 8u)) return false;
        }
        return true;
    }
    bool operator==(const FragmentExportFormats&) const = default;
};

// Build the normalized value from the raw register and each MRT's output class.
constexpr FragmentExportFormats
make_fragment_export_formats(uint32_t spi_shader_col_format,
                             const FragmentOutputClass (&classes)[8]) {
    FragmentExportFormats out;
    for (uint32_t mrt = 0; mrt < 8u; ++mrt) {
        const uint32_t code = (spi_shader_col_format >> (mrt * 4u)) & 0xFu;
        if (code >= 5u && code <= 8u) out.compressed_formats |= code << (mrt * 4u);
        if (classes[mrt] == FragmentOutputClass::Uint) out.uint_outputs |= uint8_t(1u << mrt);
        if (classes[mrt] == FragmentOutputClass::Sint) out.sint_outputs |= uint8_t(1u << mrt);
    }
    return out;
}

} // namespace prosper::gpu
