// fragment_color_export.cpp -- SpirvCompute's fragment colour-output declaration and the decode
// of compressed colour exports (#4703). Kept apart from shader_io.cpp, whose bodies are a verbatim
// generated move, so these can carry the project's formatting.

#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

namespace prosper::gpu {

void SpirvCompute::declare_color_outputs(uint32_t t_ptr_out_float) {
    uint32_t t_ptr_out_u = 0, t_ptr_out_i = 0;
    for (uint32_t mrt = 0; mrt < v_color.size(); ++mrt) {
        if (!v_color[mrt]) continue;
        uint32_t pointer = t_ptr_out_float;
        if (color_output_class[mrt] == FragmentOutputClass::Uint) {
            if (!t_ptr_out_u) {
                const uint32_t vec = t_v4u();
                t_ptr_out_u = id();
                put(types, Op_TypePointer, {t_ptr_out_u, SC_Output, vec});
            }
            pointer = t_ptr_out_u;
        } else if (color_output_class[mrt] == FragmentOutputClass::Sint) {
            if (!t_ptr_out_i) {
                const uint32_t vec = t_v4i();
                t_ptr_out_i = id();
                put(types, Op_TypePointer, {t_ptr_out_i, SC_Output, vec});
            }
            pointer = t_ptr_out_i;
        }
        put(types, Op_Variable, {pointer, v_color[mrt], SC_Output});
    }
}

// The packing of every compressed format is the half-dword split the guest compiler emits for it
// (v_cvt_pkrtz_f16_f32, v_cvt_pknorm_{u,i}16_f32, v_cvt_pk_{u,i}16_{u,i}32): channel 0 in bits
// [15:0], channel 1 in [31:16]. An integer format is a 16-bit integer, NOT an f16 -- decoding Kena's
// UINT16 lighting-channel value 1 as a half gave 5.96e-8, which an RGBA8 target stored as 0 (#4703).
//
// A format whose class disagrees with the attachment's (an f16 export into an integer target, or a
// u16 into a float one) is converted by value. The guest compiler never pairs them, so nothing
// observed pins the hardware answer. CONFIDENCE: HIGH for the matching pairs, LOW for mismatches.
uint32_t SpirvCompute::compressed_export_channel(uint32_t dword, uint32_t which,
                                                 ColorExportFormat format,
                                                 FragmentOutputClass output_class) {
    const uint32_t offset = which ? 16u : 0u;
    switch (format) {
        case ColorExportFormat::Uint16Abgr: {
            const uint32_t value = bfe_u(dword, uconst(offset), uconst(16));
            return output_class == FragmentOutputClass::Float ? cvt_u2f(value) : value;
        }
        case ColorExportFormat::Sint16Abgr: {
            const uint32_t value = bfe_s(dword, uconst(offset), uconst(16));
            return output_class == FragmentOutputClass::Float ? cvt_i2f(value) : value;
        }
        default: break;
    }
    const uint32_t value =
        format == ColorExportFormat::Unorm16Abgr   ? unpack_norm(dword, offset, 16, false, 65535.0f)
        : format == ColorExportFormat::Snorm16Abgr ? unpack_norm(dword, offset, 16, true, 32767.0f)
                                                   : unpack_half(dword, which);
    if (output_class == FragmentOutputClass::Uint) return cvt_f2u(value);
    if (output_class == FragmentOutputClass::Sint) return cvt_f2i(value);
    return value;
}

}   // namespace prosper::gpu
