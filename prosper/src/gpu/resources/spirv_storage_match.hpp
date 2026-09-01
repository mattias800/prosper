#pragma once
// #3204: does a compute binding's guest format match the SPIR-V storage-image format the recompiled
// module declares?
//
// Extracted from `execute_item`, where it was an unnamed `||` chain. The question it answers is
// narrow and load-bearing: when the module's declared storage format and the guest's declared data
// format agree NATIVELY, the backend can bind the guest image directly; when they do not, the
// binding falls to a repacking path. Getting it wrong in the permissive direction binds an image
// the shader will read with the wrong element type.
//
// The pairs are enumerated rather than derived because there is no general rule -- SPIR-V storage
// formats and Gen5 data formats are two independent vocabularies, and only these combinations have
// been confirmed to round-trip. A pair that is absent is not "unsupported"; it is "not yet
// established", and the difference matters when adding one.
#include "gpu/resources/shader_resources.hpp"

namespace prosper::gpu {

// The module-side half of the question, lifted out of the reflected descriptor so this stays a pure
// comparison of two vocabularies rather than a reach into frontend types.
struct SpirvStorageDeclaration {
    bool numeric_class_is_uint = false;   // the module declares a uint image
    bool numeric_class_is_float = false;  // ...or a float image
    bool atomic_access = false;           // atomics force the linear SSBO path, never a native bind
    uint32_t storage_image_format = 0;    // the SPIR-V Image Format operand
};

// The SPIR-V Image Format operand values are passed IN by the caller rather than restated here.
// Duplicating them would create a second source of truth for constants this header has no way
// to verify, and a wrong one would silently reclassify a binding as native.
// A float storage image binds natively on the module's word alone: the float paths agree on element
// width by construction, so there is no format pair to enumerate.
inline bool spirv_native_float_storage(const SpirvStorageDeclaration& decl) {
    return decl.numeric_class_is_float;
}

// A uint storage image must additionally match the guest's format AND component count, because the
// uint vocabulary distinguishes widths the float one does not.
inline bool spirv_native_uint_storage(const SpirvStorageDeclaration& decl,
                                      DataFormat guest_format, uint32_t components,
                                      uint32_t fmt_r32ui, uint32_t fmt_r16ui,
                                      uint32_t fmt_r8ui, uint32_t fmt_rgba8ui) {
    if (!decl.numeric_class_is_uint) return false;
    // Atomics are served by a linear SSBO view, never a native storage image, so a native match
    // here would bind the wrong thing however well the formats agree.
    if (decl.atomic_access) return false;
    if (guest_format == DataFormat::Uint32 && components == 1)
        return decl.storage_image_format == fmt_r32ui;
    if (guest_format == DataFormat::Uint8 && components == 1)
        return decl.storage_image_format == fmt_r8ui;
    if (guest_format == DataFormat::Uint8 && components == 4)
        return decl.storage_image_format == fmt_rgba8ui;
    if (guest_format == DataFormat::Uint16 && components == 1)
        return decl.storage_image_format == fmt_r16ui;
    return false;
}

}  // namespace prosper::gpu
