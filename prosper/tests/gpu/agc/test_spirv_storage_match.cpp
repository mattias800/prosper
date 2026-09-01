// #3204: does a compute binding's guest format match the SPIR-V storage-image format the module
// declares? Extracted from execute_item, where it was an unnamed `||` chain.
//
// The permissive direction is the dangerous one: a false "native" binds the guest image directly and
// the shader reads it with the wrong element type. Every arm below therefore has a rejection partner.
#include "gpu/resources/spirv_storage_match.hpp"
#include <cstdio>

using prosper::gpu::DataFormat;
using prosper::gpu::SpirvStorageDeclaration;
using prosper::gpu::spirv_native_float_storage;
using prosper::gpu::spirv_native_uint_storage;

// Stand-ins for the SPIR-V Image Format operands. The real values live at the call site and are
// passed in; this test only needs them to be DISTINCT, which is the property the predicate relies on.
static constexpr uint32_t R32UI = 101, R16UI = 102, R8UI = 103, RGBA8UI = 104;

static int fails = 0;
#define CHECK(c, m) do { if (!(c)) { printf("  [FAIL] %s\n", m); ++fails; } \
                         else printf("  [ok]   %s\n", m); } while (0)

static SpirvStorageDeclaration uint_decl(uint32_t fmt, bool atomic = false) {
    SpirvStorageDeclaration d{}; d.numeric_class_is_uint = true; d.atomic_access = atomic;
    d.storage_image_format = fmt; return d;
}

int main() {
    printf("== spirv storage match (#3204) ==\n");

    // Float: the module's word is sufficient, no format pair to check.
    SpirvStorageDeclaration f{}; f.numeric_class_is_float = true;
    CHECK(spirv_native_float_storage(f), "a float storage image is native on the module's word");
    SpirvStorageDeclaration u{}; u.numeric_class_is_uint = true;
    CHECK(!spirv_native_float_storage(u), "a uint declaration is not a float storage image");

    // The four established uint pairs.
    CHECK(spirv_native_uint_storage(uint_decl(R32UI), DataFormat::Uint32, 1, R32UI, R16UI, R8UI, RGBA8UI),
          "Uint32 x1 <-> R32ui");
    CHECK(spirv_native_uint_storage(uint_decl(R8UI), DataFormat::Uint8, 1, R32UI, R16UI, R8UI, RGBA8UI),
          "Uint8 x1 <-> R8ui");
    CHECK(spirv_native_uint_storage(uint_decl(RGBA8UI), DataFormat::Uint8, 4, R32UI, R16UI, R8UI, RGBA8UI),
          "Uint8 x4 <-> Rgba8ui");
    CHECK(spirv_native_uint_storage(uint_decl(R16UI), DataFormat::Uint16, 1, R32UI, R16UI, R8UI, RGBA8UI),
          "Uint16 x1 <-> R16ui");

    // The module's declared format must match the guest's -- a right-width guess is not enough.
    CHECK(!spirv_native_uint_storage(uint_decl(R16UI), DataFormat::Uint32, 1, R32UI, R16UI, R8UI, RGBA8UI),
          "Uint32 x1 is NOT native against an R16ui declaration");
    CHECK(!spirv_native_uint_storage(uint_decl(R8UI), DataFormat::Uint8, 4, R32UI, R16UI, R8UI, RGBA8UI),
          "component count is part of the pair: Uint8 x4 is not R8ui");

    // Atomics are served by a linear SSBO view, so a native bind would hand back the wrong object
    // however well the formats agree. This arm is why the check is not just a format comparison.
    CHECK(!spirv_native_uint_storage(uint_decl(R32UI, /*atomic=*/true), DataFormat::Uint32, 1,
                                     R32UI, R16UI, R8UI, RGBA8UI),
          "atomic access is never a native storage image, even on a matching pair");

    // A float guest format has no uint pair -- absent means "not established", not "equivalent".
    CHECK(!spirv_native_uint_storage(uint_decl(R32UI), DataFormat::Float32, 1, R32UI, R16UI, R8UI, RGBA8UI),
          "Float32 has no uint pair");
    // ...and a non-uint module never takes this path at all.
    SpirvStorageDeclaration neither{}; neither.storage_image_format = R32UI;
    CHECK(!spirv_native_uint_storage(neither, DataFormat::Uint32, 1, R32UI, R16UI, R8UI, RGBA8UI),
          "a module declaring neither class is not a native uint storage image");

    printf(fails ? "== FAIL: %d ==\n" : "== PASS ==\n", fails);
    return fails ? 1 : 0;
}
