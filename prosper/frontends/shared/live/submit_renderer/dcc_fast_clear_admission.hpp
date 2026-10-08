// dcc_fast_clear_admission.hpp -- may this sampled surface be materialized from its DCC clear code?
//
// The component/format/kind rule lives in gfx10_dcc_fast_clear_admits (src/gpu/texture/tile.cpp,
// unit-tested there). This wrapper only supplies the metadata kind, asked of the guest's own
// descriptor, and only for narrow surfaces so the common four-component path pays nothing.
#pragma once

#include <cstdint>

#include "gpu/execute/gpu_execute.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"

namespace prosper::frontend {

// VK_FORMAT_R8G8B8A8_UNORM; Vulkan enum values are fixed by the specification. Kept as a number so
// this header does not pull in the Vulkan headers.
inline constexpr uint32_t kDecodedRgba8Format = 37u;

// `decoded_format` is the VkFormat the sampled upload decodes to.
inline bool dcc_fast_clear_admitted(const prosper::gpu::ShaderResource& r,
                                    uint32_t decoded_format) {
    const bool decoded_rgba8 = decoded_format == kDecodedRgba8Format;
    if (r.num_components >= 3u) return true;   // the kind is asked only of narrow surfaces
    const auto kind =
        prosper::gpu::classify_compression_metadata_kind(prosper::gpu::MetadataKindRequest{
            r.metadata_addr, r.gpu_addr, r.format, r.num_components, r.img_dim});
    const bool never_depth = prosper::gpu::gfx10_dcc_format_never_depth(r.format);
    return prosper::gpu::gfx10_dcc_fast_clear_admits(
        r.num_components, decoded_rgba8, kind == prosper::gpu::CompressionMetadataKind::Dcc,
        kind == prosper::gpu::CompressionMetadataKind::Htile, never_depth);
}

}   // namespace prosper::frontend
