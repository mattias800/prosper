#include "fixtures/render_resource_contracts.h"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/recompiler/spirv_builder.hpp"
#include "fixtures/gpu_detile_upload.h"
#include "fixtures/backend_color_formats.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stddef.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

namespace prosper::test {
// clang-format off: preserve the byte-identical moved bodies.

bool backend_volume_view_valid(uint32_t depth, uint32_t first, uint32_t count) {
    if (!depth) return !first && !count;
    return count && first < depth && count <= depth - first;
}
bool backend_color_volume_view_valid(const BackendColorTarget& target) {
    return backend_volume_view_valid(target.volume_depth, target.volume_first_slice,
                                     target.volume_slice_count) &&
           std::all_of(target.volume_slots.begin(), target.volume_slots.end(), [](const auto& v) {
               return backend_volume_view_valid(v.depth, v.first_slice, v.slice_count);
           });
}

std::array<uint32_t, 4> backend_sampled_component_swizzle(
    const FrameResource& resource) {
    std::array<uint32_t, 4> result{
        resource.swizzle[0], resource.swizzle[1],
        resource.swizzle[2], resource.swizzle[3]};
    // CB_COLOR ALT stores COLOR_8_8_8_8 as BGRA, while prosper deliberately materializes that
    // attachment as canonical RGBA8. SQ_IMG_RSRC DST_SEL is expressed against the guest's format
    // components, so translate R<->B selectors into the canonical host image's component space.
    // Constants, G/A, ordinary textures, and targets whose storage order was already RGBA remain
    // unchanged. This is composition, not suppression: semantic permutations still survive.
    // The integer BGRA / A2R10G10B10 orders are canonicalized the same way (#4703).
    const VkFormat guest = resource.render_target_guest_format;
    if (((guest == VK_FORMAT_B8G8R8A8_UNORM || guest == VK_FORMAT_B8G8R8A8_SRGB) &&
         backend_color_format(guest) == VK_FORMAT_R8G8B8A8_UNORM) ||
        guest == VK_FORMAT_B8G8R8A8_UINT || guest == VK_FORMAT_B8G8R8A8_SINT ||
        guest == VK_FORMAT_A2R10G10B10_UINT_PACK32) {
        for (uint32_t& selector : result) {
            if (selector == 4u) selector = 6u;
            else if (selector == 6u) selector = 4u;
        }
    }
    return result;
}

bool backend_storage_image_numeric_contract_valid(const FrameResource& resource) {
    if (!resource.is_storage_image) return true;
    return resource.storage_image_contract_valid &&
           resource.storage_image_numeric_class !=
               prosper::gpu::SpirvImageNumericClass::Unknown &&
           resource.storage_image_numeric_class ==
               backend_image_numeric_class(resource.texture_format);
}

// A guest 2D_ARRAY is the same shape as the MSAA plane array with a different provenance: N
// ordinary color layers laid out one after another, carried through the SAME `sample_count`
// channel, and owed the same proof that every layer is readable before any Vulkan object or memcpy
// exists. Everything downstream is already layer-generic -- the staging buffer sizes itself
// `tw*th*td*sample_count*bpp`, the image takes `arrayLayers = sample_count`, and the view selects
// VK_IMAGE_VIEW_TYPE_2D_ARRAY above 1 -- so once a frontend publishes a layer count, this predicate
// is what stands between it and the upload (#325). Mip generation is already excluded for
// `sample_count > 1`, which is what stops a generated chain bleeding across layer boundaries.
bool backend_texture_array_span_valid(const FrameResource& resource) {
    // A borrowed GPU image has no CPU span to overrun: its owner created it with exactly
    // tw x th x sample_count layers (the retained depth-array snapshot,
    // tests/fixtures/retained_depth_array_gpu.h) and keeps it alive through the lease.
    const bool borrowed_array = !resource.tex_rgba && resource.borrowed_compute_image &&
        resource.borrowed_compute_image_lease && !resource.borrowed_compute_vertical_stack_layers;
    if (resource.img_dim != 5u || resource.td != 1u || resource.is_storage_image ||
        (!resource.tex_rgba && !borrowed_array) || !resource.tw || !resource.th ||
        resource.sample_count < 2u || resource.sample_count > kBackendMaxArrayLayers)
        return false;
    const uint32_t bpp = backend_color_bytes_per_pixel(resource.texture_format);
    if (!bpp) return false;
    if (borrowed_array) return true;
    const size_t row_bytes = static_cast<size_t>(resource.tw) * bpp;
    if (row_bytes / bpp != resource.tw) return false;
    if (resource.th > SIZE_MAX / row_bytes) return false;
    const size_t layer_bytes = row_bytes * resource.th;
    if (resource.sample_count > SIZE_MAX / layer_bytes) return false;
    return resource.tex_byte_size >= layer_bytes * resource.sample_count;
}

bool backend_texture_plane_span_valid(const FrameResource& resource) {
    if (resource.gpu_detile) {
        const auto& upload = *resource.gpu_detile;
        const bool raw = upload.program && upload.program->output_kind ==
            prosper::gpu::Float16DetileOutput::RawUvec4;
        return !resource.tex_rgba && resource.is_storage_image == raw &&
               (!raw || (resource.img_dim == 1 && resource.storage_image_contract_valid &&
                         resource.storage_image_numeric_class == prosper::gpu::SpirvImageNumericClass::Uint)) &&
               (!resource.storage_image_writeback ||
                // NOLINTNEXTLINE(concurrency-mt-unsafe): preserve the original per-call texture-share selector read.
                (raw && !std::getenv("PROSPER_NO_BACKEND_TEXTURE_SHARE") &&
                 upload.readback.buffer && upload.readback.memory && upload.readback.mapped)) &&
               !resource.has_uniform_color && !resource.persistent_texture_id &&
               !resource.persistent_render_target_id && !resource.persistent_depth_target_id &&
               !resource.borrowed_compute_image &&
               ((resource.img_dim == 3 && upload.faces == 6) ||
                (resource.img_dim == 1 && upload.faces == 1)) &&
               resource.sample_count == 1 && resource.td == 1 &&
               resource.declared_mip_levels == 1 &&
               resource.tw == upload.width && uint64_t(resource.th) == uint64_t(upload.height) * upload.faces &&
               resource.texture_format == (raw ? VK_FORMAT_R32G32B32A32_UINT : VK_FORMAT_R8G8B8A8_UNORM) &&
               upload.program && upload.program->pipeline && upload.input && upload.output &&
               upload.descriptors;
    }
    if (resource.sample_count == 1u) return true;
    if (resource.img_dim == 5u) return backend_texture_array_span_valid(resource);
    if (resource.sample_count != 4u || resource.img_dim != 6u || resource.td != 1u ||
        resource.declared_mip_levels != 1u || resource.is_storage_image ||
        backend_color_format(resource.texture_format) != VK_FORMAT_R32_SFLOAT ||
        !resource.tex_rgba || !resource.tw || !resource.th)
        return false;
    const size_t row_bytes = static_cast<size_t>(resource.tw) * sizeof(float);
    if (resource.tw && row_bytes / resource.tw != sizeof(float) ||
        resource.th > SIZE_MAX / row_bytes)
        return false;
    const size_t plane_bytes = row_bytes * resource.th;
    if (resource.sample_count > SIZE_MAX / plane_bytes) return false;
    const size_t required = plane_bytes * resource.sample_count;
    return resource.tex_byte_size >= required;
}
// clang-format on

} // namespace prosper::test
