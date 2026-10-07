// fragment_export_state.cpp -- see fragment_export_state.hpp.
#include "gpu/state/fragment_export_state.hpp"

#include "gpu/state/vk_translate.hpp"

namespace prosper::gpu {

FragmentOutputClass color_format_output_class(uint32_t vk_format) {
    switch (static_cast<VkFormat>(vk_format)) {
        case VkFormat::R8_UINT:
        case VkFormat::R8G8_UINT:
        case VkFormat::R8G8B8A8_UINT:
        case VkFormat::B8G8R8A8_UINT:
        case VkFormat::A2R10G10B10_UINT_PACK32:
        case VkFormat::A2B10G10R10_UINT_PACK32:
        case VkFormat::R16_UINT:
        case VkFormat::R16G16_UINT:
        case VkFormat::R16G16B16A16_UINT:
        case VkFormat::R32_UINT:
        case VkFormat::R32G32_UINT:
        case VkFormat::R32G32B32A32_UINT: return FragmentOutputClass::Uint;
        case VkFormat::R8_SINT:
        case VkFormat::R8G8_SINT:
        case VkFormat::R8G8B8A8_SINT:
        case VkFormat::B8G8R8A8_SINT:
        case VkFormat::R16_SINT:
        case VkFormat::R16G16_SINT:
        case VkFormat::R16G16B16A16_SINT:
        case VkFormat::R32_SINT:
        case VkFormat::R32G32_SINT:
        case VkFormat::R32G32B32A32_SINT: return FragmentOutputClass::Sint;
        default: return FragmentOutputClass::Float;
    }
}

FragmentExportFormats fragment_export_formats(const ResolvedPipelineState& pipeline) {
    FragmentOutputClass classes[8]{};
    for (uint32_t mrt = 0; mrt < 8u && mrt < pipeline.color_targets.size(); ++mrt) {
        uint32_t format = pipeline.color_targets[mrt].format;
        // Direct/synthetic states populate only the named MRT0/MRT1 fields.
        if (!format && mrt == 0) format = pipeline.color0_format;
        if (!format && mrt == 1) format = pipeline.color1_format;
        classes[mrt] = color_format_output_class(format);
    }
    return make_fragment_export_formats(pipeline.spi_shader_col_format, classes);
}

} // namespace prosper::gpu
