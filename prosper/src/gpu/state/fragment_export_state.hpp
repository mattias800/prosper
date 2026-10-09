// fragment_export_state.hpp -- derive a draw's fragment colour-export compile input (#4703) from
// its resolved pipeline state: SPI_SHADER_COL_FORMAT, and the numeric class of each MRT's attachment.
//
// The class comes from the GUEST colour-target format (vk_color_format's answer). The backend keeps
// every integer guest format as an integer attachment of the same signedness
// (tests/fixtures/backend_color_formats.h, pinned by test_integer_color_export), so the module's
// uvec4/ivec4 output and the attachment agree without this layer knowing the backend's tables.
//
// Formats travel as raw VkFormat values here: this header is included by gpu_execute.hpp, and
// naming prosper::gpu::VkFormat in it would shadow ::VkFormat in every frontend file that uses the
// namespace.
#pragma once
#include "gpu/recompiler/fragment_export_formats.hpp"
#include "gpu/state/render_state.hpp"

#include <cstdint>

namespace prosper::gpu {

// The output class of a colour target whose format is the VkFormat value `vk_format`.
FragmentOutputClass color_format_output_class(uint32_t vk_format);

FragmentExportFormats fragment_export_formats(const ResolvedPipelineState& pipeline);

}   // namespace prosper::gpu
