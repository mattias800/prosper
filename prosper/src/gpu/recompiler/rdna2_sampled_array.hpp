// Sampled array declarations must match the view owned by each backend. The graphics uploader
// and compute's reflection-selected views deliberately have different contracts.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/bc_decode.hpp"

namespace prosper::gpu {
struct SampledArrayContract {
    bool resource_arrayed = false;
    bool graphics_float_array = false;
    bool declared_arrayed = false;
    bool float_array_gather = false;
};

inline SampledArrayContract sampled_array_contract(const Rdna2Inst& in,
                                                   const ShaderResource& resource, bool compute) {
    SampledArrayContract result;
    // BC graphics arrays follow the uploaded resource shape, even for a layer-zero 2D read.
    // A Float32 graphics DIM=2D read instead selects the renderer's retained base slice.
    result.resource_arrayed =
        guest_texture_is_uploaded_array(resource.img_dim, resource.depth, resource.format);
    result.graphics_float_array = !compute && result.resource_arrayed &&
                                  resource.format == DataFormat::Float32 &&
                                  resource.cls == ResourceClass::Texture;
    result.declared_arrayed =
        result.resource_arrayed && !(result.graphics_float_array && in.mimg_dim == 1u);
    const bool gather = in.opcode == 0x47u || in.opcode == 0x40u || in.opcode == 0x57u;
    // CONFIDENCE: HIGH — DIM5 gather consumes [u,v,layer] (after an optional spatial offset).
    // The retained compute invocation supplies a dynamic float layer on a Float16 T#. Compute
    // owns full reflection-selected array views, including one-layer arrays; graphics upload
    // eligibility cannot remove that operand. Query the same selected view's layer count.
    const bool compute_float_array =
        compute && in.mimg_dim == 5u && resource.img_dim == 5u && resource.depth != 0u &&
        resource.cls == ResourceClass::Texture &&
        (resource.num_components == 1u || resource.num_components == 2u ||
         resource.num_components == 4u) &&
        (resource.format == DataFormat::Float16 || resource.format == DataFormat::Float32);
    result.declared_arrayed |= compute_float_array && (gather || in.opcode == 0x0eu);
    result.float_array_gather =
        in.mimg_dim == 5u && gather && (result.graphics_float_array || compute_float_array);
    return result;
}
} // namespace prosper::gpu
