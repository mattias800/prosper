#include "gpu/execute/empty_scalar_buffer.hpp"

#include "gpu/agc/agc_shader_layout.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/execute/scalar_buffer_bounds.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <optional>

namespace prosper::gpu {
std::optional<ShaderResource> empty_scalar_buffer_resource(const SrtUse& use, const uint32_t* code,
                                                           size_t dwords) {
    if (!code || use.kind != 1 || !use.zero_record_raw || use.instruction_format != UINT32_MAX ||
        use.required_size != 0u || use.use_pc >= dwords)
        return std::nullopt;
    const auto descriptor = decode_buffer_descriptor(use.v4.data());
    if (descriptor.num_records != 0u || descriptor.size_bytes != 0u) return std::nullopt;
    auto instruction = rdna2_decode_one(code + use.use_pc, dwords - use.use_pc);
    instruction.pc = use.use_pc;
    if (!instruction.len_dwords || instruction.len_dwords > dwords - use.use_pc ||
        !sbuffer_access_is_fully_oob(instruction, descriptor, use.scalar_oob_offset_known,
                                     use.scalar_oob_byte_offset))
        return std::nullopt;
    // CONFIDENCE: HIGH for this existing fold/materializer contract, shared with compute.
    // A no-backing per-PC marker reaches the emitter's constant-zero path without an SRT tag.
    ShaderResource resource;
    resource.cls = ResourceClass::ConstantBuffer;
    resource.format = DataFormat::Unknown;
    resource.num_components = 0;
    resource.fetch_pc = use.use_pc;
    for (uint32_t component = 0; component < 4u; ++component)
        resource.swizzle[component] = (use.v4[3] >> (component * 3u)) & 0x7u;
    return resource;
}
} // namespace prosper::gpu
