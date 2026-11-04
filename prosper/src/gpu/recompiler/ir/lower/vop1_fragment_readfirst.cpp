#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstdint>
#include "gpu/recompiler/rdna2_native_fragment_collectives.hpp"
#include "gpu/recompiler/ir/lower/vop1_fragment_readfirst.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"

namespace prosper::gpu::ir {
uint32_t native_fragment_readfirst(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                   uint32_t source) {
    // Preserve every successful original module. This service supplies the missing real
    // broadcast during the guarded retry; it grants no ordinary input or memory value.
    if (!arbitrary_placeholder_retry_active() || !b.is_fragment || b.is_compute || b.is_vertex ||
        b.is_fragment_packet() || b.ngg_one_lane || b.ngg_workgroup_shell ||
        b.guest_stage != GuestShaderStage::Fragment ||
        b.physical_domain != PhysicalExecutionDomain::Invocation ||
        (b.wave_size != 32 && b.wave_size != 64) || in.fmt != Rdna2Format::VOP1 || in.opcode != 2 ||
        in.src[0].kind != OperandKind::VGPR || in.dst.value < 0 || in.dst.value > 107 ||
        in.has_modifier || in.has_sdwa || in.has_dpp || in.src_abs[0] || in.src_neg[0] ||
        in.clamp || in.omod)
        return 0;
    // The existing closed local-index waterfall already has an exact lane-local lowering.
    // Leave its admitted retry modules unchanged as well as strict successful modules.
    if (scalar_scope_local_index_waterfall(b, in)) return 0;
    if (!native_fragment_collective_retry_active()) {
        request_native_fragment_collective_retry();
        return 0;
    }
    return fragment_readfirst_value(b, source, rs.exec);
}
} // namespace prosper::gpu::ir
