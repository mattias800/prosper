// A new arbitrary admission cannot make a lane-local READFIRST into a scalar broadcast.
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_scalar_exec_scopes.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <memory>
#include <string>

namespace prosper::gpu {
bool refuse_arbitrary_scalar_publication(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    // Only the legacy lane-local lowering calls this. Dispatcher/owned broadcast services
    // bypass it. Preserve strict successful modules and the existing agreeing-source proof.
    if (!arbitrary_placeholder_retry_active() || scalar_scope_first_lane_agrees(b, rs, in) ||
        scalar_scope_local_index_waterfall(b, in))
        return false;
    const auto* program = rs.placeholder_origin_program ? rs.placeholder_origin_program.get()
                                                        : b.placeholder_origin_program.get();
    PlaceholderRoots roots;
    if (program && in.src[0].kind == OperandKind::VGPR)
        roots =
            placeholder_roots_before(*program, in.pc, {{PlaceholderFile::Vector, in.src[0].value}});
    roots.insert({PlaceholderOrigin::UnmodelledScalarPublication,
                  {PlaceholderFile::Scalar, in.dst.value},
                  in.pc});
    b.diagnostic.placeholder_origins =
        std::make_shared<const std::string>(format_placeholder_origins(roots));
    b.stage_reject_pc = in.pc;
    b.stage_reject_reason = "unmodelled-scalar-publication";
    log_recompile_diagnostic(b.diagnostic, "unmodelled-scalar-publication", "alu",
                             "pc=%u legacy READFIRST requires a scalar broadcast", in.pc);
    return true;
}
} // namespace prosper::gpu
