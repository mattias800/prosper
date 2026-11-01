// An unloaded alternative cannot authorize a lost value in a runtime-selected VGPR bank.
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_relative_vgpr_source.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <cstdint>
#include <memory>
#include <string>

namespace prosper::gpu {
namespace {
bool reject_index(SpirvCompute& b, const Rdna2Inst& in, bool& ok) {
    const PlaceholderRoots roots{
        {PlaceholderOrigin::UnmodelledTransfer, {PlaceholderFile::Scalar, 124}, in.pc}};
    b.diagnostic.placeholder_origins =
        std::make_shared<const std::string>(format_placeholder_origins(roots));
    b.stage_reject_pc = in.pc;
    b.stage_reject_reason = "hard-placeholder-read";
    log_recompile_diagnostic(b.diagnostic, "hard-placeholder-read", "alu",
                             "pc=%u relative source has an unbounded or out-of-range index", in.pc);
    ok = false;
    return true;
}
} // namespace

bool refuse_arbitrary_placeholder_relative_source(SpirvCompute& b, const RegState& rs,
                                                  const Rdna2Inst& in, bool& ok) {
    if (!placeholder_recompile_guards_active() || b.is_fragment_packet()) return false;
    if (refuse_arbitrary_placeholder_operand(b, rs, in, {OperandKind::Special, 124}, &ok))
        return true;
    const auto m0 = rs.sreg.find(124);
    if (m0 == rs.sreg.end()) return reject_index(b, in, ok);
    // Allocation bounds selection only; each possible source keeps its independent roots.
    const auto sources = relative_vgpr_sources(b, rs, in);
    if (!sources || sources->empty()) return reject_index(b, in, ok);
    for (int reg : *sources) {
        if (refuse_arbitrary_placeholder_operand(b, rs, in, {OperandKind::VGPR, reg}, &ok))
            return true;
    }
    return false;
}
}   // namespace prosper::gpu
