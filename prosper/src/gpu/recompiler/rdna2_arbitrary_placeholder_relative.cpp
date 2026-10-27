// An unloaded alternative cannot authorize a lost value in a runtime-selected VGPR bank.
#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/rdna2_relative_index_values.hpp"
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
    if (!arbitrary_placeholder_retry_active() || b.is_fragment_packet()) return false;
    if (refuse_arbitrary_placeholder_operand(b, rs, in, {OperandKind::Special, 124}, &ok))
        return true;
    const auto m0 = rs.sreg.find(124);
    if (m0 == rs.sreg.end()) return reject_index(b, in, ok);
    // This existing integer query bounds values only. Every selected source independently
    // retains its actual value/provenance check; the union of other alternatives supplies none.
    const auto offsets = relative_index_values(b, rs, m0->second, rs.exec);
    if (!offsets || offsets->empty()) return reject_index(b, in, ok);
    for (uint32_t offset : *offsets) {
        const uint64_t reg = static_cast<uint64_t>(in.src[0].value) + offset;
        if (reg > 255) return reject_index(b, in, ok);
        if (refuse_arbitrary_placeholder_operand(b, rs, in,
                                                 {OperandKind::VGPR, static_cast<int>(reg)}, &ok))
            return true;
    }
    return false;
}
}   // namespace prosper::gpu
