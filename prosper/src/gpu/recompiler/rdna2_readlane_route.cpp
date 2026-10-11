#include "gpu/recompiler/rdna2_readlane_route.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
#include <set>

namespace prosper::gpu {
bool has_ordinary_cfg_readlane(const std::vector<Rdna2Inst>& ins) {
    std::set<int> cfg_writelane_spill_arrays;
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361)
            cfg_writelane_spill_arrays.insert(in.dst.value);
    }
    return std::any_of(ins.begin(), ins.end(), [&](const Rdna2Inst& in) {
        return !in.is_end && in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 &&
               !cfg_writelane_spill_arrays.contains(in.src[0].value);
    });
}
bool can_retry_readlane_barrier_phases(const std::vector<Rdna2Inst>& instructions,
                                       const RegState& initial) {
    const auto phased = analyze_barrier_phased_compute(instructions);
    if (!phased.found || phased.guarded || !initial.sreg_entry_m0.empty()) return false;
    // CONFIDENCE: HIGH: the phase dispatcher has a common Workgroup continuation/liveness test,
    // but #3314 does not preserve every opaque entry-M0 token at a region boundary. Keep the new
    // route conservative, including implicit relative-register effects; previously admitted routes
    // are unchanged. Mixed WRITELANE spill service is a separate contract, not this retry's scope.
    for (const auto& in : instructions) {
        if (in.is_end) break;
        if (rdna2_escapes_decoded_effects(in) ||
            (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) ||
            (in.fmt == Rdna2Format::VOP1 &&
             (in.opcode == kVop1OpcodeMovreldB32 || in.opcode == 0x43)))
            return false;
        if (in.dst.value == 124 && scalar_implicit_destination_read_width(in)) return false;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const auto& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) continue;
            const uint32_t width = scalar_alu_source_words(in, source);
            if (operand.value == 124 || (!width && operand.value >= 106 && operand.value <= 124) ||
                (operand.value >= 0 && operand.value < 124 &&
                 (width == UINT32_MAX || width > static_cast<uint32_t>(124 - operand.value))))
                return false;
        }
    }
    return true;
}
}   // namespace prosper::gpu
