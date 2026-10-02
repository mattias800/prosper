#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"

namespace prosper::gpu {
namespace {
// A bounded first execution slice, not an assertion that every fragment opcode is supported.
// In particular, ordinary compute's LOD0 and fragment's implicit WQM/OpKill shortcuts are NOT
// acceptable substitutions for the absent raster/quad ABI. Inventory before emitting any words.
const char* packet_instruction_gap(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::SOPK &&
        (in.opcode == 0x13 || in.opcode == 0x15)) return "packet-mode-write";
    if (in.fmt == Rdna2Format::SOPP &&
        (in.opcode == 0x24 || in.opcode == 0x25)) return "packet-mode-write";
    if (in.fmt == Rdna2Format::SOP1 &&
        (in.opcode == 0x0a || in.opcode == 0x09)) return "packet-wqm-unimplemented";
    if (in.fmt == Rdna2Format::VINTRP) return "packet-interpolation-unavailable";
    if (in.fmt == Rdna2Format::MIMG) return "packet-image-derivative-or-effect-unimplemented";
    if (in.fmt == Rdna2Format::DS) return "packet-ds-op-unimplemented";
    if (in.fmt == Rdna2Format::MUBUF ||
        in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::FLAT ||
        in.fmt == Rdna2Format::SMEM) return "packet-memory-effect-unimplemented";
    if (in.has_modifier || in.has_sdwa || in.has_dpp || in.clamp || in.omod ||
        std::any_of(std::begin(in.src_abs), std::end(in.src_abs), [](bool x) { return x; }) ||
        std::any_of(std::begin(in.src_neg), std::end(in.src_neg), [](bool x) { return x; }))
        return "packet-modifier-unimplemented";
    switch (in.fmt) {
        case Rdna2Format::SOPP:
            return in.is_end || in.opcode == 0 || sopp_opcode_is_direct_branch(in.opcode)
                ? nullptr : "packet-control-unimplemented";
        case Rdna2Format::SOP1:
            return in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64 ||
                in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64
                ? nullptr : "packet-scalar-op-unimplemented";
        case Rdna2Format::SOP2:
            return in.opcode == kSop2OpcodeAddU32 || in.opcode == kSop2OpcodeCselectB32
                ? nullptr : "packet-scalar-op-unimplemented";
        case Rdna2Format::SOPK:
            return in.opcode == kSopkOpcodeMovkI32 ? nullptr : "packet-scalar-op-unimplemented";
        case Rdna2Format::VOP1:
            return in.opcode == 1 ? nullptr : "packet-valu-op-unimplemented";
        case Rdna2Format::VOPC:
            return (in.opcode >= 0xc0 && in.opcode <= 0xc7) ||
                   (in.opcode >= 0xd0 && in.opcode <= 0xd7)
                ? nullptr : "packet-fp-or-compare-unimplemented";
        case Rdna2Format::VOP3:
            return in.opcode == 0x360 ? nullptr : "packet-valu-op-unimplemented";
        case Rdna2Format::EXP:
            if (in.exp_compr) return "packet-compressed-export-unimplemented";
            if (in.exp_target < kFragmentColorOutputs) return nullptr;
            if (in.exp_target == 8 && in.exp_en && !(in.exp_en & ~5u)) return nullptr;
            if (in.exp_target == 9 && !in.exp_en) return nullptr;
            return "packet-export-target-unimplemented";
        default: return "packet-format-unimplemented";
    }
}
} // namespace

FragmentPacketProgram recompile_fragment_packet(const FragmentInvocationPacket& packet,
                                                RecompileDiagnosticContext diagnostic) {
    const auto reject = [&](const std::string& reason, uint32_t pc = UINT32_MAX) {
        FragmentPacketProgram result;
        result.rejection = reason;
        // Packet failures are always announced, not hidden behind PROSPER_DBG. No code/input/output
        // artifact is returned on any failure, so an unsupported effect cannot partially execute.
        log_recompile_diagnostic(diagnostic, "fragment-packet-reject", "terminal",
                                 "pc=%u reason=%s", pc, reason.c_str());
        std::fprintf(stderr, "[fragment-packet-reject] program=0x%llx pc=%u reason=%s\n",
                     static_cast<unsigned long long>(diagnostic.program_address), pc, reason.c_str());
        return result;
    };
    if (packet.guest_code.empty() || packet.guest_code.size() > 4096 ||
        packet.vgprs.size() > 256 || packet.sgprs.size() > 106)
        return reject("packet-input-budget");
    if (!packet.mask_state_available ||
        !std::all_of(packet.slots_available.begin(), packet.slots_available.end(),
                     [](bool available) { return available; }))
        return reject("packet-invocation-state-unavailable");
    if (!packet.float_mode.canonical() || !packet.float_flags.canonical() ||
        !packet.float_transport.canonical() ||
        std::any_of(packet.export_enabled.begin(), packet.export_enabled.end(),
                    [](uint8_t value) { return value > 1; }))
        return reject("packet-launch-state-invalid");
    // The compute diagnostic cap can add an internal GDS witness and terminate a guest program.
    // Neither is in this owned packet ABI. Pin one observed operation so a concurrent environment
    // change cannot arm it after this refusal check; do not silently inherit a lossy compute cap.
    TripBoundOperation trip_operation;
    if (trip_operation.settings().bound &&
        (!trip_operation.settings().only_program ||
         trip_operation.settings().only_program == diagnostic.program_address))
        return reject("packet-diagnostic-trip-bound-unimplemented");
    std::map<int, uint32_t> columns, scalars;
    for (uint32_t column = 0; column < packet.vgprs.size(); ++column) {
        const auto& input = packet.vgprs[column];
        if (input.reg > 255 || !columns.emplace(static_cast<int>(input.reg), column).second)
            return reject("packet-vgpr-input-invalid");
    }
    for (const auto& [reg, value] : packet.sgprs)
        if (reg > 105 || !scalars.emplace(static_cast<int>(reg), value).second)
            return reject("packet-sgpr-input-invalid");

    std::vector<Rdna2Inst> ins;
    rdna2_walk(packet.guest_code.data(), packet.guest_code.size(), ins);
    if (ins.empty() || !ins.back().is_end ||
        ins.back().pc + ins.back().len_dwords != packet.guest_code.size())
        return reject("packet-code-not-one-complete-program");
    std::set<uint32_t> pcs;
    std::map<uint32_t, uint32_t> exports;
    for (const auto& in : ins) pcs.insert(in.pc);
    for (const auto& in : ins) {
        if (const char* gap = packet_instruction_gap(in)) return reject(gap, in.pc);
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const uint32_t target = branch_target(in);
            // Static forward order preserves every EXP occurrence; a last-value record would not
            // faithfully retain repeated loop exports. General loops need an owned event-log ABI.
            if (target <= in.pc) return reject("packet-backedge-unimplemented", in.pc);
            if (!pcs.contains(target)) return reject("packet-branch-target-invalid", in.pc);
        }
        if (in.fmt == Rdna2Format::EXP) {
            if (exports.size() >= 64) return reject("packet-export-budget", in.pc);
            exports.emplace(in.pc, static_cast<uint32_t>(exports.size()));
        }
        const auto need_vgpr = [&](int reg) { return columns.contains(reg); };
        // VOP3 storage calls the low field VDST, but READLANE architecturally writes an SGPR.
        // Use the decoder's opcode-aware writer inventory, not its overlapping field spelling.
        for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word)
            if (!need_vgpr(in.dst.value + static_cast<int>(word)))
                return reject("packet-vgpr-input-unavailable", in.pc);
        bool missing_scalar = false, invalid_scalar_write = false;
        for_each_scalar_write(in, [&](int reg, uint32_t width) {
            const bool mask_pair = in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeMovB64 && (reg == 106 || reg == 126);
            if ((width == 2 && (reg & 1)) || reg < 0 ||
                (reg + static_cast<int>(width) > 106 && !mask_pair))
                invalid_scalar_write = true;
            for (uint32_t word = 0; word < width; ++word)
                if (reg + static_cast<int>(word) <= 105 &&
                    !scalars.contains(reg + static_cast<int>(word))) missing_scalar = true;
        });
        if (invalid_scalar_write) return reject("packet-scalar-destination-unimplemented", in.pc);
        if (missing_scalar) return reject("packet-sgpr-input-unavailable", in.pc);
        const bool pair_source = in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == kSop1OpcodeMovB64 || in.opcode == kSop1OpcodeBcnt1I32B64 ||
             in.opcode == kSop1OpcodeFf1I32B64);
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if (in.fmt == Rdna2Format::EXP && !(in.exp_en & (1u << source))) continue;
            const auto& operand = in.src[source];
            if (operand.kind == OperandKind::VGPR && !need_vgpr(operand.value))
                return reject("packet-vgpr-input-unavailable", in.pc);
            if (operand.kind == OperandKind::SGPR) {
                if (pair_source && ((operand.value & 1) || operand.value > 104))
                    return reject("packet-scalar-pair-input-invalid", in.pc);
                if (!scalars.contains(operand.value))
                    return reject("packet-sgpr-input-unavailable", in.pc);
                if (in.fmt == Rdna2Format::SOP1 &&
                    (in.opcode == kSop1OpcodeMovB64 || in.opcode == kSop1OpcodeBcnt1I32B64 ||
                     in.opcode == kSop1OpcodeFf1I32B64) && !scalars.contains(operand.value + 1))
                    return reject("packet-sgpr-input-unavailable", in.pc);
            }
            if (pair_source && operand.kind != OperandKind::SGPR &&
                !(operand.kind == OperandKind::Special &&
                  (operand.value == 106 || operand.value == 126)))
                return reject("packet-scalar-pair-input-invalid", in.pc);
            if (operand.kind == OperandKind::Special && operand.value != 106 &&
                operand.value != 107 && operand.value != 126 && operand.value != 127 &&
                operand.value != 253)
                return reject("packet-special-input-unavailable", in.pc);
        }
    }
    if (exports.empty()) return reject("packet-no-export");

    FragmentPacketProgram result;
    result.input_stride = static_cast<uint32_t>(columns.size()) + 4;
    result.exports_per_lane = static_cast<uint32_t>(exports.size());
    const uint32_t record_stride = result.exports_per_lane * kFragmentPacketExportWords;
    result.input_words.resize(kFragmentPacketLanes * result.input_stride);
    result.output_words.resize(kFragmentPacketLanes * record_stride, 0);
    for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
        const uint32_t base = lane * result.input_stride;
        for (const auto& [reg, column] : columns)
            result.input_words[base + column] = packet.vgprs[column].words[lane];
        result.input_words[base + columns.size()] = (packet.exec_mask >> lane) & 1u;
        result.input_words[base + columns.size() + 1] = (packet.vcc_mask >> lane) & 1u;
        result.input_words[base + columns.size() + 2] = packet.scc;
        result.input_words[base + columns.size() + 3] = packet.export_enabled[lane];
    }
    SpirvCompute b;
    b.diagnostic = diagnostic;
    b.fragment_float_mode = packet.float_mode;
    b.fragment_float_flags = packet.float_flags;
    b.float_transport = packet.float_transport;
    b.begin(result.input_stride, nullptr, 64, 1, 1, 64, 0, true, true);
    b.guest_stage = GuestShaderStage::Fragment; // physical GLCompute/workgroup remains independent
    std::vector<uint32_t> marker;
    b.pstr(marker, "Prosper.GuestFragmentPacket=64;NoRasterPackingAuthority");
    b.putv(b.debug, Op_ModuleProcessed, marker);
    RegState state;
    for (const auto& [reg, column] : columns) state.vreg[reg] = b.load_input(column);
    for (const auto& [reg, value] : scalars) state.sreg[reg] = b.uconst(value);
    state.max_vgpr = columns.empty() ? 0 : columns.rbegin()->first;
    const uint32_t state_base = static_cast<uint32_t>(columns.size());
    state.exec = b.ucmp(Op_INotEqual, b.load_input(state_base), b.uconst(0));
    state.vcc = b.ucmp(Op_INotEqual, b.load_input(state_base + 1), b.uconst(0));
    state.scc = b.ucmp(Op_INotEqual, b.load_input(state_base + 2), b.uconst(0));
    state.exec_narrowed = true;
    const uint32_t enabled = b.load_input(state_base + 3);
    const auto export_record = [&](RegState& current, const Rdna2Inst& in) {
        const uint32_t base = exports.at(in.pc) * kFragmentPacketExportWords;
        const uint32_t fields[] = {b.uconst(1), b.sel(current.exec, b.uconst(1), b.uconst(0)),
            enabled, b.uconst(in.exp_target), b.uconst(in.exp_en), b.uconst(in.exp_compr),
            b.uconst((in.words[0] >> 11) & 1u), b.uconst((in.words[0] >> 12) & 1u)};
        for (uint32_t field = 0; field < 8; ++field)
            b.store_output_word(fields[field], record_stride, base + field, 0);
        for (uint32_t channel = 0; channel < 4; ++channel) {
            if (!(in.exp_en & (1u << channel))) continue;
            bool valid = true;
            const uint32_t bits = operand_bits(b, current, in, in.src[channel], &valid);
            if (!valid) return false;
            b.store_output_word(bits, record_stride, base + 8 + channel, 0);
        }
        return true; // Never OpKill, never reset EXEC, never reinterpret as a framebuffer address.
    };
    // Force the common synchronized path even for a simple forward branch. Native fragment safe-
    // branch linearization and structured subgroup votes are not packet execution authority.
    TerminalRejectCapture causes;
    if (!emit_cfg_state_machine(b, state, ins, {}, nullptr, true, false, export_record,
                                packet.guest_code.data(), packet.guest_code.size(), 0, false)) {
        const auto records = causes.take();
        return reject(records.empty() ? "packet-guest-emission-refused:no-cause-recorded"
            : "packet-guest-emission-refused:" + records.back().first + ":" +
                records.back().second.substr(0, 1024));
    }
    result.spirv = b.finish();
    if (result.spirv.empty()) return reject("packet-module-finalization-refused");
    return result;
}
} // namespace prosper::gpu
