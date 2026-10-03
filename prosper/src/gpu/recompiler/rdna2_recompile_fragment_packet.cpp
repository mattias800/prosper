#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_alu_support.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/fragment_packet_services.hpp"
#include "gpu/recompiler/fragment_packet_definedness.hpp"
#include <bitset>

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
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x09)
        return "packet-wqm-b32-unimplemented";
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x0a) {
        const auto pair = [](const Operand& op) {
            return (op.kind == OperandKind::SGPR && op.value >= 0 &&
                    op.value <= 104 && !(op.value & 1)) ||
                   ((op.kind == OperandKind::SGPR || op.kind == OperandKind::Special) &&
                    (op.value == 106 || op.value == 126));
        };
        if (!pair(in.dst)) return "packet-wqm-destination-form-unimplemented";
        if (!pair(in.src[0]) &&
            !(in.src[0].kind == OperandKind::InlineInt &&
              (in.src[0].value == 0 || in.src[0].value == -1)))
            return "packet-wqm-source-form-unimplemented";
    }
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
                           in.opcode == 0x08 || in.opcode == kSop1OpcodeAndSaveexecB64 ||
                           in.opcode == kSop1OpcodeBcnt1I32B64 ||
                           in.opcode == kSop1OpcodeFf1I32B64 || in.opcode == 0x0a
                       ? nullptr
                       : "packet-scalar-op-unimplemented";
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
            if (in.opcode == 0x365 || in.opcode == 0x366) {
                // AMD's RDNA2 machine-readable opcodes 869/870: MBCNT
                // consumes one raw dword, or one canonical half of a live wave mask.
                // Numeric operands use guest logical lane positions, not active population.
                // Bool-domain words require the synchronized, MUST-filtered CFG service.
                if ((in.words[0] >> 11) & 0xfu)
                    return "packet-mbcnt-modifier-unimplemented";
                const auto numeric = [](const Operand& op) {
                    return op.kind == OperandKind::VGPR || op.kind == OperandKind::SGPR ||
                           op.kind == OperandKind::InlineInt ||
                           op.kind == OperandKind::InlineFloat || op.kind == OperandKind::Literal;
                };
                const auto& mask = in.src[0];
                if (mask.kind == OperandKind::Special) {
                    const bool high = in.opcode == 0x366;
                    if (mask.value == 106 || mask.value == 107 ||
                        mask.value == 126 || mask.value == 127) {
                        if (mask.value != (high ? 107 : 106) &&
                            mask.value != (high ? 127 : 126))
                            return "packet-mbcnt-mask-half-unimplemented";
                    } else return "packet-mbcnt-source-kind-unimplemented";
                } else if (!numeric(mask)) return "packet-mbcnt-source-kind-unimplemented";
                if (!numeric(in.src[1]) &&
                    !(in.src[1].kind == OperandKind::Special && in.src[1].value == 253))
                    return "packet-mbcnt-accumulator-kind-unimplemented";
                return nullptr;
            }
            if (in.opcode != 0x360) return "packet-valu-op-unimplemented";
            // RDNA2's READLANE source is VGPR-or-LDS, not generic VOP3 SSRC. The
            // synchronized service reads the VGPR bank; never alias a decoded SGPR
            // number into that bank. LDS-direct is outside this owned-register slice.
            if (in.src[0].kind != OperandKind::VGPR)
                return "packet-readlane-source-kind-unimplemented";
            // OPR_SSRC_LANESEL supplies an ordinary scalar register or inline integer
            // 0..63 in this slice. A varying VGPR selector cannot become scalar state;
            // literal, float and special selector encodings are not admitted here.
            if (in.src[1].kind != OperandKind::SGPR &&
                !(in.src[1].kind == OperandKind::InlineInt &&
                  in.src[1].value >= 0 && in.src[1].value <= 63))
                return "packet-readlane-selector-kind-unimplemented";
            return nullptr;
        case Rdna2Format::EXP:
            if (in.exp_compr) return "packet-compressed-export-unimplemented";
            if (in.exp_target < kFragmentColorOutputs) return nullptr;
            if (in.exp_target == 8 && in.exp_en && !(in.exp_en & ~5u)) return nullptr;
            if (in.exp_target == 9 && !in.exp_en) return nullptr;
            return "packet-export-target-unimplemented";
        default: return "packet-format-unimplemented";
    }
}

// Register allocation is not an entry-value proof. This packet slice admits only forward direct
// branches, so a bounded instruction-level DAG meet can inspect every structurally reachable
// scalar read before admitting the program. No supplied SCC/EXEC value prunes conditional arms.
// Physical VCC/EXEC masks retain their separate dispatcher domain/lifetime proof; ordinary SGPR
// words here may hold either data or saved masks, and initialization does not authorize a cast.
const char* packet_scalar_initialization_gap(const std::vector<Rdna2Inst>& ins,
                                           const std::map<int, uint32_t>& supplied,
                                           uint32_t& failure_pc, bool resource_variant) {
    using Words = std::bitset<106>;
    std::vector<Words> entry(ins.size());
    std::vector<bool> reachable(ins.size(), false);
    std::map<uint32_t, size_t> index;
    for (size_t i = 0; i < ins.size(); ++i) index.emplace(ins[i].pc, i);
    for (const auto& [reg, value] : supplied) entry.front().set(reg);
    reachable.front() = true;
    const auto meet = [&](size_t next, const Words& defined) {
        if (!reachable[next]) { entry[next] = defined; reachable[next] = true; }
        else entry[next] &= defined;
    };
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!reachable[i]) continue;
        const auto& in = ins[i];
        failure_pc = in.pc;
        Words defined = entry[i];
        const auto available = [&](int first, uint32_t count) {
            if (first < 0 || first > 105 || !count || count > (resource_variant ? 8u : 2u) ||
                first + static_cast<int>(count) > 106) return false;
            for (uint32_t word = 0; word < count; ++word)
                if (!defined.test(first + word)) return false;
            return true;
        };
        const uint32_t implicit = scalar_implicit_destination_read_width(in);
        if (implicit && !available(in.dst.value, implicit))
            return "packet-sgpr-read-before-definition";
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if (in.fmt == Rdna2Format::EXP && !(in.exp_en & (1u << source))) continue;
            const auto& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR) continue;
            const uint32_t width = resource_variant && in.fmt == Rdna2Format::SMEM
                ? (source == 0 ? 4u : 1u)
                : resource_variant && in.fmt == Rdna2Format::MIMG
                ? (source == 1 ? 8u : source == 2 ? 4u : 1u)
                : scalar_alu_source_words(in, source);
            if (!width || width == UINT32_MAX || width > (resource_variant ? 8u : 2u))
                return "packet-scalar-read-form-unimplemented";
            if (!available(operand.value, width))
                return "packet-sgpr-read-before-definition";
        }
        // Read OLD sources first (including pair overlaps); only genuine unconditional writers
        // in the whole-program admitted packet inventory can establish a new reaching value.
        for_each_scalar_write(in, [&](int first, uint32_t count) {
            for (uint32_t word = 0; word < count; ++word)
                if (first + static_cast<int>(word) <= 105)
                    defined.set(first + word);
        });
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            sopp_opcode_is_direct_branch(in.opcode);
        if (branch) meet(index.at(branch_target(in)), defined);
        if ((!branch || in.opcode != 0x02) && i + 1 < ins.size()) meet(i + 1, defined);
    }
    failure_pc = UINT32_MAX;
    return nullptr;
}
} // namespace

FragmentPacketProgram recompile_fragment_packet(const FragmentInvocationPacket& packet,
                                                RecompileDiagnosticContext diagnostic) {
    return recompile_fragment_packet_impl(packet, diagnostic, nullptr);
}

FragmentPacketProgram recompile_fragment_packet_impl(const FragmentInvocationPacket& packet,
                                                     RecompileDiagnosticContext diagnostic,
                                                     PacketResourceServices* services,
                                                     PacketWaveDataLayout* wave_data) {
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
    if (!std::all_of(packet.slots_available.begin(), packet.slots_available.end(),
                     [](bool available) { return available; }))
        return reject("packet-invocation-state-unavailable");
    if (packet.quad_topology != FragmentPacketQuadTopology::Unknown &&
        packet.quad_topology != FragmentPacketQuadTopology::ConsecutiveLogicalQuads)
        return reject("packet-quad-topology-invalid");
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
        const char* gap = packet_instruction_gap(in);
        if (services && gap && !packet_resource_instruction_gap(in)) gap = nullptr;
        else if (services && in.fmt == Rdna2Format::SMEM)
            gap = packet_resource_instruction_gap(in);
        else if (services && in.fmt == Rdna2Format::VINTRP && in.opcode == 0 &&
                 in.dst.value == in.src[0].value) gap = "packet-parameter-p1-alias-mode-unavailable";
        if (gap) return reject(gap, in.pc);
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x0a &&
            packet.quad_topology != FragmentPacketQuadTopology::ConsecutiveLogicalQuads)
            return reject("packet-quad-topology-unavailable", in.pc);
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
        bool invalid_scalar_write = false;
        for_each_scalar_write(in, [&](int reg, uint32_t width) {
            const bool mask_pair = in.fmt == Rdna2Format::SOP1 &&
                (in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x0a) &&
                (reg == 106 || reg == 126);
            const bool resource_m0 = services && in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeMovB32 && reg == 124;
            if (!resource_m0 && ((width == 2 && (reg & 1)) || reg < 0 ||
                (reg + static_cast<int>(width) > 106 && !mask_pair))
                ) invalid_scalar_write = true;
        });
        if (invalid_scalar_write) return reject("packet-scalar-destination-unimplemented", in.pc);
        const bool pair_source =
            in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == kSop1OpcodeMovB64 || in.opcode == kSop1OpcodeBcnt1I32B64 ||
             in.opcode == kSop1OpcodeFf1I32B64 || in.opcode == 0x0a || in.opcode == 0x08 ||
             in.opcode == kSop1OpcodeAndSaveexecB64);
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if (in.fmt == Rdna2Format::EXP && !(in.exp_en & (1u << source))) continue;
            const auto& operand = in.src[source];
            if (operand.kind == OperandKind::SGPR) {
                if (pair_source && ((operand.value & 1) || operand.value > 104))
                    return reject("packet-scalar-pair-input-invalid", in.pc);
            }
            if (pair_source && operand.kind != OperandKind::SGPR &&
                !(operand.kind == OperandKind::Special &&
                  (operand.value == 106 || operand.value == 126)) &&
                !(in.opcode == 0x0a && operand.kind == OperandKind::InlineInt &&
                  (operand.value == 0 || operand.value == -1)))
                return reject("packet-scalar-pair-input-invalid", in.pc);
            if (operand.kind == OperandKind::Special && operand.value != 106 &&
                operand.value != 107 && operand.value != 126 && operand.value != 127 &&
                operand.value != 253 && !(services && in.fmt == Rdna2Format::SMEM &&
                    source == 1 && operand.value == 125))
                return reject("packet-special-input-unavailable", in.pc);
        }
    }
    if (exports.empty()) return reject("packet-no-export");
    uint32_t scalar_failure_pc = UINT32_MAX;
    if (const auto* gap = packet_scalar_initialization_gap(ins, scalars, scalar_failure_pc, services != nullptr))
        return reject(gap, scalar_failure_pc);
    if (services)
        if (const auto* gap = packet_resource_preflight(services->input, ins, scalar_failure_pc))
            return reject(gap, scalar_failure_pc);

    const auto requirements = fragment_packet_vgpr_requirements(packet.guest_code, ins);
    if (!requirements.rejection.empty()) return reject(requirements.rejection);
    if (!requirements.masks.rejection.empty()) return reject(requirements.masks.rejection);
    const uint8_t mask_availability = fragment_packet_initial_mask_availability(packet);
    uint32_t mask_failure_pc = UINT32_MAX;
    if (const auto* gap = fragment_packet_missing_initial_mask(requirements.masks,
                                                               mask_availability, mask_failure_pc))
        return reject(gap, mask_failure_pc);
    bool runtime_definedness = wave_data != nullptr;
    for (uint32_t reg = 0; reg < 256; ++reg)
        if (requirements.storage.test(reg)) {
            const auto column = columns.find(reg);
            runtime_definedness |= column == columns.end() ||
                                   packet.vgprs[column->second].available_mask != UINT64_MAX;
        }
    runtime_definedness &= !requirements.reads.empty();

    FragmentPacketProgram result;
    result.initial_mask_availability = mask_availability;
    result.demanded_initial_masks = requirements.masks.demanded;
    result.input_stride =
        static_cast<uint32_t>(columns.size()) * (runtime_definedness ? 2u : 1u) + 4;
    result.exports_per_lane = static_cast<uint32_t>(exports.size());
    const uint32_t record_stride = result.exports_per_lane * kFragmentPacketExportWords;
    result.input_words.resize(kFragmentPacketLanes * result.input_stride);
    result.output_words.resize(kFragmentPacketLanes * record_stride, 0);
    for (uint32_t lane = 0; lane < kFragmentPacketLanes; ++lane) {
        const uint32_t base = lane * result.input_stride;
        for (const auto& [reg, column] : columns)
            result.input_words[base + column] = packet.vgprs[column].words[lane];
        if (runtime_definedness)
            for (const auto& [reg, column] : columns)
                result.input_words[base + columns.size() + column] =
                    (packet.vgprs[column].available_mask >> lane) & 1u;
        const auto state_base = columns.size() * (runtime_definedness ? 2u : 1u);
        result.input_words[base + state_base] = (packet.exec_mask >> lane) & 1u;
        result.input_words[base + state_base + 1] = (packet.vcc_mask >> lane) & 1u;
        result.input_words[base + state_base + 2] = packet.scc;
        result.input_words[base + state_base + 3] = packet.export_enabled[lane];
    }
    if (services) {
        for (const auto& buffer : services->input.buffers) {
            services->buffer_offsets.emplace(buffer.pc, static_cast<uint32_t>(result.input_words.size()));
            result.input_words.insert(result.input_words.end(), buffer.words.begin(), buffer.words.end());
        }
        // One owned coefficient tuple per lane for each exact VINTRP PC. Duplicated coefficients
        // are input bytes, not baked output constants; the service consumes the actual I/J/old VDST.
        for (const auto& in : ins) if (in.fmt == Rdna2Format::VINTRP) {
            services->parameter_offsets.emplace(in.pc, static_cast<uint32_t>(result.input_words.size()));
            for (uint32_t lane = 0; lane < 64; ++lane) {
                const auto primitive = services->input.parameter_cache.quad_primitive[lane / 4];
                const auto it = std::find_if(services->input.parameter_cache.parameters.begin(),
                    services->input.parameter_cache.parameters.end(), [&](const auto& p) {
                        return p.primitive == primitive && p.attribute == in.vintrp_attr && p.channel == in.vintrp_chan;
                    });
                if (it == services->input.parameter_cache.parameters.end()) return reject("packet-parameter-tuple-unavailable", in.pc);
                result.input_words.insert(result.input_words.end(), {it->p0, it->p10, it->p20});
            }
        }
        services->output.status_offset = static_cast<uint32_t>(result.output_words.size());
        result.output_words.resize(result.output_words.size() + 64 * kFragmentResourceStatusWords, 0);
    }
    if (runtime_definedness) {
        result.vgpr_status_offset = static_cast<uint32_t>(result.output_words.size());
        result.output_words.resize(result.output_words.size() + 64 * kFragmentPacketVgprStatusWords,
                                   0);
        for (const auto& [pc, reads] : requirements.reads)
            for (const auto& read : reads)
                result.vgpr_failure_sites.push_back(
                    {pc, read.reg, static_cast<uint32_t>(read.kind)});
    }
    if (wave_data) configure_packet_wave_data(packet, ins, result, *services, *wave_data);
    SpirvCompute b;
    b.diagnostic = diagnostic;
    b.fragment_float_mode = packet.float_mode;
    b.fragment_float_flags = packet.float_flags;
    b.float_transport = packet.float_transport;
    b.begin(result.input_stride, nullptr, 64, 1, 1, 64, 0, true, true);
    const auto wave_emission =
        wave_data ? begin_packet_wave_data(b, *wave_data) : PacketWaveEmission{};
    b.guest_stage = GuestShaderStage::Fragment; // physical GLCompute/workgroup remains independent
    b.packet_quad_topology = packet.quad_topology;
    std::vector<uint32_t> marker;
    b.pstr(marker, "Prosper.GuestFragmentPacket=64;NoRasterPackingAuthority");
    b.putv(b.debug, Op_ModuleProcessed, marker);
    if (runtime_definedness) {
        marker.clear();
        b.pstr(marker, kPacketVgprValidityMarker);
        b.putv(b.debug, Op_ModuleProcessed, marker);
    }
    RegState state;
    for (const auto& [reg, column] : columns) state.vreg[reg] = b.load_input(column);
    uint32_t scalar_column = 0;
    for (const auto& [reg, value] : scalars)
        state.sreg[reg] =
            wave_data ? b.load_packet_word(b.uconst(wave_data->scalar_offsets.at(scalar_column++)))
                      : b.uconst(value);
    // Allocated missing storage is an INTERNAL placeholder, never entry authority. Per-logical-
    // lane validity checks every actual read before transactional publication, including implicit
    // P2/wide resource ranges, EXEC-ignoring peer selection and the unchanged raw inactive EXP ABI.
    state.max_vgpr = columns.empty() ? 0 : columns.rbegin()->first;
    for (uint32_t reg = 0; reg < 256; ++reg)
        if (requirements.storage.test(reg)) {
            if (!state.vreg.contains(reg)) state.vreg.emplace(reg, b.uconst(0));
            state.max_vgpr = std::max(state.max_vgpr, static_cast<int>(reg));
        }
    const uint32_t state_base =
        static_cast<uint32_t>(columns.size()) * (runtime_definedness ? 2u : 1u);
    if (mask_availability & kPacketInitialExec)
        state.exec = b.ucmp(Op_INotEqual, b.load_input(state_base), b.uconst(0));
    if (mask_availability & kPacketInitialVcc)
        state.vcc = b.ucmp(Op_INotEqual, b.load_input(state_base + 1), b.uconst(0));
    if (mask_availability & kPacketInitialScc)
        state.scc = b.ucmp(Op_INotEqual, b.load_input(state_base + 2), b.uconst(0));
    state.exec_narrowed = true;
    const uint32_t enabled = b.load_input(state_base + 3);
    if (services) services->begin(b);
    PacketVgprDefinedness definedness{requirements};
    if (runtime_definedness) definedness.begin(b, columns);
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
    const std::function<int(RegState&, const Rdna2Inst&)> service_callback = services
        ? std::function<int(RegState&, const Rdna2Inst&)>([&](RegState& current, const Rdna2Inst& in) {
            return services->emit(b, current, in);
          }) : std::function<int(RegState&, const Rdna2Inst&)>{};
    if (!emit_cfg_state_machine(b, state, ins, {}, nullptr, true, false, export_record,
                                packet.guest_code.data(), packet.guest_code.size(), 0, false,
                                service_callback, runtime_definedness ? &definedness : nullptr,
                                mask_availability == 7 ? nullptr : &requirements.masks)) {
        const auto records = causes.take();
        return reject(records.empty() ? "packet-guest-emission-refused:no-cause-recorded"
            : "packet-guest-emission-refused:" + records.back().first + ":" +
                records.back().second.substr(0, 1024));
    }
    if (services) services->finish(b);
    if (runtime_definedness) definedness.finish(b, result.vgpr_status_offset);
    if (wave_data) finish_packet_wave_data(b, wave_emission);
    result.spirv = b.finish();
    if (result.spirv.empty()) return reject("packet-module-finalization-refused");
    return result;
}
} // namespace prosper::gpu
