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
const char* packet_instruction_gap(const Rdna2Inst& in, GraphicsPacketStage stage) {
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
    if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::FLAT)
        return "packet-memory-effect-unimplemented";
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
                in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64 ||
                in.opcode == 0x0a
                ? nullptr : "packet-scalar-op-unimplemented";
        case Rdna2Format::SOP2:
            return in.opcode == kSop2OpcodeAddU32 || in.opcode == kSop2OpcodeAddcU32 ||
                           in.opcode == kSop2OpcodeCselectB32 || in.opcode == kSop2OpcodeAndB32 ||
                           in.opcode == 0x1eu || in.opcode == 0x20u || in.opcode == 0x27u
                       ? nullptr
                       : "packet-scalar-op-unimplemented";
        case Rdna2Format::SOPC:
            return in.opcode >= 0x06u && in.opcode <= 0x0bu ? nullptr
                                                            : "packet-scalar-compare-unimplemented";
        case Rdna2Format::SMEM:
            return in.opcode == 2u || in.opcode == 3u ? nullptr
                                                      : "packet-memory-effect-unimplemented";
        case Rdna2Format::SOPK:
            return in.opcode == kSopkOpcodeMovkI32 ? nullptr : "packet-scalar-op-unimplemented";
        case Rdna2Format::VOP1:
            if (in.opcode == 2u && in.src[0].kind != OperandKind::VGPR)
                return "packet-readfirstlane-source-kind-unimplemented";
            return in.opcode == 1 || in.opcode == 2 || in.opcode == 6 || in.opcode == 7
                       ? nullptr
                       : "packet-valu-op-unimplemented";
        case Rdna2Format::VOP2:
            // The direct-stage bridge uses these existing ordinary emitters for index-derived
            // positions and scalar-window colours. It retains the actual producing float profile;
            // this grants no image, interpolation, derivative, modifier or arbitrary ALU support.
            return in.opcode == 0x03u || in.opcode == 0x04u || in.opcode == 0x08u ||
                           in.opcode == 0x16u || in.opcode == 0x1bu
                       ? nullptr
                       : "packet-valu-op-unimplemented";
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
            if (stage == GraphicsPacketStage::Vertex)
                return (in.exp_target == 12u && in.exp_en == 15u) ||
                               (in.exp_target >= 32u && in.exp_target < 64u && in.exp_en == 15u)
                           ? nullptr
                           : "packet-vertex-export-unimplemented";
            if (in.exp_target < kFragmentColorOutputs) return nullptr;
            if (in.exp_target == 8 && in.exp_en && !(in.exp_en & ~5u)) return nullptr;
            if (in.exp_target == 9 && !in.exp_en) return nullptr;
            return "packet-export-target-unimplemented";
        default: return "packet-format-unimplemented";
    }
}

// These are reads of OLD physical scalar words, including the register OFFSET before raw SMEM
// overwrites its destination. Generic scalar ALU inventory deliberately returns zero for SMEM.
uint32_t packet_scalar_source_words(const Rdna2Inst& in, uint32_t source) {
    if (in.fmt == Rdna2Format::SMEM) return source == 0u ? 2u : source == 1u ? 1u : UINT32_MAX;
    return scalar_alu_source_words(in, source);
}

// Register allocation is not an entry-value proof. This packet slice admits direct branches.
// A finite instruction-level MUST fixed point inspects every structurally reachable
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
    // Loops meet the entry path as well as every backedge; a definition produced only on a
    // later visit can never authorize the first visit. No supplied SCC prunes either arm.
    std::vector<size_t> pending{0};
    size_t transfers = 0;
    while (!pending.empty()) {
        if (++transfers > ins.size() * 108u) return "packet-scalar-proof-budget";
        const size_t i = pending.back();
        pending.pop_back();
        const auto& in = ins[i];
        Words defined = entry[i];
        for_each_scalar_write(in, [&](int first, uint32_t count) {
            for (uint32_t word = 0; word < count; ++word)
                if (first + static_cast<int>(word) >= 0 && first + static_cast<int>(word) <= 105)
                    defined.set(first + word);
        });
        if (in.is_end) continue;
        const auto meet = [&](size_t next) {
            const Words merged = reachable[next] ? entry[next] & defined : defined;
            if (!reachable[next] || merged != entry[next]) {
                entry[next] = merged;
                reachable[next] = true;
                pending.push_back(next);
            }
        };
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (branch) meet(index.at(branch_target(in)));
        if ((!branch || in.opcode != kSoppOpcodeBranch) && i + 1 < ins.size()) meet(i + 1);
    }
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
                                       ? (source == 1   ? 8u
                                          : source == 2 ? 4u
                                                        : 1u)
                                       : packet_scalar_source_words(in, source);
            if (!width || width == UINT32_MAX || width > (resource_variant ? 8u : 2u))
                return "packet-scalar-read-form-unimplemented";
            if (!available(operand.value, width))
                return "packet-sgpr-read-before-definition";
        }
        // OLD source/pair/implicit reads are checked against the final incoming MUST state,
        // before this instruction's unconditional writer can define its destination.
    }
    failure_pc = UINT32_MAX;
    return nullptr;
}
} // namespace

bool complete_graphics_packet_locals(const FragmentInvocationPacket& packet, std::string& refusal,
                                     bool entry_flags_observed) {
    refusal.clear();
    const auto reject = [&](const char* reason, uint32_t pc = UINT32_MAX) {
        refusal = std::string(reason) + ":pc=" + std::to_string(pc);
        return false;
    };
    std::vector<Rdna2Inst> ins;
    if (packet.guest_code.empty() || packet.guest_code.size() > 4096u ||
        rdna2_walk(packet.guest_code.data(), packet.guest_code.size(), ins) !=
            packet.guest_code.size() ||
        ins.empty() || !ins.back().is_end || !packet.mask_state_available)
        return reject("stage-input-original-unavailable");
    struct Defined {
        std::array<uint64_t, 256>
            vector{};   // initialized architectural slots, not host executions
        std::bitset<106> scalar;
        uint64_t vcc_defined = 0;
        bool scc_defined = false;
        uint64_t possible_exec = 0, certain_exec = 0;
        bool operator==(const Defined&) const = default;
    };
    Defined initial;
    std::bitset<256> supplied_vectors;
    for (const auto& column : packet.vgprs) {
        if (column.reg >= 256u || supplied_vectors[column.reg])
            return reject("stage-input-vector-duplicate");
        supplied_vectors.set(column.reg);
        initial.vector[column.reg] = column.available_mask;
    }
    for (const auto& [reg, value] : packet.sgprs) {
        (void)value;
        if (reg >= 106u || initial.scalar[reg]) return reject("stage-input-scalar-duplicate");
        initial.scalar.set(reg);
    }
    initial.possible_exec = initial.certain_exec = packet.exec_mask;
    // Offline packet callers explicitly supply these states. Live stage assembly observes only
    // launch EXEC; its default storage zeros are not SCC/VCC entry values.
    initial.scc_defined = entry_flags_observed;
    initial.vcc_defined = entry_flags_observed ? UINT64_MAX : 0;
    std::map<uint32_t, size_t> indices;
    std::vector<std::vector<size_t>> edges(ins.size());
    for (size_t index = 0; index < ins.size(); ++index) {
        const auto& in = ins[index];
        if (!indices.emplace(in.pc, index).second || packet_instruction_gap(in, packet.stage))
            return reject("stage-input-instruction-unimplemented", in.pc);
        for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word) {
            const int reg = in.dst.value + int(word);
            if (reg < 0 || reg >= 256) return reject("stage-input-vector-destination", in.pc);
        }
        bool invalid = false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word) {
                const int reg = base + int(word);
                if ((reg < 0 || reg >= 106) && reg != 106 && reg != 107 && reg != 126 && reg != 127)
                    invalid = true;
            }
        });
        if (invalid) return reject("stage-input-scalar-destination", in.pc);
    }
    for (size_t index = 0; index < ins.size(); ++index) {
        const auto& in = ins[index];
        if (in.is_end) continue;
        bool fallthrough = true;
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !indices.contains(uint32_t(target)))
                return reject("stage-input-control-unimplemented", in.pc);
            edges[index].push_back(indices.at(uint32_t(target)));
            fallthrough = in.opcode != kSoppOpcodeBranch;
        }
        if (fallthrough) {
            if (index + 1 == ins.size()) return reject("stage-input-control-unimplemented", in.pc);
            edges[index].push_back(index + 1);
        }
    }
    std::vector<Defined> incoming(ins.size());
    std::vector<bool> reached(ins.size(), false);
    incoming[0] = initial;
    reached[0] = true;
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
        const size_t index = pending.back();
        pending.pop_back();
        const auto& in = ins[index];
        Defined next = incoming[index];
        // Only slots guaranteed to execute this write gain a definition. A later ordinary VALU/
        // EXP needs every possibly active slot; READFIRST additionally needs real lane0 whenever
        // empty EXEC remains possible, and READLANE needs its independently selectable source.
        for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word)
            next.vector[in.dst.value + word] |= next.certain_exec;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t word = 0; word < width; ++word)
                if (base + int(word) >= 0 && base + int(word) < 106)
                    next.scalar.set(base + word);
                else if (base + int(word) == 106 || base + int(word) == 107)
                    next.vcc_defined |= uint64_t(UINT32_MAX) << ((base + word - 106) * 32u);
        });
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode))
            next.vcc_defined |= next.certain_exec;
        if (in.fmt == Rdna2Format::SOPC ||
            (in.fmt == Rdna2Format::SOP1 && !sop1_opcode_leaves_scc_unmodified(in.opcode)) ||
            (in.fmt == Rdna2Format::SOP2 && in.opcode != kSop2OpcodeCselectB32))
            next.scc_defined = true;
        if (rdna2_instruction_may_change_exec(in)) {
            const auto preserved_entry_pair = [&](const Operand& source, uint64_t& value) {
                if (source.kind == OperandKind::Special && source.value == 126 &&
                    next.possible_exec == next.certain_exec) {
                    value = next.possible_exec;
                    return true;
                }
                if (source.kind == OperandKind::InlineInt &&
                    (source.value == 0 || source.value == -1)) {
                    value = source.value ? UINT64_MAX : 0;
                    return true;
                }
                if (source.kind != OperandKind::SGPR || source.value < 0 || source.value > 104)
                    return false;
                auto low =
                    std::find_if(packet.sgprs.begin(), packet.sgprs.end(), [&](const auto& word) {
                        return word.first == uint32_t(source.value);
                    });
                auto high =
                    std::find_if(packet.sgprs.begin(), packet.sgprs.end(), [&](const auto& word) {
                        return word.first == uint32_t(source.value + 1);
                    });
                if (low == packet.sgprs.end() || high == packet.sgprs.end()) return false;
                bool unchanged = true;
                for (const auto& writer : ins)
                    for_each_scalar_write(writer, [&](int reg, uint32_t width) {
                        if (reg < source.value + 2 && source.value < reg + int(width))
                            unchanged = false;
                    });
                if (!unchanged) return false;
                value = uint64_t(low->second) | uint64_t(high->second) << 32u;
                return true;
            };
            uint64_t exact = 0;
            if (in.fmt == Rdna2Format::SOP1 &&
                (in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x0au) && in.dst.value == 126 &&
                preserved_entry_pair(in.src[0], exact)) {
                if (in.opcode == 0x0au)
                    for (uint32_t quad = 0; quad < 16u; ++quad)
                        if (exact & (uint64_t(15u) << (quad * 4u)))
                            exact |= uint64_t(15u) << (quad * 4u);
                next.possible_exec = next.certain_exec = exact;
            } else {
                next.possible_exec = UINT64_MAX;
                next.certain_exec = 0;
            }
        }
        for (size_t target : edges[index]) {
            Defined merged = next;
            if (reached[target]) {
                for (uint32_t reg = 0; reg < 256u; ++reg)
                    merged.vector[reg] &= incoming[target].vector[reg];
                merged.scalar &= incoming[target].scalar;
                merged.vcc_defined &= incoming[target].vcc_defined;
                merged.scc_defined = merged.scc_defined && incoming[target].scc_defined;
                merged.possible_exec |= incoming[target].possible_exec;
                merged.certain_exec &= incoming[target].certain_exec;
            }
            if (!reached[target] || merged != incoming[target]) {
                incoming[target] = merged;
                reached[target] = true;
                pending.push_back(target);
            }
        }
    }
    for (size_t index = 0; index < ins.size(); ++index) {
        if (!reached[index]) continue;
        const auto& in = ins[index];
        const auto& defined = incoming[index];
        const bool implicit_scc =
            (in.fmt == Rdna2Format::SOP2 &&
             (in.opcode == kSop2OpcodeCselectB32 || in.opcode == kSop2OpcodeAddcU32)) ||
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 4u || in.opcode == 5u));
        if (implicit_scc && !defined.scc_defined)
            return reject("stage-input-scc-uninitialized", in.pc);
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 6u || in.opcode == 7u) &&
            defined.vcc_defined != UINT64_MAX)
            return reject("stage-input-vcc-uninitialized", in.pc);
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if (in.fmt == Rdna2Format::EXP && !(in.exp_en & (1u << source))) continue;
            const auto& operand = in.src[source];
            if (operand.kind == OperandKind::Special) {
                if (operand.value == 253 && !defined.scc_defined)
                    return reject("stage-input-scc-uninitialized", in.pc);
                if (operand.value == 106 || operand.value == 107) {
                    const uint32_t width = packet_scalar_source_words(in, source);
                    const uint64_t needed = width == 2u ? UINT64_MAX
                                                        : uint64_t(UINT32_MAX)
                                                              << ((operand.value - 106) * 32u);
                    if ((defined.vcc_defined & needed) != needed)
                        return reject("stage-input-vcc-uninitialized", in.pc);
                }
                if (operand.value == 251 && defined.vcc_defined != UINT64_MAX)
                    return reject("stage-input-vcc-uninitialized", in.pc);
            }
            if (operand.kind == OperandKind::VGPR)
                for (uint32_t word = 0; word < rdna2_vgpr_source_span(in, source); ++word) {
                    const int reg = operand.value + int(word);
                    const bool first = in.fmt == Rdna2Format::VOP1 && in.opcode == 2u;
                    const bool lane = in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360u;
                    const uint64_t required =
                        in.fmt == Rdna2Format::EXP || lane ? UINT64_MAX
                        : first ? defined.possible_exec | (defined.certain_exec ? 0u : 1u)
                                : defined.possible_exec;
                    if (reg < 0 || reg >= 256 || (defined.vector[reg] & required) != required)
                        return reject("stage-input-vector-uninitialized", in.pc);
                }
            if (operand.kind == OperandKind::SGPR) {
                const uint32_t width = packet_scalar_source_words(in, source);
                if (!width || width == UINT32_MAX || width > 2u)
                    return reject("stage-input-scalar-form-unimplemented", in.pc);
                for (uint32_t word = 0; word < width; ++word) {
                    const int reg = operand.value + int(word);
                    if (reg < 0 || reg >= 106 || !defined.scalar[reg])
                        return reject("stage-input-scalar-uninitialized", in.pc);
                }
            }
        }
        for (uint32_t word = 0; word < scalar_implicit_destination_read_width(in); ++word) {
            const int reg = in.dst.value + int(word);
            if (reg < 0 || reg >= 106 || !defined.scalar[reg])
                return reject("stage-input-scalar-uninitialized", in.pc);
        }
    }
    // Inputs stay authentic caller entry words. The common dispatcher allocates ordinary vector
    // and scalar scratch internally; neither bank's storage becomes entry authority across APIs.
    return true;
}

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
    if (packet.stage != GraphicsPacketStage::Fragment &&
        packet.stage != GraphicsPacketStage::Vertex)
        return reject("packet-stage-invalid");
    if (services && (packet.stage != GraphicsPacketStage::Fragment || !packet.raw_windows.empty()))
        return reject("packet-resource-raw-window-domain-unimplemented");
    if (!packet.mask_state_available ||
        !std::all_of(packet.slots_available.begin(), packet.slots_available.end(),
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
    const auto certificates = rdna2_raw_wave_wide_certificates(ins);
    std::map<uint32_t, std::pair<RawWaveWideCertificate, const PacketRawWaveWindow*>> windows;
    uint64_t total_window_words = 0;
    for (const auto& window : packet.raw_windows) {
        const auto proof =
            std::find_if(certificates.begin(), certificates.end(), [&](const auto& certificate) {
                return certificate.load_pc == window.load_pc;
            });
        if (proof == certificates.end() || !scalars.contains(proof->base_sgpr) ||
            !scalars.contains(proof->base_sgpr + 1u))
            return reject("packet-window-code-or-base-unproved");
        const uint64_t base = uint64_t(scalars.at(proof->base_sgpr)) |
                              (uint64_t(scalars.at(proof->base_sgpr + 1u)) << 32u);
        const uint64_t begin_offset = uint64_t(int64_t(proof->offset_min) + proof->immediate);
        const uint64_t bytes = uint64_t(proof->offset_max) - proof->offset_min + proof->bytes;
        if (!base || (base & 3u) || base > UINT64_MAX - begin_offset ||
            base + begin_offset > UINT64_MAX - bytes || window.guest_base != base ||
            window.guest_begin != base + begin_offset || window.words.size() != bytes / 4u ||
            (bytes & 3u) || !windows.emplace(window.load_pc, std::pair{*proof, &window}).second)
            return reject("packet-window-shape-invalid", window.load_pc);
        total_window_words += window.words.size();
        if (windows.size() > 8u || total_window_words > 16u * 1024u * 1024u)
            return reject("packet-window-budget");   // refuse the whole domain, never clamp it
    }
    for (const auto& in : ins) pcs.insert(in.pc);
    // Storage inventory only: no value is supplied by allocating a destination or source slot.
    std::bitset<256> owned_vector_storage;
    for (const auto& in : ins) {
        for (uint32_t word = 0; word < rdna2_vgpr_write_count(in); ++word) {
            const int reg = in.dst.value + int(word);
            if (reg < 0 || reg >= 256)
                return reject("packet-vector-destination-unimplemented", in.pc);
            owned_vector_storage.set(reg);
        }
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if (in.fmt == Rdna2Format::EXP && !(in.exp_en & (1u << source))) continue;
            if (in.src[source].kind != OperandKind::VGPR) continue;
            for (uint32_t word = 0; word < rdna2_vgpr_source_span(in, source); ++word) {
                const int reg = in.src[source].value + int(word);
                if (reg < 0 || reg >= 256)
                    return reject("packet-vector-source-unimplemented", in.pc);
                owned_vector_storage.set(reg);
            }
        }
    }
    for (const auto& in : ins) {
        const char* gap = packet_instruction_gap(in, packet.stage);
        if (services && gap && !packet_resource_instruction_gap(in)) gap = nullptr;
        else if (services && in.fmt == Rdna2Format::SMEM)
            gap = packet_resource_instruction_gap(in);
        else if (services && in.fmt == Rdna2Format::VINTRP && in.opcode == 0 &&
                 in.dst.value == in.src[0].value) gap = "packet-parameter-p1-alias-mode-unavailable";
        // The independently published resource API retains its closed forward instruction slice.
        // This live raw-window extension must not expand that API or mix its input authorities.
        if (services && in.fmt == Rdna2Format::VOP1 && in.opcode != 1u &&
            !packet_special_f32_opcode(in.opcode))
            gap = "packet-valu-op-unimplemented";
        if (services && in.fmt == Rdna2Format::VOP2 && in.opcode != 3u && in.opcode != 8u)
            gap = "packet-valu-op-unimplemented";
        if (services && in.fmt == Rdna2Format::SOPC) gap = "packet-format-unimplemented";
        if (services && in.fmt == Rdna2Format::SOP2 && in.opcode != kSop2OpcodeAddU32 &&
            in.opcode != kSop2OpcodeCselectB32)
            gap = "packet-scalar-op-unimplemented";
        if (gap) return reject(gap, in.pc);
        if (!services && in.fmt == Rdna2Format::SMEM && !windows.contains(in.pc))
            return reject("packet-raw-window-unavailable", in.pc);
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == 0x0a &&
            packet.quad_topology != FragmentPacketQuadTopology::ConsecutiveLogicalQuads)
            return reject("packet-quad-topology-unavailable", in.pc);
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const uint32_t target = branch_target(in);
            if (!pcs.contains(target)) return reject("packet-branch-target-invalid", in.pc);
            if (services && target <= in.pc) return reject("packet-backedge-unimplemented", in.pc);
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
        const bool pair_source = in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == kSop1OpcodeMovB64 || in.opcode == kSop1OpcodeBcnt1I32B64 ||
             in.opcode == kSop1OpcodeFf1I32B64 || in.opcode == 0x0a);
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
                !((in.opcode == 0x0a || (in.opcode == kSop1OpcodeMovB64 && in.dst.value == 126)) &&
                  operand.kind == OperandKind::InlineInt &&
                  (operand.value == 0 || operand.value == -1)))
                return reject("packet-scalar-pair-input-invalid", in.pc);
            if (operand.kind == OperandKind::Special && operand.value != 106 &&
                operand.value != 107 && operand.value != 126 && operand.value != 127 &&
                operand.value != 253 &&
                !(services && in.fmt == Rdna2Format::SMEM && source == 1 && operand.value == 125))
                return reject("packet-special-input-unavailable", in.pc);
        }
    }
    if (exports.empty()) return reject("packet-no-export");
    // The output ABI owns one record per static EXP. Permit repeated dynamic READFIRST/load
    // events, but reject any EXP in a control-flow cycle before producing even scratch output.
    std::map<uint32_t, size_t> instruction_index;
    for (size_t i = 0; i < ins.size(); ++i) instruction_index.emplace(ins[i].pc, i);
    std::vector<std::vector<size_t>> successors(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (branch) successors[i].push_back(instruction_index.at(branch_target(in)));
        if ((!branch || in.opcode != kSoppOpcodeBranch) && i + 1 < ins.size())
            successors[i].push_back(i + 1);
    }
    for (const auto& [pc, ordinal] : exports) {
        const size_t start = instruction_index.at(pc);
        std::vector<size_t> pending = successors[start];
        std::vector<bool> visited(ins.size(), false);
        while (!pending.empty()) {
            const size_t next = pending.back();
            pending.pop_back();
            if (next == start) return reject("packet-repeated-export-unimplemented", pc);
            if (visited[next]) continue;
            visited[next] = true;
            pending.insert(pending.end(), successors[next].begin(), successors[next].end());
        }
    }
    uint32_t scalar_failure_pc = UINT32_MAX;
    if (const auto* gap = packet_scalar_initialization_gap(ins, scalars, scalar_failure_pc, services != nullptr))
        return reject(gap, scalar_failure_pc);
    auto requirements = fragment_packet_vgpr_requirements(packet.guest_code, ins);
    // The independently published forward packet/resource domain owns VGP1 runtime validity.
    // The live integer/raw-window domain also admits READFIRST and repeated events; its separate
    // whole-original, per-slot MUST proof must succeed before emitting any module. A failed narrow
    // resource inventory is never enough to admit a resource packet or a missing stage input.
    const bool owned_input_proof =
        !services && (packet.stage == GraphicsPacketStage::Vertex || !windows.empty() ||
                      !requirements.rejection.empty());
    if (services) {
        if (const auto* gap = packet_resource_preflight(services->input, ins, scalar_failure_pc))
            return reject(gap, scalar_failure_pc);
    } else if (owned_input_proof) {
        std::string input_failure;
        if (!complete_graphics_packet_locals(packet, input_failure))
            return reject("packet-input-definition-unproved:" + input_failure);
    }

    if (owned_input_proof) {
        // This is NOT a resource read certificate. The strict stage proof above owns admission;
        // the inventory below only allocates the original read/write slots for that proved path.
        requirements = {};
        requirements.storage = owned_vector_storage;
    } else if (!requirements.rejection.empty())
        return reject(requirements.rejection);
    bool runtime_definedness = wave_data != nullptr;
    for (uint32_t reg = 0; reg < 256; ++reg)
        if (requirements.storage.test(reg)) {
            const auto column = columns.find(reg);
            runtime_definedness |= column == columns.end() ||
                                   packet.vgprs[column->second].available_mask != UINT64_MAX;
        }
    runtime_definedness &= !requirements.reads.empty();

    FragmentPacketProgram result;
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
    b.guest_stage = packet.stage == GraphicsPacketStage::Fragment
                        ? GuestShaderStage::Fragment
                        : GuestShaderStage::Vertex;   // physical GLCompute is independent
    b.guest_wave_event_domain = GuestWaveEventDomain::Owned64;
    for (const auto& [pc, window] : windows) {
        const uint32_t first_word = static_cast<uint32_t>(result.input_words.size());
        result.input_words.insert(result.input_words.end(), window.second->words.begin(),
                                  window.second->words.end());
        b.packet_raw_windows.emplace(pc, std::pair{window.first, first_word});
    }
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

    state.exec = b.ucmp(Op_INotEqual, b.load_input(state_base), b.uconst(0));
    state.vcc = b.ucmp(Op_INotEqual, b.load_input(state_base + 1), b.uconst(0));
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
    // Force the common synchronized path for every dynamic event instance. Native fragment safe-
    // branch linearization and structured subgroup votes are not packet execution authority.
    TerminalRejectCapture causes;
    const std::function<int(RegState&, const Rdna2Inst&)> service_callback = services
        ? std::function<int(RegState&, const Rdna2Inst&)>([&](RegState& current, const Rdna2Inst& in) {
            return services->emit(b, current, in);
          }) : std::function<int(RegState&, const Rdna2Inst&)>{};
    if (!emit_cfg_state_machine(b, state, ins, {}, nullptr, true, !windows.empty(), export_record,
                                packet.guest_code.data(), packet.guest_code.size(), 0, false,
                                service_callback, runtime_definedness ? &definedness : nullptr)) {

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

std::vector<uint32_t> build_owned_vertex_export_commit(const std::vector<uint32_t>& parameters,
                                                       uint32_t vertices_per_instance,
                                                       const PixelInputMapping* pixel_inputs,
                                                       FloatTransportConfig float_transport) {
    if (!float_transport.canonical() || !vertices_per_instance || vertices_per_instance > 4096u ||
        parameters.size() > 31u || !std::is_sorted(parameters.begin(), parameters.end()) ||
        std::adjacent_find(parameters.begin(), parameters.end()) != parameters.end() ||
        std::any_of(parameters.begin(), parameters.end(),
                    [](uint32_t target) { return target >= 32u; }))
        return {};
    SpirvCompute b;
    b.float_transport = float_transport;
    b.begin_vertex();
    const uint32_t array = b.id(), block = b.id(), pointer = b.id(), storage = b.id();
    b.t_ptr_sb_u32 = b.id();
    b.put(b.deco, Op_Decorate, {array, Dec_ArrayStride, 4u});
    b.put(b.deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
    b.put(b.deco, Op_Decorate, {block, Dec_Block});
    b.put(b.deco, Op_Decorate, {storage, Dec_DescriptorSet, 0});
    b.put(b.deco, Op_Decorate, {storage, Dec_Binding, 0});
    b.put(b.deco, Op_Decorate, {storage, 24u});   // NonWritable
    b.put(b.types, Op_TypeRuntimeArray, {array, b.t_u32});
    b.put(b.types, Op_TypeStruct, {block, array});
    b.put(b.types, Op_TypePointer, {pointer, SC_StorageBuffer, block});
    b.put(b.types, Op_TypePointer, {b.t_ptr_sb_u32, SC_StorageBuffer, b.t_u32});
    b.declare_external_storage_buffer(pointer, storage);
    b.cbuf_var.emplace(0u, storage);
    const uint32_t occurrence =
        b.ibin(Op_IAdd, b.load_vertex_index(),
               b.ibin(Op_IMul, b.load_instance_index(), b.uconst(vertices_per_instance)));
    const uint32_t base =
        b.ibin(Op_IMul, occurrence, b.uconst(uint32_t(parameters.size() + 1u) * 4u));
    const auto component = [&](uint32_t word) {
        return b.cbuf_load(b.ibin(Op_IAdd, base, b.uconst(word)), 0u);
    };
    b.export_position(component(0), component(1), component(2), component(3));
    const uint32_t passthrough = pixel_inputs ? pixel_inputs->effective_passthrough_mask() : 0u;
    for (uint32_t ordinal = 0; ordinal < parameters.size(); ++ordinal) {
        const uint32_t source = parameters[ordinal];
        const uint32_t first = (ordinal + 1u) * 4u;
        const auto publish = [&](uint32_t location) {
            b.export_param(location, component(first), component(first + 1u), component(first + 2u),
                           component(first + 3u));
        };
        // Match the ordinary vertex compiler's exact producing PARAM-to-pixel-input routing.
        if (!pixel_inputs || !(pixel_inputs->valid_mask & (1u << source))) {
            if (!pixel_inputs || pixel_inputs->consumes(source)) publish(source);
        }
        if (pixel_inputs)
            for (uint32_t input = 0; input < pixel_inputs->controls.size(); ++input) {
                if (!(pixel_inputs->valid_mask & (1u << input)) || !pixel_inputs->consumes(input))
                    continue;
                const uint32_t raw_offset = pixel_inputs->controls[input] & 0x3fu;
                const uint32_t offset =
                    (passthrough & (1u << input)) ? raw_offset & 0x1fu : raw_offset;
                if (offset == source) publish(input);
            }
    }
    return b.finish();
}
std::vector<uint32_t> build_owned_fragment_export_commit(uint32_t records,
                                                         FloatTransportConfig profile) {
    if (!records || records > 16384u || !profile.canonical()) return {};
    SpirvCompute b;
    b.float_transport = profile;
    b.begin_fragment(nullptr, 1u);
    b.put(b.caps, Op_Capability, {Cap_Geometry});
    const uint32_t primitive_pointer = b.id(), primitive_var = b.id(), signed_primitive = b.id();
    b.put(b.types, Op_TypePointer, {primitive_pointer, SC_Input, b.t_i32});
    b.put(b.types, Op_Variable, {primitive_pointer, primitive_var, SC_Input});
    b.put(b.deco, Op_Decorate, {primitive_var, Dec_BuiltIn, 7u});   // PrimitiveId
    b.put(b.deco, Op_Decorate, {primitive_var, Dec_Flat});
    b.iface.push_back(primitive_var);
    b.put(b.code, Op_Load, {b.t_i32, signed_primitive, primitive_var});
    const uint32_t primitive = b.i2u(signed_primitive);
    const uint32_t x = b.fragcoord_component(0u), y = b.fragcoord_component(1u);
    const uint32_t array = b.id(), block = b.id(), pointer = b.id(), storage = b.id();
    b.t_ptr_sb_u32 = b.id();
    b.put(b.deco, Op_Decorate, {array, Dec_ArrayStride, 4u});
    b.put(b.deco, Op_MemberDecorate, {block, 0, Dec_Offset, 0});
    b.put(b.deco, Op_Decorate, {block, Dec_Block});
    b.put(b.deco, Op_Decorate, {storage, Dec_DescriptorSet, 1u});
    // Set1/binding0 is reserved for the renderer's persistent internal GDS buffer.
    b.put(b.deco, Op_Decorate, {storage, Dec_Binding, 1u});
    b.put(b.deco, Op_Decorate, {storage, 24u});   // NonWritable
    b.put(b.types, Op_TypeRuntimeArray, {array, b.t_u32});
    b.put(b.types, Op_TypeStruct, {block, array});
    b.put(b.types, Op_TypePointer, {pointer, SC_StorageBuffer, block});
    b.put(b.types, Op_TypePointer, {b.t_ptr_sb_u32, SC_StorageBuffer, b.t_u32});
    b.declare_external_storage_buffer(pointer, storage);
    b.cbuf_var.emplace(0u, storage);
    const uint32_t entry = b.cur_block;
    const uint32_t loop = b.id(), body = b.id(), cont = b.id(), done = b.id();
    const uint32_t index = b.id(), next = b.id();
    b.emit_branch(loop);
    b.emit_label(loop);
    b.put(b.code, Op_Phi, {b.t_u32, index, b.uconst(0), entry, next, cont});
    const uint32_t room = b.ucmp(Op_ULessThan, index, b.uconst(records));
    b.put(b.code, Op_LoopMerge, {done, cont, 0u});
    b.put(b.code, Op_BranchConditional, {room, body, done});
    b.emit_label(body);
    const uint32_t base = b.ibin(Op_IMul, index, b.uconst(8u));
    const auto word = [&](uint32_t field) {
        return b.cbuf_load(b.ibin(Op_IAdd, base, b.uconst(field)), 0u);
    };
    const uint32_t a = b.ucmp(Op_IEqual, word(0u), primitive);
    const uint32_t c = b.ucmp(Op_IEqual, word(1u), x), d = b.ucmp(Op_IEqual, word(2u), y);
    const uint32_t ac = b.id(), match = b.id();
    b.put(b.code, Op_LogicalAnd, {b.t_bool, ac, a, c});
    b.put(b.code, Op_LogicalAnd, {b.t_bool, match, ac, d});
    b.emit_branch(cont);
    b.emit_label(cont);
    b.put(b.code, Op_IAdd, {b.t_u32, next, index, b.uconst(1u)});
    b.put(b.code, Op_BranchConditional, {match, done, loop});
    b.emit_label(done);
    const uint32_t found = b.ucmp(Op_ULessThan, index, b.uconst(records));
    const uint32_t inspect = b.id(), publish = b.id(), discard = b.id(), finish = b.id();
    const uint32_t inner_discard = b.id(), inner_finish = b.id();
    b.put(b.code, Op_SelectionMerge, {finish, 0u});
    b.put(b.code, Op_BranchConditional, {found, inspect, discard});
    b.emit_label(inspect);
    const uint32_t selected = b.ibin(Op_IMul, index, b.uconst(8u));
    const auto selected_word = [&](uint32_t field) {
        return b.cbuf_load(b.ibin(Op_IAdd, selected, b.uconst(field)), 0u);
    };
    const uint32_t enabled = b.ucmp(Op_IEqual, selected_word(3u), b.uconst(1u));
    // This inner selection must merge at finish, not reuse the surrounding loop's continue.
    b.put(b.code, Op_SelectionMerge, {inner_finish, 0u});
    b.put(b.code, Op_BranchConditional, {enabled, publish, inner_discard});
    b.emit_label(inner_discard);
    b.put(b.code, Op_Kill, {});
    b.emit_label(publish);
    b.export_color(0u, selected_word(4u), selected_word(5u), selected_word(6u), selected_word(7u));
    b.emit_branch(inner_finish);
    b.emit_label(inner_finish);
    b.emit_branch(finish);
    b.emit_label(discard);
    b.put(b.code, Op_Kill, {});
    b.emit_label(finish);
    return b.finish();
}
} // namespace prosper::gpu
