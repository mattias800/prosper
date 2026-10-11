#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <bitset>
#include <deque>

namespace prosper::gpu {
namespace {
bool ordinary_pair(const Operand& source) {
    return (source.kind == OperandKind::SGPR && source.value >= 0 && source.value <= 104) ||
           (source.kind == OperandKind::Special && (source.value == 106 || source.value == 107));
}

bool marked_source(const RegState& rs, const Operand& source) {
    if (!ordinary_pair(source)) return false;
    for (int word : {source.value, source.value + 1})
        if (rs.sreg_merge_placeholder.contains(word) || rs.sreg_memory_pattern.contains(word) ||
            rs.unavailable_saved_mask_words.contains(word))
            return true;
    return false;
}

std::array<uint32_t, 2> source_words(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                     const Operand& source) {
    if (source.kind == OperandKind::InlineInt)
        return {b.uconst(static_cast<uint32_t>(source.value)),
                b.uconst(source.value < 0 ? UINT32_MAX : 0u)};
    if (source.kind == OperandKind::Literal && in.has_literal)
        return {b.uconst(in.literal), b.uconst(0)};
    if (!ordinary_pair(source) || marked_source(rs, source)) return {};
    std::array<uint32_t, 2> result{};
    for (int half = 0; half < 2; ++half) {
        const int word = source.value + half;
        if (sreg_word_may_be_fabricated(rs, word)) return {};
        const auto current = rs.sreg.find(word);
        if (current != rs.sreg.end())
            result[half] = current->second;
        else {
            // A mask write ends the launch-input lifetime. Do not resurrect an older input if
            // an incomplete current mask happens to coexist with an entry map spelling.
            if (rs.sreg_bool.contains(source.value) ||
                ((source.value == 106 || source.value == 107) && rs.vcc))
                return {};
            const auto input = rs.sreg_input.find(word);
            if (input == rs.sreg_input.end()) return {};
            result[half] = input->second;
        }
    }
    return result;
}

uint32_t logical_word(SpirvCompute& b, uint32_t opcode, uint32_t a, uint32_t c) {
    const auto complement = [&](uint32_t value) {
        return b.ibin(Op_BitwiseXor, value, b.uconst(UINT32_MAX));
    };
    switch (opcode) {
        case 0x0f: return b.ibin(Op_BitwiseAnd, a, c);
        case 0x11: return b.ibin(Op_BitwiseOr, a, c);
        case 0x13: return b.ibin(Op_BitwiseXor, a, c);
        case 0x15: return b.ibin(Op_BitwiseAnd, a, complement(c));
        case 0x17: return b.ibin(Op_BitwiseOr, a, complement(c));
        case 0x19: return complement(b.ibin(Op_BitwiseAnd, a, c));
        case 0x1b: return complement(b.ibin(Op_BitwiseOr, a, c));
        case 0x1d: return complement(b.ibin(Op_BitwiseXor, a, c));
        default: return 0;
    }
}
}   // namespace

bool is_native_b64_logical(const SpirvCompute& b, const Rdna2Inst& in) {
    return b.is_compute && !b.is_fragment_packet() && b.wave_size == 64 &&
           b.native_subgroup_size == 64 && in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
           in.opcode <= 0x1d && (in.opcode & 1u) && in.dst.value >= 0 && in.dst.value <= 107;
}

void initialize_native_logical_data_demand(SpirvCompute& b,
                                           const std::vector<Rdna2Inst>& original) {
    // Backward MAY liveness follows physical word generations. A later use after a definite
    // overwrite cannot certify an earlier Bool-only logical, and either branch/loop successor
    // can demand the current generation. The finite lattice has 128 bits per instruction.
    using Words = std::bitset<128>;
    const size_t count = original.size();
    std::map<uint32_t, size_t> at_pc;
    for (size_t i = 0; i < count; ++i) at_pc.emplace(original[i].pc, i);
    std::vector<std::vector<size_t>> successors(count), predecessors(count);
    std::vector<Words> before(count), after(count), reads(count), kills(count);
    std::vector<bool> relays(count);
    const auto mark = [](Words& words, int base, uint32_t width) {
        for (uint32_t half = 0; half < width && half < 128; ++half) {
            const int word = base + static_cast<int>(half);
            if (word >= 0 && word < 128) words.set(word);
        }
    };
    for (size_t i = 0; i < count; ++i) {
        const auto& in = original[i];
        if (rdna2_escapes_decoded_effects(in) || in.fmt == Rdna2Format::Unknown) {
            // An unresolved control/read footprint supplies no deadness proof.
            reads[i].set();
        }
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
        if (branch) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !at_pc.contains(static_cast<uint32_t>(target)))
                reads[i].set();
            else
                successors[i].push_back(at_pc.at(static_cast<uint32_t>(target)));
        }
        if (!in.is_end && (!branch || in.opcode != 0x02) && i + 1 < count)
            successors[i].push_back(i + 1);
        for (size_t next : successors[i]) predecessors[next].push_back(i);
        // Conditional moves can preserve the old destination, so their writer spelling is
        // not a definite kill. All implicit sources are charged before writer retirement.
        const bool conditional =
            (in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x02);
        if (!conditional)
            for_each_scalar_write(in,
                                  [&](int base, uint32_t width) { mark(kills[i], base, width); });
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind != OperandKind::SGPR)
            mark(kills[i], 106, 2);
        relays[i] =
            is_native_b64_logical(b, in) ||
            (in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x08 || in.opcode == 0x0a ||
              (in.opcode >= kSop1OpcodeAndSaveexecB64 && in.opcode <= kSop1OpcodeXnorSaveexecB64)));
        if (relays[i]) continue;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const auto& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR &&
                !(operand.kind == OperandKind::Special && operand.value >= 106 &&
                  operand.value <= 127))
                continue;
            uint32_t width = scalar_alu_source_words(in, source);
            // The shared ALU table covers explicit arithmetic widths and dynamic SGPR ranges.
            // Memory operands are descriptors/addresses, not Bool masks. Unknown footprints
            // conservatively observe the remaining words rather than supplying a false kill.
            if (in.fmt == Rdna2Format::SMEM)
                width = source == 0 ? (in.opcode >= 0x08 ? 4u : 2u) : 1u;
            if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF)
                width = source == 1 ? 4u : 1u;
            if (in.fmt == Rdna2Format::MIMG) width = source == 1 ? 8u : (source == 2 ? 4u : 1u);
            if (width == UINT32_MAX) continue;   // the shared inventory says this field is not read
            if (!width) width = 128;
            mark(reads[i], operand.value, width);
        }
        mark(reads[i], in.dst.value, scalar_implicit_destination_read_width(in));
    }
    std::deque<size_t> pending;
    std::vector<bool> queued(count, true);
    for (size_t i = count; i-- > 0;) pending.push_back(i);
    while (!pending.empty()) {
        const size_t i = pending.front();
        pending.pop_front();
        queued[i] = false;
        Words out;
        for (size_t next : successors[i]) out |= before[next];
        after[i] = out;
        Words value = (out & ~kills[i]) | reads[i];
        const auto& in = original[i];
        if (relays[i] && (out & kills[i]).any())
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const auto& operand = in.src[source];
                if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special)
                    mark(value, operand.value, 2);
            }
        if (value == before[i]) continue;
        before[i] = value;
        for (size_t previous : predecessors[i])
            if (!queued[previous]) {
                queued[previous] = true;
                pending.push_back(previous);
            }
    }
    for (size_t i = 0; i < count; ++i)
        if (is_native_b64_logical(b, original[i]) && (after[i] & kills[i]).any())
            b.native_logical_data_demand_pcs.insert(original[i].pc);
}

void initialize_native_logical_uniform_entry_proof(SpirvCompute& b,
                                                   const std::vector<Rdna2Inst>& original) {
    if (!b.is_compute || b.native_subgroup_size != 64 || b.wave_size != 64) return;
    uint32_t reentry = UINT32_MAX;
    std::set<uint32_t> pcs;
    for (const auto& in : original) pcs.insert(in.pc);
    for (const auto& in : original) {
        if (in.fmt == Rdna2Format::Unknown || rdna2_escapes_decoded_effects(in)) return;
        if (in.fmt != Rdna2Format::SOPP ||
            !(in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09)))
            continue;
        const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
        if (target < 0 || target > UINT32_MAX || !pcs.contains(static_cast<uint32_t>(target)))
            return;
        if (target <= in.pc) reentry = std::min(reentry, static_cast<uint32_t>(target));
    }
    for (const auto& in : original) {
        if (in.pc >= reentry || in.is_end || in.fmt == Rdna2Format::EXP ||
            (rdna2_instruction_may_change_exec(in) && !exec_write_sets_full_mask(in)))
            break;
        if (in.fmt == Rdna2Format::SOPP) {
            if (in.opcode == 0x04 || in.opcode == 0x05) {
                if (!branch_is_workgroup_uniform(original, in.pc, UniformBranchCondition::Scc))
                    break;
            } else if (in.opcode == 0x06 || in.opcode == 0x07) {
                if (!branch_is_workgroup_uniform(original, in.pc, UniformBranchCondition::Vcc))
                    break;
            } else if (in.opcode != 0x00 && in.opcode != 0x02 && in.opcode != 0x0a &&
                       in.opcode != 0x0c && in.opcode != 0x20 && in.opcode != 0x21 &&
                       in.opcode != 0x22)
                break;
        }
        if (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeSetregB32) break;
        // Every preceding host case/selection is reached by either the entire launch workgroup
        // or none of it: each guest branch condition is proved launch-uniform by the shared
        // closed provenance walk. No initial partial/outer gate, reentry or EXEC mutation can
        // enter this prefix. This is not an SCC-spelling or subgroup-width assumption.
        if (is_native_b64_logical(b, in)) b.native_logical_uniform_entry_pcs.insert(in.pc);
    }
}

bool native_logical_producer_has_full_participation(const SpirvCompute& b, const Rdna2Inst& in) {
    if (saved_mask_producer_has_full_participation(b, in)) return true;
    if (!b.native_logical_uniform_entry_pcs.contains(in.pc) ||
        in.pc >= b.saved_mask_original_words.size())
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return !in.synthetic_terminator && in == original;
}

bool native_logical_data_demanded(const SpirvCompute& b, const Rdna2Inst& in) {
    if (!b.native_logical_data_demand_pcs.contains(in.pc) ||
        in.pc >= b.saved_mask_original_words.size())
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return !in.synthetic_terminator && in == original;
}

std::array<uint32_t, 2> native_logical_result_words(SpirvCompute& b, const RegState& rs,
                                                    const Rdna2Inst& in, uint32_t result_mask) {
    if (!is_native_b64_logical(b, in)) return {};
    // Read both complete current operands before any destination expires an overlapping source.
    // CONFIDENCE: HIGH -- AMD 70648 section12.1 defines B64 logic on all64 source bits. Retained
    // DATA therefore needs no rendezvous with departed lanes, even in a partial host region.
    const auto a = source_words(b, rs, in, in.src[0]);
    const auto c = source_words(b, rs, in, in.src[1]);
    if (a[0] && a[1] && c[0] && c[1])
        return {logical_word(b, in.opcode, a[0], c[0]), logical_word(b, in.opcode, a[1], c[1])};
    // A Bool-only result is not scalar DATA. The sole ballot fallback requires the original
    // complete producer's participation proof; width, wave_ok and a register number are not it.
    if (!native_logical_data_demanded(b, in) ||
        !native_logical_producer_has_full_participation(b, in) || marked_source(rs, in.src[0]) ||
        marked_source(rs, in.src[1]))
        return {};
    return {b.native_wave_ballot_half(result_mask, 0), b.native_wave_ballot_half(result_mask, 1)};
}

void record_native_logical_word_gaps(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    if (!is_native_b64_logical(b, in)) return;
    for (int word : {in.dst.value, in.dst.value + 1})
        if (!rs.sreg.contains(word) || sreg_word_may_be_fabricated(rs, word) ||
            rs.sreg_memory_pattern.contains(word))
            rs.unavailable_saved_mask_words.insert(word);
}
}   // namespace prosper::gpu
