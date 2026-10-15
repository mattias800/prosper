// See rdna2_spilled_mask_halves.hpp.
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
#include "gpu/recompiler/rdna2_scalar_condition_marks.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"

#include <algorithm>
#include <cstdint>

namespace prosper::gpu {
namespace {

constexpr uint32_t kOpWritelane = 0x361;
constexpr uint32_t kOpReadlane = 0x360;

template <typename Map>
void keep_agreeing(Map& into, const Map& incoming) {
    for (auto it = into.begin(); it != into.end();) {
        const auto other = incoming.find(it->first);
        if (other == incoming.end() || !(other->second == it->second))
            it = into.erase(it);
        else
            ++it;
    }
}

template <typename Map>
void erase_instance(Map& facts, uint32_t def_pc) {
    for (auto it = facts.begin(); it != facts.end();) {
        if (it->second.def_pc == def_pc)
            it = facts.erase(it);
        else
            ++it;
    }
}

void erase_vgpr_slots(SpilledMaskHalves& state, int vgpr) {
    for (auto it = state.slots.begin(); it != state.slots.end();) {
        if (it->first.first == vgpr)
            it = state.slots.erase(it);
        else
            ++it;
    }
}

bool constant_lane(const Rdna2Inst& in) {
    return in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 &&
           in.src[1].value <= 63;
}

}  // namespace

void meet_spilled_mask_halves(SpilledMaskHalves& into, const SpilledMaskHalves& incoming) {
    keep_agreeing(into.pair_def_pc, incoming.pair_def_pc);
    keep_agreeing(into.slots, incoming.slots);
    keep_agreeing(into.sregs, incoming.sregs);
    std::erase_if(into.defined_slots,
                  [&](const auto& slot) { return !incoming.defined_slots.contains(slot); });
    into.relayed_words.insert(incoming.relayed_words.begin(), incoming.relayed_words.end());
    into.lane_written_vgprs.insert(incoming.lane_written_vgprs.begin(),
                                   incoming.lane_written_vgprs.end());
    into.relayed_vectors.insert(incoming.relayed_vectors.begin(), incoming.relayed_vectors.end());
    into.memory_vectors.insert(incoming.memory_vectors.begin(), incoming.memory_vectors.end());
    into.memory_words.insert(incoming.memory_words.begin(), incoming.memory_words.end());
    into.memory_slots.insert(incoming.memory_slots.begin(), incoming.memory_slots.end());
    meet_active_vector_words(into.active_vectors, incoming.active_vectors);
    into.exec_known_full = into.exec_known_full && incoming.exec_known_full;
    into.native_interpolation = into.native_interpolation && incoming.native_interpolation;
}

namespace {

// Whether `in` reads a relayed word: the same inputs scalar_source_marks() charges in the emitter
// (scalar operands, a read-modify-write destination, SCC), on the state BEFORE `in`. SMEM's
// address words are not data.
bool reads_relayed_word(const std::set<int>& relayed, const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::SMEM) return false;
    const auto any = [&](int first, uint32_t width) {
        for (uint32_t w = 0; w < width; ++w)
            if (relayed.contains(first + static_cast<int>(w))) return true;
        return false;
    };
    for (uint32_t k = 0; k < in.n_src && k < 3; ++k) {
        const Operand& o = in.src[k];
        const bool scalar = o.kind == OperandKind::SGPR ||
                            (o.kind == OperandKind::Special && o.value >= 106 && o.value <= 124);
        if (!scalar) continue;
        uint32_t width = scalar_alu_source_words(in, k);
        if (width == 0 || width == UINT32_MAX) width = 2;   // unknown: both words of a pair
        if (width > 2) width = static_cast<uint32_t>(std::max(0, 125 - o.value));   // a range
        if (any(o.value, width)) return true;
    }
    if (any(in.dst.value, scalar_implicit_destination_read_width(in))) return true;
    return scalar_reads_scc(in) && relayed.contains(kRelayedScc);
}

}  // namespace

SpilledVectorSourceMarks spilled_vector_source_marks(const SpilledMaskHalves& state,
                                                     const Rdna2Inst& in,
                                                     const std::set<int>& masks,
                                                     const std::set<int>& scalar_words,
                                                     bool definite_scc, uint32_t wave_size) {
    if (vector_write_leaves_old_word(in, state.native_interpolation) ||
        (state.native_interpolation && in.fmt == Rdna2Format::VINTRP))
        return {};
    const auto& defined = state.defined_slots;
    const auto& relayed = state.relayed_words;
    const auto vector_any = [&](const std::set<int>& marked) {
        for (uint32_t source = 0; source < in.n_src; ++source)
            for (uint32_t word = 0; word < vector_provenance_source_span(in, source); ++word)
                if (marked.contains(in.src[source].value + static_cast<int>(word))) return true;
        if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43)
            return std::any_of(marked.begin(), marked.end(),
                               [&](int reg) { return reg >= in.src[0].value; });
        return false;
    };
    const bool load = in.fmt == Rdna2Format::SMEM || in.fmt == Rdna2Format::MUBUF ||
                      in.fmt == Rdna2Format::MTBUF || in.fmt == Rdna2Format::MIMG ||
                      in.fmt == Rdna2Format::FLAT || in.fmt == Rdna2Format::DS;
    // An absent incoming SCC has no MAY relay entry yet. Use the pre-instruction MUST fact,
    // so a numeric copy or lane selector cannot become genuine merely at a later CFG edge.
    const bool missing_scc = !load && scalar_reads_scc(in) && !definite_scc;
    bool vector_relay =
        missing_scc || ((!load || in.fmt == Rdna2Format::DS) &&
                        (vector_any(state.relayed_vectors) || reads_relayed_word(relayed, in)));
    // A load overwrites its old DATA generation. Preserve unavailable current address inputs
    // across case edges too; neither a clean old destination nor memory storage authenticates it.
    if (load && (in.fmt == Rdna2Format::FLAT || in.fmt == Rdna2Format::MUBUF ||
                 in.fmt == Rdna2Format::MTBUF)) {
        const auto available = [&](int word) {
            return !state.relayed_vectors.contains(word) ||
                   (state.active_vectors.enabled && state.active_vectors.current.contains(word));
        };
        if (in.fmt == Rdna2Format::FLAT && in.src[0].kind == OperandKind::VGPR) {
            vector_relay =
                vector_relay || !available(in.src[0].value) || !available(in.src[0].value + 1);
        } else if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) {
            const uint32_t words = ((in.literal >> 12) & 1u) + ((in.literal >> 13) & 1u);
            for (uint32_t word = 0; word < words; ++word)
                vector_relay = vector_relay || !available(in.src[0].value + static_cast<int>(word));
            const auto& offset = in.src[2];
            const bool scalar_word =
                offset.kind == OperandKind::SGPR ||
                (offset.kind == OperandKind::Special && offset.value >= 106 && offset.value <= 124);
            if (scalar_word) {
                // Retain an architectural VCC mask beside DATA facts; actual numeric ballot
                // materialization still requires the emitter's exact native-wave contract.
                const bool vcc_mask_word = offset.kind == OperandKind::Special &&
                                           offset.value <= 107 && masks.contains(106) &&
                                           !scalar_words.contains(offset.value);
                vector_relay = vector_relay || relayed.contains(offset.value) ||
                               (!scalar_words.contains(offset.value) && !vcc_mask_word);
            } else if (offset.kind == OperandKind::Special && offset.value == 253) {
                vector_relay = vector_relay || !definite_scc;
            }
        }
    }
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 0x43) vector_relay = true;
    bool memory =
        load || vector_any(state.memory_vectors) || reads_relayed_word(state.memory_words, in);
    // An ordinary absent SGPR is fabricated before any instruction has relayed it. Retain
    // that consumed scalar dependency separately from whole-vector marks, so a defined
    // constant spill slot can stay genuine beside an unrelated marked slot.
    bool missing_scalar_word = false;
    if (!load)
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const auto& operand = in.src[source];
            const bool scalar =
                operand.kind == OperandKind::SGPR || (operand.kind == OperandKind::Special &&
                                                      operand.value >= 106 && operand.value <= 124);
            if (!scalar) continue;
            uint32_t width = scalar_alu_source_words(in, source);
            if (!width || width == UINT32_MAX) width = 2;
            if (width > 2) width = static_cast<uint32_t>(std::max(0, 125 - operand.value));
            for (uint32_t word = 0; word < width; ++word) {
                const int reg = operand.value + static_cast<int>(word);
                if (reg <= (wave_size == 32 ? 124 : 105) && !scalar_words.contains(reg) &&
                    !masks.contains(reg) && !(reg > 0 && masks.contains(reg - 1)))
                    missing_scalar_word = true;
            }
        }
    for (uint32_t word = 0; wave_size == 32 && word < scalar_implicit_destination_read_width(in);
         ++word) {
        const int reg = in.dst.value + static_cast<int>(word);
        if (!scalar_words.contains(reg) && !masks.contains(reg) &&
            !(reg > 0 && masks.contains(reg - 1)))
            missing_scalar_word = true;
    }
    vector_relay = vector_relay || missing_scalar_word;
    // Relayed words first, while `defined` still describes the slots `in` reads. A V_READLANE
    // relays unless it reads a constant-lane slot in `defined`; any other scalar write relays when
    // one of its inputs does. A compare is the only scalar writer certain to rewrite SCC.
    const bool readlane = in.fmt == Rdna2Format::VOP3 && in.opcode == kOpReadlane;
    const bool relays =
        missing_scc || missing_scalar_word || reads_relayed_word(relayed, in) ||
        (readlane ? state.lane_written_vgprs.contains(in.src[0].value) &&
                            !(constant_lane(in) &&
                              defined.contains({in.src[0].value, in.src[1].value})) ||
                        (!state.lane_written_vgprs.contains(in.src[0].value) &&
                         vector_any(state.relayed_vectors))
                  : vector_any(state.relayed_vectors));
    if (readlane && constant_lane(in) && state.lane_written_vgprs.contains(in.src[0].value))
        memory = state.memory_slots.contains({in.src[0].value, in.src[1].value});
    const bool active_defined =
        state.active_vectors.enabled &&
        active_vector_sources_defined(state.active_vectors, in, state.relayed_vectors, masks,
                                      scalar_words, relayed, definite_scc);
    const bool active_complete = active_defined && (vector_write_completes_active_word(in) ||
                                                    vector_read_completes_active_word(in));
    return {vector_relay, relays, memory, active_defined, active_complete};
}

void advance_defined_slots(SpilledMaskHalves& state, const Rdna2Inst& in,
                           const std::set<int>& masks, const std::set<int>& scalar_words,
                           const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                           const std::vector<int>& vector_writes,
                           const SpilledVectorSourceMarks& source, bool definite_scc,
                           uint32_t wave_size) {
    if (vector_write_leaves_old_word(in, state.native_interpolation)) return;
    auto& defined = state.defined_slots;
    auto& relayed = state.relayed_words;
    const bool readlane = in.fmt == Rdna2Format::VOP3 && in.opcode == kOpReadlane;
    const bool vector_relay = source.vector_relay;
    const bool relays = source.scalar_relay;
    const bool memory = source.memory;
    const bool compare =
        in.fmt == Rdna2Format::VOPC ||
        (in.fmt == Rdna2Format::VOP3 && (in.opcode < 0x100 || vop3_writes_mask_sdst(in)));
    for (const auto& [base, width] : scalar_writes)
        for (uint32_t word = 0; word < width; ++word) {
            const int reg = base + static_cast<int>(word);
            if (relays && !compare)
                relayed.insert(reg);
            else
                relayed.erase(reg);
            if (memory && !compare && !(masks.contains(base) && !scalar_words.contains(reg)))
                state.memory_words.insert(reg);
            else
                state.memory_words.erase(reg);
        }
    const bool scalar_alu = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                            in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SOPC;
    // CONFIDENCE: HIGH -- an instruction which leaves SCC unchanged cannot transfer its data
    // dependency into that condition. A genuine SCC writer instead replaces an earlier taint;
    // preserving instructions never repair it. Both wave widths use the same ISA effect inventory.
    if (scalar_instruction_writes_scc(in)) {
        if (relays)
            relayed.insert(kRelayedScc);
        else
            relayed.erase(kRelayedScc);
    } else if (!scalar_instruction_preserves_scc(in) && relays && (wave_size == 32 || scalar_alu)) {
        relayed.insert(kRelayedScc);
    }
    const bool preserve = vector_write_depends_on_old_word(in, state.exec_known_full);
    for (int vgpr : vector_writes) {
        // CONFIDENCE: HIGH -- a complete write whose every input is genuine in the current EXEC
        // lanes (active_vectors) makes those lanes genuine; the lanes outside EXEC keep the old
        // word, so the result may hold a relayed value only where the old word could. Without
        // this, a relay MAY fact on a source that is genuine wherever it is read (a word last
        // completed under a narrower EXEC) taints a destination whose old lanes are genuine, and
        // the dispatcher reloads it as a placeholder in every lane (#5173).
        if (vector_relay && !source.active_complete)
            state.relayed_vectors.insert(vgpr);
        else if (!preserve)
            state.relayed_vectors.erase(vgpr);
        if (memory)
            state.memory_vectors.insert(vgpr);
        else if (!preserve)
            state.memory_vectors.erase(vgpr);
    }
    if (rdna2_instruction_may_change_exec(in))
        state.exec_known_full =
            vector_exec_is_full_after_write(in, state.exec_known_full, wave_size);
    const auto drop_vgpr = [&](int vgpr) {
        std::erase_if(defined, [&](const auto& slot) { return slot.first == vgpr; });
    };
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == kOpWritelane) {
        state.lane_written_vgprs.insert(in.dst.value);
        if (!constant_lane(in)) return drop_vgpr(in.dst.value);
        const Operand& source = in.src[0];
        const int reg = source.value;
        const bool constant = source.kind == OperandKind::InlineInt ||
                              source.kind == OperandKind::InlineFloat ||
                              source.kind == OperandKind::Literal;
        const bool is_sgpr = source.kind == OperandKind::SGPR;
        const bool is_exec = source.kind == OperandKind::Special && (reg == 126 || reg == 127);
        const bool pair_mask = is_sgpr && (masks.contains(reg) || masks.contains(reg - 1));
        const bool defined_word = (is_sgpr || source.kind == OperandKind::Special) &&
                                  scalar_words.contains(reg) && !relayed.contains(reg);
        // SCC is a numeric 0/1 source, but is not stored in the SGPR MUST-word set. Its
        // incoming condition must be genuine on every path before this exact slot is defined.
        const bool scc_word = source.kind == OperandKind::Special && reg == 253 && definite_scc;
        const std::pair<int, int> slot{in.dst.value, in.src[1].value};
        if (memory)
            state.memory_slots.insert(slot);
        else
            state.memory_slots.erase(slot);
        if (constant || is_exec || pair_mask || defined_word || scc_word)
            defined.insert(slot);
        else
            defined.erase(slot);
    } else if (!readlane) {
        for (int vgpr : vector_writes) {
            drop_vgpr(vgpr);
            std::erase_if(state.memory_slots, [&](const auto& slot) { return slot.first == vgpr; });
        }
    }
}

void mark_relayed_words(const SpilledMaskHalves& entry, RegState& rs) {
    rs.vector_full_exec = entry.exec_known_full;
    rs.vector_native_interpolation = entry.native_interpolation;
    for (int word : entry.relayed_words)
        if (word == kRelayedScc)
            rs.scc_merge_placeholder = true;
        else
            rs.sreg_merge_placeholder.insert(word);
    for (int reg : entry.relayed_vectors) {
        auto& mark = rs.vreg_source_marks[reg];
        mark.placeholder = mark.hard = true;
    }
    for (int reg : entry.memory_vectors) rs.vreg_source_marks[reg].memory = true;
    rs.sreg_memory_pattern.insert(entry.memory_words.begin(), entry.memory_words.end());
    rs.lane_slot_memory_pattern.insert(entry.memory_slots.begin(), entry.memory_slots.end());
}

bool reassembles_spilled_mask_pair(const SpilledMaskHalves& state, const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.opcode != 0x04 || in.src[0].kind != OperandKind::SGPR)
        return false;
    const auto low = state.sregs.find(in.src[0].value);
    const auto high = state.sregs.find(in.src[0].value + 1);
    return low != state.sregs.end() && high != state.sregs.end() && low->second.half == 0 &&
           high->second.half == 1 && low->second.pair == high->second.pair &&
           low->second.def_pc == high->second.def_pc;
}

int spilled_slot_half(const SpilledMaskHalves& state, const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::VOP3 || (in.opcode != kOpWritelane && in.opcode != kOpReadlane) ||
        !constant_lane(in))
        return -1;
    const int vgpr = in.opcode == kOpWritelane ? in.dst.value : in.src[0].value;
    const auto slot = state.slots.find({vgpr, in.src[1].value});
    return slot == state.slots.end() ? -1 : slot->second.half;
}

namespace {

void carry_spilled_mask_words(SpilledMaskHalves& state, const Rdna2Inst& in) {
    // CONFIDENCE: HIGH -- `slots`/`sregs` are MUST facts naming the instance (defining pc) whose
    // half a slot or SGPR holds; every vector write to the slot VGPR, scalar write to the SGPR and
    // redefinition at that pc ends them. Two reloaded halves of one instance therefore hold its
    // value, and ActiveVectorWords.instances holds words genuine wherever that value enables a
    // lane: a destroying vector write leaves it, an EXEC change never could (the words are
    // relative to the mask, not to the EXEC current when it was saved or spilled).
    ActiveVectorWords& active = state.active_vectors;
    if (!active.enabled || in.fmt != Rdna2Format::VOP3 || !constant_lane(in)) return;
    if (in.opcode == kOpWritelane) {
        const auto slot = state.slots.find({in.dst.value, in.src[1].value});
        if (slot == state.slots.end()) return;
        const auto words = active.masks.find(slot->second.pair);
        if (words != active.masks.end())
            active.instances[slot->second.def_pc] = words->second;
        else
            active.instances.erase(slot->second.def_pc);
        return;
    }
    // V_READLANE's destination field always names a scalar register (SGPRs, then VCC at 106).
    if (in.opcode != kOpReadlane || in.dst.value < 0 || in.dst.value > 107) return;
    const int pair = in.dst.value & ~1;
    const auto low = state.sregs.find(pair);
    const auto high = state.sregs.find(pair + 1);
    if (low == state.sregs.end() || high == state.sregs.end() || low->second.half != 0 ||
        high->second.half != 1 || low->second.pair != high->second.pair ||
        low->second.def_pc != high->second.def_pc)
        return;
    const auto words = active.instances.find(low->second.def_pc);
    if (words != active.instances.end()) active.masks[pair] = words->second;
}

}   // namespace

void advance_spilled_mask_halves(SpilledMaskHalves& state, const Rdna2Inst& in,
                                 const std::set<int>& masks, const std::set<int>& mask_slot_keys,
                                 const std::vector<std::pair<int, uint32_t>>& scalar_writes,
                                 const std::vector<int>& vector_writes, int mask_write) {
    if (vector_write_leaves_old_word(in, state.native_interpolation)) return;
    const bool writelane = in.fmt == Rdna2Format::VOP3 && in.opcode == kOpWritelane;
    const bool readlane = in.fmt == Rdna2Format::VOP3 && in.opcode == kOpReadlane;

    // Vector side. A constant-lane V_WRITELANE replaces exactly one slot; a dynamic lane may
    // replace any slot of that VGPR; every other vector write ends the whole VGPR's slots.
    if (writelane) {
        if (!constant_lane(in)) {
            erase_vgpr_slots(state, in.dst.value);
        } else {
            const std::pair<int, int> key{in.dst.value, in.src[1].value};
            SpilledMaskHalf fact;
            if (in.src[0].kind == OperandKind::SGPR) {
                const int source = in.src[0].value;
                // The low word of a live MUST-mask pair, or the high word of one. The pair must
                // also be a key the dispatcher persists as a mask slot, or emit_alu's Bool
                // would not survive the block edge this fact is about.
                for (int half = 0; half < 2 && fact.pair < 0; ++half) {
                    const int pair = source - half;
                    const auto def = state.pair_def_pc.find(pair);
                    if (pair >= 0 && pair <= 104 && masks.contains(pair) &&
                        mask_slot_keys.contains(pair) && def != state.pair_def_pc.end())
                        fact = {pair, half, def->second};
                }
            }
            if (fact.pair >= 0)
                state.slots[key] = fact;
            else
                state.slots.erase(key);
        }
    } else if (!readlane) {
        for (int vgpr : vector_writes) erase_vgpr_slots(state, vgpr);
    }

    // Scalar side. Any write of a word ends its reloaded-half fact and the definition of every
    // pair that word belongs to.
    for (const auto& [base, width] : scalar_writes)
        for (uint32_t word = 0; word < width; ++word) {
            const int reg = base + static_cast<int>(word);
            state.sregs.erase(reg);
            state.pair_def_pc.erase(reg);
            state.pair_def_pc.erase(reg - 1);
        }
    if (readlane && constant_lane(in)) {
        const auto slot = state.slots.find({in.src[0].value, in.src[1].value});
        if (slot != state.slots.end()) state.sregs[in.dst.value] = slot->second;
    }

    // A new definition at this pc is a new instance: facts about the previous one must not pair
    // with facts about this one.
    if (mask_write >= 0 && mask_write <= 104) {
        erase_instance(state.slots, in.pc);
        erase_instance(state.sregs, in.pc);
        state.pair_def_pc[mask_write] = in.pc;
    }
    carry_spilled_mask_words(state, in);
}

}   // namespace prosper::gpu
