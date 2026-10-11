#include "gpu/recompiler/rdna2_portable_mask_slots.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>

namespace prosper::gpu {
namespace {
using Types = PortableMaskHalfTypes;
using Slot = std::pair<int, int>;
bool static_lane(const Rdna2Inst& in) {
    return in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 && in.src[1].value < 64;
}
bool conditional_write(const Rdna2Inst& in) {
    return (in.fmt == Rdna2Format::SOP1 &&
            (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64)) ||
           (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCmovkI32);
}
int source_half(const Types& types, const Operand& source) {
    if (source.kind == OperandKind::Special) {
        if (source.value == 126) return 0;
        if (source.value == 127) return 1;
    }
    if (source.kind != OperandKind::SGPR && source.kind != OperandKind::Special) return -1;
    if (const auto found = types.words.find(source.value); found != types.words.end())
        return static_cast<int>(found->second);
    if (types.may_words.contains(source.value)) return -1;
    if (types.complete.contains(source.value)) return 0;
    if (types.complete.contains(source.value - 1) && !types.numeric_words.contains(source.value))
        return 1;
    return -1;
}
bool complete_source(const Types& types, const Operand& source) {
    if (source.kind == OperandKind::InlineInt || source.kind == OperandKind::Literal) return true;
    if (source.kind == OperandKind::Special && (source.value == 126 || source.value == 127))
        return true;
    if (source.kind != OperandKind::SGPR && source.kind != OperandKind::Special) return false;
    const auto low = types.words.find(source.value), high = types.words.find(source.value + 1);
    if (types.may_words.contains(source.value) || types.may_words.contains(source.value + 1))
        return low != types.words.end() && high != types.words.end() && low->second == 0 &&
               high->second == 1;
    return types.complete.contains(source.value);
}
bool mask_consumer(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::SOP1)
        return in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x08 || in.opcode == 0x0a ||
               sop1_opcode_is_saveexec_b64(in.opcode);
    return in.fmt == Rdna2Format::SOP2 &&
           (in.opcode == 0x0b || (in.opcode >= 0x0f && in.opcode <= 0x1b && (in.opcode & 1u)));
}
void kill_word(Types& types, int word, bool conditional) {
    types.words.erase(word);
    types.numeric_words.erase(word);
    if (!conditional) types.may_words.erase(word);
    types.complete.erase(word);
    types.complete.erase(word - 1);
}
void kill_slots(Types& types, int vgpr) {
    std::erase_if(types.slots, [&](const auto& item) { return item.first.first == vgpr; });
    std::erase_if(types.may_slots, [&](const auto& key) { return key.first == vgpr; });
}
void transfer(Types& types, const Rdna2Inst& in, int max_vgpr) {
    const Types old = types; // sources precede every overlapping architectural destination
    for_each_scalar_write(
        in,
        [&](int base, uint32_t width) {
            for (uint32_t k = 0; k < width; ++k)
                kill_word(types, base + static_cast<int>(k), conditional_write(in));
        },
        false);
    // S_MOV_B64 can emit a complete Bool and genuine DATA together. Shipping WRITELANE reads
    // a saved pair's HIGH word through DATA; preserve that domain instead of replacing it with
    // a predicate-only slot. Track only the existing closed MOV producers, with MUST joins.
    if (in.fmt == Rdna2Format::SOP1 &&
        (in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64) &&
        (in.dst.kind == OperandKind::SGPR ||
         (in.dst.kind == OperandKind::Special && (in.dst.value == 106 || in.dst.value == 107)))) {
        const uint32_t width = in.opcode == kSop1OpcodeMovB64 ? 2u : 1u;
        for (uint32_t k = 0; k < width; ++k) {
            const int source = in.src[0].value + static_cast<int>(k);
            if (in.src[0].kind == OperandKind::InlineInt ||
                in.src[0].kind == OperandKind::Literal ||
                ((in.src[0].kind == OperandKind::SGPR || in.src[0].kind == OperandKind::Special) &&
                 old.numeric_words.contains(source) && !old.may_words.contains(source)))
                types.numeric_words.insert(in.dst.value + static_cast<int>(k));
        }
    }
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 && static_lane(in)) {
        const Slot key{in.src[0].value, in.src[1].value};
        if (old.may_slots.contains(key)) types.may_words.insert(in.dst.value);
        if (const auto found = old.slots.find(key); found != old.slots.end())
            types.words[in.dst.value] = found->second;
        return;
    }
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
        in.src[0].kind == OperandKind::SGPR) {
        if (old.may_words.contains(in.src[0].value)) types.may_words.insert(in.dst.value);
        if (const auto found = old.words.find(in.src[0].value); found != old.words.end())
            types.words[in.dst.value] = found->second;
    }
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
        if (!static_lane(in)) {
            kill_slots(types, in.dst.value);
            return;
        }
        const Slot key{in.dst.value, in.src[1].value};
        types.slots.erase(key);
        types.may_slots.erase(key);
        const int half = source_half(old, in.src[0]);
        if (half >= 0) {
            types.slots[key] = static_cast<uint32_t>(half);
            types.may_slots.insert(key);
        } else if (old.may_words.contains(in.src[0].value))
            types.may_slots.insert(key);
        return;
    }
    for_each_possible_vector_write(in, max_vgpr, [&](int vgpr) { kill_slots(types, vgpr); });
    // Retain only writers whose actual mask lowering establishes a complete predicate. Numeric
    // pair writers are not inferred to be masks from a B64 opcode alone.
    if (in.fmt == Rdna2Format::SOP1 && sop1_opcode_is_saveexec_b64(in.opcode))
        types.complete.insert(in.dst.value); // D receives OLD EXEC even when S0 is DATA
    else if (mask_consumer(in) &&
             !(in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
               in.src[0].kind == OperandKind::Literal) &&
             complete_source(old, in.src[0]) && (in.n_src < 2 || complete_source(old, in.src[1])))
        types.complete.insert(in.dst.value);
    else if (in.fmt == Rdna2Format::VOPC || in.fmt == Rdna2Format::VOP3)
        for_each_scalar_write(
            in,
            [&](int base, uint32_t width) {
                if (width == 2 && scalar_write_is_b64_mask(in, base)) types.complete.insert(base);
            },
            false);
}
void meet(Types& types, const Types& incoming) {
    std::erase_if(types.words, [&](const auto& item) {
        const auto other = incoming.words.find(item.first);
        return other == incoming.words.end() || other->second != item.second;
    });
    std::erase_if(types.slots, [&](const auto& item) {
        const auto other = incoming.slots.find(item.first);
        return other == incoming.slots.end() || other->second != item.second;
    });
    std::erase_if(types.complete, [&](int root) { return !incoming.complete.contains(root); });
    std::erase_if(types.numeric_words,
                  [&](int word) { return !incoming.numeric_words.contains(word); });
    types.may_words.insert(incoming.may_words.begin(), incoming.may_words.end());
    types.may_slots.insert(incoming.may_slots.begin(), incoming.may_slots.end());
}
std::shared_ptr<PortableMaskHalfProgram> analyze(const RegState& rs,
                                                 const std::vector<Rdna2Inst>& ins) {
    auto program = std::make_shared<PortableMaskHalfProgram>();
    if (ins.empty()) return program;
    const int max_vgpr = shader_max_vgpr(ins);
    std::map<uint32_t, size_t> indices;
    for (size_t i = 0; i < ins.size(); ++i) indices.emplace(ins[i].pc, i);
    Types entry = rs.portable_mask_half_types;
    std::erase_if(entry.numeric_words, [&](int word) {
        return rs.sreg_merge_placeholder.contains(word) || rs.sreg_memory_pattern.contains(word);
    });
    if (rs.scalar_presence_has_no_placeholders) {
        const auto seed_data = [&](const auto& words) {
            for (const auto& [word, value] : words)
                if (value && !rs.sreg_merge_placeholder.contains(word) &&
                    !rs.sreg_memory_pattern.contains(word) &&
                    !rs.unavailable_saved_mask_words.contains(word) &&
                    !entry.may_words.contains(word))
                    entry.numeric_words.insert(word);
        };
        seed_data(rs.sreg);
        seed_data(rs.sreg_input);
    }
    std::erase_if(entry.words, [&](const auto& item) {
        return rs.sreg_merge_placeholder.contains(item.first) ||
               rs.sreg_memory_pattern.contains(item.first);
    });
    std::erase_if(entry.slots, [&](const auto& item) {
        return rs.lane_slot_merge_placeholder.contains(item.first) ||
               rs.lane_slot_memory_pattern.contains(item.first);
    });
    for (const auto& [root, value] : rs.sreg_bool)
        if (value && !entry.may_words.contains(root) && !entry.may_words.contains(root + 1) &&
            !rs.sreg_bool_b32.contains(root) && !rs.sreg_merge_placeholder.contains(root) &&
            !rs.sreg_memory_pattern.contains(root))
            entry.complete.insert(root);
    if (rs.vcc && !rs.sreg.contains(106) && !rs.sreg.contains(107)) entry.complete.insert(106);
    std::vector<Types> incoming(ins.size());
    std::vector<bool> reachable(ins.size());
    incoming[0] = std::move(entry);
    reachable[0] = true;
    std::vector<size_t> pending{0};
    while (!pending.empty()) {
        const size_t i = pending.back();
        pending.pop_back();
        const auto& in = ins[i];
        Types next = incoming[i];
        transfer(next, in, max_vgpr);
        std::vector<size_t> successors;
        if (!in.is_end && !rdna2_escapes_decoded_effects(in)) {
            const bool branch =
                in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
            if (branch) {
                const uint32_t target =
                    static_cast<uint32_t>(static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16);
                if (const auto found = indices.find(target); found != indices.end())
                    successors.push_back(found->second);
            }
            if ((!branch || in.opcode != 0x02) && i + 1 < ins.size()) successors.push_back(i + 1);
        }
        for (size_t successor : successors) {
            if (!reachable[successor]) {
                reachable[successor] = true;
                incoming[successor] = next;
                pending.push_back(successor);
            } else {
                Types joined = incoming[successor];
                meet(joined, next);
                if (!(joined == incoming[successor])) {
                    incoming[successor] = std::move(joined);
                    pending.push_back(successor);
                }
            }
        }
    }
    for (size_t i = 0; i < ins.size(); ++i)
        if (reachable[i]) {
            program->before[ins[i].pc] = incoming[i];
            transfer(incoming[i], ins[i], max_vgpr);
            program->after[ins[i].pc] = std::move(incoming[i]);
        }
    return program;
}
bool marked_word(const RegState& rs, int word) {
    return rs.sreg_merge_placeholder.contains(word) || rs.sreg_memory_pattern.contains(word);
}
uint32_t pair_predicate(SpirvCompute& b, const RegState& rs, const Types& types, int root) {
    const auto low = types.words.find(root), high = types.words.find(root + 1);
    const auto lv = rs.sreg_bool.find(root), hv = rs.sreg_bool.find(root + 1);
    if (low == types.words.end() || high == types.words.end() || low->second != 0 ||
        high->second != 1 || lv == rs.sreg_bool.end() || hv == rs.sreg_bool.end() ||
        marked_word(rs, root) || marked_word(rs, root + 1))
        return 0;
    const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
    return b.bsel(b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)), hv->second, lv->second);
}
} // namespace

PortableMaskHalfBodyScope::PortableMaskHalfBodyScope(SpirvCompute& b, const RegState& rs,
                                                     const std::vector<Rdna2Inst>& ins)
    : b_(b), previous_(b.portable_mask_half_program) {
    const bool has_slots = !rs.portable_mask_half_types.may_words.empty() ||
                           !rs.portable_mask_half_types.may_slots.empty() ||
                           !rs.vgpr_lane_mask_slots.empty() ||
                           std::any_of(ins.begin(), ins.end(), [](const auto& in) {
                               return in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361;
                           });
    b.portable_mask_half_program =
        ordinary_portable_wave64_masks(b) && has_slots ? analyze(rs, ins) : nullptr;
}
PortableMaskHalfBodyScope::~PortableMaskHalfBodyScope() {
    b_.portable_mask_half_program = previous_;
}
void add_portable_mask_half_keys(const SpirvCompute& b, std::set<int>& keys) {
    if (b.portable_mask_half_program)
        for (const auto& [pc, types] : b.portable_mask_half_program->after)
            for (const auto& [word, role] : types.words) keys.insert(word);
}
bool portable_mask_half_at_entry(const SpirvCompute& b, uint32_t pc, int word) {
    if (!b.portable_mask_half_program) return false;
    const auto state = b.portable_mask_half_program->before.find(pc);
    return state != b.portable_mask_half_program->before.end() &&
           state->second.words.contains(word);
}
void load_portable_mask_half_types(const SpirvCompute& b, RegState& rs, uint32_t pc) {
    if (!b.portable_mask_half_program) return;
    const auto state = b.portable_mask_half_program->before.find(pc);
    rs.portable_mask_half_types =
        state == b.portable_mask_half_program->before.end() ? Types{} : state->second;
}
PortableMaskHalfInstruction::PortableMaskHalfInstruction(SpirvCompute& b, RegState& rs,
                                                         const Rdna2Inst& in, bool& ok)
    : rs_(rs) {
    if (!ordinary_portable_wave64_masks(b) || !b.portable_mask_half_program) return;
    const auto found = b.portable_mask_half_program->before.find(in.pc);
    if (found == b.portable_mask_half_program->before.end()) return;
    const Types& before = found->second;
    rs.portable_mask_half_types = b.portable_mask_half_program->after.at(in.pc);
    for_each_scalar_write(
        in,
        [&](int base, uint32_t width) {
            for (uint32_t k = 0; k < width; ++k) written_.insert(base + static_cast<int>(k));
        },
        false);
    auto reject = [&](const char* reason) {
        b.stage_reject_pc = in.pc;
        b.stage_reject_reason = reason;
        ok = false;
        handled_ = true;
    };
    if (rdna2_escapes_decoded_effects(in) &&
        (!before.may_words.empty() || !before.may_slots.empty())) {
        reject("portable-mask-half-unmodeled-effect");
        return;
    }
    auto publish_word = [&](int dst, uint32_t predicate) {
        rs.sreg.erase(dst);
        rs.sreg_srt.erase(dst);
        rs.sreg_bool[dst] = predicate;
        rs.sreg_bool_narrowed[dst] = true;
        rs.sreg_bool_b32.erase(dst);
    };
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360 && static_lane(in)) {
        const Slot key{in.src[0].value, in.src[1].value};
        const auto mask_slot = rs.vgpr_lane_mask_slots.find(key.first);
        if (mask_slot != rs.vgpr_lane_mask_slots.end() && mask_slot->second.contains(key.second) &&
            !before.may_slots.contains(key)) {
            reject("portable-mask-half-slot-unproven");
            return;
        }
        if (before.may_slots.contains(key)) {
            const auto role = before.slots.find(key);
            const auto vgpr = rs.vgpr_lane_mask_slots.find(key.first);
            if (role == before.slots.end() || vgpr == rs.vgpr_lane_mask_slots.end() ||
                !vgpr->second.contains(key.second) ||
                rs.lane_slot_merge_placeholder.contains(key) ||
                rs.lane_slot_memory_pattern.contains(key)) {
                reject("portable-mask-half-slot-unavailable");
                return;
            }
            publish_word(in.dst.value, vgpr->second.at(key.second));
            handled_ = true;
            return;
        }
    }
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 &&
        in.src[0].kind == OperandKind::SGPR && before.may_words.contains(in.src[0].value)) {
        const auto source = rs.sreg_bool.find(in.src[0].value);
        if (!before.words.contains(in.src[0].value) || source == rs.sreg_bool.end() ||
            marked_word(rs, in.src[0].value)) {
            reject("portable-mask-half-copy-unavailable");
            return;
        }
        const uint32_t value = source->second;
        publish_word(in.dst.value, value);
        handled_ = true;
        return;
    }
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361 && static_lane(in)) {
        const int half = source_half(before, in.src[0]);
        if (half >= 0) {
            uint32_t predicate = 0;
            if (in.src[0].kind == OperandKind::Special && in.src[0].value >= 126)
                predicate = rs.exec;
            else if (!marked_word(rs, in.src[0].value)) {
                const int root = before.words.contains(in.src[0].value) ? in.src[0].value
                                                                        : in.src[0].value - half;
                if (const auto source = rs.sreg_bool.find(root); source != rs.sreg_bool.end())
                    predicate = source->second;
                else if (root == 106 && !rs.sreg.contains(in.src[0].value))
                    predicate = rs.vcc;
            }
            if (!predicate) {
                reject("portable-mask-half-spill-unavailable");
                return;
            }
            const int vgpr = in.dst.value, lane = in.src[1].value;
            rs.vreg.erase(vgpr);
            rs.invalidated_vgpr_lane_slots.erase(vgpr);
            rs.vgpr_lane_mask_slots[vgpr][lane] = predicate;
            if (auto data = rs.vgpr_lane_slots.find(vgpr); data != rs.vgpr_lane_slots.end()) {
                data->second.erase(lane);
                if (data->second.empty()) rs.vgpr_lane_slots.erase(data);
            }
            handled_ = true;
            return;
        }
    }
    const uint32_t implicit = scalar_implicit_destination_read_width(in);
    for (uint32_t k = 0; k < implicit; ++k)
        if (before.may_words.contains(in.dst.value + static_cast<int>(k))) {
            reject("portable-mask-half-data-unavailable");
            return;
        }
    for (uint32_t i = 0; i < in.n_src; ++i) {
        const auto& source = in.src[i];
        if (source.kind != OperandKind::SGPR && source.kind != OperandKind::Special) continue;
        uint32_t width = scalar_alu_source_words(in, i);
        if (in.fmt == Rdna2Format::SMEM) width = i == 0 ? (in.opcode >= 8 ? 4u : 2u) : 1u;
        if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) width = i == 1 ? 4u : 1u;
        if (in.fmt == Rdna2Format::MIMG) width = i == 1 ? 8u : (i == 2 ? 4u : 1u);
        if (width == UINT32_MAX) continue;
        if (!width) width = 128; // an unknown footprint cannot launder a half as DATA
        bool half = false;
        for (uint32_t k = 0; k < width; ++k)
            half |= before.may_words.contains(source.value + static_cast<int>(k));
        if (!half) continue;
        if (width != 2 || !mask_consumer(in)) {
            reject("portable-mask-half-data-unavailable");
            return;
        }
        const uint32_t predicate = pair_predicate(b, rs, before, source.value);
        if (!predicate) {
            reject("portable-mask-half-pair-unavailable");
            return;
        }
        // Function-variable zeros and immutable launch inputs are not current physical words.
        // Symbolic mask halves carry only predicate projections; a B64 move must not copy an
        // older DATA spelling into its destination merely because that storage still exists.
        for (int word : {source.value, source.value + 1}) {
            rs.sreg.erase(word);
            rs.sreg_input.erase(word);
            rs.sreg_srt.erase(word);
        }
        restore_.emplace(source.value, rs.sreg_bool.at(source.value));
        rs.sreg_bool[source.value] = predicate;
    }
}
PortableMaskHalfInstruction::~PortableMaskHalfInstruction() {
    for (const auto& [word, value] : restore_)
        if (!written_.contains(word)) rs_.sreg_bool[word] = value;
}
}   // namespace prosper::gpu
