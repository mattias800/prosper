#include "gpu/recompiler/rdna2_native_participation.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"

#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include <array>
#include <bitset>
#include <deque>
#include <optional>

namespace prosper::gpu {
namespace {
constexpr uint64_t full = UINT64_MAX;
struct Word {
    bool uniform = false, genuine = false;
    std::optional<uint32_t> constant;
    bool numeric_uniform = uniform;
    bool operator==(const Word&) const = default;
};
struct Mask {
    bool genuine = false;
    std::optional<uint64_t> constant;
    bool operator==(const Mask&) const = default;
};
struct Facts {
    bool common = true;
    bool numeric_common = true;
    std::array<Word, 128> scalar{};
    std::array<Mask, 128> mask{};
    std::array<uint64_t, 256> vector{};
    std::bitset<256> local_x;
    std::array<std::optional<uint32_t>, 256> vector_constant{};
    Word scc;
    Mask exec{true, full};
    // Numeric spill services return their stored SSA, rather than shuffling ordinary VGPR DATA.
    // Uniformity belongs to that current slot producer, including false/missing edge facts.
    std::map<std::pair<int, uint32_t>, bool> numeric_slots;
    std::set<int> numeric_slot_vgprs;
    bool operator==(const Facts&) const = default;
};
Word meet(Word a, Word b) {
    Word result{a.uniform && b.uniform, a.genuine && b.genuine,
                a.constant == b.constant ? a.constant : std::nullopt};
    result.numeric_uniform = a.numeric_uniform && b.numeric_uniform;
    return result;
}
Mask meet(Mask a, Mask b) {
    return {a.genuine && b.genuine, a.constant == b.constant ? a.constant : std::nullopt};
}
Facts meet(Facts a, const Facts& b) {
    a.common &= b.common;
    a.numeric_common &= b.numeric_common;
    a.scc = meet(a.scc, b.scc);
    a.exec = meet(a.exec, b.exec);
    for (size_t i = 0; i < a.scalar.size(); ++i) {
        a.scalar[i] = meet(a.scalar[i], b.scalar[i]);
        a.mask[i] = meet(a.mask[i], b.mask[i]);
    }
    for (size_t i = 0; i < a.vector.size(); ++i) {
        a.vector[i] &= b.vector[i];
        if (a.vector_constant[i] != b.vector_constant[i]) a.vector_constant[i].reset();
    }
    a.local_x &= b.local_x;
    for (const auto& [slot, uniform] : b.numeric_slots) {
        const auto found = a.numeric_slots.find(slot);
        if (found == a.numeric_slots.end())
            a.numeric_slots.emplace(slot, false);
        else
            found->second &= uniform;
    }
    for (auto& [slot, uniform] : a.numeric_slots)
        if (!b.numeric_slots.contains(slot)) uniform = false;
    a.numeric_slot_vgprs.insert(b.numeric_slot_vgprs.begin(), b.numeric_slot_vgprs.end());
    return a;
}
bool ordinary(const SpirvCompute& b) {
    return native_compute_has_complete_entry(b) && b.native_logical_uniform_launch &&
           b.is_compute && !b.is_fragment_packet() && !b.ngg_workgroup_shell && !b.ngg_one_lane &&
           b.wave_size == 64 && b.native_subgroup_size == 64 && b.local_count == 64;
}
bool scalar_word_available(const RegState& rs, int r) {
    if (r < 0 || r >= 124 || sreg_word_may_be_fabricated(rs, r) ||
        rs.sreg_memory_pattern.contains(r) || rs.sreg_entry_m0.contains(r))
        return false;
    const auto current = rs.sreg.find(r);
    const auto input = rs.sreg_input.find(r);
    if (current == rs.sreg.end() && (rs.sreg_bool.contains(r) || rs.sreg_bool.contains(r - 1) ||
                                     ((r == 106 || r == 107) && rs.vcc)))
        return false;
    const uint32_t value = current != rs.sreg.end()       ? current->second
                           : input != rs.sreg_input.end() ? input->second
                                                          : 0;
    return value && !rs.lane_local_scalars.contains(value);
}
bool vector_word_available(const RegState& rs, int r) {
    const auto found = rs.vreg.find(r);
    if (found == rs.vreg.end() || !found->second) return false;
    const auto marks = rs.vreg_source_marks.find(r);
    return marks == rs.vreg_source_marks.end() ||
           (!marks->second.placeholder && !marks->second.hard &&
            marks->second.blanket_roots.empty());
}
Word word(const Facts& f, const Rdna2Inst& in, Operand o, uint32_t offset = 0) {
    if (o.kind == OperandKind::InlineInt)
        return {true, true, offset ? (o.value < 0 ? UINT32_MAX : 0u) : uint32_t(o.value)};
    if (o.kind == OperandKind::Literal && in.has_literal)
        return {true, true, offset ? 0u : in.literal};
    if ((o.kind == OperandKind::SGPR || o.kind == OperandKind::Special) && o.value >= 0 &&
        o.value + int(offset) < 128)
        return f.scalar[static_cast<size_t>(o.value) + offset];
    return {};
}
Word combine(Word a, Word b) {
    Word result{a.uniform && b.uniform, a.genuine && b.genuine, std::nullopt};
    result.numeric_uniform = a.numeric_uniform && b.numeric_uniform;
    return result;
}
Word scalar_sources(const Facts& f, const Rdna2Inst& in) {
    Word result{true, true, std::nullopt};
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const uint32_t width = scalar_alu_source_words(in, k);
        if (!width || width > 4) return {};
        for (uint32_t w = 0; w < width; ++w) result = combine(result, word(f, in, in.src[k], w));
    }
    return result;
}
Mask mask(const Facts& f, const Rdna2Inst& in, Operand o) {
    if (o.kind == OperandKind::InlineInt) return {true, uint64_t(int64_t(o.value))};
    if (o.kind == OperandKind::Literal && in.has_literal) return {true, in.literal};
    if (o.kind != OperandKind::SGPR && o.kind != OperandKind::Special) return {};
    if (o.value == 126) return f.exec;
    if (o.value < 0 || o.value >= 127) return {};
    if (f.mask[o.value].genuine) return f.mask[o.value];
    const auto low = word(f, in, o), high = word(f, in, o, 1);
    if (!low.uniform || !low.genuine || !high.uniform || !high.genuine) return {};
    return {true, low.constant && high.constant
                      ? std::optional<uint64_t>{uint64_t(*low.constant) |
                                                (uint64_t(*high.constant) << 32u)}
                      : std::nullopt};
}
uint64_t vector_source(const Facts& f, const Rdna2Inst& in, Operand o, uint32_t span = 1) {
    if (o.kind == OperandKind::VGPR && o.value >= 0 && o.value + int(span) <= 256) {
        uint64_t result = full;
        for (uint32_t w = 0; w < span; ++w) result &= f.vector[o.value + w];
        return result;
    }
    Word s{true, true, std::nullopt};
    for (uint32_t w = 0; w < span; ++w) s = combine(s, word(f, in, o, w));
    return s.genuine ? full : 0;
}
uint64_t vector_sources(const Facts& f, const Rdna2Inst& in) {
    uint64_t result = full;
    const uint32_t n =
        in.fmt == Rdna2Format::VOP3
            ? std::min<uint32_t>(in.n_src, vop3_architectural_source_count(in.opcode))
            : in.n_src;
    for (uint32_t k = 0; k < n; ++k) {
        const uint32_t span = in.src[k].kind == OperandKind::VGPR ? rdna2_vgpr_source_span(in, k)
                                                                  : scalar_alu_source_words(in, k);
        if (!span || span > 4) return 0;
        result &= vector_source(f, in, in.src[k], span);
    }
    return result;
}
void kill_scalar(Facts& f, int r, uint32_t width) {
    for (uint32_t w = 0; w < width && r + int(w) < 128; ++w) {
        if (r + int(w) < 0) continue;
        const int at = r + int(w);
        f.scalar[at] = {};
        f.mask[at] = {};
        if (at) f.mask[at - 1] = {};
    }
}
void write_vector(Facts& f, const Rdna2Inst& in, int r, uint64_t genuine) {
    if (r < 0 || r >= 256) return;
    const bool partial = in.has_sdwa && in.sdwa_dst_sel != 6;
    if (partial) genuine &= f.vector[r];
    // Unknown EXEC may replace any lane or preserve it. A genuine new word AND a genuine old word
    // is sufficient; an EXEC-off lane never becomes initialized merely because VDST exists.
    f.vector[r] = f.exec.constant ? (f.vector[r] & ~*f.exec.constant) | (genuine & *f.exec.constant)
                                  : f.vector[r] & genuine;
    f.local_x.reset(r);
    f.vector_constant[r].reset();
}
std::optional<uint32_t> lane_value(const Facts& f, const SpirvCompute& b, const Rdna2Inst& in,
                                   Operand o, uint32_t lane) {
    if (o.kind != OperandKind::VGPR) return word(f, in, o).constant;
    if (o.value < 0 || o.value >= 256) return {};
    if (f.local_x[o.value]) return lane % b.guest_local_size[0];
    return f.vector_constant[o.value];
}
bool integer_compare(uint32_t opcode) {
    return (opcode >= 0x81 && opcode <= 0x86) || (opcode >= 0xc1 && opcode <= 0xc6);
}
Mask compare(const Facts& f, const SpirvCompute& b, const Rdna2Inst& in) {
    const uint32_t opcode = vopc_is_cmpx(in.opcode) ? in.opcode - 0x10u : in.opcode;
    if (!integer_compare(opcode) || in.has_dpp || in.has_modifier || in.sdwa_src0_sel != 6 ||
        in.sdwa_src1_sel != 6 || in.src_abs[0] || in.src_abs[1] || in.src_neg[0] || in.src_neg[1])
        return {};
    const uint64_t active = f.exec.constant.value_or(full);
    const bool genuine = f.exec.genuine && (vector_sources(f, in) & active) == active;
    Mask result{genuine, std::nullopt};
    if (!genuine || !f.exec.constant) return result;
    uint64_t bits = 0;
    for (uint32_t lane = 0; lane < 64; ++lane) {
        if (!(active & (uint64_t{1} << lane))) continue;
        const auto a = lane_value(f, b, in, in.src[0], lane);
        const auto c = lane_value(f, b, in, in.src[1], lane);
        if (!a || !c) return result;
        bool value = false;
        switch (opcode & 7u) {
            case 1: value = opcode < 0xc0 ? int32_t(*a) < int32_t(*c) : *a < *c; break;
            case 2: value = *a == *c; break;
            case 3: value = opcode < 0xc0 ? int32_t(*a) <= int32_t(*c) : *a <= *c; break;
            case 4: value = opcode < 0xc0 ? int32_t(*a) > int32_t(*c) : *a > *c; break;
            case 5: value = *a != *c; break;
            case 6: value = opcode < 0xc0 ? int32_t(*a) >= int32_t(*c) : *a >= *c; break;
        }
        if (value) bits |= uint64_t{1} << lane;
    }
    result.constant = bits;
    return result;
}
bool scalar_opcode(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::SOP1)
        return in.opcode == 3 || in.opcode == 4 || in.opcode == 7 || in.opcode == 8 ||
               in.opcode == 0xb || in.opcode == 0x15;
    if (in.fmt != Rdna2Format::SOP2) return false;
    return in.opcode <= 0xb || (in.opcode >= 0xe && in.opcode <= 0x26) || in.opcode == 0x27 ||
           in.opcode == 0x28;
}
Mask logical_mask(Mask a, Mask c, uint32_t opcode) {
    Mask result{a.genuine && c.genuine, std::nullopt};
    if (!a.constant || !c.constant) return result;
    switch (opcode) {
        case 0xf: result.constant = *a.constant & *c.constant; break;
        case 0x11: result.constant = *a.constant | *c.constant; break;
        case 0x13: result.constant = *a.constant ^ *c.constant; break;
        case 0x15: result.constant = *a.constant & ~*c.constant; break;
        default: break;
    }
    return result;
}
Mask saveexec_mask(Mask old_exec, Mask source, uint32_t opcode) {
    Mask result{old_exec.genuine && source.genuine, std::nullopt};
    if (!old_exec.constant || !source.constant) return result;
    const uint64_t e = *old_exec.constant, s = *source.constant;
    // CONFIDENCE: HIGH -- RDNA2 SOP1 SAVEEXEC reads S0 and OLD_EXEC before publishing either
    // destination. The complemented operand differs between ANDN/ORN1 and ANDN/ORN2.
    switch (opcode) {
        case 0x24: result.constant = s & e; break;
        case 0x25: result.constant = s | e; break;
        case 0x26: result.constant = s ^ e; break;
        case 0x27: result.constant = s & ~e; break;
        case 0x28: result.constant = s | ~e; break;
        case 0x29: result.constant = ~(s & e); break;
        case 0x2a: result.constant = ~(s | e); break;
        case 0x2b: result.constant = ~(s ^ e); break;
        case 0x37: result.constant = ~s & e; break;
        case 0x38: result.constant = ~s | e; break;
        default: return {};
    }
    return result;
}
Facts transfer(const Facts& before, const SpirvCompute& b, const Rdna2Inst& in) {
    Facts f = before;
    const int dst = in.dst.value;
    if (in.fmt == Rdna2Format::Unknown || rdna2_escapes_decoded_effects(in) ||
        in.fmt == Rdna2Format::EXP || in.fmt == Rdna2Format::SOPK)
        f.common = f.numeric_common = false;
    if (in.fmt == Rdna2Format::SOPP) {
        if (in.opcode != 0 && in.opcode != 1 && in.opcode != 2 &&
            !(in.opcode >= 4 && in.opcode <= 9) && in.opcode != 0xc && in.opcode != 0x20 &&
            in.opcode != 0x21 && in.opcode != 0x22)
            f.common = false;
        // A guest barrier ends the existing physical producer proof, not scalar uniformity.
        if (in.opcode != 0 && in.opcode != 1 && in.opcode != 2 &&
            !(in.opcode >= 4 && in.opcode <= 0xa) && in.opcode != 0xc && in.opcode != 0x20 &&
            in.opcode != 0x21 && in.opcode != 0x22)
            f.numeric_common = false;
        return f;
    }
    for_each_scalar_write(in, [&](int r, uint32_t width) { kill_scalar(f, r, width); });
    if (in.fmt == Rdna2Format::VOPC || (in.fmt == Rdna2Format::VOP3 && in.opcode < 0x100)) {
        const auto result = compare(before, b, in);
        if (vopc_is_cmpx(in.opcode))
            f.exec = result;
        else if (dst >= 0 && dst < 127) {
            // The explicit writer inventory excludes implicit VCC and VOP3 compare destinations.
            // A fresh Bool mask replaces physical DATA, including any overlapping old mask alias.
            kill_scalar(f, dst, 2);
            f.mask[dst] = result;
            // The new proof applies only to fresh complete ordinary-SGPR comparisons. Physical
            // words are retained by the existing producer hook, not by a later mask read/copy.
            if (in.fmt == Rdna2Format::VOPC && dst <= 104 && scalar_write_is_b64_mask(in, dst) &&
                before.common && result.genuine)
                for (int half = 0; half < 2; ++half)
                    f.scalar[dst + half] = {true, true,
                                            result.constant ? std::optional<uint32_t>{uint32_t(
                                                                  *result.constant >> (half * 32))}
                                                            : std::nullopt};
        }
        return f;
    }
    if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a)
        kill_scalar(f, 106, 2);
    if (in.fmt == Rdna2Format::SOPC) {
        if (in.opcode <= 0xb)
            f.scc = scalar_sources(before, in);
        else if (in.opcode == 0x12 || in.opcode == 0x13) {
            const auto a = mask(before, in, in.src[0]), c = mask(before, in, in.src[1]);
            f.scc = {a.genuine && c.genuine, a.genuine && c.genuine, std::nullopt};
        } else
            f.scc = {};
        return f;
    }
    if (in.fmt == Rdna2Format::SOP1 && sop1_opcode_is_saveexec_b64(in.opcode)) {
        const auto source = mask(before, in, in.src[0]);
        if (dst >= 0 && dst < 127) f.mask[dst] = before.exec;
        f.exec = saveexec_mask(before.exec, source, in.opcode);
        f.scc = {f.exec.genuine, f.exec.genuine, std::nullopt};
        return f;
    }
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == 4) {
        const auto source = mask(before, in, in.src[0]);
        if (dst == 126) {
            f.exec = source;
            return f;
        }
        if (dst >= 0 && dst < 127 && source.genuine) f.mask[dst] = source;
    }
    if (scalar_opcode(in)) {
        auto result = scalar_sources(before, in);
        if (in.fmt == Rdna2Format::SOP2 &&
            (in.opcode == 4 || in.opcode == 5 || in.opcode == 0xa || in.opcode == 0xb))
            result = combine(result, before.scc);
        const auto a = word(before, in, in.src[0]);
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == 3) result.constant = a.constant;
        const uint32_t width = scalar_write_width(in);
        for (uint32_t half = 0; half < width && dst >= 0 && dst + int(half) < 128; ++half) {
            auto value = result;
            if (in.fmt == Rdna2Format::SOP1 && in.opcode == 4)
                value = word(before, in, in.src[0], half);
            f.scalar[dst + half] = value;
        }
        if (in.fmt == Rdna2Format::SOP2 && in.opcode >= 0xf && in.opcode <= 0x1d &&
            (in.opcode & 1u) && dst >= 0 && dst < 127)
            f.mask[dst] =
                logical_mask(mask(before, in, in.src[0]), mask(before, in, in.src[1]), in.opcode);
        if (in.fmt == Rdna2Format::SOP2 && in.opcode == 0x25 && dst == 126) {
            const auto c = word(before, in, in.src[1]);
            f.exec = {result.uniform && result.genuine, std::nullopt};
            if (a.constant && c.constant) {
                const uint32_t width_bits = *a.constant & 63u, offset = *c.constant & 63u;
                f.exec.constant = ((uint64_t{1} << width_bits) - 1u) << offset;
            }
        } else if (dst == 126 && in.fmt == Rdna2Format::SOP2 && width == 2)
            f.exec = f.mask[dst];
        else if (rdna2_instruction_may_change_exec(in))
            f.exec = {};
        // These DATA producers preserve architectural SCC. Replacing its proof with their
        // uniform operands would let a later CSELECT launder a lane-local old condition.
        if (in.fmt == Rdna2Format::SOP2 && in.opcode != kSop2OpcodeCselectB32 &&
            in.opcode != kSop2OpcodeCselectB64 && in.opcode != kSop2OpcodeBfmB32 &&
            in.opcode != kSop2OpcodeBfmB64 && in.opcode != 0x26)
            f.scc = {result.uniform, result.genuine, std::nullopt};
        else if (in.fmt == Rdna2Format::SOP1 && !sop1_opcode_leaves_scc_unmodified(in.opcode))
            f.scc = {result.uniform, result.genuine, std::nullopt};
        return f;
    }
    if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK ||
        in.fmt == Rdna2Format::SMEM) {
        f.scc = {};
        if (rdna2_instruction_may_change_exec(in)) f.exec = {};
        return f;
    }
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360) {
        const auto selector = word(before, in, in.src[1]);
        const bool fixed = selector.constant && *selector.constant < 64 && selector.genuine;
        const bool genuine =
            fixed && (vector_source(before, in, in.src[0]) & (uint64_t{1} << *selector.constant));
        if (dst >= 0 && dst < 128) f.scalar[dst] = {fixed && before.common, genuine, std::nullopt};
        if (dst >= 0 && dst < 128) {
            bool uniform = false;
            if (fixed && selector.numeric_uniform && before.numeric_common &&
                in.src[0].kind == OperandKind::VGPR) {
                const auto slot = before.numeric_slots.find({in.src[0].value, *selector.constant});
                if (slot != before.numeric_slots.end())
                    uniform = slot->second;
                else if (!before.numeric_slot_vgprs.contains(in.src[0].value))
                    uniform = genuine;
            }
            f.scalar[dst].numeric_uniform = uniform;
        }
        return f;
    }
    uint64_t genuine = vector_sources(before, in);
    if (in.fmt == Rdna2Format::MUBUF) {
        // Only the already-lowered ordinary buffer contract can supply guest words. Address and
        // descriptor fields must be genuine; compilation still validates their actual resource.
        Word descriptor{true, true, std::nullopt};
        for (uint32_t w = 0; w < 4; ++w)
            descriptor = combine(descriptor, word(before, in, in.src[1], w));
        const uint32_t address_words = (in.literal & 0x3000u) == 0x3000u ? 2u : 1u;
        genuine = descriptor.genuine ? vector_source(before, in, in.src[0], address_words) &
                                           vector_source(before, in, in.src[2])
                                     : 0;
        if (in.opcode == 0x32)
            genuine &= vector_source(before, in, in.dst);
        else if (in.opcode < 0x8 || in.opcode > 0xf)
            genuine = 0;
        if (in.mubuf_tfe || in.mubuf_lds)
            genuine = 0;   // Status/owned LDS writes are not ordinary loaded VDST words.
    } else if (in.fmt == Rdna2Format::VOP1 && (in.opcode == 2 || in.opcode == 0x43)) {
        genuine = 0;   // READFIRSTLANE is lane-local here; opaque MOVRELS is not DATA authority.
    } else if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
        f.common = false;   // spill services and mixed mask/word slots are a separate contract
        if (dst >= 0 && dst < 256) {
            const auto selector = word(before, in, in.src[1]);
            const auto source = word(before, in, in.src[0]);
            bool uniform = source.numeric_uniform;
            if (in.src[0].kind == OperandKind::SGPR || in.src[0].kind == OperandKind::Special) {
                const int r = in.src[0].value;
                // A retained physical half of a genuine mask is uniform too; current DATA
                // availability remains the separate emission-time proof.
                uniform |= (r >= 0 && r < 128 && before.mask[r].genuine) ||
                           (r > 0 && r < 128 && before.mask[r - 1].genuine);
            }
            if (selector.constant && *selector.constant < 64 && selector.numeric_uniform)
                f.numeric_slots[{dst, *selector.constant}] = uniform && before.numeric_common;
            else
                for (auto& [slot, proof] : f.numeric_slots)
                    if (slot.first == dst) proof = false;
            f.numeric_slot_vgprs.insert(dst);
        }
        genuine = 0;
    } else {
        // This is a semantic proof inventory, not decoder storage accounting. Operations with
        // implicit old VDST, carry, interpolation, spill or mask inputs need their own transfer.
        const bool supported =
            (in.fmt == Rdna2Format::VOP1 && (in.opcode == 1 || in.opcode == kVop1OpcodeFfbhU32)) ||
            (in.fmt == Rdna2Format::VOP2 &&
             (in.opcode == 1 || in.opcode == 0x25 || in.opcode == 0x26)) ||
            (in.fmt == Rdna2Format::VOP3 &&
             (in.opcode == 0x169 || in.opcode == 0x36d || in.opcode == 0x378));
        if (!supported) genuine = 0;
        if (in.fmt == Rdna2Format::VOP2 && in.opcode == 1 && !before.mask[106].genuine) genuine = 0;
    }
    if (in.has_dpp || (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x378))
        genuine = genuine == full ? full : 0;
    if (in.has_dpp && (!in.dpp_bound_ctrl || in.dpp_row_mask != 15 || in.dpp_bank_mask != 15) &&
        dst >= 0 && dst < 256)
        genuine &= before.vector[dst];
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x378 && !in.permlane_fetch_inactive &&
        !in.permlane_bound_ctrl && before.exec.constant != full && dst >= 0 && dst < 256)
        genuine &= before.vector[dst];
    for_each_possible_vector_write(in, 255, [&](int r) { write_vector(f, in, r, genuine); });
    if (before.exec.constant == full && genuine == full &&
        !(in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361))
        for_each_possible_vector_write(in, 255, [&](int r) {
            std::erase_if(f.numeric_slots, [r](const auto& slot) { return slot.first.first == r; });
            f.numeric_slot_vgprs.erase(r);
        });
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == kVop1OpcodeMovreldB32) {
        for (int r = std::max(dst, 0); r < 256; ++r) {
            f.vector[r] = 0;
            f.vector_constant[r].reset();
            f.local_x.reset(r);
        }
    }
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 1 && dst >= 0 && dst < 256 &&
        before.exec.constant == full && !in.has_sdwa && !in.has_dpp && !in.has_modifier &&
        !in.src_abs[0] && !in.src_neg[0] && !in.omod && !in.clamp) {
        if (in.src[0].kind == OperandKind::VGPR && in.src[0].value >= 0 && in.src[0].value < 256) {
            f.local_x[dst] = before.local_x[in.src[0].value];
            f.vector_constant[dst] = before.vector_constant[in.src[0].value];
        } else
            f.vector_constant[dst] = word(before, in, in.src[0]).constant;
    }
    if (in.fmt == Rdna2Format::VOP1 && in.opcode == 2 && dst >= 0 && dst < 128)
        f.scalar[dst] = {false, genuine == full, std::nullopt};
    return f;
}
}   // namespace

std::vector<uint64_t> native_participation_entry(const SpirvCompute& b, const RegState* rs) {
    if (!ordinary(b) || !rs) return {};
    std::vector<uint64_t> result{b.guest_local_size[0], b.guest_local_size[1],
                                 b.guest_local_size[2], rs->exec,
                                 rs->exec_narrowed,     rs->scc_merge_placeholder};
    const auto& launch = b.guest_compute_launch;
    result.push_back(launch.launch.known);
    result.push_back(launch.launch.initiator);
    for (const auto word : launch.launch.num_thread) result.push_back(word);
    for (const auto word : launch.launch.dimensions) result.push_back(word);
    for (const auto dimension : launch.threads) result.push_back(dimension);
    result.push_back(launch.exact_thread_extent);
    for (int r = 0; r < 128; ++r) {
        const auto current = rs->sreg.find(r), input = rs->sreg_input.find(r);
        result.push_back(current == rs->sreg.end() ? 0 : current->second);
        result.push_back(input == rs->sreg_input.end() ? 0 : input->second);
        result.push_back(scalar_word_available(*rs, r));
    }
    for (int r = 0; r < 256; ++r) {
        const auto value = rs->vreg.find(r);
        result.push_back(value == rs->vreg.end() ? 0 : value->second);
        result.push_back(vector_word_available(*rs, r));
    }
    return result;
}

std::set<uint32_t> native_mask_participation(const SpirvCompute& b, const RegState* initial,
                                             const std::vector<Rdna2Inst>& instructions,
                                             std::set<uint32_t>* numeric_vcc_moves) {
    if (numeric_vcc_moves) numeric_vcc_moves->clear();
    if (!ordinary(b) || !initial || initial->exec_narrowed || instructions.empty() ||
        instructions.size() > 4096)
        return {};
    std::map<uint32_t, size_t> at;
    for (size_t i = 0; i < instructions.size(); ++i)
        if (!at.emplace(instructions[i].pc, i).second) return {};
    Facts entry;
    for (int r = 0; r < 124; ++r)
        if (scalar_word_available(*initial, r)) entry.scalar[r] = {true, true, std::nullopt};
    for (int r = 0; r < 256; ++r)
        if (vector_word_available(*initial, r)) entry.vector[r] = full;
    const auto x = initial->vreg.find(0);
    if (x != initial->vreg.end() && x->second == b.localid_comp[0]) entry.local_x.set(0);
    std::vector<std::optional<Facts>> incoming(instructions.size());
    std::vector<bool> queued(instructions.size());
    std::deque<size_t> work;
    const auto enqueue = [&](size_t i) {
        if (!queued[i]) {
            queued[i] = true;
            work.push_back(i);
        }
    };
    incoming[0] = entry;
    enqueue(0);
    size_t steps = 0;
    while (!work.empty()) {
        if (++steps > instructions.size() * 2048u)
            return {};   // proof budget, never guest execution
        const size_t i = work.front();
        work.pop_front();
        queued[i] = false;
        const auto& in = instructions[i];
        const auto before = *incoming[i];
        auto after = transfer(before, b, in);
        const bool branch =
            in.fmt == Rdna2Format::SOPP && (in.opcode == 2 || (in.opcode >= 4 && in.opcode <= 9));
        std::optional<bool> taken;
        if (branch && in.opcode != 2) {
            const bool scc = in.opcode == 4 || in.opcode == 5;
            const auto tested = in.opcode == 6 || in.opcode == 7 ? before.mask[106] : before.exec;
            const bool genuine = scc ? before.scc.uniform && before.scc.genuine : tested.genuine;
            after.common &= genuine;
            after.numeric_common &= genuine;
            if (!scc && tested.constant) {
                const bool zero = *tested.constant == 0;
                taken = (in.opcode == 6 || in.opcode == 8) ? zero : !zero;
            }
        }
        const auto send = [&](size_t to) {
            auto next = incoming[to] ? meet(*incoming[to], after) : after;
            if (!incoming[to] || next != *incoming[to]) {
                incoming[to] = next;
                enqueue(to);
            }
        };
        if (branch && (!taken || *taken || in.opcode == 2)) {
            const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !at.contains(uint32_t(target))) return {};
            send(at.at(uint32_t(target)));
        }
        if (!in.is_end && i + 1 < instructions.size() &&
            (!branch || (in.opcode != 2 && (!taken || !*taken))))
            send(i + 1);
    }
    // CONFIDENCE: HIGH -- induction over original predecessors, including every reentry, keeps
    // all64 host workers at the same PC/ACTIVE dynamic instance. Native votes/shuffles reconverge
    // inside each operation. A fixed-peer shuffle may prove routing equality without genuine
    // selected DATA; only genuine branch dependencies and current predicate operands grant a site.
    std::set<uint32_t> result;
    for (size_t i = 0; i < instructions.size(); ++i) {
        const auto& in = instructions[i];
        if (numeric_vcc_moves && incoming[i] && incoming[i]->numeric_common &&
            in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 && in.dst.value == 106 &&
            (in.src[0].kind == OperandKind::SGPR || in.src[0].kind == OperandKind::Special) &&
            in.src[0].value >= 0 && in.src[0].value < 127 &&
            word(*incoming[i], in, in.src[0]).numeric_uniform &&
            word(*incoming[i], in, in.src[0], 1).numeric_uniform)
            numeric_vcc_moves->insert(in.pc);
        if (!incoming[i] || !incoming[i]->common || in.fmt != Rdna2Format::VOPC ||
            vopc_is_cmpx(in.opcode) || in.dst.kind != OperandKind::SGPR || in.dst.value < 0 ||
            in.dst.value > 104 || !scalar_write_is_b64_mask(in, in.dst.value))
            continue;
        if (compare(*incoming[i], b, in).genuine) result.insert(in.pc);
    }
    return result;
}

bool native_mask_predicate_available(const RegState& rs, const Rdna2Inst& in) {
    for (uint32_t k = 0; k < in.n_src; ++k) {
        const auto source = in.src[k];
        if (source.kind == OperandKind::VGPR) {
            const uint32_t width = rdna2_vgpr_source_span(in, k);
            if (!width || width > 4) return false;
            for (uint32_t w = 0; w < width; ++w)
                if (!vector_word_available(rs, source.value + int(w))) return false;
        } else if (source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) {
            const uint32_t width = scalar_alu_source_words(in, k);
            if (!width || width > 4) return false;
            for (uint32_t w = 0; w < width; ++w)
                if (!scalar_word_available(rs, source.value + int(w))) return false;
        } else if (source.kind != OperandKind::InlineInt &&
                   !(source.kind == OperandKind::Literal && in.has_literal))
            return false;
    }
    return true;
}
bool native_mask_participation_site(const SpirvCompute& b, const Rdna2Inst& in) {
    return ordinary(b) && saved_mask_physical_launch_matches_original(b) &&
           b.native_participating_mask_pcs.contains(in.pc);
}
}   // namespace prosper::gpu
