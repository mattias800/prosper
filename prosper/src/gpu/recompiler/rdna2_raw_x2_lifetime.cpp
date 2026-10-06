// Raw x2 lifetimes distinguish pointer provenance from two physical numeric words.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/execute/sopp_cfg.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace prosper::gpu {
// Prove S_LOAD_DWORDX2 DESCRIPTOR-TABLE POINTER loads.
//
// A shader whose resources live entirely in an SRT reaches them through pointers: the driver places
// a 64-bit table pointer in user data, and the shader chases it -- `s_load_dwordx2 s[2:3], s[0:1],
// imm` -- before loading the actual V#/T#/S# out of the pointed-to table with an x4/x8. Such a
// pointer is never data. Its only use is as the SBASE of another raw scalar load.
//
// Uncharted: Legacy of Thieves (PPSA05684) declares ZERO sharps in every shader
// (sharp_resource_count {0,0,0,0}, srt_size_dw 2..9), so every graphics resource arrives this way.
// The pointer load fell through to the constant-buffer path, found no declared cbuf, and rejected
// the whole shader -- `[smem-reject] pc=1 reason=unresolved-cbuf op=0x1 src0=s0` -- so every
// fragment shader in the title failed and no draw could be realized (#3616).
//
// The front half already resolves the entire chain: resolve_dynamic_fetch follows exactly these
// pointers out of guest memory and publishes each descriptor at its exact consumer PC. The pointer
// is therefore provenance in SPIR-V, the same standing the x16 and x2-fragment shapes below and
// above already have, and zero placeholders represent it exactly.
//
// The admission is a whole-stream USE proof, never opcode-wide. The loaded pair must be read ONLY
// as a raw `s_load_*` SBASE -- never as scalar data, an s_buffer_load V# base, an image or buffer
// descriptor, or an address -- and at least one such pointer use must exist, so the proof asserts a
// shape rather than merely failing to find a counterexample. Scanning EVERY instruction rather than
// the forward path makes control flow irrelevant: if no read anywhere treats the value as data, no
// path can — and "every read" has to include the IMPLICIT destination reads that decode no source
// operand, or the claim is false for the whole SOPK family. CONFIDENCE: HIGH for the admitted shape.
//
// A later REDEFINITION of the pair is deliberately not disqualifying, and that is the one subtle
// point. These shaders reuse the low SGPRs hard -- `s_load_dwordx8 s[0:7], s[38:39]` lands on top of
// the s[2:3] pointer a few instructions later -- so requiring sole ownership rejected every real
// candidate. It is not needed for soundness: this proof decides only what OUR load's destination
// holds, the scan already refuses if ANY instruction anywhere reads the pair as data, and a read
// that belongs to a later definition is either another SBASE use (harmless) or a data read (which
// rejects us conservatively). Dropping the check is strictly more permissive and equally sound.
std::unordered_set<uint32_t> rdna2_proven_smem_pointer_loads(const std::vector<Rdna2Inst>& ins) {
    std::unordered_set<uint32_t> proven;
    if (ins.empty()) return proven;

    auto scalar_operand = [](const Operand& operand) {
        return operand.kind == OperandKind::SGPR || (operand.kind == OperandKind::Special &&
                                                     operand.value >= 106 && operand.value <= 124);
    };

    for (size_t load_index = 0; load_index < ins.size(); ++load_index) {
        const Rdna2Inst& load = ins[load_index];
        if (load.is_end || load.fmt != Rdna2Format::SMEM || load.opcode != kSmemOpcodeLoadDwordX2 ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 || load.dst.value + 1 > 105 ||
            // Immediate-only. A register SOFFSET is the fragment shape the sibling proof owns.
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            static_cast<int32_t>(load.literal) < 0)
            continue;

        const int lo = load.dst.value, hi = lo + 1;
        auto touches = [&](int first, uint32_t words) {
            if (first < 0 || !words) return false;
            return first <= hi && first + static_cast<int>(words) > lo;
        };

        bool valid = true, used_as_pointer = false;
        for (size_t index = 0; valid && index < ins.size(); ++index) {
            if (index == load_index) continue;
            const Rdna2Inst& in = ins[index];
            if (in.is_end) continue;

            if (in.fmt == Rdna2Format::SMEM) {
                // SBASE is 2 dwords for a raw s_load and 4 for an s_buffer_load's V#. Only the raw
                // form, naming the pair exactly, is the pointer use admitted here.
                const bool raw_pointer_base =
                    in.opcode < 0x08u && scalar_operand(in.src[0]) && in.src[0].value == lo;
                if (raw_pointer_base) used_as_pointer = true;
                const uint32_t base_words = in.opcode >= 0x08u ? 4u : 2u;
                if (!raw_pointer_base && scalar_operand(in.src[0]) &&
                    touches(in.src[0].value, base_words))
                    valid = false;
                if (valid && scalar_operand(in.src[1]) && touches(in.src[1].value, 1))
                    valid = false;
                continue;
            }
            if (in.fmt == Rdna2Format::MIMG) {
                if ((in.src[1].kind == OperandKind::SGPR && touches(in.src[1].value, 8)) ||
                    (scalar_operand(in.src[2]) && touches(in.src[2].value, 4)))
                    valid = false;
                continue;
            }
            if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) {
                if ((in.src[1].kind == OperandKind::SGPR && touches(in.src[1].value, 4)) ||
                    (scalar_operand(in.src[2]) && touches(in.src[2].value, 1)))
                    valid = false;
                continue;
            }
            // An instruction can read its own DESTINATION without naming it as a source, and the
            // loop below cannot see that: SOPK decodes no source operands at all (`n_src` stays 0),
            // so `s_cmpk_eq_i32 s2, 0`, `s_addk_i32`, `s_mulk_i32` and `s_cmovk_i32` -- and SOP1's
            // conditional moves and bitset forms -- would read the pointer pair as ordinary data
            // with this proof none the wiser. It would then be admitted, placeholdered with zero,
            // and the shader would compute on that zero and render silently wrong, which is exactly
            // the outcome the whole proof exists to prevent. The sibling x16 proof already makes
            // this check; the helper was written for this case and says so.
            const uint32_t implicit_read = scalar_implicit_destination_read_width(in);
            if (implicit_read && touches(in.dst.value, implicit_read)) {
                valid = false;
                break;
            }
            for (uint32_t source = 0; valid && source < in.n_src; ++source) {
                if (!scalar_operand(in.src[source])) continue;
                const uint32_t words = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                                               in.fmt == Rdna2Format::SOPC ||
                                               in.fmt == Rdna2Format::VOP3
                                           ? 2u
                                           : 1u;
                if (touches(in.src[source].value, words)) valid = false;
            }
        }
        if (valid && used_as_pointer) proven.insert(load.pc);
    }
    return proven;
}

// A raw x2 load often supplies pointer/descriptor provenance; merely having a known SBASE
// does not justify adding a data binding. Admit one when BOTH loaded words are directly read
// by supported B32 scalar operations before a control transfer. Those reads require actual
// current bytes. This is not a claim that derivatives cannot later form descriptors or
// pointers: loading the same bytes remains correct for those paths too. Original-word
// lifetimes use the emitter's complete scalar-write inventory, including secondary writes.
// Branches before the candidate load do not affect its own observation window.
std::vector<uint32_t> rdna2_proven_raw_x2_data_loads(const std::vector<Rdna2Inst>& ins) {
    std::vector<uint32_t> proven;
    std::unordered_set<uint32_t> instruction_pcs;
    for (const Rdna2Inst& in : ins) instruction_pcs.insert(in.pc);
    for (size_t load_index = 0; load_index < ins.size(); ++load_index) {
        const Rdna2Inst& load = ins[load_index];
        if (load.fmt != Rdna2Format::SMEM || load.opcode != kSmemOpcodeLoadDwordX2 ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            (load.dst.value > 104 && load.dst.value != 106) ||
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            static_cast<int32_t>(load.literal) < 0)
            continue;

        const int first = load.dst.value;
        const bool vcc_data_pair = first == 106;
        bool live[2] = {true, true};
        bool used[2] = {false, false};
        bool valid = true;
        bool vcc_mask_replaced = false;
        // The vector form below may cross one forward branch after both observations. Its
        // invocation-local buffer binding must still name the entry pointer on every path to the
        // load, and a later backedge must not rerun the load with a different SBASE.
        bool vector_entry_pointer = load.src[0].kind == OperandKind::SGPR &&
                                    load.src[0].value >= 0 && load.src[0].value + 1 <= 105;
        for (size_t i = 0; i < load_index && vector_entry_pointer; ++i) {
            const Rdna2Inst& earlier = ins[i];
            if (earlier.fmt == Rdna2Format::Unknown || !earlier.len_dwords ||
                (earlier.fmt == Rdna2Format::SOP1 && earlier.opcode >= 0x20u &&
                 earlier.opcode <= 0x22u) ||
                (earlier.fmt == Rdna2Format::SOPK && earlier.opcode == 0x16u)) {
                vector_entry_pointer = false;
                break;
            }
            for_each_scalar_write(earlier, [&](int base, uint32_t width) {
                vector_entry_pointer &= base + static_cast<int>(width) <= load.src[0].value ||
                                        base > load.src[0].value + 1;
            });
        }
        bool reenters_load = false;
        for (size_t i = load_index + 1; i < ins.size(); ++i) {
            const Rdna2Inst& later = ins[i];
            if (later.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(later.opcode)) {
                const int64_t target =
                    static_cast<int64_t>(later.pc) + later.len_dwords + later.simm16;
                reenters_load |= target <= static_cast<int64_t>(load.pc);
            }
        }
        bool vector_observed = false;
        auto touches_live = [&](int base, uint32_t width) {
            if (base < 0 || !width) return false;
            for (int word = 0; word < 2; ++word)
                if (live[word] && base <= first + word &&
                    static_cast<uint32_t>(first + word - base) < width)
                    return true;
            return false;
        };
        for (size_t index = load_index + 1; index < ins.size() && valid; ++index) {
            const Rdna2Inst& in = ins[index];
            if (in.fmt == Rdna2Format::Unknown || in.len_dwords == 0) {
                valid = false;
                break;
            }
            if (in.is_end) break;
            // Physical VCC_LO/HI may carry two ordinary scalar words, but the previous VCC
            // predicate is a separate emitter view. Keep following the data lifetime until a
            // fresh compare replaces that predicate. Every other VCC use/write before then is
            // refused; merely seeing both scalar conversions is not a mask-safety proof.
            if (vcc_data_pair && in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                in.dst.kind == OperandKind::Special && in.dst.value == 106) {
                vcc_mask_replaced = used[0] && used[1];
                valid = vcc_mask_replaced;
                break;
            }
            if (vcc_data_pair) {
                if ((in.fmt == Rdna2Format::VOP2 &&
                     (in.opcode == 0x01u || (in.opcode >= 0x28u && in.opcode <= 0x2au))) ||
                    (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x06u || in.opcode == 0x07u))) {
                    valid = false;
                    break;
                }
                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    if (base <= 107 && base + static_cast<int>(width) > 106) valid = false;
                });
                if (!valid) break;
            }
            // WAITCNT, NOP and CLAUSE are the only SOPP instructions allowed while the
            // loaded pair is live. A branch could re-enter a use without this load.
            if (in.fmt == Rdna2Format::SOPP && in.opcode != 0x00 && in.opcode != 0x0c &&
                in.opcode != 0x20) {
                const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
                if (vector_observed && used[0] && used[1] && vector_entry_pointer &&
                    !reenters_load && sopp_opcode_is_direct_branch(in.opcode) &&
                    target > static_cast<int64_t>(in.pc) && target <= UINT32_MAX &&
                    instruction_pcs.contains(static_cast<uint32_t>(target)))
                    break;
                valid = false;
                break;
            }
            // SETPC/SWAPPC/RFE leave this local instruction stream. Direct scalar
            // observations already made before this transfer still need current bytes;
            // make no claim about their derivatives or later uses.
            const bool indirect_transfer =
                in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20 && in.opcode <= 0x22;
            if (scalar_implicit_destination_read_width(in) &&
                touches_live(in.dst.value, scalar_implicit_destination_read_width(in))) {
                valid = false;
                break;
            }
            const bool scalar_data_op = !vcc_data_pair && in.fmt == Rdna2Format::SOP2 &&
                                        (in.opcode == 0x1e || in.opcode == 0x27);
            const bool vector_data_op = !vcc_data_pair && vector_entry_pointer && !reenters_load &&
                                        in.fmt == Rdna2Format::VOP2 && !in.has_sdwa &&
                                        !in.has_dpp && in.len_dwords == 1 &&
                                        (in.opcode == 0x05u || in.opcode == 0x08u);
            const bool vcc_data_op = vcc_data_pair && vector_entry_pointer && !reenters_load &&
                                     in.fmt == Rdna2Format::VOP1 && in.opcode == 0x06u &&
                                     !in.has_sdwa && !in.has_dpp && in.len_dwords == 1;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const Operand& operand = in.src[source];
                if (operand.kind != OperandKind::SGPR &&
                    !(operand.kind == OperandKind::Special && operand.value >= 106 &&
                      operand.value <= 124))
                    continue;
                uint32_t width = 1;
                if (in.fmt == Rdna2Format::SMEM && source == 0)
                    width = in.opcode >= 8 ? 4 : 2;
                else if (in.fmt == Rdna2Format::MIMG && source == 1)
                    width = 8;
                else if (in.fmt == Rdna2Format::MIMG && source == 2)
                    width = 4;
                else if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) &&
                         source == 1)
                    width = 4;
                else if (!scalar_data_op && !vector_data_op &&
                         (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                          in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::VOP3))
                    width = 2;   // conservative for B64 forms
                if (!touches_live(operand.value, width)) continue;
                if (!scalar_data_op && !vector_data_op && !vcc_data_op) {
                    valid = false;
                    break;
                }
                if (vector_data_op) vector_observed = true;
                for (int word = 0; word < 2; ++word)
                    if (live[word] && operand.value == first + word) used[word] = true;
            }
            if (!valid) break;
            if (indirect_transfer) break;
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (int word = 0; word < 2; ++word)
                    if (base >= 0 && base <= first + word &&
                        static_cast<uint32_t>(first + word - base) < width)
                        live[word] = false;
            });
            if (!live[0] && !live[1]) break;
        }
        if (valid && used[0] && used[1] && (!vcc_data_pair || vcc_mask_replaced))
            proven.push_back(load.pc);
    }
    return proven;
}

namespace {
bool scalar(const Operand& op) {
    return op.kind == OperandKind::SGPR ||
           (op.kind == OperandKind::Special && op.value >= 106 && op.value <= 124);
}
bool overlap(int base, uint32_t width, int first) {
    return base >= 0 && width && base <= first + 1 &&
           (width == UINT32_MAX || int64_t(base) + width > first);
}
bool implicit_vcc_read(const Rdna2Inst& in) {
    return (in.fmt == Rdna2Format::VOP2 &&
            (in.opcode == 1u || (in.opcode >= 0x28u && in.opcode <= 0x2au))) ||
           (in.fmt == Rdna2Format::SOPP && (in.opcode == 6u || in.opcode == 7u));
}
bool writes_pair(const Rdna2Inst& in, int first) {
    bool writes = false;
    for_each_scalar_write(in,
                          [&](int base, uint32_t width) { writes |= overlap(base, width, first); });
    if (first == 106 && in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
        in.dst.value == 106)
        writes = true;
    return writes;
}
bool raw_x2(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SMEM && in.opcode == kSmemOpcodeLoadDwordX2 &&
           in.dst.kind == OperandKind::SGPR && in.dst.value == 106 && scalar(in.src[0]) &&
           in.src[1].kind == OperandKind::Special && in.src[1].value == 125 &&
           static_cast<int32_t>(in.literal) >= 0 && !(in.literal & 3u) &&
           in.literal <= 0x100000u - 8u;
}
bool b32_numeric(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::SOP1) return in.opcode == kSop1OpcodeMovB32;
    if (in.fmt != Rdna2Format::SOP2 || scalar_write_width(in) != 1u) return false;
    switch (in.opcode) {
        case 0x00:
        case 0x01:
        case 0x02:
        case 0x03:
        case 0x06:
        case 0x07:
        case 0x08:
        case 0x09:
        case 0x1e:
        case 0x20:
        case 0x22:
        case 0x26:
        case 0x27:
        case 0x2e:
        case 0x2f:
        case 0x30:
        case 0x31:
        case 0x35:
        case 0x36: return true;
        default: return false;
    }
}
// Both original words must reach an actual B32 multiply before any replacement. Subsequent
// replacements must derive from those owned words or real constants on every incoming path;
// arbitrary scalar scratch and the obsolete Bool predicate confer no numeric authority.
// All branches in the numeric lifetime are followed until a fresh compare or program exit;
// later mask-phase loops cannot re-enter that expired lifetime.
bool numeric_lifetime(const std::vector<Rdna2Inst>& ins,
                      const std::unordered_map<uint32_t, size_t>& by_pc, size_t child) {
    const int first = ins[child].dst.value;
    struct State {
        size_t index;
        bool multiplied;
        std::bitset<128> owned;
    };
    std::bitset<128> initial;
    initial.set(first);
    initial.set(first + 1);
    std::vector<State> pending{{child + 1, false, initial}};
    std::vector<std::array<std::optional<std::bitset<128>>, 2>> incoming(ins.size());
    while (!pending.empty()) {
        auto state = pending.back();
        pending.pop_back();
        if (state.index >= ins.size()) return false;
        auto& prior = incoming[state.index][state.multiplied];
        if (prior) {
            state.owned &= *prior;
            if (state.owned == *prior) continue;
        }
        prior = state.owned;
        const auto& in = ins[state.index];
        if (in.is_end) {
            if (!state.multiplied) return false;
            continue;
        }
        bool multiplied = state.multiplied;
        const auto owned_source = [&](const Operand& op) {
            if (scalar(op)) return op.value >= 0 && op.value < 128 && state.owned.test(op.value);
            return op.kind == OperandKind::InlineInt || op.kind == OperandKind::InlineFloat ||
                   op.kind == OperandKind::Literal ||
                   (op.kind == OperandKind::Special && op.value == 125);
        };
        const bool owned_result = b32_numeric(in) && scalar(in.dst) &&
                                  in.n_src == (in.fmt == Rdna2Format::SOP1 ? 1u : 2u) &&
                                  std::all_of(in.src, in.src + in.n_src, owned_source);
        if (first == 106 && in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind == OperandKind::Special && in.dst.value == 106) {
            if (!multiplied) return false;
            continue;   // the emitter creates a new predicate from this compare's real operands
        }
        if (first == 106 && implicit_vcc_read(in)) return false;
        const bool mul = in.fmt == Rdna2Format::SOP2 && in.opcode == 0x26u && in.n_src == 2u &&
                         scalar(in.src[0]) && scalar(in.src[1]) &&
                         ((in.src[0].value == first && in.src[1].value == first + 1) ||
                          (in.src[1].value == first && in.src[0].value == first + 1));
        if (scalar_implicit_destination_read_width(in) &&
            overlap(in.dst.value, scalar_implicit_destination_read_width(in), first))
            return false;
        for (uint32_t k = 0; k < in.n_src; ++k) {
            if (!scalar(in.src[k])) continue;
            const uint32_t width = b32_numeric(in) ? 1u : scalar_alu_source_words(in, k);
            if (!overlap(in.src[k].value, width, first)) continue;
            const bool conversion = in.fmt == Rdna2Format::VOP1 && in.opcode == 6u && !in.has_dpp &&
                                    !in.has_sdwa && in.len_dwords == 1u;
            if (!owned_source(in.src[k]) || (!multiplied && !mul) ||
                (!b32_numeric(in) && !conversion))
                return false;
        }
        if (mul) multiplied = true;
        if (writes_pair(in, first) && (!multiplied || !owned_result)) return false;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (int reg = 0; reg < 128; ++reg)
                if (base >= 0 && reg >= base &&
                    (width == UINT32_MAX || uint64_t(reg - base) < width))
                    state.owned.reset(reg);
        });
        if (owned_result && in.dst.value >= 0 && in.dst.value < 128) state.owned.set(in.dst.value);
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const uint32_t target =
                static_cast<uint32_t>(int64_t(in.pc) + in.len_dwords + in.simm16);
            pending.push_back({by_pc.at(target), multiplied, state.owned});
            if (in.opcode == kSoppOpcodeBranch) continue;
        }
        pending.push_back({state.index + 1, multiplied, state.owned});
    }
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.fmt != Rdna2Format::SOPP || !sopp_opcode_is_direct_branch(in.opcode)) continue;
        const auto target =
            by_pc.at(static_cast<uint32_t>(int64_t(in.pc) + in.len_dwords + in.simm16));
        if ((incoming[target][0] || incoming[target][1]) && !(incoming[i][0] || incoming[i][1]))
            return false;   // a later predicate phase must not restore an expired numeric definition
    }
    return true;
}
}   // namespace

std::vector<RawNestedWideChain> rdna2_owned_raw_x2_chains(const std::vector<Rdna2Inst>& ins) {
    std::vector<RawNestedWideChain> result;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        // Anything that can write a register its operands do not name, or run code off the
        // decoded edges, voids every chain. This used to name SOP1 0x28..0x2a for the M0-relative
        // moves; those three are B64 saveexec forms, and the relative moves went unrefused (#4559).
        if (in.fmt == Rdna2Format::Unknown || !in.len_dwords || !by_pc.emplace(in.pc, i).second ||
            rdna2_may_write_unnamed_register_or_leave_cfg(in))
            return {};
    }
    for (const auto& in : ins) {
        if (in.fmt != Rdna2Format::SOPP || in.is_end) continue;
        if (sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !by_pc.contains(static_cast<uint32_t>(target)))
                return {};
        } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier)
            return {};
    }
    for (size_t parent = 0; parent + 1 < ins.size(); ++parent) {
        const auto& load = ins[parent];
        if (!raw_x2(load) || load.src[0].kind != OperandKind::SGPR || load.src[0].value < 0 ||
            load.src[0].value > 104)
            continue;
        bool stable = true;
        for (size_t i = 0; i < parent && stable; ++i) {
            stable &=
                !(ins[i].fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(ins[i].opcode));
            stable &= !writes_pair(ins[i], load.src[0].value);
        }
        if (!stable) continue;
        const int first = load.dst.value;
        for (size_t child = parent + 1; child < ins.size(); ++child) {
            const auto& in = ins[child];
            if (raw_x2(in) && in.src[0].value == first && in.dst.value == first) {
                const bool reenters_load =
                    std::any_of(ins.begin(), ins.end(), [&](const auto& branch) {
                        return branch.fmt == Rdna2Format::SOPP &&
                               sopp_opcode_is_direct_branch(branch.opcode) &&
                               int64_t(branch.pc) + branch.len_dwords + branch.simm16 <= in.pc;
                    });
                if (!reenters_load && numeric_lifetime(ins, by_pc, child))
                    result.push_back({load.pc, in.pc, 8u, 8u, load.literal, in.literal});
                break;   // this read replaces the pointer lifetime, whether admitted or refused
            }
            if (in.is_end ||
                (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) ||
                (first == 106 && implicit_vcc_read(in)) || writes_pair(in, first))
                break;
            bool read = scalar_implicit_destination_read_width(in) &&
                        overlap(in.dst.value, scalar_implicit_destination_read_width(in), first);
            for (uint32_t k = 0; k < in.n_src; ++k)
                if (scalar(in.src[k]))
                    read |= overlap(in.src[k].value, scalar_alu_source_words(in, k), first);
            if (read) break;
        }
    }
    return result;
}

void end_smem_vcc_data_lifetime(RegState& rs, const Rdna2Inst& in, uint32_t n) {
    if (n != 2u || in.dst.value != 106 || !rs.smem_raw_x2_data_loads.contains(in.pc)) return;
    rs.vcc = 0;
    rs.vcc_wave_uniform = 0;
    for (int reg = 105; reg <= 107; ++reg) {
        rs.sreg_bool.erase(reg);
        rs.sreg_bool_b32.erase(reg);
        rs.sreg_bool_narrowed.erase(reg);
    }
}

bool emit_owned_raw_x2(SpirvCompute& b, RegState& rs, const Rdna2Inst& in,
                       const ShaderResourceTable* rt, bool& ok) {
    if (in.opcode != kSmemOpcodeLoadDwordX2) return false;
    const auto chain =
        std::find_if(rs.smem_owned_raw_x2_chains.begin(), rs.smem_owned_raw_x2_chains.end(),
                     [&](const auto& c) { return c.parent_pc == in.pc || c.child_pc == in.pc; });
    if (chain == rs.smem_owned_raw_x2_chains.end()) {
        // An owned x2 carrier is not general constant-buffer authority. If the complete original
        // lifetime was refused, a legacy resource lookup must not resurrect that load.
        if (rt && std::any_of(rt->resources.begin(), rt->resources.end(), [&](const auto& r) {
                return r.fetch_pc == in.pc && r.owned_nested_snapshot_bytes == 8u;
            })) {
            ok = false;
            return true;
        }
        return false;
    }
    const auto* parent =
        rt ? owned_nested_snapshot_at(*rt, chain->parent_pc, 8u, compiler_resource_has_host_data)
           : nullptr;
    const auto* child =
        rt ? owned_nested_snapshot_at(*rt, chain->child_pc, 8u, compiler_resource_has_host_data)
           : nullptr;
    if (!b.is_compute || !parent || !child) {
        ok = false;
        return true;
    }
    const auto* resource = in.pc == chain->parent_pc ? parent : child;
    for (uint32_t k = 0; k < 2; ++k) {
        rs.sreg[in.dst.value + static_cast<int>(k)] = b.cbuf_load(b.uconst(k), resource->binding);
        rs.sreg_srt.erase(in.dst.value + static_cast<int>(k));
    }
    end_smem_vcc_data_lifetime(rs, in, 2u);
    return true;
}
}   // namespace prosper::gpu
