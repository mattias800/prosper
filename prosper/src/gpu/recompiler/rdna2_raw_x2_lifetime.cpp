// Raw x2 pointer and numeric use proofs extracted without behavior changes.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <algorithm>
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
        return operand.kind == OperandKind::SGPR ||
               (operand.kind == OperandKind::Special &&
                operand.value >= 106 && operand.value <= 124);
    };

    for (size_t load_index = 0; load_index < ins.size(); ++load_index) {
        const Rdna2Inst& load = ins[load_index];
        if (load.is_end || load.fmt != Rdna2Format::SMEM ||
            load.opcode != kSmemOpcodeLoadDwordX2 ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + 1 > 105 ||
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
                const bool raw_pointer_base = in.opcode < 0x08u &&
                    scalar_operand(in.src[0]) && in.src[0].value == lo;
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
            if (implicit_read && touches(in.dst.value, implicit_read)) { valid = false; break; }
            for (uint32_t source = 0; valid && source < in.n_src; ++source) {
                if (!scalar_operand(in.src[source])) continue;
                const uint32_t words =
                    in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                    in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::VOP3 ? 2u : 1u;
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
            if (later.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(later.opcode)) {
                const int64_t target = static_cast<int64_t>(later.pc) +
                    later.len_dwords + later.simm16;
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
            if (vcc_data_pair && in.fmt == Rdna2Format::VOPC &&
                !vopc_is_cmpx(in.opcode) && in.dst.kind == OperandKind::Special &&
                in.dst.value == 106) {
                vcc_mask_replaced = used[0] && used[1];
                valid = vcc_mask_replaced;
                break;
            }
            if (vcc_data_pair) {
                if ((in.fmt == Rdna2Format::VOP2 &&
                     (in.opcode == 0x01u ||
                      (in.opcode >= 0x28u && in.opcode <= 0x2au))) ||
                    (in.fmt == Rdna2Format::SOPP &&
                     (in.opcode == 0x06u || in.opcode == 0x07u))) {
                    valid = false;
                    break;
                }
                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    if (base <= 107 && base + static_cast<int>(width) > 106)
                        valid = false;
                });
                if (!valid) break;
            }
            // WAITCNT, NOP and CLAUSE are the only SOPP instructions allowed while the
            // loaded pair is live. A branch could re-enter a use without this load.
            if (in.fmt == Rdna2Format::SOPP && in.opcode != 0x00 &&
                in.opcode != 0x0c && in.opcode != 0x20) {
                const int64_t target = static_cast<int64_t>(in.pc) +
                    in.len_dwords + in.simm16;
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
            const bool indirect_transfer = in.fmt == Rdna2Format::SOP1 &&
                in.opcode >= 0x20 && in.opcode <= 0x22;
            if (scalar_implicit_destination_read_width(in) &&
                touches_live(in.dst.value, scalar_implicit_destination_read_width(in))) {
                valid = false;
                break;
            }
            const bool scalar_data_op = !vcc_data_pair && in.fmt == Rdna2Format::SOP2 &&
                (in.opcode == 0x1e || in.opcode == 0x27);
            const bool vector_data_op = !vcc_data_pair && vector_entry_pointer && !reenters_load &&
                in.fmt == Rdna2Format::VOP2 && !in.has_sdwa && !in.has_dpp &&
                in.len_dwords == 1 && (in.opcode == 0x05u || in.opcode == 0x08u);
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
                    width = 2; // conservative for B64 forms
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

} // namespace prosper::gpu
