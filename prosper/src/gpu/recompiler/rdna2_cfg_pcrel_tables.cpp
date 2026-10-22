// Original PC-relative table format proofs.
#include "gpu/recompiler/rdna2_cfg_pcrel_tables.hpp"
#include "gpu/recompiler/rdna2_cfg_pcrel_parts.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {

// True when a typed-buffer BUF_FMT stores exactly 32 bits per component and applies no conversion,
// so a typed fetch of `components` components is byte-identical to the same number of raw dwords.
// rdna2_buffer_format maps 20/21/22 to Uint32/Sint32/Float32 (n=1), 62/63/64 (n=2), 72/73/74 (n=3)
// and 75/76/77 (n=4); every other value either narrows, normalizes or packs. The component count
// must MATCH the opcode's, because a typed load whose format is narrower than the opcode
// default-fills the missing components (0,0,0,1) instead of reading them.
// CONFIDENCE: HIGH, and the reason is a mitigation rather than the table's authority. The table it
// reads (`agc_shader_layout.cpp:137`) labels itself "HIGH on the anchors, MED on the rest", and of
// these twelve only 64/74/77 are anchors — the live case, BUF_FMT 22, is not. What makes HIGH
// defensible is that for all twelve the NON-folded path is also a plain dword copy
// (`rdna2_emit_alu.cpp`, `value = load_dword(kidx);  // raw 32-bit component`), so a wrong row here
// would make the fold wrong in exactly the way the ordinary path is already wrong. The fold
// introduces no divergence of its own.
bool rdna2_buffer_format_is_raw_dwords(uint32_t format, uint32_t components) {
    switch (format) {
        case 20:
        case 21:
        case 22: return components == 1;
        case 62:
        case 63:
        case 64: return components == 2;
        case 72:
        case 73:
        case 74: return components == 3;
        case 75:
        case 76:
        case 77: return components == 4;
        default: return false;
    }
}

// True when a V# word3's DST_SEL routes the first `components` channels straight through, so a
// TYPED fetch returns the stored dwords in their stored order.
//
// This does not apply to the untyped fold: a raw `buffer_load_dword*` ignores DST_SEL entirely —
// RDNA2 ISA Table 31 gives every non-FORMAT buffer op a DST SEL of "identity", and gives MTBUF
// `tbuffer_*` the same, so only MUBUF `buffer_*_format_*` reads the field at all (#2869). A
// FORMAT load does not, and getting it wrong is silent -- Sonic Frontiers' own embedded-table V#
// carries word3 = 0x10005004, i.e. DST_SEL = (X, 0, 0, 0), so a four-component typed fetch through
// that same descriptor would return the stored X and three CONSTANT ZEROES, not four dwords. Its
// actual load is `tbuffer_load_format_x`, which writes only X, and X is identity -- but a fold that
// checked only the format would have been wrong for the neighbouring shape and could not have said
// so. Selector fields for components the opcode does not write are not examined.
// DST_SEL is v[3][11:0], three bits per channel; SQ_SEL identity is 4/5/6/7 (X/Y/Z/W).
bool rdna2_buffer_dst_sel_is_identity(uint32_t descriptor_word3, uint32_t components) {
    for (uint32_t channel = 0; channel < components && channel < 4u; ++channel)
        if (((descriptor_word3 >> (3u * channel)) & 0x7u) != 4u + channel) return false;
    return true;
}

PcrelTables detect_pcrel_tables(const std::vector<Rdna2Inst>& ins, const uint32_t* code,
                                size_t dwords, size_t* required_dwords) {
    PcrelTables out;
    std::unordered_map<int, uint64_t>
        pcoff;   // reg -> byte offset into the code blob (pair LO half)
    std::unordered_set<int> pchi;   // regs holding the matching HI half
    std::unordered_map<int, uint32_t>
        kconst;   // reg -> compile-time constant (s_mov_b32 literal/inline)
    std::unordered_map<int, uint32_t> fact_pc;   // reg -> pc where its current fact was established
    std::vector<uint32_t> br_targets;   // all branch targets (for the entry-soundness check)
    for (const auto& in : ins) {
        if (in.is_end) break;
        if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) && in.opcode >= 0x02 &&
            in.opcode <= 0x09 && in.opcode != 0x03)
            br_targets.push_back(in.pc + in.len_dwords + (uint32_t)(int32_t)in.simm16);
    }
    auto kill = [&](int r) {
        pcoff.erase(r);
        pchi.erase(r);
        kconst.erase(r);
        fact_pc.erase(r);
    };
    auto imm_of = [&](const Operand& o, const Rdna2Inst& in, uint32_t* v) -> bool {
        if (o.kind == OperandKind::Literal) {
            *v = in.literal;
            return true;
        }
        if (o.kind == OperandKind::InlineInt) {
            *v = (uint32_t)o.value;
            return true;
        }
        if (o.kind == OperandKind::SGPR) {
            auto it = kconst.find(o.value);
            if (it != kconst.end()) {
                *v = it->second;
                return true;
            }
        }
        return false;
    };
    for (const auto& in : ins) {
        if (in.is_end) break;
        switch (in.fmt) {
            case Rdna2Format::SOP1:
                cfg_pcrel_scalar_unary(std::cref(kill), in, pcoff, pchi, fact_pc, std::cref(imm_of),
                                       kconst);
                break;
            case Rdna2Format::SOP2:
                cfg_pcrel_scalar_binary(in, pcoff, std::cref(imm_of), std::cref(kill), fact_pc,
                                        pchi);
                break;
            case Rdna2Format::SOPK:
                cfg_pcrel_scalar_immediate(in, std::cref(kill), kconst, fact_pc);
                break;
            case Rdna2Format::SMEM:
                cfg_pcrel_scalar_load(in, pcoff, pchi, kconst, fact_pc, br_targets, dwords,
                                      required_dwords, out, code, std::cref(kill));
                break;
            case Rdna2Format::VOP1:
            case Rdna2Format::VOPC:
            case Rdna2Format::VOP3: cfg_pcrel_vector_writes(in, std::cref(kill)); break;
            case Rdna2Format::MTBUF:
                cfg_pcrel_typed_load(in, pcoff, pchi, kconst, fact_pc, br_targets, dwords,
                                     required_dwords, out, code);
                break;
            case Rdna2Format::MUBUF:
                cfg_pcrel_raw_load(in, pcoff, pchi, kconst, fact_pc, br_targets, dwords,
                                   required_dwords, out, code);
                break;
            default: break;   // formats that don't write SGPRs
        }
    }
    return out;
}

void cfg_pcrel_scalar_unary(
    const std::function<void(int)>& kill, const prosper::gpu::Rdna2Inst& in,
    std::unordered_map<int, uint64_t>& pcoff, std::unordered_set<int>& pchi,
    std::unordered_map<int, uint32_t>& fact_pc,
    const std::function<bool(const Operand&, const Rdna2Inst&, uint32_t*)>& imm_of,
    std::unordered_map<int, uint32_t>& kconst) {
    switch (in.fmt) {
        case Rdna2Format::SOP1:
            kill(in.dst.value);
            if (in.opcode == 0x1f) {   // s_getpc_b64: pair = address of NEXT inst
                kill(in.dst.value + 1);
                pcoff[in.dst.value] = (uint64_t)(in.pc + in.len_dwords) * 4u;
                pchi.insert(in.dst.value + 1);
                fact_pc[in.dst.value] = in.pc;
                fact_pc[in.dst.value + 1] = in.pc;
            } else if (in.opcode == 0x03) {   // s_mov_b32 <const>
                uint32_t v;
                if (imm_of(in.src[0], in, &v)) {
                    kconst[in.dst.value] = v;
                    fact_pc[in.dst.value] = in.pc;
                }
            } else if (in.opcode == 0x04) {   // s_mov_b64 (pair write)
                kill(in.dst.value + 1);
            }
            break;
        default: break;
    }
}

void cfg_pcrel_scalar_binary(
    const prosper::gpu::Rdna2Inst& in, std::unordered_map<int, uint64_t>& pcoff,
    const std::function<bool(const Operand&, const Rdna2Inst&, uint32_t*)>& imm_of,
    const std::function<void(int)>& kill, std::unordered_map<int, uint32_t>& fact_pc,
    std::unordered_set<int>& pchi) {
    switch (in.fmt) {
        case Rdna2Format::SOP2: {
            int d = in.dst.value;
            if (in.opcode == 0x00) {   // s_add_u32: pcrel-lo + immediate
                uint32_t imm;
                uint64_t base;
                bool got = false;
                if (in.src[0].kind == OperandKind::SGPR && pcoff.count(in.src[0].value) &&
                    imm_of(in.src[1], in, &imm)) {
                    base = pcoff[in.src[0].value];
                    got = true;
                } else if (in.src[1].kind == OperandKind::SGPR && pcoff.count(in.src[1].value) &&
                           imm_of(in.src[0], in, &imm)) {
                    base = pcoff[in.src[1].value];
                    got = true;
                }
                kill(d);
                if (got) {
                    pcoff[d] = base + imm;
                    fact_pc[d] = in.pc;
                }
            } else if (in.opcode == 0x04) {   // s_addc_u32: pcrel-hi + 0 (+carry; a real
                // carry needs the lo word to wrap over a <4 KB offset — not a shader-blob shape)
                uint32_t imm;
                bool hi_ok = (in.src[0].kind == OperandKind::SGPR && pchi.count(in.src[0].value) &&
                              imm_of(in.src[1], in, &imm) && imm == 0) ||
                             (in.src[1].kind == OperandKind::SGPR && pchi.count(in.src[1].value) &&
                              imm_of(in.src[0], in, &imm) && imm == 0);
                kill(d);
                if (hi_ok) {
                    pchi.insert(d);
                    fact_pc[d] = in.pc;
                }
            } else {
                kill(d);
                if (in.opcode == 0x29) kill(d + 1);   // s_bfe_u64 writes a pair
            }
            break;
        }
        default: break;
    }
}

void cfg_pcrel_scalar_immediate(const prosper::gpu::Rdna2Inst& in,
                                const std::function<void(int)>& kill,
                                std::unordered_map<int, uint32_t>& kconst,
                                std::unordered_map<int, uint32_t>& fact_pc) {
    switch (in.fmt) {
        case Rdna2Format::SOPK:
            // s_movk_i32 is also a common way to fill V# word2 (num_records). Astro Bot's
            // fullscreen-table VS uses it immediately before s_getpc_b64; treating every SOPK
            // as an unknown write prevented the otherwise-proven embedded table from folding.
            // The immediate is sign-extended by the ISA, matching the scalar register value.
            if (sopk_writes_scalar_data(in.opcode)) kill(in.dst.value);
            if (in.opcode == 0x00) {
                kconst[in.dst.value] = static_cast<uint32_t>(static_cast<int32_t>(in.simm16));
                fact_pc[in.dst.value] = in.pc;
            }
            break;
        default: break;
    }
}

void cfg_pcrel_vector_writes(const prosper::gpu::Rdna2Inst& in,
                             const std::function<void(int)>& kill) {
    switch (in.fmt) {
        case Rdna2Format::VOP1:
            if (in.opcode == 0x02) kill(in.dst.value);   // v_readfirstlane -> SGPR
            break;
        case Rdna2Format::VOPC:
            if (in.dst.kind == OperandKind::SGPR && in.dst.value <= 105) {
                kill(in.dst.value);
                kill(in.dst.value + 1);
            }   // e64 compare -> SGPR pair
            break;
        case Rdna2Format::VOP3:
            if (in.sdst.kind == OperandKind::SGPR) {
                kill(in.sdst.value);
                kill(in.sdst.value + 1);
            }
            break;
        default: break;
    }
}

}   // namespace prosper::gpu
