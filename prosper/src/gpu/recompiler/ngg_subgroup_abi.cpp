// ngg_subgroup_abi.cpp -- see ngg_subgroup_abi.hpp.
#include "gpu/recompiler/ngg_subgroup_abi.hpp"

#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace prosper::gpu {
namespace {

constexpr int kVcc = 106;
constexpr int kM0 = 124;
constexpr int kExecLo = 126;
constexpr int kExecHi = 127;
constexpr uint32_t kS3GsWaveId = 0x00ff0000u;

// The three launch VGPRs the shell does not supply: v4 (adjacency offsets 4/5) and v6/v7 (ES user
// VGPRs). Bit k of a mask names kTrackedVgpr[k].
constexpr int kTrackedVgpr[3] = {4, 6, 7};
uint8_t tracked_bit(int reg) {
    for (int k = 0; k < 3; ++k)
        if (kTrackedVgpr[k] == reg) return static_cast<uint8_t>(1u << k);
    return 0;
}

struct State {
    std::bitset<128> sdef;   // scalar registers written on EVERY path (launch values included)
    bool s3_overwritten = false;   // s3 no longer holds the launch value on every path
    bool exec_full = false;   // EXEC is all-ones on every path
    uint8_t vall = 0;   // tracked VGPRs written for all 64 lanes on every path
    uint8_t vcur = 0;   // tracked VGPRs written for every lane active in the current EXEC
    bool scc = false;   // SCC written on every path
    // Saved copies of EXEC (#3135): each slot's SGPR pair holds the EXEC of some earlier point on
    // every path (s_mov_b64 sP, exec or a SAVEEXEC), with `vcur` the tracked VGPRs written for
    // every lane of that EXEC. `exec_equal`: EXEC has not been written since, so a VGPR written now
    // covers that saved EXEC too. Restoring EXEC from a pair (s_mov_b64 exec, sP) brings back
    // exactly those lanes, and with them the slot's vcur -- the compiler's if/else idiom. Several
    // slots, because the idiom nests: Kena's 11562c72 saves the loop EXEC into s[4:5] and the inner
    // EXEC into s[6:7], restores s[6:7] inside the loop and s[4:5] before the back edge.
    struct SavedExec {
        int pair = -1;
        uint8_t vcur = 0;
        bool exec_equal = false;
        bool operator==(const SavedExec&) const = default;
    };
    std::array<SavedExec, 4> saved{};
    SavedExec* saved_slot(int pair) {
        for (SavedExec& slot : saved)
            if (slot.pair == pair) return &slot;
        return nullptr;
    }
    void meet(const State& o) {
        sdef &= o.sdef;
        scc = scc && o.scc;
        s3_overwritten = s3_overwritten && o.s3_overwritten;
        exec_full = exec_full && o.exec_full;
        vall &= o.vall;
        vcur &= o.vcur;
        // A pair saved on both paths keeps the definitions both paths had; any other is dropped.
        for (SavedExec& slot : saved) {
            if (slot.pair < 0) continue;
            const auto other =
                std::find_if(o.saved.begin(), o.saved.end(),
                             [&](const SavedExec& x) { return x.pair == slot.pair; });
            if (other == o.saved.end()) {
                slot = SavedExec{};
                continue;
            }
            slot.vcur &= other->vcur;
            slot.exec_equal = slot.exec_equal && other->exec_equal;
        }
    }
    bool operator==(const State& o) const {
        return sdef == o.sdef && s3_overwritten == o.s3_overwritten && exec_full == o.exec_full &&
               vall == o.vall && vcur == o.vcur && scc == o.scc && saved == o.saved;
    }
};

bool constant_operand(const Rdna2Inst& in, const Operand& op, uint32_t& value) {
    if (op.kind == OperandKind::InlineInt) {
        value = static_cast<uint32_t>(op.value);
        return true;
    }
    if (op.kind == OperandKind::Literal) {
        value = in.literal;
        return true;
    }
    return false;
}

uint32_t field_mask(uint32_t offset, uint32_t width) {
    if (width == 0 || offset >= 32) return 0;
    if (width >= 32 - offset) return 0xffffffffu << offset;
    return ((1u << width) - 1u) << offset;
}

uint32_t sdwa_select_mask(uint8_t select) {
    if (select <= 3) return 0xffu << (8u * select);
    if (select == 4) return 0xffffu;
    if (select == 5) return 0xffff0000u;
    return 0xffffffffu;
}

// Which bits of the 32-bit scalar operand `index` the instruction can observe. Only a few exact
// shapes narrow it; everything else demands the whole register. Used for s3, whose [23:16] the shell
// does not supply.
uint32_t demanded_bits(const Rdna2Inst& in, uint32_t index) {
    uint32_t c = 0;
    if (in.fmt == Rdna2Format::SOP2 && index == 0 && constant_operand(in, in.src[1], c)) {
        switch (in.opcode) {
            case 0x1e: return 0xffffffffu >> (c & 31u);   // s_lshl_b32
            case 0x20: return 0xffffffffu << (c & 31u);   // s_lshr_b32
            case 0x27:
            case 0x28: return field_mask(c & 31u, (c >> 16) & 0x7fu);   // s_bfe_[ui]32
            case 0x0e: return c;   // s_and_b32
            default: break;
        }
    }
    if (in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0e && index == 1 &&
        constant_operand(in, in.src[0], c))
        return c;
    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x148 && index == 0) {   // v_bfe_u32
        uint32_t offset = 0, width = 0;
        if (constant_operand(in, in.src[1], offset) && constant_operand(in, in.src[2], width))
            return field_mask(offset & 31u, width & 31u);
    }
    if (in.has_sdwa && (in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 ||
                        in.fmt == Rdna2Format::VOPC)) {
        if (index == 0) return sdwa_select_mask(in.sdwa_src0_sel);
        if (index == 1) return sdwa_select_mask(in.sdwa_src1_sel);
    }
    return 0xffffffffu;
}

bool vector_format(Rdna2Format fmt) {
    switch (fmt) {
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
        case Rdna2Format::VOP3P:
        case Rdna2Format::VOPC:
        case Rdna2Format::DS:
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
        case Rdna2Format::MIMG:
        case Rdna2Format::FLAT:
        case Rdna2Format::EXP:
        case Rdna2Format::VINTRP: return true;
        default: return false;
    }
}

// Data dwords one DS DATA field carries. Unknown opcodes take the widest packet (fail-closed for
// the read inventory, which may only over-report).
uint32_t ds_data_dwords(uint32_t opcode) {
    if (opcode == 0x35 || (opcode >= 0x36 && opcode <= 0x3f) ||
        (opcode >= 0x76 && opcode <= 0x78) || (opcode >= 0xa2 && opcode <= 0xa7) ||
        opcode == 0xb1 || opcode == 0xfe || opcode == 0xff)
        return 0;   // loads, swizzle (source is ADDR), append/consume/ordered count
    if (opcode <= 0x34 || opcode == 0xa0 || opcode == 0xa1 || (opcode >= 0xb0 && opcode <= 0xb3))
        return 1;
    if (opcode >= 0x40 && opcode <= 0x6f) return 2;
    if (opcode == 0xde) return 3;
    return 4;
}

// Full-dword VGPR results of a DS load, by opcode. Partial (D16) loads and anything not listed write
// no definite value.
uint32_t ds_full_result_dwords(uint32_t opcode) {
    switch (opcode) {
        case 0x35:
        case 0x36:
        case 0x39:
        case 0x3a:
        case 0x3b:
        case 0x3c:
        case 0xb1:
        case 0xb2:
        case 0xb3: return 1;
        case 0x37:
        case 0x38:
        case 0x76: return 2;
        case 0xfe: return 3;
        case 0x77:
        case 0x78:
        case 0xff: return 4;
        default: return opcode >= 0x20 && opcode <= 0x34 ? 1u : 0u;   // 32-bit returning atomics
    }
}

bool vop_rmw_destination(const Rdna2Inst& in) {
    // Operations that read their own destination: MAC/FMAC/DOT accumulators, lane writes, permlane
    // (preserves out-of-row lanes), swaps, DPP (masked rows/banks keep the old value), SDWA with a
    // partial destination, and VOP3 f16 results selecting the high half.
    if (in.has_dpp) return true;
    if (in.has_sdwa && (in.sdwa_dst_sel != 6 || in.sdwa_dst_unused == 2)) return true;
    const uint32_t vop2 = in.fmt == Rdna2Format::VOP2 ? in.opcode
                          : in.fmt == Rdna2Format::VOP3 && in.opcode >= 0x100 && in.opcode < 0x140
                              ? in.opcode - 0x100
                              : UINT32_MAX;
    if (vop2 == 0x02 || vop2 == 0x06 || vop2 == 0x0d || vop2 == 0x1f || vop2 == 0x2b ||
        vop2 == 0x36 || vop2 == 0x37 || vop2 == 0x3c)
        return true;
    if (in.fmt == Rdna2Format::VOP3 &&
        (in.opcode == 0x361 || in.opcode == 0x377 || in.opcode == 0x378 || (in.vop3p_opsel & 0x8u)))
        return true;
    if (in.fmt == Rdna2Format::VOP1 && (in.opcode == 0x65 || in.opcode == 0x68)) return true;
    return false;
}

// Number of VGPRs a vector source names, classified per form; 0 means "not classified", and the
// caller refuses the program. Every 64-bit-operand form of the gfx10.3 VALU is listed (RDNA2 ISA,
// VOPC/VOP1/VOP3 opcode tables); VOP2 and VOP3P have none. CONFIDENCE: MED on the completeness of
// those lists, which is why the VOP3-only opcode space outside them is refused rather than charged.
uint32_t vgpr_source_span(const Rdna2Inst& in, uint32_t index) {
    if (in.fmt == Rdna2Format::DS) return index == 0 ? 1u : ds_data_dwords(in.opcode);
    const bool vop3 = in.fmt == Rdna2Format::VOP3;
    // VOPC, natively or VOP3-encoded (opcode < 0x100): f64 compares and classes 0x20-0x3f, i64
    // 0xa0-0xbf (class_f64 0xa8/0xb8 takes a 32-bit mask in src1), u64 0xe0-0xf7.
    if (in.fmt == Rdna2Format::VOPC || (vop3 && in.opcode < 0x100)) {
        const uint32_t op = in.opcode;
        if ((op == 0xa8 || op == 0xb8) && index == 1) return 1;
        if ((op >= 0x20 && op <= 0x3f) || (op >= 0xa0 && op <= 0xbf) || (op >= 0xe0 && op <= 0xf7))
            return 2;
        return 1;
    }
    // VOP1, natively or VOP3-encoded (0x180 + op): the f64 sources.
    if (in.fmt == Rdna2Format::VOP1 || (vop3 && in.opcode >= 0x180 && in.opcode < 0x200)) {
        const uint32_t op = in.fmt == Rdna2Format::VOP1 ? in.opcode : in.opcode - 0x180;
        switch (op) {
            case 0x03:
            case 0x0f:
            case 0x15:   // cvt_i32/f32/u32 from f64
            case 0x17:
            case 0x18:
            case 0x19:
            case 0x1a:   // trunc/ceil/rndne/floor_f64
            case 0x2f:
            case 0x31:
            case 0x34:   // rcp/rsq/sqrt_f64
            case 0x3c:
            case 0x3d:
            case 0x3e:   // frexp_exp/frexp_mant/fract_f64
                return 2;
            default: return 1;
        }
    }
    if (in.fmt == Rdna2Format::VOP2 || in.fmt == Rdna2Format::VOP3P) return 1;
    if (!vop3) return 1;
    if (in.opcode >= 0x100 && in.opcode < 0x140) return 1;   // VOP3-encoded VOP2
    if (in.opcode >= 0x140 && in.opcode < 0x180) {
        switch (in.opcode) {
            case 0x14c:
            case 0x160:
            case 0x16e:
            case 0x170:   // fma/div_fixup/div_scale/div_fmas f64
                return 2;
            case 0x164:
            case 0x165:
            case 0x166:
            case 0x167:   // add/mul/min/max_f64
                return index < 2 ? 2u : 1u;
            case 0x168:
            case 0x174:   // ldexp_f64 / trig_preop_f64: src0 is f64
                return index == 0 ? 2u : 1u;
            case 0x172:
            case 0x173:   // qsad_pk_u16_u8 / mqsad_pk_u16_u8: src0, src2 are b64
                return index == 1 ? 1u : 2u;
            case 0x175:   // mqsad_u32_u8: src0 b64, src2 b128
                return index == 0 ? 2u : index == 2 ? 4u : 1u;
            case 0x176:
            case 0x177:   // mad_u64_u32 / mad_i64_i32: src2 is 64-bit
                return index == 2 ? 2u : 1u;
            case 0x161:
            case 0x162:
            case 0x163: return 0;   // not in the gfx10.3 table
            // 0x178 v_xor3_b32 (three 32-bit sources); 0x179-0x17f are not classified.
            default: return in.opcode <= 0x178 ? 1u : 0u;
        }
    }
    if (in.opcode == 0x2ff || in.opcode == 0x300 || in.opcode == 0x301)   // b64 shifts: src1
        return index == 1 ? 2u : 1u;
    if (in.opcode >= 0x300) return 1;   // the 32-bit VOP3-only tail (add3, lshl_add, permlane...)
    return 0;   // 0x200-0x2fe: VOP3 interpolation and unassigned space
}

// Definite full-dword VGPR results for the lanes in EXEC (under-reporting is the safe direction).
uint32_t definite_vgpr_results(const Rdna2Inst& in) {
    if (in.dst.kind != OperandKind::VGPR || in.dst.value < 0) return 0;
    switch (in.fmt) {
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
        case Rdna2Format::VOP3P:
            // Partial results: DPP (masked rows/banks keep the old value), SDWA sub-dword or
            // preserving destinations, single-lane/permuted lane writes and f16 high-half results.
            if (in.has_dpp || (in.has_sdwa && (in.sdwa_dst_sel != 6 || in.sdwa_dst_unused == 2)) ||
                (in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x361 || in.opcode == 0x377 ||
                                                 in.opcode == 0x378 || (in.vop3p_opsel & 0x8u))) ||
                dynamic_vgpr_destination(in))
                return 0;
            return rdna2_vgpr_write_count(in);
        case Rdna2Format::DS: return ds_full_result_dwords(in.opcode);
        case Rdna2Format::MUBUF:
            if (in.mubuf_lds || in.opcode > 0x0f || (in.opcode > 0x03 && in.opcode < 0x08))
                return 0;
            return rdna2_vgpr_write_count(in);
        case Rdna2Format::MTBUF: return in.opcode <= 0x03 ? rdna2_vgpr_write_count(in) : 0u;
        case Rdna2Format::MIMG:
            return in.mimg_d16 || rdna2_instruction_may_write_memory(in)
                       ? 0u
                       : rdna2_vgpr_write_count(in);
        default: return 0;
    }
}

enum class ExecEffect : uint8_t { None, Full, Narrow, Other };

ExecEffect exec_effect(const Rdna2Inst& in, bool& writes_lo, bool& writes_hi) {
    writes_lo = writes_hi = false;
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        for (uint32_t w = 0; w < width; ++w) {
            writes_lo |= base + static_cast<int>(w) == kExecLo;
            writes_hi |= base + static_cast<int>(w) == kExecHi;
        }
    });
    const bool cmpx = (in.fmt == Rdna2Format::VOPC || in.fmt == Rdna2Format::VOP3) &&
                      in.opcode < 0x100 && vopc_is_cmpx(in.opcode);
    const bool saveexec = in.fmt == Rdna2Format::SOP1 && sop1_opcode_writes_exec_b64(in.opcode);
    if (cmpx || saveexec) writes_lo = writes_hi = true;
    if (!writes_lo && !writes_hi && !rdna2_instruction_may_change_exec(in)) return ExecEffect::None;
    if (exec_write_sets_full_mask(in)) return ExecEffect::Full;
    const auto is_exec = [](const Operand& op) {
        return (op.kind == OperandKind::SGPR || op.kind == OperandKind::Special) &&
               op.value == kExecLo;
    };
    if (cmpx) return ExecEffect::Narrow;
    if (in.fmt == Rdna2Format::SOP1 &&
        (in.opcode == kSop1OpcodeAndSaveexecB64 || in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
         in.opcode == kSop1OpcodeAndn1WrexecB64))
        return ExecEffect::Narrow;
    if (in.fmt == Rdna2Format::SOP2 && writes_lo && writes_hi &&
        ((in.opcode == kSop2OpcodeAndB64 && (is_exec(in.src[0]) || is_exec(in.src[1]))) ||
         (in.opcode == 0x15 && is_exec(in.src[0]))))   // s_andn2_b64 exec, exec, x
        return ExecEffect::Narrow;
    return ExecEffect::Other;
}

struct ScalarRead {
    int reg;
    uint32_t demanded;
};
struct Reads {
    std::vector<ScalarRead> scalar;
    std::vector<int> vgpr;
    bool exec = false;
    bool scc = false;
    bool unclassified = false;   // a vector source whose width this inventory does not know
};

void add_scalar(Reads& reads, const Rdna2Inst& in, const Operand& op, uint32_t words,
                uint32_t index) {
    if (op.kind == OperandKind::Special && (op.value == 251 || op.value == 252)) {
        const int base = op.value == 251 ? kVcc : kExecLo;   // VCCZ / EXECZ
        reads.scalar.push_back({base, 0xffffffffu});
        reads.scalar.push_back({base + 1, 0xffffffffu});
        return;
    }
    if (op.kind != OperandKind::SGPR && op.kind != OperandKind::Special) return;
    if (op.value < 0 || op.value > kExecHi || op.value == 125) return;   // NULL and beyond
    for (uint32_t w = 0; w < words; ++w) {
        const int reg = op.value + static_cast<int>(w);
        if (reg > kExecHi) break;
        reads.scalar.push_back({reg, w == 0 ? demanded_bits(in, index) : 0xffffffffu});
    }
}

void add_vgpr(Reads& reads, const Operand& op, uint32_t span) {
    if (op.kind != OperandKind::VGPR || op.value < 0) return;
    for (uint32_t w = 0; w < span; ++w) reads.vgpr.push_back(op.value + static_cast<int>(w));
}

// SCC as an implicit input: s_cbranch_scc0/1, s_addc/s_subb, s_cselect, s_cmov, s_cmovk, and any
// operand naming SCC (253).
bool reads_scc(const Rdna2Inst& in) {
    for (uint32_t k = 0; k < in.n_src; ++k)
        if (in.src[k].kind == OperandKind::Special && in.src[k].value == 253) return true;
    switch (in.fmt) {
        case Rdna2Format::SOPP: return in.opcode == 0x04 || in.opcode == 0x05;
        case Rdna2Format::SOP2:
            return in.opcode == 0x04 || in.opcode == 0x05 || in.opcode == 0x0a || in.opcode == 0x0b;
        case Rdna2Format::SOP1:
            return in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64;
        case Rdna2Format::SOPK: return in.opcode == kSopkOpcodeCmovkI32;
        default: return false;
    }
}

// SCC as a definite output. Under-reporting a writer only refuses more programs.
bool writes_scc(const Rdna2Inst& in) {
    switch (in.fmt) {
        case Rdna2Format::SOPC: return true;
        case Rdna2Format::SOP1: return sop1_opcode_writes_scc(in.opcode);
        case Rdna2Format::SOP2:
            // add/sub/min/max, the bitwise forms, shifts, bfe, absdiff and lshlN_add write SCC;
            // cselect, bfm, mul_i32, pack and mul_hi do not.
            return in.opcode <= 0x09 || (in.opcode >= 0x0e && in.opcode <= 0x23) ||
                   (in.opcode >= 0x27 && in.opcode <= 0x2a) || in.opcode == 0x2c ||
                   (in.opcode >= 0x2e && in.opcode <= 0x31);
        case Rdna2Format::SOPK: return in.opcode >= 0x03 && in.opcode <= 0x0f;   // cmpk, addk
        default: return false;
    }
}

Reads instruction_reads(const Rdna2Inst& in) {
    Reads reads;
    reads.exec = vector_format(in.fmt);
    reads.scc = reads_scc(in);
    switch (in.fmt) {
        case Rdna2Format::SOP1:
        case Rdna2Format::SOP2:
        case Rdna2Format::SOPC:
            for (uint32_t k = 0; k < in.n_src; ++k) {
                const uint32_t words = scalar_alu_source_words(in, k);
                if (words != UINT32_MAX) add_scalar(reads, in, in.src[k], words ? words : 2u, k);
            }
            if (in.fmt == Rdna2Format::SOP1 && sop1_opcode_writes_exec_b64(in.opcode))
                reads.exec = true;
            if (const uint32_t implicit = scalar_implicit_destination_read_width(in))
                add_scalar(reads, in, in.dst, implicit, 3);
            break;
        case Rdna2Format::SOPK:
            if (const uint32_t implicit = scalar_implicit_destination_read_width(in))
                add_scalar(reads, in, in.dst, implicit, 3);
            break;
        case Rdna2Format::SOPP:
            if (in.opcode == 0x08 || in.opcode == 0x09) reads.exec = true;   // EXECZ/NZ
            if (in.opcode == 0x06 || in.opcode == 0x07) {   // VCCZ/NZ
                reads.scalar.push_back({kVcc, 0xffffffffu});
                reads.scalar.push_back({kVcc + 1, 0xffffffffu});
            }
            // s_sendmsghalt / s_ttracedata read M0. GS_ALLOC_REQ's M0 has its own named check.
            if (in.opcode == 0x11 || in.opcode == 0x16) reads.scalar.push_back({kM0, 0xffffffffu});
            break;
        case Rdna2Format::SMEM:
            add_scalar(reads, in, in.src[0], smem_opcode_is_buffer_load(in.opcode) ? 4u : 2u, 0);
            add_scalar(reads, in, in.src[1], 1, 1);
            break;
        case Rdna2Format::VOP1:
        case Rdna2Format::VOP2:
        case Rdna2Format::VOP3:
        case Rdna2Format::VOP3P:
        case Rdna2Format::VOPC: {
            // A VOP3 encoding always carries three source fields; the ones its opcode does not
            // read decode as s0 and must not count as reads (a phantom s0 refuses the program).
            const uint32_t sources =
                in.fmt == Rdna2Format::VOP3
                    ? std::min<uint32_t>(in.n_src, vop3_architectural_source_count(in.opcode))
                    : in.n_src;
            for (uint32_t k = 0; k < sources; ++k) {
                const uint32_t words = scalar_alu_source_words(in, k);
                if (words != UINT32_MAX) add_scalar(reads, in, in.src[k], words ? words : 2u, k);
                if (in.src[k].kind != OperandKind::VGPR) continue;
                const uint32_t span = vgpr_source_span(in, k);
                if (!span) reads.unclassified = true;
                add_vgpr(reads, in.src[k], std::max(1u, span));
            }
            // Implicit VCC: VOP2 cndmask and carry-in in every encoding that has no src2 (e32,
            // SDWA, DPP), and v_div_fmas_f32/f64.
            if ((in.fmt == Rdna2Format::VOP2 &&
                 (in.opcode == 0x01 || (in.opcode >= 0x28 && in.opcode <= 0x2a))) ||
                (in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x16f || in.opcode == 0x170))) {
                reads.scalar.push_back({kVcc, 0xffffffffu});
                reads.scalar.push_back({kVcc + 1, 0xffffffffu});
            }
            if (vop_rmw_destination(in))
                add_vgpr(reads, in.dst, std::max(1u, rdna2_vgpr_write_count(in)));
            break;
        }
        case Rdna2Format::DS:
            for (uint32_t k = 0; k < in.n_src; ++k)
                add_vgpr(reads, in.src[k], vgpr_source_span(in, k));
            if (in.opcode == 0x3d || in.opcode == 0x3e || in.opcode == 0x3f || in.opcode == 0xb0 ||
                in.opcode == 0xb1 || (in.opcode >= 0x19 && in.opcode <= 0x1d))
                reads.scalar.push_back({kM0, 0xffffffffu});
            if (in.opcode >= 0xa2 && in.opcode <= 0xa7) add_vgpr(reads, in.dst, 1);   // D16 merge
            break;
        case Rdna2Format::MUBUF:
        case Rdna2Format::MTBUF:
            // VADDR holds one register per enabled OFFEN / IDXEN (the decoder packs them at bits
            // 12 / 13 of `literal`).
            add_vgpr(reads, in.src[0], ((in.literal >> 12) & 1u) + ((in.literal >> 13) & 1u));
            add_scalar(reads, in, in.src[1], 4, 1);
            add_scalar(reads, in, in.src[2], 1, 2);
            if (rdna2_instruction_may_write_memory(in))
                add_vgpr(reads, in.dst, std::max(1u, rdna2_vgpr_destination_span(in)));
            break;
        case Rdna2Format::MIMG:
            // A non-NSA address runs from VADDR for up to 13 dwords depending on the opcode, dim
            // and A16; charge it through v7, the highest launch VGPR this check guards.
            add_vgpr(reads, in.src[0],
                     in.src[0].value >= 0 && in.src[0].value < 7 ? 8u - in.src[0].value : 4u);
            for (uint32_t k = 0; k < in.mimg_nsa && k < 3; ++k)
                for (uint32_t b = 0; b < 4; ++b)
                    reads.vgpr.push_back(static_cast<int>((in.words[2 + k] >> (8 * b)) & 0xffu));
            add_scalar(reads, in, in.src[1], in.mimg_r128 ? 4u : 8u, 1);
            if (in.opcode >= 0x20 && in.opcode <= 0x6f) add_scalar(reads, in, in.src[2], 4, 2);
            if (rdna2_instruction_may_write_memory(in) || in.mimg_d16)
                add_vgpr(reads, in.dst, std::max(1u, rdna2_vgpr_destination_span(in)));
            break;
        case Rdna2Format::FLAT:
            add_vgpr(reads, in.src[0], 2);
            add_scalar(reads, in, in.src[1], in.flat_segment == 1 ? 1u : 2u, 1);
            add_vgpr(reads, in.dst, 4);
            break;
        case Rdna2Format::EXP:
            for (uint32_t c = 0; c < 4; ++c) {
                if (!(in.exp_en & (1u << c))) continue;
                add_vgpr(reads, in.exp_compr ? in.src[c / 2] : in.src[c], 1);
            }
            break;
        default: break;
    }
    return reads;
}

bool is_exec_operand(const Operand& op) {
    return (op.kind == OperandKind::SGPR || op.kind == OperandKind::Special) && op.value == kExecLo;
}

// The even SGPR pair below EXEC that `op` names, or -1.
int sgpr_pair(const Operand& op) {
    return op.kind == OperandKind::SGPR && op.value >= 0 && op.value + 1 < kVcc &&
                   (op.value & 1) == 0
               ? op.value
               : -1;
}

void transfer(State& s, const Rdna2Inst& in) {
    if (writes_scc(in)) s.scc = true;
    if (const uint32_t results = definite_vgpr_results(in)) {
        for (uint32_t w = 0; w < results; ++w) {
            const uint8_t bit = tracked_bit(in.dst.value + static_cast<int>(w));
            s.vcur |= bit;
            if (s.exec_full) s.vall |= bit;
            for (State::SavedExec& slot : s.saved)
                if (slot.pair >= 0 && slot.exec_equal) slot.vcur |= bit;
        }
    }
    for_each_scalar_write(in, [&](int base, uint32_t width) {
        for (uint32_t w = 0; w < width; ++w) {
            const int reg = base + static_cast<int>(w);
            if (reg >= 0 && reg <= kExecHi) s.sdef.set(static_cast<size_t>(reg));
            if (reg == 3) s.s3_overwritten = true;
            for (State::SavedExec& slot : s.saved)
                if (slot.pair >= 0 && (reg == slot.pair || reg == slot.pair + 1))
                    slot = State::SavedExec{};   // the copy is gone
        }
    });
    // A save of EXEC: s_mov_b64 sP, exec (EXEC unchanged), or a SAVEEXEC (sP = the EXEC before
    // the instruction changes it).
    const bool sop1 = in.fmt == Rdna2Format::SOP1;
    const int dst_pair = sgpr_pair(in.dst);
    const bool copy_of_exec = sop1 && in.opcode == kSop1OpcodeMovB64 && is_exec_operand(in.src[0]);
    if (dst_pair >= 0 && (copy_of_exec || (sop1 && sop1_opcode_is_saveexec_b64(in.opcode)))) {
        // The write above already cleared a slot this pair had; take a free one. With every slot
        // in use the oldest record is dropped, which only forgets definitions (conservative).
        State::SavedExec* slot = s.saved_slot(-1);
        if (!slot) {
            std::rotate(s.saved.begin(), s.saved.begin() + 1, s.saved.end());
            slot = &s.saved.back();
        }
        *slot = {dst_pair, s.vcur, copy_of_exec};
    }
    const int restored_pair = sop1 && in.opcode == kSop1OpcodeMovB64 && in.dst.value == kExecLo
                                  ? sgpr_pair(in.src[0])
                                  : -1;
    State::SavedExec* const restore = restored_pair >= 0 ? s.saved_slot(restored_pair) : nullptr;
    // The shared writer inventory names SGPR destinations; the implicit VCC results of an e32
    // compare (encoded as the VCC special operand) and of the e32 carry operations are added here.
    const bool vopc_to_vcc = in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                             in.dst.kind == OperandKind::Special && in.dst.value == kVcc;
    const bool carry_to_vcc = in.fmt == Rdna2Format::VOP2 && !in.has_sdwa && !in.has_dpp &&
                              in.opcode >= 0x28 && in.opcode <= 0x2a;
    if (vopc_to_vcc || carry_to_vcc) {
        s.sdef.set(kVcc);
        s.sdef.set(kVcc + 1);
    }
    bool lo = false, hi = false;
    const ExecEffect effect = exec_effect(in, lo, hi);
    switch (effect) {
        case ExecEffect::None: break;
        case ExecEffect::Full:
            s.exec_full = true;
            s.vcur = s.vall;
            s.sdef.set(kExecLo);
            s.sdef.set(kExecHi);
            break;
        case ExecEffect::Narrow:
            s.exec_full = false;
            s.sdef.set(kExecLo);
            s.sdef.set(kExecHi);
            break;
        case ExecEffect::Other:
            s.exec_full = false;
            s.vcur = s.vall;
            if (lo) s.sdef.set(kExecLo);
            if (hi) s.sdef.set(kExecHi);
            break;
    }
    // Any EXEC change ends "EXEC is the saved copy" -- including the implicit EXEC writes no
    // explicit destination names (the B32 SAVEEXEC/WREXEC family), which leave lo/hi false.
    if (effect != ExecEffect::None)
        for (State::SavedExec& slot : s.saved) slot.exec_equal = false;
    if (restore) {
        s.vcur = s.vall | restore->vcur;
        restore->exec_equal = true;
    }
}

const char* sgpr_reason(int reg) {
    if (reg == 0 || reg == 1) return "ngg-abi-read-s0-s1";
    if (reg == 2) return "ngg-abi-read-s2";
    if (reg == 4 || reg == 5) return "ngg-abi-read-s4-s5";
    if (reg == 6 || reg == 7) return "ngg-abi-read-s6-s7";
    if (reg == kVcc || reg == kVcc + 1) return "ngg-abi-read-undefined-vcc";
    if (reg == kM0) return "ngg-abi-read-undefined-m0";
    if (reg >= 108 && reg <= 123) return "ngg-abi-read-ttmp";
    if (reg == kExecLo || reg == kExecHi) return "ngg-abi-exec-read-before-write";
    return "ngg-abi-read-undefined-sgpr";
}

bool is_branch(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOPP &&
           ((in.opcode >= 0x02 && in.opcode <= 0x09 && in.opcode != 0x03) ||
            (in.opcode >= 0x17 && in.opcode <= 0x1a));
}

void refuse(NggSubgroupAbiFacts& facts, const char* reason, uint32_t pc, const char* fmt = nullptr,
            int value = 0) {
    if (!facts.refusal.empty()) return;
    char detail[64] = {};
    if (fmt) std::snprintf(detail, sizeof(detail), fmt, value);
    char text[192];
    std::snprintf(text, sizeof(text), "reason=%s pc=%u%s%s", reason, pc, fmt ? " " : "", detail);
    facts.reason = reason;
    facts.refusal = text;
}

}   // namespace

NggSubgroupAbiFacts analyze_ngg_subgroup_abi(const std::vector<Rdna2Inst>& ins,
                                             const NggSubgroupAbiLaunch& launch) {
    NggSubgroupAbiFacts facts;
    if (ins.empty() || !ins.back().is_end) {
        refuse(facts, "ngg-abi-undecoded", ins.empty() ? 0u : ins.back().pc);
        return facts;
    }
    std::unordered_map<uint32_t, size_t> index_of;
    for (size_t i = 0; i < ins.size(); ++i) index_of[ins[i].pc] = i;

    // Instruction-level refusals that need no dataflow.
    std::map<uint32_t, const Rdna2Inst*> exports;   // target -> EXP
    for (const auto& in : ins) {
        if (in.fmt == Rdna2Format::Unknown) {
            refuse(facts, "ngg-abi-undecoded", in.pc);
            return facts;
        }
        if (rdna2_escapes_decoded_effects(in) ||
            (in.fmt == Rdna2Format::VOP1 && ((in.opcode >= 0x42 && in.opcode <= 0x44) ||
                                             in.opcode == 0x48 || in.opcode == 0x68))) {
            refuse(facts, "ngg-abi-control-transfer", in.pc);
            return facts;
        }
        if (is_branch(in) && !index_of.contains(branch_target(in))) {
            refuse(facts, "ngg-abi-cfg-unresolved", in.pc);
            return facts;
        }
        const bool gws_or_ordered = in.fmt == Rdna2Format::DS &&
                                    ((in.opcode >= 0x19 && in.opcode <= 0x1d) || in.opcode == 0x3f);
        if (rdna2_instruction_may_write_memory(in) ||
            (in.fmt == Rdna2Format::DS && (in.ds_gds || gws_or_ordered)) ||
            (in.fmt == Rdna2Format::FLAT && in.flat_segment == 1) ||
            (in.fmt == Rdna2Format::MUBUF && in.mubuf_lds)) {
            refuse(facts, "ngg-side-effect", in.pc);
            return facts;
        }
        if (in.fmt == Rdna2Format::SOPP && (in.opcode == 0x10 || in.opcode == 0x11)) {
            if (in.opcode != 0x10 || in.simm16 != 9) {
                refuse(facts, "ngg-sendmsg-unsupported", in.pc, "simm16=0x%x", in.simm16 & 0xffff);
                return facts;
            }
            facts.alloc_request_pcs.push_back(in.pc);
        }
        if (in.fmt != Rdna2Format::EXP) continue;
        if (in.exp_compr) {
            refuse(facts, "ngg-export-compressed", in.pc);
            return facts;
        }
        const uint32_t target = in.exp_target;
        const bool param = target >= kExpTargetParam0 && target < kExpTargetParam0 + 32u;
        if (target != kExpTargetPrim && target != kExpTargetPos0 && target != kExpTargetPos1 &&
            !param) {
            refuse(facts, "ngg-export-target-unsupported", in.pc, "target=%d",
                   static_cast<int>(target));
            return facts;
        }
        if (target == kExpTargetPrim && in.exp_en != 1u) {
            refuse(facts, "ngg-export-prim-channels", in.pc, "en=0x%x",
                   static_cast<int>(in.exp_en));
            return facts;
        }
        // POS1 carries point size (x), edge flag (y), the layer (z) and the viewport index (w);
        // only the layer is modelled downstream, so any other channel is refused here.
        if (target == kExpTargetPos1 && (in.exp_en & ~0x4u) != 0) {
            refuse(facts, "ngg-export-pos1-channels", in.pc, "en=0x%x",
                   static_cast<int>(in.exp_en));
            return facts;
        }
        if (!exports.emplace(target, &in).second) {
            refuse(facts, "ngg-export-duplicate-target", in.pc, "target=%d",
                   static_cast<int>(target));
            return facts;
        }
    }
    // An EXP inside any [target, branch] range of a backward branch can execute more than once.
    for (const auto& in : ins) {
        if (!is_branch(in) || branch_target(in) > in.pc) continue;
        for (const auto& [target, exp] : exports) {
            if (exp->pc >= branch_target(in) && exp->pc <= in.pc) {
                refuse(facts, "ngg-export-in-cycle", exp->pc, "loop-branch=%d",
                       static_cast<int>(in.pc));
                return facts;
            }
        }
    }
    if (!exports.contains(kExpTargetPrim)) {
        refuse(facts, "ngg-export-missing-prim", ins.back().pc);
        return facts;
    }
    if (!exports.contains(kExpTargetPos0)) {
        refuse(facts, "ngg-export-missing-pos0", ins.back().pc);
        return facts;
    }

    // MUST dataflow to a fixpoint (greatest fixpoint: unreached states start at TOP).
    State entry;
    entry.sdef.set(3);
    for (uint32_t k = 0; k < launch.user_sgprs && 8u + k <= 105u; ++k) entry.sdef.set(8u + k);
    if (launch.user_data_address_known) {
        entry.sdef.set(0);
        entry.sdef.set(1);
    }
    std::vector<State> in_state(ins.size());
    std::vector<bool> reached(ins.size(), false);
    in_state[0] = entry;
    reached[0] = true;
    std::vector<size_t> worklist{0};
    const auto successors = [&](size_t i) {
        std::vector<size_t> next;
        const Rdna2Inst& in = ins[i];
        if (in.is_end) return next;
        if (is_branch(in)) next.push_back(index_of.at(branch_target(in)));
        if (!(in.fmt == Rdna2Format::SOPP && in.opcode == 0x02) && i + 1 < ins.size())
            next.push_back(i + 1);
        return next;
    };
    while (!worklist.empty()) {
        const size_t i = worklist.back();
        worklist.pop_back();
        State out = in_state[i];
        transfer(out, ins[i]);
        for (size_t next : successors(i)) {
            if (!reached[next]) {
                reached[next] = true;
                in_state[next] = out;
                worklist.push_back(next);
                continue;
            }
            State joined = in_state[next];
            joined.meet(out);
            if (!(joined == in_state[next])) {
                in_state[next] = joined;
                worklist.push_back(next);
            }
        }
    }

    for (size_t i = 0; i < ins.size(); ++i) {
        if (!reached[i]) continue;
        const Rdna2Inst& in = ins[i];
        const State& s = in_state[i];
        const Reads reads = instruction_reads(in);
        if (reads.exec && !(s.sdef.test(kExecLo) && s.sdef.test(kExecHi))) {
            refuse(facts, "ngg-abi-exec-read-before-write", in.pc);
            return facts;
        }
        if (reads.unclassified) {
            refuse(facts, "ngg-abi-unclassified-vector-width", in.pc, "op=0x%x",
                   static_cast<int>(in.opcode));
            return facts;
        }
        if (reads.scc && !s.scc) {
            refuse(facts, "ngg-abi-read-undefined-scc", in.pc);
            return facts;
        }
        for (const ScalarRead& read : reads.scalar) {
            if (read.reg == 3 && !s.s3_overwritten && (read.demanded & kS3GsWaveId)) {
                refuse(facts, "ngg-abi-read-s3-gs-wave-id", in.pc, "demanded=0x%08x",
                       static_cast<int>(read.demanded));
                return facts;
            }
            if (!s.sdef.test(static_cast<size_t>(read.reg))) {
                refuse(facts, sgpr_reason(read.reg), in.pc, "s%d", read.reg);
                return facts;
            }
        }
        for (int reg : reads.vgpr) {
            const uint8_t bit = tracked_bit(reg);
            if (bit && !(s.vcur & bit)) {
                refuse(facts, reg == 4 ? "ngg-abi-read-v4" : "ngg-abi-read-v6-v7", in.pc, "v%d",
                       reg);
                return facts;
            }
        }
        if (in.fmt == Rdna2Format::SOPP && in.opcode == 0x10 && !s.sdef.test(kM0)) {
            refuse(facts, "ngg-sendmsg-m0-unproven", in.pc);
            return facts;
        }
    }
    // A merged NGG program must allocate its outputs; without GS_ALLOC_REQ every block it writes
    // would be invalid at runtime, so refuse it here where the refused-shader index can see it.
    if (facts.alloc_request_pcs.empty()) {
        refuse(facts, "ngg-sendmsg-missing", ins.back().pc);
        return facts;
    }
    // Every M0 writer must be a scalar ALU instruction, so the value GS_ALLOC_REQ sends is wave data
    // and not a lane-local or memory-loaded word.
    {
        for (const auto& in : ins) {
            bool writes_m0 = false;
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                writes_m0 |= base <= kM0 && kM0 < base + static_cast<int>(width);
            });
            if (writes_m0 && in.fmt != Rdna2Format::SOP1 && in.fmt != Rdna2Format::SOP2 &&
                in.fmt != Rdna2Format::SOPK) {
                refuse(facts, "ngg-sendmsg-m0-unproven", in.pc, "writer-fmt=%d",
                       static_cast<int>(in.fmt));
                return facts;
            }
        }
    }

    // The record layout: flags, PRIM, POS0, [POS1], PARAM targets ascending.
    NggExportRecordLayout& layout = facts.layout;
    uint32_t word = kNggRecordPos0Word + 4u;
    layout.prim_channels = exports.at(kExpTargetPrim)->exp_en;
    layout.pos0_channels = exports.at(kExpTargetPos0)->exp_en;
    if (const auto pos1 = exports.find(kExpTargetPos1); pos1 != exports.end()) {
        layout.pos1_word = word;
        layout.pos1_channels = pos1->second->exp_en;
        word += 4u;
    }
    for (const auto& [target, exp] : exports) {
        if (target < kExpTargetParam0) continue;
        if (layout.param_targets.size() == kNggMaxParamTargets) {
            refuse(facts, "ngg-export-too-many-params", exp->pc);
            return facts;
        }
        if (layout.param_targets.empty()) layout.first_param_word = word;
        layout.param_targets.push_back(target);
        layout.param_channels.push_back(exp->exp_en);
        word += 4u;
    }
    layout.words_per_lane = word;
    return facts;
}

}   // namespace prosper::gpu
