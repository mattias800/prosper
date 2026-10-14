// rdna2_instruction_effects.hpp — ONE table of what each RDNA2 instruction reads and writes.
//
// The recompiler has dozens of analyses that each need to know, per opcode, whether an instruction
// reads or writes SCC, how many SGPRs its destination receives, and which architectural registers
// it touches without naming them. Each used to keep a private opcode list, and those lists drifted:
// one SCC-reader list matched s_mov_b32 instead of s_cmov_b32, a scalar-width list wrote one dword
// for s_ashr_i64, and two copies of one width function disagreed about VOPC (#5146). This module is
// the single answer. Analyses ask it; they do not keep their own lists.
//
// Scope today: every opcode the decoder can produce in the scalar ALU formats SOP2, SOP1, SOPK,
// SOPC and SOPP, and in the VALU formats VOP1, VOP2, VOPC and VOP3. Each slot of those opcode
// spaces is either a named instruction or an explicit RESERVED encoding (one RDNA2 does not define,
// which the decoder still accepts because it only splits fields). `test_rdna2_instruction_effects`
// fails if a decodable opcode has neither. A VALU row describes its sources, its vector result
// (which lanes and VGPRs it writes, from what) and its implicit reads; see Rdna2EffectsStatus::Valu.
// Other formats have no per-opcode entries yet; the question helpers below answer the questions
// whose answer is a property of the whole format (no VALU or memory instruction touches SCC) and
// report Unknown otherwise. An opcode with no entry is never a silent "no effect": its status is
// Unknown and every helper says so.
//
// Primary source: AMD "RDNA 2" Instruction Set Architecture Reference Guide (doc 70648), the
// per-format opcode tables in chapter 12 (12.1 SOP2, 12.2 SOPK, 12.3 SOP1, 12.4 SOPC, 12.5 SOPP,
// 12.7-12.12 VALU) and the "Sets SCC?" tables in chapter 5 (5.3-5.8). Which opcode numbers
// exist was cross-checked by disassembling every encoding with `llvm-mc -disassemble`.
#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace prosper::gpu {

enum class Rdna2EffectsStatus : uint8_t {
    Unknown,   // no entry: every answer about this opcode is "unknown"; callers stay conservative
    Reserved,   // the decoder accepts the encoding but RDNA2 assigns no instruction to it
    Known,
    // A VALU instruction (VOP1, VOP2, VOPC, VOP3). Its name, source widths, vector-result fields
    // (lane_flow, vdst_dwords, vdst_half, accumulates, m0_relative) and implicit VCC/EXEC/M0 READS
    // are facts. Its scalar destination and implicit writes are NOT described and read as zero, so
    // known() stays false: ask describes_sources() / describes_vector_result() instead.
    Valu,
};

// Architectural registers an instruction reads or writes WITHOUT naming them in an operand.
// An operand that names one (an SSRC of 253 = SCC, 106 = VCC_LO, ...) is not recorded here; it is
// visible in the decoded operands.
enum Rdna2ImplicitRegister : uint8_t {
    kRdna2ImplicitNone = 0,
    kRdna2ImplicitScc = 1u << 0,
    kRdna2ImplicitVcc = 1u << 1,
    kRdna2ImplicitExec = 1u << 2,
    kRdna2ImplicitM0 = 1u << 3,
};

// What a VALU instruction does with the lanes of a wave (ISA 70648 12.7-12.12 pseudo-code).
enum class Rdna2LaneFlow : uint8_t {
    None,   // no lane result: v_nop, v_pipeflush, v_clrexcp (and every non-VALU row)
    LaneWise,   // each lane's result is a function of that lane's operands only (VOPC included)
    LaneIndex,   // lane-wise, and the lane's own index is an operand (v_mbcnt_lo/hi)
    ReadLane,   // an SGPR receives one lane's value (v_readfirstlane, v_readlane)
    WriteLane,   // one lane, chosen by an operand, is written regardless of EXEC (v_writelane)
    CrossLane,   // a lane reads another lane's source (v_permlane16, v_permlanex16)
    Relative,   // a source and/or destination VGPR is indexed by M0 (v_movrel*, v_swaprel)
    Swap,   // VDST and SRC0 exchange values in each lane (v_swap_b32)
    Interpolate,   // reads LDS attribute data addressed through M0 (the VOP3 v_interp_* forms)
};

// How an instruction consumes SCC implicitly. Data: SCC is an operand of the result (s_cselect,
// s_addc/s_subb carry-in, s_cmov/s_cmovk). Branch: SCC selects the next PC (s_cbranch_scc0/1).
enum class Rdna2SccRead : uint8_t { None, Data, Branch };

// The kind of EXEC update an instruction performs. SaveExec: D = old EXEC, then EXEC = op(S0, EXEC).
// WrExec: EXEC = op(S0, EXEC), then D = new EXEC. SubvectorLoop: s_subvector_loop_begin/end move
// EXEC halves through D.
enum class Rdna2ExecEffect : uint8_t { None, SaveExec, WrExec, SubvectorLoop };

// Whether a source or destination is addressed relative to M0 rather than at the encoded register:
// s_movrels reads SGPR[S0 + M0], s_movreld writes SGPR[D + M0], s_movrelsd_2 does both.
enum class Rdna2M0Relative : uint8_t { None, Source, Destination, Both };

struct Rdna2InstructionEffects {
    Rdna2EffectsStatus status = Rdna2EffectsStatus::Unknown;
    const char* name = nullptr;   // the llvm-mc mnemonic; nullptr unless Known
    Rdna2SccRead scc_read = Rdna2SccRead::None;
    uint8_t implicit_reads = kRdna2ImplicitNone;   // Rdna2ImplicitRegister bits; includes SCC
    uint8_t implicit_writes = kRdna2ImplicitNone;   // whenever scc_read != None / SCC is written
    // Consecutive SGPRs written through the decoded destination field (Rdna2Inst::dst). 0 when the
    // field is not written: it is a source (s_cmpk_*, s_setreg_b32, s_waitcnt_*) or unused.
    uint8_t sdst_dwords = 0;
    // Consecutive SGPRs read through source operand 0/1/2 when it names a scalar register
    // (SSRC0/SSRC1 for SALU, SRC0-SRC2 for VALU). 0 means the instruction has no such source.
    // SOPK and SOPP have no SSRC fields; SOPK forms that read their SDST field are not described
    // here yet. A VALU lane-mask source (SRC2 of v_cndmask_b32 and of the carry-in forms) is
    // charged the Wave64 pair.
    uint8_t ssrc_dwords[3] = {0, 0, 0};
    Rdna2M0Relative m0_relative = Rdna2M0Relative::None;
    Rdna2ExecEffect exec = Rdna2ExecEffect::None;
    // ---- Vector result (VALU rows; status Valu). ----
    Rdna2LaneFlow lane_flow = Rdna2LaneFlow::None;
    // Consecutive VGPRs the VDST field receives: 2 for a 64-bit result, 4 for v_mqsad_u32_u8,
    // 0 when VDST is not a VGPR result (VOPC writes a lane mask; v_readlane/v_readfirstlane write
    // an SGPR; v_nop has none).
    uint8_t vdst_dwords = 0;
    // The result is one 16-bit value (D.f16/u16/i16) written into one half of the VDST register:
    // the other half is not produced by this instruction. Packed results (v_pk_*, v_cvt_pk*,
    // v_pack_b32_f16, v_sat_pk_u8_i16) are full words.
    bool vdst_half = false;
    // The old VDST value of the same lane is an operand (v_mac/v_fmac/v_dot*c: D = ... + D).
    bool accumulates = false;
    // SRC0's VGPR is written too, with the old VDST (v_swap_b32, v_swaprel_b32).
    bool writes_src0 = false;

    bool known() const { return status == Rdna2EffectsStatus::Known; }
    bool describes_sources() const {
        return status == Rdna2EffectsStatus::Known || status == Rdna2EffectsStatus::Valu;
    }
    bool describes_vector_result() const { return status == Rdna2EffectsStatus::Valu; }
};

// The table entry for (fmt, opcode). Returns an entry with status Unknown for a format without a
// table or an opcode outside its format's space; never throws and never allocates.
const Rdna2InstructionEffects& rdna2_instruction_effects(Rdna2Format fmt, uint32_t opcode);
inline const Rdna2InstructionEffects& rdna2_instruction_effects(const Rdna2Inst& in) {
    return rdna2_instruction_effects(in.fmt, in.opcode);
}

// Formats whose whole decodable opcode space has entries: the scalar ALU formats with full
// effects, and VOP1, VOP2, VOPC and VOP3 with the VALU facts (status Valu).
bool rdna2_effects_format_is_tabulated(Rdna2Format fmt);
// The opcodes [first, end) the decoder can produce in a tabulated format: the opcode field minus
// the encodings another format claims first (SOP2 0x60-0x7F are SOPK/SOP1/SOPC/SOPP, SOPK
// 0x1D-0x1F are SOP1/SOPC/SOPP, VOP2 0x3E/0x3F are VOPC/VOP1, VOP3 0x000-0x0FF decode as VOPC).
// {0, 0} for a format without a table.
struct Rdna2OpcodeRange {
    uint32_t first = 0;
    uint32_t end = 0;
};
Rdna2OpcodeRange rdna2_effects_opcode_range(Rdna2Format fmt);

// Every row of the table: the named instructions, then one Reserved row per reserved opcode. For
// coverage and consistency tests.
struct Rdna2InstructionEffectsRow {
    Rdna2Format fmt;
    uint32_t opcode;
    Rdna2InstructionEffects effects;
};
std::span<const Rdna2InstructionEffectsRow> rdna2_instruction_effects_rows();

// ---- Question helpers ------------------------------------------------------------------------
// A three-valued answer. Unknown means the table cannot say; each caller decides which way is
// conservative for its own question and says so where it collapses Unknown to a bool.
enum class Rdna2Answer : uint8_t { No, Yes, Unknown };

// Does the instruction write SCC? Per entry for the scalar formats. Every other decoded format is
// No: SCC is written only by scalar ALU instructions (ISA 70648 sections 3.8 and 5.3). Unknown for
// a reserved or untabulated scalar opcode and for Rdna2Format::Unknown.
Rdna2Answer rdna2_writes_scc(Rdna2Format fmt, uint32_t opcode);
// Does the instruction read SCC implicitly, as data or as a branch condition? The same format rule
// as above: outside the scalar formats SCC is only ever read through an explicit operand (SSRC/SRC
// value 253), which this does not inspect.
Rdna2Answer rdna2_reads_scc_implicitly(Rdna2Format fmt, uint32_t opcode);
// The data-read half of the above: s_cselect, s_addc/s_subb, s_cmov, s_cmovk. Branches are No.
Rdna2Answer rdna2_reads_scc_as_data(Rdna2Format fmt, uint32_t opcode);

// True when any decoded operand names SCC explicitly (value 253, "{zeros, SCC}").
bool rdna2_operand_reads_scc(const Rdna2Inst& in);

// Does any instruction in `ins` read SCC, implicitly (as data or branch condition) or through an
// explicit SCC operand? A reserved or unknown scalar opcode counts as a reader: this is the
// question a caller asks before deciding it may skip computing SCC, so the safe answer is yes.
bool rdna2_program_reads_scc(const std::vector<Rdna2Inst>& ins);

// ---- VALU result questions ------------------------------------------------------------------
// Opcode-level answers; Unknown for a reserved or untabulated encoding and for a non-VALU format.
// The Rdna2Inst forms below add the encoding's own modifiers (DPP, SDWA destination select).

// Is each written lane a function of that same lane's operands only (and, for an accumulator, its
// own old VDST)? Yes for LaneWise, LaneIndex and Swap; No for the cross-lane, selected-lane,
// M0-relative and interpolation flows and for an instruction with no lane result.
Rdna2Answer rdna2_valu_lane_mapped(Rdna2Format fmt, uint32_t opcode);
// Does the result depend on which lanes exist or are active, rather than only on the lane's own
// operands? Yes for ReadLane, WriteLane, CrossLane and LaneIndex.
Rdna2Answer rdna2_valu_lane_dependent(Rdna2Format fmt, uint32_t opcode);
// Does any VDST word written keep (part of) its old value: an accumulator, a 16-bit result
// (the other half), a selected-lane write (the other lanes) or an M0-relative destination?
Rdna2Answer rdna2_valu_result_keeps_old_vdst(Rdna2Format fmt, uint32_t opcode);
// VGPRs that receive a complete, lane-wise result computed from the explicit operands alone
// (no accumulator, no 16-bit half): vdst_dwords for such an instruction, 0 otherwise or when the
// table cannot say.
uint32_t rdna2_valu_complete_result_dwords(Rdna2Format fmt, uint32_t opcode);
// The same for one decoded instruction: 0 for a DPP form (masked rows/banks keep old lanes) and
// for an SDWA destination that is not a padded whole dword; a 16-bit result counts as complete
// only through the SDWA DWORD + UNUSED_PAD form, which zero-fills the other half.
uint32_t rdna2_valu_complete_result_dwords(const Rdna2Inst& in);
// Does a VALU instruction write a VGPR chosen at run time by M0 rather than its encoded VDST
// (v_movreld, v_movrelsd, v_movrelsd_2, v_swaprel in every encoding)? Storage inventories treat
// such a write as reaching every VGPR from the encoded register up.
bool rdna2_writes_m0_relative_vgpr(const Rdna2Inst& in);

}   // namespace prosper::gpu
