// rdna2_instruction_effects.cpp — the per-opcode effects table. See the header for the contract.
//
// Every row names its source. "ISA 12.x" is the per-format opcode table of AMD doc 70648, which
// gives each instruction's pseudo-code (what D, SCC, EXEC and M0 receive); "ISA 5.x" is the
// chapter-5 summary tables with their "Sets SCC?" column. The opcode numbers and the reserved
// holes were cross-checked against `llvm-mc -mcpu=gfx1030 -disassemble` over each whole opcode
// space; where the two sources differ the row says so and carries a CONFIDENCE mark.
#include "gpu/recompiler/rdna2_instruction_effects.hpp"

#include "gpu/recompiler/ir/effects/operand_effects.hpp"

#include <array>
#include <cstddef>

namespace prosper::gpu {
namespace {

using F = Rdna2Format;
using Row = Rdna2InstructionEffectsRow;

constexpr uint8_t kScc = kRdna2ImplicitScc;
constexpr uint8_t kVcc = kRdna2ImplicitVcc;
constexpr uint8_t kExec = kRdna2ImplicitExec;
constexpr uint8_t kM0 = kRdna2ImplicitM0;

// One named instruction. `dst`/`s0`/`s1` are dword counts; `reads`/`writes` are implicit registers.
constexpr Rdna2InstructionEffects op(const char* name, uint8_t dst, uint8_t s0, uint8_t s1,
                                     uint8_t reads = 0, uint8_t writes = 0,
                                     Rdna2SccRead scc_read = Rdna2SccRead::None,
                                     Rdna2ExecEffect exec = Rdna2ExecEffect::None,
                                     Rdna2M0Relative relative = Rdna2M0Relative::None) {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::Known;
    e.name = name;
    e.scc_read = scc_read;
    e.implicit_reads = reads;
    e.implicit_writes = writes;
    e.sdst_dwords = dst;
    e.ssrc_dwords[0] = s0;
    e.ssrc_dwords[1] = s1;
    e.ssrc_dwords[2] = 0;
    e.m0_relative = relative;
    e.exec = exec;
    return e;
}
// Common shapes.
constexpr Rdna2InstructionEffects alu(const char* name, uint8_t dst, uint8_t s0, uint8_t s1,
                                      bool writes_scc) {
    return op(name, dst, s0, s1, 0, writes_scc ? kScc : 0);
}
constexpr Rdna2InstructionEffects scc_data(const char* name, uint8_t dst, uint8_t s0, uint8_t s1,
                                           bool writes_scc = false) {
    return op(name, dst, s0, s1, kScc, writes_scc ? kScc : 0, Rdna2SccRead::Data);
}
constexpr Rdna2InstructionEffects saveexec(const char* name, uint8_t width) {
    return op(name, width, width, 0, kExec, kExec | kScc, Rdna2SccRead::None,
              Rdna2ExecEffect::SaveExec);
}
constexpr Rdna2InstructionEffects wrexec(const char* name, uint8_t width) {
    return op(name, width, width, 0, kExec, kExec | kScc, Rdna2SccRead::None,
              Rdna2ExecEffect::WrExec);
}
constexpr Rdna2InstructionEffects compare(const char* name, uint8_t s0, uint8_t s1) {
    return op(name, 0, s0, s1, 0, kScc);
}
constexpr Rdna2InstructionEffects plain(const char* name) {
    return op(name, 0, 0, 0);
}
// The same row with its PC effect recorded.
constexpr Rdna2InstructionEffects control(Rdna2InstructionEffects e, Rdna2ControlFlow flow) {
    e.control = flow;
    return e;
}
// One SMEM instruction: what SDATA receives, the SBASE width (a pair, or a V# quad) and the
// SOFFSET register (one dword), when the instruction has them.
constexpr Rdna2InstructionEffects smem(const char* name, Rdna2ScalarMemory kind, uint8_t dst,
                                       uint8_t base, uint8_t offset) {
    Rdna2InstructionEffects e = op(name, dst, base, offset);
    e.smem = kind;
    return e;
}
// The same row with SRC2 recorded as a lane mask.
constexpr Rdna2InstructionEffects lane_mask_src2(Rdna2InstructionEffects e) {
    e.lane_mask_src2 = true;
    return e;
}
// A VALU instruction: its source widths, and by default the common result shape -- one VGPR
// written lane-wise from the explicit operands. The wrappers below state every other shape.
constexpr Rdna2InstructionEffects valu(const char* name, uint8_t s0, uint8_t s1, uint8_t s2) {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::Valu;
    e.name = name;
    e.ssrc_dwords[0] = s0;
    e.ssrc_dwords[1] = s1;
    e.ssrc_dwords[2] = s2;
    e.lane_flow = Rdna2LaneFlow::LaneWise;
    e.vdst_dwords = 1;
    return e;
}
// A VOPC compare: lane-wise, and its result is a lane mask, not a VGPR.
constexpr Rdna2InstructionEffects vcmp(const char* name, uint8_t s0, uint8_t s1) {
    Rdna2InstructionEffects e = valu(name, s0, s1, 0);
    e.vdst_dwords = 0;
    e.scalar_result = {Rdna2ScalarDestination::Dst, 2, true};
    return e;
}
#include "gpu/recompiler/ir/effects/valu_results.inc"

// A VALU instruction with no lane result and no operands (v_nop, v_pipeflush, v_clrexcp).
constexpr Rdna2InstructionEffects no_result(const char* name) {
    Rdna2InstructionEffects e = valu(name, 0, 0, 0);
    e.lane_flow = Rdna2LaneFlow::None;
    e.vdst_dwords = 0;
    return e;
}
// VDST receives `dwords` consecutive VGPRs (0: the result is not a VGPR).
constexpr Rdna2InstructionEffects wide(Rdna2InstructionEffects e, uint8_t dwords) {
    e.vdst_dwords = dwords;
    return e;
}
// The result is a 16-bit value written into one half of VDST.
constexpr Rdna2InstructionEffects half(Rdna2InstructionEffects e) {
    e.vdst_half = true;
    return e;
}
// The old VDST of the same lane is an operand.
constexpr Rdna2InstructionEffects accumulate(Rdna2InstructionEffects e) {
    e.accumulates = true;
    return e;
}
constexpr Rdna2InstructionEffects flow(Rdna2InstructionEffects e, Rdna2LaneFlow lanes,
                                       uint8_t reads = 0) {
    e.lane_flow = lanes;
    if (lanes == Rdna2LaneFlow::ReadLane) e.scalar_result = {Rdna2ScalarDestination::Dst, 1, false};
    e.implicit_reads = reads;
    return e;
}
// SRC0's VGPR also receives a value: the old VDST (v_swap_b32, v_swaprel_b32).
constexpr Rdna2InstructionEffects swaps(Rdna2InstructionEffects e) {
    e.writes_src0 = true;
    return e;
}
constexpr Rdna2InstructionEffects relative(Rdna2InstructionEffects e, Rdna2M0Relative which) {
    e.lane_flow = Rdna2LaneFlow::Relative;
    e.m0_relative = which;
    e.implicit_reads = kM0;
    return e;
}
constexpr Rdna2InstructionEffects reserved() {
    Rdna2InstructionEffects e;
    e.status = Rdna2EffectsStatus::Reserved;
    return e;
}

#include "gpu/recompiler/ir/effects/scalar_alu.inc"
#include "gpu/recompiler/ir/effects/scalar_memory.inc"
#include "gpu/recompiler/ir/effects/vector_alu.inc"
#include "gpu/recompiler/ir/effects/vector_compare.inc"
#include "gpu/recompiler/ir/effects/vector3.inc"
#include "gpu/recompiler/ir/effects/packed.inc"
#include "gpu/recompiler/ir/effects/memory_constructors.inc"
#include "gpu/recompiler/ir/effects/ds.inc"
#include "gpu/recompiler/ir/effects/buffer.inc"
#include "gpu/recompiler/ir/effects/typed_buffer.inc"
#include "gpu/recompiler/ir/effects/flat.inc"
#include "gpu/recompiler/ir/effects/image.inc"
#include "gpu/recompiler/ir/effects/misc_memory.inc"
#include "gpu/recompiler/ir/effects/reserved.inc"

#include "gpu/recompiler/ir/effects/lookup.inc"
}   // namespace

const Rdna2InstructionEffects& rdna2_instruction_effects(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2InstructionEffects* entry = entry_for(fmt, opcode);
    return entry ? *entry : kUnknown;
}

bool rdna2_effects_format_is_tabulated(Rdna2Format fmt) {
    return range_for(fmt).end != 0;
}

Rdna2OpcodeRange rdna2_effects_opcode_range(Rdna2Format fmt) {
    return range_for(fmt);
}

std::span<const Rdna2InstructionEffectsRow> rdna2_instruction_effects_rows() {
    static const std::vector<Row> rows = all_rows();
    return rows;
}

namespace {
const Rdna2InstructionEffects* valu_entry(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    return e.describes_vector_result() ? &e : nullptr;
}
Rdna2Answer answer(bool yes) {
    return yes ? Rdna2Answer::Yes : Rdna2Answer::No;
}
}   // namespace

Rdna2Answer rdna2_valu_lane_mapped(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2InstructionEffects* e = valu_entry(fmt, opcode);
    if (!e) return Rdna2Answer::Unknown;
    return answer(e->lane_flow == Rdna2LaneFlow::LaneWise ||
                  e->lane_flow == Rdna2LaneFlow::LaneIndex || e->lane_flow == Rdna2LaneFlow::Swap);
}

Rdna2Answer rdna2_valu_lane_dependent(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2InstructionEffects* e = valu_entry(fmt, opcode);
    if (!e) return Rdna2Answer::Unknown;
    return answer(
        e->lane_flow == Rdna2LaneFlow::ReadLane || e->lane_flow == Rdna2LaneFlow::WriteLane ||
        e->lane_flow == Rdna2LaneFlow::CrossLane || e->lane_flow == Rdna2LaneFlow::LaneIndex);
}

Rdna2Answer rdna2_valu_result_keeps_old_vdst(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2InstructionEffects* e = valu_entry(fmt, opcode);
    if (!e) return Rdna2Answer::Unknown;
    return answer(e->accumulates || e->vdst_half || e->lane_flow == Rdna2LaneFlow::WriteLane ||
                  e->m0_relative == Rdna2M0Relative::Destination ||
                  e->m0_relative == Rdna2M0Relative::Both);
}

uint32_t rdna2_valu_complete_result_dwords(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2InstructionEffects* e = valu_entry(fmt, opcode);
    if (!e || e->accumulates || e->vdst_half) return 0;
    if (e->lane_flow != Rdna2LaneFlow::LaneWise && e->lane_flow != Rdna2LaneFlow::LaneIndex)
        return 0;
    return e->vdst_dwords;
}

uint32_t rdna2_valu_complete_result_dwords(const Rdna2Inst& in) {
    if (in.has_dpp) return 0;
    if (in.has_sdwa && (in.sdwa_dst_sel != 6 || in.sdwa_dst_unused == 2)) return 0;
    const Rdna2InstructionEffects* e = valu_entry(in.fmt, in.opcode);
    if (!e) return 0;
    // CONFIDENCE: HIGH -- SDWA DWORD + UNUSED_PAD places the 16-bit result in [15:0] and pads the
    // rest with zero (ISA 6.8), so it defines the whole word; the plain form defines one half.
    if (e->vdst_half && in.has_sdwa && !e->accumulates &&
        (e->lane_flow == Rdna2LaneFlow::LaneWise || e->lane_flow == Rdna2LaneFlow::LaneIndex))
        return e->vdst_dwords;
    return rdna2_valu_complete_result_dwords(in.fmt, in.opcode);
}

uint32_t rdna2_valu_source_count(Rdna2Format fmt, uint32_t opcode) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    // Unknown and interpolation forms (an attribute field, not a register) keep every field.
    // Accumulators read old VDST through e.accumulates, independently of encoded source fields.
    // Their widths already describe the explicit sources (ISA 70648 12.7: D = S0*S1 + D).
    if (!e.describes_vector_result() || e.lane_flow == Rdna2LaneFlow::Interpolate) return 3;
    uint32_t count = 0;
    for (uint32_t source = 0; source < 3; ++source)
        if (e.ssrc_dwords[source]) count = source + 1;
    return count;
}

uint32_t rdna2_valu_vgpr_source_span(const Rdna2Inst& in, uint32_t source) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(in);
    if (!e.describes_sources() || source >= 3) return 0;
    return e.ssrc_dwords[source];
}

uint32_t rdna2_smem_load_dwords(uint32_t opcode) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(Rdna2Format::SMEM, opcode);
    return e.known() && e.smem == Rdna2ScalarMemory::Load ? e.sdst_dwords : 0u;
}

uint32_t rdna2_smem_destination_dwords(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SMEM) return 0;
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(in);
    if (!e.known() || (e.smem == Rdna2ScalarMemory::Atomic && !in.smem_glc)) return 0;
    return e.sdst_dwords;
}

uint32_t rdna2_smem_address_dwords(uint32_t opcode) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(Rdna2Format::SMEM, opcode);
    return e.known() ? e.ssrc_dwords[0] : 4u;
}

bool rdna2_writes_m0_relative_vgpr(const Rdna2Inst& in) {
    const Rdna2InstructionEffects* e = valu_entry(in.fmt, in.opcode);
    return e && (e->m0_relative == Rdna2M0Relative::Destination ||
                 e->m0_relative == Rdna2M0Relative::Both);
}

Rdna2Answer rdna2_writes_scc(Rdna2Format fmt, uint32_t opcode) {
    if (never_touches_scc(fmt)) return Rdna2Answer::No;
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    if (!e.known()) return Rdna2Answer::Unknown;
    return (e.implicit_writes & kScc) ? Rdna2Answer::Yes : Rdna2Answer::No;
}

Rdna2Answer rdna2_reads_scc_implicitly(Rdna2Format fmt, uint32_t opcode) {
    if (never_touches_scc(fmt)) return Rdna2Answer::No;
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    if (!e.known()) return Rdna2Answer::Unknown;
    return e.scc_read != Rdna2SccRead::None ? Rdna2Answer::Yes : Rdna2Answer::No;
}

Rdna2Answer rdna2_reads_scc_as_data(Rdna2Format fmt, uint32_t opcode) {
    if (never_touches_scc(fmt)) return Rdna2Answer::No;
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(fmt, opcode);
    if (!e.known()) return Rdna2Answer::Unknown;
    return e.scc_read == Rdna2SccRead::Data ? Rdna2Answer::Yes : Rdna2Answer::No;
}

// Declared in rdna2_decode.hpp, where their callers already look for them.
bool sop1_opcode_leaves_scc_unmodified(uint32_t opcode) {
    return rdna2_writes_scc(Rdna2Format::SOP1, opcode) == Rdna2Answer::No;
}
bool sop1_opcode_writes_scc(uint32_t opcode) {
    return rdna2_writes_scc(Rdna2Format::SOP1, opcode) == Rdna2Answer::Yes;
}
namespace {
bool sop1_exec_update(uint32_t opcode, Rdna2ExecEffect kind, uint8_t width) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(Rdna2Format::SOP1, opcode);
    return e.known() && e.exec == kind && e.sdst_dwords == width;
}
}   // namespace
bool sop1_opcode_is_saveexec_b32(uint32_t opcode) {
    return sop1_exec_update(opcode, Rdna2ExecEffect::SaveExec, 1);
}
bool sop1_opcode_is_wrexec_b32(uint32_t opcode) {
    return sop1_exec_update(opcode, Rdna2ExecEffect::WrExec, 1);
}
bool sop1_opcode_is_saveexec_b64(uint32_t opcode) {
    return sop1_exec_update(opcode, Rdna2ExecEffect::SaveExec, 2);
}
bool sop1_opcode_is_wrexec_b64(uint32_t opcode) {
    return sop1_exec_update(opcode, Rdna2ExecEffect::WrExec, 2);
}
uint32_t rdna2_sop2_dest_dwords(uint32_t opcode) {
    const Rdna2InstructionEffects& e = rdna2_instruction_effects(Rdna2Format::SOP2, opcode);
    return e.known() ? e.sdst_dwords : 2u;   // fail closed: a reserved encoding counts as a pair
}

bool rdna2_operand_reads_scc(const Rdna2Inst& in) {
    for (uint32_t source = 0; source < in.n_src && source < 4; ++source)
        if (in.src[source].kind == OperandKind::Special && in.src[source].value == 253) return true;
    return false;
}

bool rdna2_program_reads_scc(const std::vector<Rdna2Inst>& ins) {
    for (const Rdna2Inst& in : ins)
        if (rdna2_operand_reads_scc(in) ||
            rdna2_reads_scc_implicitly(in.fmt, in.opcode) != Rdna2Answer::No)
            return true;
    return false;
}

}   // namespace prosper::gpu
