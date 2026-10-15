// sop2_b64_routes.cpp — SOP2 64-bit routes: s_bfm_b64, B64 mask logic, 64-bit shifts and s_bfe_u64.
// Relocated verbatim from emit_alu's SOP2 case by tools/refactor/extract_case_routes.py (ADR 0012).
// These routes still emit SPIR-V directly; they are the mask-domain and proof-carrying shapes the
// IR does not model yet. Each returns nullopt when the instruction is not its shape.
#include "gpu/recompiler/ir/lower/scalar_lowering_internal.hpp"

namespace prosper::gpu::ir {

std::optional<bool> Sop2Lowering::route_bfm_b64() {
    if (in.opcode == kSop2OpcodeBfmB64) { // s_bfm_b64
        if (b.is_vertex && !b.ngg_one_lane) { ok = false; return true; }
        // Wave masks are represented as one bool per SPIR-V invocation.  Constructing the
        // architectural 64-bit integer and then splitting it would lose that domain, so test
        // the current lane directly: bits [offset, offset+width) are set, truncated at bit 63.
        // This is also exact for width==0 (the ISA expression produces an empty mask).
        const uint32_t width = b.ibin(Op_BitwiseAnd, val(in.src[0]), b.uconst(63));
        const uint32_t offset = b.ibin(Op_BitwiseAnd, val(in.src[1]), b.uconst(63));
        // Vertex NGG is the same one-lane approximation as the inline-mask helpers: its
        // represented guest lane is lane zero and it has no compute LocalInvocationIndex.
        //
        // Everything else takes the guest lane from guest_lane_id(), NOT linear_localid.
        // linear_localid is a compute-only value: in a fragment shader it is never
        // assigned and stays 0, which is the "no id" sentinel, so the lane test below was
        // emitted as `OpBitwiseAnd %uint %0 %63` -- an operand naming id 0. spirv-val
        // rejects that module ("Id is 0"), and NVIDIA's driver dereferences it while
        // compiling the pipeline: Kena: Bridge of Spirits (PPSA01802) died inside
        // nvoglv64.dll on its first level load, on `s_bfm_b64 <mask>, 12, 44`.
        // guest_lane_id() is the fragment lane from SubgroupLocalInvocationId (and records
        // the exact-wave-width contract that lane id carries), linear_localid for compute.
        const uint32_t lane = b.ngg_one_lane ? b.uconst(0) :
            b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1));
        const uint32_t at_or_after = b.ucmp(Op_UGreaterThanEqual, lane, offset);
        const uint32_t within_width = b.ucmp(
            Op_ULessThan, b.ibin(Op_ISub, lane, offset), width);
        const uint32_t r = b.land(at_or_after, within_width);
        if (in.dst.value == 126 || in.dst.value == 127) {
            rs.exec = r;
            rs.exec_narrowed = true;
        } else if (in.dst.value == 106 || in.dst.value == 107) {
            rs.vcc = r;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            mask_write_clobbers_pair(rs, in.dst.value);
        } else {
            rs.sreg_bool[in.dst.value] = r;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            mask_write_clobbers_pair(rs, in.dst.value);
            record_s_bfm_b64_scalar_data(b, rs, in, width, offset, ok);
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_b64_mask_logical() {
    if (in.opcode == 0x0f || in.opcode == 0x11 || in.opcode == 0x13 || in.opcode == 0x15 ||
        in.opcode == 0x17 || in.opcode == 0x19 || in.opcode == 0x1b || in.opcode == 0x1d) {
        // 64-bit wave-mask LOGICAL ops on per-lane bools: AND/OR/XOR/ANDN2 plus the
        // complementary ORN2/NAND/NOR/XNOR family. Used for lane-mask arithmetic around
        // divergent control flow / ballot. SCC=(result!=0) is a cross-lane reduction we can't
        // form per-lane — POISON rs.scc so a later consumer rejects instead of silently
        // reading an older s_cmp's value (the adjacent scc-branch shapes are claimed by the
        // mask_test/waterfall linearizers and never read rs.scc).
        auto is_exec = [](const Operand& o){ return o.value == 126 || o.value == 127; };
        auto mask = [&](const Operand& o) -> uint32_t {
            if ((o.value == 106 || o.value == 107) && rs.vcc) return rs.vcc;
            if (o.value == 126 || o.value == 127) return rs.exec;
            if (o.kind == OperandKind::SGPR) { auto it = rs.sreg_bool.find(o.value);
                if (it != rs.sreg_bool.end()) return it->second; }
            if (o.kind == OperandKind::InlineInt)
                return inline_int_mask_bit(b, o.value);
            if (o.kind == OperandKind::Literal && in.has_literal)
                return constant_mask_bit(b, in.literal, 0u);
            if (const uint32_t bit = scalar_pair_lane_bit(b, rs, o)) return bit;
            // Name the missing representation. A wave-mask op that cannot resolve
            // an operand rejects the whole shader, and the reject line downstream says
            // only `mode=unresolved-operand` -- which cannot distinguish these states,
            // and they point at different places to look:
            //
            //   sreg_bool=0, sreg absent     no representation reached here. Could be lost mask
            //                                provenance, ordinary scalar data never classified, or a
            //                                genuinely undefined register -- this line does not
            //                                distinguish them.
            //   sreg present, proven         the words are definitely assigned. Necessary for a
            //                                scalar-pair projection and NOT sufficient: the wave/lane
            //                                contract is a separate question this line is silent on.
            //   sreg present, unproven       definite assignment is unproven SOMEWHERE in this
            //                                shader. The flag is a broad state property -- it cannot
            //                                say which register, which predecessor or which dynamic
            //                                path carries a placeholder, so on its own it does not
            //                                convict these operands.
            //
            // So this narrows the question rather than answering it. In particular a missing
            // `sreg_bool` entry is compatible with the operand HAVING originated from EXEC/VCC and the
            // provenance having been dropped: absence here is not evidence the source was never a mask.
            //
            // Sonic Frontiers' dropped HDR producer (#2790) reports the third row, which was
            // indistinguishable from the other two for a full investigation round. Ungated and deduped
            // like [mimg-unresolved] (further down this file): it fires only on a path that has already
            // failed, so its volume is bounded by the defect it reports.
            //
            // The dedup key includes program_address, which matters: every shader has a
            // pc 328, so a pc-only key would let the first program to reach one silence
            // every later program and attribute its line to a shader nobody is looking at.
            // `--retry-failed-stage` used to recompile through an entry point that left
            // that field 0, collapsing the key across programs; it now passes the real
            // address, so this attributes correctly on both the live and the replay path.
            {
                static std::mutex probe_mutex;
                static std::set<std::tuple<uint64_t, uint32_t, int>> probe_reported;
                bool first = false;
                {
                    std::lock_guard lock(probe_mutex);
                    // Capped like divloop_reject's set: a long-lived process compiling many
                    // shaders must not grow this without bound. Past the cap the diagnostic
                    // goes quiet rather than the set growing.
                    if (probe_reported.size() < 4096)
                        first = probe_reported.emplace(b.diagnostic.program_address, in.pc,
                                                       o.value).second;
                }
                const char* stage_name = b.is_compute  ? "compute"
                                       : b.is_fragment ? "fragment"
                                                       : "vertex";
                if (first)
                    std::fprintf(
                        stderr,
                        "[wave-mask-unresolved] program=0x%llx pc=%u operand=%d kind=%u "
                        "sreg_bool=%d sreg=%d/%d sreg_input=%d/%d "
                        "no_placeholders=%d merge_placeholder=%d/%d stage=%s wave=%u "
                        "native_sg=%u\n",
                        static_cast<unsigned long long>(b.diagnostic.program_address),
                        in.pc, o.value,
                        static_cast<unsigned>(
                            static_cast<std::underlying_type_t<OperandKind>>(o.kind)),
                        static_cast<int>(rs.sreg_bool.contains(o.value)),
                        static_cast<int>(rs.sreg.contains(o.value)),
                        static_cast<int>(rs.sreg.contains(o.value + 1)),
                        static_cast<int>(rs.sreg_input.contains(o.value)),
                        static_cast<int>(rs.sreg_input.contains(o.value + 1)),
                        static_cast<int>(rs.scalar_presence_has_no_placeholders),
                        static_cast<int>(rs.sreg_merge_placeholder.contains(o.value)),
                        static_cast<int>(rs.sreg_merge_placeholder.contains(o.value + 1)),
                        stage_name, b.wave_size, b.native_subgroup_size);
            }
            return 0;
        };
        uint32_t m0 = mask(in.src[0]), m1 = mask(in.src[1]);
        if (!m0 || !m1) { ok = false; return true; }
        rs.scc = 0;   // poison: hardware SCC=(result!=0) is unrepresentable per-lane
        const uint32_t n0 = b.logical_not(m0), n1 = b.logical_not(m1);
        const uint32_t x = b.bsel(m0, n1, m1);               // xor = m0 ? !m1 : m1
        uint32_t r = in.opcode == 0x0f ? b.land(m0, m1)      // and
                   : in.opcode == 0x11 ? b.lor(m0, m1)       // or
                   : in.opcode == 0x13 ? x                    // xor
                   : in.opcode == 0x15 ? b.land(m0, n1)      // andn2
                   : in.opcode == 0x17 ? b.lor(m0, n1)       // orn2
                   : in.opcode == 0x19 ? b.lor(n0, n1)       // nand
                   : in.opcode == 0x1b ? b.land(n0, n1)      // nor
                   : b.logical_not(x);                        // xnor
        const auto words = native_logical_result_words(b, rs, in, r);
        if (native_logical_data_demanded(b, in) && (!words[0] || !words[1])) {
            b.stage_reject_pc = in.pc;
            b.stage_reject_reason = "native-logical-data-words-unavailable";
            ok = false;
            return true;
        }
        if (is_exec(in.dst)) {
            record_exec_mask_subset(rs, r, m0, m1, in.opcode, rs.exec);
            rs.exec = r;
            rs.exec_narrowed = true;
        } else if (in.dst.value == 106 || in.dst.value == 107) {
            rs.vcc = r;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            mask_write_clobbers_pair(rs, in.dst.value);
        } else {
            rs.sreg_bool[in.dst.value] = r;
            rs.sreg_bool_narrowed[in.dst.value] = true;
            mask_write_clobbers_pair(rs, in.dst.value);
        }
        if (words[0] && words[1]) {
            rs.sreg[in.dst.value] = words[0];
            rs.sreg[in.dst.value + 1] = words[1];
        }
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_lshr_b64_exec_pack() {
    // The NGG wave-packing s_lshr_b64 form (dst = EXEC) sets EXEC to the count of active
    // vertices/primitives in the wave. A per-invocation SPIR-V shader has no wave to pack,
    // so leave EXEC full. Handle that special form before the ordinary scalar-pair shift.
    if (in.opcode == 0x21) {
        if (in.dst.value == 126 || in.dst.value == 127) {
            rs.scc = 0; // poison: hardware SCC=(result!=0) over the packed mask is cross-lane
            return true;
        }
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_shift_b64() {
    if (in.opcode == 0x1f || in.opcode == 0x21) {
        // Ordinary s_lshl/lshr_b64 operates on a complete scalar-data pair. Generated
        // compute shaders frequently borrow VCC_LO/HI for 64-bit address arithmetic; that
        // lifetime coexists with VCC's per-lane predicate view, so retain both exactly.
        auto scalar_word = [&](int reg, uint32_t& value) {
            auto current = rs.sreg.find(reg);
            if (current != rs.sreg.end()) { value = current->second; return true; }
            auto input = rs.sreg_input.find(reg);
            if (input != rs.sreg_input.end()) { value = input->second; return true; }
            return false;
        };
        uint32_t source_lo = 0, source_hi = 0;
        const Operand& source = in.src[0];
        if (source.kind == OperandKind::SGPR ||
            (source.kind == OperandKind::Special &&
             source.value >= 106 && source.value < 124)) {
            if (!scalar_word(source.value, source_lo) ||
                !scalar_word(source.value + 1, source_hi)) {
                ok = false; return true;
            }
        } else if (source.kind == OperandKind::InlineInt) {
            source_lo = b.uconst(static_cast<uint32_t>(source.value));
            source_hi = b.uconst(source.value < 0 ? UINT32_MAX : 0u);
        } else if (source.kind == OperandKind::Literal) {
            source_lo = b.uconst(in.literal);
            source_hi = b.uconst(0);
        } else {
            ok = false; return true;
        }
        const uint32_t amount = b.ibin(
            Op_BitwiseAnd, val(in.src[1]), b.uconst(63));
        if (!ok) return true;
        const uint32_t result = b.u64_shift(
            in.opcode == 0x1f ? Op_ShiftLeftLogical : Op_ShiftRightLogical,
            b.u64_from_lohi(source_lo, source_hi), amount);
        const uint32_t lo = b.u64_lo(result), hi = b.u64_hi(result);
        rs.scc = b.ucmp(
            Op_INotEqual, b.ibin(Op_BitwiseOr, lo, hi), b.uconst(0));
        const bool writes_exec = in.dst.value == 126 || in.dst.value == 127;
        const bool writes_vcc = in.dst.value == 106 || in.dst.value == 107;
        if (writes_exec || writes_vcc) {
            uint32_t lane = b.ibin(
                Op_BitwiseAnd, b.guest_lane_id(), b.uconst(b.wave_size - 1));
            const uint32_t mask_bit = b.u64_bit(result, lane);
            if (writes_exec) {
                rs.exec = mask_bit;
                rs.exec_narrowed = true;
                rs.sreg.erase(in.dst.value);
                rs.sreg.erase(in.dst.value + 1);
            } else {
                rs.vcc = mask_bit;
                rs.sreg_bool[in.dst.value] = mask_bit;
                rs.sreg_bool_narrowed[in.dst.value] = true;
                rs.sreg[in.dst.value] = lo;
                rs.sreg[in.dst.value + 1] = hi;
            }
        } else {
            rs.sreg[in.dst.value] = lo;
            rs.sreg[in.dst.value + 1] = hi;
            rs.sreg_bool.erase(in.dst.value);
            rs.sreg_bool.erase(in.dst.value + 1);
            rs.sreg_bool_narrowed.erase(in.dst.value);
            rs.sreg_bool_narrowed.erase(in.dst.value + 1);
        }
        rs.sreg_bool_b32.erase(in.dst.value);
        rs.sreg_bool_b32.erase(in.dst.value + 1);
        rs.sreg_srt.erase(in.dst.value);
        rs.sreg_srt.erase(in.dst.value + 1);
        return true;
    }
    return std::nullopt;
}

std::optional<bool> Sop2Lowering::route_bfe_u64() {
    if (in.opcode == kSop2OpcodeBfeU64) { // s_bfe_u64
        // BFE's sources are scalar DATA even when its destination is the architectural
        // EXEC/VCC mask pair. Keep the complete uniform source pair long enough to extract
        // the bit belonging to this emulated lane; treating the destination as ordinary
        // SGPR data leaves rs.exec/rs.vcc stale, while rejecting EXEC drops valid NGG
        // merged-stage prologues. A Vulkan vertex invocation represents the one live guest
        // lane for that vertex; compute/fragment shells retain their real wave lane.
        auto high = [&]() -> uint32_t {
            const Operand& source = in.src[0];
            if (source.kind == OperandKind::SGPR) {
                auto value = rs.sreg.find(source.value + 1);
                if (value != rs.sreg.end()) return value->second;
                auto input = rs.sreg_input.find(source.value + 1);
                return input == rs.sreg_input.end() ? b.uconst(0) : input->second;
            }
            if (source.kind == OperandKind::Special &&
                source.value >= 106 && source.value < 124) {
                auto value = rs.sreg.find(source.value + 1);
                if (value != rs.sreg.end()) return value->second;
                ok = false; return b.uconst(0);
            }
            if (source.kind == OperandKind::InlineInt)
                return b.uconst(source.value < 0 ? UINT32_MAX : 0u);
            if (source.kind == OperandKind::Literal) return b.uconst(0);
            ok = false; return b.uconst(0);
        };
        const uint32_t source_lo = val(in.src[0]);
        const uint32_t source_hi = high();
        const uint32_t control = val(in.src[1]);
        if (!ok) return true;
        const uint32_t offset = b.ibin(Op_BitwiseAnd, control, b.uconst(0x3f));
        const uint32_t width = b.ibin(
            Op_BitwiseAnd,
            b.ibin(Op_ShiftRightLogical, control, b.uconst(16)),
            b.uconst(0x7f));
        const uint32_t result = b.bfe_u64(
            b.u64_from_lohi(source_lo, source_hi), offset, width);
        const bool writes_exec = in.dst.value == 126 || in.dst.value == 127;
        const bool writes_vcc = in.dst.value == 106 || in.dst.value == 107;
        const auto has_scalar_word = [&](int reg) {
            return rs.sreg.contains(reg) || rs.sreg_input.contains(reg);
        };
        const auto has_scalar_source = [&](const Operand& operand, bool wide) {
            if (operand.kind == OperandKind::InlineInt ||
                operand.kind == OperandKind::Literal) return true;
            if (operand.kind != OperandKind::SGPR &&
                operand.kind != OperandKind::Special) return false;
            return has_scalar_word(operand.value) &&
                (!wide || has_scalar_word(operand.value + 1));
        };
        const bool retain_vcc_scalar = writes_vcc && has_scalar_source(in.src[0], true) &&
                                       has_scalar_source(in.src[1], false) &&
                                       (rs.scalar_presence_has_no_placeholders ||
                                        b.vcc_bfe_u64_scalar_result_pcs.contains(in.pc) ||
                                        b.vcc_local_scalar_write_pcs.contains(in.pc));
        if (writes_exec || writes_vcc) {
            uint32_t lane = b.guest_lane_id();
            lane = b.ibin(Op_BitwiseAnd, lane, b.uconst(b.wave_size - 1));
            const uint32_t mask_bit = b.u64_bit(result, lane);
            // The exact 64-bit scalar result can carry SCC and both VCC data words.
            // A mask-only source cannot: its per-lane predicate is not a 64-bit value.
            rs.scc = retain_vcc_scalar
                ? b.ucmp(Op_INotEqual,
                         b.ibin(Op_BitwiseOr, b.u64_lo(result), b.u64_hi(result)),
                         b.uconst(0))
                : 0;
            if (writes_exec) {
                rs.exec = mask_bit;
                rs.exec_narrowed = true;
            } else {
                rs.vcc = mask_bit;
                rs.sreg_bool[in.dst.value] = mask_bit;
                rs.sreg_bool_narrowed[in.dst.value] = true;
            }
            mask_write_clobbers_pair(rs, in.dst.value);
            if (retain_vcc_scalar) {
                rs.sreg[in.dst.value] = b.u64_lo(result);
                rs.sreg[in.dst.value + 1] = b.u64_hi(result);
            }
        } else {
            const uint32_t lo = b.u64_lo(result), hi = b.u64_hi(result);
            rs.sreg[in.dst.value] = lo;
            rs.sreg[in.dst.value + 1] = hi;
            rs.scc = b.ucmp(Op_INotEqual,
                            b.ibin(Op_BitwiseOr, lo, hi), b.uconst(0));
            rs.sreg_srt.erase(in.dst.value);
            rs.sreg_srt.erase(in.dst.value + 1);
        }
        return true;
    }
    return std::nullopt;
}

} // namespace prosper::gpu::ir
