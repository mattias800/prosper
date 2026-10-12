#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_spilled_mask_halves.hpp"
#include <bitset>
#include <deque>

namespace prosper::gpu {
namespace {
std::set<int> copy_entry_masks(const RegState* initial) {
    std::set<int> roots{126, 127};
    if (!initial) return roots;
    if (initial->vcc) {
        roots.insert(106);
        roots.insert(107);
    }
    for (const auto& [root, value] : initial->sreg_bool)
        if (value && root >= 0 && root < 128) roots.insert(root);
    return roots;
}

bool native_vcc_copy(const SpirvCompute& b, const Rdna2Inst& in) {
    return b.is_compute && !b.is_fragment_packet() && !b.ngg_workgroup_shell && !b.ngg_one_lane &&
           b.wave_size == 64 && b.native_subgroup_size == 64 && in.fmt == Rdna2Format::SOP1 &&
           in.opcode == kSop1OpcodeMovB64 && in.dst.kind == OperandKind::SGPR &&
           in.dst.value == 106 && in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 &&
           in.src[0].value <= 104;
}

// CONFIDENCE: HIGH -- this fragment route preserves the same current B64 predicate.
// Its instruction-order MUST source and generation deadness do not authorize any physical
// words or a ballot after lanes left. Packet/portable and unrelated execution domains differ.
bool fragment_copy_domain(const SpirvCompute& b) {
    return b.is_fragment && !b.is_compute && !b.is_vertex && !b.ngg_workgroup_shell &&
           !b.ngg_one_lane && b.guest_stage == GuestShaderStage::Fragment &&
           b.physical_domain == PhysicalExecutionDomain::Invocation && b.wave_size == 64 &&
           !b.allow_b32_masks;
}

bool fragment_vcc_copy(const SpirvCompute& b, const Rdna2Inst& in) {
    return fragment_copy_domain(b) && in.fmt == Rdna2Format::SOP1 &&
           in.opcode == kSop1OpcodeMovB64 && in.dst.kind == OperandKind::SGPR &&
           in.dst.value == 106 && in.src[0].kind == OperandKind::SGPR && in.src[0].value >= 0 &&
           in.src[0].value <= 104;
}

std::array<uint32_t, 10> copy_entry_domain(const SpirvCompute& b) {
    return {static_cast<uint32_t>(b.guest_stage),
            static_cast<uint32_t>(b.physical_domain),
            b.wave_size,
            b.native_subgroup_size,
            b.is_compute,
            b.is_fragment,
            b.is_vertex,
            b.ngg_workgroup_shell,
            b.ngg_one_lane,
            b.allow_b32_masks};
}

bool ordinary_pair(const Operand& source) {
    return (source.kind == OperandKind::SGPR && source.value >= 0 && source.value <= 104) ||
           (source.kind == OperandKind::Special && (source.value == 106 || source.value == 107));
}

bool marked_source(const RegState& rs, const Operand& source) {
    if (!ordinary_pair(source)) return false;
    for (int word : {source.value, source.value + 1})
        if (rs.sreg_merge_placeholder.contains(word) || rs.sreg_memory_pattern.contains(word) ||
            rs.unavailable_saved_mask_words.contains(word))
            return true;
    return false;
}

std::array<uint32_t, 2> source_words(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                     const Operand& source) {
    if (source.kind == OperandKind::InlineInt)
        return {b.uconst(static_cast<uint32_t>(source.value)),
                b.uconst(source.value < 0 ? UINT32_MAX : 0u)};
    if (source.kind == OperandKind::Literal && in.has_literal)
        return {b.uconst(in.literal), b.uconst(0)};
    if (!ordinary_pair(source) || marked_source(rs, source)) return {};
    std::array<uint32_t, 2> result{};
    for (int half = 0; half < 2; ++half) {
        const int word = source.value + half;
        if (sreg_word_may_be_fabricated(rs, word)) return {};
        const auto current = rs.sreg.find(word);
        if (current != rs.sreg.end())
            result[half] = current->second;
        else {
            // A mask write ends the launch-input lifetime. Do not resurrect an older input if
            // an incomplete current mask happens to coexist with an entry map spelling.
            if (rs.sreg_bool.contains(source.value) ||
                ((source.value == 106 || source.value == 107) && rs.vcc))
                return {};
            const auto input = rs.sreg_input.find(word);
            if (input == rs.sreg_input.end()) return {};
            result[half] = input->second;
        }
    }
    return result;
}

uint32_t logical_word(SpirvCompute& b, uint32_t opcode, uint32_t a, uint32_t c) {
    const auto complement = [&](uint32_t value) {
        return b.ibin(Op_BitwiseXor, value, b.uconst(UINT32_MAX));
    };
    switch (opcode) {
        case 0x0f: return b.ibin(Op_BitwiseAnd, a, c);
        case 0x11: return b.ibin(Op_BitwiseOr, a, c);
        case 0x13: return b.ibin(Op_BitwiseXor, a, c);
        case 0x15: return b.ibin(Op_BitwiseAnd, a, complement(c));
        case 0x17: return b.ibin(Op_BitwiseOr, a, complement(c));
        case 0x19: return complement(b.ibin(Op_BitwiseAnd, a, c));
        case 0x1b: return complement(b.ibin(Op_BitwiseOr, a, c));
        case 0x1d: return complement(b.ibin(Op_BitwiseXor, a, c));
        default: return 0;
    }
}
}   // namespace

void seed_native_logical_entry_marks(SpilledMaskHalves& marks, const RegState& initial) {
    for (const auto& [reg, source] : initial.vreg_source_marks) {
        if (source.placeholder || source.hard) marks.relayed_vectors.insert(reg);
        if (source.memory) marks.memory_vectors.insert(reg);
    }
    marks.exec_known_full = initial.exec && !initial.exec_narrowed;
    marks.native_interpolation = initial.vector_native_interpolation;
    marks.relayed_words.insert(initial.sreg_merge_placeholder.begin(),
                               initial.sreg_merge_placeholder.end());
    marks.relayed_words.insert(initial.unavailable_saved_mask_words.begin(),
                               initial.unavailable_saved_mask_words.end());
    marks.memory_words.insert(initial.sreg_memory_pattern.begin(),
                              initial.sreg_memory_pattern.end());
    if (initial.scc_merge_placeholder) marks.relayed_words.insert(kRelayedScc);
}

bool native_logical_data_word(const RegState& rs, int word, uint32_t& value) {
    // Ordinary DATA retains its existing MAY marks; only a missing physical mask word is a
    // transactional gap here. Native logical result authority checks those marks separately.
    if (rs.unavailable_saved_mask_words.contains(word)) return false;
    const auto current = rs.sreg.find(word);
    if (current != rs.sreg.end()) {
        value = current->second;
        return true;
    }
    // A current mask cannot resurrect a scalar launch spelling from an older generation.
    if (rs.sreg_bool.contains(word) || rs.sreg_bool.contains(word - 1) ||
        ((word == 106 || word == 107) && rs.vcc))
        return false;
    const auto input = rs.sreg_input.find(word);
    if (input == rs.sreg_input.end()) return false;
    value = input->second;
    return true;
}

bool is_native_b64_logical(const SpirvCompute& b, const Rdna2Inst& in) {
    return b.is_compute && !b.is_fragment_packet() && b.wave_size == 64 &&
           b.native_subgroup_size == 64 && in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
           in.opcode <= 0x1d && (in.opcode & 1u) && in.dst.value >= 0 && in.dst.value <= 107;
}

void initialize_native_logical_data_demand(SpirvCompute& b, const std::vector<Rdna2Inst>& original,
                                           const RegState* initial) {
    // Backward MAY liveness follows physical word generations. A later use after a definite
    // overwrite cannot certify an earlier Bool-only logical, and either branch/loop successor
    // can demand the current generation. The finite lattice has 128 bits per instruction.
    using Words = std::bitset<128>;
    b.native_bool_only_vcc_copy_pcs.clear();
    b.fragment_bool_only_vcc_copy_pcs.clear();
    b.fragment_bool_only_saved_mask_pcs.clear();
    b.native_vcc_copy_entry_domain = copy_entry_domain(b);
    b.native_vcc_copy_entry_masks = copy_entry_masks(initial);
    b.native_vcc_copy_entry_b32_masks =
        initial ? initial->sreg_bool_b32 : std::unordered_set<int>{};
    const size_t count = original.size();
    std::map<uint32_t, size_t> at_pc;
    for (size_t i = 0; i < count; ++i) at_pc.emplace(original[i].pc, i);
    std::vector<std::vector<size_t>> successors(count), predecessors(count);
    std::vector<Words> before(count), after(count), reads(count), kills(count);
    std::vector<bool> relays(count);
    const auto mark = [](Words& words, int base, uint32_t width) {
        for (uint32_t half = 0; half < width && half < 128; ++half) {
            const int word = base + static_cast<int>(half);
            if (word >= 0 && word < 128) words.set(word);
        }
    };
    // A CSELECT is a Bool relay only when both operands have that domain on every
    // incoming path. Mere pair width does not choose the emitter's mask-first route.
    // Seed the MUST lattice from the actual caller state, never from SGPR numbers.
    std::vector<Words> masks_before(count), masks_after(count);
    for (size_t i = 0; i < count; ++i) {
        const auto& in = original[i];
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
        if (branch) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target >= 0 && target <= UINT32_MAX &&
                at_pc.contains(static_cast<uint32_t>(target)))
                successors[i].push_back(at_pc.at(static_cast<uint32_t>(target)));
        }
        if (!in.is_end && (!branch || in.opcode != 0x02) && i + 1 < count)
            successors[i].push_back(i + 1);
        for (size_t next : successors[i]) predecessors[next].push_back(i);
    }
    std::vector<bool> reachable(count);
    std::deque<size_t> visit;
    if (count) {
        reachable[0] = true;
        visit.push_back(0);
    }
    while (!visit.empty()) {
        const size_t i = visit.front();
        visit.pop_front();
        for (size_t next : successors[i])
            if (!reachable[next]) {
                reachable[next] = true;
                visit.push_back(next);
            }
    }
    Words entry;
    entry.set(126);
    if (!fragment_copy_domain(b)) entry.set(127);
    if (initial) {
        if (initial->vcc && (!fragment_copy_domain(b) || !initial->sreg_bool_b32.contains(106))) {
            entry.set(106);
            if (!fragment_copy_domain(b)) entry.set(107);
        }
        for (const auto& [root, value] : initial->sreg_bool)
            if (value && root >= 0 && root < 128 &&
                (!fragment_copy_domain(b) || ((root <= 104 || root == 106 || root == 126) &&
                                              !initial->sreg_bool_b32.contains(root))))
                entry.set(root);
    }
    const auto mask_source = [&](const Words& masks, const Operand& source) {
        if (source.kind == OperandKind::InlineInt) return true;
        // A fragment B64 source must remain complete through every relay. A move/select
        // cannot turn an independent high word or B32 seed into a whole-mask root.
        if (fragment_copy_domain(b) &&
            !((source.kind == OperandKind::SGPR && source.value >= 0 && source.value <= 104) ||
              ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
               (source.value == 106 || source.value == 126))))
            return false;
        return (source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
               source.value >= 0 && source.value < 128 && masks.test(source.value);
    };
    const auto saveexec = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOP1 && ((in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                                                in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
                                               in.opcode == kSop1OpcodeAndn1SaveexecB64 ||
                                               in.opcode == kSop1OpcodeOrn1SaveexecB64);
    };
    for (size_t i = 0; i < count; ++i)
        if (reachable[i]) masks_after[i].set();
    bool masks_changed = true;
    while (masks_changed) {
        masks_changed = false;
        for (size_t i = 0; i < count; ++i) {
            if (!reachable[i]) continue;
            Words incoming;
            incoming.set();
            if (i == 0) incoming &= entry;
            for (size_t previous : predecessors[i])
                if (reachable[previous]) incoming &= masks_after[previous];
            masks_before[i] = incoming;
            Words value = incoming;
            const auto& in = original[i];
            const bool copy_mask = in.fmt == Rdna2Format::SOP1 &&
                                   (in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x08 ||
                                    in.opcode == 0x0a || in.opcode == kSop1OpcodeQuadmaskB64) &&
                                   mask_source(incoming, in.src[0]);
            const bool select_mask = in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0b &&
                                     mask_source(incoming, in.src[0]) &&
                                     mask_source(incoming, in.src[1]);
            // CONFIDENCE: HIGH -- the existing fragment SOP2 emitter computes these logical
            // results pointwise from genuine Bool operands. This is only the original MUST
            // transfer; generation DATA demand and current-source validation remain separate.
            const bool fragment_logical =
                fragment_copy_domain(b) && in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
                in.opcode <= 0x1d && (in.opcode & 1u) && in.dst.kind == OperandKind::SGPR &&
                ((in.dst.value >= 0 && in.dst.value <= 104) || in.dst.value == 106 ||
                 in.dst.value == 126) &&
                mask_source(incoming, in.src[0]) && mask_source(incoming, in.src[1]);
            const bool conditional =
                in.fmt == Rdna2Format::SOP1 &&
                (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64);
            // A conditional scalar replacement is not a definite DATA kill, but it cannot
            // prove the old mask survives on every path. Retire all overlapping mask roots.
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (int root = std::max(0, base - 1);
                     root < base + static_cast<int>(width) && root < 128; ++root)
                    value.reset(root);
            });
            if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
                in.dst.kind != OperandKind::SGPR)
                for (int root : {105, 106, 107}) value.reset(root);
            if (in.fmt == Rdna2Format::Unknown || rdna2_escapes_decoded_effects(in)) value.reset();
            int root = -1;
            if (!conditional && (copy_mask || select_mask || saveexec(in) ||
                                 is_native_b64_logical(b, in) || fragment_logical))
                root = in.dst.value;
            if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode))
                root = in.dst.kind == OperandKind::SGPR ? in.dst.value : 106;
            if (vop3_writes_mask_sdst(in)) root = in.sdst.value;
            if (root >= 0 && root < 128 &&
                (!fragment_copy_domain(b) || root <= 104 || root == 106 || root == 126))
                value.set(root);
            if (root == 106 && !fragment_copy_domain(b)) value.set(107);
            value.set(126);
            if (!fragment_copy_domain(b)) value.set(127);
            if (value != masks_after[i]) {
                masks_after[i] = value;
                masks_changed = true;
            }
        }
    }
    for (size_t i = 0; i < count; ++i) {
        const auto& in = original[i];
        if (rdna2_escapes_decoded_effects(in) || in.fmt == Rdna2Format::Unknown) {
            // An unresolved control/read footprint supplies no deadness proof.
            reads[i].set();
        }
        const bool branch = in.fmt == Rdna2Format::SOPP &&
                            (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
        if (branch) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !at_pc.contains(static_cast<uint32_t>(target)))
                reads[i].set();
        }
        // Conditional moves can preserve the old destination, so their writer spelling is
        // not a definite kill. All implicit sources are charged before writer retirement.
        const bool conditional =
            (in.fmt == Rdna2Format::SOP1 &&
             (in.opcode == kSop1OpcodeCmovB32 || in.opcode == kSop1OpcodeCmovB64)) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x02);
        if (!conditional)
            for_each_scalar_write(in,
                                  [&](int base, uint32_t width) { mark(kills[i], base, width); });
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            in.dst.kind != OperandKind::SGPR)
            mark(kills[i], 106, 2);
        const bool source_mask = mask_source(masks_before[i], in.src[0]);
        const bool select_mask = in.fmt == Rdna2Format::SOP2 && in.opcode == 0x0b && source_mask &&
                                 mask_source(masks_before[i], in.src[1]);
        const bool bool_logical = in.fmt == Rdna2Format::SOP2 && in.opcode >= 0x0f &&
                                  in.opcode <= 0x1d && (in.opcode & 1u);
        relays[i] = bool_logical || select_mask ||
                    (in.fmt == Rdna2Format::SOP1 && source_mask &&
                     (in.opcode == kSop1OpcodeMovB64 || in.opcode == 0x08 || in.opcode == 0x0a ||
                      in.opcode == kSop1OpcodeQuadmaskB64 || saveexec(in)));
        const bool mask_reduction =
            in.fmt == Rdna2Format::SOP1 && source_mask &&
            (in.opcode == kSop1OpcodeBcnt1I32B64 || in.opcode == kSop1OpcodeFf1I32B64 ||
             in.opcode == kSop1OpcodeFlbitI32B64);
        if (relays[i] || mask_reduction) continue;
        // Canonical MBCNT halves consume the saved Bool view, just like the emitter's
        // mbcnt_source_bit route. Its accumulator remains numeric DATA. A mismatched half
        // or a source without an all-path mask root supplies no such exemption.
        const bool mbcnt =
            in.fmt == Rdna2Format::VOP3 && (in.opcode == 0x365 || in.opcode == 0x366);
        const int mbcnt_root = in.src[0].value - (in.opcode == 0x366 ? 1 : 0);
        const bool mbcnt_mask =
            mbcnt && mbcnt_root >= 0 && mbcnt_root < 128 && masks_before[i].test(mbcnt_root) &&
            (in.src[0].kind == OperandKind::SGPR ||
             (in.src[0].kind == OperandKind::Special && (mbcnt_root == 106 || mbcnt_root == 126)));
        // CONFIDENCE: HIGH -- the ordinary V_CNDMASK e64 emitter consumes source2 from
        // sreg_bool/VCC, not scalar DATA. Only its instruction-order MUST mask domain
        // supplies this exemption; data sources0/1 and unresolved physical generations
        // remain reads. VCC_HI's independent B32 route has no such Wave64 MUST fact here.
        const auto& condition = in.src[2];
        const bool cndmask_mask =
            in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101 && in.n_src == 3 &&
            mask_source(masks_before[i], condition) &&
            ((condition.kind == OperandKind::SGPR && condition.value >= 0 &&
              condition.value <= 105) ||
             ((condition.kind == OperandKind::SGPR || condition.kind == OperandKind::Special) &&
              condition.value == 106));
        for (uint32_t source = 0; source < in.n_src; ++source) {
            if ((source == 0 && mbcnt_mask) || (source == 2 && cndmask_mask)) continue;
            const auto& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR &&
                !(operand.kind == OperandKind::Special && operand.value >= 106 &&
                  operand.value <= 127))
                continue;
            uint32_t width = scalar_alu_source_words(in, source);
            // The shared ALU table covers explicit arithmetic widths and dynamic SGPR ranges.
            // Memory operands are descriptors/addresses, not Bool masks. Unknown footprints
            // conservatively observe the remaining words rather than supplying a false kill.
            if (in.fmt == Rdna2Format::SMEM)
                width = source == 0 ? (in.opcode >= 0x08 ? 4u : 2u) : 1u;
            if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF)
                width = source == 1 ? 4u : 1u;
            if (in.fmt == Rdna2Format::MIMG) width = source == 1 ? 8u : (source == 2 ? 4u : 1u);
            if (width == UINT32_MAX) continue;   // the shared inventory says this field is not read
            if (!width) width = 128;
            mark(reads[i], operand.value, width);
        }
        mark(reads[i], in.dst.value, scalar_implicit_destination_read_width(in));
    }
    std::deque<size_t> pending;
    std::vector<bool> queued(count, true);
    for (size_t i = count; i-- > 0;) pending.push_back(i);
    while (!pending.empty()) {
        const size_t i = pending.front();
        pending.pop_front();
        queued[i] = false;
        Words out;
        for (size_t next : successors[i]) out |= before[next];
        after[i] = out;
        Words value = (out & ~kills[i]) | reads[i];
        const auto& in = original[i];
        if (relays[i] && (out & kills[i]).any()) {
            if (saveexec(in))
                mark(value, 126, 2);   // saved DATA is OLD EXEC, independent of the new source
            else
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    const auto& operand = in.src[source];
                    if (operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special)
                        mark(value, operand.value, 2);
                }
        }
        if (value == before[i]) continue;
        before[i] = value;
        for (size_t previous : predecessors[i])
            if (!queued[previous]) {
                queued[previous] = true;
                pending.push_back(previous);
            }
    }
    for (size_t i = 0; i < count; ++i) {
        // CONFIDENCE: HIGH -- this reuses the same original CFG's backward MAY DATA liveness
        // and instruction-order MUST mask domains. It withholds representation only; it grants
        // neither physical words nor a new mask fact. Unknown footprints already read all words.
        if (fragment_copy_domain(b) && reachable[i] && !(after[i] & kills[i]).any())
            for_each_scalar_write(original[i], [&](int root, uint32_t width) {
                if (width == 2 && root >= 0 && root <= 104 && masks_after[i].test(root) &&
                    scalar_write_is_b64_mask(original[i], root))
                    b.fragment_bool_only_saved_mask_pcs.insert(original[i].pc);
            });
        if (is_native_b64_logical(b, original[i]) && (after[i] & kills[i]).any())
            b.native_logical_data_demand_pcs.insert(original[i].pc);
        else if (native_vcc_copy(b, original[i]) && relays[i] && !(after[i] & kills[i]).any())
            b.native_bool_only_vcc_copy_pcs.insert(original[i].pc);
        else if (fragment_vcc_copy(b, original[i]) && relays[i] && !(after[i] & kills[i]).any())
            b.fragment_bool_only_vcc_copy_pcs.insert(original[i].pc);
    }
}

void initialize_native_logical_uniform_entry_proof(SpirvCompute& b,
                                                   const std::vector<Rdna2Inst>& original,
                                                   const RegState* initial) {
    if (!native_compute_has_complete_entry(b) || !b.native_logical_uniform_launch ||
        b.ngg_workgroup_shell || !b.is_compute || b.native_subgroup_size != 64 || b.wave_size != 64)
        return;
    uint32_t reentry = UINT32_MAX;
    std::set<uint32_t> pcs;
    for (const auto& in : original) pcs.insert(in.pc);
    for (const auto& in : original) {
        if (in.fmt == Rdna2Format::Unknown || rdna2_escapes_decoded_effects(in)) return;
        if (in.fmt != Rdna2Format::SOPP ||
            !(in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09)))
            continue;
        const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
        if (target < 0 || target > UINT32_MAX || !pcs.contains(static_cast<uint32_t>(target)))
            return;
        if (target <= in.pc) reentry = std::min(reentry, static_cast<uint32_t>(target));
    }
    bool genuine_mov_entry = b.local_count == 64 && initial && !initial->exec_narrowed;
    for (const auto& in : original) {
        if (in.pc >= reentry || in.is_end || in.fmt == Rdna2Format::EXP ||
            (rdna2_instruction_may_change_exec(in) && !exec_write_sets_full_mask(in)))
            break;
        if (in.fmt == Rdna2Format::SOPP) {
            if (in.opcode == 0x04 || in.opcode == 0x05) {
                if (!branch_is_workgroup_uniform(original, in.pc, UniformBranchCondition::Scc))
                    break;
            } else if (in.opcode == 0x06 || in.opcode == 0x07) {
                if (!branch_is_workgroup_uniform(original, in.pc, UniformBranchCondition::Vcc))
                    break;
            } else if (in.opcode != 0x00 && in.opcode != 0x02 && in.opcode != 0x0a &&
                       in.opcode != 0x0c && in.opcode != 0x20 && in.opcode != 0x21 &&
                       in.opcode != 0x22)
                break;
        }
        if (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeSetregB32) break;
        genuine_mov_entry &= native_uniform_mov_entry_sources(initial, in);
        // Every preceding host case/selection is reached by either the entire launch workgroup
        // or none of it: each guest branch condition is proved launch-uniform by the shared
        // closed provenance walk. No initial partial/outer gate, reentry or EXEC mutation can
        // enter this prefix. This is not an SCC-spelling or subgroup-width assumption.
        if (is_native_b64_logical(b, in)) b.native_logical_uniform_entry_pcs.insert(in.pc);
        // Only canonical current EXEC is admitted here. Unlike the old Special-destination
        // bypass, this conditional producer requires genuine launch DATA for every preceding
        // scalar read, in addition to the original scalar-uniform control proof above.
        if (genuine_mov_entry && in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
            in.src[0].kind == OperandKind::Special && in.src[0].value == 126 &&
            in.dst.kind == OperandKind::SGPR &&
            ((in.dst.value >= 0 && in.dst.value <= 104) || in.dst.value == 106 ||
             in.dst.value == 126))
            b.native_uniform_exec_mov_pcs.insert(in.pc);
    }
}

bool native_logical_producer_has_full_participation(const SpirvCompute& b, const Rdna2Inst& in) {
    if (!saved_mask_physical_launch_matches_original(b)) return false;
    if (saved_mask_producer_has_full_participation(b, in)) return true;
    if (!b.native_logical_uniform_entry_pcs.contains(in.pc) ||
        in.pc >= b.saved_mask_original_words.size())
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return !in.synthetic_terminator && in == original;
}

bool native_logical_data_demanded(const SpirvCompute& b, const Rdna2Inst& in) {
    if (!b.native_logical_data_demand_pcs.contains(in.pc) ||
        in.pc >= b.saved_mask_original_words.size())
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return !in.synthetic_terminator && in == original;
}

bool native_vcc_copy_entry_masks_match(const SpirvCompute& b, const RegState* initial) {
    return native_vcc_copy_execution_domain_matches(b) &&
           b.native_vcc_copy_entry_masks == copy_entry_masks(initial) &&
           b.native_vcc_copy_entry_b32_masks ==
               (initial ? initial->sreg_bool_b32 : std::unordered_set<int>{});
}

bool native_vcc_copy_execution_domain_matches(const SpirvCompute& b) {
    return b.native_vcc_copy_entry_domain == copy_entry_domain(b);
}

bool native_vcc_copy_is_bool_only(const SpirvCompute& b, const Rdna2Inst& in) {
    const bool proved_copy =
        (native_vcc_copy(b, in) && b.native_bool_only_vcc_copy_pcs.contains(in.pc)) ||
        (fragment_vcc_copy(b, in) && b.fragment_bool_only_vcc_copy_pcs.contains(in.pc));
    if (!proved_copy || b.native_vcc_copy_entry_domain != copy_entry_domain(b) ||
        in.pc >= b.saved_mask_original_words.size())
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return !in.synthetic_terminator && in == original;
}

bool fragment_saved_mask_is_bool_only(const SpirvCompute& b, const Rdna2Inst& in, int root) {
    if (!fragment_copy_domain(b) || root < 0 || root > 104 ||
        !b.fragment_bool_only_saved_mask_pcs.contains(in.pc) ||
        !native_vcc_copy_execution_domain_matches(b) ||
        in.pc >= b.saved_mask_original_words.size() || !scalar_write_is_b64_mask(in, root))
        return false;
    auto original = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                     b.saved_mask_original_words.size() - in.pc);
    original.pc = in.pc;
    return !in.synthetic_terminator && in == original;
}

std::array<uint32_t, 2> native_logical_result_words(SpirvCompute& b, const RegState& rs,
                                                    const Rdna2Inst& in, uint32_t result_mask) {
    if (!is_native_b64_logical(b, in)) return {};
    // Read both complete current operands before any destination expires an overlapping source.
    // CONFIDENCE: HIGH -- AMD 70648 section12.1 defines B64 logic on all64 source bits. Retained
    // DATA therefore needs no rendezvous with departed lanes, even in a partial host region.
    const auto a = source_words(b, rs, in, in.src[0]);
    const auto c = source_words(b, rs, in, in.src[1]);
    if (a[0] && a[1] && c[0] && c[1])
        return {logical_word(b, in.opcode, a[0], c[0]), logical_word(b, in.opcode, a[1], c[1])};
    // A Bool-only result is not scalar DATA. The sole ballot fallback requires the original
    // complete producer's participation proof; width, wave_ok and a register number are not it.
    if (!native_logical_data_demanded(b, in) ||
        !native_logical_producer_has_full_participation(b, in) || marked_source(rs, in.src[0]) ||
        marked_source(rs, in.src[1]))
        return {};
    return {b.native_wave_ballot_half(result_mask, 0), b.native_wave_ballot_half(result_mask, 1)};
}

void record_native_logical_word_gaps(const SpirvCompute& b, RegState& rs, const Rdna2Inst& in) {
    if (!is_native_b64_logical(b, in)) return;
    for (int word : {in.dst.value, in.dst.value + 1})
        if (!rs.sreg.contains(word) || sreg_word_may_be_fabricated(rs, word) ||
            rs.sreg_memory_pattern.contains(word))
            rs.unavailable_saved_mask_words.insert(word);
}
}   // namespace prosper::gpu
