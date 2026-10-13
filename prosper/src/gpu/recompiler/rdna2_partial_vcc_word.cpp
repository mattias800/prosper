#include "gpu/recompiler/rdna2_partial_vcc_word.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <array>
#include <map>
#include <optional>

namespace prosper::gpu {
namespace {
struct Facts {
    bool vcc = false;
    bool wave_control = true;
    bool scc_uniform = false;
    std::array<std::optional<uint32_t>, 128> words{};
};

void meet(Facts& into, const Facts& other) {
    into.vcc &= other.vcc;
    into.wave_control &= other.wave_control;
    into.scc_uniform &= other.scc_uniform;
    for (size_t word = 0; word < into.words.size(); ++word)
        if (into.words[word] != other.words[word]) into.words[word].reset();
}

bool plain_move(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB32 && in.n_src == 1 &&
           !in.synthetic_terminator;
}

std::optional<uint32_t> constant_word(const Facts& before, const Rdna2Inst& in, size_t index = 0) {
    if (index >= in.n_src) return std::nullopt;
    const auto source = in.src[index];
    if (source.kind == OperandKind::InlineInt) return static_cast<uint32_t>(source.value);
    if (source.kind == OperandKind::Literal && in.has_literal) return in.literal;
    if ((source.kind == OperandKind::SGPR || source.kind == OperandKind::Special) &&
        source.value >= 0 && source.value < 124)
        return before.words[source.value];
    return std::nullopt;
}

bool lane_context(const SpirvCompute& b) {
    // CONFIDENCE: HIGH -- ordinary compute retains actual logical LocalInvocationId. This is
    // a predicate bit, never a physical ballot or evidence of raw/native launch participation.
    return b.is_compute && !b.is_vertex && !b.is_fragment && !b.is_fragment_packet() &&
           b.guest_stage == GuestShaderStage::Compute &&
           b.physical_domain == PhysicalExecutionDomain::Workgroup && !b.ngg_workgroup_shell &&
           !b.ngg_one_lane && !b.allow_b32_masks && b.wave_size == 64 && b.linear_localid &&
           (!b.native_subgroup_size || b.native_subgroup_size == 64) && b.local_count >= 64 &&
           b.local_count % 64 == 0 && !b.invocation_guard_merge;
}
} // namespace

void initialize_partial_vcc_word_proof(SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                       const uint32_t* code, size_t dwords, bool partial_entry) {
    if (b.partial_vcc_word_analysis_done) {
        // Recursive slices or a changed original stream may revoke, never reseed, this proof.
        if (partial_entry || !code || dwords != b.partial_vcc_original_words.size() ||
            !std::equal(b.partial_vcc_original_words.begin(), b.partial_vcc_original_words.end(),
                        code))
            b.partial_vcc_word_pcs.clear();
        else {
            std::vector<Rdna2Inst> original;
            rdna2_walk(code, dwords, original);
            if (ins != original) b.partial_vcc_word_pcs.clear();
        }
        return;
    }
    b.partial_vcc_word_analysis_done = true;
    if (partial_entry || !lane_context(b) || !code || !dwords || ins.empty() || ins.size() > 4096)
        return;
    if (std::none_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return plain_move(in) && (in.dst.value == 106 || in.dst.value == 107);
        }))
        return;
    std::vector<Rdna2Inst> original;
    rdna2_walk(code, dwords, original);
    if (ins != original) return;
    b.partial_vcc_original_words.assign(code, code + dwords);
    // The invocation-local READFIRST fallback can reach both scalar branches and the comparison
    // itself through scalar/vector relays. A builder service flag is not proof that this body
    // actually dispatches its exact common-phase service. Keep this new recovery out of every
    // READFIRST program until that producing route is bound; existing admissions are unchanged.
    if (std::any_of(ins.begin(), ins.end(), [](const Rdna2Inst& in) {
            return in.fmt == Rdna2Format::VOP1 && in.opcode == 0x02;
        }))
        return;

    std::map<uint32_t, size_t> at;
    for (size_t i = 0; i < ins.size(); ++i)
        if (!at.emplace(ins[i].pc, i).second) return;
    std::vector<std::vector<size_t>> successors(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.synthetic_terminator || in.fmt == Rdna2Format::Unknown ||
            rdna2_escapes_decoded_effects(in))
            return;
        if (in.is_end) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (in.fmt == Rdna2Format::SOPP && !branch && in.opcode != 0 && in.opcode != 0x0c &&
            in.opcode != 0x20 && in.opcode != 0x21 && in.opcode != 0x22)
            return;
        if (branch) {
            const int64_t target = int64_t{in.pc} + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX) return;
            const auto found = at.find(static_cast<uint32_t>(target));
            // No old logical HIGH view crosses a host loop generation on this new route.
            if (found == at.end() || found->second <= i) return;
            successors[i].push_back(found->second);
        }
        if (!(branch && in.opcode == kSoppOpcodeBranch) && i + 1 < ins.size())
            successors[i].push_back(i + 1);
    }

    std::vector<std::optional<Facts>> incoming(ins.size());
    incoming[0] = Facts{};   // initial false/missing VCC and scalar SSA are not genuine producers
    for (size_t i = 0; i < ins.size(); ++i) {
        if (!incoming[i]) continue;
        const auto& in = ins[i];
        const auto before = *incoming[i];
        auto after = before;
        // CONFIDENCE: HIGH -- a scalar instruction belongs to the whole guest wave. The legacy
        // READFIRST fallback can instead drive invocation-local SCC control. A reaching VCC
        // producer alone cannot authorize partial replacement inside that host branch. Only
        // genuine numeric constant/copy comparisons establish this bounded SCC proof; once an
        // unproved scalar source or branch is crossed, no later compare recovers that authority.
        if (in.fmt == Rdna2Format::SOPC)
            after.scc_uniform =
                in.opcode <= 0x0bu && constant_word(before, in, 0) && constant_word(before, in, 1);
        else if (in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK ||
                 (in.fmt == Rdna2Format::SOP1 && !sop1_opcode_leaves_scc_unmodified(in.opcode)))
            after.scc_uniform = false;
        if (scalar_reads_scc(in) ||
            (in.fmt == Rdna2Format::SOPP && (in.opcode == 4u || in.opcode == 5u)))
            after.wave_control &= before.scc_uniform;
        const auto source = plain_move(in) ? constant_word(before, in) : std::nullopt;
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            for (uint32_t half = 0; half < width; ++half) {
                const int word = base + static_cast<int>(half);
                if (word >= 0 && word < 128) after.words[word].reset();
                if (word == 106 || word == 107) after.vcc = false;
            }
        });
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode) &&
            !(in.dst.kind == OperandKind::SGPR && in.dst.value <= 105)) {
            after.words[106].reset();
            after.words[107].reset();
            after.vcc = !in.has_sdwa && !in.has_dpp;
        }
        if (in.fmt == Rdna2Format::VOP2 && in.opcode >= 0x28 && in.opcode <= 0x2a) {
            after.vcc = false;
            after.words[106].reset();
            after.words[107].reset();
        }
        if (plain_move(in) && source && in.dst.value >= 0 && in.dst.value < 124) {
            after.words[in.dst.value] = source;
            if (in.dst.value == 106 || in.dst.value == 107) {
                after.vcc = before.vcc;
                if (before.vcc && before.wave_control) b.partial_vcc_word_pcs.emplace(in.pc, in);
            }
        }
        for (size_t to : successors[i]) {
            if (incoming[to])
                meet(*incoming[to], after);
            else
                incoming[to] = after;
        }
    }
}

uint32_t partial_vcc_word_predicate(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in,
                                    uint32_t source) {
    // CONFIDENCE: HIGH -- AMD 70648 sections 3.6/3.9 and Table SOP1: MOV_B32 replaces exactly
    // one scalar dword, independently of EXEC. Retain the other genuine comparison bits without
    // publishing a numeric ballot or a complete saved-SGPR mask alias.
    const auto proof = b.partial_vcc_word_pcs.find(in.pc);
    if (proof == b.partial_vcc_word_pcs.end()) return 0;
    const auto decline = [&] {
        // Static source facts are necessary, not permission to replace an unavailable current
        // generation. A declined partial writer retires VCC in the ordinary caller. Revoke this
        // new recovery so even a later CFG zero placeholder cannot stand for its lost sibling.
        b.partial_vcc_word_pcs.clear();
        return 0u;
    };
    if (!lane_context(b) || !rs.vcc || in.synthetic_terminator || in != proof->second)
        return decline();
    if (in.src[0].kind == OperandKind::SGPR || in.src[0].kind == OperandKind::Special) {
        const int word = in.src[0].value;
        const auto current = rs.sreg.find(word);
        if (current == rs.sreg.end() || current->second != source ||
            sreg_word_may_be_fabricated(rs, word) || rs.sreg_memory_pattern.contains(word) ||
            rs.sreg_srt.contains(word) || rs.sreg_entry_m0.contains(word) ||
            rs.unavailable_saved_mask_words.contains(word) ||
            rs.lane_local_scalars.contains(source))
            return decline();
    }
    const uint32_t lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
    const uint32_t high = b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32));
    const uint32_t bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
    const uint32_t replacement = b.ucmp(
        Op_INotEqual, b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, source, bit), b.uconst(1)),
        b.uconst(0));
    return b.bsel(in.dst.value == 107 ? high : b.bsel(high, b.bfalse(), b.btrue()), replacement,
                  rs.vcc);
}

void publish_s_mov_b32_word(RegState& rs, const Rdna2Inst& in, uint32_t source,
                            uint32_t predicate) {
    rs.sreg[in.dst.value] = source;
    // Preserve the existing MOV descriptor transfer: a selected four-word descriptor replaces
    // the old destination tag, including ordinary scalar self-copies.
    const auto tag = rs.sreg_srt.find(in.src[0].value);
    if ((in.src[0].kind == OperandKind::SGPR ||
         (in.src[0].kind == OperandKind::Special && in.src[0].value >= 106 &&
          in.src[0].value <= 123)) &&
        tag != rs.sreg_srt.end())
        rs.sreg_srt[in.dst.value] = tag->second;
    else
        rs.sreg_srt.erase(in.dst.value);
    if (predicate) {
        rs.vcc = predicate;
        rs.vcc_wave_uniform = 0;   // the replacement can change a previous uniform truth value
    }
}
}   // namespace prosper::gpu
