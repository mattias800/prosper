#include "gpu/recompiler/rdna2_native_entry_exec_words.hpp"
#include "gpu/recompiler/compute_dispatch_launch.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_legacy_vcc_writes.hpp"
#include "gpu/recompiler/ir/ops/scalar_alu.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_native_logical_words.hpp"
#include "gpu/recompiler/rdna2_native_launch.hpp"
#include "gpu/recompiler/rdna2_saved_mask_words.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct NativeEntryExecWordPlan {
    const SpirvCompute* owner = nullptr;
    ComputeDispatchLaunchProfile launch;
    std::vector<Rdna2Inst> original;
    std::vector<uint32_t> words;
    std::set<uint32_t> producers;
    std::vector<std::vector<size_t>> predecessors;
    uint32_t partial = 0, last_group = 0;
    bool revoked = false;
};
namespace {
using Identities = std::bitset<127>;
bool native_context(const SpirvCompute& b) {
    return b.is_compute && !b.is_fragment && !b.is_vertex && !b.is_fragment_packet() &&
           b.guest_stage == GuestShaderStage::Compute &&
           b.physical_domain == PhysicalExecutionDomain::Workgroup && !b.ngg_workgroup_shell &&
           !b.ngg_one_lane && !b.allow_b32_masks && b.wave_size == 64 &&
           b.native_subgroup_size == 64 && b.local_count == 64;
}
std::optional<uint32_t> entry_partial(const ComputeDispatchLaunchProfile& profile) {
    if (profile.wave_size != 64 || profile.local != std::array<uint32_t, 3>{64, 1, 1} ||
        profile.threads[1] != 1 || profile.threads[2] != 1)
        return {};
    if (compute_dispatch_has_complete_groups(profile)) return 0;
    // A closed 1D thread-dimension launch. PARTIAL_TG_EN and its exact producing word are
    // required; normalized geometry alone cannot supply any initial EXEC bits.
    const auto& raw = profile.launch;
    const uint32_t remainder = profile.threads[0] % 64;
    if (!profile.exact_thread_extent || raw.known != ComputeDispatchLaunchState::kAll ||
        raw.initiator != 0x27u || raw.dimensions != profile.threads || !remainder ||
        raw.num_thread != std::array<uint32_t, 3>{64u | (remainder << 16), 0x10001u, 0x10001u})
        return {};
    return remainder;
}
bool fresh_boundary(const SpirvCompute& b) {
    if (b.saved_mask_word_analysis_done || b.invocation_guard_merge) return false;
    for (size_t at = 0; at < b.code.size();) {
        const uint32_t width = b.code[at] >> 16, op = b.code[at] & 0xffffu;
        if (!width || width > b.code.size() - at || op == Op_BranchConditional || op == Op_Switch ||
            op == Op_LoopMerge || op == Op_Kill)
            return false;
        at += width;
    }
    return true;
}
bool closed_program(const std::vector<Rdna2Inst>& ins,
                    std::vector<std::vector<size_t>>& successors) {
    if (ins.empty() || ins.front().pc || !ins.back().is_end) return false;
    std::map<uint32_t, size_t> at;
    uint64_t next = 0;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        const auto& e = rdna2_instruction_effects(in);
        if (in.pc != next || !in.len_dwords || in.synthetic_terminator ||
            e.status == Rdna2EffectsStatus::Unknown || e.status == Rdna2EffectsStatus::Reserved ||
            e.status == Rdna2EffectsStatus::Unmodeled || e.control >= Rdna2ControlFlow::Trap ||
            e.m0_relative != Rdna2M0Relative::None || rdna2_escapes_decoded_effects(in) ||
            in.fmt == Rdna2Format::EXP || in.fmt == Rdna2Format::VINTRP ||
            (in.fmt == Rdna2Format::SOPK && (in.opcode == 0x13 || in.opcode == 0x15)))
            return false;
        at.emplace(in.pc, i);
        next += in.len_dwords;
        if (next > UINT32_MAX) return false;
    }
    successors.resize(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        const bool branch =
            in.fmt == Rdna2Format::SOPP && (in.opcode == 2 || (in.opcode >= 4 && in.opcode <= 9));
        if (in.fmt == Rdna2Format::SOPP && !branch && !in.is_end && in.opcode != 0 &&
            in.opcode != 0xa && in.opcode != 0xc && in.opcode != 0x20 && in.opcode != 0x21 &&
            in.opcode != 0x22)
            return false;
        if (branch) {
            const int64_t target = int64_t(in.pc) + in.len_dwords + in.simm16;
            if (target < 0 || target > UINT32_MAX || !at.contains(uint32_t(target))) return false;
            successors[i].push_back(at.at(uint32_t(target)));
        }
        if (!in.is_end && i + 1 < ins.size() && (!branch || in.opcode != 2))
            successors[i].push_back(i + 1);
    }
    return true;
}
void kill(Identities& values, int base, uint32_t width) {
    for (int root = 0; root < 127; ++root)
        if (base < root + 2 && int64_t(root) < int64_t(base) + width) values.reset(root);
}
Identities transfer(Identities before, const Rdna2Inst& in) {
    auto after = before;
    for_each_scalar_write(in, [&](int base, uint32_t width) { kill(after, base, width); });
    if (rdna2_instruction_may_change_exec(in)) kill(after, 126, 2);
    const bool compare =
        in.fmt == Rdna2Format::VOPC || (in.fmt == Rdna2Format::VOP3 && in.opcode < 0x100);
    if (compare && !vopc_is_cmpx(in.opcode)) kill(after, in.dst.value, 2);
    // A conservative VCC alias kill, independently of the explicit writer inventory. In
    // particular CMPX writes EXEC and preserves VCC; ordinary VALU cannot create Entry facts.
    const bool valu = in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 ||
                      in.fmt == Rdna2Format::VOP3 || in.fmt == Rdna2Format::VOP3P;
    if (valu && !(compare && vopc_is_cmpx(in.opcode))) kill(after, 106, 2);
    if (rdna2_instruction_effects(in).implicit_writes & kRdna2ImplicitVcc) kill(after, 106, 2);
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
        (in.src[0].kind == OperandKind::SGPR || in.src[0].kind == OperandKind::Special) &&
        in.src[0].value >= 0 && in.src[0].value < 127 && in.dst.value >= 0 && in.dst.value < 127)
        after.set(in.dst.value, before.test(in.src[0].value));
    if (in.fmt == Rdna2Format::SOP1 && sop1_opcode_is_saveexec_b64(in.opcode)) {
        after.reset(126);
        if (in.dst.value >= 0 && in.dst.value < 126) after.set(in.dst.value, before.test(126));
    }
    return after;
}
std::set<uint32_t> entry_producers(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins) {
    std::vector<std::vector<size_t>> successors;
    if (!closed_program(ins, successors)) return {};
    std::vector<std::optional<Identities>> incoming(ins.size());
    incoming[0] = Identities{}.set(126);
    std::deque<size_t> work{0};
    std::vector<bool> queued(ins.size());
    queued[0] = true;
    size_t steps = 0;
    while (!work.empty()) {
        if (++steps > ins.size() * 2048u) return {};
        const size_t i = work.front();
        work.pop_front();
        queued[i] = false;
        const auto& state = incoming[i];
        if (!state) return {};
        const auto after = transfer(*state, ins[i]);
        for (size_t next : successors[i]) {
            auto& slot = incoming[next];
            auto value = after;
            if (slot) {
                value = *slot & after;
                if (*slot == value) continue;
            }
            slot = value;
            if (!queued[next]) {
                work.push_back(next);
                queued[next] = true;
            }
        }
    }
    std::set<uint32_t> result;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& state = incoming[i];
        if (state && state->test(126) && native_entry_exec_word_move(b, ins[i]))
            result.insert(ins[i].pc);
    }
    return result;
}
// This proof grants no value or definedness. Each observed word needs one invariant reaching
// expression. Multiple definitions, value cycles, vector publications and late memory loads are
// unproved; a complete current overwrite may retire all of them without changing older routes.
class UniformWords {
    const NativeEntryExecWordPlan& plan;
    size_t budget = 8192;
    using Key = std::pair<size_t, int>;
    std::map<Key, bool> memo;
    std::set<Key> visiting;
    bool writes(const Rdna2Inst& in, int word) const {
        if (word == 128) return rdna2_writes_scc(in.fmt, in.opcode) != Rdna2Answer::No;
        bool found = (word == 106 || word == 107) && legacy_valu_writes_vcc(in);
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            found |= word >= base && int64_t(word) < int64_t(base) + width;
        });
        return found;
    }
    std::optional<size_t> definition(size_t before, int word) {
        const size_t entry = plan.original.size();
        std::set<size_t> definitions, seen;
        auto pending = plan.predecessors[before];
        if (!before) definitions.insert(entry);
        while (!pending.empty()) {
            if (!budget) return {};
            --budget;
            const size_t at = pending.back();
            pending.pop_back();
            if (!seen.insert(at).second) continue;
            if (writes(plan.original[at], word)) {
                definitions.insert(at);
            } else {
                if (!at) definitions.insert(entry);
                const auto& incoming = plan.predecessors[at];
                pending.insert(pending.end(), incoming.begin(), incoming.end());
            }
            if (definitions.size() > 1) return {};
        }
        if (definitions.empty()) return {};
        return *definitions.begin();
    }
    bool prefix_load(size_t at) const {
        const auto& in = plan.original[at];
        if (in.fmt != Rdna2Format::SMEM || in.opcode != 0) return false;
        for (const auto& candidate : plan.original) {
            if (candidate.pc < in.pc &&
                (rdna2_may_write_guest_memory(candidate) ||
                 (candidate.fmt == Rdna2Format::SOPP && candidate.opcode == 0xa)))
                return false;
            const bool branch =
                candidate.fmt == Rdna2Format::SOPP &&
                (candidate.opcode == 2 || (candidate.opcode >= 4 && candidate.opcode <= 9));
            if (!branch) continue;
            const int64_t target = int64_t(candidate.pc) + candidate.len_dwords + candidate.simm16;
            if (candidate.pc < in.pc || target <= in.pc) return false;
        }
        return true;
    }
    bool source(size_t at, const Operand& operand, uint32_t width) {
        if (operand.kind == OperandKind::InlineInt || operand.kind == OperandKind::InlineFloat ||
            operand.kind == OperandKind::Literal)
            return true;
        if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special) return false;
        if (operand.value == 125) return true;
        if (operand.value == 253) return word(at, 128);
        if (!width || width > 2 || operand.value < 0 || operand.value + int(width) > 126)
            return false;
        for (uint32_t part = 0; part < width; ++part)
            if (!word(at, operand.value + int(part))) return false;
        return true;
    }
    bool expression(size_t at, int destination) {
        const auto& in = plan.original[at];
        if ((destination == 106 || destination == 107) && plan.producers.contains(in.pc))
            return true; // the separately authenticated original entry word, not current EXEC
        const bool load = prefix_load(at);
        const bool move = in.fmt == Rdna2Format::SOP1 &&
                          (in.opcode == kSop1OpcodeMovB32 || in.opcode == kSop1OpcodeMovB64);
        if (!load && !move && !ir::find_scalar_op(in.fmt, in.opcode)) return false;
        for (uint32_t k = 0; k < in.n_src; ++k) {
            // S_LOAD_DWORD has a real two-word scalar address and a one-word offset. ALU
            // widths deliberately do not describe memory operands; no new memory facts are used.
            const uint32_t width = load ? (k == 0 ? 2u : 1u) : scalar_alu_source_words(in, k);
            if (!source(at, in.src[k], width)) return false;
        }
        if ((rdna2_instruction_effects(in).implicit_reads & kRdna2ImplicitScc) && !word(at, 128))
            return false;
        const auto old = scalar_implicit_destination_read_width(in);
        for (uint32_t part = 0; part < old; ++part)
            if (!word(at, in.dst.value + int(part))) return false;
        return true;
    }

public:
    explicit UniformWords(const NativeEntryExecWordPlan& original) : plan(original) {}
    bool word(size_t before, int physical) {
        const Key key{before, physical};
        if (const auto found = memo.find(key); found != memo.end()) return found->second;
        if (!visiting.insert(key).second) return false;
        const auto producer = definition(before, physical);
        const bool result =
            producer &&
            (*producer == plan.original.size() ? physical != 128 : expression(*producer, physical));
        visiting.erase(key);
        memo.emplace(key, result);
        return result;
    }
};
bool uniform_vcc_pair(const SpirvCompute& b, const Rdna2Inst& in) {
    const auto& plan = b.native_entry_exec_word_plan;
    if (!plan || plan->revoked || b.saved_mask_original_fragment_entry || plan->owner != &b ||
        plan->launch != b.guest_compute_launch || plan->words != b.saved_mask_original_words)
        return false;
    const auto found =
        std::lower_bound(plan->original.begin(), plan->original.end(), in.pc,
                         [](const auto& candidate, uint32_t pc) { return candidate.pc < pc; });
    if (found == plan->original.end() || in != *found) return false;
    const auto before = size_t(found - plan->original.begin());
    UniformWords proof(*plan);
    return proof.word(before, 106) && proof.word(before, 107);
}
bool original_region(const NativeEntryExecWordPlan& plan, const std::vector<Rdna2Inst>& region) {
    if (!std::is_sorted(region.begin(), region.end(),
                        [](const auto& a, const auto& c) { return a.pc < c.pc; }))
        return false;
    for (const auto& in : region) {
        if (in.synthetic_terminator && in.is_end && in.fmt == Rdna2Format::SOPP && in.opcode == 1)
            continue;
        const auto found =
            std::lower_bound(plan.original.begin(), plan.original.end(), in.pc,
                             [](const auto& candidate, uint32_t pc) { return candidate.pc < pc; });
        if (found == plan.original.end() || in != *found) return false;
    }
    return true;
}
}   // namespace
bool native_entry_exec_word_move(const SpirvCompute& b, const Rdna2Inst& in) {
    return native_context(b) && !in.synthetic_terminator && in.fmt == Rdna2Format::SOP1 &&
           in.opcode == kSop1OpcodeMovB64 && in.dst.value == 106 &&
           in.src[0].kind == OperandKind::Special && in.src[0].value == 126;
}
void initialize_native_entry_exec_words(SpirvCompute& b, const ComputeDispatchLaunchProfile& launch,
                                        const std::vector<Rdna2Inst>& selected,
                                        const uint32_t* code, size_t dwords) {
    if (b.native_entry_exec_word_plan || !native_context(b) || !fresh_boundary(b) || !code ||
        !dwords)
        return;
    const auto partial = entry_partial(launch);
    if (!partial) return;
    std::vector<Rdna2Inst> original;
    rdna2_walk(code, dwords, original);
    if (selected != original) return;
    std::vector<std::vector<size_t>> successors;
    if (!closed_program(original, successors)) return;
    auto producers = entry_producers(b, original);
    auto plan = std::make_shared<NativeEntryExecWordPlan>();
    plan->owner = &b;
    plan->launch = launch;
    plan->original = std::move(original);
    plan->words.assign(code, code + dwords);
    plan->producers = std::move(producers);
    plan->predecessors.resize(plan->original.size());
    for (size_t from = 0; from < successors.size(); ++from)
        for (size_t to : successors[from]) plan->predecessors[to].push_back(from);
    plan->partial = *partial;
    plan->last_group = launch.threads[0] / 64;
    b.native_entry_exec_word_plan = std::move(plan);
}
void constrain_native_entry_exec_words_region(SpirvCompute& b, const std::vector<Rdna2Inst>& ins) {
    if (b.native_entry_exec_word_plan && !original_region(*b.native_entry_exec_word_plan, ins)) {
        auto revoked = std::make_shared<NativeEntryExecWordPlan>(*b.native_entry_exec_word_plan);
        revoked->producers.clear();
        revoked->revoked = true;
        b.native_entry_exec_word_plan = std::move(revoked);
    }
}
bool native_entry_exec_word_proven(const SpirvCompute& b, const Rdna2Inst& in) {
    const auto& plan = b.native_entry_exec_word_plan;
    if (!plan || plan->owner != &b || !native_entry_exec_word_move(b, in) ||
        !plan->producers.contains(in.pc) || plan->launch != b.guest_compute_launch ||
        plan->launch.local != b.guest_local_size || b.saved_mask_original_fragment_entry ||
        plan->launch != b.saved_mask_original_compute_launch ||
        plan->words != b.saved_mask_original_words ||
        !b.native_logical_data_beyond_mask_spill_pcs.contains(in.pc) ||
        (plan->partial && !b.groupid[0]))
        return false;
    const auto original =
        std::lower_bound(plan->original.begin(), plan->original.end(), in.pc,
                         [](const auto& candidate, uint32_t pc) { return candidate.pc < pc; });
    return original != plan->original.end() && in == *original;
}
uint32_t native_compute_vcc_data_mask(SpirvCompute& b, const RegState& rs, const Rdna2Inst& in) {
    uint32_t low = 0, high = 0;
    if (!native_context(b) || !entry_partial(b.guest_compute_launch) || !uniform_vcc_pair(b, in) ||
        b.guest_compute_launch.local != b.guest_local_size ||
        b.guest_compute_launch != b.saved_mask_original_compute_launch ||
        !native_logical_data_word(rs, 106, low) || !native_logical_data_word(rs, 107, high) ||
        sreg_word_may_be_fabricated(rs, 106) || sreg_word_may_be_fabricated(rs, 107) ||
        rs.lane_local_scalars.contains(low) || rs.lane_local_scalars.contains(high))
        return 0;
    // The physical pair is already genuine DATA. Observe its guest lane bit; no ballot or
    // previously retired predicate can supply a missing half or replace these current words.
    const auto lane = b.ibin(Op_BitwiseAnd, b.guest_lane_id(), b.uconst(63));
    const auto word = b.sel(b.ucmp(Op_UGreaterThanEqual, lane, b.uconst(32)), high, low);
    const auto bit = b.ibin(Op_BitwiseAnd, lane, b.uconst(31));
    return b.ucmp(Op_INotEqual,
                  b.ibin(Op_BitwiseAnd, b.ibin(Op_ShiftRightLogical, word, bit), b.uconst(1)),
                  b.uconst(0));
}
std::optional<std::array<uint32_t, 2>> native_entry_exec_words(SpirvCompute& b,
                                                               const Rdna2Inst& in) {
    if (!native_entry_exec_word_proven(b, in) ||
        saved_mask_producer_has_full_participation(b, in) ||
        native_uniform_exec_mov_has_full_participation(b, in))
        return {};
    const auto& plan = *b.native_entry_exec_word_plan;
    if (!plan.partial) return std::array<uint32_t, 2>{b.uconst(UINT32_MAX), b.uconst(UINT32_MAX)};
    // These are the recorded launch's entry bits, not a vote of the current participants or EXEC.
    const uint64_t bits = (uint64_t{1} << plan.partial) - 1;
    const auto last = b.ucmp(Op_IEqual, b.groupid[0], b.uconst(plan.last_group));
    return std::array<uint32_t, 2>{
        b.sel(last, b.uconst(uint32_t(bits)), b.uconst(UINT32_MAX)),
        b.sel(last, b.uconst(uint32_t(bits >> 32)), b.uconst(UINT32_MAX))};
}
}   // namespace prosper::gpu
