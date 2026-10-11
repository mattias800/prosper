#include "gpu/recompiler/rdna2_entry_exec_constants.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>

namespace prosper::gpu {
namespace {
constexpr uint32_t sub_u32 = 0x01, subb_u32 = 0x05;

std::array<uint32_t, 9> entry_domain(const ComputeShaderConfig& c) {
    return {c.local_x,
            c.local_y,
            c.local_z,
            c.threads_x,
            c.threads_y,
            c.threads_z,
            c.wave_size,
            c.native_subgroup_size,
            c.exact_thread_extent};
}

bool complete_portable_launch(const ComputeShaderConfig& c) {
    // CONFIDENCE: HIGH. RDNA2 70648 section 3.12.1 initializes EXEC to the valid work-item mask.
    // Neither normalized dimensions, a host Bool nor host subgroup width proves physical guest
    // EXEC bits. Complete local groups additionally need a whole number of guest Wave64s here.
    if (c.wave_size != 64 || c.native_subgroup_size || !compute_dispatch_has_complete_groups(c))
        return false;
    const uint64_t xy = uint64_t{c.local_x} * c.local_y;
    const uint64_t count = xy * c.local_z;
    return count >= 64 && count % 64 == 0;
}

bool audited_integer_read(const Rdna2Inst& in) {
    // InlineInt(-1) preserves U32 bits for these integer operations. It is NOT a substitute for
    // EXEC's bits in floating-point source decoding. Keep opcode, SCC/borrow, width and destination.
    if (in.fmt != Rdna2Format::SOP2 || (in.opcode != sub_u32 && in.opcode != subb_u32) ||
        in.n_src != 2 || in.dst.kind != OperandKind::SGPR || in.dst.value < 0 ||
        in.dst.value > 105 || in.has_modifier || in.has_sdwa || in.has_dpp || in.clamp || in.omod)
        return false;
    for (uint32_t s = 0; s < 2; ++s)
        if (in.src_neg[s] || in.src_abs[s]) return false;
    return true;
}

bool exec_word(const Operand& source) {
    return source.kind == OperandKind::Special && (source.value == 126 || source.value == 127);
}

bool known_control(const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOPP) return true;
    return in.opcode == 0 || in.is_end || sopp_opcode_is_direct_branch(in.opcode) ||
           in.opcode == kSoppOpcodeBarrier || in.opcode == 0x0c || in.opcode == 0x20;
}
}   // namespace

std::vector<uint32_t> portable_entry_exec_constant_reads(const std::vector<Rdna2Inst>& ins,
                                                         const ComputeShaderConfig& config) {
    if (!complete_portable_launch(config) || ins.empty() || ins.front().pc != 0 ||
        !ins.back().is_end || ins.back().synthetic_terminator)
        return {};
    std::unordered_set<uint32_t> starts;
    uint64_t next = 0;
    for (const auto& in : ins) {
        if (in.pc != next || !in.len_dwords || in.fmt == Rdna2Format::Unknown ||
            in.synthetic_terminator || rdna2_escapes_decoded_effects(in) ||
            in.fmt == Rdna2Format::EXP || !known_control(in))
            return {};
        starts.insert(in.pc);
        next += in.len_dwords;
    }
    std::vector<uint32_t> reads;
    for (const auto& in : ins) {
        // The original first straight-line epoch only. Restoring EXEC later does not restore
        // original-entry authority, and a barrier/selection begins a different body epoch.
        if (rdna2_instruction_may_change_exec(in) || in.fmt == Rdna2Format::SOPK ||
            (in.fmt == Rdna2Format::SOPP &&
             (sopp_opcode_is_direct_branch(in.opcode) || in.opcode == kSoppOpcodeBarrier)))
            break;
        if (audited_integer_read(in) && (exec_word(in.src[0]) || exec_word(in.src[1])))
            reads.push_back(in.pc);
    }
    if (reads.empty()) return {};
    for (const auto& in : ins) {
        if (in.fmt != Rdna2Format::SOPP || !sopp_opcode_is_direct_branch(in.opcode)) continue;
        const int64_t target = int64_t{in.pc} + in.len_dwords + in.simm16;
        // No edge may re-enter any specialized read under a later EXEC generation. Also reject
        // invalid targets instead of treating the decoded local span as a new root program.
        if (target < 0 || target > std::numeric_limits<uint32_t>::max() ||
            !starts.contains(static_cast<uint32_t>(target)) || target <= reads.back())
            return {};
    }
    return reads;
}

PortableEntryExecRoot::PortableEntryExecRoot(const uint32_t* code, size_t dwords,
                                             const ComputeShaderConfig& config)
    : domain_(entry_domain(config)), launch_(config.dispatch_launch) {
    if (!code || !dwords || !complete_portable_launch(config)) return;
    if (rdna2_walk(code, dwords, original_) != dwords) return;
    reads_ = portable_entry_exec_constant_reads(original_, config);
    if (!reads_.empty()) words_.assign(code, code + dwords);
}

void PortableEntryExecRoot::specialize(const uint32_t* code, size_t dwords,
                                       const ComputeShaderConfig& config,
                                       std::vector<Rdna2Inst>& instructions) const {
    if (reads_.empty() || !code || dwords != words_.size() || entry_domain(config) != domain_ ||
        config.dispatch_launch != launch_ || instructions != original_ ||
        !std::equal(words_.begin(), words_.end(), code))
        return;
    for (auto& in : instructions) {
        if (!std::binary_search(reads_.begin(), reads_.end(), in.pc)) continue;
        for (uint32_t s = 0; s < 2; ++s)
            if (exec_word(in.src[s])) in.src[s] = {OperandKind::InlineInt, -1};
    }
}
}   // namespace prosper::gpu
