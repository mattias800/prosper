#include "gpu/recompiler/rdna2_compute_quad_init.hpp"

#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_recompile_shared.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include <array>
#include <bitset>
#include <optional>
#include <queue>

namespace prosper::gpu {
bool compute_dpp_quad_or(const Rdna2Inst& in) {
    return in.fmt == Rdna2Format::VOP2 && in.opcode == 0x1cu && in.has_dpp && !in.has_modifier &&
           !in.has_sdwa && in.dpp_ctrl < 256 && in.dpp_bound_ctrl && in.dpp_row_mask == 15 &&
           in.dpp_bank_mask == 15 && in.n_src == 2 && in.dst.kind == OperandKind::VGPR &&
           in.dst.value >= 0 && in.dst.value < 256 && in.src[0].kind == OperandKind::VGPR &&
           in.src[0].value >= 0 && in.src[0].value < 256 && in.src[1].kind == OperandKind::VGPR &&
           in.src[1].value >= 0 && in.src[1].value < 256 && !in.src_neg[0] && !in.src_neg[1] &&
           !in.src_abs[0] && !in.src_abs[1] && !in.clamp && !in.omod;
}
namespace {
using Coverage = std::bitset<256>;
Coverage full() {
    return Coverage{}.set();
}
Coverage atom(uint32_t generation) {
    Coverage result;
    for (uint32_t assignment = 0; assignment < 256; ++assignment)
        result[assignment] = (assignment & (1u << generation)) != 0;
    return result;
}
struct Facts {
    Coverage exec = full();
    std::array<Coverage, 256> vector{};
};
bool true_constant(const SpirvCompute& b, uint32_t value) {
    for (size_t at = 0; at < b.types.size();) {
        const uint32_t words = b.types[at] >> 16u;
        if (!words || words > b.types.size() - at) return false;
        if ((b.types[at] & 0xffffu) == Op_ConstantTrue && words == 3 &&
            b.types[at + 1] == b.t_bool && b.types[at + 2] == value)
            return true;
        at += words;
    }
    return false;
}
bool clean_entry_vector(const RegState& entry, int reg) {
    const auto marks = entry.vreg_source_marks.find(reg);
    return marks == entry.vreg_source_marks.end() ||
           (!marks->second.placeholder && !marks->second.hard && !marks->second.memory &&
            marks->second.blanket_roots.empty());
}
Coverage source(const Facts& facts, const Rdna2Inst& in, const Operand& o) {
    if (o.kind == OperandKind::InlineInt || o.kind == OperandKind::InlineFloat ||
        (o.kind == OperandKind::Literal && in.has_literal))
        return full();
    if (o.kind == OperandKind::VGPR && o.value >= 0 && o.value < 256)
        return facts.vector[static_cast<size_t>(o.value)];
    return {};
}
bool pure_word(const Rdna2Inst& in) {
    if (in.has_modifier || in.has_dpp || in.clamp || in.omod) return false;
    if (in.fmt == Rdna2Format::VOP1) return in.opcode == 0x01u;
    if (in.fmt == Rdna2Format::VOP2)
        return (in.opcode >= 0x16u && in.opcode <= 0x1du) || in.opcode == 0x25u;
    if (in.fmt == Rdna2Format::VOP3) return in.opcode == 0x148u || in.opcode == 0x36fu;
    return false;
}
std::optional<Coverage> exec_replacement(const Facts& before, const Rdna2Inst& in) {
    if (in.fmt != Rdna2Format::SOP1 || in.dst.value != 126) return std::nullopt;
    if (in.opcode == 0x08u && in.src[0].kind == OperandKind::Special && in.src[0].value == 126)
        return ~before.exec;
    if (in.opcode == kSop1OpcodeMovB64 && in.src[0].kind == OperandKind::InlineInt) {
        if (in.src[0].value == -1) return full();
        if (in.src[0].value == 0) return Coverage{};
    }
    return std::nullopt;
}
bool opaque_control(const Rdna2Inst& in) {
    return (in.fmt == Rdna2Format::SOP1 && in.opcode >= kSop1OpcodeSetpcB64 &&
            in.opcode <= kSop1OpcodeRfeB64) ||
           (in.fmt == Rdna2Format::SOPK &&
            (in.opcode == kSopkOpcodeCallB64 || in.opcode == kSopkOpcodeSubvectorLoopBegin ||
             in.opcode == kSopkOpcodeSubvectorLoopEnd));
}
} // namespace

std::set<uint32_t> proven_compute_quad_or_sources(SpirvCompute& b, const RegState& entry,
                                                  const std::vector<Rdna2Inst>& ins,
                                                  const uint32_t* code, size_t dwords,
                                                  uint32_t initial_active) {
    std::set<uint32_t> proven;
    if (!b.is_compute || !b.native_logical_uniform_launch || !b.guest_compute_complete_groups ||
        b.ngg_workgroup_shell || b.ngg_one_lane || b.is_fragment_packet() || initial_active ||
        b.invocation_guard_merge || entry.exec_narrowed || !true_constant(b, entry.exec) || !code ||
        !dwords || (b.wave_size != 32 && b.wave_size != 64))
        return proven;
    const auto local = entry.vreg.find(0);
    if (local == entry.vreg.end() || local->second != b.localid_comp[0]) return proven;
    if (!clean_entry_vector(entry, 0)) return proven;
    std::vector<Rdna2Inst> original;
    rdna2_walk(code, dwords, original);
    if (original != ins || ins.empty() || ins.front().pc ||
        std::any_of(ins.begin(), ins.end(), opaque_control))
        return proven;
    const auto safe = safe_execz_branches(original);
    std::map<uint32_t, size_t> index;
    for (size_t i = 0; i < ins.size(); ++i) index.emplace(ins[i].pc, i);
    std::vector<std::vector<size_t>> next(ins.size()), previous(ins.size());
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& in = ins[i];
        if (in.is_end || (in.fmt == Rdna2Format::SOPP && in.opcode == 0x12u)) continue;
        const bool branch = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        // Reuse the existing guest side-effect proof, never a new guessed EXECZ exception.
        const bool linear = branch && in.opcode == 0x08u && safe.contains(in.pc);
        if (branch && !linear) {
            const auto target = index.find(branch_target(in));
            if (target == index.end()) return {};
            next[i].push_back(target->second);
        }
        if ((!branch || linear || in.opcode != kSoppOpcodeBranch) && i + 1 < ins.size())
            next[i].push_back(i + 1);
        for (size_t j : next[i]) previous[j].push_back(i);
    }
    for (size_t site = 0; site < ins.size(); ++site) {
        if (!compute_dpp_quad_or(ins[site])) continue;
        std::vector<bool> cone(ins.size(), false);
        std::vector<size_t> pending{site};
        cone[site] = true;
        while (!pending.empty()) {
            const auto i = pending.back();
            pending.pop_back();
            for (size_t j : previous[i])
                if (!cone[j]) {
                    cone[j] = true;
                    pending.push_back(j);
                }
        }
        if (!cone[0]) continue;
        std::vector<uint32_t> degree(ins.size(), 0);
        std::queue<size_t> ready;
        size_t cone_size = 0;
        for (size_t i = 0; i < ins.size(); ++i)
            if (cone[i]) {
                ++cone_size;
                for (size_t j : previous[i]) degree[i] += cone[j];
                if (!degree[i]) ready.push(i);
            }
        std::vector<size_t> order;
        while (!ready.empty()) {
            const size_t i = ready.front();
            ready.pop();
            order.push_back(i);
            for (size_t j : next[i])
                if (cone[j] && !--degree[j]) ready.push(j);
        }
        // A static producer PC cannot authenticate a second dynamic loop generation.
        if (order.size() != cone_size || order.front() != 0) continue;
        std::vector<std::optional<Facts>> outgoing(ins.size());
        uint32_t generations = 0;
        bool closed = true;
        for (size_t i : order) {
            Facts facts;
            if (i == 0) {
                facts.vector[0] = full();
                for (int component = 1; component < 3; ++component) {
                    const auto vgpr = entry.vreg.find(component);
                    if (vgpr != entry.vreg.end() && vgpr->second == b.localid_comp[component] &&
                        clean_entry_vector(entry, component))
                        facts.vector[static_cast<size_t>(component)] = full();
                }
            } else {
                bool first = true;
                for (size_t j : previous[i])
                    if (cone[j] && outgoing[j]) {
                        if (first) {
                            facts = *outgoing[j];
                            first = false;
                        } else {
                            if (facts.exec != outgoing[j]->exec) {
                                closed = false;
                                break;
                            }
                            for (size_t r = 0; r < facts.vector.size(); ++r)
                                facts.vector[r] &= outgoing[j]->vector[r];
                        }
                    }
                if (first) {
                    closed = false;
                    break;
                }
            }
            if (!closed) break;
            const auto& in = ins[i];
            if (i == site) {
                if (source(facts, in, in.src[0]).all() && source(facts, in, in.src[1]).all() &&
                    (facts.exec.all() || facts.vector[static_cast<size_t>(in.dst.value)].all()))
                    proven.insert(in.pc);
                break;
            }
            const Facts before = facts;
            Coverage value;
            if (pure_word(in) || compute_dpp_quad_or(in)) {
                value = full();
                for (uint32_t k = 0; k < in.n_src; ++k) value &= source(before, in, in.src[k]);
                if (compute_dpp_quad_or(in) && !value.all()) value.reset();
            }
            for_each_possible_vector_write(in, 255, [&](int r) {
                if (r < 0 || r >= 256) return;
                // WRITELANE ignores EXEC and can replace an unknown selected lane. This proof
                // has no scalar payload/selector authority, so retire the whole possible word.
                if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361u) {
                    facts.vector[static_cast<size_t>(r)].reset();
                    return;
                }
                auto written = value;
                if (in.has_sdwa && in.sdwa_dst_sel != 6)
                    written &= before.vector[static_cast<size_t>(r)];
                facts.vector[static_cast<size_t>(r)] =
                    (before.vector[static_cast<size_t>(r)] & ~before.exec) |
                    (written & before.exec);
            });
            if (rdna2_instruction_may_change_exec(in)) {
                if (const auto exact = exec_replacement(before, in))
                    facts.exec = *exact;
                else {
                    if (generations == 8) {
                        closed = false;
                        break;
                    }
                    facts.exec = atom(generations++);
                }
            }
            outgoing[i] = facts;
        }
    }
    return proven;
}
}   // namespace prosper::gpu
