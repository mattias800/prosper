#include "gpu/recompiler/rdna2_loop_exec_data.hpp"
// A code-only DATA bound. This supplies neither route permission nor physical mask words.
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include "gpu/recompiler/rdna2_cfg_branch_helpers.hpp"
#include "gpu/recompiler/rdna2_cfg_registers.hpp"
#include "gpu/recompiler/rdna2_cfg_wave_uniform.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_instruction_effects.hpp"
#include "gpu/recompiler/rdna2_uniform_counter_data.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_vector_word_marks.hpp"
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <vector>
namespace prosper::gpu {
namespace {
using BoundMasks = std::bitset<128>;
constexpr std::size_t no_successor = std::numeric_limits<std::size_t>::max();
bool pair_root(int reg) {
    return (reg >= 0 && reg <= 104 && (reg & 1) == 0) || reg == 106 || reg == 126;
}
bool operand_within(const Operand& operand, const Rdna2Inst& in, const BoundMasks& masks) {
    if ((operand.kind == OperandKind::SGPR || operand.kind == OperandKind::Special) &&
        pair_root(operand.value))
        return masks.test(operand.value);
    return (operand.kind == OperandKind::InlineInt && operand.value == 0) ||
           (operand.kind == OperandKind::Literal && in.has_literal && in.literal == 0);
}
bool transfer_bound(const Rdna2Inst& in, const BoundMasks& before, BoundMasks& after) {
    after = before;
    bool valid = true, touches_exec = rdna2_instruction_may_change_exec(in);
    const auto kill = [&](int base, uint32_t width) {
        if (base < 0 || base > 255 || width > 128 || width > 256u - static_cast<unsigned>(base)) {
            valid = false;
            return;
        }
        touches_exec |= base <= 127 && base + static_cast<int>(width) > 126;
        for (uint32_t k = 0; k < width; ++k)
            for (int root : {base + static_cast<int>(k) - 1, base + static_cast<int>(k)})
                if (root >= 0 && root < 128) after.reset(root);
    };
    for_each_scalar_write(in, kill);
    const auto& effects = rdna2_instruction_effects(in.fmt, in.opcode);
    if (effects.sdst_dwords) kill(in.dst.value, effects.sdst_dwords);
    // The explicit writer inventory omits VOP2's implicit carry destination.
    if (in.fmt == Rdna2Format::VOP2 && !cannot_write_vcc(in)) kill(106, 2);
    if (!valid) return false;
    const auto set_pair = [&](bool value) {
        if ((in.dst.kind == OperandKind::SGPR || in.dst.kind == OperandKind::Special) &&
            pair_root(in.dst.value))
            after.set(in.dst.value, value);
    };
    const bool exec = before.test(126);
    if (in.fmt == Rdna2Format::VOPC) {
        if (vopc_is_cmpx(in.opcode))
            after.set(126, exec);
        else
            set_pair(exec);
        return !touches_exec || vopc_is_cmpx(in.opcode);
    }
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeAndSaveexecB64) {
        if (in.dst.value < 0 || in.dst.value > 104 || !pair_root(in.dst.value)) return false;
        set_pair(exec);
        after.set(126, exec || operand_within(in.src[0], in, before));
        return true;
    }
    bool handled = false, bound = false;
    if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64) {
        handled = true;
        bound = operand_within(in.src[0], in, before);
    } else if (in.fmt == Rdna2Format::SOP2 && (in.opcode == 0x0fu || in.opcode == 0x15u)) {
        handled = true;
        bound = operand_within(in.src[0], in, before) ||
                (in.opcode == 0x0fu && operand_within(in.src[1], in, before));
    }
    if (handled) set_pair(bound);
    return !touches_exec || (handled && in.dst.value == 126);
}
struct BoundGraph {
    std::vector<const Rdna2Inst*> body;
    std::map<uint32_t, std::size_t> indices;
    std::vector<std::array<std::size_t, 2>> successors;
};
bool make_bound_graph(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const DivLoop& loop,
                      BoundGraph& graph) {
    if (loop.header_pc >= loop.backedge_pc) return false;
    uint32_t next = loop.header_pc;
    for (const auto& in : ins) {
        const auto& effects = rdna2_instruction_effects(in.fmt, in.opcode);
        const auto flow = effects.control;
        const bool scalar = in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                            in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::SOPK ||
                            in.fmt == Rdna2Format::SOPP || in.fmt == Rdna2Format::SMEM;
        const bool valu = in.fmt == Rdna2Format::VOP1 || in.fmt == Rdna2Format::VOP2 ||
                          in.fmt == Rdna2Format::VOP3 || in.fmt == Rdna2Format::VOP3P ||
                          in.fmt == Rdna2Format::VOPC;
        if ((scalar && !effects.known()) || (valu && !effects.describes_sources())) return false;
        const bool direct = in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode);
        if (in.fmt == Rdna2Format::Unknown ||
            (!direct && flow != Rdna2ControlFlow::None && flow != Rdna2ControlFlow::End))
            return false;
        if (in.pc < loop.header_pc || in.pc > loop.backedge_pc) {
            if (in.fmt == Rdna2Format::SOPP && !direct && !sopp_is_noop(in) && !in.is_end &&
                in.opcode != kSoppOpcodeBarrier)
                return false;
            if (direct) {
                const auto target = branch_target(in);
                if (target >= loop.header_pc && target <= loop.backedge_pc) return false;
            }
            continue;
        }
        if (graph.body.size() >= 2048 || in.pc != next || !in.len_dwords ||
            in.synthetic_terminator || in.is_end || in.pc >= b.saved_mask_original_words.size() ||
            in.len_dwords > b.saved_mask_original_words.size() - in.pc ||
            rdna2_escapes_decoded_effects(in) ||
            (in.fmt == Rdna2Format::SOPP && !direct && !sopp_is_noop(in)))
            return false;
        auto decoded = rdna2_decode_one(b.saved_mask_original_words.data() + in.pc,
                                        b.saved_mask_original_words.size() - in.pc);
        decoded.pc = in.pc;
        if (decoded != in) return false;
        graph.indices.emplace(in.pc, graph.body.size());
        graph.body.push_back(&in);
        next += in.len_dwords;
    }
    if (graph.body.empty() || graph.body.back()->pc != loop.backedge_pc) return false;
    const auto& last = *graph.body.back();
    if (last.fmt != Rdna2Format::SOPP || !sopp_opcode_is_direct_branch(last.opcode) ||
        branch_target(last) != loop.header_pc)
        return false;
    for (const auto* in : graph.body) {
        std::array<std::size_t, 2> edges{no_successor, no_successor};
        unsigned count = 0;
        const auto edge = [&](uint32_t pc) {
            const auto found = graph.indices.find(pc);
            if (found != graph.indices.end())
                edges[count++] = found->second;
            else if (pc != loop.exit_pc)
                return false;
            return true;
        };
        if (in->fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in->opcode)) {
            if (!edge(branch_target(*in))) return false;
            if (in->opcode == kSoppOpcodeBranch) {
                graph.successors.push_back(edges);
                continue;
            }
        }
        if (!edge(in->pc + in->len_dwords)) return false;
        graph.successors.push_back(edges);
    }
    return true;
}
bool graph_keeps_bound(const BoundGraph& graph, BoundMasks entry) {
    const auto size = graph.body.size();
    entry.set(126);
    std::vector<BoundMasks> states(size), outputs(size);
    std::vector<bool> reached(size, false), queued(size, false);
    std::deque<std::size_t> pending{0};
    states[0] = entry;
    reached[0] = queued[0] = true;
    std::size_t visits = 0;
    while (!pending.empty()) {
        if (++visits > size * 130) return false;
        const auto index = pending.front();
        pending.pop_front();
        queued[index] = false;
        if (!transfer_bound(*graph.body[index], states[index], outputs[index])) return false;
        for (auto successor : graph.successors[index]) {
            if (successor == no_successor) continue;
            auto incoming = outputs[index];
            // Assume only the header's EXEC bound. Carried saved pairs still meet the real back edges.
            if (successor == 0) incoming.set(126);
            auto merged = reached[successor] ? (states[successor] & incoming) : incoming;
            if (reached[successor] && merged == states[successor]) continue;
            states[successor] = merged;
            reached[successor] = true;
            if (!queued[successor]) {
                queued[successor] = true;
                pending.push_back(successor);
            }
        }
    }
    bool back = false;
    for (std::size_t index = 0; index < size; ++index)
        if (reached[index])
            for (auto successor : graph.successors[index])
                if (successor == 0) {
                    back = true;
                    if (!outputs[index].test(126)) return false;
                }
    return back;
}
}   // namespace
bool native_loop_data_exec_bound(const SpirvCompute& b, const std::vector<Rdna2Inst>& ins,
                                 const DivLoop& loop, const RegState& entry) {
    if (b.saved_mask_original_words.size() > 65536 ||
        loop.exit_pc >= b.saved_mask_original_words.size() || entry.vector_wave_size != 64 ||
        !entry.exec || !entry.exec_narrowed || entry.vector_full_exec ||
        !native_exact_retry_authentic(b, b.saved_mask_original_words.data(),
                                      b.saved_mask_original_words.size()))
        return false;
    // EXEC exits need their own uniform-counter and scalar-live-out boundary; no new route.
    if (loop.condition != DivLoop::Condition::Scc &&
        !uniform_counter_loop_data_boundary(b, ins, loop, entry))
        return false;
    BoundGraph graph;
    if (!make_bound_graph(b, ins, loop, graph)) return false;
    BoundMasks initial;
    initial.set(126);
    for (const auto& [root, value] : entry.sreg_bool)
        if (pair_root(root) && !entry.sreg_bool_b32.contains(root) && value &&
            exec_within(entry, value, entry.exec))
            initial.set(root);
    if (entry.vcc && exec_within(entry, entry.vcc, entry.exec)) initial.set(106);
    return graph_keeps_bound(graph, initial);
}
}   // namespace prosper::gpu
