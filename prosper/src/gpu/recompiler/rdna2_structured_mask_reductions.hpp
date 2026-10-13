#pragma once

// Structured Wave64 mask-reduction inventory; promoted mechanically from the CFG emitter.
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace prosper::gpu {

inline std::unordered_set<uint32_t>
proven_structured_wave64_mask_reduction_pcs(const std::vector<Rdna2Inst>& ins) {
    size_t count = 0;
    while (count < ins.size() && !ins[count].is_end) ++count;
    if (!count) return {};

    std::unordered_map<uint32_t, size_t> index_for_pc;
    for (size_t i = 0; i < count; ++i) index_for_pc.emplace(ins[i].pc, i);

    std::vector<std::set<int>> incoming(count);
    std::vector<bool> reachable(count, false);
    std::vector<size_t> pending{0};
    reachable[0] = true;
    while (!pending.empty()) {
        const size_t index = pending.back();
        pending.pop_back();
        std::set<int> saved = incoming[index];
        const Rdna2Inst& in = ins[index];

        for_each_scalar_write(
            in,
            [&](int base, uint32_t width) {
                for (auto it = saved.begin(); it != saved.end();) {
                    const int root = *it;
                    if (base < root + 2 && root < base + static_cast<int>(width))
                        it = saved.erase(it);
                    else
                        ++it;
                }
            },
            /*wave32_one_word_masks=*/false);
        if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
            in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 105 &&
            in.src[0].kind == OperandKind::Special && in.src[0].value == 106 && saved.contains(106))
            saved.insert(in.dst.value);
        // Every SAVEEXEC variant writes OLD_EXEC to SDST before replacing architectural EXEC with
        // its logical result. OLD_EXEC is always one complete Wave64 mask, so an ordinary pair or
        // VCC destination becomes an exact mask producer without requiring source provenance. Do
        // not seed an EXEC destination: its architectural state after the instruction is the new
        // logical result, not the saved value.
        const bool saves_old_exec =
            in.fmt == Rdna2Format::SOP1 &&
            ((in.opcode >= kSop1OpcodeAndSaveexecB64 && in.opcode <= kSop1OpcodeXnorSaveexecB64) ||
             in.opcode == kSop1OpcodeAndn1SaveexecB64 || in.opcode == kSop1OpcodeOrn1SaveexecB64);
        if (saves_old_exec && in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 &&
            in.dst.value <= 106)
            saved.insert(in.dst.value);
        // A non-CMPX vector compare writes a fresh architectural VCC predicate. This deliberately
        // stays narrower than the dispatcher's general mask transfer: the structured override only
        // needs to certify the captured VOPC -> S_MOV_B64 -> reduction chain, and an ordinary scalar
        // VCC write above kills the fact before it can seed a saved pair.
        if (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) {
            const int mask_destination =
                in.dst.kind == OperandKind::SGPR && in.dst.value <= 105 ? in.dst.value : 106;
            saved.insert(mask_destination);
        }

        std::vector<size_t> successors;
        auto add_successor = [&](uint32_t pc) {
            const auto found = index_for_pc.find(pc);
            if (found != index_for_pc.end() &&
                std::find(successors.begin(), successors.end(), found->second) == successors.end())
                successors.push_back(found->second);
        };
        const bool indirect_pc = in.fmt == Rdna2Format::SOP1 && in.opcode >= kSop1OpcodeSetpcB64 &&
                                 in.opcode <= kSop1OpcodeRfeB64;
        if (!indirect_pc && in.fmt == Rdna2Format::SOPP &&
            sopp_opcode_is_direct_branch(in.opcode)) {
            add_successor(branch_target(in));
            if (in.opcode != kSoppOpcodeBranch && index + 1 < count)
                successors.push_back(index + 1);
        } else if (!indirect_pc && in.fmt == Rdna2Format::SOPP && in.opcode == 0x12u) {
            // S_TRAP transfers control outside the decoded shader stream.
        } else if (!indirect_pc && index + 1 < count) {
            successors.push_back(index + 1);
        }

        for (size_t successor : successors) {
            if (!reachable[successor]) {
                reachable[successor] = true;
                incoming[successor] = saved;
                pending.push_back(successor);
                continue;
            }
            std::set<int> joined;
            std::set_intersection(incoming[successor].begin(), incoming[successor].end(),
                                  saved.begin(), saved.end(), std::inserter(joined, joined.end()));
            if (joined != incoming[successor]) {
                incoming[successor] = std::move(joined);
                pending.push_back(successor);
            }
        }
    }

    std::unordered_set<uint32_t> proven;
    for (size_t i = 0; i < count; ++i) {
        const Rdna2Inst& in = ins[i];
        if (!reachable[i] || in.fmt != Rdna2Format::SOP1 ||
            (in.opcode != kSop1OpcodeBcnt1I32B64 && in.opcode != kSop1OpcodeFf1I32B64) ||
            in.src[0].kind != OperandKind::SGPR)
            continue;
        if (incoming[i].contains(in.src[0].value)) proven.insert(in.pc);
    }
    return proven;
}

}   // namespace prosper::gpu
