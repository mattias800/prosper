// See rdna2_mask_half_alias.hpp. Moved out of emit_cfg_state_machine.
#include "gpu/recompiler/rdna2_mask_half_alias.hpp"

#include "gpu/recompiler/rdna2_barrier_mask_halves.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace prosper::gpu {

void analyze_wave64_mask_half_aliases(
    SpirvCompute& b, const std::vector<Rdna2Inst>& ins, const std::vector<uint32_t>& starts,
    const std::vector<std::vector<uint32_t>>& successors,
    const VectorWriteVisitor& for_each_possible_vector_write,
    std::vector<std::map<int, uint32_t>>& wave64_mask_half_sreg_in,
    std::vector<bool>& wave64_mask_half_reachable, const NativeBarrierHalfProof* barrier_halves,
    std::map<std::pair<int, int>, uint32_t>* terminal_slots, uint32_t terminal_pc) {
    if (terminal_slots) terminal_slots->clear();
    if (b.is_compute && b.wave_size == 64 && b.native_subgroup_size == 64 && !starts.empty()) {
        struct Half {
            uint32_t word;
            bool imported;
            bool operator==(const Half&) const = default;
        };
        struct MaskHalfState {
            std::map<int, Half> sreg;
            std::map<std::pair<int, int>, Half> slot;
            bool operator==(const MaskHalfState&) const = default;
        };
        auto special_half = [](const Operand& source) -> int {
            if (source.kind != OperandKind::Special) return -1;
            // EXEC is always a live mask in RegState. VCC_LO/HI may instead be scalar scratch, and
            // their physical encodings do not carry a runtime domain tag; treating those words as
            // masks here can turn a dispatcher placeholder into a ballot of false. Admit VCC only
            // after a future proof is explicitly tied to the Wave64 mask-domain MUST analysis.
            if (source.value == 126) return 0;
            if (source.value == 127) return 1;
            return -1;
        };
        auto meet = [](MaskHalfState& dst, const MaskHalfState& incoming) {
            for (auto it = dst.sreg.begin(); it != dst.sreg.end();) {
                const auto other = incoming.sreg.find(it->first);
                if (other == incoming.sreg.end() || other->second.word != it->second.word)
                    it = dst.sreg.erase(it);
                else {
                    it->second.imported |= other->second.imported;
                    ++it;
                }
            }
            for (auto it = dst.slot.begin(); it != dst.slot.end();) {
                const auto other = incoming.slot.find(it->first);
                if (other == incoming.slot.end() || other->second.word != it->second.word)
                    it = dst.slot.erase(it);
                else {
                    it->second.imported |= other->second.imported;
                    ++it;
                }
            }
        };
        auto transfer = [&](MaskHalfState& state, const Rdna2Inst& in, bool record) {
            // Only the unchanged branch-free original prefix proves that a new post-barrier
            // ballot sees every physical lane. Forget imported facts before any other site,
            // including facts relayed through SGPRs/re-spills. Local preexisting rules remain.
            if (barrier_halves && !barrier_halves->participant_pcs().contains(in.pc)) {
                std::erase_if(state.sreg, [](const auto& fact) { return fact.second.imported; });
                std::erase_if(state.slot, [](const auto& fact) { return fact.second.imported; });
            }
            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360) {
                for_each_scalar_write(
                    in,
                    [&](int base, uint32_t width) {
                        for (uint32_t word = 0; word < width; ++word)
                            state.sreg.erase(base + static_cast<int>(word));
                    },
                    /*wave32_one_word_masks*/ false);
                if (in.src[1].kind == OperandKind::InlineInt && in.src[1].value >= 0 &&
                    in.src[1].value <= 63) {
                    const std::pair<int, int> key{in.src[0].value, in.src[1].value};
                    const auto half = state.slot.find(key);
                    if (half != state.slot.end()) {
                        state.sreg[in.dst.value] = half->second;
                        if (record) b.wave64_mask_readlane_half_for_pc[in.pc] = half->second.word;
                    }
                }
                return;
            }

            for_each_scalar_write(
                in,
                [&](int base, uint32_t width) {
                    for (uint32_t word = 0; word < width; ++word)
                        state.sreg.erase(base + static_cast<int>(word));
                },
                /*wave32_one_word_masks*/ false);

            if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361) {
                // A dynamic selector may overwrite any lane and therefore kills every known slot
                // in this VGPR. A constant selector updates only its named slot.
                if (in.src[1].kind != OperandKind::InlineInt || in.src[1].value < 0 ||
                    in.src[1].value > 63) {
                    for (auto it = state.slot.begin(); it != state.slot.end();) {
                        if (it->first.first == in.dst.value)
                            it = state.slot.erase(it);
                        else
                            ++it;
                    }
                    return;
                }
                const std::pair<int, int> key{in.dst.value, in.src[1].value};
                int half = special_half(in.src[0]);
                bool imported = false;
                if (half < 0 && in.src[0].kind == OperandKind::SGPR) {
                    const auto source = state.sreg.find(in.src[0].value);
                    if (source != state.sreg.end()) {
                        half = static_cast<int>(source->second.word);
                        imported = source->second.imported;
                        if (record) b.wave64_mask_writelane_alias_pcs.insert(in.pc);
                    }
                }
                if (half < 0)
                    state.slot.erase(key);
                else
                    state.slot[key] = Half{static_cast<uint32_t>(half), imported};
                return;
            }

            for_each_possible_vector_write(in, [&](int vgpr) {
                for (auto it = state.slot.begin(); it != state.slot.end();) {
                    if (it->first.first == vgpr)
                        it = state.slot.erase(it);
                    else
                        ++it;
                }
            });
        };

        std::vector<MaskHalfState> half_in(starts.size());
        if (barrier_halves)
            for (const auto& [slot, half] : barrier_halves->slots())
                half_in.front().slot.emplace(slot, Half{half, true});
        std::vector<bool> half_reachable(starts.size(), false);
        half_reachable.front() = true;
        std::vector<uint32_t> pending{0};
        while (!pending.empty()) {
            const uint32_t block = pending.back();
            pending.pop_back();
            MaskHalfState state = half_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins)
                if (in.pc >= lo && in.pc < hi && !in.is_end) transfer(state, in, /*record*/ false);
            for (uint32_t successor : successors[block]) {
                if (!half_reachable[successor]) {
                    half_reachable[successor] = true;
                    half_in[successor] = state;
                    pending.push_back(successor);
                    continue;
                }
                MaskHalfState joined = half_in[successor];
                meet(joined, state);
                if (!(joined == half_in[successor])) {
                    half_in[successor] = std::move(joined);
                    pending.push_back(successor);
                }
            }
        }
        for (uint32_t block = 0; block < starts.size(); ++block) {
            if (!half_reachable[block]) continue;
            wave64_mask_half_reachable[block] = true;
            for (const auto& [reg, half] : half_in[block].sreg)
                wave64_mask_half_sreg_in[block].emplace(reg, half.word);
            if (terminal_slots && starts[block] == terminal_pc)
                for (const auto& [slot, half] : half_in[block].slot)
                    terminal_slots->emplace(slot, half.word);
            MaskHalfState state = half_in[block];
            const uint32_t lo = starts[block];
            const uint32_t hi = block + 1 < starts.size() ? starts[block + 1] : UINT32_MAX;
            for (const auto& in : ins)
                if (in.pc >= lo && in.pc < hi && !in.is_end) transfer(state, in, /*record*/ true);
        }
    }
}

}   // namespace prosper::gpu
