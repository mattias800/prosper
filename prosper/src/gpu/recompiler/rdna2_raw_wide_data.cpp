// Distinguish descriptor provenance from scalar data in typeless raw x4/x8 SMEM loads.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
// A raw wide load backed by a dispatch-time CPU snapshot cannot observe an earlier shader write
// to that same guest allocation. Without an alias proof, keep potentially writing instructions
// on the ordinary unresolved route. Read-only operations remain eligible.
bool rdna2_may_write_guest_memory(const Rdna2Inst& in) {
    if (in.fmt == Rdna2Format::FLAT) return true;
    if (in.fmt == Rdna2Format::SMEM) return in.opcode >= 0x10u;
    if (in.fmt == Rdna2Format::MTBUF) return in.opcode > 0x03u;
    if (in.fmt == Rdna2Format::MUBUF)
        return !(in.opcode <= 0x03u ||
                 (in.opcode >= 0x08u && in.opcode <= 0x0fu));
    if (in.fmt == Rdna2Format::MIMG)
        return in.opcode != 0x00u && in.opcode != 0x0eu && in.opcode != 0x27u && in.opcode != 0x2fu;
    return false;
}

namespace {

struct RawWideState {
    size_t index;
    uint16_t live;
};

// One load's live words, tracked along every direct branch until overwrite or exit. An uncertain
// control edge or ordinary scalar read is a refusal to replace those words with zero. This does
// not certify descriptor provenance; resource resolution still decides whether a consumer works.
class RawWideLifetime {
public:
    RawWideLifetime(const std::vector<Rdna2Inst>& instructions,
                    const std::unordered_map<uint32_t, size_t>& pc_indices,
                    size_t load_index, uint32_t word_count)
        : ins(instructions), by_pc(pc_indices), start(load_index),
          first(instructions[load_index].dst.value), words(word_count) {}

    bool requires_backing() const {
        if (first + static_cast<int>(words) > 106) return true;
        std::vector<RawWideState> pending;
        std::unordered_set<uint64_t> visited;
        if (start + 1 < ins.size())
            pending.push_back({start + 1, static_cast<uint16_t>((1u << words) - 1u)});
        while (!pending.empty()) {
            const RawWideState state = pending.back();
            pending.pop_back();
            if (!state.live || state.index >= ins.size() || state.index == start) continue;
            const uint64_t key = (static_cast<uint64_t>(state.index) << 8u) | state.live;
            if (!visited.insert(key).second) continue;
            const Rdna2Inst& in = ins[state.index];
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords ||
                (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u))
                return true;
            if (in.is_end) continue;
            if (reads_data(in, state.live)) return true;
            const uint16_t live = kill_written_words(in, state.live);
            if (live && !enqueue_successors(in, state.index, live, pending)) return true;
        }
        return false;
    }

    bool has_numeric_reader_or_uncertain_path() const {
        // Scalar data is MAY provenance (OR at joins). A separate MUST fact identifies exact
        // fresh mask roots used by the emitter's Bool consumers (AND at joins). A compare can
        // replace that Bool without proving that both physical scalar words were overwritten.
        // Keep the possible high-word numeric dependency alive for every ordinary data reader.
        struct State {
            size_t index;
            std::bitset<128> regs;
            bool scc;
            std::bitset<128> masks;
        };
        if (start + 1 >= ins.size()) return false;
        State initial{start + 1, {}, false, {}};
        initial.masks = saved_exec_masks_at_load();
        initial.masks.set(126); // entry EXEC cannot depend on this subsequent, unreplayed load
        for (uint32_t word = 0; word < words; ++word)
            initial.regs.set(static_cast<size_t>(first + static_cast<int>(word)));
        std::vector<State> pending{initial};
        std::vector<std::array<std::bitset<128>, 2>> seen(ins.size());
        std::vector<std::array<std::bitset<128>, 2>> seen_masks(ins.size());
        std::vector<std::array<bool, 2>> seen_any(ins.size());
        const auto mask_register = [](const Operand& operand) {
            if (operand.kind != OperandKind::SGPR && operand.kind != OperandKind::Special)
                return -1;
            return operand.value >= 0 &&
                (operand.value <= 105 || operand.value == 106 || operand.value == 126)
                ? operand.value : -1;
        };
        const auto supported_compare = [](const Rdna2Inst& in) {
            // Match emit_alu's exact effective compare families. CMPX publishes EXEC only;
            // its SDST-shaped decoder field cannot manufacture a saved-mask lifetime.
            const uint32_t op = vopc_is_cmpx(in.opcode) ? in.opcode - 0x10u : in.opcode;
            return in.fmt == Rdna2Format::VOPC &&
                (op <= 0x0fu || (op >= 0x81u && op <= 0x86u) || op == 0x88u ||
                 (op >= 0x89u && op <= 0x8eu) || (op >= 0xa1u && op <= 0xa6u) ||
                 (op >= 0xa9u && op <= 0xaeu) || (op >= 0xc1u && op <= 0xc6u) ||
                 (op >= 0xc9u && op <= 0xceu) || (op >= 0xe1u && op <= 0xe6u));
        };
        auto writes_scc = [](const Rdna2Inst& in) {
            if (in.fmt == Rdna2Format::SOPC) return true;
            if (in.fmt == Rdna2Format::SOP1)
                return !sop1_opcode_leaves_scc_unmodified(in.opcode);
            if (in.fmt == Rdna2Format::SOPK)
                return (in.opcode >= kSopkOpcodeCmpkFirst &&
                        in.opcode <= kSopkOpcodeCmpkLast) ||
                       in.opcode == kSopkOpcodeAddkI32;
            if (in.fmt == Rdna2Format::SOP2)
                return in.opcode != 0x0au && in.opcode != 0x0bu &&
                       in.opcode != kSop2OpcodeBfmB32 &&
                       in.opcode != kSop2OpcodeBfmB64 &&
                       in.opcode != 0x26u &&
                       in.opcode != kSop2OpcodePackLlB32B16 &&
                       in.opcode != 0x35u && in.opcode != 0x36u;
            return false;
        };
        size_t processed = 0;
        while (!pending.empty()) {
            State state = std::move(pending.back());
            pending.pop_back();
            if ((!state.regs.any() && !state.scc) || state.index >= ins.size()) continue;
            if (state.index == start) return true; // a replayed load needs a new byte observation
            const size_t slot = state.scc ? 1u : 0u;
            if (seen_any[state.index][slot] &&
                (state.regs & ~seen[state.index][slot]).none() &&
                (seen_masks[state.index][slot] & ~state.masks).none()) continue;
            if (seen_any[state.index][slot])
                seen_masks[state.index][slot] &= state.masks;
            else
                seen_masks[state.index][slot] = state.masks;
            seen_any[state.index][slot] = true;
            seen[state.index][slot] |= state.regs;
            state.regs = seen[state.index][slot];
            state.masks = seen_masks[state.index][slot];
            if (++processed > 32768) return true;
            const Rdna2Inst& in = ins[state.index];
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords) return true;
            if (in.is_end) continue;
            if ((in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u &&
                 in.opcode <= 0x22u) ||
                (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCallB64) ||
                (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x28u &&
                 in.opcode <= 0x2au)) return true; // indirect control/relative SGPR write

            if (state.scc && in.fmt == Rdna2Format::SOPP &&
                (in.opcode == 0x04u || in.opcode == 0x05u)) return true;
            if (in.fmt == Rdna2Format::SOPP &&
                (in.opcode == 0x08u || in.opcode == 0x09u) &&
                !state.masks.test(126)) return true;
            const auto independent_mask = [&](const Operand& operand) {
                if (operand.kind == OperandKind::InlineInt)
                    return operand.value == 0 || operand.value == -1;
                const int root = mask_register(operand);
                return root >= 0 && state.masks.test(static_cast<size_t>(root));
            };
            const bool mask_logical = in.fmt == Rdna2Format::SOP2 &&
                (in.opcode == 0x0fu || in.opcode == 0x11u || in.opcode == 0x13u ||
                 in.opcode == 0x15u || in.opcode == 0x17u || in.opcode == 0x19u ||
                 in.opcode == 0x1bu || in.opcode == 0x1du);
            const bool mask_move = in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeMovB64;
            const bool mask_saveexec = in.fmt == Rdna2Format::SOP1 &&
                in.opcode >= kSop1OpcodeAndSaveexecB64 &&
                in.opcode <= kSop1OpcodeXnorSaveexecB64;
            const bool independent_transfer = mask_register(in.dst) >= 0 &&
                ((mask_logical && independent_mask(in.src[0]) &&
                                  independent_mask(in.src[1])) ||
                 (mask_move && independent_mask(in.src[0])) ||
                 (mask_saveexec && independent_mask(in.src[0]) && state.masks.test(126)));
            const bool implicit_scc_read = state.scc &&
                ((in.fmt == Rdna2Format::SOP2 &&
                  (in.opcode == 0x04u || in.opcode == 0x05u ||
                   in.opcode == 0x0au || in.opcode == 0x0bu)) ||
                 (in.fmt == Rdna2Format::SOP1 &&
                  (in.opcode == kSop1OpcodeCmovB32 ||
                   in.opcode == kSop1OpcodeCmovB64)) ||
                 (in.fmt == Rdna2Format::SOPK &&
                  in.opcode == kSopkOpcodeCmovkI32));
            const uint32_t copy_words = in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeMovB32 ? 1u :
                in.fmt == Rdna2Format::SOP1 &&
                in.opcode == kSop1OpcodeMovB64 ? 2u : 0u;
            const bool plain_copy = copy_words && in.dst.kind == OperandKind::SGPR &&
                (in.src[0].kind == OperandKind::SGPR ||
                 in.src[0].kind == OperandKind::Special);
            bool derived_read = implicit_scc_read;
            if (!plain_copy) {
                const uint32_t implicit = scalar_implicit_destination_read_width(in);
                for (uint32_t k = 0; k < implicit; ++k)
                    if (in.dst.value >= 0 && in.dst.value + static_cast<int>(k) < 128 &&
                        state.regs.test(static_cast<size_t>(in.dst.value + k)))
                        derived_read = true;
                for (uint32_t source = 0; source < in.n_src; ++source) {
                    const Operand& operand = in.src[source];
                    if (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x101u &&
                        source == 2u && independent_mask(operand)) continue;
                    if (operand.kind == OperandKind::Special && operand.value == 253) {
                        derived_read |= state.scc;
                        continue;
                    }
                    if ((operand.kind != OperandKind::SGPR &&
                         !(operand.kind == OperandKind::Special &&
                           operand.value >= 106 && operand.value <= 127)) ||
                        descriptor_source(in, source)) continue;
                    for (uint32_t k = 0; k < source_width(in, source); ++k)
                        if (operand.value >= 0 &&
                            operand.value + static_cast<int>(k) < 128 &&
                            state.regs.test(static_cast<size_t>(operand.value + k)))
                            derived_read = true;
                }
            }
            const bool scalar_result = in.fmt == Rdna2Format::SOP1 ||
                in.fmt == Rdna2Format::SOP2 || in.fmt == Rdna2Format::SOPK;
            if (derived_read && !scalar_result && in.fmt != Rdna2Format::SOPC) return true;
            if (derived_read && scalar_result &&
                (in.dst.kind != OperandKind::SGPR ||
                 (in.fmt == Rdna2Format::SOPK &&
                  in.opcode == kSopkOpcodeSetregB32) ||
                 rdna2_instruction_may_change_exec(in)) &&
                !independent_transfer) return true;

            // Evaluate inputs before expiring overlapping roots. Both siblings are invalidated
            // conservatively: a saved Bool must never authorize a later physical data lifetime.
            const int compare_root = in.dst.value == 126 ? -1 : mask_register(in.dst);
            const bool fresh_compare = supported_compare(in) &&
                !vopc_is_cmpx(in.opcode) && compare_root >= 0 &&
                !derived_read && state.masks.test(126);
            // The emitter narrows EXEC with CMPX's explicit comparison. Independent inputs
            // preserve an already independent EXEC, without overwriting SDST/VCC or proving
            // anything about their numeric high words.
            const bool independent_cmpx = supported_compare(in) &&
                vopc_is_cmpx(in.opcode) && !derived_read && state.masks.test(126);
            const bool fresh_exec = independent_cmpx || (independent_transfer &&
                (mask_saveexec || mask_register(in.dst) == 126));
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k) {
                    const int reg = base + static_cast<int>(k);
                    if (reg >= 0 && reg < 128) state.masks.reset(static_cast<size_t>(reg));
                    if (reg > 0 && reg <= 128) state.masks.reset(static_cast<size_t>(reg - 1));
                }
            });
            if (in.fmt == Rdna2Format::VOPC && in.dst.value == 106 &&
                !vopc_is_cmpx(in.opcode)) state.masks.reset(106);
            if (rdna2_instruction_may_change_exec(in)) state.masks.reset(126);
            if (fresh_compare) state.masks.set(static_cast<size_t>(compare_root));
            if (independent_transfer) state.masks.set(static_cast<size_t>(mask_register(in.dst)));
            if (fresh_exec) state.masks.set(126);

            std::array<bool, 2> copied{};
            if (plain_copy)
                for (uint32_t k = 0; k < copy_words; ++k)
                    copied[k] = (k == 0 && state.scc &&
                                 in.src[0].kind == OperandKind::Special &&
                                 in.src[0].value == 253) ||
                        (in.src[0].value >= 0 &&
                         in.src[0].value + static_cast<int>(k) < 128 &&
                         state.regs.test(static_cast<size_t>(in.src[0].value + k)));
            std::bitset<128> produced;
            const bool conditional_write =
                (in.fmt == Rdna2Format::SOP1 &&
                 (in.opcode == kSop1OpcodeCmovB32 ||
                  in.opcode == kSop1OpcodeCmovB64)) ||
                (in.fmt == Rdna2Format::SOPK &&
                 in.opcode == kSopkOpcodeCmovkI32);
            const bool definite_scalar_write = !conditional_write &&
                (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
                 in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SMEM ||
                 (in.fmt == Rdna2Format::VOPC && !vopc_is_cmpx(in.opcode)) ||
                 (in.fmt == Rdna2Format::VOP3 && in.opcode == 0x360u));
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k) {
                    const int reg = base + static_cast<int>(k);
                    if (reg < 0 || reg >= 128) continue;
                    if (definite_scalar_write)
                        state.regs.reset(static_cast<size_t>(reg));
                    if (derived_read && scalar_result &&
                        (in.dst.kind == OperandKind::SGPR || independent_transfer) &&
                        base == in.dst.value)
                        produced.set(static_cast<size_t>(reg));
                }
            }, /*wave32_one_word_masks*/true);
            if (in.fmt == Rdna2Format::VOPC && in.dst.value == 106 &&
                !vopc_is_cmpx(in.opcode))
                state.regs.reset(106); // only the guaranteed VCC low word, no width assumption
            state.regs |= produced;
            if (plain_copy)
                for (uint32_t k = 0; k < copy_words; ++k)
                    if (copied[k] && in.dst.value >= 0 &&
                        in.dst.value + static_cast<int>(k) < 128)
                        state.regs.set(static_cast<size_t>(in.dst.value + k));
            if (writes_scc(in)) state.scc = derived_read;
            if (!state.regs.any() && !state.scc) continue;

            auto enqueue = [&](size_t next) {
                // MAY words grow and MUST mask roots shrink at joins, including backedges.
                // The finite worklist must converge within the cap above. Reexecuting this
                // load still requires a fresh observation, not its earlier descriptor proof.
                if (next == start) return false;
                pending.push_back({next, state.regs, state.scc, state.masks});
                return true;
            };
            if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target_pc = static_cast<int64_t>(in.pc) +
                    in.len_dwords + in.simm16;
                if (target_pc < 0 || target_pc > UINT32_MAX) return true;
                const auto target = by_pc.find(static_cast<uint32_t>(target_pc));
                if (target == by_pc.end()) {
                    if (target_pc <= ins.back().pc) return true;
                } else if (!enqueue(target->second)) return true;
                if (in.opcode == kSoppOpcodeBranch) continue;
            } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
                       in.opcode != 0x0au && in.opcode != 0x10u &&
                       in.opcode != 0x16u && in.opcode != 0x17u) {
                return true;
            }
            if (state.index + 1 < ins.size() && !enqueue(state.index + 1)) return true;
        }
        return false;
    }

    bool has_reader_after_bypassed_load(size_t entry) const {
        std::vector<RawWideState> pending{{entry,
            static_cast<uint16_t>((1u << words) - 1u)}};
        std::unordered_set<uint64_t> visited;
        while (!pending.empty()) {
            const RawWideState state = pending.back();
            pending.pop_back();
            if (!state.live || state.index >= ins.size()) continue;
            const uint64_t key = (static_cast<uint64_t>(state.index) << 8u) | state.live;
            if (!visited.insert(key).second) continue;
            if (visited.size() > 8192) return true;
            const Rdna2Inst& in = ins[state.index];
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords) return true;
            if (in.is_end) continue;
            const uint32_t implicit = scalar_implicit_destination_read_width(in);
            if (implicit && overlaps(in.dst.value, implicit, state.live)) return true;
            for (uint32_t source = 0; source < in.n_src; ++source) {
                const Operand& operand = in.src[source];
                if ((operand.kind == OperandKind::SGPR ||
                     (operand.kind == OperandKind::Special &&
                      operand.value >= 106 && operand.value <= 124)) &&
                    overlaps(operand.value, source_width(in, source), state.live))
                    return true;
            }
            const uint16_t live = kill_written_words(in, state.live);
            if (live && !enqueue_successors(in, state.index, live, pending)) return true;
        }
        return false;
    }

private:
    const std::vector<Rdna2Inst>& ins;
    const std::unordered_map<uint32_t, size_t>& by_pc;
    size_t start;
    int first;
    uint32_t words;

    std::bitset<128> saved_exec_masks_at_load() const {
        // Only actual mask saves, on every path reaching this load, may seed its Bool facts.
        // Their value predates this load; no scalar bits, wave width or zero are inferred.
        std::vector<std::bitset<128>> incoming(start + 1);
        std::vector<bool> reached(start + 1);
        reached[0] = true;
        for (size_t index = 0; index < start; ++index) {
            if (!reached[index]) continue;
            const auto& in = ins[index];
            if (in.fmt == Rdna2Format::Unknown || !in.len_dwords ||
                (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u) ||
                (in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x28u && in.opcode <= 0x2au) ||
                (in.fmt == Rdna2Format::SOPK && in.opcode == kSopkOpcodeCallB64)) return {};
            auto masks = incoming[index];
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k) {
                    const int reg = base + static_cast<int>(k);
                    if (reg >= 0 && reg < 128) masks.reset(static_cast<size_t>(reg));
                    if (reg > 0 && reg <= 128) masks.reset(static_cast<size_t>(reg - 1));
                }
            });
            if (in.fmt == Rdna2Format::SOP1 && in.opcode == kSop1OpcodeMovB64 &&
                in.dst.kind == OperandKind::SGPR && in.dst.value >= 0 && in.dst.value <= 104 &&
                in.src[0].kind == OperandKind::Special && in.src[0].value == 126)
                masks.set(static_cast<size_t>(in.dst.value));
            const auto enter = [&](size_t next) {
                if (next > start) return;
                if (reached[next]) incoming[next] &= masks;
                else incoming[next] = masks;
                reached[next] = true;
            };
            if (in.is_end) continue;
            if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target_pc = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
                if (target_pc <= in.pc || target_pc > UINT32_MAX) return {};
                const auto target = by_pc.find(static_cast<uint32_t>(target_pc));
                if (target == by_pc.end()) {
                    if (target_pc <= ins.back().pc) return {};
                } else {
                    if (target->second <= index) return {};
                    enter(target->second);
                }
                if (in.opcode == kSoppOpcodeBranch) continue;
            } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
                       in.opcode != 0x0au && in.opcode != 0x10u &&
                       in.opcode != 0x16u && in.opcode != 0x17u) return {};
            enter(index + 1);
        }
        auto masks = reached[start] ? incoming[start] : std::bitset<128>{};
        // The load itself overwrites data and invalidates an overlapping saved pair.
        for_each_scalar_write(ins[start], [&](int base, uint32_t width) {
            for (uint32_t k = 0; k < width; ++k) {
                const int reg = base + static_cast<int>(k);
                if (reg >= 0 && reg < 128) masks.reset(static_cast<size_t>(reg));
                if (reg > 0 && reg <= 128) masks.reset(static_cast<size_t>(reg - 1));
            }
        });
        return masks;
    }

    bool overlaps(int base, uint32_t width, uint16_t live) const {
        if (base < 0) return false;
        for (uint32_t word = 0; word < words; ++word) {
            const int reg = first + static_cast<int>(word);
            if ((live & (1u << word)) && base <= reg &&
                reg < base + static_cast<int>(width)) return true;
        }
        return false;
    }

    static uint32_t source_width(const Rdna2Inst& in, uint32_t source) {
        if (in.fmt == Rdna2Format::MIMG) {
            if (source == 1) return 8;
            if (source == 2) return 4;
        }
        if ((in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF) && source == 1)
            return 4;
        if (in.fmt == Rdna2Format::SMEM && source == 0)
            return in.opcode >= 8u ? 4u : 2u;
        if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
            in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::VOP3 ||
            in.fmt == Rdna2Format::VOPC) {
            const uint32_t width = scalar_alu_source_words(in, source);
            return width == UINT32_MAX ? 0u : width;
        }
        return 1;
    }

    static bool descriptor_source(const Rdna2Inst& in, uint32_t source) {
        if (in.fmt == Rdna2Format::MIMG) return source == 1 || source == 2;
        if (in.fmt == Rdna2Format::MUBUF || in.fmt == Rdna2Format::MTBUF)
            return source == 1;
        return in.fmt == Rdna2Format::SMEM && in.opcode >= 8u && source == 0;
    }

    bool reads_data(const Rdna2Inst& in, uint16_t live) const {
        const uint32_t implicit_width = scalar_implicit_destination_read_width(in);
        if (implicit_width && overlaps(in.dst.value, implicit_width, live)) return true;
        for (uint32_t source = 0; source < in.n_src; ++source) {
            const Operand& operand = in.src[source];
            if (operand.kind != OperandKind::SGPR &&
                !(operand.kind == OperandKind::Special &&
                  operand.value >= 106 && operand.value <= 124)) continue;
            if (overlaps(operand.value, source_width(in, source), live) &&
                !descriptor_source(in, source)) return true;
        }
        return false;
    }

    uint16_t kill_written_words(const Rdna2Inst& in, uint16_t live) const {
        for_each_scalar_write(in, [&](int base, uint32_t width) {
            if (base < 0) return;
            for (uint32_t word = 0; word < words; ++word)
                if (base <= first + static_cast<int>(word) &&
                    first + static_cast<int>(word) < base + static_cast<int>(width))
                    live &= static_cast<uint16_t>(~(1u << word));
        }, /*wave32_one_word_masks*/true);
        return live;
    }

    bool enqueue_successors(const Rdna2Inst& in, size_t index, uint16_t live,
                            std::vector<RawWideState>& pending) const {
        if (in.fmt == Rdna2Format::SOPP && sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target_pc = static_cast<int64_t>(in.pc) +
                static_cast<int64_t>(in.len_dwords) + in.simm16;
            if (target_pc < 0 || target_pc > UINT32_MAX) return false;
            const auto target = by_pc.find(static_cast<uint32_t>(target_pc));
            if (target == by_pc.end()) {
                // A branch beyond the decoded program ends this path. A missing internal target
                // is unproven and cannot retain placeholder authority.
                if (target_pc <= ins.back().pc) return false;
            } else {
                pending.push_back({target->second, live});
            }
            if (in.opcode == kSoppOpcodeBranch) return true;
        } else if (in.fmt == Rdna2Format::SOPP && !sopp_is_noop(in) &&
                   in.opcode != 0x0au && in.opcode != 0x10u &&
                   in.opcode != 0x16u && in.opcode != 0x17u) {
            // 0x10/0x16/0x17 have no register/control effect in the emitter. All other unknown
            // transfers might enter a reader outside this decoded path.
            return false;
        }
        if (index + 1 < ins.size()) pending.push_back({index + 1, live});
        return true;
    }
};

} // namespace

std::vector<uint32_t> rdna2_raw_wide_data_loads(const std::vector<Rdna2Inst>& ins) {
    std::vector<uint32_t> data_loads;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t index = 0; index < ins.size(); ++index)
        by_pc.emplace(ins[index].pc, index);
    const bool has_guest_write = std::any_of(ins.begin(), ins.end(),
                                             rdna2_may_write_guest_memory);
    for (size_t index = 0; index < ins.size(); ++index) {
        const Rdna2Inst& load = ins[index];
        if (load.fmt != Rdna2Format::SMEM ||
            (load.opcode != 0x2u && load.opcode != 0x3u) ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + (load.opcode == 0x2u ? 4 : 8) > 128)
            continue;
        const uint32_t words = load.opcode == 0x2u ? 4u : 8u;
        const RawWideLifetime lifetime(ins, by_pc, index, words);
        const bool register_offset = load.src[1].kind != OperandKind::Special ||
            load.src[1].value != 125;
        // Scalar arithmetic may assemble a descriptor even when SOFFSET is a register.
        // The derived-use proof must find an actual non-descriptor observer before forbidding
        // the established exact-fetch-PC descriptor route. Copies and descriptor patches alone
        // are provenance, not scalar data observed by the emitted module. A shader-side memory
        // write could invalidate the front half's dispatch-time descriptor snapshot, so keep
        // newly admitted register-offset patches fail-visible until an alias proof exists.
        if (lifetime.requires_backing() &&
            (lifetime.has_numeric_reader_or_uncertain_path() ||
             (register_offset && has_guest_write)))
            data_loads.push_back(load.pc);
    }
    return data_loads;
}

// A small, deliberately stricter subset of the above refusal population can use a current-byte
// buffer. The predicate above reports uncertainty as "needs backing"; it must never itself grant
// admission. Here the entire decoded program has only valid forward edges, and the raw pointer is
// an unchanged entry pair. A load then observes one dispatch-local upload on every visit.
static std::vector<uint32_t> proven_immediate_wide_data_loads(
        const std::vector<Rdna2Inst>& ins, bool owned_read_point) {
    std::vector<uint32_t> proven;
    if (ins.empty()) return proven;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i)
        if (ins[i].fmt == Rdna2Format::Unknown || !ins[i].len_dwords ||
            !by_pc.emplace(ins[i].pc, i).second) return proven;
    for (const Rdna2Inst& in : ins) {
        if ((in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20u && in.opcode <= 0x22u) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x16u)) return proven;
        if (in.fmt != Rdna2Format::SOPP || in.is_end) continue;
        if (sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = static_cast<int64_t>(in.pc) + in.len_dwords + in.simm16;
            if (target <= static_cast<int64_t>(in.pc) || target > UINT32_MAX ||
                !by_pc.contains(static_cast<uint32_t>(target))) return proven;
        } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier &&
                   // GS_ALLOC_REQ reads M0 to request parameter-cache space; it changes no
                   // scalar value or guest source bytes (RDNA2 ISA 12.5.1). The emitter already
                   // supports this NGG prelude. Other messages remain outside this proof.
                   !(in.opcode == 0x10u && in.simm16 == 9))
            return proven;
    }
    const auto needs_backing = rdna2_raw_wide_data_loads(ins);
    for (size_t i = 0; i < ins.size(); ++i) {
        const Rdna2Inst& load = ins[i];
        const uint32_t words = load.opcode == 0x2u ? 4u : 8u;
        if (load.fmt != Rdna2Format::SMEM ||
            (load.opcode != 0x2u && load.opcode != 0x3u) ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + static_cast<int>(words) > 106 ||
            load.src[0].kind != OperandKind::SGPR || load.src[0].value < 0 ||
            load.src[0].value + 1 > 105 ||
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            static_cast<int32_t>(load.literal) < 0 || (load.literal & 3u) ||
            !std::binary_search(needs_backing.begin(), needs_backing.end(), load.pc))
            continue;
        bool stable_entry_pointer = true;
        const RawWideLifetime lifetime(ins, by_pc, i, words);
        for (const Rdna2Inst& in : ins) {
            // Forward-only control permits a textual prefix scan: any earlier global/image
            // store might alias the raw source after the CPU upload was captured.
            if ((owned_read_point || in.pc < load.pc) && rdna2_may_write_guest_memory(in))
                stable_entry_pointer = false;
            // A predecessor may bypass the load only if every bypass path overwrites all
            // would-be loaded words before reading them. This admits branch-to-after-region
            // shapes without allowing a joined reader to see a fabricated definition.
            if (in.pc < load.pc && in.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(in.opcode)) {
                const int64_t target = static_cast<int64_t>(in.pc) +
                    in.len_dwords + in.simm16;
                if (target > static_cast<int64_t>(load.pc) &&
                    lifetime.has_reader_after_bypassed_load(
                        by_pc.at(static_cast<uint32_t>(target))))
                    stable_entry_pointer = false;
            }
            if (!owned_read_point || in.pc < load.pc)
                for_each_scalar_write(in, [&](int base, uint32_t width) {
                    if (base >= 0 && base <= load.src[0].value + 1 &&
                        base + static_cast<int>(width) > load.src[0].value)
                        stable_entry_pointer = false;
                });
            if (!stable_entry_pointer) break;
        }
        if (stable_entry_pointer && lifetime.has_numeric_reader_or_uncertain_path())
            proven.push_back(load.pc);
    }
    return proven;
}

std::vector<uint32_t> rdna2_proven_raw_immediate_wide_data_loads(
        const std::vector<Rdna2Inst>& ins) {
    return proven_immediate_wide_data_loads(ins, false);
}

std::vector<uint32_t> rdna2_proven_raw_register_wide_data_loads(
        const std::vector<Rdna2Inst>& ins, std::vector<uint32_t>* scalar_source_pcs) {
    if (scalar_source_pcs) scalar_source_pcs->clear();
    // Reuse the entry-pointer, forward-CFG, bypass-reader and guest-write proofs. Only the
    // candidate's addressing mode changes here; its loaded-word lifetime is unchanged.
    auto immediate = ins;
    for (auto& load : immediate)
        if (load.fmt == Rdna2Format::SMEM &&
            (load.opcode == 0x2u || load.opcode == 0x3u))
            load.src[1] = {OperandKind::Special, 125};
    const auto entry_proven = rdna2_proven_raw_immediate_wide_data_loads(immediate);
    const auto owned_parents = proven_immediate_wide_data_loads(ins, true);
    std::vector<uint32_t> proven;
    for (size_t i = 0; i < ins.size(); ++i) {
        const auto& load = ins[i];
        const auto& offset = load.src[1];
        const bool ordinary = offset.kind == OperandKind::SGPR &&
                              offset.value >= 0 && offset.value <= 105;
        const bool vcc = offset.kind == OperandKind::Special &&
                         (offset.value == 106 || offset.value == 107);
        if ((!ordinary && !vcc) ||
            !std::binary_search(entry_proven.begin(), entry_proven.end(), load.pc))
            continue;
        // The compact fold can omit implicit VALU mask writes. Authenticate the entire reaching
        // scalar-definition chain against the FULL stream before trusting a concrete fold value.
        std::bitset<128> needed;
        needed.set(static_cast<size_t>(offset.value));
        bool scalar_prefix = true;
        std::vector<uint32_t> sources;
        for (size_t j = i; j-- > 0;) {
            const auto& writer = ins[j];
            // This first subset does not select between incoming scalar definitions. A later
            // branch remains allowed by the shared lifetime proof, including a bypass overwrite.
            if (writer.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(writer.opcode)) {
                scalar_prefix = false;
                break;
            }
            const bool valu = writer.fmt == Rdna2Format::VOP1 ||
                              writer.fmt == Rdna2Format::VOP2 ||
                              writer.fmt == Rdna2Format::VOP3 ||
                              writer.fmt == Rdna2Format::VOPC;
            // Conservative for implicit writers too: no VALU may intervene in a live VCC
            // dependency. An earlier VALU before its scalar replacement has no such dependency.
            if (valu && (needed.test(106) || needed.test(107))) {
                scalar_prefix = false;
                break;
            }
            std::bitset<128> writes;
            for_each_scalar_write(writer, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k)
                    if (base + static_cast<int>(k) >= 0 &&
                        base + static_cast<int>(k) < 128)
                        writes.set(static_cast<size_t>(base + static_cast<int>(k)));
            });
            if ((writes & needed).none()) continue;
            // A complete immediate parent supplies the SAME observed words to the fold and
            // emitted stage. Later writes to its entry pointer cannot retarget this earlier read.
            // The parent proof still rejects every potentially aliasing guest write in the shader.
            if (std::binary_search(owned_parents.begin(), owned_parents.end(), writer.pc)) {
                needed &= ~writes;
                sources.push_back(writer.pc);
                // The exact parent already proves its entry pointer and reaching definition.
                // Once every selector word resolves, unrelated earlier control is irrelevant.
                // The child entry-pointer authority remains required by entry_proven above.
                if (needed.none()) break;
                continue;
            }
            // A bounded immediate raw x1 fetch is a latched scalar value. Its entry pointer
            // needs to survive only UNTIL this read, unlike the wide source pointer, whose
            // full-program lifetime is authenticated above. GTA overwrites this source pair
            // after the read while preserving the loaded scalar that supplies SOFFSET.
            const bool immediate_scalar_read = writer.fmt == Rdna2Format::SMEM &&
                writer.opcode == 0u && writer.dst.kind == OperandKind::SGPR &&
                writer.dst.value >= 0 && writer.dst.value <= 105 &&
                writer.src[0].kind == OperandKind::SGPR && writer.src[0].value >= 0 &&
                writer.src[0].value < 105 && writer.src[1].kind == OperandKind::Special &&
                writer.src[1].value == 125 && writer.literal == 0u;
            if (immediate_scalar_read) {
                bool entry_at_read = true;
                for (size_t prefix = 0; prefix < j && entry_at_read; ++prefix) {
                    const auto& before = ins[prefix];
                    if (rdna2_may_write_guest_memory(before) ||
                        (before.fmt == Rdna2Format::SOPP &&
                         sopp_opcode_is_direct_branch(before.opcode)) ||
                        (before.fmt == Rdna2Format::SOP1 && before.opcode >= 0x28u &&
                         before.opcode <= 0x2au)) {
                        entry_at_read = false;
                        break;
                    }
                    for_each_scalar_write(before, [&](int base, uint32_t width) {
                        if (base >= 0 && base <= writer.src[0].value + 1 &&
                            base + static_cast<int>(width) > writer.src[0].value)
                            entry_at_read = false;
                    });
                }
                if (!entry_at_read) {
                    scalar_prefix = false;
                    break;
                }
                // The existing exact-PC x1 resource supplies the emitted numeric value;
                // realization still requires mapped current bytes for this draw's pointer.
                needed &= ~writes;
                sources.push_back(writer.pc);
                continue;
            }
            // Scalar moves and B32 arithmetic are per-draw data, rather than masks or conditional
            // definitions. Concrete operands/opcode support are checked again by the live fold.
            const bool scalar_writer =
                (writer.fmt == Rdna2Format::SOP1 && writer.opcode == 0x03u) ||
                (writer.fmt == Rdna2Format::SOP2 && scalar_write_width(writer) == 1u &&
                 writer.opcode != 0x0au && writer.opcode != 0x04u && writer.opcode != 0x05u) ||
                (writer.fmt == Rdna2Format::SOPK && writer.opcode == 0x00u);
            if (!scalar_writer) {
                scalar_prefix = false;
                break;
            }
            needed &= ~writes;
            for (uint32_t k = 0; k < writer.n_src; ++k) {
                const auto& source = writer.src[k];
                if (source.kind == OperandKind::SGPR ||
                    (source.kind == OperandKind::Special &&
                     (source.value == 106 || source.value == 107))) {
                    if (source.value < 0 || source.value > 107) scalar_prefix = false;
                    else needed.set(static_cast<size_t>(source.value));
                } else if (source.kind == OperandKind::VGPR ||
                           (source.kind == OperandKind::Special && source.value != 125)) {
                    scalar_prefix = false;
                }
            }
            if (!scalar_prefix) break;
        }
        // Ordinary entry SGPRs are resolved from this draw. VCC cannot be an uninitialized entry
        // value: its complete scalar definition must have been found above.
        if (scalar_prefix && !needed.test(106) && !needed.test(107)) {
            proven.push_back(load.pc);
            if (scalar_source_pcs)
                scalar_source_pcs->insert(scalar_source_pcs->end(), sources.begin(), sources.end());
        }
    }
    if (scalar_source_pcs) {
        std::sort(scalar_source_pcs->begin(), scalar_source_pcs->end());
        scalar_source_pcs->erase(std::unique(scalar_source_pcs->begin(), scalar_source_pcs->end()),
                                scalar_source_pcs->end());
    }
    return proven;
}

std::vector<uint32_t> rdna2_owned_raw_wide_data_loads(const std::vector<Rdna2Inst>& ins) {
    const auto strict = rdna2_proven_raw_immediate_wide_data_loads(ins);
    const auto read_points = proven_immediate_wide_data_loads(ins, true);
    std::vector<uint32_t> owned;
    std::set_difference(read_points.begin(), read_points.end(), strict.begin(), strict.end(),
                        std::back_inserter(owned));
    std::vector<uint32_t> scalar_sources;
    rdna2_proven_raw_register_wide_data_loads(ins, &scalar_sources);
    for (uint32_t pc : scalar_sources)
        if (std::binary_search(read_points.begin(), read_points.end(), pc))
            owned.push_back(pc);
    std::sort(owned.begin(), owned.end());
    owned.erase(std::unique(owned.begin(), owned.end()), owned.end());
    return owned;
}

std::vector<uint32_t> rdna2_proven_raw_nested_wide_data_loads(
        const std::vector<Rdna2Inst>& ins) {
    std::vector<uint32_t> proven;
    if (ins.empty()) return proven;
    const auto parents = rdna2_proven_raw_immediate_wide_data_loads(ins);
    const auto needs_backing = rdna2_raw_wide_data_loads(ins);
    if (parents.empty() || needs_backing.empty()) return proven;
    std::unordered_map<uint32_t, size_t> by_pc;
    for (size_t i = 0; i < ins.size(); ++i) {
        if (ins[i].fmt == Rdna2Format::Unknown || !ins[i].len_dwords ||
            !by_pc.emplace(ins[i].pc, i).second) return {};
    }
    for (const Rdna2Inst& in : ins) {
        if (in.fmt != Rdna2Format::SOPP || in.is_end) continue;
        if (sopp_opcode_is_direct_branch(in.opcode)) {
            const int64_t target = static_cast<int64_t>(in.pc) +
                in.len_dwords + in.simm16;
            if (target <= static_cast<int64_t>(in.pc) || target > UINT32_MAX ||
                !by_pc.contains(static_cast<uint32_t>(target))) return {};
        } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier) {
            return {};
        }
    }
    for (size_t child_index = 0; child_index < ins.size(); ++child_index) {
        const Rdna2Inst& child = ins[child_index];
        const int child_width = child.opcode == 0x2u ? 4 : 8;
        if (child.fmt != Rdna2Format::SMEM ||
            (child.opcode != 0x2u && child.opcode != 0x3u) ||
            child.dst.kind != OperandKind::SGPR ||
            // Numeric snapshots write ordinary SGPRs, not the emitter's typed VCC mask.
            child.dst.value < 0 || child.dst.value > 106 - child_width ||
            child.src[0].kind != OperandKind::SGPR ||
            child.src[0].value < 0 || child.src[0].value + 1 >= 106 ||
            child.src[1].kind != OperandKind::Special || child.src[1].value != 125 ||
            static_cast<int32_t>(child.literal) < 0 || (child.literal & 3u) ||
            !std::binary_search(needs_backing.begin(), needs_backing.end(), child.pc))
            continue;
        size_t parent_index = ins.size();
        for (size_t i = 0; i < child_index; ++i) {
            const Rdna2Inst& candidate = ins[i];
            if (candidate.fmt == Rdna2Format::SMEM &&
                (candidate.opcode == 0x2u || candidate.opcode == 0x3u) &&
                candidate.dst.kind == OperandKind::SGPR &&
                candidate.dst.value == child.src[0].value &&
                std::binary_search(parents.begin(), parents.end(), candidate.pc))
                parent_index = i;
        }
        if (parent_index == ins.size()) continue;
        bool stable = true;
        const RawWideLifetime child_lifetime(ins, by_pc, child_index,
                                             child.opcode == 0x2u ? 4u : 8u);
        for (size_t i = 0; i < child_index && stable; ++i) {
            const Rdna2Inst& preceding = ins[i];
            if (i > parent_index)
                for_each_scalar_write(preceding, [&](int base, uint32_t width) {
                    if (base >= 0 && base < child.src[0].value + 2 &&
                        base + static_cast<int>(width) > child.src[0].value)
                        stable = false;
                });
            if (i < parent_index && preceding.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(preceding.opcode)) {
                const int64_t target = static_cast<int64_t>(preceding.pc) +
                    preceding.len_dwords + preceding.simm16;
                if (target > static_cast<int64_t>(ins[parent_index].pc) &&
                    target <= static_cast<int64_t>(child.pc)) stable = false;
            }
            // A branch around the child can join a numeric reader with no child definition.
            // The parent-dominance check above does not cover this second edge.
            if (preceding.fmt == Rdna2Format::SOPP &&
                sopp_opcode_is_direct_branch(preceding.opcode)) {
                const int64_t target = static_cast<int64_t>(preceding.pc) +
                    preceding.len_dwords + preceding.simm16;
                if (target > static_cast<int64_t>(child.pc) &&
                    child_lifetime.has_reader_after_bypassed_load(
                        by_pc.at(static_cast<uint32_t>(target)))) stable = false;
            }
        }
        if (stable) proven.push_back(child.pc);
    }
    return proven;
}

std::vector<uint32_t> rdna2_raw_nested_numeric_loads(const std::vector<Rdna2Inst>& ins) {
    const auto numeric = rdna2_raw_wide_data_loads(ins);
    std::vector<uint32_t> result;
    for (size_t index = 0; index < ins.size(); ++index) {
        const auto& load = ins[index];
        if (load.fmt != Rdna2Format::SMEM || load.src[0].kind != OperandKind::SGPR ||
            load.src[1].kind != OperandKind::Special || load.src[1].value != 125 ||
            !std::binary_search(numeric.begin(), numeric.end(), load.pc)) continue;
        bool written = false;
        for (size_t before = 0; before < index; ++before)
            for_each_scalar_write(ins[before], [&](int base, uint32_t width) {
                written |= base < load.src[0].value + 2 &&
                           load.src[0].value < base + static_cast<int>(width);
            });
        if (written) result.push_back(load.pc);
    }
    return result;
}

std::vector<RawNestedWideChain> rdna2_owned_nested_wide_chains(
        const std::vector<Rdna2Inst>& ins) {
    if (std::any_of(ins.begin(), ins.end(), rdna2_may_write_guest_memory)) return {};
    const auto children = rdna2_proven_raw_nested_wide_data_loads(ins);
    const auto parents = rdna2_proven_raw_immediate_wide_data_loads(ins);
    std::vector<RawNestedWideChain> chains;
    for (uint32_t pc : children) {
        const auto child = std::find_if(ins.begin(), ins.end(),
            [pc](const Rdna2Inst& in) { return in.pc == pc; });
        if (child == ins.end()) return {};
        const Rdna2Inst* parent = nullptr;
        for (auto it = ins.begin(); it != child; ++it)
            if (it->fmt == Rdna2Format::SMEM &&
                (it->opcode == 0x2u || it->opcode == 0x3u) &&
                it->dst.kind == OperandKind::SGPR &&
                it->dst.value == child->src[0].value &&
                std::binary_search(parents.begin(), parents.end(), it->pc)) parent = &*it;
        if (!parent) return {};
        // One bounded hop only. The effective snapshots are short; offsets cannot authenticate
        // a giant allocation, negative addressing, or a later pointer chain.
        const uint32_t parent_bytes = parent->opcode == 0x2u ? 16u : 32u;
        const uint32_t child_bytes = child->opcode == 0x2u ? 16u : 32u;
        if (uint64_t(parent->literal) + parent_bytes > 0x100000u ||
            uint64_t(child->literal) + child_bytes > 0x100000u) continue;
        chains.push_back({parent->pc, child->pc, parent_bytes, child_bytes,
                          parent->literal, child->literal});
    }
    return chains;
}

} // namespace prosper::gpu
