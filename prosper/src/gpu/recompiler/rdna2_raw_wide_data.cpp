// Distinguish descriptor provenance from scalar data in typeless raw x4/x8 SMEM loads.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
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
        return in.opcode != 0x00u && in.opcode != 0x0eu && in.opcode != 0x27u;
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
        // May-provenance, not a value proof: a read on ANY reachable path needs the actual
        // bytes. Merge arriving register/SCC origins with OR and reprocess when they grow.
        // Scalar descriptor assembly remains provenance-only until a non-descriptor consumer.
        struct State {
            size_t index;
            std::bitset<128> regs;
            bool scc;
        };
        if (start + 1 >= ins.size()) return false;
        State initial{start + 1, {}, false};
        for (uint32_t word = 0; word < words; ++word)
            initial.regs.set(static_cast<size_t>(first + static_cast<int>(word)));
        std::vector<State> pending{initial};
        std::vector<std::array<std::bitset<128>, 2>> seen(ins.size());
        std::vector<std::array<bool, 2>> seen_any(ins.size());
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
                (state.regs & ~seen[state.index][slot]).none()) continue;
            seen_any[state.index][slot] = true;
            seen[state.index][slot] |= state.regs;
            state.regs = seen[state.index][slot];
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
                    if (operand.kind == OperandKind::Special && operand.value == 253) {
                        derived_read |= state.scc;
                        continue;
                    }
                    if ((operand.kind != OperandKind::SGPR &&
                         !(operand.kind == OperandKind::Special &&
                           operand.value >= 106 && operand.value <= 124)) ||
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
                 rdna2_instruction_may_change_exec(in))) return true;

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
                 in.fmt == Rdna2Format::SOPK || in.fmt == Rdna2Format::SMEM);
            for_each_scalar_write(in, [&](int base, uint32_t width) {
                for (uint32_t k = 0; k < width; ++k) {
                    const int reg = base + static_cast<int>(k);
                    if (reg < 0 || reg >= 128) continue;
                    if (definite_scalar_write)
                        state.regs.reset(static_cast<size_t>(reg));
                    if (derived_read && scalar_result &&
                        in.dst.kind == OperandKind::SGPR &&
                        base == in.dst.value)
                        produced.set(static_cast<size_t>(reg));
                }
            }, /*wave32_one_word_masks*/true);
            state.regs |= produced;
            if (plain_copy)
                for (uint32_t k = 0; k < copy_words; ++k)
                    if (copied[k] && in.dst.value >= 0 &&
                        in.dst.value + static_cast<int>(k) < 128)
                        state.regs.set(static_cast<size_t>(in.dst.value + k));
            if (writes_scc(in)) state.scc = derived_read;
            if (!state.regs.any() && !state.scc) continue;

            auto enqueue = [&](size_t next) {
                if (next <= start) return false; // unsupported re-entry to producer or prefix
                pending.push_back({next, state.regs, state.scc});
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
        });
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
std::vector<uint32_t> rdna2_proven_raw_immediate_wide_data_loads(
        const std::vector<Rdna2Inst>& ins) {
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
        } else if (!sopp_is_noop(in) && in.opcode != kSoppOpcodeBarrier)
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
            if (in.pc < load.pc && rdna2_may_write_guest_memory(in))
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
        if (child.fmt != Rdna2Format::SMEM ||
            (child.opcode != 0x2u && child.opcode != 0x3u) ||
            child.dst.kind != OperandKind::SGPR ||
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
        }
        if (stable) proven.push_back(child.pc);
    }
    return proven;
}

} // namespace prosper::gpu
