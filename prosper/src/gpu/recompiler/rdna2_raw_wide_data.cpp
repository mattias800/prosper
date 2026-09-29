// Distinguish descriptor provenance from scalar data in typeless raw x4/x8 SMEM loads.
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_cfg_support.hpp"
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace prosper::gpu {
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
        // VOPC includes two-word integer compares. Over-approximating its other forms can only
        // refuse a placeholder, never hide a live high half.
        if (in.fmt == Rdna2Format::SOP1 || in.fmt == Rdna2Format::SOP2 ||
            in.fmt == Rdna2Format::SOPC || in.fmt == Rdna2Format::VOP3 ||
            in.fmt == Rdna2Format::VOPC) return 2;
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
    for (size_t index = 0; index < ins.size(); ++index) {
        const Rdna2Inst& load = ins[index];
        if (load.fmt != Rdna2Format::SMEM ||
            (load.opcode != 0x2u && load.opcode != 0x3u) ||
            (load.src[1].kind == OperandKind::Special && load.src[1].value == 125) ||
            load.dst.kind != OperandKind::SGPR || load.dst.value < 0 ||
            load.dst.value + (load.opcode == 0x2u ? 4 : 8) > 128)
            continue;
        const uint32_t words = load.opcode == 0x2u ? 4u : 8u;
        if (RawWideLifetime(ins, by_pc, index, words).requires_backing())
            data_loads.push_back(load.pc);
    }
    return data_loads;
}

} // namespace prosper::gpu
