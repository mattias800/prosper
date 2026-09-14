#pragma once

#include "gpu/recompiler/rdna2_decode.hpp"
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <utility>
#include <vector>

namespace prosper::gpu {

// Immutable control metadata for one exact, ordered, compact scalar-fold stream.
// It contains no register values, guest pointers, descriptor bytes or publication state.
struct FoldControlStep {
    uint32_t restore_slot = UINT32_MAX;
    uint32_t save_slot = UINT32_MAX;
    uint32_t restore_source_pc = UINT32_MAX;
    bool reset_zero_mip = false;
    bool changes_exec = false;
    uint16_t zero_mip_vgpr = UINT16_MAX; // absent shape; valid VGPRs are 0..255
};
struct FoldControlPlan {
    std::vector<FoldControlStep> steps;
    uint32_t snapshot_count = 0;
    bool cfg_known = true;
    uint64_t allocated_bytes() const {
        return sizeof(*this) + steps.capacity() * sizeof(FoldControlStep);
    }
};

// Cold-path diagnostic: counts actual constructions, including uncached specializations.
// No counter is touched when a warm fold borrows its immutable plan.
inline std::atomic<uint64_t> fold_control_plan_builds{0};

inline FoldControlPlan build_fold_control_plan(const std::vector<Rdna2Inst>& ins) {
    fold_control_plan_builds.fetch_add(1, std::memory_order_relaxed);
    FoldControlPlan plan;
    plan.steps.resize(ins.size());
    // Instruction-only facts are valid even when the stream has an indirect CFG. Derive them
    // before that conservative early return; the live zero-value/EXEC proof stays per invocation.
    for (size_t k = 0; k < ins.size(); ++k) {
        auto& step = plan.steps[k];
        step.changes_exec = rdna2_instruction_may_change_exec(ins[k]);
        uint32_t mip = 0;
        if (rdna2_mimg_zero_mip_shape(ins[k], &mip))
            step.zero_mip_vgpr = static_cast<uint16_t>(mip);
    }
    const auto branch = [](const Rdna2Inst& in) {
        return in.fmt == Rdna2Format::SOPP &&
            (in.opcode == 0x02 || (in.opcode >= 0x04 && in.opcode <= 0x09));
    };
    const auto target = [](const Rdna2Inst& in) {
        return int64_t(in.pc) + int64_t(in.len_dwords) + int64_t(in.simm16);
    };
    for (const auto& in : ins) {
        if ((in.fmt == Rdna2Format::SOP1 && in.opcode >= 0x20 && in.opcode <= 0x22) ||
            (in.fmt == Rdna2Format::SOPK && in.opcode == 0x16)) {
            plan.cfg_known = false;
            for (auto& step : plan.steps) step.reset_zero_mip = true;
            return plan;
        }
    }
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    std::vector<uint32_t> starts;
    if (!ins.empty()) starts.push_back(ins.front().pc);
    for (const auto& in : ins) {
        if (!branch(in)) continue;
        const int64_t t = target(in);
        if (t >= 0 && t <= UINT32_MAX) {
            edges.emplace_back(uint32_t(t), in.pc);
            starts.push_back(uint32_t(t));
        }
        starts.push_back(in.pc + in.len_dwords);
    }
    std::sort(edges.begin(), edges.end());
    std::sort(starts.begin(), starts.end());
    uint32_t previous_block = UINT32_MAX;
    for (size_t k = 0; k < ins.size(); ++k) {
        const uint32_t pc = ins[k].pc;
        const auto block_end = std::upper_bound(starts.begin(), starts.end(), pc);
        const uint32_t block = block_end == starts.begin() ? UINT32_MAX : *std::prev(block_end);
        plan.steps[k].reset_zero_mip = k != 0 && block != previous_block;
        previous_block = block;
        if (k == 0) continue;
        const auto& prev = ins[k - 1];
        // Compaction may have removed a fall-through block: retained adjacency is insufficient.
        if (prev.fmt != Rdna2Format::SOPP || prev.opcode != 0x02 ||
            prev.pc + prev.len_dwords != pc) continue;
        const auto lo = std::lower_bound(edges.begin(), edges.end(), std::pair{pc, uint32_t{0}});
        const auto hi = std::upper_bound(edges.begin(), edges.end(), std::pair{pc, UINT32_MAX});
        if (std::distance(lo, hi) != 1 || lo->second >= pc) continue;
        const auto source = std::lower_bound(ins.begin(), ins.end(), lo->second,
            [](const Rdna2Inst& in, uint32_t source_pc) { return in.pc < source_pc; });
        if (source == ins.end() || source->pc != lo->second) continue;
        auto& save = plan.steps[size_t(source - ins.begin())].save_slot;
        if (save == UINT32_MAX) save = plan.snapshot_count++;
        plan.steps[k].restore_slot = save;
        plan.steps[k].restore_source_pc = lo->second;
    }
    return plan;
}

} // namespace prosper::gpu
