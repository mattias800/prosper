#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_placeholder_origins.hpp"
#include "gpu/recompiler/ir/cfg/guest_cfg.hpp"
#include <vector>
#include <cstdint>
#include <utility>
#include "gpu/recompiler/rdna2_placeholder_origin_trace.hpp"
#include "gpu/recompiler/rdna2_vector_old_word.hpp"
#include <algorithm>
#include <set>

namespace prosper::gpu::detail {
PlaceholderOriginTrace::PlaceholderOriginTrace(const PlaceholderOriginProgram& p)
    : program(p), cfg(ir::cfg::build_guest_cfg(p.instructions, {})),
      predecessors(p.instructions.size()), full_exec(p.instructions.size(), true) {
    std::vector<bool> exits(cfg.blocks.size(), true);
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto block : cfg.rpo) {
            bool full = block == 0 ? p.entry_full_exec : true;
            for (auto pred : cfg.preds[block])
                if (cfg.reachable[pred]) full = full && exits[pred];
            const auto& b = cfg.blocks[block];
            for (uint32_t i = b.first; i <= b.last; ++i) {
                full_exec[i] = full;
                const auto& in = p.instructions[i];
                if (rdna2_instruction_may_change_exec(in))
                    full = vector_exec_is_full_after_write(in, full, p.wave_size);
            }
            if (exits[block] != full) {
                exits[block] = full;
                changed = true;
            }
        }
    }
    for (uint32_t block = 0; block < cfg.blocks.size(); ++block) {
        const auto& b = cfg.blocks[block];
        for (uint32_t i = b.first + 1; i <= b.last; ++i) predecessors[i].push_back(i - 1);
        for (auto pred : cfg.preds[block])
            if (cfg.reachable[pred]) predecessors[b.first].push_back(cfg.blocks[pred].last);
    }
}

PlaceholderRoots PlaceholderOriginTrace::before(uint32_t pc,
                                                const std::vector<PlaceholderWord>& words,
                                                uint32_t work_limit) const {
    const auto& ins = program.instructions;
    const auto at = std::find_if(ins.begin(), ins.end(), [&](const auto& x) { return x.pc == pc; });
    PlaceholderRoots roots;
    if (at == ins.end() || predecessors.empty()) {
        for (auto word : words) roots.insert({PlaceholderOrigin::Unreachable, word, pc});
        return roots;
    }
    using Work = std::pair<uint32_t, PlaceholderWord>;
    std::vector<Work> work;
    std::set<Work> seen;
    const auto enqueue_before = [&](uint32_t index, PlaceholderWord word) {
        if (index == 0) {
            const auto entry_roots = entry(word);
            roots.insert(entry_roots.begin(), entry_roots.end());
        }
        for (auto pred : predecessors[index]) work.emplace_back(pred, word);
    };
    for (auto word : words) enqueue_before(static_cast<uint32_t>(at - ins.begin()), word);
    while (!work.empty()) {
        const auto [index, word] = work.back();
        work.pop_back();
        if (!seen.emplace(index, word).second) continue;
        if (seen.size() > work_limit) {
            roots.insert({PlaceholderOrigin::TraceBudget, word, ins[index].pc});
            break;
        }
        const auto& in = ins[index];
        if (!writes(in, word)) {
            enqueue_before(index, word);
            continue;
        }
        if (word.file == PlaceholderFile::Spill &&
            !(in.fmt == Rdna2Format::VOP3 && in.opcode == 0x361u)) {
            roots.insert({PlaceholderOrigin::Spill, word, in.pc});
            continue;
        }
        const auto inputs = sources(in, word);
        // The legacy read inventory intentionally declines uninventoried image forms and
        // implicit LDS/GDS sources. Name that gap instead of inventing an unloaded root.
        if ((in.fmt == Rdna2Format::MIMG && image_addresses(in).empty()) ||
            (in.fmt == Rdna2Format::DS && (in.ds_gds || inputs.empty())))
            roots.insert({PlaceholderOrigin::UnmodelledTransfer, word, in.pc});
        for (auto input : inputs) enqueue_before(index, input);
        if (word.file == PlaceholderFile::Vector &&
            vector_write_depends_on_old_word(in, full_exec[index]))
            enqueue_before(index, word);
        if (word.file == PlaceholderFile::Spill &&
            (word.lane < 0 || in.src[1].kind != OperandKind::InlineInt))
            enqueue_before(index, word);
    }
    if (roots.empty())
        for (auto word : words) roots.insert({PlaceholderOrigin::LostDefinition, word, pc});
    return roots;
}
} // namespace prosper::gpu::detail

namespace prosper::gpu {
PlaceholderRoots placeholder_roots_before(const PlaceholderOriginProgram& program, uint32_t pc,
                                          const std::vector<PlaceholderWord>& words) {
    return detail::PlaceholderOriginTrace(program).before(pc, words);
}
} // namespace prosper::gpu
