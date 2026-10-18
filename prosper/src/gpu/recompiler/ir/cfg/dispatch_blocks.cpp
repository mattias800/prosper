// dispatch_blocks.cpp -- see dispatch_blocks.hpp and AGENTS.md.
#include "gpu/recompiler/ir/cfg/dispatch_blocks.hpp"

#include <algorithm>
#include <utility>

namespace prosper::gpu::ir::cfg {

namespace {

thread_local std::vector<DispatchBlocksRecord>* g_capture = nullptr;

std::string join(const std::vector<uint32_t>& values) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); ++i)
        out += (i ? "," : "") + std::to_string(values[i]);
    return out + "]";
}

}   // namespace

DispatchBlocks dispatch_blocks(const GuestCfg& cfg) {
    DispatchBlocks out;
    out.successors.resize(cfg.blocks.size());
    for (size_t b = 0; b < cfg.blocks.size(); ++b) {
        const GuestBlock& block = cfg.blocks[b];
        out.starts.push_back(block.pc);
        auto& edges = out.successors[b];
        auto add = [&](uint32_t successor) {
            if (successor != kNone && std::find(edges.begin(), edges.end(), successor) == edges.end())
                edges.push_back(successor);
        };
        if (block.term == Terminator::Branch || terminator_is_conditional(block.term))
            add(block.succ[1]);
        if (block.term != Terminator::Branch) add(block.succ[0]);
    }
    return out;
}

std::string dispatch_blocks_difference(const DispatchBlocks& dispatcher, const DispatchBlocks& cfg) {
    const size_t n = std::min(dispatcher.starts.size(), cfg.starts.size());
    for (size_t k = 0; k < n; ++k)
        if (dispatcher.starts[k] != cfg.starts[k])
            return "block " + std::to_string(k) + " starts at pc " +
                   std::to_string(dispatcher.starts[k]) + ", guest cfg pc " +
                   std::to_string(cfg.starts[k]);
    if (dispatcher.starts.size() != cfg.starts.size())
        return "blocks " + std::to_string(dispatcher.starts.size()) + ", guest cfg " +
               std::to_string(cfg.starts.size());
    for (size_t k = 0; k < n; ++k)
        if (k >= dispatcher.successors.size() || k >= cfg.successors.size() ||
            dispatcher.successors[k] != cfg.successors[k])
            return "block " + std::to_string(k) + " (pc " + std::to_string(cfg.starts[k]) +
                   ") successors " +
                   (k < dispatcher.successors.size() ? join(dispatcher.successors[k]) : "?") +
                   ", guest cfg " + (k < cfg.successors.size() ? join(cfg.successors[k]) : "?");
    return {};
}

bool adopt_guest_cfg_blocks(std::span<const Rdna2Inst> ins, const SplitPolicy& policy,
                            std::vector<uint32_t>& starts,
                            std::vector<std::vector<uint32_t>>& successors) {
    DispatchBlocks own{starts, successors};
    const GuestCfg cfg = build_guest_cfg(ins, policy);
    DispatchBlocks from_cfg = dispatch_blocks(cfg);
    DispatchBlocksRecord record;
    record.services = static_cast<size_t>(
        std::count_if(cfg.blocks.begin(), cfg.blocks.end(),
                      [](const GuestBlock& block) { return block.service_event != kNone; }));
    record.entry_pc = ins.empty() ? 0 : ins.front().pc;
    record.blocks = own.starts.size();
    record.cfg_blocks = from_cfg.starts.size();
    record.difference = dispatch_blocks_difference(own, from_cfg);
    record.adopted = record.difference.empty();
    if (record.adopted) {
        starts = std::move(from_cfg.starts);
        successors = std::move(from_cfg.successors);
    }
    if (g_capture) g_capture->push_back(record);
    return record.adopted;
}

DispatchBlocksCapture::DispatchBlocksCapture() : previous_(g_capture) { g_capture = &records_; }

DispatchBlocksCapture::~DispatchBlocksCapture() { g_capture = previous_; }

}   // namespace prosper::gpu::ir::cfg
