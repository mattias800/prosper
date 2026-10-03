#include "gpu/execute/ordered_graphics_read_point_internal.hpp"
#include "gpu/execute/gpu_execute.hpp"

#include <atomic>

namespace prosper::gpu {

// Historical failures belong to the baseline; only a new failure breaks this submit's epoch.
std::shared_ptr<const OrderedGraphicsReadPoint>
OrderedGraphicsReadPointIssuer::issue(uint64_t submit, uint64_t order,
                                      const std::vector<DrawItem>& pending, uint64_t vertex,
                                      uint64_t fragment) const {
    if (!submit || !dependencies_ok || !pending.empty() || !baseline.known) return {};
    const auto current = graphics_producer_status();
    if (!current.known || current.pending || current.failures != baseline.failures) return {};
    std::vector<OrderedGraphicsReadPoint::Source> sources;
    for (uint64_t address : {vertex, fragment}) {
        // Complete ordinary registered programs only. A separately allocated continuation or
        // fused body needs its own association contract, not a guessed front-only identity.
        if (!address || prosper_agc_shader_continuation_for_code(address) ||
            prosper_agc_fused_back_header_for_front(address))
            return {};
        auto original = registered_graphics_read_source(address);
        if (!original.words || original.words->empty() || !original.chains) return {};
        sources.push_back({address, std::move(original.words), std::move(original.chains)});
    }
    const auto after = graphics_producer_status();
    if (!after.known || after.pending || after.failures != baseline.failures) return {};
    static std::atomic<uint64_t> next_identity{1};
    const uint64_t identity = next_identity.fetch_add(1, std::memory_order_relaxed);
    if (!identity) return {};
    return std::shared_ptr<const OrderedGraphicsReadPoint>(
        new OrderedGraphicsReadPoint(identity, submit, order, baseline.failures, std::move(sources),
                                     epoch_, epoch_->sequence.load(std::memory_order_acquire)));
}

std::shared_ptr<const std::vector<uint32_t>>
OrderedGraphicsReadPoint::source(uint64_t address) const {
    for (const auto& candidate : sources_)
        if (candidate.address == address) return candidate.words;
    return {};
}

bool OrderedGraphicsReadPoint::owns_chains(uint64_t address,
                                           const std::vector<RawNestedWideChain>& chains) const {
    for (const auto& candidate : sources_)
        if (candidate.address == address) return candidate.chains && *candidate.chains == chains;
    return false;
}

bool OrderedGraphicsReadPoint::valid_for(
    uint64_t submit, uint64_t order, uint64_t source_address,
    const std::shared_ptr<const std::vector<uint32_t>>& owner) const {
    if (!identity_ || !submit || submit != submit_ || order != order_ || !owner) return false;
    const auto expected = source(source_address);
    if (!expected || expected.get() != owner.get() || expected.owner_before(owner) ||
        owner.owner_before(expected))
        return false;
    const auto ordered = [&] {
        return epoch_ && epoch_->active.load(std::memory_order_acquire) &&
               epoch_->sequence.load(std::memory_order_acquire) == sequence_;
    };
    if (!ordered()) return false;
    const auto current = graphics_producer_status();
    return current.known && !current.pending && current.failures == failure_generation_ &&
           ordered();
}

}   // namespace prosper::gpu
