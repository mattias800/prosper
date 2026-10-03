#pragma once

#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/execute/graphics_execution_activity.hpp"

namespace prosper::gpu {
struct DrawItem;

// Executor implementation detail, deliberately not included by gpu_execute.hpp. Its dependency
// latch is driven by actual ordered outcomes, never a public supplied completion flag.
struct OrderedGraphicsReadPointIssuer {
    const GraphicsProducerStatus baseline;
    bool dependencies_ok = true;

private:
    const GraphicsExecutionActivity& execution_;
    const std::shared_ptr<OrderedGraphicsReadPoint::Epoch> epoch_ =
        std::make_shared<OrderedGraphicsReadPoint::Epoch>();

public:
    OrderedGraphicsReadPointIssuer(GraphicsProducerStatus initial,
                                   const GraphicsExecutionActivity& execution)
        : baseline(initial), execution_(execution) {}
    OrderedGraphicsReadPointIssuer(const OrderedGraphicsReadPointIssuer&) = delete;
    OrderedGraphicsReadPointIssuer& operator=(const OrderedGraphicsReadPointIssuer&) = delete;
    ~OrderedGraphicsReadPointIssuer() { epoch_->active.store(false, std::memory_order_release); }
    void advance() { epoch_->sequence.fetch_add(1, std::memory_order_acq_rel); }
    std::shared_ptr<const OrderedGraphicsReadPoint> issue(uint64_t submit, uint64_t order,
                                                          const std::vector<DrawItem>& pending,
                                                          uint64_t vertex, uint64_t fragment) const;
};
} // namespace prosper::gpu
