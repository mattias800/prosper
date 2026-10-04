#pragma once

#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/execute/graphics_execution_activity.hpp"
#include "gpu/execute/scalar_bank_read_point.hpp"

namespace prosper::gpu {
struct DrawItem;
class FragmentScalarBank;

// Executor implementation detail, deliberately not included by gpu_execute.hpp. Its dependency
// latch is driven by actual ordered outcomes, never a public supplied completion flag.
struct OrderedGraphicsReadPointIssuer {
    const GraphicsProducerStatus baseline;
    bool dependencies_ok = true;

private:
    const GraphicsExecutionActivity& execution_;
    const std::shared_ptr<OrderedGraphicsReadPoint::Epoch> epoch_ =
        std::make_shared<OrderedGraphicsReadPoint::Epoch>();
    // Actual executor append/submission events cover this exact queued span. Origins retain
    // original physical allocation births through later VA reuse, not just current target VAs.
    bool scalar_effects_known_ = true;
    size_t scalar_covered_draws_ = 0;
    std::shared_ptr<const OrderedScalarBankReadPoint::PendingDraw> scalar_effects_;
    std::shared_ptr<const std::vector<prosper::GuestDirectAllocation>> scalar_outputs_ =
        std::make_shared<const std::vector<prosper::GuestDirectAllocation>>();

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
    void record_queued_draw(const GpuState&, const DrawItem&, uint64_t submit);
    void submitted_span() {
        scalar_effects_known_ = true;
        scalar_covered_draws_ = 0;
        scalar_effects_.reset();
        scalar_outputs_ = std::make_shared<const std::vector<prosper::GuestDirectAllocation>>();
    }
    std::shared_ptr<const OrderedScalarBankReadPoint>
    issue_scalar(uint64_t submit, uint64_t order, const GpuState&,
                 const std::vector<DrawItem>& pending, std::string& refusal) const;
};
struct OrderedScalarDrawInputs {
    std::shared_ptr<const OrderedScalarBankReadPoint> point;
    std::shared_ptr<const FragmentScalarBank> bank;
};
// The actual ordered realizer calls this before native resource folding. A failed scalar seal
// remains a named refusal; it never borrows RAW's empty-span permission or flushes GPU work.
OrderedScalarDrawInputs prepare_ordered_scalar_draw(OrderedGraphicsReadPointIssuer&,
                                                    uint64_t submit, uint64_t order,
                                                    const GpuState&,
                                                    const std::vector<DrawItem>& pending);
} // namespace prosper::gpu
