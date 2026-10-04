#pragma once
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace prosper::gpu {
struct GpuState;
class NativeGraphicsStageCompilation;
class OrderedScalarBankReadPoint;
class OriginalFragmentDrawProducer;
// Complete original no-writer facts and the actual selected module association for one draw.
// This is a retained historical producer, NOT descriptor/read-point permission. A bank FS is
// deliberately distinct from a native FS; lower consumers never infer original effects from
// the absence of reflected writes in a substituted/omitted module.
class OriginalGraphicsDrawEffects {
    friend std::shared_ptr<const OriginalGraphicsDrawEffects>
    seal_original_graphics_draw_effects(const OrderedScalarBankReadPoint&, const GpuState&,
                                        uint64_t,
                                        std::shared_ptr<const NativeGraphicsStageCompilation>,
                                        std::shared_ptr<const NativeGraphicsStageCompilation>,
                                        std::shared_ptr<const OriginalFragmentDrawProducer>);
    const std::shared_ptr<const NativeGraphicsStageCompilation> vertex_, fragment_;
    const std::shared_ptr<const OriginalFragmentDrawProducer> original_fragment_;
    const uint64_t submit_, order_;
    OriginalGraphicsDrawEffects(std::shared_ptr<const NativeGraphicsStageCompilation> vertex,
                                std::shared_ptr<const NativeGraphicsStageCompilation> fragment,
                                std::shared_ptr<const OriginalFragmentDrawProducer> original,
                                uint64_t submit, uint64_t order)
        : vertex_(std::move(vertex)), fragment_(std::move(fragment)),
          original_fragment_(std::move(original)), submit_(submit), order_(order) {}

public:
    uint64_t source_submit() const { return submit_; }
    bool matches_draw(uint64_t submit, uint64_t order,
                      const std::shared_ptr<const std::vector<uint32_t>>& vertex,
                      const std::shared_ptr<const std::vector<uint32_t>>& fragment,
                      std::span<const uint32_t> geometry,
                      std::span<const uint32_t> fragment_words) const;
};
} // namespace prosper::gpu
