#pragma once
// Materialize a lazily-sampled persistent colour target's CPU pixels by reading the GPU target back.
// Moved verbatim out of image_resources.cpp; used only by materialize_image_resource.

#include "shared/live/live_renderer_internal.hpp"   // RttSurf
#include "diagnostics/readback_refusal.hpp"
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace prosper::frontend::submit_renderer {

inline void materialize_lazy_sampled_color_target(RttSurf& surface, uint64_t address,
                                         VkFormat format, uint64_t expected_bytes) {
    std::vector<uint8_t> materialized;
    std::string error;
    const bool readback_ok = [&] {
        namespace refusal = prosper::diagnostics::readback_refusal;
        if (!refusal::selected(address))
            return prosper::test::readback_persistent_color_target(
                address, surface.w, surface.h, format, materialized, error);
        refusal::Record record{};
        record.context.caller = refusal::Caller::LazySampled;
        record.context.frontend_gpu_valid = surface.gpu_valid
            ? refusal::ObservedBool::Yes : refusal::ObservedBool::No;
        record.context.frontend_cpu_pixels = surface.rgba
            ? refusal::ObservedBool::Yes : refusal::ObservedBool::No;
        const bool result = prosper::test::readback_persistent_color_target(
            address, surface.w, surface.h, format, materialized, error, 0, &record);
        refusal::emit(record);
        return result;
    }();
    if (readback_ok && materialized.size() == expected_bytes) {
        surface.rgba = std::make_shared<const std::vector<uint8_t>>(std::move(materialized));
    } else {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) < 24)
            fprintf(stderr,
                    "[rtt] lazy sampled target readback failed: "
                    "base=0x%llx extent=%ux%u error=%s\n",
                    (unsigned long long)address, surface.w, surface.h, error.c_str());
    }
}

} // namespace prosper::frontend::submit_renderer
