#include "shared/live/submit_renderer/native_bc_chain_audit.hpp"

#include "gpu/texture/bc_decode.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace prosper::frontend::submit_renderer {

NativeBcChainAudit audit_native_bc_chain(const uint8_t* chain, size_t chain_bytes, uint32_t width,
                                         uint32_t height, uint32_t levels, uint32_t block_bytes,
                                         prosper::gpu::DataFormat format) {
    NativeBcChainAudit audit;
    if (!chain || !width || !height || !block_bytes) return audit;
    std::vector<uint8_t> above;
    std::vector<uint8_t> level_rgba;
    uint32_t above_width = 0;
    uint32_t above_height = 0;
    size_t offset = 0;
    // A 32-bit extent has at most 32 levels; more would shift by the type's width.
    for (uint32_t level = 0; level < std::min(levels, 32u); ++level) {
        const uint32_t level_width = std::max(width >> level, 1u);
        const uint32_t level_height = std::max(height >> level, 1u);
        const size_t level_bytes =
            static_cast<size_t>((level_width + 3u) / 4u) * ((level_height + 3u) / 4u) * block_bytes;
        if (level_bytes > chain_bytes - offset) break;
        level_rgba.assign(static_cast<size_t>(level_width) * level_height * 4u, 0);
        if (!prosper::gpu::bc_decode_surface(level_rgba.data(), chain + offset, level_bytes,
                                             level_width, level_height, format))
            return {};
        if (level) {
            const auto above_at = [&](uint32_t x, uint32_t y, uint32_t channel) {
                x = std::min(x, above_width - 1u);
                y = std::min(y, above_height - 1u);
                return static_cast<int>(
                    above[(static_cast<size_t>(y) * above_width + x) * 4u + channel]);
            };
            const auto box = [&](uint32_t x, uint32_t y, uint32_t channel) {
                return (above_at(2u * x, 2u * y, channel) + above_at(2u * x + 1u, 2u * y, channel) +
                        above_at(2u * x, 2u * y + 1u, channel) +
                        above_at(2u * x + 1u, 2u * y + 1u, channel) + 2) /
                       4;
            };
            double difference = 0.0;
            double shifted = 0.0;
            for (uint32_t y = 0; y < level_height; ++y)
                for (uint32_t x = 0; x < level_width; ++x)
                    for (uint32_t channel = 0; channel < 3u; ++channel) {
                        const size_t texel = (static_cast<size_t>(y) * level_width + x) * 4u;
                        const int value = level_rgba[texel + channel];
                        difference += std::abs(value - box(x, y, channel));
                        if (level == 1u)
                            shifted += std::abs(
                                value - box((x + level_width / 2u) % level_width, y, channel));
                    }
            const double samples = static_cast<double>(level_width) * level_height * 3.0;
            audit.level_mad.push_back(difference / samples);
            if (level == 1u) audit.shifted_control = shifted / samples;
        }
        above.swap(level_rgba);
        above_width = level_width;
        above_height = level_height;
        offset += level_bytes;
    }
    return audit;
}

void log_native_bc_chain_audit(const std::vector<uint8_t>& chain,
                               const prosper::gpu::MipChainPlan& plan, uint32_t levels,
                               uint32_t block_bytes, const prosper::gpu::ShaderResource& resource) {
    if (plan.levels.empty()) return;
    const uint32_t width = plan.levels[0].width;
    const uint32_t height = plan.levels[0].height;
    const NativeBcChainAudit audit = audit_native_bc_chain(
        chain.data(), chain.size(), width, height, levels, block_bytes, resource.format);
    std::string per_level;
    for (size_t i = 0; i < audit.level_mad.size(); ++i) {
        std::array<char, 32> field{};
        const bool tail = i + 1u < plan.levels.size() && plan.levels[i + 1u].in_tail;
        std::snprintf(field.data(), field.size(), "%s%.2f%s", i ? "," : "", audit.level_mad[i],
                      tail ? "t" : "");
        per_level += field.data();
    }
    std::fprintf(stderr,
                 "[native-bc-audit] addr=0x%llx fmt=%u %ux%u levels=%u tile=%u l1_vs_box0=%.2f "
                 "shifted_control=%.2f levels_mad=%s\n",
                 static_cast<unsigned long long>(resource.gpu_addr),
                 static_cast<unsigned>(resource.format), width, height, levels, resource.tile_mode,
                 audit.level_mad.empty() ? 0.0 : audit.level_mad[0], audit.shifted_control,
                 per_level.c_str());
}

} // namespace prosper::frontend::submit_renderer
