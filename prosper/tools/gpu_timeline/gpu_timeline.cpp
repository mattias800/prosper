#include "gpu/gpu_timeline.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>

using namespace prosper::gpu;

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--records")) {
        std::fprintf(stderr, "usage: gpu_timeline <capture.prgtl> [--records]\n");
        return 2;
    }
    GpuTimelineFile timeline;
    std::string error;
    if (!read_gpu_timeline(argv[1], timeline, error)) {
        std::fprintf(stderr, "gpu_timeline: %s: %s\n", argv[1], error.c_str());
        return 1;
    }

    uint64_t last_ns = 0, draws = 0, dispatches = 0;
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> target_extents;
    for (const auto& submit : timeline.submits) {
        last_ns = std::max(last_ns, submit.elapsed_ns);
        draws += submit.draw_count;
        dispatches += submit.dispatch_count;
        if (submit.color0_width && submit.color0_height)
            target_extents[{submit.color0_width, submit.color0_height}]++;
    }
    for (const auto& present : timeline.presents) last_ns = std::max(last_ns, present.elapsed_ns);
    const double seconds = static_cast<double>(last_ns) / 1e9;
    std::printf("timeline version=%u revision=%s title=%s route=%s\n",
                timeline.version, timeline.metadata.revision.c_str(), timeline.metadata.title_id.c_str(),
                timeline.metadata.input_route.c_str());
    std::printf("duration=%.3fs submits=%zu presents=%zu details=%zu producers=%zu "
                "draws=%llu dispatches=%llu truncated_tail=%s\n",
                seconds, timeline.submits.size(), timeline.presents.size(), timeline.details.size(),
                timeline.producers.size(),
                static_cast<unsigned long long>(draws), static_cast<unsigned long long>(dispatches),
                timeline.truncated_tail ? "yes" : "no");
    if (seconds > 0)
        std::printf("rates submits=%.1f/s presents=%.1f/s\n",
                    timeline.submits.size() / seconds, timeline.presents.size() / seconds);
    for (const auto& [extent, count] : target_extents)
        std::printf("target %ux%u submits=%llu\n", extent.first, extent.second,
                    static_cast<unsigned long long>(count));
    for (const auto& detail : timeline.details)
        std::printf("detail submit=%llu realized=%u/%u draws %u/%u dispatches operations=%u missing=%u "
                    "versions=%u shaders/%u resources bytes=%llu capture=%s\n",
                    static_cast<unsigned long long>(detail.submit_no), detail.realized_draw_count,
                    detail.semantic_draw_count, detail.realized_dispatch_count,
                    detail.semantic_dispatch_count, detail.operation_count,
                    detail.missing_operation_count, detail.shader_version_count,
                    detail.resource_version_count, static_cast<unsigned long long>(detail.resource_bytes),
                    detail.capture_path.c_str());
    for (const auto& producer : timeline.producers)
        std::printf("producer consumer=%llu/op%u resource=%016llx/%ux%u -> %s"
                    " submit=%llu draw=%llu order=%llu target=%016llx/%ux%u\n",
                    static_cast<unsigned long long>(producer.consumer_submit_no),
                    producer.consumer_operation,
                    static_cast<unsigned long long>(producer.resource_addr),
                    producer.resource_width, producer.resource_height,
                    producer.resolved ? "resolved" : "unresolved",
                    static_cast<unsigned long long>(producer.producer_submit_no),
                    static_cast<unsigned long long>(producer.producer_draw_index),
                    static_cast<unsigned long long>(producer.producer_command_order),
                    static_cast<unsigned long long>(producer.producer_target_addr),
                    producer.producer_width, producer.producer_height);

    if (argc == 3) {
        size_t si = 0, pi = 0, di = 0, xi = 0;
        while (si < timeline.submits.size() || pi < timeline.presents.size() ||
               di < timeline.details.size() || xi < timeline.producers.size()) {
            const uint64_t ss = si < timeline.submits.size() ? timeline.submits[si].sequence : UINT64_MAX;
            const uint64_t ps = pi < timeline.presents.size() ? timeline.presents[pi].sequence : UINT64_MAX;
            const uint64_t ds = di < timeline.details.size() ? timeline.details[di].sequence : UINT64_MAX;
            const uint64_t xs = xi < timeline.producers.size() ? timeline.producers[xi].sequence : UINT64_MAX;
            if (ss <= ps && ss <= ds && ss <= xs) {
                const auto& s = timeline.submits[si++];
                std::printf("%llu %.6f submit=%llu draws=%u dispatches=%u order=%llu..%llu "
                            "target=%016llx/%ux%u\n",
                            static_cast<unsigned long long>(s.sequence), s.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(s.submit_no), s.draw_count, s.dispatch_count,
                            static_cast<unsigned long long>(s.first_command_order),
                            static_cast<unsigned long long>(s.last_command_order),
                            static_cast<unsigned long long>(s.color0_base),
                            s.color0_width, s.color0_height);
            } else if (ps <= ds && ps <= xs) {
                const auto& p = timeline.presents[pi++];
                std::printf("%llu %.6f present=%llu latest-submit=%llu buffer=%d flip-arg=%lld extent=%ux%u\n",
                            static_cast<unsigned long long>(p.sequence), p.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(p.present_count),
                            static_cast<unsigned long long>(p.latest_submit_no), p.buffer_index,
                            static_cast<long long>(p.flip_arg), p.width, p.height);
            } else if (ds <= xs) {
                const auto& d = timeline.details[di++];
                std::printf("%llu %.6f detail-submit=%llu realized=%u/%u+%u/%u operations=%u missing=%u "
                            "versions=%u/%u bytes=%llu capture=%s\n",
                            static_cast<unsigned long long>(d.sequence), d.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(d.submit_no), d.realized_draw_count,
                            d.semantic_draw_count, d.realized_dispatch_count, d.semantic_dispatch_count,
                            d.operation_count, d.missing_operation_count, d.shader_version_count,
                            d.resource_version_count, static_cast<unsigned long long>(d.resource_bytes),
                            d.capture_path.c_str());
            } else {
                const auto& x = timeline.producers[xi++];
                std::printf("%llu %.6f producer consumer=%llu/op%u resource=%016llx/%ux%u "
                            "future=%u resolved=%s submit=%llu draw=%llu order=%llu\n",
                            static_cast<unsigned long long>(x.sequence), x.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(x.consumer_submit_no), x.consumer_operation,
                            static_cast<unsigned long long>(x.resource_addr), x.resource_width,
                            x.resource_height, x.future_writer_operation, x.resolved ? "yes" : "no",
                            static_cast<unsigned long long>(x.producer_submit_no),
                            static_cast<unsigned long long>(x.producer_draw_index),
                            static_cast<unsigned long long>(x.producer_command_order));
            }
        }
    }
    if (timeline.truncated_tail)
        std::fprintf(stderr, "gpu_timeline: warning: ignored a truncated final record\n");
    return 0;
}
