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
    std::printf("timeline version=1 revision=%s title=%s route=%s\n",
                timeline.metadata.revision.c_str(), timeline.metadata.title_id.c_str(),
                timeline.metadata.input_route.c_str());
    std::printf("duration=%.3fs submits=%zu presents=%zu draws=%llu dispatches=%llu truncated_tail=%s\n",
                seconds, timeline.submits.size(), timeline.presents.size(),
                static_cast<unsigned long long>(draws), static_cast<unsigned long long>(dispatches),
                timeline.truncated_tail ? "yes" : "no");
    if (seconds > 0)
        std::printf("rates submits=%.1f/s presents=%.1f/s\n",
                    timeline.submits.size() / seconds, timeline.presents.size() / seconds);
    for (const auto& [extent, count] : target_extents)
        std::printf("target %ux%u submits=%llu\n", extent.first, extent.second,
                    static_cast<unsigned long long>(count));

    if (argc == 3) {
        size_t si = 0, pi = 0;
        while (si < timeline.submits.size() || pi < timeline.presents.size()) {
            const bool take_submit = pi == timeline.presents.size() ||
                (si < timeline.submits.size() &&
                 timeline.submits[si].sequence < timeline.presents[pi].sequence);
            if (take_submit) {
                const auto& s = timeline.submits[si++];
                std::printf("%llu %.6f submit=%llu draws=%u dispatches=%u order=%llu..%llu "
                            "target=%016llx/%ux%u\n",
                            static_cast<unsigned long long>(s.sequence), s.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(s.submit_no), s.draw_count, s.dispatch_count,
                            static_cast<unsigned long long>(s.first_command_order),
                            static_cast<unsigned long long>(s.last_command_order),
                            static_cast<unsigned long long>(s.color0_base),
                            s.color0_width, s.color0_height);
            } else {
                const auto& p = timeline.presents[pi++];
                std::printf("%llu %.6f present=%llu latest-submit=%llu buffer=%d flip-arg=%lld extent=%ux%u\n",
                            static_cast<unsigned long long>(p.sequence), p.elapsed_ns / 1e9,
                            static_cast<unsigned long long>(p.present_count),
                            static_cast<unsigned long long>(p.latest_submit_no), p.buffer_index,
                            static_cast<long long>(p.flip_arg), p.width, p.height);
            }
        }
    }
    if (timeline.truncated_tail)
        std::fprintf(stderr, "gpu_timeline: warning: ignored a truncated final record\n");
    return 0;
}
