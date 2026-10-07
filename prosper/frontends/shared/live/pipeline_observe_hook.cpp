// pipeline_observe_hook.cpp -- see the header. Everything here is diagnostic: nothing it does is
// visible to the guest or to the dispatch it observes.
#include "shared/live/pipeline_observe_hook.hpp"

#include <atomic>
#include <cstdio>
#include <cstdlib>

#include "diagnostics/env_cache.hpp"
#include "shared/compute/pipeline_observer.hpp"

namespace prosper::frontend {
namespace {

PipelineObserver& observer() {
    // Immortal: the atexit report below is registered before this is first constructed, so a
    // function-local static would be destroyed BEFORE the report reads it, and an executor thread
    // flushing during shutdown could reach it too (#4664 review). Never destroyed, never freed.
    static PipelineObserver& instance = *new PipelineObserver;
    return instance;
}

void print_report(const char* when) {
    std::fprintf(stderr, "[pipeline-observe] report (%s): what ADR 0009 pipelining would have allowed\n", when);
    for (const std::string& line : observer().report()) std::fprintf(stderr, "%s\n", line.c_str());
}

void on_ordered_operation(prosper::gpu::OrderedOperationKind kind, uint32_t /*count*/) {
    observer().other_operation(kind == prosper::gpu::OrderedOperationKind::Dma
                                   ? ObservedOperation::Dma
                                   : ObservedOperation::GraphicsSpan);
}

}  // namespace

bool pipeline_observe_enabled() { return PROSPER_ENV_ON("PROSPER_PIPELINE_OBSERVE"); }

std::vector<BindingRange> dispatch_binding_ranges(const std::vector<BoundBuffer>& buffers,
                                                  const std::vector<BoundImage>& images) {
    std::vector<BindingRange> ranges;
    ranges.reserve(buffers.size() + images.size());
    for (const BoundBuffer& b : buffers) {
        if (!b.resource || b.resource->host_data || !b.guest_bytes) continue;
        ranges.push_back({b.resource->gpu_addr, b.guest_bytes, /*read=*/true, /*write=*/b.writable});
    }
    for (const BoundImage& i : images) {
        if (!i.resource || i.resource->host_data || !i.guest_bytes) continue;
        ranges.push_back({i.resource->gpu_addr, i.guest_bytes,
                          /*read=*/!(i.storage && i.storage_write_only),
                          /*write=*/i.storage && i.storage_writeback});
    }
    return ranges;
}

void pipeline_observe_dispatch(
    const prosper::gpu::ComputeItem& item, const std::vector<BoundBuffer>& buffers,
    const std::vector<BoundImage>& images, std::chrono::steady_clock::time_point start,
    const std::optional<std::chrono::steady_clock::time_point>& pipeline_start,
    const std::optional<std::chrono::steady_clock::time_point>& dispatch_end,
    std::chrono::steady_clock::time_point loop_exit) {
    static const bool installed = [] {
        prosper::gpu::set_ordered_operation_observer(&on_ordered_operation);
        std::atexit([] { print_report("exit"); });
        return true;
    }();
    (void)installed;
    const auto span_ms = [](auto from, auto to) {
        return std::max(0.0, std::chrono::duration<double, std::milli>(to - from).count());
    };
    DispatchTimes times;
    if (pipeline_start) times.setup_ms = span_ms(start, *pipeline_start);
    if (pipeline_start && dispatch_end) times.gpu_ms = span_ms(*pipeline_start, *dispatch_end);
    if (dispatch_end) times.writeback_ms = span_ms(*dispatch_end, loop_exit);
    observer().dispatch(item.submit_no, dispatch_binding_ranges(buffers, images), times);
    static std::atomic<uint64_t> calls{0};
    if (calls.fetch_add(1, std::memory_order_relaxed) % 256 == 255) print_report("periodic");
}

}  // namespace prosper::frontend
