#include "diagnostics/readback_refusal.hpp"

#include "diagnostics/exit_reports.hpp"
#include "gpu/diagnostics/watch_list.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <vector>

namespace prosper::diagnostics::readback_refusal {
namespace {
constexpr size_t kMaxTargets = 8;
constexpr uint64_t kMaxRecords = 64;
struct State {
    std::array<uint64_t, kMaxTargets> targets{};
    size_t target_count = 0;
    std::atomic<uint64_t> seen{0};
    std::atomic<uint64_t> write_completed{0};
    std::atomic<uint64_t> write_failed{0};
    std::atomic<uint64_t> suppressed{0};
};
constinit std::atomic<State*> active{nullptr};

const char* bool_name(ObservedBool value) {
    switch (value) {
    case ObservedBool::Yes: return "yes";
    case ObservedBool::No: return "no";
    default: return "NOT_OBSERVED";
    }
}
const char* caller_name(Caller caller) {
    switch (caller) {
    case Caller::ComputeSnapshot: return "compute-snapshot";
    case Caller::LazySampled: return "lazy-sampled";
    default: return "UNKNOWN";
    }
}
} // namespace

void initialize() {
    // Called at frontend registration, never by lookup/readback or a counted submit hook.
    // Keep registered-report state alive through the repository's several exit paths.
    static State* const state = []() -> State* {
        const char* spec = std::getenv("PROSPER_RTT_READBACK_TRACE_TARGETS");
        if (!spec) return nullptr;
        std::vector<uint64_t> parsed;
        if (!prosper::gpu::parse_hex_watch_list(spec, parsed)) {
            std::fprintf(stderr, "[rtt-readback] selection=malformed armed=no "
                                 "expected=nonzero-0x-addresses history=UNKNOWN\n");
            return nullptr;
        }
        std::array<uint64_t, kMaxTargets> targets{};
        size_t count = 0;
        for (uint64_t address : parsed) {
            if (std::find(targets.begin(), targets.begin() + count, address) !=
                targets.begin() + count) continue;
            if (count == targets.size()) {
                std::fprintf(stderr, "[rtt-readback] selection=too-many-targets armed=no "
                                     "maximum=8 history=UNKNOWN\n");
                return nullptr;
            }
            targets[count++] = address;
        }
        auto* result = new State;
        result->targets = targets;
        result->target_count = count;
        register_exit_report([result] {
            const uint64_t seen = result->seen.load(std::memory_order_relaxed);
            std::fprintf(stderr, "[rtt-readback] RUN SNAPSHOT quiescence=unverified "
                                 "scope=selected-early-refusals seen=%llu fputs-accepted=%llu "
                                 "fputs-failed=%llu suppressed=%llu partition=unverified "
                                 "selected-request-population=NOT_OBSERVED "
                                 "history=UNKNOWN producer-tokens=NOT_OBSERVED\n",
                         static_cast<unsigned long long>(seen),
                         static_cast<unsigned long long>(result->write_completed.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(result->write_failed.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(result->suppressed.load(std::memory_order_relaxed)));
        });
        std::fprintf(stderr, "[rtt-readback] armed=yes targets=%zu maximum-records=64 "
                             "scope=compute-snapshot,lazy-sampled early-refusals-only "
                             "clock=steady-us history=UNKNOWN producer-tokens=NOT_OBSERVED\n", count);
        return result;
    }();
    active.store(state, std::memory_order_release);
}

bool selected(uint64_t address) {
    State* state = active.load(std::memory_order_acquire);
    if (!state) return false;
    return std::find(state->targets.begin(), state->targets.begin() + state->target_count,
                     address) != state->targets.begin() + state->target_count;
}

const char* reason_name(Reason reason) {
    switch (reason) {
    case Reason::GlobalUnproven: return "global-unproven";
    case Reason::LookupNotPerformed: return "lookup-not-performed";
    case Reason::ExactKeyAbsent: return "exact-key-absent";
    case Reason::PresentInvalid: return "present-invalid";
    case Reason::NoImage: return "no-image";
    case Reason::UndefinedLayout: return "undefined-layout";
    default: return "not-early-refused";
    }
}

size_t format_record(char* output, size_t capacity, const Record& record) {
    if (record.reason == Reason::NotRefused) {
        if (capacity) output[0] = '\0';
        return 0;
    }
    char canonical[24] = "NOT_OBSERVED";
    char layout[24] = "NOT_OBSERVED";
    char submit[24] = "UNKNOWN";
    char draw[24] = "UNKNOWN";
    char program[24] = "UNKNOWN";
    if (record.canonical_known)
        std::snprintf(canonical, sizeof(canonical), "%u", record.canonical_format);
    if (record.layout_known)
        std::snprintf(layout, sizeof(layout), "%u", record.layout);
    if (record.context.known & SubmitKnown)
        std::snprintf(submit, sizeof(submit), "%llu", static_cast<unsigned long long>(record.context.submit));
    if (record.context.known & DrawKnown)
        std::snprintf(draw, sizeof(draw), "%llu", static_cast<unsigned long long>(record.context.draw));
    if (record.context.known & ProgramKnown)
        std::snprintf(program, sizeof(program), "0x%llx", static_cast<unsigned long long>(record.context.program));
    const int required = std::snprintf(
        output, capacity,
        "[rtt-readback] source-us=%llu clock=steady-us reason=%s caller=%s "
        "key=0x%llx/%ux%u/requested-format=%u/canonical-format=%s/volume-depth=%u "
        "key-present=%s valid=%s image-present=%s layout=%s "
        "context-known=%u submit=%s draw=%s program=%s "
        "frontend-gpu-valid=%s frontend-cpu-vector-present=%s "
        "producer-tokens=NOT_OBSERVED pins=NOT_OBSERVED history=UNKNOWN "
        "gate=NOT_OBSERVED final-consumer-disposition=UNKNOWN\n",
        static_cast<unsigned long long>(record.source_us), reason_name(record.reason),
        caller_name(record.context.caller), static_cast<unsigned long long>(record.address),
        record.width, record.height, record.requested_format, canonical, record.volume_depth,
        bool_name(record.lookup.key_present), bool_name(record.lookup.valid),
        bool_name(record.image_present), layout, record.context.known,
        submit, draw, program,
        bool_name(record.context.frontend_gpu_valid), bool_name(record.context.frontend_cpu_pixels));
    return required > 0 ? static_cast<size_t>(required) : 0;
}

void emit(const Record& record, FILE* stream) {
    if (record.reason == Reason::NotRefused) return;
    State* state = active.load(std::memory_order_acquire);
    if (!state) return;
    const uint64_t ordinal = state->seen.fetch_add(1, std::memory_order_relaxed);
    if (ordinal == kMaxRecords)
        std::fprintf(stream, "[rtt-readback] record-limit=64 further-records=suppressed "
                             "history=UNKNOWN\n");
    if (ordinal >= kMaxRecords) {
        state->suppressed.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    char line[1024];
    format_record(line, sizeof(line), record);
    if (std::fputs(line, stream) >= 0)
        state->write_completed.fetch_add(1, std::memory_order_relaxed);
    else state->write_failed.fetch_add(1, std::memory_order_relaxed);
}

} // namespace prosper::diagnostics::readback_refusal
