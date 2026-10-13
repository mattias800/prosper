#include "gpu/diagnostics/wait_producer_trace.hpp"
#include "diagnostics/diag_clock.hpp"
#include "diagnostics/diag_ratelimit.hpp"
#include "diagnostics/env_cache.hpp"
#include "host/platform/immortal.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>

namespace prosper::gpu {
namespace {
constexpr uint64_t kProducerLimit = 4096, kRecordLimit = 512;
constexpr size_t kLabelLimit = 64;
struct TraceState {
    std::mutex mutex;
    WaitProducerTraceCounters counters;
    std::array<uint64_t, kLabelLimit> labels{};
    uint64_t event = 0;
};
TraceState& trace_state() {
    static Immortal<TraceState> state;
    static_assert(std::is_trivially_destructible_v<decltype(state)>);
    static_assert(std::is_same_v<decltype(state), Immortal<TraceState>>);
    return *state;
}
thread_local WaitTraceSubmitIdentity submit_identity;
thread_local const PendingWriterIdentity* applying_producer = nullptr;
thread_local const Pm4Command* applying_command = nullptr;

bool watched(TraceState& state, uint64_t address) {
    return std::find(state.labels.begin(), state.labels.begin() + state.counters.labels, address) !=
           state.labels.begin() + state.counters.labels;
}
void note_limit(const char* name, uint64_t ordinal) {
    if (diag_should_print(ordinal, 1))
        std::fprintf(stderr, "[wait-producer] limit=%s omitted-ordinal=%llu data=TRUNCATED\n", name,
                     static_cast<unsigned long long>(ordinal));
}
std::array<char, 768> format_identity(const char* role, const PendingWriterIdentity* producer) {
    std::array<char, 768> out{};
    const PendingWriterIdentity empty;
    const auto& p = producer ? *producer : empty;
    const int written = std::snprintf(
        out.data(), out.size(),
        " %s-present=%u %s-occurrence=%llu %s-origin=%u %s-queue-known=%u %s-queue=0x%x"
        " %s-submit-known=%u %s-submit=%llu %s-packet-known=%u %s-packet=0x%llx"
        " %s-order-known=%u %s-order=%llu %s-captured-us=%llu %s-kind=%u"
        " %s-address=0x%llx %s-bytes=%llu"
        " %s-payload-known=%u %s-payload=0x%llx",
        role, producer ? 1u : 0u, role, static_cast<unsigned long long>(p.occurrence), role,
        p.submit.origin, role, p.submit.queue_known ? 1u : 0u, role, p.submit.queue, role,
        p.submit.submit_known ? 1u : 0u, role, static_cast<unsigned long long>(p.submit.submit),
        role, p.packet_known ? 1u : 0u, role, static_cast<unsigned long long>(p.packet), role,
        p.order_known ? 1u : 0u, role, static_cast<unsigned long long>(p.stream_order), role,
        static_cast<unsigned long long>(p.captured_us), role, p.kind, role,
        static_cast<unsigned long long>(p.address), role, static_cast<unsigned long long>(p.bytes),
        role, p.payload_known ? 1u : 0u, role, static_cast<unsigned long long>(p.payload));
    if (written < 0 || static_cast<size_t>(written) >= out.size())
        std::snprintf(out.data(), out.size(), " %s-data=TRUNCATED", role);
    return out;
}
uint64_t begin_record(TraceState& state) {
    const uint64_t ordinal = ++state.event;
    if (state.counters.records >= kRecordLimit) {
        note_limit("records", ++state.counters.record_overflow);
        return 0;
    }
    ++state.counters.records;
    return ordinal;
}
} // namespace

bool wait_producer_trace_enabled() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): boot-pinned, synchronized static; never rearmed.
    return PROSPER_ENV_ON("PROSPER_WAIT_PRODUCER_TRACE");
}

PendingWriterTrace capture_pending_writer_identity(const Pm4Command& command) {
    if (!wait_producer_trace_enabled()) return {};
    using K = Pm4Command::Kind;
    if (command.kind != K::ReleaseMem && command.kind != K::EventWrite &&
        command.kind != K::WriteData && command.kind != K::DmaData)
        return {};
    auto& state = trace_state();
    std::lock_guard lock(state.mutex);
    if (state.counters.producers >= kProducerLimit) {
        note_limit("producers", ++state.counters.missing_producers);
        return {};
    }
    // WD points to PendWrite's owned payload by this boundary. The tag keeps values and numeric
    // packet identity only, so later command-buffer recycling cannot change the observation.
    auto* producer = new (std::nothrow) PendingWriterIdentity;
    if (!producer) {
        note_limit("allocation", ++state.counters.missing_producers);
        return {};
    }
    producer->occurrence = ++state.counters.producers;
    producer->submit = submit_identity;
    const auto payload = reinterpret_cast<uint64_t>(command.payload);
    producer->packet_known = payload >= sizeof(uint32_t);
    producer->packet = producer->packet_known ? payload - sizeof(uint32_t) : 0;
    producer->order_known = command.stream_order != 0;
    producer->stream_order = command.stream_order;
    producer->captured_us = diagnostics::diag_now_us();
    producer->kind = static_cast<uint32_t>(command.kind);
    if (command.kind == K::ReleaseMem) {
        producer->address = command.rel_addr;
        producer->bytes = command.rel_data_sel == 1 ? 4 : 8;
        producer->payload_known =
            command.rel_value_valid && (command.rel_data_sel == 1 || command.rel_data_sel == 2);
        producer->payload =
            command.rel_data_sel == 1 ? uint32_t(command.rel_value) : command.rel_value;
    } else if (command.kind == K::EventWrite) {
        producer->address = command.event_addr;
        producer->bytes = 8;
    } else if (command.kind == K::WriteData) {
        producer->address = command.wd_addr;
        producer->bytes = uint64_t(command.wd_num) * 4;
        producer->payload_known =
            command.wd_valid && command.wd_data && command.wd_num && command.wd_num <= 2;
        if (producer->payload_known)
            std::memcpy(&producer->payload, command.wd_data, producer->bytes);
    } else if (command.kind == K::DmaData) {
        producer->address = command.dd_dst;
        producer->bytes = command.dd_bytes;
        producer->payload_known = command.dd_valid && command.dd_bytes && command.dd_bytes <= 8 &&
                                  !(command.dd_sels & kDmaDataAddressSource) &&
                                  command.dd_src <= UINT32_MAX;
        if (producer->payload_known) {
            const uint32_t word = uint32_t(command.dd_src);
            for (size_t offset = 0; offset < command.dd_bytes; offset += sizeof(word))
                std::memcpy(reinterpret_cast<uint8_t*>(&producer->payload) + offset, &word,
                            std::min<size_t>(sizeof(word), command.dd_bytes - offset));
        }
    }
    return PendingWriterTrace(producer);
}

void observe_wait_producer_decision(const Pm4Command& command,
                                    const WaitRegMemPredicateSample& decision,
                                    const PendingWriterIdentity* producer) {
    if (!wait_producer_trace_enabled()) return;
    auto& state = trace_state();
    std::lock_guard lock(state.mutex);
    ++state.counters.decisions;
    if (!watched(state, command.wm_addr)) {
        if (state.counters.labels == kLabelLimit) {
            note_limit("labels", ++state.counters.label_overflow);
            return;
        }
        state.labels[state.counters.labels++] = command.wm_addr;
    }
    const uint64_t ordinal = begin_record(state);
    if (!ordinal) return;
    PendingWriterIdentity consumer;
    consumer.submit = submit_identity;
    const uint64_t packet_payload = reinterpret_cast<uint64_t>(command.payload);
    consumer.packet_known = packet_payload >= sizeof(uint32_t);
    consumer.packet = consumer.packet_known ? packet_payload - sizeof(uint32_t) : 0;
    consumer.order_known = command.stream_order != 0;
    consumer.stream_order = command.stream_order;
    const auto consumer_text = format_identity("consumer", &consumer);
    const auto producer_text = format_identity("producer", producer);
    std::fprintf(stderr,
                 "[wait-producer] ord=%llu phase=wait ts-us=%llu address=0x%llx%s%s"
                 " raw-valid=%u raw=0x%llx overlay=%u effective=0x%llx masked=0x%llx"
                 " supported=%u satisfied=%u ref=0x%llx mask=0x%llx\n",
                 static_cast<unsigned long long>(ordinal),
                 static_cast<unsigned long long>(diagnostics::diag_now_us()),
                 static_cast<unsigned long long>(command.wm_addr), consumer_text.data(),
                 producer_text.data(), decision.readable ? 1u : 0u,
                 static_cast<unsigned long long>(decision.raw_value),
                 decision.overlay_touched ? 1u : 0u,
                 static_cast<unsigned long long>(decision.effective_value),
                 static_cast<unsigned long long>(decision.masked_value),
                 decision.comparison_supported ? 1u : 0u, decision.satisfied ? 1u : 0u,
                 static_cast<unsigned long long>(command.wm_ref),
                 static_cast<unsigned long long>(command.wm_mask));
}

void observe_pending_writer_store(const Pm4Command& command, uint64_t address, uint64_t bytes) {
    if (!wait_producer_trace_enabled() || applying_command != &command) return;
    auto& state = trace_state();
    std::lock_guard lock(state.mutex);
    ++state.counters.stores;
    if (!watched(state, address)) return;
    const uint64_t ordinal = begin_record(state);
    if (!ordinal) return;
    const auto producer_text = format_identity("producer", applying_producer);
    std::fprintf(stderr,
                 "[wait-producer] ord=%llu phase=store-notified ts-us=%llu address=0x%llx%s"
                 " bytes=%llu\n",
                 static_cast<unsigned long long>(ordinal),
                 static_cast<unsigned long long>(diagnostics::diag_now_us()),
                 static_cast<unsigned long long>(address), producer_text.data(),
                 static_cast<unsigned long long>(bytes));
}

WaitProducerTraceCounters wait_producer_trace_counters() {
    if (!wait_producer_trace_enabled()) return {};
    auto& state = trace_state();
    std::lock_guard lock(state.mutex);
    return state.counters;
}

WaitTraceSubmitScope::WaitTraceSubmitScope(WaitTraceSubmitIdentity identity) {
    if (!wait_producer_trace_enabled()) return;
    active_ = true;
    previous_ = submit_identity;
    submit_identity = identity;
}
WaitTraceSubmitScope::~WaitTraceSubmitScope() {
    if (active_) submit_identity = previous_;
}
PendingWriterTraceApplyScope::PendingWriterTraceApplyScope(const PendingWriterIdentity* producer,
                                                           const Pm4Command& command) {
    if (!wait_producer_trace_enabled()) return;
    active_ = true;
    previous_ = applying_producer;
    previous_command_ = applying_command;
    applying_command = &command;
    applying_producer = producer;
}
PendingWriterTraceApplyScope::~PendingWriterTraceApplyScope() {
    if (active_) {
        applying_producer = previous_;
        applying_command = previous_command_;
    }
}
}   // namespace prosper::gpu
