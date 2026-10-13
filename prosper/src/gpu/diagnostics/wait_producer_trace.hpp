#pragma once
// Bounded pending-writer attribution. These identities observe an existing operation; they never
// satisfy a wait, publish a write, or infer a queue from adjacent log lines or worker context.
#include "gpu/pm4/pm4_decode.hpp"
#include "gpu/pm4/wait_regmem_sample.hpp"
#include <cstdint>
#include <memory>

namespace prosper::gpu {

struct WaitTraceSubmitIdentity {
    // Queue is the actual u32 submit operand, not a proven queue-generation identifier.
    uint8_t origin = 0;
    bool queue_known = false;
    uint32_t queue = 0;
    bool submit_known = false;
    uint64_t submit = 0;
};
struct PendingWriterIdentity {
    // Unique inside this process's bounded capture, including repeated packets/queue handles.
    uint64_t occurrence = 0;
    WaitTraceSubmitIdentity submit;
    uint64_t packet = 0, stream_order = 0, captured_us = 0;
    bool packet_known = false, order_known = false;
    uint32_t kind = 0;
    uint64_t address = 0, bytes = 0, payload = 0;
    bool payload_known = false;
};
using PendingWriterTrace = std::unique_ptr<const PendingWriterIdentity>;

bool wait_producer_trace_enabled();   // Boot-pinned diagnostic, not a guest-behavior selector.
PendingWriterTrace capture_pending_writer_identity(const Pm4Command& command);
void observe_wait_producer_decision(const Pm4Command& command,
                                    const WaitRegMemPredicateSample& decision,
                                    const PendingWriterIdentity* producer);
void observe_pending_writer_store(const Pm4Command& command, uint64_t address, uint64_t bytes);

struct WaitProducerTraceCounters {
    uint64_t producers = 0, missing_producers = 0, decisions = 0, stores = 0;
    uint64_t labels = 0, label_overflow = 0, records = 0, record_overflow = 0;
};
WaitProducerTraceCounters wait_producer_trace_counters();

// Only the actual HLE submit boundary supplies this context. Direct folds retain unknown identity.
class WaitTraceSubmitScope {
public:
    explicit WaitTraceSubmitScope(WaitTraceSubmitIdentity identity);
    ~WaitTraceSubmitScope();
    WaitTraceSubmitScope(const WaitTraceSubmitScope&) = delete;
    WaitTraceSubmitScope& operator=(const WaitTraceSubmitScope&) = delete;

private:
    bool active_ = false;
    WaitTraceSubmitIdentity previous_;
};

// The detached worker carries the enqueue-owned identity into an attempted application. A store
// record requires the existing after-write notification; a skipped/unmapped effect supplies none.
class PendingWriterTraceApplyScope {
public:
    PendingWriterTraceApplyScope(const PendingWriterIdentity* producer, const Pm4Command& command);
    ~PendingWriterTraceApplyScope();
    PendingWriterTraceApplyScope(const PendingWriterTraceApplyScope&) = delete;
    PendingWriterTraceApplyScope& operator=(const PendingWriterTraceApplyScope&) = delete;

private:
    bool active_ = false;
    const PendingWriterIdentity* previous_ = nullptr;
    const Pm4Command* previous_command_ = nullptr;
};

} // namespace prosper::gpu
