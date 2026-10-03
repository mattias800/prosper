#include "hle/kernel/apr_submission.hpp"
#include "hle/kernel/sce_errno.hpp"
#include "host/memory/guest_memory_copy.hpp"
#include "host/platform/immortal.hpp"
#include <atomic>
#include <mutex>
#include <unordered_map>

namespace prosper {
namespace {
struct Execution { uint32_t pending = 0, failure = 0; };
struct ExecutionState {
    std::mutex mutex;
    std::unordered_map<uint64_t, Execution> batches;
    bool capacity_exhausted = false;
};
Immortal<ExecutionState> execution_state;
static_assert(std::is_trivially_destructible_v<decltype(execution_state)>);
constexpr size_t max_execution_states = 4096;
// Every minted handle names an already-completed eager submission. Keeping the contiguous
// completed range permits repeat waits without retaining one allocation per historical submit.
// These are opaque HLE handles, not an inferred console bit encoding. Never wrap or reuse them.
std::atomic<uint64_t> next_completed_id{1};
constexpr uint32_t refused = hle::sce_kernel_error(hle::FreeBsdErrno::EInval);
}

void apr_execution_reset(uint64_t cb) {
    std::lock_guard lock(execution_state->mutex);
    auto it = execution_state->batches.find(cb);
    // A rewind cannot complete a read still executing on another thread.
    if (it != execution_state->batches.end() && !it->second.pending) execution_state->batches.erase(it);
}

AprReadExecution::AprReadExecution(uint64_t cb) : cb_(cb), status_(refused) {
    std::lock_guard lock(execution_state->mutex);
    if (execution_state->batches.size() >= max_execution_states && !execution_state->batches.count(cb_)) {
        // Refuse further untracked batches rather than evicting a real failure or claiming
        // completion after dropping its pending state. Successful batches normally erase here.
        execution_state->capacity_exhausted = true;
        return;
    }
    tracked_ = true;
    ++execution_state->batches[cb_].pending;
}
AprReadExecution::~AprReadExecution() {
    if (!tracked_) return;
    std::lock_guard lock(execution_state->mutex);
    auto& state = execution_state->batches.at(cb_);
    --state.pending;
    if (!state.failure && status_) state.failure = status_;
    if (!state.pending && !state.failure) execution_state->batches.erase(cb_);
}
uint64_t AprReadExecution::finish(uint64_t status) {
    status_ = static_cast<uint32_t>(status);
    return tracked_ ? status : hle::sce_kernel_error(hle::FreeBsdErrno::ENoMem);
}
uint32_t apr_execution_status(uint64_t cb) {
    std::lock_guard lock(execution_state->mutex);
    const auto it = execution_state->batches.find(cb);
    if (it == execution_state->batches.end()) return execution_state->capacity_exhausted
        ? hle::sce_kernel_error(hle::FreeBsdErrno::ENoMem) : 0;
    if (it->second.pending) return hle::sce_kernel_error(hle::FreeBsdErrno::EBusy);
    return it->second.failure;
}

uint64_t apr_publish_completed_submit(uint64_t result, uint64_t id_slot, uint32_t* id) {
    // Original callers read status32 and failing-offset32 from result, then ID32 from a
    // distinct slot. On success the failing offset is zero. Failed read builders return their
    // established SCE error before reaching this function: no APR-engine error enum is guessed.
    const uint32_t completed_result[2] = {0, 0};
    if (result && !host::guest_write_exact(result, completed_result, sizeof completed_result))
        return hle::sce_kernel_error(hle::FreeBsdErrno::EFault);
    uint64_t next = next_completed_id.load(std::memory_order_acquire);
    do {
        if (next > UINT32_MAX) return hle::sce_kernel_error(hle::FreeBsdErrno::ENoSpc);
    } while (!next_completed_id.compare_exchange_weak(next, next + 1,
                std::memory_order_acq_rel, std::memory_order_acquire));
    const uint32_t submitted_id = static_cast<uint32_t>(next);
    if (id_slot && !host::guest_write_exact(id_slot, &submitted_id, sizeof submitted_id))
        return hle::sce_kernel_error(hle::FreeBsdErrno::EFault);
    if (id) *id = submitted_id;
    return 0;
}
uint64_t apr_wait_completed_submit(uint32_t id) {
    if (id && id < next_completed_id.load(std::memory_order_acquire)) return 0;
    // The hardware's invalid-ID error and reuse policy remain unmeasured. EINVAL is the HLE
    // refusal policy for a handle we never issued, not a claim about the console namespace.
    return refused;
}
}
