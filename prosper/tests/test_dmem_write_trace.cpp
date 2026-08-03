#include "host/guest_write_watch.hpp"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;
#define CHECK(condition, message) do { \
    if (!(condition)) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; } \
} while (0)

std::atomic<int64_t> writer_tid{0};
std::atomic<bool> request_contention{false};
std::atomic<bool> contention_held{false};
std::atomic<bool> contention_hook_entered{false};
std::atomic<bool> contention_released{false};
std::atomic<bool> unrelated_lock_held{false};
std::atomic<bool> release_unrelated_lock{false};
std::atomic<bool> unrelated_contention_hook_entered{false};
std::atomic<bool> stepping_rearm_result{true};
std::atomic<bool> stepping_kernel_write_succeeded{false};
std::atomic<bool> pause_invalidation_step{false};
std::atomic<bool> invalidation_step_ready{false};
std::atomic<bool> release_invalidation_step{false};
std::atomic<bool> invalidation_canary_active{false};
std::atomic<int> last_step_action{-1};
std::atomic<uint32_t> trap_count{0};
volatile sig_atomic_t post_store_marker = 0;
volatile sig_atomic_t marker_when_first_completed = -1;

void contention_hook() {
    contention_hook_entered.store(true, std::memory_order_release);
    while (!contention_released.load(std::memory_order_acquire)) sched_yield();
}

void unrelated_contention_hook() {
    unrelated_contention_hook_entered.store(true, std::memory_order_release);
    // Let a mutant that enters the lock path terminate deterministically instead of exhausting the
    // full retry bound. The named assertion below still proves that production never invoked us.
    release_unrelated_lock.store(true, std::memory_order_release);
}

void trace_signal_handler(int sig, siginfo_t* info, void* raw_context) {
    auto* context = static_cast<ucontext_t*>(raw_context);
    const int64_t tid = static_cast<int64_t>(syscall(SYS_gettid));
    if (sig == SIGSEGV && info && info->si_addr &&
        (context->uc_mcontext.gregs[REG_ERR] & 2)) {
        const auto action = prosper::host::guest_write_watch_handle_fault_ex(
            reinterpret_cast<uint64_t>(info->si_addr),
            static_cast<uint64_t>(context->uc_mcontext.gregs[REG_RIP]), tid);
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep)
            context->uc_mcontext.gregs[REG_EFL] |= 0x100;
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep &&
            pause_invalidation_step.exchange(false, std::memory_order_acq_rel)) {
            invalidation_step_ready.store(true, std::memory_order_release);
            while (!release_invalidation_step.load(std::memory_order_acquire)) sched_yield();
        }
        if (action == prosper::host::GuestWriteWatchFaultAction::SingleStep &&
            !request_contention.exchange(true, std::memory_order_acq_rel)) {
            while (!contention_held.load(std::memory_order_acquire)) sched_yield();
        }
        if (action != prosper::host::GuestWriteWatchFaultAction::NotHandled) return;
    }
    if (sig == SIGTRAP && info && info->si_code == TRAP_TRACE) {
        trap_count.fetch_add(1, std::memory_order_relaxed);
        prosper::host::GuestDmemWriteTraceEvent event{};
        const auto action = prosper::host::guest_dmem_write_trace_complete_step(
            tid, static_cast<uint64_t>(context->uc_mcontext.gregs[REG_RIP]), event);
        last_step_action.store(static_cast<int>(action), std::memory_order_release);
        if (action == prosper::host::GuestDmemWriteTraceStepAction::LockTimeout) _exit(92);
        if (action == prosper::host::GuestDmemWriteTraceStepAction::Complete ||
            action == prosper::host::GuestDmemWriteTraceStepAction::CompleteInvalid) {
            if (action == prosper::host::GuestDmemWriteTraceStepAction::Complete &&
                event.ordinal == 1 && marker_when_first_completed == -1)
                marker_when_first_completed = post_store_marker;
            context->uc_mcontext.gregs[REG_EFL] &= ~static_cast<greg_t>(0x100);
            return;
        }
        // A targeted mutation that drops pending ownership must fail the named assertion below rather
        // than trapping forever. Production has no such compensation and would take its fatal path.
        if (action == prosper::host::GuestDmemWriteTraceStepAction::NotHandled &&
            invalidation_canary_active.load(std::memory_order_acquire)) {
            context->uc_mcontext.gregs[REG_EFL] &= ~static_cast<greg_t>(0x100);
            return;
        }
    }
    _exit(90);
}

void install_altstack() {
    // Intentionally leaked until thread exit: the kernel may still reference it until then.
    auto* storage = new std::vector<uint8_t>(128 * 1024);
    stack_t stack{};
    stack.ss_sp = storage->data();
    stack.ss_size = storage->size();
    if (sigaltstack(&stack, nullptr) != 0) _exit(91);
}

__attribute__((noinline)) void store_byte(volatile uint8_t* address, uint8_t value) {
    *address = value;
    __asm__ volatile("" ::: "memory");
}

__attribute__((noinline)) void exchange_byte(volatile uint8_t* address, uint8_t value) {
    __asm__ volatile("xchgb %0, (%1)" : "+q"(value) : "r"(address) : "memory");
}

// The marker store is the exact instruction after the protected selected-byte store. The first
// TRAP_TRACE must capture/re-arm before this instruction is allowed to execute.
__attribute__((noinline)) void store_byte_then_mark(volatile uint8_t* address, uint8_t value,
                                                   volatile sig_atomic_t* marker) {
    __asm__ volatile(
        "movb %b1, (%0)\n\t"
        "movl $1, (%2)\n\t"
        :
        : "r"(address), "q"(value), "r"(marker)
        : "memory");
}

__attribute__((noinline)) void store_qword_then_mark(volatile uint8_t* address, uint64_t value,
                                                     volatile sig_atomic_t* marker) {
    __asm__ volatile(
        "movq %1, (%0)\n\t"
        "movl $1, (%2)\n\t"
        :
        : "r"(address), "r"(value), "r"(marker)
        : "memory");
}

} // namespace

int main() {
    constexpr uint64_t page = 0x1000;
    constexpr uint64_t allocation_size = page * 3;
    constexpr uint64_t physical = 0x740000;
    constexpr uint64_t offset = page + 0x120;
    constexpr uint32_t bytes = 16;
    constexpr uint32_t chain = 7;

    struct sigaction action{};
    action.sa_sigaction = trace_signal_handler;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    CHECK(sigaction(SIGSEGV, &action, nullptr) == 0, "install SIGSEGV trace handler");
    CHECK(sigaction(SIGTRAP, &action, nullptr) == 0, "install SIGTRAP trace handler");
    install_altstack();
    prosper::host::guest_write_watch_set_fault_onstack(true);

    const uint8_t mov_imm8_mem[] = {0xc6, 0x00, 0x7f};
    const uint8_t mov_imm32_mem[] = {0xc7, 0x00, 1, 0, 0, 0};
    const uint8_t mov_imm16_mem[] = {0x66, 0xc7, 0x00, 1, 0};
    const uint8_t mov_imm64_sign_extended_mem[] = {0x48, 0xc7, 0x00, 1, 0, 0, 0};
    const uint8_t mov_register_only[] = {0x48, 0x89, 0xc0};
    const uint8_t c7_other_extension[] = {0xc7, 0x08, 1, 0, 0, 0};
    const uint8_t c6_other_extension[] = {0xc6, 0x38, 0x7f};
    const uint8_t legacy_prefix_after_rex[] = {0x48, 0x66, 0x89, 0x00};
    const uint8_t lock_prefixed_mov[] = {0xf0, 0x48, 0x89, 0x00};
    CHECK(prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
              mov_imm8_mem) == 1 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  mov_imm32_mem) == 4 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  mov_imm16_mem) == 2 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  mov_imm64_sign_extended_mem) == 8,
          "bounded store decoder accepts C6/C7 /0 memory forms with exact operand widths");
    CHECK(prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
              mov_register_only) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  c7_other_extension) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  c6_other_extension) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  legacy_prefix_after_rex) == 0 &&
              prosper::host::guest_dmem_write_trace_decode_contiguous_store_size(
                  lock_prefixed_mov) == 0,
          "bounded store decoder rejects register ModRM, other C6/C7 extensions, and ambiguous prefixes");

    CHECK(setenv("PROSPER_DMEM_CALLER", "1", 1) == 0 &&
              setenv("PROSPER_DMEM_WRITE_TRACE", "7:0x3000:2:0x1120:16", 1) == 0 &&
              setenv("PROSPER_DMEM_WRITE_TRACE_MAX_EVENTS", "3", 1) == 0,
          "set five-field occurrence-selector environment");
    prosper::host::guest_dmem_write_trace_init_from_environment();
    auto parsed = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(parsed.status == prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation &&
              parsed.config.allocation_occurrence == 2 && parsed.config.offset == offset &&
              parsed.config.size == bytes && parsed.config.max_events == 3,
          "five-field environment syntax retains explicit occurrence and bounds");
    CHECK(setenv("PROSPER_DMEM_WRITE_TRACE", "7:0x3000:0x1120:16", 1) == 0,
          "set legacy four-field unique-selector environment");
    prosper::host::guest_dmem_write_trace_init_from_environment();
    parsed = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(parsed.status == prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation &&
              parsed.config.allocation_occurrence == 0 && parsed.config.offset == offset,
          "four-field environment syntax preserves strict-uniqueness mode");
    unsetenv("PROSPER_DMEM_WRITE_TRACE");
    unsetenv("PROSPER_DMEM_WRITE_TRACE_MAX_EVENTS");
    unsetenv("PROSPER_DMEM_CALLER");

    const prosper::host::GuestDmemWriteTraceConfig invalid_config{
        chain, allocation_size, offset,
        static_cast<uint32_t>(prosper::host::kGuestDmemWriteTraceMaxBytes + 1), 3};
    CHECK(!prosper::host::guest_dmem_write_trace_configure(invalid_config) &&
              prosper::host::guest_dmem_write_trace_snapshot().invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::InvalidConfig,
          "invalid selector is fail-visible before any mapping is touched");

    auto invalid_occurrence_config = invalid_config;
    invalid_occurrence_config.size = bytes;
    invalid_occurrence_config.allocation_occurrence =
        prosper::host::kGuestDmemWriteTraceMaxAllocationOccurrence + 1;
    CHECK(!prosper::host::guest_dmem_write_trace_configure(invalid_occurrence_config) &&
              prosper::host::guest_dmem_write_trace_snapshot().invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::InvalidConfig,
          "allocation occurrence above the explicit bound is fail-visible");

    const prosper::host::GuestDmemWriteTraceConfig config{
        chain, allocation_size, offset, bytes, 3};
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "dynamic dmem writer trace configuration is accepted");
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
              prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation,
          "configured trace waits for its allocation identity");

    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain + 1);
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size - page, chain);
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().allocation_matches == 0,
          "wrong chain and wrong allocation size cannot move the selector lever");

    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
              prosper::host::GuestDmemWriteTraceStatus::WaitingMapping,
          "exact caller-chain and allocation size resolve before mapping");

    auto* mapping = static_cast<uint8_t*>(
        mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(mapping != MAP_FAILED, "map canary direct-memory range");
    if (mapping == MAP_FAILED) return 1;
    std::memset(mapping + offset, 0x11, bytes);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(mapping), allocation_size, physical, 0x3);

    auto armed = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(armed.status == prosper::host::GuestDmemWriteTraceStatus::Armed,
          "matching map creation arms the selected physical pages");
    CHECK(armed.mapping_matches == 1 && armed.target_addr ==
              reinterpret_cast<uint64_t>(mapping + offset),
          "dynamic selector resolves the mapping-relative virtual address");
    CHECK(armed.initial[0] == 0x11 && armed.initial[bytes - 1] == 0x11,
          "arm records the selected initial bytes");

    // Both the production dirty watch and the diagnostic own the same page protection. Releasing the
    // production owner must leave the page RO for the still-Armed trace. A kernel write gives a
    // deterministic protection probe without consuming the diagnostic's CPU-fault event.
    auto reset_overlap_watch = prosper::host::GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(mapping + offset), bytes);
    CHECK(static_cast<bool>(reset_overlap_watch),
          "production watch can share an already trace-armed page");
    reset_overlap_watch.reset();
    int reset_probe[2] = {-1, -1};
    CHECK(pipe(reset_probe) == 0, "production-reset protection probe pipe created");
    const uint8_t reset_payload = 0x5a;
    CHECK(write(reset_probe[1], &reset_payload, 1) == 1,
          "production-reset protection probe payload queued");
    errno = 0;
    const ssize_t reset_write = read(reset_probe[0], mapping + offset, 1);
    CHECK(reset_write < 0 && errno == EFAULT &&
              prosper::host::guest_dmem_write_trace_snapshot().status ==
                  prosper::host::GuestDmemWriteTraceStatus::Armed,
          "production reset preserves the diagnostic owner's real read-only protection");
    close(reset_probe[0]);
    close(reset_probe[1]);

    auto production_watch = prosper::host::GuestWriteWatch::create(
        reinterpret_cast<uint64_t>(mapping + offset), bytes);
    CHECK(static_cast<bool>(production_watch),
          "production overlap watch arms before the diagnostic step canary");

    const int64_t creator_tid = static_cast<int64_t>(syscall(SYS_gettid));
    prosper::host::guest_dmem_write_trace_set_contention_hook_for_test(
        &unrelated_contention_hook);
    std::thread unrelated_locker([] {
        prosper::host::guest_dmem_write_trace_lock_state_for_test();
        unrelated_lock_held.store(true, std::memory_order_release);
        while (!release_unrelated_lock.load(std::memory_order_acquire)) sched_yield();
        prosper::host::guest_dmem_write_trace_unlock_state_for_test();
        unrelated_lock_held.store(false, std::memory_order_release);
    });
    while (!unrelated_lock_held.load(std::memory_order_acquire)) sched_yield();
    prosper::host::GuestDmemWriteTraceEvent unrelated_event{};
    const auto unrelated_step = prosper::host::guest_dmem_write_trace_complete_step(
        creator_tid, 0x1234, unrelated_event);
    const bool unrelated_lock_still_held = unrelated_lock_held.load(std::memory_order_acquire);
    release_unrelated_lock.store(true, std::memory_order_release);
    unrelated_locker.join();
    CHECK(unrelated_step == prosper::host::GuestDmemWriteTraceStepAction::NotHandled &&
              !unrelated_contention_hook_entered.load(std::memory_order_acquire) &&
              unrelated_lock_still_held,
          "unrelated TRAP bypasses contended dmem state through the lock-free pending-TID gate");

    prosper::host::guest_dmem_write_trace_set_contention_hook_for_test(&contention_hook);
    std::thread locker([&] {
        while (!request_contention.load(std::memory_order_acquire)) sched_yield();
        stepping_rearm_result.store(production_watch.rearm(), std::memory_order_release);
        int step_probe[2] = {-1, -1};
        if (pipe(step_probe) == 0) {
            const uint8_t step_payload = 0x6b;
            if (write(step_probe[1], &step_payload, 1) == 1) {
                const ssize_t wrote = read(step_probe[0], mapping + offset - 0x40, 1);
                stepping_kernel_write_succeeded.store(wrote == 1, std::memory_order_release);
            }
            close(step_probe[0]);
            close(step_probe[1]);
        }
        prosper::host::guest_dmem_write_trace_lock_state_for_test();
        contention_held.store(true, std::memory_order_release);
        while (!contention_hook_entered.load(std::memory_order_acquire)) sched_yield();
        prosper::host::guest_dmem_write_trace_unlock_state_for_test();
        contention_released.store(true, std::memory_order_release);
    });
    std::thread writer([&] {
        install_altstack();
        writer_tid.store(static_cast<int64_t>(syscall(SYS_gettid)), std::memory_order_release);
        store_byte_then_mark(mapping + offset, 0x42, &post_store_marker);
        exchange_byte(mapping + offset - 0x20, 0x77); // unknown RMW span, outside selected range
        store_byte(mapping + offset + 1, 0x43);      // selected again; proves re-arm
        store_byte(mapping + offset - 0x21, 0x78);   // fourth fault exceeds bounded history
    });
    writer.join();
    locker.join();
    prosper::host::guest_dmem_write_trace_set_contention_hook_for_test(nullptr);

    const auto result = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(writer_tid.load(std::memory_order_acquire) != 0 &&
              writer_tid.load(std::memory_order_acquire) != creator_tid,
          "canary writer runs on a thread other than the mapping creator");
    CHECK(result.status == prosper::host::GuestDmemWriteTraceStatus::Overflow &&
              result.overflow_events == 1,
          "the first event beyond the configured bound is explicit overflow");
    CHECK(result.event_count == 3 && result.page_faults >= 3 &&
              result.completed_steps == 3 && result.rearms == 3,
          "fault, single-step, bounded history, and re-arm mechanisms all ran");
    CHECK(contention_hook_entered.load(std::memory_order_acquire) &&
              marker_when_first_completed == 0 && trap_count.load(std::memory_order_relaxed) == 3,
          "contended first TRAP captures and re-arms before the next guest instruction executes");
    CHECK(!stepping_rearm_result.load(std::memory_order_acquire) &&
              stepping_kernel_write_succeeded.load(std::memory_order_acquire) &&
              production_watch.query() == prosper::host::GuestWriteWatchQuery::Dirty,
          "production rearm is refused during the trace RW step and cannot re-protect it early");
    production_watch.reset();

    const auto& first = result.events[0];
    CHECK(first.selected && first.decoded_write_size == 1 &&
              first.coverage_valid_before && first.rearmed,
          "first selected write has continuous process-wide pre-fault coverage");
    CHECK(first.tid == writer_tid.load(std::memory_order_acquire) &&
              first.writer_rip != 0 && first.next_rip != 0,
          "cross-thread canary retains writer thread and instruction identity");
    CHECK(first.before[0] == 0x11 && first.after[0] == 0x42,
          "cross-thread canary retains exact before/after selected bytes");

    const auto& adjacent = result.events[1];
    CHECK(!adjacent.selected && adjacent.selection_uncertain &&
              adjacent.decoded_write_size == 0 &&
              adjacent.before[0] == 0x42 && adjacent.after[0] == 0x42 &&
              result.selection_uncertain_faults == 1,
          "unchanged off-range fault is explicit unknown because a same-value crossing store is possible");

    const auto& repeated = result.events[2];
    CHECK(repeated.selected && repeated.decoded_write_size == 1 && repeated.rearmed &&
              repeated.before[0] == 0x42 &&
              repeated.before[1] == 0x11 && repeated.after[1] == 0x43,
          "a second selected store is captured after the real re-arm path");
    CHECK(!repeated.coverage_valid_before && result.coverage_gaps >= 3,
          "single-step RW windows invalidate later negative coverage instead of claiming completeness");

    // Hold a real writer in the SIGSEGV handler after SingleStep publication. A sibling invalidation
    // must retain that TID's pending ownership until its store executes and TRAP_TRACE is consumed;
    // CompleteInvalid tells the signal caller to clear TF without re-arming stale topology.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for pending-step invalidation control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    volatile sig_atomic_t invalidation_marker = 0;
    pause_invalidation_step.store(true, std::memory_order_release);
    invalidation_step_ready.store(false, std::memory_order_release);
    release_invalidation_step.store(false, std::memory_order_release);
    invalidation_canary_active.store(true, std::memory_order_release);
    last_step_action.store(-1, std::memory_order_release);
    std::thread invalidated_writer([&] {
        install_altstack();
        store_byte_then_mark(mapping + offset + 2, 0x66, &invalidation_marker);
    });
    while (!invalidation_step_ready.load(std::memory_order_acquire)) sched_yield();
    prosper::host::guest_write_watch_notify_physical_write(physical + offset + 2, 1);
    const auto invalidating = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(invalidating.status == prosper::host::GuestDmemWriteTraceStatus::Stepping &&
              invalidating.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::PhysicalWrite,
          "sibling invalidation retains the published stepping owner and explicit reason");
    release_invalidation_step.store(true, std::memory_order_release);
    invalidated_writer.join();
    invalidation_canary_active.store(false, std::memory_order_release);
    const auto invalidated = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(last_step_action.load(std::memory_order_acquire) == static_cast<int>(
              prosper::host::GuestDmemWriteTraceStepAction::CompleteInvalid) &&
              invalidation_marker == 1 &&
              invalidated.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              invalidated.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::PhysicalWrite &&
              invalidated.completed_steps == 1 && invalidated.event_count == 1 &&
              !invalidated.events[0].rearmed,
          "invalidated pending TF trap is consumed, cleared, retained, and returned explicitly invalid");

    // CR2 identifies the first faulting byte, not the qword's full span. Start four bytes before the
    // selected interval and change four selected bytes with one instruction; post-step evidence must
    // upgrade this crossing store from unknown to selected.
    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for crossing-store control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    volatile sig_atomic_t crossing_marker = 0;
    store_qword_then_mark(mapping + offset - 4, 0x24232221ddccbbaaull, &crossing_marker);
    const auto crossing = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(crossing_marker == 1 && crossing.status ==
              prosper::host::GuestDmemWriteTraceStatus::Armed &&
              crossing.event_count == 1 && crossing.selected_faults == 1 &&
              crossing.selection_uncertain_faults == 0 && crossing.events[0].selected &&
              crossing.events[0].decoded_write_size == 8 &&
              !crossing.events[0].selection_uncertain &&
              crossing.events[0].fault_addr == reinterpret_cast<uint64_t>(mapping + offset - 4) &&
              std::memcmp(crossing.events[0].before.data(), crossing.events[0].after.data(), bytes) != 0,
          "qword beginning before the range is selected when its completed store changes selected bytes");

    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for mapping-topology control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    CHECK(prosper::host::guest_dmem_write_trace_snapshot().status ==
              prosper::host::GuestDmemWriteTraceStatus::Armed,
          "existing complete mapping re-arms the reset selector");
    auto* second_alias = static_cast<uint8_t*>(
        mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(second_alias != MAP_FAILED, "map second-alias topology control");
    if (second_alias != MAP_FAILED) {
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(second_alias), allocation_size, physical, 0x3);
        const auto topology = prosper::host::guest_dmem_write_trace_snapshot();
        CHECK(topology.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
                  topology.invalid_reason ==
                      prosper::host::GuestDmemWriteTraceInvalidReason::MappingTopologyChanged,
              "a writable alias created after arm invalidates the trace instead of hiding its gap");
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(second_alias), allocation_size);
        munmap(second_alias, allocation_size);
    }

    prosper::host::guest_write_watch_notify_direct_mapping_removed(
        reinterpret_cast<uint64_t>(mapping), allocation_size);
    munmap(mapping, allocation_size);

    CHECK(prosper::host::guest_dmem_write_trace_configure(config),
          "selector can be reset for ambiguity control");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical + allocation_size, allocation_size, chain);
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical + allocation_size * 2, allocation_size, chain);
    const auto ambiguous = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(ambiguous.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              ambiguous.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::AmbiguousAllocation &&
              ambiguous.allocation_matches == 3 && ambiguous.selected_occurrence == 1,
          "default selector invalidates on occurrence 2 but keeps an honest observed census");

    auto occurrence_config = config;
    occurrence_config.allocation_occurrence = 2;
    CHECK(prosper::host::guest_dmem_write_trace_configure(occurrence_config),
          "explicit bounded occurrence selector is accepted");
    prosper::host::guest_dmem_write_trace_notify_allocation(
        physical, allocation_size, chain);
    const auto insufficient = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(insufficient.status == prosper::host::GuestDmemWriteTraceStatus::WaitingAllocation &&
              insufficient.config.allocation_occurrence == 2 &&
              insufficient.allocation_matches == 1 && insufficient.selected_occurrence == 0 &&
              insufficient.target_phys == 0,
          "insufficient occurrence stays waiting and proves requested/observed/selected state");

    auto* occurrence_two_mapping = static_cast<uint8_t*>(
        mmap(nullptr, allocation_size, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    CHECK(occurrence_two_mapping != MAP_FAILED, "map occurrence-2 selection control");
    if (occurrence_two_mapping != MAP_FAILED) {
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            reinterpret_cast<uint64_t>(occurrence_two_mapping), allocation_size,
            physical + allocation_size, 0x3);
        prosper::host::guest_dmem_write_trace_notify_allocation(
            physical + allocation_size, allocation_size, chain);
        prosper::host::guest_dmem_write_trace_notify_allocation(
            physical + allocation_size * 2, allocation_size, chain);
        const auto selected_two = prosper::host::guest_dmem_write_trace_snapshot();
        CHECK(selected_two.status == prosper::host::GuestDmemWriteTraceStatus::Armed &&
                  selected_two.config.allocation_occurrence == 2 &&
                  selected_two.allocation_matches == 3 && selected_two.selected_occurrence == 2 &&
                  selected_two.target_phys == physical + allocation_size + offset &&
                  selected_two.target_addr ==
                      reinterpret_cast<uint64_t>(occurrence_two_mapping + offset),
              "occurrence 2 moves the lever to the second allocation while observing later matches");
        prosper::host::guest_write_watch_notify_direct_mapping_removed(
            reinterpret_cast<uint64_t>(occurrence_two_mapping), allocation_size);
        munmap(occurrence_two_mapping, allocation_size);
    }
    prosper::host::guest_write_watch_set_fault_onstack(false);

    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}
