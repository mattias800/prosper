#include "host/guest_write_watch.hpp"

#include <atomic>
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
std::atomic<uint32_t> trap_count{0};
volatile sig_atomic_t post_store_marker = 0;
volatile sig_atomic_t marker_when_first_completed = -1;

void contention_hook() {
    contention_hook_entered.store(true, std::memory_order_release);
    while (!contention_released.load(std::memory_order_acquire)) sched_yield();
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
        if (action == prosper::host::GuestDmemWriteTraceStepAction::LockTimeout) _exit(92);
        if (action == prosper::host::GuestDmemWriteTraceStepAction::Complete) {
            if (event.ordinal == 1) marker_when_first_completed = post_store_marker;
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

    const prosper::host::GuestDmemWriteTraceConfig invalid_config{
        chain, allocation_size, offset,
        static_cast<uint32_t>(prosper::host::kGuestDmemWriteTraceMaxBytes + 1), 3};
    CHECK(!prosper::host::guest_dmem_write_trace_configure(invalid_config) &&
              prosper::host::guest_dmem_write_trace_snapshot().invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::InvalidConfig,
          "invalid selector is fail-visible before any mapping is touched");

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

    const int64_t creator_tid = static_cast<int64_t>(syscall(SYS_gettid));
    prosper::host::guest_dmem_write_trace_set_contention_hook_for_test(&contention_hook);
    std::thread locker([] {
        while (!request_contention.load(std::memory_order_acquire)) sched_yield();
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
        store_byte(mapping + offset - 0x20, 0x77);   // same protected page, outside selected range
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

    const auto& first = result.events[0];
    CHECK(first.selected && first.coverage_valid_before && first.rearmed,
          "first selected write has continuous process-wide pre-fault coverage");
    CHECK(first.tid == writer_tid.load(std::memory_order_acquire) &&
              first.writer_rip != 0 && first.next_rip != 0,
          "cross-thread canary retains writer thread and instruction identity");
    CHECK(first.before[0] == 0x11 && first.after[0] == 0x42,
          "cross-thread canary retains exact before/after selected bytes");

    const auto& adjacent = result.events[1];
    CHECK(!adjacent.selected && adjacent.before[0] == 0x42 && adjacent.after[0] == 0x42,
          "same-page out-of-range store is distinguished from a selected-range writer");

    const auto& repeated = result.events[2];
    CHECK(repeated.selected && repeated.rearmed && repeated.before[0] == 0x42 &&
              repeated.before[1] == 0x11 && repeated.after[1] == 0x43,
          "a second selected store is captured after the real re-arm path");
    CHECK(!repeated.coverage_valid_before && result.coverage_gaps >= 3,
          "single-step RW windows invalidate later negative coverage instead of claiming completeness");

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
    const auto ambiguous = prosper::host::guest_dmem_write_trace_snapshot();
    CHECK(ambiguous.status == prosper::host::GuestDmemWriteTraceStatus::Invalid &&
              ambiguous.invalid_reason ==
                  prosper::host::GuestDmemWriteTraceInvalidReason::AmbiguousAllocation &&
              ambiguous.allocation_matches == 2,
          "a second dynamic allocation match invalidates the selector instead of silently picking one");
    prosper::host::guest_write_watch_set_fault_onstack(false);

    if (failures) return 1;
    std::puts("== PASS ==");
    return 0;
}
