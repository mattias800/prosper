// Actual encoded PM4 -> CommandProcessor -> PendQueue/ledger/window controls. No Vulkan device
// is created: a validation-layer environment alone cannot turn these into rendering evidence.
#include "diagnostics/perf/perf_alarms.hpp"
#include "gpu/pm4/command_processor.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#include <io.h>
#define route_dup _dup
#define route_dup2 _dup2
#define route_close _close
#define route_fileno _fileno
#else
#include <unistd.h>
#define route_dup dup
#define route_dup2 dup2
#define route_close close
#define route_fileno fileno
#endif

using namespace prosper::gpu;
namespace perf = prosper::diagnostics::perf;
static unsigned& failure_count() {
    static unsigned value = 0;
    return value;
}
static void check(bool condition, const char* label) {
    std::printf("[%s] %s\n", condition ? "ok" : "FAIL", label);
    if (!condition) ++failure_count();
}
static void environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
using Counts = std::array<uint64_t, 6>;
static Counts counts() {
    constexpr std::array fields{perf::Counter::WaitRegMemDirectEvaluations,
        perf::Counter::WaitRegMemDirectCompareFalse, perf::Counter::WaitRegMemDirectUnreadable,
        perf::Counter::WaitRegMemDirectUnsupported, perf::Counter::WaitRegMemDirectFalseProceed,
        perf::Counter::WaitRegMemDirectFalseDefer};
    Counts result{};
    for (size_t i = 0; i < fields.size(); ++i)
        result[i] = perf::ledger().counters[static_cast<size_t>(fields[i])].load();
    return result;
}
static void delta(const Counts& before, Counts expected, bool disabled, const char* label) {
    const auto after = counts();
    if (disabled) expected.fill(0);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] += before[i];
    check(after == expected, label);
}
static uint32_t header(uint32_t total, uint32_t record) {
    return 0xc0000000u | ((total - 2u) << 16u) | (IT_NOP << 8u) |
           ((record & (R_NUM - 1u)) << 2u);
}
static void wait(std::vector<uint32_t>& stream, uint64_t address, uint64_t reference,
                 uint32_t function = 3u, uint64_t mask = UINT64_MAX) {
    stream.insert(stream.end(), {header(8, R_WAIT_MEM_64), uint32_t(address),
        uint32_t(address >> 32u), uint32_t(mask), uint32_t(mask >> 32u),
        uint32_t(reference), uint32_t(reference >> 32u), function});
}
static void release(std::vector<uint32_t>& stream, uint64_t address, uint64_t value) {
    stream.insert(stream.end(), {header(7, R_RELEASE_MEM), uint32_t(address),
        uint32_t(address >> 32u), 2u, uint32_t(value), uint32_t(value >> 32u), 0x2du});
}
static void dma(std::vector<uint32_t>& stream, uint64_t destination, uint64_t source,
                uint32_t bytes, bool immediate = false) {
    stream.insert(stream.end(), {header(7, R_DMA_DATA), uint32_t(destination),
        uint32_t(destination >> 32u), uint32_t(source), uint32_t(source >> 32u), bytes,
        immediate ? 0u : kDmaDataAddressSource});
}
static size_t fold(const std::vector<uint32_t>& stream, GpuState& state) {
    return run_command_buffer(stream.data(), stream.size(), state);
}
static std::string read(FILE* file) {
    std::rewind(file);
    std::string result;
    char buffer[4096];
    while (size_t n = std::fread(buffer, 1, sizeof buffer, file)) result.append(buffer, n);
    return result;
}
static std::string summary(perf::AlarmEngine& engine) {
    FILE* file = std::tmpfile();
    if (!file) { check(false, "summary scratch opens"); return {}; }
    engine.write_summary(file);
    std::fflush(file);
    auto result = read(file);
    std::fclose(file);
    return result;
}

int main(int argc, char** argv) {
    if (argc != 3 || (std::strcmp(argv[2], "proceed") && std::strcmp(argv[2], "defer"))) {
        std::fprintf(stderr, "usage: test_wait_regmem_packet_route <jsonl-output> <proceed|defer>\n");
        return 2;
    }
    const bool deferred = !std::strcmp(argv[2], "defer");
    environment("PROSPER_WAIT_DEFER", deferred ? "1" : "0");
    environment("PROSPER_EOP_WRITE_SYNC", nullptr);
    const bool disabled = !perf::enabled();
    std::printf("Actual CPU packet route; mode=%s observers=%s; Vulkan UNRUN\n",
                argv[2], disabled ? "disabled" : "enabled");
    std::unique_ptr<FILE, decltype(&std::fclose)> captured(std::tmpfile(), &std::fclose);
    if (!captured) return 2;
    const int saved = route_dup(route_fileno(stderr));
    if (saved < 0) return 2;
    if (route_dup2(route_fileno(captured.get()), route_fileno(stderr)) < 0) {
        route_close(saved);
        return 2;
    }

    // A real active submit holds the queued ReleaseMem private. Its overlay must satisfy the wait
    // without publishing the word. The false arm prints the retained raw/effective distinction.
    for (bool satisfied : {false, true}) {
        alignas(8) uint64_t label = 0;
        std::vector<uint32_t> stream;
        release(stream, reinterpret_cast<uint64_t>(&label), 2u);
        wait(stream, reinterpret_cast<uint64_t>(&label), satisfied ? 2u : 3u);
        GpuState state;
        const auto before = counts();
        prosper_gpu_submit_scope_begin();
        check(fold(stream, state) == 2u, "release/wait decode through actual packet route");
        check(label == 0, "pending release remains private inside its actual submit scope");
        check(prosper_gpu_submit_scope_active(), "overlay arm has a real active submit scope");
        delta(before, {1, satisfied ? 0u : 1u, 0, 0,
            !satisfied && !deferred ? 1u : 0u, !satisfied && deferred ? 1u : 0u}, disabled,
            satisfied ? "real pending overlay satisfies the direct predicate" :
                        "real pending overlay preserves false reason and actual policy action");
        check(last_fold_deferred() == (!satisfied && deferred),
              "observations preserve the selected queue pause policy");
        prosper_gpu_submit_scope_end();
        prosper_gpu_drain_completion_writes();
        check(label == 2u, "pending release publishes after the submit return checkpoint");
        // Recover even if a private overlay mutant incorrectly deferred the satisfying arm.
        // No local label may leave scope while a fixture-owned barrier still refers to it.
        label = satisfied ? 2u : 3u;
        const auto before_recheck = counts();
        flush_deferred_streams();
        check(!deferred_pending(), "overlay arm leaves no deferred stream");
        delta(before_recheck, {}, disabled, "deferred rechecks are outside the direct denominator");
    }

    // Numeric false action is tied to actual downstream effects, not only counter contents.
    {
        alignas(8) uint64_t condition = 0, downstream = 0;
        std::vector<uint32_t> stream;
        wait(stream, reinterpret_cast<uint64_t>(&condition), 1u);
        release(stream, reinterpret_cast<uint64_t>(&downstream), 9u);
        GpuState state;
        const auto before = counts();
        fold(stream, state);
        prosper_gpu_drain_completion_writes();
        check(downstream == (deferred ? 0u : 9u), "false action preserves actual downstream visibility");
        delta(before, {1, 1, 0, 0, deferred ? 0u : 1u, deferred ? 1u : 0u}, disabled,
              "actual false proceed/defer action is counted once");
        if (deferred) {
            std::vector<uint32_t> later;
            wait(later, reinterpret_cast<uint64_t>(&downstream), 9u);
            const auto gated_before = counts();
            GpuState gated;
            fold(later, gated);
            delta(gated_before, {}, disabled, "wait on an already gated address is not a direct evaluation");
        }
        condition = 1u;
        const auto recheck_before = counts();
        flush_deferred_streams();
        // Releasing the barrier now hands completion effects to the post-submit FIFO. Wait for
        // those writes before rechecking a later deferred stream that consumes the same label.
        prosper_gpu_drain_completion_writes();
        flush_deferred_streams();
        check(downstream == 9u && !deferred_pending(), "producer releases all downstream effects in order");
        delta(recheck_before, {}, disabled, "released deferred waits do not inflate direct incidence");
    }

    // Invalid packet/address gates are distinct from a valid packet whose address is unreadable.
    {
        const auto before = counts();
        GpuState state;
        for (uint64_t address : {uint64_t{0}, uint64_t{0x1001}}) {
            std::vector<uint32_t> stream;
            wait(stream, address, 1u);
            fold(stream, state);
        }
        std::vector<uint32_t> short_packet{header(4, R_WAIT_MEM_64), 0x1000u, 0, 0};
        fold(short_packet, state);
        delta(before, {}, disabled, "invalid, zero and unaligned waits remain outside direct incidence");
        check(!deferred_pending(), "invalid wait gates do not create a deferred stream");
    }
    {
        alignas(8) uint64_t condition = 1u;
        GpuState state;
        std::vector<uint32_t> unreadable, unsupported;
        wait(unreadable, 0x1000u, 0u, 0u);
        const auto before_unreadable = counts();
        fold(unreadable, state);
        delta(before_unreadable, {1, 0, 1, 0, deferred ? 0u : 1u, deferred ? 1u : 0u}, disabled,
              "actual unreadable route refuses before comparison function zero");
        wait(unsupported, reinterpret_cast<uint64_t>(&condition), 1u, 7u);
        const auto before_unsupported = counts();
        fold(unsupported, state);
        delta(before_unsupported, {1, 0, 0, 1, deferred ? 0u : 1u, deferred ? 1u : 0u}, disabled,
              "actual readable unsupported route has its own primary reason");
        if (deferred) {
            // These intentionally impossible barriers exercise the existing bounded liveness
            // backstop; its age starts at the FIRST deferred recheck, not at direct folding.
            // The unreadable barrier expires immediately; the readable unsupported barrier
            // records its first blocked time and remains queued until a later timed recheck.
            const auto recheck_before = counts();
            flush_deferred_streams();
            check(deferred_pending(), "readable unsupported fixture barrier arms the existing watchdog");
            std::this_thread::sleep_for(std::chrono::milliseconds(1100));
            flush_deferred_streams();
            delta(recheck_before, {}, disabled, "timeout rechecks remain outside direct incidence");
            check(!deferred_pending(), "impossible fixture barriers are reaped after the existing timeout");
        }
    }

    // Host-stack VA disjointness cannot prove physical topology. Preserve the existing fail-closed
    // result explicitly rather than mistaking this arm for a satisfying retained-effect overlay.
    {
        alignas(8) uint64_t source = 0x11223344u, destination = 0, label = 0;
        std::vector<uint32_t> stream;
        dma(stream, reinterpret_cast<uint64_t>(&destination), reinterpret_cast<uint64_t>(&source), 8u);
        dma(stream, reinterpret_cast<uint64_t>(&label), 1u, 8u, true);
        wait(stream, reinterpret_cast<uint64_t>(&label), 0x100000001ull);
        GpuState state;
        const auto before = counts();
        check(prosper::guest_memory_topology_relation(reinterpret_cast<uint64_t>(&destination), 8u,
                  reinterpret_cast<uint64_t>(&label), 8u) ==
                  prosper::GuestMemoryTopologyRelation::Unknown,
              "unregistered retained DMA and label topology is explicitly unknown");
        check(fold(stream, state) == 3u && state.dma_execution_rejected,
              "unknown-topology retained-effect wait preserves the original rejection");
        delta(before, {}, disabled, "unknown-topology retained rejection is not direct incidence");
        check(!deferred_pending(), "unknown-topology arm leaves no deferred stream");
    }

    // Use real kernel direct-memory registration to prove the copy destination is physically
    // disjoint from the waited label. This reaches the actual retained immediate-effect overlay.
    {
        prosper::register_builtin_hle();
        const auto allocate = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
        const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
        const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
        const auto release_mapping = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
        constexpr uint64_t owned_length = 0x10000u;
        constexpr uint64_t owned_budget_end = 16ull * 1024u * 1024u * 1024u;
        uint64_t owned_physical = 0, owned_view = 0;
        const bool routes = allocate && map && unmap && release_mapping;
        check(routes, "retained-effect fixture resolves real memory HLE routes");
        const bool allocated = routes && allocate(0, owned_budget_end, owned_length, owned_length,
            0, reinterpret_cast<uint64_t>(&owned_physical)) == 0;
        const bool mapped = allocated && map(reinterpret_cast<uint64_t>(&owned_view), owned_length,
            0x2u, 0, owned_physical, owned_length) == 0 && owned_view;
        check(mapped, "retained-effect fixture maps authoritative direct-memory topology");
        if (mapped) for (bool accepted : {false, true}) {
            auto* source = reinterpret_cast<uint64_t*>(owned_view);
            auto* destination = reinterpret_cast<uint64_t*>(owned_view + 0x100u);
            auto* label = reinterpret_cast<uint64_t*>(owned_view + 0x200u);
            *source = 0x11223344u;
            *destination = *label = 0;
            check(prosper::guest_memory_topology_relation(reinterpret_cast<uint64_t>(destination),
                      8u, reinterpret_cast<uint64_t>(label), 8u) ==
                      prosper::GuestMemoryTopologyRelation::Disjoint,
                  "registered retained copy and waited label are physically disjoint");
            std::vector<uint32_t> stream;
            dma(stream, reinterpret_cast<uint64_t>(destination), reinterpret_cast<uint64_t>(source), 8u);
            dma(stream, reinterpret_cast<uint64_t>(label), 1u, 8u, true);
            wait(stream, reinterpret_cast<uint64_t>(label), accepted ? 0x100000001ull : 2u);
            GpuState state;
            const auto before = counts();
            check(fold(stream, state) == 3u, "retained DMA/effect/wait decode through actual packets");
            check(state.dma_execution_rejected == !accepted,
                  "retained-effect wait keeps its original acceptance/refusal result");
            check(*destination == 0 && *label == 0 && state.ordered_memory_effects.size() == 1u,
                  "retained route keeps its copy and immediate effect private during folding");
            delta(before, {}, disabled, "ordered retained-effect acceptance/refusal is not direct incidence");
            check(!deferred_pending(), "retained-effect arm leaves no deferred stream");
        }
        if (owned_view)
            check(unmap(owned_view, owned_length, 0, 0, 0, 0) == 0,
                  "retained-effect fixture unmaps its owned view after all folds");
        if (allocated)
            check(release_mapping(owned_physical, owned_length, 0, 0, 0, 0) == 0,
                  "retained-effect fixture releases its owned direct memory");
    }

    perf::EngineConfig config;
    config.log = nullptr;
    config.window_ns = 100;
    config.jsonl_path = argv[1];
    {
        perf::AlarmEngine engine(config);
        engine.on_flip(1000, perf::ledger(), 60);
        alignas(8) uint64_t condition = 0;
        std::vector<uint32_t> stream;
        wait(stream, reinterpret_cast<uint64_t>(&condition), 1u);
        const auto before = counts();
        for (unsigned i = 0; i < 1100u; ++i) {
            GpuState state;
            fold(stream, state);
            if (deferred) {
                condition = 1u;
                flush_deferred_streams();
                condition = 0;
            }
        }
        delta(before, {1100, 1100, 0, 0, deferred ? 0u : 1100u, deferred ? 1100u : 0u}, disabled,
              "suppressed warnings still count every actual direct evaluation and action");
        check(!deferred_pending(), "suppression arm reaps every deferred fixture barrier");
        engine.on_flip(1100, perf::ledger(), 60);
        const auto closed = summary(engine);
        check(closed.find(disabled ? "data=NO DATA evaluations=0" :
                                    "data=OBSERVED evaluations=1100 compare-false=1100") != std::string::npos,
              "actual closed window exports packet-route population with absence explicit");
        check(closed.find("completed-windows=1 snapshot=relaxed") != std::string::npos,
              "actual packet-route report keeps relaxed closed-window scope");
        condition = 1u;
        GpuState trailing;
        fold(stream, trailing);
        check(summary(engine) == closed, "unclosed trailing packet observation stays out of report");
    }
    FILE* json = std::fopen(argv[1], "r");
    check(json != nullptr, "actual route window JSONL opens");
    if (json) {
        const auto text = read(json);
        std::fclose(json);
        check(text.find(disabled ? "\"wait_regmem_direct_evaluations\":0" :
                                  "\"wait_regmem_direct_evaluations\":1100") != std::string::npos,
              "actual route JSONL retains direct population");
    }
    std::fflush(stderr);
    const int restored = route_dup2(saved, route_fileno(stderr));
    const int closed = route_close(saved);
    if (restored < 0 || closed < 0) return 2;
    const auto diagnostics = read(captured.get());
    captured.reset();
    std::fwrite(diagnostics.data(), 1, diagnostics.size(), stderr);
    size_t warning_lines = 0;
    for (size_t cursor = 0; (cursor = diagnostics.find("NOT satisfied at fold time:", cursor)) !=
            std::string::npos; ++cursor) ++warning_lines;
    check(warning_lines == 41u, "real warning cadence prints first forty and ordinal1024 only");
    check(diagnostics.find("decision-raw=0x0 decision-overlay=1 decision-effective=0x2") !=
              std::string::npos,
          "real warning retains raw and pending-overlay predicate values separately");
    check(diagnostics.find("decision-value-valid=0 decision-raw=0x0 decision-overlay=0") !=
              std::string::npos,
          "real unreadable warning marks placeholder values invalid");
    check(diagnostics.find("decision-supported=0") != std::string::npos,
          "real unsupported warning retains comparison support");
    check(!deferred_pending() && !prosper_gpu_submit_scope_active(),
          "actual route fixture exits with no deferred streams or submit scope");
    std::printf("== %s: %u failures ==\n", failure_count() ? "FAIL" : "PASS", failure_count());
    return failure_count() ? 1 : 0;
}
