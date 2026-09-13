// #3407: exercise buffer decisions through the production backend, without COMPUTELOG's hashes.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "shared/live/live_compute.hpp"
#include "shared/perf/performance_capture.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <vector>
#if defined(__linux__)
#include <csignal>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>
#endif

using namespace prosper::gpu;
namespace {
constexpr uint32_t bytes = 2u << 20;
constexpr uint64_t code_address = 0x3407b00100ull;
constexpr uint64_t submit = 3407;
constexpr uint32_t fill[] = {0xb1b1b1b1u, 0xb2b2b2b2u, 0xb3b3b3b3u, 0xb4b4b4b4u};
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
void env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
std::string hex(uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}
#if defined(__linux__)
void fault(int signal, siginfo_t* info, void* context) {
#if defined(__x86_64__)
    const bool write_fault = context &&
        (static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 2) != 0;
#else
    const bool write_fault = context != nullptr;
#endif
    if (signal == SIGSEGV && write_fault && info && info->si_addr &&
        prosper::host::guest_write_watch_handle_fault(
            reinterpret_cast<uint64_t>(info->si_addr))) return;
    static constexpr char message[] = "buffer timing fixture: unexpected SIGSEGV\n";
    (void)!write(STDERR_FILENO, message, sizeof(message) - 1);
    _exit(86);
}
bool install_fault_handler() {
    static uint8_t stack_memory[256 * 1024];
    stack_t stack{};
    stack.ss_sp = stack_memory;
    stack.ss_size = sizeof(stack_memory);
    struct sigaction action{};
    action.sa_sigaction = fault;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    return sigaltstack(&stack, nullptr) == 0 && sigaction(SIGSEGV, &action, nullptr) == 0;
}
#endif
struct GuestBuffer {
    uint32_t* data = nullptr;
#if !defined(__linux__)
    std::vector<uint32_t> storage;
#endif
    explicit GuestBuffer(uint64_t backing_offset) {
#if defined(__linux__)
        void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) return;
        data = static_cast<uint32_t*>(mapping);
        prosper::host::guest_write_watch_notify_direct_mapping_added(
            address(), bytes, backing_offset, 0x3);
#else
        (void)backing_offset;
        storage.resize(bytes / sizeof(uint32_t));
        data = storage.data();
#endif
        std::fill_n(data, bytes / sizeof(uint32_t), 0xababababu);
    }
    ~GuestBuffer() {
#if defined(__linux__)
        if (data) {
            prosper::host::guest_write_watch_notify_direct_mapping_removed(address(), bytes);
            munmap(data, bytes);
        }
#endif
    }
    GuestBuffer(const GuestBuffer&) = delete;
    GuestBuffer& operator=(const GuestBuffer&) = delete;
    uint64_t address() const { return reinterpret_cast<uint64_t>(data); }
    bool correct() const {
        return std::equal(std::begin(fill), std::end(fill), data) &&
            std::all_of(data + 4, data + bytes / sizeof(uint32_t),
                        [](uint32_t word) { return word == 0xababababu; });
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: fixture MODE CAPTURE_DIRECTORY\n"); return 2; }
    const std::string mode = argv[1];
    const bool hosted_overlap = mode == "hosted-overlap";
    const bool promotion = mode == "promotion" || mode == "promotion-cpu";
    const bool cpu_result = mode == "promotion-cpu";
    if (mode != "selected" && mode != "wrong-code" && mode != "wrong-hash" &&
        mode != "disabled" && mode != "capture-idle" && mode != "capture-armed" &&
        !promotion && !hosted_overlap) return 2;
    env("PROSPER_COMPUTELOG", nullptr);
    env("PROSPER_COMPUTELOG_CODE", nullptr);
    env("PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB", cpu_result ? nullptr : "1");
    env("PROSPER_NO_PERSISTENT_COMPUTE_BUFFER_RESULTS", cpu_result ? "1" : nullptr);
    env("PROSPER_COMPUTE_WRITE_WATCH_PROMOTE_HITS", promotion ? nullptr : "1");
    env("PROSPER_NO_DISK_PIPELINE_CACHE", "1");
    env("PROSPER_COMPUTE_BUFFER_TIMING", mode == "disabled" ? nullptr : "1");
    env("PROSPER_COMPUTE_TIMING_CAPTURE_ONLY",
        mode == "capture-idle" || mode == "capture-armed" ? "1" : nullptr);
#if defined(__linux__)
    if (!install_fault_handler()) return 3;
    prosper::host::guest_write_watch_set_fault_onstack(true);
    constexpr bool watch = true;
#else
    constexpr bool watch = false;
#endif
    GuestBuffer first(0x10000000), second(0x10200000);
    if (!first.data || !second.data) return 3;

    // Same one-invocation stores as test_game_compute's two-target regression. Equal values
    // make the shared-address arm independent of store ordering between aliased descriptors.
    static constexpr uint32_t shader[] = {
        0x7e080280u, 0x7e000204u, 0x7e020205u, 0x7e040206u, 0x7e060207u,
        0xe01c2000u, 0x80000004u,
        0x7e0a020cu, 0x7e0c020du, 0x7e0e020eu, 0x7e10020fu,
        0xe01c2000u, 0x80020504u, 0xbf810000u,
    };
    ShaderResource a{};
    a.cls = ResourceClass::ConstantBuffer;
    a.format = DataFormat::Uint32;
    a.num_components = 4;
    a.binding = 0; a.sgpr_base = 0; a.stride = 16;
    a.gpu_addr = first.address(); a.size = bytes;
    ShaderResource b = a;
    b.binding = 1; b.sgpr_base = 8;
    auto resources = std::make_shared<ShaderResourceTable>();
    resources->resources = {a, b};
    ComputeShaderConfig config;
    config.local_x = 1;
    config.user_sgprs = {0, 0, 0, 0, fill[0], fill[1], fill[2], fill[3],
                        0, 0, 0, 0, fill[0], fill[1], fill[2], fill[3]};
    ComputeItem item;
    item.spirv = recompile_compute(shader, std::size(shader), resources.get(), config);
    if (item.spirv.empty()) return 4;
    item.resources = resources;
    item.user_sgprs = config.user_sgprs;
    item.code_addr = code_address;
    item.submit_no = submit;
    item.launch.threads_x = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_x = item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    const auto hash = gpu_capture_hash(reinterpret_cast<const uint8_t*>(item.spirv.data()),
                                       item.spirv.size() * sizeof(uint32_t));
    env("PROSPER_COMPUTE_TIMING_CODE", hex(code_address + (mode == "wrong-code")).c_str());
    env("PROSPER_COMPUTE_TIMING_HASH", hex(hash ^ (mode == "wrong-hash")).c_str());
    std::fprintf(stderr, "[buffer-timing-fixture] mode=%s code=%s hash=%s watch=%u bytes=%u\n",
                 mode.c_str(), hex(code_address).c_str(), hex(hash).c_str(), unsigned(watch), bytes);
    auto& capture = prosper::perf::interactive_performance_capture();
    if (mode == "capture-armed") {
        const auto now = std::chrono::steady_clock::now().time_since_epoch();
        const auto armed = capture.arm(argv[2], "TEST", "compute buffer fixture", "fixture",
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count(),
            std::chrono::system_clock::now());
        if (!armed.ok || !capture.detailed_timing_active()) return 5;
    }
    auto dispatch = [&](unsigned index) {
        ComputeItem next = item;
        next.dispatch_index = index;
        next.command_order = index * 10;
        return next;
    };
    unsigned notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t address, uint64_t length, const char*) {
        if ((address == first.address() || address == second.address()) && length == bytes)
            ++notifications;
    });
    if (hosted_overlap) {
        unsigned private_notifications = 0;
        bool private_large = false;
        set_guest_gpu_write_observer([&](uint64_t address, uint64_t length, const char*) {
            if (!address || ((address == first.address() || address == second.address()) &&
                             length == (private_large ? bytes : 16u)))
                ++private_notifications;
        });
        // #3619: distinct advertised identities can still publish into the same hosted bytes.
        // A small private output must not hide from B's earlier-write overlap check.
        for (bool large_is_private : {false, true}) {
            private_large = large_is_private;
            for (bool reversed : {false, true}) {
                for (bool overlap : {false, true}) {
                    std::fprintf(stderr, "[hosted-overlap-case] private-large=%u reversed=%u overlap=%u\n",
                                 unsigned(private_large), unsigned(reversed), unsigned(overlap));
                    auto table = std::make_shared<ShaderResourceTable>(*resources);
                    const unsigned small = reversed ? 1 : 0, large = 1 - small;
                    auto& sa = table->resources[small];
                    auto& lb = table->resources[large];
                    sa.gpu_addr = private_large ? second.address() : 0;
                    sa.size = 16;
                    sa.host_data = reinterpret_cast<uint8_t*>(second.data);
                    sa.host_data_size = 16;
                    lb.gpu_addr = private_large ? 0 : first.address();
                    lb.host_data = reinterpret_cast<uint8_t*>(first.data);
                    lb.host_data_size = bytes;
                    item.resources = table;
                    item.user_sgprs = config.user_sgprs;
                    // Warm the exact large-result baseline with a disjoint private output.
                    for (unsigned n = 0; n < 2; ++n)
                        check(prosper::frontend::execute_live_compute_items({dispatch(10 + n)}),
                              "hosted overlap baseline warmup executes");
                    check(first.correct(), "warm hosted buffer has exact baseline bytes");
                    const auto before = prosper::frontend::live_compute_buffer_gpu_result_skips();
                    if (overlap) sa.host_data = reinterpret_cast<uint8_t*>(first.data);
                    for (unsigned lane = 0; lane < 4; ++lane)
                        item.user_sgprs[small * 8 + 4 + lane] = fill[lane] ^ 0xffffffffu;
                    check(prosper::frontend::execute_live_compute_items({dispatch(12)}),
                          "private hosted overlap dispatch executes");
                    for (unsigned lane = 0; lane < 4; ++lane) {
                        const uint32_t expected = overlap && reversed ? fill[lane] ^ 0xffffffffu : fill[lane];
                        check(first.data[lane] == expected,
                              "hosted outputs preserve last-writer bytes");
                    }
                    check(std::all_of(first.data + 4, first.data + bytes / 4,
                                      [](uint32_t v) { return v == 0xababababu; }),
                          "hosted overlap leaves the complete large-buffer tail intact");
                    const auto after = prosper::frontend::live_compute_buffer_gpu_result_skips();
                    check(after == before + (overlap && !reversed ? 0u : 1u),
                          "only an earlier effective overlap vetoes the large GPU result skip");
                    // A subsequent writer must repair the final private mutation.
                    sa.host_data = reinterpret_cast<uint8_t*>(second.data);
                    item.user_sgprs = config.user_sgprs;
                    check(prosper::frontend::execute_live_compute_items({dispatch(13)}) && first.correct(),
                          "subsequent hosted writer repairs any last private output");
                }
            }
        }
        check(private_notifications == 0, "private hosted outputs do not announce architectural writes");
        set_guest_gpu_write_observer({});
        if (!failures) std::fprintf(stderr, "[buffer-timing-fixture] success mode=%s\n", mode.c_str());
        return failures ? 1 : 0;
    }
    if (promotion) {
        const auto initial_skips = prosper::frontend::live_compute_buffer_gpu_result_skips();
        for (unsigned index = 1; index <= 9; ++index) {
            const bool mutation = index == 3 || index == 8;
            const auto skips_before = prosper::frontend::live_compute_buffer_gpu_result_skips();
            if (mutation) first.data[0] ^= 0xffffffffu;
            auto next = dispatch(index);
            next.submit_no = submit + index;
            // Every call owns a distinct real submit journal. Reusing the item or incrementing
            // its submit number inside ONE scope would accidentally test journal skips instead.
            const auto result = execute_ordered_items(
                {{SubmitOperationKind::Dispatch, index, next.command_order}}, {}, {next}, {},
                [&](const std::vector<ComputeItem>& items) {
                    return prosper::frontend::execute_live_compute_items(items);
                }, 1, 1);
            check(result.compute_executed && first.correct(),
                  "promotion dispatch publishes exact result and preserves untouched tail");
            check(notifications == index,
                  "promotion dispatch publishes one owner despite aliased bindings");
            const auto skips_after = prosper::frontend::live_compute_buffer_gpu_result_skips();
            if (cpu_result || mutation || index == 1)
                check(skips_after == skips_before,
                      "cold/changed/CPU-only result cannot use GPU-identical shortcut");
            else
                check(skips_after == skips_before + 1,
                      "unchanged promotion dispatch proves its retained GPU result");
        }
        if (cpu_result)
            check(prosper::frontend::live_compute_buffer_gpu_result_skips() == initial_skips,
                  "CPU promotion arm never uses retained GPU-result comparison");
        set_guest_gpu_write_observer({});
        capture.cancel();
        if (!failures)
            std::fprintf(stderr, "[buffer-timing-fixture] success mode=%s\n", mode.c_str());
        return failures ? 1 : 0;
    }
    // A real submit scope supplies the journal; each callback still executes the Vulkan backend.
    const auto ordered = execute_ordered_items(
        {{SubmitOperationKind::Dispatch, 1, 10}, {SubmitOperationKind::Dispatch, 2, 20}},
        {}, {dispatch(1), dispatch(2)}, {},
        [&](const std::vector<ComputeItem>& items) {
            const bool ok = prosper::frontend::execute_live_compute_items(items);
            check(ok && first.correct(), "ordered dispatch publishes exact shader result");
            return ok;
        }, 1, 1);
    check(ordered.compute_executed, "ordered submit executed compute");
    check(notifications == 2, "one notification per owner, despite two alias bindings");
    auto run = [&](unsigned index) {
        check(prosper::frontend::execute_live_compute_items({dispatch(index)}),
              "direct dispatch executes");
        check(first.correct(), "direct dispatch preserves exact output and untouched tail");
    };
    run(3); // No submit journal: exact comparison establishes the write watches.
    run(4); // Clean watch, followed by exact retained GPU-result comparison.
    const auto skips = prosper::frontend::live_compute_buffer_gpu_result_skips();
    first.data[0] ^= 0xffffffffu; // Real external guest write, without a GPU notification.
    run(5); // Dirty first MiB must upload and repair the modified word.
    check(prosper::frontend::live_compute_buffer_gpu_result_skips() == skips,
          "guest mutation refuses stale retained-result skip");
    run(6);
    check(prosper::frontend::live_compute_buffer_gpu_result_skips() > skips,
          "unchanged result skip recovers after guest repair");
    auto distinct = std::make_shared<ShaderResourceTable>(*resources);
    distinct->resources[1].gpu_addr = second.address();
    item.resources = distinct;
    run(7);
    check(second.correct(), "separate second owner receives its own shader result");
    check(notifications == 8, "six alias dispatches plus two distinct owners publish once each");
    set_guest_gpu_write_observer({});
    capture.cancel();
    if (!failures) std::fprintf(stderr, "[buffer-timing-fixture] success mode=%s\n", mode.c_str());
    return failures ? 1 : 0;
}
