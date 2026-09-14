// Watched CPU fills must restore only guest-authorized writes and retain clear semantics.
#include "fixtures/render_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "host/memory/guest_memory_map.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "shared/live/live_compute.hpp"
#include <array>
#include <csignal>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

using namespace prosper::gpu;
using namespace prosper::host;
namespace {
int failures = 0;
volatile sig_atomic_t faults = 0;
constexpr size_t bytes = 8192, words = bytes / 4;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
void watch_fault(int signal, siginfo_t* info, void* context) {
#if defined(__x86_64__)
    const bool write = context && (static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 2);
#else
    const bool write = context != nullptr;
#endif
    if (signal == SIGSEGV && write && info && info->si_addr &&
        guest_write_watch_handle_fault(reinterpret_cast<uint64_t>(info->si_addr))) {
        faults = faults + 1;
        return;
    }
    _exit(86);
}
ComputeItem fill(uint32_t* target, uint32_t records, std::array<uint32_t, 4> pattern) {
    static constexpr uint32_t code[]{0xd7460004u, 0x04010c08u, 0x7e000204u,
        0x7e020205u, 0x7e040206u, 0x7e060207u, 0xe01c2000u, 0x80000004u, 0xbf810000u};
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer; r.binding = 2; r.sgpr_base = 0;
    r.format = DataFormat::Uint32; r.num_components = 4; r.stride = 16;
    r.gpu_addr = reinterpret_cast<uint64_t>(target); r.size = bytes;
    ComputeItem item;
    item.resources = std::make_shared<ShaderResourceTable>();
    item.resources->resources = {r};
    item.user_sgprs = {uint32_t(r.gpu_addr), uint32_t(r.gpu_addr >> 32) | (16u << 16),
                      uint32_t(bytes / 16), 0, pattern[0], pattern[1], pattern[2], pattern[3]};
    item.cpu_fast_path = classify_compute_cpu_fast_path(code, std::size(code));
    item.code_addr = 0x3407f111;
    item.launch.local_x = 64; item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = (records + 63) / 64;
    item.launch.groups_y = item.launch.groups_z = 1;
    item.launch.threads_x = records; item.launch.threads_y = item.launch.threads_z = 1;
    item.recompile_config_available = true;
    auto& config = item.recompile_config;
    config.user_sgprs = item.user_sgprs;
    config.local_x = 64; config.local_y = config.local_z = 1;
    config.tgid_x_en = true; config.tgid_y_en = config.tgid_z_en = false;
    config.exact_thread_extent = true;
    config.threads_x = records; config.threads_y = config.threads_z = 1;
    item.spirv = recompile_compute(code, std::size(code), item.resources.get(), config);
    check(!item.spirv.empty() && item.cpu_fast_path == ComputeCpuFastPath::FillSgprUvec4,
          "fixture compiles and matches the exact production fill classifier");
    return item;
}
void execute_fill(const ComputeItem& item) {
    const uint64_t address = item.resources->resources[0].gpu_addr;
    const auto before = guest_write_watch_stats();
    const auto cpu = prosper::frontend::live_compute_cpu_fill_dispatches();
    unsigned notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t addr, uint64_t size, const char* origin) {
        check(std::strcmp(origin, "gpu-preserving") != 0, "fill remains an architectural overwrite");
        if (addr == address && size == bytes) ++notifications;
    });
    check(prosper::frontend::execute_live_compute_items({item}), "live fill completes");
    set_guest_gpu_write_observer({});
    check(prosper::frontend::live_compute_cpu_fill_dispatches() == cpu + 1, "watched fill uses CPU shortcut");
    check(guest_write_watch_stats().host_write_notifies == before.host_write_notifies + 1,
          "exactly one host-write preparation, including a permission retry");
    check(notifications == 1, "one ordinary notification covers the declared guest resource");
    check(faults == 0, "CPU store does not depend on fault recovery to prepare watched pages");
}
void refusal(const ComputeItem& item, bool preparation) {
    const auto before = guest_write_watch_stats();
    const auto cpu = prosper::frontend::live_compute_cpu_fill_dispatches();
    unsigned notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t, uint64_t, const char*) { ++notifications; });
    check(!prosper::frontend::live_compute_cpu_fast_path_for_test(item).has_value(),
          "unsafe CPU destination declines without invoking unrelated GPU fallback");
    set_guest_gpu_write_observer({});
    check(guest_write_watch_stats().host_write_notifies == before.host_write_notifies + preparation,
          "refusal prepares watches only after address, descriptor and extent guards");
    check(prosper::frontend::live_compute_cpu_fill_dispatches() == cpu && notifications == 0,
          "declined CPU fill publishes no architectural overwrite");
}
void depth_clear(uint32_t* guest) {
    // Guest metadata is already zero, while the renderer owns nonzero depth.
    // A byte-preserving shortcut would incorrectly keep rejecting this consumer.
    #include "../tools/boot_trace/refvs.inc"
    std::vector<uint32_t> vs(kRefVs, kRefVs + std::size(kRefVs));
    auto with_depth = [&](uint32_t bits) {
        auto module = vs;
        for (size_t i = 5; i < module.size(); i += module[i] >> 16)
            if ((module[i] & 0xffffu) == 43 && (module[i] >> 16) == 4 &&
                module[i + 1] == 6 && module[i + 2] == 0x25) module[i + 3] = bits;
        return module;
    };
    const uint32_t green_code[]{0x7e000280, 0x7e0202f2, 0x7e040280, 0x7e0602f2,
                               0xf800180f, 0x03020100, 0xbf810000};
    ResolvedPipelineState writer;
    writer.topology = 3; writer.color_write_mask = 15;
    writer.depth_test_enable = writer.depth_write_enable = true;
    writer.depth_compare_op = 7; writer.has_depth_clear = true; writer.depth_clear_value = 0;
    writer.depth_read_base = writer.depth_write_base = 0x3407d00000ull;
    writer.htile_data_base = reinterpret_cast<uint64_t>(guest);
    ResolvedPipelineState reader = writer;
    reader.depth_write_enable = false; reader.depth_compare_op = 6;
    prosper::test::BackendDraw w, r;
    w.vs = with_depth(0x3f400000); w.fs = recompile_fragment(green_code, std::size(green_code));
    w.ps = &writer; w.vcount = 3;
    r.vs = with_depth(0x3f000000); r.fs = w.fs; r.ps = &reader; r.vcount = 3;
    const auto produced = prosper::test::render_draws_rgba({w}, 64, 64, nullptr, nullptr, true);
    auto green_visible = [&]() {
        const auto pixels = prosper::test::render_draws_rgba({r}, 64, 64, nullptr, nullptr, true);
        return pixels.size() == 64 * 64 * 4 && pixels[(32 * 64 + 32) * 4 + 1] > 192;
    };
    check(produced.size() == 64 * 64 * 4 && !green_visible(),
          "rendered depth rejects a farther consumer before the clear");
    unsigned ordinary = 0, preserving = 0;
    set_guest_gpu_write_observer([&](uint64_t address, uint64_t size, const char* origin) {
        if (std::strcmp(origin, "gpu-preserving") == 0) ++preserving; else ++ordinary;
        const auto* previous = guest_gpu_write_origin();
        set_guest_gpu_write_origin(origin);
        prosper::test::invalidate_persistent_ds_guest_write(address, size);
        set_guest_gpu_write_origin(previous);
    });
    notify_guest_gpu_write_preserving_bytes(reinterpret_cast<uint64_t>(guest), bytes);
    check(preserving == 1 && !green_visible(), "byte-preserving control retains rendered depth");
    auto watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(guest), bytes);
    check(watch && !guest_writable(reinterpret_cast<uint64_t>(guest), bytes),
          "equal-value clear also starts on a protected destination");
    const auto cpu = prosper::frontend::live_compute_cpu_fill_dispatches();
    check(prosper::frontend::execute_live_compute_items({fill(guest, bytes / 16, {0, 0, 0, 0})}) &&
              prosper::frontend::live_compute_cpu_fill_dispatches() == cpu + 1,
          "equal-value metadata clear executes through watched CPU fill");
    check(ordinary == 1 && green_visible(), "logical zero-to-zero clear invalidates rendered depth");
    set_guest_gpu_write_observer({});
}
void exercise(uint32_t* guest, uint32_t* alias) {
    std::fill_n(guest, words, 0xccccccccu);
    auto item = fill(guest, 65, {1u, 0x80000000u, 0x7fc01234u, 0xffffffffu});
    execute_fill(item); // Unarmed control uses the existing path.
    std::fill_n(guest, words, 0xccccccccu);
    auto watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(guest), bytes);
    check(watch && watch.query() == GuestWriteWatchQuery::Unchanged &&
              !guest_writable(reinterpret_cast<uint64_t>(guest), bytes) &&
              !guest_writable(reinterpret_cast<uint64_t>(alias), bytes),
          "both physical aliases are really protected before the retry");
    execute_fill(item);
    for (size_t i = 0; i < words; ++i)
        check(guest[i] == (i < 65 * 4 ? item.user_sgprs[4 + i % 4] : 0xccccccccu),
              "exact nonzero pattern and partial-launch padding survive");
    check(watch.query() == GuestWriteWatchQuery::Dirty, "CPU fill invalidates the original watch");
    check(watch.rearm(), "watch rearms for an alias-directed zero fill");
    execute_fill(fill(alias, bytes / 16, {0, 0, 0, 0}));
    check(std::all_of(guest, guest + words, [](uint32_t x) { return x == 0; }),
          "zero fill through a watched alias updates the same physical storage");
    watch.reset();
    depth_clear(guest);
    watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(guest), bytes);

    check(watch.rearm(), "watch rearms for malformed-input refusals");
    auto bad = item;
    bad.resources = std::make_shared<ShaderResourceTable>(*item.resources);
    bad.resources->resources[0].size = 16;
    refusal(bad, false);
    bad = item; bad.launch.local_x = 32; refusal(bad, false);
    for (uint64_t address : {uint64_t{0}, uint64_t{1}, UINT64_MAX - 7}) {
        bad = item;
        bad.resources = std::make_shared<ShaderResourceTable>(*item.resources);
        bad.resources->resources[0].gpu_addr = address;
        refusal(bad, false);
    }
    check(watch.query() == GuestWriteWatchQuery::Unchanged,
          "rejected shape/address does not dirty the watched destination");

    // Hosted replay storage retains priority over an unusable advertised address.
    std::vector<uint32_t> hosted(65 * 4, 0);
    auto replay = item;
    replay.resources = std::make_shared<ShaderResourceTable>(*item.resources);
    auto& resource = replay.resources->resources[0];
    resource.gpu_addr = 1; resource.host_data = reinterpret_cast<uint8_t*>(hosted.data());
    resource.host_data_size = hosted.size() * 4;
    check(prosper::frontend::live_compute_cpu_fast_path_for_test(replay) == true,
          "complete hosted prefix remains writable without probing its advertised identity");
    for (size_t i = 0; i < hosted.size(); ++i)
        check(hosted[i] == item.user_sgprs[4 + i % 4], "hosted fill preserves every pattern bit");
    resource.host_data_size -= 1;
    refusal(replay, false);

    // Guest permission revocation must survive the attempted watch disarm. The sibling
    // stays writable; membership in that physical page must not authorize this alias.
    guest_write_watch_notify_direct_mapping_protection(reinterpret_cast<uint64_t>(alias), bytes, 1);
    check(mprotect(alias, bytes, PROT_READ) == 0, "guest alias is genuinely read-only");
    notify_guest_page_protection_changed();
    auto readonly = fill(alias, 65, {7, 7, 7, 7});
    refusal(readonly, true);
    check(!guest_writable(reinterpret_cast<uint64_t>(alias), bytes) &&
              std::all_of(guest, guest + words, [](uint32_t x) { return x == 0; }),
          "retry preserves genuine read-only permissions and bytes");
    watch.reset();
}
}
int main() {
    static uint8_t stack_memory[256 * 1024];
    stack_t stack{}; stack.ss_sp = stack_memory; stack.ss_size = sizeof(stack_memory);
    struct sigaction action{};
    action.sa_sigaction = watch_fault; action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    if (sigaltstack(&stack, nullptr) || sigaction(SIGSEGV, &action, nullptr)) return 2;
    guest_write_watch_set_fault_onstack(true);
    FILE* backing = std::tmpfile();
    if (!backing || ftruncate(fileno(backing), bytes)) return 2;
    void* guest = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(backing), 0);
    void* alias = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(backing), 0);
    if (guest == MAP_FAILED || alias == MAP_FAILED) return 2;
    constexpr uint64_t physical = 0x3407f0000ull;
    guest_write_watch_notify_direct_mapping_added(reinterpret_cast<uint64_t>(guest), bytes, physical, 3);
    guest_write_watch_notify_direct_mapping_added(reinterpret_cast<uint64_t>(alias), bytes, physical, 3);
    check(prosper::test::render_vk_ctx().ok, "renderer device initializes before live compute fallback");
    if (!prosper::test::render_vk_ctx().ok) return 2;
    exercise(static_cast<uint32_t*>(guest), static_cast<uint32_t*>(alias));
    guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(guest), bytes);
    guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(alias), bytes);
    munmap(guest, bytes); munmap(alias, bytes); std::fclose(backing);
    guest_write_watch_set_fault_onstack(false);
    std::printf("cpu_fill_watch_retry: %d failures\n", failures);
    return failures != 0;
}
