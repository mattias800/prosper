// Reuse complete fill patterns only with current owner authority; retain logical clear semantics.
#include "fixtures/render_runner.h"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "host/memory/guest_memory_map.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "shared/live/live_compute.hpp"
#include "hle/dispatch/dispatch.hpp"
#include <array>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sys/ucontext.h>
#include <unistd.h>

using namespace prosper::gpu;
using namespace prosper::host;
namespace {
int failures = 0;
volatile sig_atomic_t faults = 0;
constexpr size_t bytes = 8192, words = bytes / 4;
constexpr uint64_t depth_plane_offset = 2u * 1024u * 1024u;
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
ComputeItem fill(uint32_t* target, uint32_t records, std::array<uint32_t, 4> pattern,
                 uint32_t extent = bytes, DataFormat format = DataFormat::Uint32) {
    static constexpr uint32_t code[]{0xd7460004u, 0x04010c08u, 0x7e000204u,
        0x7e020205u, 0x7e040206u, 0x7e060207u, 0xe01c2000u, 0x80000004u, 0xbf810000u};
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer; r.binding = 2; r.sgpr_base = 0;
    r.format = format; r.num_components = 4; r.stride = 16;
    r.gpu_addr = reinterpret_cast<uint64_t>(target); r.size = extent;
    ComputeItem item;
    item.resources = std::make_shared<ShaderResourceTable>();
    item.resources->resources = {r};
    item.user_sgprs = {uint32_t(r.gpu_addr), uint32_t(r.gpu_addr >> 32) | (16u << 16),
                      uint32_t(extent / 16), 0, pattern[0], pattern[1], pattern[2], pattern[3]};
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
enum class ClearCoverage { Full, PartialPrefix, UntouchedTail };
void depth_clear(uint32_t* guest, uint32_t extent = bytes, bool cached_fill = false,
                 ClearCoverage coverage = ClearCoverage::Full) {
    const bool untouched_tail = coverage == ClearCoverage::UntouchedTail;
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
    writer.depth_read_base = writer.depth_write_base =
        reinterpret_cast<uint64_t>(guest) + depth_plane_offset;
    writer.htile_data_base = reinterpret_cast<uint64_t>(guest) + (untouched_tail ? extent / 2 : 0);
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
    uint64_t clear_address = 0, clear_bytes = 0;
    set_guest_gpu_write_observer([&](uint64_t address, uint64_t size, const char* origin) {
        if (std::strcmp(origin, "gpu-preserving") == 0) ++preserving;
        else { ++ordinary; clear_address = address; clear_bytes = size; }
        const auto* previous = guest_gpu_write_origin();
        set_guest_gpu_write_origin(origin);
        prosper::test::invalidate_persistent_ds_guest_write(address, size);
        set_guest_gpu_write_origin(previous);
    });
    notify_guest_gpu_write_preserving_bytes(reinterpret_cast<uint64_t>(guest), extent);
    check(preserving == 1 && !green_visible(), "byte-preserving control retains rendered depth");
    auto watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(guest), extent);
    check(watch && !guest_writable(reinterpret_cast<uint64_t>(guest), extent),
          "equal-value clear also starts on a protected destination");
    const auto cpu = prosper::frontend::live_compute_cpu_fill_dispatches();
    const auto reused = prosper::frontend::live_compute_cached_fill_dispatches();
    const auto compared = prosper::frontend::live_compute_buffer_gpu_result_skips();
    const auto submitted = prosper::frontend::live_compute_queue_submit_attempts();
    const auto records = coverage == ClearCoverage::Full ? extent / 16 : 65u;
    check(prosper::frontend::execute_live_compute_items({fill(guest, records, {0, 0, 0, 0}, extent)}) &&
              prosper::frontend::live_compute_cpu_fill_dispatches() == cpu,
          "equal-value metadata clear executes through the selected path");
    check(prosper::frontend::live_compute_cached_fill_dispatches() == reused + cached_fill,
          "depth clear exercises the intended cached or GPU fallback path");
    const bool gpu_compare = !cached_fill && extent >= (1u << 20);
    check(prosper::frontend::live_compute_buffer_gpu_result_skips() == compared + gpu_compare,
          "GPU comparator and cached/host comparison depth cases are distinguished");
    check(prosper::frontend::live_compute_queue_submit_attempts() ==
              submitted + (!cached_fill), "depth clear has expected submission count");
    check(ordinary == 1 && clear_address == reinterpret_cast<uint64_t>(guest) &&
              clear_bytes == records * 16,
          "known fill forwards exactly its written range as an architectural clear");
    check(preserving == (coverage == ClearCoverage::Full ? 1u : 2u),
          "partial clear retains the ordinary full-binding unchanged notification");
    check(green_visible() == !untouched_tail,
          untouched_tail ? "partial clear preserves rendered depth in the untouched buffer tail"
                         : "logical zero-to-zero clear invalidates rendered depth");
    set_guest_gpu_write_observer({});
}
void cached_fills(uint32_t* guest, uint32_t* alias, uint32_t extent) {
    using namespace prosper::frontend;
    const std::array<uint32_t, 4> a{1u, 0x80000000u, 0x7fc01234u, 0xffffffffu};
    const std::array<uint32_t, 4> b{7u, 8u, 9u, 10u};
    auto full_a = fill(guest, extent / 16, a, extent);
    auto full_b = fill(guest, extent / 16, b, extent);
    std::fill_n(guest, extent / 4, 0xccccccccu);
    auto external_watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(guest), extent);
    check(bool(external_watch), "full-buffer fixture has a real guest watch");
    auto run = [&](const ComputeItem& item, bool expect_reuse, bool success = true) {
        check(external_watch.rearm(), "protect destination to exercise GPU/cache path");
        check(!guest_writable(reinterpret_cast<uint64_t>(guest), extent),
              "CPU store shortcut cannot explain cached-fill result");
        const auto reused = live_compute_cached_fill_dispatches();
        const auto submitted = live_compute_queue_submit_attempts();
        const auto cpu = live_compute_cpu_fill_dispatches();
        check(execute_live_compute_items({item}) == success, "full-buffer dispatch completion agrees");
        check(live_compute_cpu_fill_dispatches() == cpu, "cache test does not take CPU store shortcut");
        check(live_compute_cached_fill_dispatches() == reused + expect_reuse,
              "only a currently proven complete pattern can skip dispatch");
        check(live_compute_queue_submit_attempts() == submitted + !expect_reuse,
              "cached fill avoids a real queue submission");
    };
    auto equal = [&](const std::array<uint32_t, 4>& pattern) {
        for (size_t i = 0; i < extent / 4; ++i)
            if (guest[i] != pattern[i % 4]) return false;
        return true;
    };
    run(full_a, false);
    check(equal(a), "first completed fill publishes every exact pattern bit");
    run(full_a, false); // Exact source refresh establishes the owner's own watch.
    run(full_a, true);
    check(equal(a), "reused complete fill preserves all bytes");
    check(external_watch.query() == GuestWriteWatchQuery::Unchanged,
          "proven unchanged fill preserves guest-byte watch authority");

    run(full_b, false);
    check(equal(b), "changed requested pattern is executed");
    run(full_b, true);

    // A real guest write through the physical alias must revoke cached content proof.
    check(!guest_writable(reinterpret_cast<uint64_t>(alias), extent),
          "physical alias is write-protected before the external store");
    const auto previous_faults = faults;
    reinterpret_cast<volatile uint32_t*>(alias)[3] = 0xdeadbeefu;
    check(faults == previous_faults + 1, "alias mutation reaches the actual watch fault handler");
    run(full_b, false);
    check(equal(b), "fill repairs externally changed guest bytes");
    run(full_b, true);

    auto ordinary = full_a;
    ordinary.cpu_fast_path = ComputeCpuFastPath::None;
    run(ordinary, false);
    check(equal(a), "intervening ordinary GPU writer changed the owner");
    run(full_b, false);
    check(equal(b), "previous pattern proof cannot survive an arbitrary GPU writer");
    run(full_b, true);

    const auto partial_a = fill(guest, 65, a, extent);
    run(partial_a, false);
    bool prefix_and_tail = true;
    for (size_t i = 0; i < extent / 4; ++i)
        prefix_and_tail &= guest[i] == (i < 65 * 4 ? a[i % 4] : b[i % 4]);
    check(prefix_and_tail, "partial fill neither overwrites nor proves the padded tail");
    run(full_a, false);
    check(equal(a), "partial prefix cannot stamp the untouched tail as the same pattern");
    run(full_b, false);
    run(full_b, true);

    auto mismatched_extent = partial_a;
    mismatched_extent.launch = full_a.launch;
    run(mismatched_extent, false);
    run(full_a, false);
    check(equal(a), "full launch cannot hide a truncated emitted shader extent");
    run(full_a, true);

    live_compute_fail_next_buffer_readback_for_test();
    run(full_b, false, false);
    check(equal(a), "failed result publication leaves previous guest bytes intact");
    run(full_b, false);
    check(equal(b), "failed completed result must not authorize a later cached fill");
    run(full_b, true);

    auto zero = fill(guest, extent / 16, {0, 0, 0, 0}, extent);
    zero.cpu_fast_path = ComputeCpuFastPath::None;
    run(zero, false); // Equal GPU primary/baseline, with no uniform-fill proof.
    depth_clear(guest, extent, false); // GPU comparator reports unchanged.
    depth_clear(guest, extent, true);  // Existing complete pattern bypasses submission.
    depth_clear(guest, extent, false, ClearCoverage::UntouchedTail);
    depth_clear(guest, extent, false, ClearCoverage::PartialPrefix);
    depth_clear(guest);               // Small uncached result uses host memcmp.

    const std::array<uint32_t, 4> fp{0x7fc01234u, 0x80000000u, 0x7f800000u, 0xff800000u};
    auto float_fill = fill(guest, extent / 16, fp, extent, DataFormat::Float32);
    run(float_fill, false);
    check(equal(fp), "native float fill preserves NaN payload, signed zero and infinity bits");
    run(float_fill, true);
    const std::array<uint32_t, 4> si{0x80000000u, 0xffffffffu, 0x7fffffffu, 0u};
    auto signed_fill = fill(guest, extent / 16, si, extent, DataFormat::Sint32);
    run(signed_fill, false);
    check(equal(si), "native signed fill preserves every integer bit");
    run(signed_fill, true);
}
void overridden_fills(uint32_t* guest, uint32_t extent) {
    using namespace prosper::frontend;
    auto item = fill(guest, extent / 16, {1, 2, 3, 4}, extent);
    // The replacement writes literal zeros instead of the requested SGPR pattern.
    const uint32_t code[]{0xd7460004u, 0x04010c08u, 0x7e000280u,
        0x7e020280u, 0x7e040280u, 0x7e060280u, 0xe01c2000u, 0x80000004u, 0xbf810000u};
    const auto replacement = recompile_compute(code, std::size(code), item.resources.get(),
                                                item.recompile_config);
    check(!replacement.empty(), "replacement full writer compiles");
    const char* temporary = std::getenv("TMPDIR");
    const auto path = std::filesystem::path(temporary ? temporary : ".") /
        ("cached-fill-override-" + std::to_string(getpid()) + ".spv");
    {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(replacement.data()), replacement.size() * 4);
        check(bool(file), "replacement module is written completely");
    }
    check(setenv("PROSPER_LOAD_COMPUTE_SPIRV", ("0x3407f111:" + path.string()).c_str(), 1) == 0,
          "replacement is selected before the first compute dispatch");
    std::fill_n(guest, extent / 4, 0xccccccccu);
    auto watch = GuestWriteWatch::create(reinterpret_cast<uint64_t>(guest), extent);
    for (unsigned i = 0; i < 4; ++i) {
        check(watch.rearm(), "replacement destination remains watched");
        const auto reused = live_compute_cached_fill_dispatches();
        const auto submitted = live_compute_queue_submit_attempts();
        const auto cpu = live_compute_cpu_fill_dispatches();
        check(execute_live_compute_items({item}), "replacement dispatch completes");
        check(live_compute_cpu_fill_dispatches() == cpu &&
                  live_compute_cached_fill_dispatches() == reused &&
                  live_compute_queue_submit_attempts() == submitted + 1,
              "original shader pattern never authorizes reuse of replacement output");
        check(std::all_of(guest, guest + extent / 4, [](uint32_t value) { return value == 0; }),
              "actual replacement output remains authoritative");
    }
    item.code_addr += 0x100; // Same owner and pattern, outside the targeted override.
    check(watch.rearm(), "non-overridden retry also uses the watched path");
    const auto reused = live_compute_cached_fill_dispatches();
    const auto submitted = live_compute_queue_submit_attempts();
    check(execute_live_compute_items({item}) &&
              live_compute_cached_fill_dispatches() == reused &&
              live_compute_queue_submit_attempts() == submitted + 1,
          "an overridden result cannot leave a latent stamp of the original pattern");
    for (size_t i = 0; i < extent / 4; ++i)
        check(guest[i] == item.user_sgprs[4 + i % 4],
              "non-overridden retry writes the real requested pattern");
    check(watch.rearm(), "successful non-overridden publication can be reused");
    check(execute_live_compute_items({item}) &&
              live_compute_cached_fill_dispatches() == reused + 1 &&
              live_compute_queue_submit_attempts() == submitted + 1,
          "only successful original publication enables later cached reuse");
    std::filesystem::remove(path);
}
}
int main(int argc, char** argv) {
    const bool override_mode = argc > 1 && std::strcmp(argv[1], "--cached-override") == 0;
    const size_t mapping_bytes = 4u * 1024u * 1024u;
    const uint32_t work_bytes = 2u * 1024u * 1024u;
    static uint8_t stack_memory[256 * 1024];
    stack_t stack{}; stack.ss_sp = stack_memory; stack.ss_size = sizeof(stack_memory);
    struct sigaction action{};
    action.sa_sigaction = watch_fault; action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    if (sigaltstack(&stack, nullptr) || sigaction(SIGSEGV, &action, nullptr)) return 2;
    guest_write_watch_set_fault_onstack(true);
    prosper::register_builtin_hle();
    const auto allocate = prosper::Hle::lookup(prosper::nid_hash("sceKernelAllocateDirectMemory"));
    const auto map = prosper::Hle::lookup(prosper::nid_hash("sceKernelMapDirectMemory"));
    const auto unmap = prosper::Hle::lookup(prosper::nid_hash("sceKernelMunmap"));
    const auto release = prosper::Hle::lookup(prosper::nid_hash("sceKernelReleaseDirectMemory"));
    if (!allocate || !map || !unmap || !release) return 2;
    uint64_t physical = 0, guest_address = 0, alias_address = 0;
    if (allocate(0, 0x200000000ull, mapping_bytes, 0x10000, 0,
                 reinterpret_cast<uint64_t>(&physical)) != 0) return 2;
    struct DirectMappingGuard {
        prosper::HleFn unmap, release;
        uint64_t physical, guest = 0, alias = 0, bytes;
        ~DirectMappingGuard() {
            if (alias) unmap(alias, bytes, 0, 0, 0, 0);
            if (guest) unmap(guest, bytes, 0, 0, 0, 0);
            release(physical, bytes, 0, 0, 0, 0);
        }
    } backing{unmap, release, physical, 0, 0, mapping_bytes};
    if (map(reinterpret_cast<uint64_t>(&guest_address), mapping_bytes, 2, 0,
            physical, 0x10000) != 0 || !guest_address) return 2;
    backing.guest = guest_address;
    if (map(reinterpret_cast<uint64_t>(&alias_address), mapping_bytes, 2, 0,
            physical, 0x10000) != 0 || !alias_address) return 2;
    backing.alias = alias_address;
    auto* guest = reinterpret_cast<uint32_t*>(guest_address);
    auto* alias = reinterpret_cast<uint32_t*>(alias_address);
    check(prosper::guest_memory_topology_relation(
              guest_address, work_bytes, alias_address, work_bytes) ==
              prosper::GuestMemoryTopologyRelation::Overlap &&
          prosper::guest_memory_topology_relation(
              guest_address, work_bytes,
              guest_address + depth_plane_offset, 64u * 64u * sizeof(float)) ==
              prosper::GuestMemoryTopologyRelation::Disjoint,
          "buffer aliases and retained depth have the intended tracked physical topology");
    check(prosper::test::render_vk_ctx().ok, "renderer device initializes before live compute fallback");
    if (!prosper::test::render_vk_ctx().ok) return 2;
    if (override_mode)
        overridden_fills(guest, work_bytes);
    else
        cached_fills(guest, alias, work_bytes);
    guest_write_watch_set_fault_onstack(false);
    std::printf("cached_compute_fill: %d failures\n", failures);
    return failures != 0;
}
