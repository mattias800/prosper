// #3407: export-watch demand must defer cost without manufacturing content authority.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "host/memory/guest_write_watch.hpp"
#include "shared/live/live_compute.hpp"
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t width = 64, height = 16, words = width * height;
constexpr size_t bytes = words * sizeof(uint32_t);
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
void watch_fault(int signal, siginfo_t* info, void* context) {
#if defined(__x86_64__)
    const bool write_fault = context &&
        (static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 2) != 0;
#else
    const bool write_fault = context != nullptr;
#endif
    if (signal == SIGSEGV && write_fault && info && info->si_addr &&
        prosper::host::guest_write_watch_handle_fault(reinterpret_cast<uint64_t>(info->si_addr)))
        return;
    _exit(86);
}
ComputeItem writer(const ShaderResource& image, uint32_t value) {
    ShaderResourceTable table;
    table.resources = {image};
    std::vector<uint32_t> code{0x7e080300u, 0x7e0002ffu, value}; // v4=tid; v0=value
    for (uint32_t y = 0; y < height; ++y)
        code.insert(code.end(), {0x7e0a0280u + y, 0xf0200108u, 0x00020004u});
    code.push_back(0xbf810000u);
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);
    config.local_x = width;
    config.local_y = config.local_z = 1;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Uint32, 1);
    ComputeItem result;
    result.spirv = recompile_compute(code.data(), code.size(), &table, config);
    result.resources = std::make_shared<ShaderResourceTable>(table);
    result.user_sgprs = config.user_sgprs;
    result.code_addr = 0x34070001ull + value;
    result.dispatch_index = 1;
    result.command_order = 10;
    result.launch.threads_x = result.launch.local_x = width;
    result.launch.threads_y = result.launch.threads_z = 1;
    result.launch.local_y = result.launch.local_z = 1;
    result.launch.groups_x = result.launch.groups_y = result.launch.groups_z = 1;
    const auto reflection = validate_spirv_descriptor_interface(
        result.spirv, &table, 0, SpirvShaderStage::Compute, false);
    const auto* binding = find_spirv_descriptor_binding(reflection, 0, image.binding);
    check(!result.spirv.empty() && reflection.ok() && reflection.storage_image_writes_complete &&
              binding && binding->writable && !binding->readable &&
              binding->storage_image_format == kSpirvImageFormatR32ui,
          "producer is actual write-only native R32_UINT storage");
    return result;
}
bool borrow(const ShaderResource& image) {
    auto sampled = image;
    sampled.cls = ResourceClass::Texture;
    prosper::frontend::LiveComputeImageImport lease;
    const bool ok = prosper::frontend::import_live_compute_storage_image(sampled, bytes, lease);
    check(ok == lease.valid(), "public import result agrees with lease validity");
    return ok; // Drop each lease before any subsequent producer mutates its owner.
}
void run(uint32_t* guest, uint32_t* alias) {
    ShaderResource image{};
    image.cls = ResourceClass::StorageImage;
    image.binding = 5; image.sgpr_base = 8;
    image.img_dim = 1; image.width = width; image.height = height; image.depth = 1;
    image.format = DataFormat::Uint32; image.num_components = 1;
    image.gpu_addr = reinterpret_cast<uint64_t>(guest); image.size = bytes;
    for (unsigned c = 0; c < 4; ++c) image.swizzle[c] = 4 + c;
    constexpr uint32_t first_value = 0x13579bdfu, second_value = 0x2468ace0u;
    const auto first = writer(image, first_value), second = writer(image, second_value);
    auto exact = [&](uint32_t value) {
        return std::all_of(guest, guest + words, [=](uint32_t word) { return word == value; });
    };
    const auto before = prosper::host::guest_write_watch_stats();
    for (const auto* item : {&first, &second}) {
        DrawItem draw;
        draw.draw_index = 1; draw.command_order = 20;
        unsigned consumers = 0;
        const auto result = execute_ordered_items(
            {{SubmitOperationKind::Dispatch, 1, 10}, {SubmitOperationKind::Draw, 1, 20}},
            {draw}, {*item},
            [&](const std::vector<DrawItem>&, uint32_t, uint32_t) {
                ++consumers;
                check(guest_gpu_write_tracking_active(), "same-submit consumer has ordered authority");
                check(borrow(image), "first and repeated same-submit consumers borrow without demand");
                return RenderedFrame{};
            }, [](const std::vector<ComputeItem>& items) {
                return prosper::frontend::execute_live_compute_items(items);
            }, width, height);
        check(result.compute_executed && consumers == 1, "producer and actual consumer callback execute");
    }
    check(exact(second_value), "every guest texel contains the completed second result");
    const auto journal_only = prosper::host::guest_write_watch_stats();
    check(journal_only.create_attempts == before.create_attempts &&
              journal_only.registrations == before.registrations,
          "repeated journal-only publication creates no page watch");
    check(!guest_gpu_write_tracking_active(), "standalone borrow has no ordered journal");
    check(!borrow(image), "first out-of-journal borrow declines and requests future authority");
    check(prosper::host::guest_write_watch_stats().create_attempts == journal_only.create_attempts,
          "requesting demand does not arm a watch against potentially stale bytes");

    prosper::frontend::live_compute_fail_next_storage_readback_for_test();
    check(!prosper::frontend::execute_live_compute_items({first}), "injected changed-result readback fails");
    check(!borrow(image), "failed producer cannot authorize a requested export");
    check(prosper::frontend::execute_live_compute_items({first}) && exact(first_value),
          "successful retry repairs the complete guest result");
    const auto published = prosper::host::guest_write_watch_stats();
    check(published.registrations > journal_only.registrations,
          "successful publication services demand preserved through failure invalidation");
    check(borrow(image), "completed requested publication permits cross-submit borrow");
    check(prosper::host::guest_write_watch_stats().unchanged > published.unchanged,
          "cross-submit positive actually consults an unchanged registered watch");

    const auto cpu_before = prosper::host::guest_write_watch_stats();
    guest[0] ^= 1;
    check(!borrow(image), "real CPU write revokes demanded export authority");
    check(prosper::host::guest_write_watch_stats().faults > cpu_before.faults,
          "CPU mutation really passes through the installed page-fault handler");
    check(prosper::frontend::execute_live_compute_items({second}) && exact(second_value) && borrow(image),
          "ordinary writeback repairs CPU mutation and rearms requested export");
    const auto alias_before = prosper::host::guest_write_watch_stats();
    alias[words - 1] ^= 1;
    check(guest[words - 1] == (second_value ^ 1u), "alias store changes the same physical guest bytes");
    check(!borrow(image), "real store through a second VA revokes original export authority");
    check(prosper::host::guest_write_watch_stats().faults > alias_before.faults,
          "physical alias is protected by the demanded registration");
}
}
int main() {
    static uint8_t stack_memory[256 * 1024];
    stack_t stack{};
    stack.ss_sp = stack_memory; stack.ss_size = sizeof(stack_memory);
    struct sigaction action{};
    action.sa_sigaction = watch_fault; action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    if (sigaltstack(&stack, nullptr) != 0 || sigaction(SIGSEGV, &action, nullptr) != 0) return 2;
    prosper::host::guest_write_watch_set_fault_onstack(true);
    FILE* backing = std::tmpfile();
    if (!backing || ftruncate(fileno(backing), bytes) != 0) return 2;
    void* mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(backing), 0);
    void* alias = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fileno(backing), 0);
    if (mapping == MAP_FAILED || alias == MAP_FAILED) return 2;
    std::fill_n(static_cast<uint32_t*>(mapping), words, 0xccccccccu);
    constexpr uint64_t physical = 0x340700000ull;
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(mapping), bytes, physical, 0x3);
    prosper::host::guest_write_watch_notify_direct_mapping_added(
        reinterpret_cast<uint64_t>(alias), bytes, physical, 0x3);
    run(static_cast<uint32_t*>(mapping), static_cast<uint32_t*>(alias));
    prosper::host::guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(mapping), bytes);
    prosper::host::guest_write_watch_notify_direct_mapping_removed(reinterpret_cast<uint64_t>(alias), bytes);
    munmap(mapping, bytes); munmap(alias, bytes); std::fclose(backing);
    prosper::host::guest_write_watch_set_fault_onstack(false);
    std::printf("demand_compute_export_watch: %d failures\n", failures);
    return failures ? 1 : 0;
}
