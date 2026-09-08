// Independent storage outputs must publish the final overlapping guest bytes.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "shared/live/live_compute.hpp"
#if defined(__linux__)
#include "host/memory/guest_write_watch.hpp"
#include <csignal>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
using namespace prosper::gpu;
namespace {
constexpr uint32_t width = 64, height = 16, texels = width * height;
enum class Journal { Active, Inactive, Overflow };
Journal journal = Journal::Active;
// Heap backings deliberately have no registered guest page watch. Without a usable
// journal Linux declines a safe borrow; Windows can prove it with its exact mirror.
bool can_retain(bool history_exhausted = false) {
#if defined(_WIN32)
    return true;
#else
    return journal != Journal::Inactive && !history_exhausted;
#endif
}
int failures = 0;
void check(bool ok, const char *message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}
ComputeItem item(const std::vector<uint32_t> &code, const ShaderResourceTable &resources,
                 uint64_t address, uint32_t index) {
    ComputeShaderConfig config;
    config.user_sgprs.resize(24);
    config.local_x = width;
    config.local_y = config.local_z = 1;
    config.tidig_comp_cnt = 0;
    config.native_storage_format_support = native_storage_format_support_bit(DataFormat::Uint32, 1);
    ComputeItem result;
    result.spirv = recompile_compute(code.data(), code.size(), &resources, config);
    result.user_sgprs = config.user_sgprs;
    result.resources = std::make_shared<ShaderResourceTable>(resources);
    result.code_addr = address;
    result.dispatch_index = index;
    result.command_order = index * 10;
    result.launch.threads_x = result.launch.local_x = width;
    result.launch.threads_y = result.launch.threads_z = 1;
    result.launch.local_y = result.launch.local_z = 1;
    result.launch.groups_x = result.launch.groups_y = result.launch.groups_z = 1;
    return result;
}

template <class Callback> void execute_sequence(std::vector<ComputeItem> items, Callback callback) {
    std::vector<SubmitOperation> operations;
    for (size_t i = 0; i < items.size(); ++i) {
        items[i].dispatch_index = i + 1;
        items[i].command_order = (i + 1) * 10;
        operations.push_back(
            {SubmitOperationKind::Dispatch, uint32_t(i + 1), uint32_t((i + 1) * 10)});
    }
    if (journal == Journal::Inactive) {
        check(!guest_gpu_write_tracking_active(),
              "inactive journal arm really has no ordered scope");
        for (const auto &i : items)
            callback(std::vector<ComputeItem>{i});
    } else {
        execute_ordered_items(
            operations, {}, items,
            [](const std::vector<DrawItem> &, uint32_t, uint32_t) { return RenderedFrame{}; },
            callback, 1, 1);
    }
}
struct Fixture {
    std::vector<uint32_t> guest = std::vector<uint32_t>(texels + 16, 0xccccccccu);
    std::vector<uint32_t> other = std::vector<uint32_t>(texels + 16, 0xeeeeeeeeu);
    std::vector<uint32_t> observed = std::vector<uint32_t>(texels, 0xdeadbeefu);
    ShaderResource a{}, b{};
    bool overlap, alias, hosted;
    uint32_t noise = 0;
    Fixture(bool overlap_, bool alias_, bool hosted_ = false)
        : overlap(overlap_), alias(alias_), hosted(hosted_) {
        a.cls = ResourceClass::StorageImage;
        a.binding = 5;
        a.sgpr_base = 8;
        a.img_dim = 1;
        a.format = DataFormat::Uint32;
        a.num_components = 1;
        a.width = width;
        a.height = height;
        a.depth = 1;
        a.size = texels * 4;
        a.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
        for (unsigned c = 0; c < 4; ++c)
            a.swizzle[c] = 4 + c;
        b = a;
        b.binding = 6;
        b.sgpr_base = 16;
        if (!alias) {
            b.height = height / 2;
            b.size = texels * 2;
        }
        if (!overlap || hosted)
            b.gpu_addr = reinterpret_cast<uint64_t>(other.data());
        if (hosted) {
            b.host_data = reinterpret_cast<uint8_t *>(guest.data());
            b.host_data_size = b.size;
        }
    }
    ComputeItem writer(uint32_t av, uint32_t bv, int only = -1) {
        ShaderResourceTable table;
        if (only != 1)
            table.resources.push_back(a);
        if (only != 0)
            table.resources.push_back(b);
        std::vector<uint32_t> code{0x7e080300u};
        for (unsigned output = 0; output < 2; ++output) {
            if (only >= 0 && output != unsigned(only))
                continue;
            code.insert(code.end(), {0x7e0002ffu, output ? bv : av});
            for (uint32_t y = 0; y < (output ? b.height : a.height); ++y)
                code.insert(code.end(),
                            {0x7e0a0280u + y, 0xf0200108u, output ? 0x00040004u : 0x00020004u});
        }
        code.push_back(0xbf810000u);
        auto result = item(code, table, 0x34760001u, 1);
        const auto reflection = validate_spirv_descriptor_interface(
            result.spirv, &table, 0, SpirvShaderStage::Compute, false);
        check(reflection.ok() && reflection.storage_image_writes_complete,
              "complete writer reflection");
        for (const auto &r : table.resources) {
            const auto *d = find_spirv_descriptor_binding(reflection, 0, r.binding);
            check(d && d->writable && !d->readable &&
                      d->storage_image_format == kSpirvImageFormatR32ui,
                  "each output really is native typed R32_UINT storage");
        }
        return result;
    }
    ShaderResource sampled() {
        auto r = a;
        r.cls = ResourceClass::Texture;
        return r;
    }
    ComputeItem reader() {
        ShaderResource output{};
        output.cls = ResourceClass::ConstantBuffer;
        output.binding = 2;
        output.sgpr_base = 0;
        output.format = DataFormat::Uint32;
        output.num_components = 1;
        output.stride = 4;
        output.size = texels * 4;
        output.gpu_addr = reinterpret_cast<uint64_t>(observed.data());
        ShaderResourceTable table;
        table.resources = {output, sampled()};
        std::vector<uint32_t> code{0x7e080300u};
        for (uint32_t y = 0; y < height; ++y)
            code.insert(code.end(), {0x7e0a0280u + y, 0xf0000108u, 0x00020804u, 0xbf8c3f70u,
                                     0xe0702000u | (y * width * 4u), 0x80000804u});
        code.push_back(0xbf810000u);
        return item(code, table, 0x34760002u, 2);
    }
    void prepare_borrow() {
        if (journal != Journal::Overflow)
            return;
        check(guest_gpu_write_tracking_active(), "overflow arm has an active journal");
        const auto snapshot = guest_gpu_write_snapshot();
        for (size_t i = 0; i <= kGuestGpuWriteJournalCapacity; ++i) {
            ++noise;
            notify_guest_gpu_write(reinterpret_cast<uint64_t>(&noise), sizeof(noise));
        }
        check(guest_gpu_writes_since(snapshot, a.gpu_addr, a.size) == GuestGpuWriteQuery::Unknown,
              "overflow arm really exhausts journal authority");
    }
    void check_graphics(bool expected) {
        prosper::frontend::LiveComputeImageImport imported;
        const bool borrowed =
            prosper::frontend::import_live_compute_storage_image(sampled(), a.size, imported);
        check(borrowed == expected && borrowed == imported.valid(),
              "graphics export requires a current exact result and valid lease");
    }
    void run(uint32_t av, uint32_t bv) {
        if (alias)
            bv = av; // Folded descriptors store the same value, without conflicting invocations.
        auto producer = writer(av, bv), consumer = reader();
        check(!producer.spirv.empty() && !consumer.spirv.empty(),
              "real writer and dependent reader compile");
        std::vector<uint32_t> expected(texels, av);
        if (overlap)
            std::fill_n(expected.begin(), b.width * b.height, bv);
        std::fill(observed.begin(), observed.end(), 0xdeadbeefu);
        bool ok = true;
        unsigned calls = 0;
        const auto before = prosper::frontend::live_compute_storage_transfer_seeds();
        execute_sequence({producer, consumer}, [&](const std::vector<ComputeItem> &items) {
            ++calls;
            const bool executed = prosper::frontend::execute_live_compute_items(items);
            ok &= executed;
            if (calls == 1) {
                check(std::equal(expected.begin(), expected.end(), guest.begin()),
                      "guest bytes contain the final ordered output composite");
                if (!overlap) {
                    check(std::all_of(other.begin(), other.begin() + b.width * b.height,
                                      [&](uint32_t v) { return v == bv; }),
                          "disjoint B publishes every texel");
                    check(std::all_of(other.begin() + b.width * b.height, other.end(),
                                      [](uint32_t v) { return v == 0xeeeeeeeeu; }),
                          "disjoint B preserves padding/tail");
                } else if (hosted) {
                    check(std::all_of(other.begin(), other.end(),
                                      [](uint32_t v) { return v == 0xeeeeeeeeu; }),
                          "host-backed output leaves its advertised storage untouched");
                }
                prepare_borrow();
                check_graphics((!overlap || alias) && can_retain(journal == Journal::Overflow));
            }
            return executed;
        });
        if (overlap && !alias)
            check_graphics(false);
        check(ok && calls == 2, "producer and sampled GPU consumer both execute");
        check(observed == expected, "every sampled GPU output matches final guest bytes");
        const auto transfers = prosper::frontend::live_compute_storage_transfer_seeds() - before;
        check((transfers > 0) == ((!overlap || alias) && can_retain(journal == Journal::Overflow)),
              "only an exact retained result authorizes GPU transfer");
        check(std::all_of(guest.begin() + texels, guest.end(),
                          [](uint32_t v) { return v == 0xccccccccu; }),
              "output preserves guest backing tail");
    }
    void prewarm_b(bool host_baseline) {
        const auto before = prosper::frontend::live_compute_image_result_snapshot_bytes();
        if (host_baseline)
            prosper::frontend::live_compute_force_next_image_result_host_fallback_for_test();
        check(prosper::frontend::execute_live_compute_items({writer(0, 0x2468ace0u, 1)}),
              "late B establishes its independent result before overlapping A exists");
        check(prosper::frontend::live_compute_image_result_snapshot_bytes() - before ==
                  (host_baseline ? b.size : 0u),
              "late B establishes requested CPU or GPU baseline");
    }
    void dcc_overlap(bool hosted_metadata) {
        // A is linear; B's separate tiled pixels use a metadata plane overlapping A.
        b.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const size_t tiled_bytes = tiled_surface_bytes(width, b.height, b.tile_mode, 0, 4);
        other.resize(tiled_bytes / 4 + 16, 0xeeeeeeeeu);
        b.gpu_addr = reinterpret_cast<uint64_t>(other.data());
        b.size = tiled_bytes;
        b.compression_enabled = b.write_compress_enabled = b.meta_pipe_aligned = true;
        b.metadata_addr = a.gpu_addr;
        const size_t metadata_bytes = gpu_capture_dcc_metadata_footprint(b);
        check(metadata_bytes && metadata_bytes <= a.size && !(metadata_bytes % 4),
              "bounded real DCC footprint overlaps A");
        if (!metadata_bytes || metadata_bytes > a.size || metadata_bytes % 4)
            return;
        b.dcc_metadata_size = metadata_bytes;
        if (hosted_metadata) {
            b.metadata_addr = reinterpret_cast<uint64_t>(advertised_metadata.data());
            b.dcc_metadata_host_data = reinterpret_cast<uint8_t *>(guest.data());
            b.dcc_metadata_host_data_size = metadata_bytes;
        }
        std::fill(guest.begin(), guest.end(),
                  0xffffffffu); // valid all-uncompressed initial metadata
        std::vector<uint32_t> expected(texels, 0x13579bdfu);
        std::fill_n(expected.begin(), metadata_bytes / 4, 0xffffffffu);
        unsigned calls = 0;
        bool ok = true;
        execute_sequence({writer(0x13579bdfu, 0x2468ace0u), reader()},
                         [&](const std::vector<ComputeItem> &items) {
                             ++calls;
                             const bool executed =
                                 prosper::frontend::execute_live_compute_items(items);
                             ok &= executed;
                             if (calls == 1) {
                                 check(std::equal(expected.begin(), expected.end(), guest.begin()),
                                       "DCC reset is part of final A composite");
                                 prepare_borrow();
                                 check_graphics(false);
                             }
                             return executed;
                         });
        check(ok && calls == 2 && observed == expected,
              "sampled A observes overlapping DCC metadata reset");
        std::vector<uint32_t> linear(width * b.height);
        detile_surface(reinterpret_cast<uint8_t *>(linear.data()),
                       reinterpret_cast<const uint8_t *>(other.data()), width, b.height,
                       b.tile_mode, 0, 4);
        check(
            std::all_of(linear.begin(), linear.end(), [](uint32_t v) { return v == 0x2468ace0u; }),
            "separate DCC pixel owner publishes its own exact output");
        check(std::all_of(advertised_metadata.begin(), advertised_metadata.end(),
                          [](uint8_t v) { return v == 0x97; }),
              "hosted DCC preserves advertised metadata backing");
        check_graphics(false);
    }
    std::vector<uint8_t> advertised_metadata = std::vector<uint8_t>(65536, 0x97);
    void self_dcc() {
        a.tile_mode = static_cast<uint32_t>(TileMode::Sw64KbRX);
        const size_t tiled_bytes = tiled_surface_bytes(width, height, a.tile_mode, 0, 4);
        guest.resize(tiled_bytes / 4 + 16, 0xffffffffu);
        std::fill(guest.begin(), guest.end(), 0xffffffffu);
        a.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
        a.size = tiled_bytes;
        a.compression_enabled = a.write_compress_enabled = a.meta_pipe_aligned = true;
        a.metadata_addr = a.gpu_addr;
        const size_t metadata_bytes = gpu_capture_dcc_metadata_footprint(a);
        check(metadata_bytes && metadata_bytes < tiled_bytes,
              "bounded self-overlapping DCC metadata");
        if (!metadata_bytes || metadata_bytes >= tiled_bytes)
            return;
        a.dcc_metadata_size = metadata_bytes;
        std::vector<uint32_t> expected(texels);
        unsigned calls = 0;
        bool ok = true;
        execute_sequence(
            {writer(0x13579bdfu, 0, 0), reader()}, [&](const std::vector<ComputeItem> &items) {
                ++calls;
                const bool executed = prosper::frontend::execute_live_compute_items(items);
                ok &= executed;
                if (calls == 1) {
                    check(std::all_of(reinterpret_cast<uint8_t *>(guest.data()),
                                      reinterpret_cast<uint8_t *>(guest.data()) + metadata_bytes,
                                      [](uint8_t v) { return v == 0xff; }),
                          "self-overlapping metadata reset changes stored pixels");
                    detile_surface(reinterpret_cast<uint8_t *>(expected.data()),
                                   reinterpret_cast<uint8_t *>(guest.data()), width, height,
                                   a.tile_mode, 0, 4);
                    check(expected.front() == 0xffffffffu &&
                              std::find(expected.begin(), expected.end(), 0x13579bdfu) !=
                                  expected.end(),
                          "self-DCC oracle distinguishes reset bytes from original private image");
                    prepare_borrow();
                    check_graphics(false);
                }
                return executed;
            });
        check(ok && calls == 2 && observed == expected,
              "sampled self-DCC view observes final physical guest composite");
        check_graphics(false);
    }
    void pinned_overlap() {
        prosper::frontend::LiveComputeImageImport pinned;
        unsigned calls = 0;
        bool ok = true;
        std::vector<uint32_t> expected(texels, 0x31415926u);
        std::fill_n(expected.begin(), b.width * b.height, 0x2468ace0u);
        execute_sequence({writer(0x13579bdfu, 0, 0), writer(0x31415926u, 0x2468ace0u), reader()},
                         [&](const std::vector<ComputeItem> &items) {
                             ++calls;
                             const bool executed =
                                 prosper::frontend::execute_live_compute_items(items);
                             ok &= executed;
                             if (calls == 1)
                                 check(prosper::frontend::import_live_compute_storage_image(
                                           sampled(), a.size, pinned) == can_retain(),
                                       "hold real graphics lease across overlapping dispatch");
                             if (calls == 2) {
                                 prepare_borrow();
                                 check_graphics(false);
                             }
                             return executed;
                         });
        check(ok && calls == 3 && observed == expected,
              "pinned allocation survives conflict while new consumers see current bytes");
        check(pinned.valid() == can_retain(),
              "authority invalidation preserves outstanding lease lifetime");
    }
    ComputeItem buffer_writer(bool alongside_image) {
        ShaderResource output{};
        output.cls = ResourceClass::ConstantBuffer;
        output.binding = 2;
        output.sgpr_base = 0;
        output.format = DataFormat::Uint32;
        output.num_components = 1;
        output.stride = 4;
        output.size = b.width * b.height * 4;
        output.gpu_addr = b.gpu_addr;
        output.host_data = b.host_data;
        output.host_data_size = b.host_data_size;
        ShaderResourceTable table;
        table.resources = {output};
        std::vector<uint32_t> code{0x7e080300u, 0x7e1002ffu, 0x2468ace0u};
        for (uint32_t y = 0; y < b.height; ++y)
            code.insert(code.end(), {0xe0702000u | (y * width * 4u), 0x80000804u});
        if (alongside_image) {
            table.resources.push_back(a);
            code.insert(code.end(), {0x7e0002ffu, 0x13579bdfu});
            for (uint32_t y = 0; y < a.height; ++y)
                code.insert(code.end(), {0x7e0a0280u + y, 0xf0200108u, 0x00020004u});
        }
        code.push_back(0xbf810000u);
        return item(code, table, 0x34760004u, 1);
    }
    void image_after_buffer(bool host_baseline) {
        const auto before = prosper::frontend::live_compute_image_result_snapshot_bytes();
        if (host_baseline)
            prosper::frontend::live_compute_force_next_image_result_host_fallback_for_test();
        unsigned calls = 0;
        bool ok = true;
        execute_sequence(
            {writer(0x13579bdfu, 0, 0), buffer_writer(true), reader()},
            [&](const std::vector<ComputeItem> &items) {
                ++calls;
                const bool executed = prosper::frontend::execute_live_compute_items(items);
                ok &= executed;
                if (calls == 1) {
                    check(prosper::frontend::live_compute_image_result_snapshot_bytes() - before ==
                              (host_baseline ? a.size : 0u),
                          "A establishes requested baseline before buffer conflict");
                    check_graphics(can_retain());
                }
                if (calls == 2) {
                    prepare_borrow();
                    check_graphics(false);
                }
                return executed;
            });
        check(ok && calls == 3, "image after overlapping buffer executes");
        check(std::all_of(guest.begin(), guest.begin() + texels,
                          [](uint32_t v) { return v == 0x13579bdfu; }) &&
                  std::all_of(observed.begin(), observed.end(),
                              [](uint32_t v) { return v == 0x13579bdfu; }),
              "unchanged image must restore bytes overwritten by earlier buffer writeback");
    }
    void internal_output() {
        std::vector<uint32_t> internal(texels, 0xccccccccu);
        auto producer = buffer_writer(true);
        auto table = std::make_shared<ShaderResourceTable>(*producer.resources);
        table->resources[0].gpu_addr = 0; // Same convention as GDS and private storage masks.
        table->resources[0].host_data = reinterpret_cast<uint8_t *>(internal.data());
        table->resources[0].host_data_size = internal.size() * 4;
        producer.resources = table;
        unsigned private_notifications = 0, calls = 0;
        set_guest_gpu_write_observer([&](uint64_t address, uint64_t, const char *) {
            if (!address || address == reinterpret_cast<uint64_t>(internal.data()))
                ++private_notifications;
        });
        execute_sequence({producer, reader()}, [&](const std::vector<ComputeItem> &items) {
            ++calls;
            const bool ok = prosper::frontend::execute_live_compute_items(items);
            check(ok, "internal output and unrelated guest image execute");
            if (calls == 1) {
                prepare_borrow();
                check_graphics(can_retain(journal == Journal::Overflow));
            }
            return ok;
        });
        set_guest_gpu_write_observer({});
        check(calls == 2 && private_notifications == 0,
              "internal backing neither announces guest writes nor blocks unrelated publication");
        check(std::all_of(internal.begin(), internal.begin() + texels / 2,
                          [](uint32_t v) { return v == 0x2468ace0u; }) &&
                  std::all_of(internal.begin() + texels / 2, internal.end(),
                              [](uint32_t v) { return v == 0xccccccccu; }),
              "internal output writes exact bytes and preserves its tail");
        check(std::all_of(observed.begin(), observed.end(),
                          [](uint32_t v) { return v == 0x13579bdfu; }),
              "unrelated image remains a current sampled result");
    }
    void submission_failure() {
        unsigned calls = 0;
        execute_sequence(
            {writer(0x13579bdfu, 0, 0), writer(0x31415926u, 0x2468ace0u)},
            [&](const std::vector<ComputeItem> &items) {
                ++calls;
                if (calls == 2)
                    prosper::frontend::live_compute_force_next_queue_submit_device_lost_for_test();
                const bool executed = prosper::frontend::execute_live_compute_items(items);
                check(executed == (calls == 1),
                      "device-loss arm reaches the real failed submit branch");
                check_graphics(calls == 1);
                return executed;
            });
        check(calls == 2, "failure is injected after successful retained A");
        check_graphics(false);
    }
    void unbound_hosted(bool buffer = false) {
        check(hosted, "unbound regression has an effective hosted overlap");
        constexpr uint32_t av = 0x13579bdfu, bv = 0x2468ace0u;
        auto first = writer(av, bv, 0), second = buffer ? buffer_writer(false) : writer(av, bv, 1),
             consumer = reader();
        std::vector<uint32_t> expected(texels, av);
        std::fill_n(expected.begin(), b.width * b.height, bv);
        unsigned calls = 0;
        bool ok = true;
        execute_sequence({first, second, consumer}, [&](const std::vector<ComputeItem> &items) {
            ++calls;
            const bool executed = prosper::frontend::execute_live_compute_items(items);
            ok &= executed;
            if (calls == 1)
                check_graphics(can_retain()); // Actual retained A positive before it is absent.
            if (calls == 2) {
                check(std::equal(expected.begin(), expected.end(), guest.begin()),
                      "hosted-only writer creates guest composite");
                prepare_borrow();
                check_graphics(false);
            }
            return executed;
        });
        check(ok && calls == 3 && observed == expected,
              "unbound cached view observes actual hosted destination writes");
        check_graphics(false);
    }
};
#if defined(__linux__)
void watch_fault(int signal, siginfo_t *info, void *context) {
#if defined(__x86_64__)
    const bool write_fault =
        context && (static_cast<ucontext_t *>(context)->uc_mcontext.gregs[REG_ERR] & 2) != 0;
#else
    const bool write_fault = context != nullptr;
#endif
    if (signal == SIGSEGV && write_fault && info && info->si_addr &&
        prosper::host::guest_write_watch_handle_fault(reinterpret_cast<uint64_t>(info->si_addr)))
        return;
    _exit(86);
}
void watched_cross_submit() {
    static uint8_t stack_memory[256 * 1024];
    stack_t stack{};
    stack.ss_sp = stack_memory;
    stack.ss_size = sizeof(stack_memory);
    struct sigaction action{};
    action.sa_sigaction = watch_fault;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    const bool installed =
        sigaltstack(&stack, nullptr) == 0 && sigaction(SIGSEGV, &action, nullptr) == 0;
    check(installed, "install red-zone-safe real write-watch fault handler");
    if (!installed)
        return;
    prosper::host::guest_write_watch_set_fault_onstack(true);
    constexpr size_t bytes = 8192;
    void *mapping =
        mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    check(mapping != MAP_FAILED, "allocate actual registered guest mapping");
    if (mapping == MAP_FAILED)
        return;
    const uint64_t address = reinterpret_cast<uint64_t>(mapping);
    prosper::host::guest_write_watch_notify_direct_mapping_added(address, bytes, 0x347600000ull,
                                                                 0x3);
    std::fill_n(static_cast<uint32_t *>(mapping), bytes / 4, 0xccccccccu);
    {
        Fixture f(true, false);
        f.a.gpu_addr = f.b.gpu_addr = address;
        const auto before = prosper::host::guest_write_watch_stats();
        execute_sequence({f.writer(0x13579bdfu, 0, 0)}, [](const std::vector<ComputeItem> &items) {
            const bool ok = prosper::frontend::execute_live_compute_items(items);
            check(ok, "watched producer executes");
            return ok;
        });
        check(!guest_gpu_write_tracking_active(), "producer journal ended before watched borrow");
        f.check_graphics(true);
        const auto after = prosper::host::guest_write_watch_stats();
        check(after.registrations > before.registrations && after.unchanged > before.unchanged,
              "cross-submit positive really uses a registered unchanged page watch");
        execute_sequence({f.writer(0x31415926u, 0x2468ace0u)},
                         [](const std::vector<ComputeItem> &items) {
                             const bool ok = prosper::frontend::execute_live_compute_items(items);
                             check(ok, "watched conflict executes");
                             return ok;
                         });
        f.check_graphics(false);
        check(prosper::frontend::execute_live_compute_items({f.reader()}),
              "cross-submit watched consumer executes");
        for (size_t i = 0; i < texels; ++i) {
            const uint32_t expected = i < texels / 2 ? 0x2468ace0u : 0x31415926u;
            if (static_cast<uint32_t *>(mapping)[i] != expected || f.observed[i] != expected) {
                check(false, "watched guest and sampled consumer preserve final composite");
                break;
            }
        }
    }
    prosper::host::guest_write_watch_notify_direct_mapping_removed(address, bytes);
    munmap(mapping, bytes);
}
#endif
struct BufferFixture {
    static constexpr size_t words = 1024 * 1024 / 4;
    std::vector<uint32_t> guest = std::vector<uint32_t>(words + 64, 0xccccccccu);
    ShaderResource a{}, b{};
    explicit BufferFixture(bool overlap) {
        a.cls = ResourceClass::ConstantBuffer;
        a.binding = 2;
        a.sgpr_base = 0;
        a.format = DataFormat::Uint32;
        a.num_components = 1;
        a.stride = 4;
        a.size = words * 4;
        a.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
        b = a;
        b.binding = 3;
        b.sgpr_base = 4;
        b.gpu_addr += overlap ? 128 : 0;
    }
    ComputeItem writer(bool both, uint32_t av = 0x13579bdfu) {
        ShaderResourceTable table;
        table.resources = both ? std::vector<ShaderResource>{a, b} : std::vector<ShaderResource>{b};
        std::vector<uint32_t> code{0x7e080300u};
        if (both)
            code.insert(code.end(), {0x7e1002ffu, av, 0xe0702000u, 0x80000804u});
        code.insert(code.end(), {0x7e1002ffu, 0x2468ace0u, 0xe0702000u, 0x80010804u, 0xbf810000u});
        return item(code, table, 0x34760003u, 1);
    }
    void run() {
        const auto before = prosper::frontend::live_compute_buffer_gpu_result_skips();
        execute_sequence({writer(false), writer(false), writer(true)},
                         [&](const std::vector<ComputeItem> &items) {
                             const bool ok = prosper::frontend::execute_live_compute_items(items);
                             check(ok, "overlapping buffer writer executes");
                             return ok;
                         });
        check(prosper::frontend::live_compute_buffer_gpu_result_skips() == before + 1,
              "unchanged isolated B skips once; conflicting B must restore its bytes");
        for (size_t i = 0; i < guest.size(); ++i) {
            const uint32_t expected = i < 32 ? 0x13579bdfu : i < 96 ? 0x2468ace0u : 0xccccccccu;
            if (guest[i] != expected) {
                check(false, "buffer composite and full untouched tail are exact");
                break;
            }
        }
    }
};
} // namespace
int main(int argc, char **argv) {
#if defined(__linux__)
    if (argc == 2 && !std::strcmp(argv[1], "--watched")) {
        watched_cross_submit();
        return failures ? 1 : 0;
    }
#endif
    if (argc == 2 && !std::strcmp(argv[1], "--submit-failure")) {
        Fixture failure(true, false);
        failure.submission_failure();
        return failures ? 1 : 0;
    }
    if (argc == 2 && !std::strcmp(argv[1], "--unjournaled"))
        journal = Journal::Inactive;
    else if (argc == 2 && !std::strcmp(argv[1], "--overflow"))
        journal = Journal::Overflow;
    else if (argc != 1)
        return 2;
    // Simultaneous identities prevent a prior arm's cached sampled view from hiding publication.
    Fixture conflict(true, false), disjoint(false, false), alias(true, true),
        hosted(true, false, true), unbound(true, false, true);
    for (auto *f : {&conflict, &disjoint, &alias, &hosted}) {
        f->run(0x13579bdfu, 0x2468ace0u);
        f->run(0x31415926u, 0x2468ace0u); // Changed early A, unchanged overlapping late B.
    }
    unbound.unbound_hosted();
    Fixture gpu_baseline(true, false), host_baseline(true, false), dcc(false, false),
        hosted_dcc(false, false);
    gpu_baseline.prewarm_b(false);
    gpu_baseline.run(0x13579bdfu, 0x2468ace0u);
    host_baseline.prewarm_b(true);
    host_baseline.run(0x13579bdfu, 0x2468ace0u);
    dcc.dcc_overlap(false);
    hosted_dcc.dcc_overlap(true);
    Fixture self(true, false), pinned(true, false);
    self.self_dcc();
    pinned.pinned_overlap();
    Fixture buffer_unbound(true, false, true), buffer_gpu(true, false, true),
        buffer_cpu(true, false, true);
    buffer_unbound.unbound_hosted(true);
    buffer_gpu.image_after_buffer(false);
    buffer_cpu.image_after_buffer(true);
    Fixture internal(false, false);
    internal.internal_output();
    BufferFixture buffers(true);
    buffers.run();
    return failures ? 1 : 0;
}
