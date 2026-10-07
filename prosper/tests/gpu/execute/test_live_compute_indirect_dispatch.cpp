// test_live_compute_indirect_dispatch -- #3656, the Vulkan half: the production compute backend
// consumes a device-resolved indirect dispatch (ComputeItem::indirect_args_addr) with
// vkCmdDispatchIndirect from a retained GPU buffer, instead of the CPU reading the triplet.
//
// HOW THE ARMS DISCRIMINATE. A GPU PRODUCER (a real compute dispatch) writes a valid triplet into a
// 1 MiB buffer the backend retains. The test then overwrites the same range in GUEST memory with a
// DELIBERATELY WRONG triplet, {7,7,7}, using a plain store the retained buffer's write journal never
// sees. A consumer that read its counts from guest memory launches 7 groups and tramples the canary
// words after the expected extent; the device route launches what the producer wrote. The expected
// triplet is a different, small value in every arm, so a launch that ignored the argument buffer
// (for instance by using item.launch's zero) writes nothing and fails the same checks.
//
// The producer and the consumer are the same program: the production fill kernel
// (record = (tgid.x << 6) + tid.x; out[record] = {s4,s5,s6,s7}). Group count x therefore decides
// exactly how many 16-byte records are written, which makes the expected output derivable by hand.
#include "shared/live/live_compute.hpp"
#include "shared/live/indirect_dispatch.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

using namespace prosper::gpu;
using prosper::frontend::indirect_dispatch_backend_stats;

static int failures = 0;
#define CHECK(condition, message)                                                   \
    do {                                                                            \
        if (!(condition)) {                                                         \
            std::printf("  [FAIL] %s\n", message);                                  \
            ++failures;                                                             \
        } else {                                                                    \
            std::printf("  [ok]   %s\n", message);                                  \
        }                                                                           \
    } while (0)

// The exact production fill stream (classify_compute_cpu_fast_path's FillSgprUvec4).
static constexpr uint32_t kFillCode[] = {
    0xd7460004u, 0x04010c08u, 0x7e000204u, 0x7e020205u, 0x7e040206u,
    0x7e060207u, 0xe01c2000u, 0x80000004u, 0xbf810000u,
};

static constexpr uint32_t kArgumentBufferBytes = 1u << 20;     // the retention threshold
static constexpr uint32_t kOutputBytes = 8192;                  // 512 records
static constexpr uint32_t kCanary = 0xCCCCCCCCu;

// Zeroed, 4 KiB-aligned guest-style storage.
struct Storage {
    std::vector<uint8_t> raw;
    uint8_t* base = nullptr;
    explicit Storage(size_t bytes) : raw(bytes + 4096) {
        base = reinterpret_cast<uint8_t*>(
            (reinterpret_cast<uintptr_t>(raw.data()) + 4095u) & ~uintptr_t{4095u});
    }
    uint32_t* words() { return reinterpret_cast<uint32_t*>(base); }
};

// One dispatch of the fill program writing `pattern` over `extent` bytes at `target`. When `records`
// is non-zero the thread extent is exact (the producer); otherwise the launch is the group-count
// form whose counts the caller supplies either directly or through indirect_args_addr.
static ComputeItem fill_item(uint8_t* target, uint32_t extent, std::array<uint32_t, 4> pattern,
                             uint32_t records, uint64_t indirect_args_addr, uint64_t code_addr) {
    ShaderResource r{};
    r.cls = ResourceClass::ConstantBuffer;
    r.binding = 2;
    r.sgpr_base = 0;
    r.format = DataFormat::Uint32;
    r.num_components = 4;
    r.stride = 16;
    r.gpu_addr = reinterpret_cast<uint64_t>(target);
    r.size = extent;
    ComputeItem item;
    item.resources = std::make_shared<ShaderResourceTable>();
    item.resources->resources = {r};
    item.user_sgprs = {uint32_t(r.gpu_addr), uint32_t(r.gpu_addr >> 32) | (16u << 16),
                       extent / 16, 0, pattern[0], pattern[1], pattern[2], pattern[3]};
    item.code_addr = code_addr;
    item.launch.local_x = 64;
    item.launch.local_y = item.launch.local_z = 1;
    item.recompile_config_available = true;
    auto& config = item.recompile_config;
    config.user_sgprs = item.user_sgprs;
    config.local_x = 64;
    config.local_y = config.local_z = 1;
    config.tgid_x_en = true;
    config.tgid_y_en = config.tgid_z_en = false;
    if (records) {
        item.launch.groups_x = (records + 63) / 64;
        item.launch.groups_y = item.launch.groups_z = 1;
        item.launch.threads_x = records;
        item.launch.threads_y = item.launch.threads_z = 1;
        config.exact_thread_extent = true;
        config.threads_x = records;
        config.threads_y = config.threads_z = 1;
    } else {
        // Unresolved launch: zero groups and an unspecialized thread extent, exactly what the
        // executor hands over for a device-resolved dispatch.
        item.indirect_args_addr = indirect_args_addr;
        item.indirect_dispatch = indirect_args_addr != 0;
    }
    item.spirv = recompile_compute(kFillCode, std::size(kFillCode), item.resources.get(), config);
    return item;
}

// The words every output record must hold after `records` records were written, canary elsewhere.
static bool output_matches(const uint32_t* out, uint32_t records,
                           std::array<uint32_t, 4> pattern) {
    for (uint32_t i = 0; i < kOutputBytes / 4; ++i) {
        const uint32_t expected = i / 4 < records ? pattern[i % 4] : kCanary;
        if (out[i] != expected) return false;
    }
    return true;
}

static void reset_output(uint32_t* out) {
    std::fill(out, out + kOutputBytes / 4, kCanary);
}

struct Counters {
    uint64_t device, fallback, rejected;
};
static Counters counters() {
    auto& s = indirect_dispatch_backend_stats();
    return {s.device_dispatches.load(), s.host_fallbacks.load(), s.rejected.load()};
}

// Runs `items` as ONE ordered submit through the production compute backend. The retained buffer's
// write journal exists only inside such a submit (execute_ordered_items arms it), and that journal is
// what makes a retained buffer authoritative for a later dispatch in the same submit -- which is
// exactly the producer -> indirect-consumer shape this change serves. `after_each(index)` runs
// immediately after each dispatch completes and writes back, so a test can model a CPU store that
// lands between the producer and the consumer.
template <typename AfterEach>
static std::vector<bool> run_submit(const std::vector<ComputeItem>& submitted,
                                    AfterEach after_each) {
    // The ordered executor finds each item through ComputeItem::dispatch_index, so give every one
    // its position; left at zero they would all collapse onto the last item.
    std::vector<ComputeItem> items = submitted;
    std::vector<SubmitOperation> operations;
    for (size_t i = 0; i < items.size(); ++i) {
        items[i].dispatch_index = i;
        operations.push_back({SubmitOperationKind::Dispatch, i, 100 + i * 100});
    }
    std::vector<bool> results(items.size(), false);
    size_t next = 0;
    const LiveRenderFn render = [](const std::vector<DrawItem>&, uint32_t, uint32_t) {
        return std::vector<uint8_t>(4, 0);
    };
    const LiveComputeFn compute = [&](const std::vector<ComputeItem>& one) {
        const bool ok = prosper::frontend::execute_live_compute_items(one);
        results[next] = ok;
        after_each(next);
        ++next;
        return ok;
    };
    execute_ordered_items(operations, {}, items, render, compute, 1, 1);
    return results;
}
static std::vector<bool> run_submit(const std::vector<ComputeItem>& items) {
    return run_submit(items, [](size_t) {});
}

int main() {
    std::printf("== test_live_compute_indirect_dispatch ==\n");
    Storage args_storage(kArgumentBufferBytes + 4096);
    Storage out_storage(kOutputBytes);
    uint32_t* const args = args_storage.words();
    uint32_t* const out = out_storage.words();
    const uint64_t args_base = reinterpret_cast<uint64_t>(args);
    const std::array<uint32_t, 4> kProducer = {2, 1, 3, 5};      // offsets 0/4/8 -> x = 2/1/3
    const std::array<uint32_t, 4> kConsumerPattern = {0x11111111u, 0x22222222u, 0x33333333u,
                                                      0x44444444u};
    constexpr uint32_t kWords = kArgumentBufferBytes / 4;
    constexpr uint32_t kRecords = kArgumentBufferBytes / 16;

    auto producer = [&](std::array<uint32_t, 4> pattern, uint64_t code) {
        return fill_item(args_storage.base, kArgumentBufferBytes, pattern, kRecords, 0, code);
    };
    auto consumer = [&](uint64_t address, uint64_t code) {
        return fill_item(out_storage.base, kOutputBytes, kConsumerPattern, 0, address, code);
    };
    // Overwrite the argument range in guest memory with a triplet no arm expects. A plain store:
    // nothing journals it, so the retained buffer stays authoritative for what the GPU wrote.
    auto scribble_wrong = [](uint64_t address) {
        uint32_t* p = reinterpret_cast<uint32_t*>(address);
        p[0] = p[1] = p[2] = 7;
    };
    // producer -> (CPU scribbles the argument range) -> consumer, as one ordered submit.
    auto produce_then_consume = [&](std::array<uint32_t, 4> pattern, uint64_t address,
                                    bool scribble, uint64_t code) {
        std::vector<ComputeItem> items = {producer(pattern, code), consumer(address, code + 1)};
        const bool built = !items[0].spirv.empty() && !items[1].spirv.empty();
        reset_output(out);
        const std::vector<bool> ran = run_submit(items, [&](size_t index) {
            if (index == 0 && scribble) scribble_wrong(address);
        });
        return std::make_pair(built, ran.size() == 2 && ran[0] && ran[1]);
    };

    // ---- Arm 1: producer -> indirect consumer, offset 0 ------------------------------------------
    {
        const Counters before = counters();
        const auto [built, ran] = produce_then_consume(kProducer, args_base, true, 0x36560010u);
        const Counters after = counters();
        CHECK(built && ran, "the GPU producer and the indirect consumer both execute");
        CHECK(after.device == before.device + 1 && after.fallback == before.fallback,
              "the selected arm is vkCmdDispatchIndirect: one device dispatch, no host fallback");
        CHECK(output_matches(out, 2 * 64, kConsumerPattern),
              "the consumer launched the GPU's x=2 (128 records), not the guest's wrong x=7");
    }

    const auto device_limits = prosper::frontend::live_compute_workgroup_count_limits_for_test();
    const bool limits_observed = std::all_of(device_limits.begin(), device_limits.end(),
                                           [](uint32_t limit) { return limit != 0; });
    CHECK(limits_observed, "observe the selected device's limits before deciding expected launches");
    if (!limits_observed) return 1;
    std::printf("  device workgroup-count limits: %u x %u x %u\n",
                device_limits[0], device_limits[1], device_limits[2]);
    std::array<uint32_t, 4> excessive_counts{1, 1, 1, 0};
    const auto excessive_axis = std::find_if(device_limits.begin(), device_limits.end(),
        [](uint32_t limit) { return limit != std::numeric_limits<uint32_t>::max(); });
    const bool excessive_representable = excessive_axis != device_limits.end();
    if (excessive_representable)
        excessive_counts[excessive_axis - device_limits.begin()] = *excessive_axis + 1u;

    // ---- Arm 2: nonzero buffer offsets -------------------------------------------------------------
    {
        const Counters before = counters();
        const auto [built, ran] = produce_then_consume(kProducer, args_base + 4, true, 0x36560020u);
        const Counters after = counters();
        CHECK(built && ran && after.device == before.device + 1 &&
                  after.fallback == before.fallback,
              "an offset of 4 bytes stays on the device route");
        CHECK(output_matches(out, 1 * 64, kConsumerPattern),
              "offset 4 holds {1,3,5}: one group of records, so the offset was honoured");
    }
    {
        const auto [built, ran] = produce_then_consume(kProducer, args_base + 8, true, 0x36560030u);
        CHECK(built && ran && output_matches(out, 3 * 64, kConsumerPattern),
              "offset 8 holds {3,5,2}: three groups of records");
    }

    // ---- Arm 3: the last valid 12 bytes of the retained buffer -----------------------------------
    {
        const uint64_t last = args_base + kArgumentBufferBytes - 12;
        const Counters before = counters();
        const auto [built, ran] = produce_then_consume(kProducer, last, true, 0x36560040u);
        const Counters after = counters();
        // Word kWords-3 is word 1 of its record: {1,3,5}.
        CHECK(args[kWords - 3] == 7 && args[kWords - 2] == 7 && args[kWords - 1] == 7,
              "fixture: the guest's last 12 bytes were overwritten with {7,7,7}");
        CHECK(built && ran && after.device == before.device + 1 &&
                  after.fallback == before.fallback,
              "a range ending exactly at the buffer's end is still covered");
        CHECK(output_matches(out, 1 * 64, kConsumerPattern),
              "the last-12-bytes triplet {1,3,5} was honoured");
    }

    // ---- Arm 4: the arguments are overwritten between two dispatches ------------------------------
    {
        std::vector<ComputeItem> items = {producer(kProducer, 0x36560050u),
                                          consumer(args_base, 0x36560051u),
                                          producer({4, 1, 1, 0}, 0x36560052u),
                                          consumer(args_base, 0x36560053u)};
        uint32_t first_out[kOutputBytes / 4] = {};
        reset_output(out);
        const std::vector<bool> ran = run_submit(items, [&](size_t index) {
            if (index == 1) std::memcpy(first_out, out, sizeof(first_out)); // first consumer done
        });
        CHECK(ran.size() == 4 && ran[0] && ran[1] && ran[2] && ran[3], "all four dispatches execute");
        CHECK(output_matches(first_out, 2 * 64, kConsumerPattern),
              "the first consumer saw the first producer's x=2");
        CHECK(output_matches(out, 4 * 64, kConsumerPattern),
              "the second consumer saw the NEWER producer's x=4 (it also re-wrote the first 128)");
    }

    // ---- Arm 5: a zero count is a hardware no-op decided by the device ----------------------------
    {
        const Counters before = counters();
        const uint64_t out_base = reinterpret_cast<uint64_t>(out);
        uint32_t output_writes = 0;
        set_guest_gpu_write_observer([&](uint64_t address, uint64_t size, const char*) {
            if (address < out_base + kOutputBytes && out_base < address + size) ++output_writes;
        });
        const auto [built, ran] = produce_then_consume({0, 1, 1, 0}, args_base, true, 0x36560060u);
        set_guest_gpu_write_observer({});
        const Counters after = counters();
        CHECK(built && ran && after.device == before.device + 1 && after.rejected == before.rejected,
              "zero counts dispatch (and launch nothing) without being reported as an error");
        CHECK(output_matches(out, 0, kConsumerPattern), "nothing was written");
        CHECK(output_writes == 0,
              "a zero-group launch publishes no write: nothing downstream is invalidated");
    }

    // ---- Arm 5b: a DIRECT launch with a zero group count is the same hardware no-op ----------------
    // It used to be reported as "exceeds the workgroup-count limit" and declined, which dropped the
    // dispatch as a failure (and poisoned its producer epoch) although nothing was wrong (#4131).
    {
        reset_output(out);
        const uint64_t out_base = reinterpret_cast<uint64_t>(out);
        uint32_t output_writes = 0;
        set_guest_gpu_write_observer([&](uint64_t address, uint64_t size, const char*) {
            if (address < out_base + kOutputBytes && out_base < address + size) ++output_writes;
        });
        ComputeItem zero = consumer(0, 0x36560065u);   // direct form: no indirect arguments
        zero.launch.groups_x = zero.launch.groups_y = zero.launch.groups_z = 0;
        const std::vector<bool> ran = run_submit({zero});
        set_guest_gpu_write_observer({});
        CHECK(ran.size() == 1 && ran[0],
              "a direct zero-group launch is a neutral no-op, not a declined dispatch");
        CHECK(output_matches(out, 0, kConsumerPattern) && output_writes == 0,
              "a direct zero-group launch writes and publishes nothing");
    }

    // ---- Arm 6: a device-produced count above the device limit is refused, not launched -----------
    if (excessive_representable) {
        const Counters before = counters();
        std::vector<ComputeItem> items = {producer(excessive_counts, 0x36560070u),
                                          consumer(args_base, 0x36560071u)};
        reset_output(out);
        const std::vector<bool> ran = run_submit(items);
        const Counters after = counters();
        CHECK(ran.size() == 2 && ran[0],
              "the dispatch completes instead of hanging or losing the device");
        CHECK(after.device == before.device + 1 && after.rejected == before.rejected + 1,
              "the validation pass zeroed the launch and the backend reported it");
        CHECK(ran.size() == 2 && !ran[1],
              "the refused launch is a DECLINE, exactly like the host route's over-limit refusal");
        CHECK(output_matches(out, 0, kConsumerPattern), "no group ran: canaries intact");
    }

    // ---- Arm 6b: the limit is per AXIS ------------------------------------------------------------
    if (device_limits[1] != std::numeric_limits<uint32_t>::max()) {
        // Derive an actually illegal Y count from this device, not the portable minimum.
        std::vector<ComputeItem> items = {producer({1, device_limits[1] + 1u, 1, 0}, 0x36560072u),
                                          consumer(args_base, 0x36560073u)};
        reset_output(out);
        const Counters before = counters();
        const std::vector<bool> ran = run_submit(items);
        const Counters after = counters();
        CHECK(ran.size() == 2 && ran[0] && !ran[1] && after.rejected == before.rejected + 1,
              "y above its own axis limit is refused even though x=1 is within range");
    }
    {
        // The expected outcome comes from the property query, never the backend's rejection counter.
        // A shared min-of-axes limit must fail this arm when the actual X limit admits 70000.
        const bool expected_refusal = 70000u > device_limits[0];
        std::vector<ComputeItem> items = {producer({70000, 1, 1, 0}, 0x36560074u),
                                          consumer(args_base, 0x36560075u)};
        reset_output(out);
        const Counters before = counters();
        const std::vector<bool> ran = run_submit(items);
        const Counters after = counters();
        CHECK(ran.size() == 2 && ran[0] && ran[1] == !expected_refusal &&
                  after.device == before.device + 1 &&
                  after.rejected == before.rejected + (expected_refusal ? 1u : 0u),
              "x=70000 matches the independently observed X limit");
        CHECK(output_matches(out, expected_refusal ? 0 : kOutputBytes / 16, kConsumerPattern),
              "the independent limit expectation determines all output words and canaries");
        std::printf("  note: x=70000 must be %s on this device\n",
                    expected_refusal ? "refused" : "run");
    }

    // ---- Arm 7: a range that is not wholly inside a retained buffer is resolved on the host --------
    {
        // 8 bytes inside the buffer, 4 outside it: the buffer cannot supply the whole triplet.
        const uint64_t straddle = args_base + kArgumentBufferBytes - 8;
        args[kWords] = 1;                    // the word past the end, in guest memory only
        const Counters before = counters();
        const auto [built, ran] = produce_then_consume(kProducer, straddle, false, 0x36560080u);
        const Counters after = counters();
        CHECK(built && ran && after.device == before.device &&
                  after.fallback == before.fallback + 1,
              "an undersized range falls back to the host read; no device dispatch");
        // Words kWords-2 and kWords-1 are {3,5}; the third is the 1 stored above: x=3.
        CHECK(output_matches(out, 3 * 64, kConsumerPattern),
              "the host fallback launched the guest triplet {3,5,1}");
    }

    // ---- Arm 8: an unaligned range is refused by both routes ----------------------------------------
    {
        const auto [built, ran] = produce_then_consume(kProducer, args_base + 2, false, 0x36560090u);
        CHECK(built && !ran && output_matches(out, 0, kConsumerPattern),
              "an unaligned triplet is declined and launches nothing");
    }

    // ---- Arm 9: arguments with no retained GPU owner are resolved on the host ------------------------
    {
        Storage loose(4096);
        uint32_t* p = loose.words();
        p[0] = 2; p[1] = 1; p[2] = 1;
        std::vector<ComputeItem> items = {consumer(reinterpret_cast<uint64_t>(p), 0x365600A0u)};
        reset_output(out);
        const Counters before = counters();
        const std::vector<bool> ran = run_submit(items);
        const Counters after = counters();
        CHECK(ran.size() == 1 && ran[0] && after.device == before.device &&
                  after.fallback == before.fallback + 1,
              "no authoritative buffer covers the range: host fallback, no device dispatch");
        CHECK(output_matches(out, 2 * 64, kConsumerPattern), "the host fallback launched {2,1,1}");
    }

    // ---- Arm 10: a bound resource that covers the arguments is an alias -------------------------------
    {
        std::vector<ComputeItem> items = {producer(kProducer, 0x365600B0u),
                                          consumer(args_base, 0x365600B1u)};
        ShaderResource alias{};
        alias.cls = ResourceClass::ConstantBuffer;
        alias.binding = 7;
        alias.gpu_addr = args_base;
        alias.size = 64;
        items[1].resources->resources.push_back(alias);
        reset_output(out);
        const Counters before = counters();
        const std::vector<bool> ran = run_submit(items);
        const Counters after = counters();
        CHECK(ran.size() == 2 && ran[0] && ran[1] && after.device == before.device &&
                  after.fallback == before.fallback + 1,
              "an aliasing resource keeps the dispatch off the device route");
        CHECK(output_matches(out, 2 * 64, kConsumerPattern),
              "the host fallback launched the triplet at the aliased range ({2,1,3})");
    }

    // ---- Arm 11: a journaled write to the range in the same submit revokes authority ----------------
    {
        // Journal a write over the argument range between producer and consumer, as a retained GPU
        // operation (or an HLE copy) would. The buffer is no longer provably current: host route.
        std::vector<ComputeItem> items = {producer(kProducer, 0x365600C0u),
                                          consumer(args_base, 0x365600C1u)};
        reset_output(out);
        const Counters before = counters();
        const std::vector<bool> ran = run_submit(items, [&](size_t index) {
            if (index == 0) notify_guest_gpu_write(args_base, 64);
        });
        const Counters after = counters();
        CHECK(ran.size() == 2 && ran[0] && ran[1] && after.device == before.device &&
                  after.fallback == before.fallback + 1,
              "a journaled write over the range revokes the buffer's authority");
        CHECK(output_matches(out, 2 * 64, kConsumerPattern),
              "the host fallback launched the guest's current triplet {2,1,3}");
    }

    // ---- Arm 12: actual validator with synthetic unequal limits ---------------------------------
    // Only one 128-worker group runs the validation shader. The large values are DATA, not launch
    // dimensions, so this safely distinguishes the shared-minimum bug on equal-limit CI devices too.
    struct ValidatorCase {
        std::array<uint32_t, 8> input, expected;
        const char* name;
    };
    const ValidatorCase validator_cases[] = {
        {{70000, 4, 5, 9, 100000, 8, 16, 0xABCD},
         {70000, 4, 5, 0, 100000, 8, 16, 0xABCD}, "unequal limits admit a legal large X count"},
        {{100000, 8, 16, 9, 100000, 8, 16, 0xABCD},
         {100000, 8, 16, 0, 100000, 8, 16, 0xABCD}, "each axis admits its exact boundary"},
        {{1, 9, 1, 0, 100000, 8, 16, 0xABCD},
         {0, 0, 0, 1, 100000, 8, 16, 0xABCD}, "Y above its independent limit refuses the whole launch"},
        {{1, 1, 17, 0, 100000, 8, 16, 0xABCD},
         {0, 0, 0, 1, 100000, 8, 16, 0xABCD}, "Z above its independent limit refuses the whole launch"},
    };
    uint64_t validator_code = 0x365600D0u;
    for (const auto& test : validator_cases) {
        Storage record_storage(4096);
        std::memcpy(record_storage.base, test.input.data(), sizeof(test.input));
        ShaderResource record{};
        record.cls = ResourceClass::ConstantBuffer;
        record.binding = 0;
        record.format = DataFormat::Uint32;
        record.num_components = 1;
        record.stride = 4;
        record.gpu_addr = reinterpret_cast<uint64_t>(record_storage.base);
        record.size = sizeof(test.input);
        ComputeItem item;
        item.resources = std::make_shared<ShaderResourceTable>();
        item.resources->resources = {record};
        item.spirv = build_compute_indirect_dispatch_validate();
        item.code_addr = validator_code++;
        item.user_sgprs = {0};
        item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
        item.launch.local_x = item.launch.threads_x = 128;
        item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
        const bool ran = prosper::frontend::execute_live_compute_items({item});
        CHECK(ran && std::memcmp(record_storage.base, test.expected.data(), sizeof(test.expected)) == 0,
              test.name);
    }

    // ---- Arm 13: post-fence map failure declines without publishing, then recovery ----------------
    // Seed a retained output too. After the failed consumer, a deliberately unjournaled CPU store
    // changes that output's next argument triplet. Failure authority must prevent a stale device pin.
    Storage fault_output(kArgumentBufferBytes);
    const uint64_t fault_base = reinterpret_cast<uint64_t>(fault_output.base);
    std::vector<std::array<uint32_t, 4>> fault_counts{{0, 1, 1, 0}};
    if (excessive_representable) fault_counts.push_back(excessive_counts);
    uint64_t fault_code = 0x365600E0u;
    for (const auto counts : fault_counts) {
        const Counters before = counters();
        reset_output(out);
        std::vector<ComputeItem> items = {
            producer(counts, fault_code++),
            fill_item(fault_output.base, kArgumentBufferBytes, {1, 1, 1, 0}, kRecords, 0, fault_code++),
            fill_item(fault_output.base, kArgumentBufferBytes, kConsumerPattern, 0, args_base, fault_code++),
            consumer(fault_base, fault_code++),
        };
        uint32_t output_writes = 0;
        const auto ran = run_submit(items, [&](size_t index) {
            if (index == 1) {
                set_guest_gpu_write_observer([&](uint64_t address, uint64_t size, const char*) {
                    if (address < fault_base + kArgumentBufferBytes && fault_base < address + size)
                        ++output_writes;
                });
                prosper::frontend::live_compute_fail_next_indirect_readback_for_test();
            } else if (index == 2) {
                auto* words = fault_output.words();
                words[0] = 2; words[1] = words[2] = 1;
            }
        });
        set_guest_gpu_write_observer({});
        const Counters after = counters();
        CHECK(ran.size() == 4 && ran[0] && ran[1] && !ran[2] && ran[3],
              "post-fence indirect result-map failure declines; the following dispatch recovers");
        CHECK(!prosper::frontend::live_compute_indirect_readback_fault_pending_for_test(),
              "the post-fence fault was consumed, not missed or consumed during setup");
        CHECK(output_writes == 0, "an unreadable launch result publishes no output write");
        CHECK(after.device == before.device + 1 && after.fallback == before.fallback + 1,
              "failure authority prevents the following consumer from pinning a stale retained result");
        CHECK(output_matches(out, 2 * 64, kConsumerPattern),
              "recovery consumes the new guest triplet, not the retained GPU's stale x=1");
    }
    if (!excessive_representable)
        std::printf("  note: no representable count exceeds this device; synthetic refusal cases ran\n");

    std::printf(failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
