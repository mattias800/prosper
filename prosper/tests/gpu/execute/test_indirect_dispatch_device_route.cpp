// test_indirect_dispatch_device_route -- #3656: an eligible indirect compute dispatch reaches the
// backend with its argument ADDRESS and unresolved launch dimensions, instead of the executor
// copying the 12-byte triplet on the CPU. This file covers the executor half with a recording
// backend; the Vulkan half (vkCmdDispatchIndirect itself) is test_live_compute_indirect_dispatch.
//
// How each arm discriminates. The guest argument memory is initialised to a DELIBERATELY WRONG
// triplet, {7,7,7}, that no consumer in this file should ever launch. An executor that still copies
// the triplet on the CPU hands the backend seven groups and no argument address, so the first arm
// fails on both fields; reverting the change reddens it. The refusal arms are the opposite shape:
// they hold the unchanged ordered CPU copy to its old contract (groups == the guest triplet).
#include "gpu/execute/gpu_execute.hpp"
#include <gtest/gtest.h>
#include "gpu/pm4/pm4_registers.hpp"
#include "hle/dispatch/dispatch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper::gpu;
namespace P = prosper::agc::Pm4;

static int fails = 0;
#define CHECK(c, m) EXPECT_TRUE(c) << (m)

alignas(256) static const uint32_t kNoopCs[] = {0xBF810000u};
struct Recorded {
    uint32_t calls = 0;
    uint64_t args_addr = 0;
    ComputeLaunchDimensions launch{};
    bool indirect = false;
    ComputeCpuFastPath fast_path = ComputeCpuFastPath::None;
};

struct Fixture {
    ShaderReg registers[2] = {{P::COMPUTE_PGM_LO, 0}, {P::COMPUTE_PGM_HI, 0}};
    AgcShaderHeader header{};
    void* registered = nullptr;
    bool ok = false;

    Fixture(const uint32_t* code, size_t bytes) {
        header.file_header = 0x34333231u;
        header.version = 0x18;
        header.sh_registers = registers;
        header.shader_size = static_cast<uint32_t>(bytes);
        header.type = 0;
        header.num_sh_registers = 2;
        auto create_shader = prosper::Hle::lookup("f3dg2CSgRKY");
        ok = create_shader &&
             create_shader(reinterpret_cast<uint64_t>(&registered),
                           reinterpret_cast<uint64_t>(&header), reinterpret_cast<uint64_t>(code),
                           0, 0, 0) == 0 && registered == &header;
    }
    GpuState state() const {
        GpuState st;
        st.sh[P::COMPUTE_PGM_LO] = registers[0].value;
        st.sh[P::COMPUTE_PGM_HI] = registers[1].value;
        return st;
    }
};

static GpuState::Dispatch indirect_dispatch(uint64_t args_addr, uint64_t order = 10) {
    GpuState::Dispatch d;
    d.indirect = true;
    d.indirect_args_addr = args_addr;
    d.command_order = order;
    return d;
}

// Runs one submit against a recording backend. `device_capable` is what the backend declares.
static Recorded run(const GpuState& st, bool device_capable, bool* executed = nullptr) {
    Recorded rec;
    set_submit_compute([&](const std::vector<ComputeItem>& items) {
        for (const ComputeItem& item : items) {
            ++rec.calls;
            rec.args_addr = item.indirect_args_addr;
            rec.launch = item.launch;
            rec.indirect = item.indirect_dispatch;
            rec.fast_path = item.cpu_fast_path;
        }
        return !items.empty();
    });
    set_submit_compute_indirect_dispatch(device_capable);
    const bool ran = execute_nonrender_submit_work(st, 1440);
    if (executed) *executed = ran;
    set_submit_compute({});
    return rec;
}

TEST(IndirectDispatchDeviceRoute, Contract) {
    std::printf("== test_indirect_dispatch_device_route ==\n");
    prosper::register_agc_hle();   // the shader registry sceAgcCreateShader fills
    Fixture noop(kNoopCs, sizeof(kNoopCs));
    CHECK(noop.ok, "register a descriptor-free compute program");
    if (!noop.ok) FAIL() << "legacy early exit";

    alignas(4) uint32_t wrong[3] = {7, 7, 7};        // never a valid launch for these tests
    alignas(4) uint32_t four_words[4] = {2, 3, 4, 5};
    const uint64_t wrong_addr = reinterpret_cast<uint64_t>(wrong);

    // ---- Arm 1: the device route -------------------------------------------------------------
    {
        GpuState st = noop.state();
        st.dispatches.push_back(indirect_dispatch(wrong_addr));
        const IndirectDispatchStats before = indirect_dispatch_stats();
        const Recorded rec = run(st, true);
        const IndirectDispatchStats after = indirect_dispatch_stats();
        CHECK(rec.calls == 1, "the dispatch reaches the backend exactly once");
        CHECK(rec.args_addr == wrong_addr,
              "the backend is handed the guest ARGUMENT ADDRESS, not resolved dimensions");
        CHECK(rec.launch.groups_x == 0 && rec.launch.groups_y == 0 && rec.launch.groups_z == 0 &&
                  rec.launch.threads_x == 0 && rec.launch.threads_y == 0 &&
                  rec.launch.threads_z == 0,
              "the launch is UNKNOWN (zero), so the stale {7,7,7} cannot have been copied");
        CHECK(rec.indirect, "the item keeps its indirect provenance for F8 reporting");
        CHECK(after.device_resolved == before.device_resolved + 1 &&
                  after.cpu_resolved == before.cpu_resolved,
              "the census counts one device-resolved dispatch and no CPU resolution");
        CHECK(wrong[0] == 7 && wrong[1] == 7 && wrong[2] == 7, "guest argument memory untouched");
    }

    // ---- Arm 2: a backend that does not declare the capability keeps the CPU copy --------------
    {
        GpuState st = noop.state();
        st.dispatches.push_back(indirect_dispatch(wrong_addr));
        const IndirectDispatchStats before = indirect_dispatch_stats();
        const Recorded rec = run(st, false);
        const IndirectDispatchStats after = indirect_dispatch_stats();
        CHECK(rec.calls == 1 && rec.args_addr == 0 && rec.launch.groups_x == 7 &&
                  rec.launch.groups_y == 7 && rec.launch.groups_z == 7,
              "no capability: the ordered CPU copy resolves {7,7,7} exactly as before");
        CHECK(after.cpu_resolved == before.cpu_resolved + 1 &&
                  after.device_resolved == before.device_resolved,
              "no capability: counted as a CPU resolution");
    }

    // ---- Arm 3: thread-dimension mode is refused ----------------------------------------------
    {
        GpuState st = noop.state();
        GpuState::Dispatch d = indirect_dispatch(wrong_addr);
        d.modifier |= P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_MASK
                      << P::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS_SHIFT;
        st.dispatches.push_back(d);
        const Recorded rec = run(st, true);
        CHECK(rec.calls == 1 && rec.args_addr == 0 && rec.launch.threads_x == 7 &&
                  rec.launch.groups_x == 7,
              "thread-dimension mode needs the CPU count: refused, resolved from the guest triplet");
    }

    // ---- Arm 4: an unaligned argument address is refused ---------------------------------------
    {
        GpuState st = noop.state();
        st.dispatches.push_back(
            indirect_dispatch(reinterpret_cast<uint64_t>(four_words) + 1));
        bool executed = true;
        const Recorded rec = run(st, true, &executed);
        CHECK(rec.calls == 0 && !executed,
              "an unaligned triplet is refused by both routes and never reaches the backend");
    }

    // ---- Arm 5: an unreadable argument range is refused ----------------------------------------
    {
        GpuState st = noop.state();
        st.dispatches.push_back(indirect_dispatch(0x10));
        bool executed = true;
        const Recorded rec = run(st, true, &executed);
        CHECK(rec.calls == 0 && !executed, "an unreadable triplet never reaches the backend");
    }

    // ---- Arm 6: zero counts are the device's to resolve -----------------------------------------
    {
        alignas(4) uint32_t zeros[3] = {0, 0, 0};
        GpuState st = noop.state();
        st.dispatches.push_back(indirect_dispatch(reinterpret_cast<uint64_t>(zeros)));
        const Recorded rec = run(st, true);
        CHECK(rec.calls == 1 && rec.args_addr == reinterpret_cast<uint64_t>(zeros),
              "zero counts are not read on the CPU: the device sees them and launches no groups");
    }

    // ---- Arm 7: every launch-dependent specialization refuses an unresolved launch ------------
    // Hand-built items, one token each. The base item is launch-free, so a predicate that always
    // answers false (or always true) cannot satisfy both halves.
    {
        ComputeItem base;
        base.indirect_args_addr = wrong_addr;
        base.indirect_dispatch = true;
        CHECK(gpu_indirect_dispatch_launch_free(base), "control: an item with no launch proof is launch-free");
        struct Case { const char* name; void (*apply)(ComputeItem&); };
        const Case cases[] = {
            {"CPU fast path (launch-shaped fill)", [](ComputeItem& i) {
                i.cpu_fast_path = ComputeCpuFastPath::FillSgprUvec4; }},
            {"null-guarded raw store proof", [](ComputeItem& i) {
                i.null_guarded_raw_store_validated = true; }},
            {"nullable-output raw buffer proof", [](ComputeItem& i) {
                i.nullable_output_raw_buffer_validated = true; }},
            {"GTA V cf9200 no-backing proof", [](ComputeItem& i) {
                i.gta5_cf9200_no_backing_validated = true; }},
            {"trip-bound witness (sized by the group count)", [](ComputeItem& i) {
                i.trip_witness_instrumented = true; }},
            {"exact thread extent guard", [](ComputeItem& i) {
                i.recompile_config.exact_thread_extent = true; }},
            {"specialized thread extent x", [](ComputeItem& i) {
                i.recompile_config.threads_x = 64; }},
            {"specialized thread extent y", [](ComputeItem& i) {
                i.recompile_config.threads_y = 2; }},
            {"specialized thread extent z", [](ComputeItem& i) {
                i.recompile_config.threads_z = 2; }},
            {"a launch that was resolved after all", [](ComputeItem& i) {
                i.launch.groups_x = 1; }},
        };
        for (const Case& c : cases) {
            ComputeItem item = base;
            c.apply(item);
            char message[160];
            std::snprintf(message, sizeof(message), "refused: %s", c.name);
            CHECK(!gpu_indirect_dispatch_launch_free(item), message);
        }
    }

    // ---- Arm 8: an argument range that any bound resource covers is an alias -------------------
    {
        alignas(4) uint32_t backing[16] = {};
        auto resource_at = [](const void* base, uint32_t bytes) {
            ShaderResource r{};
            r.cls = ResourceClass::ConstantBuffer;
            r.gpu_addr = reinterpret_cast<uint64_t>(base);
            r.size = bytes;
            return r;
        };
        ComputeItem item;
        item.indirect_args_addr = reinterpret_cast<uint64_t>(&backing[4]);   // bytes [16, 28)
        CHECK(!gpu_indirect_dispatch_args_alias_resource(item), "control: no resource table, no alias");
        item.resources = std::make_shared<ShaderResourceTable>();
        item.resources->resources = {resource_at(&backing[8], 32)};   // starts at byte 32
        CHECK(!gpu_indirect_dispatch_args_alias_resource(item), "a disjoint resource is not an alias");
        item.resources->resources = {resource_at(&backing[0], 64)};   // covers everything
        CHECK(gpu_indirect_dispatch_args_alias_resource(item), "a resource covering the range is an alias");
        item.resources->resources = {resource_at(&backing[6], 16)};   // [24, 40) overlaps the last word
        CHECK(gpu_indirect_dispatch_args_alias_resource(item), "a resource overlapping the tail is an alias");
    }

    EXPECT_EQ(fails, 0);
}
