// ds_read_b32 ... gds in a compute stage (#4553).
//
// GDS is the device-global data share. prosper keeps it as one 64 KiB buffer every dispatch
// binds, and lowered the plain store (ds_write_b32 ... gds) and the append/consume counters
// against it, but refused the plain read. MOUSE: P.I. For Hire runs a small compute program
// every frame that reads a word out of GDS and stores it into a buffer (`d8da0000 00000000`,
// ds_read_b32 v0, v0 gds), which is how an engine copies a counter into the arguments of a
// later draw; with the read refused, the whole dispatch was skipped.
//
// These cases run on the production compute backend: one dispatch stores, a later one reads
// what it stored and stores it somewhere else, and the test looks at the buffer afterwards.
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kRead = 0xd8da0000u;    // ds_read_b32 ... gds
constexpr uint32_t kWrite = 0xd8360000u;   // ds_write_b32 ... gds

// One GDS instance shared by every dispatch of a case, as on the console.
struct Gds {
    std::array<uint8_t, size_t{64} * 1024> bytes{};
    std::shared_ptr<ShaderResourceTable> table = std::make_shared<ShaderResourceTable>();

    Gds() {
        ShaderResource resource;
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 1;
        resource.binding = kComputeInternalGdsBinding;
        resource.size = static_cast<uint32_t>(bytes.size());
        resource.stride = 4;
        resource.host_data = bytes.data();
        resource.host_data_size = bytes.size();
        table->resources.push_back(resource);
    }
    uint32_t word(uint32_t byte_address) const {
        uint32_t value = 0;
        std::memcpy(&value, bytes.data() + byte_address, sizeof value);
        return value;
    }
    void set(uint32_t byte_address, uint32_t value) {
        std::memcpy(bytes.data() + byte_address, &value, sizeof value);
    }
    ComputeItem dispatch(const std::vector<uint32_t>& code, uint32_t threads, uint64_t order) {
        ComputeItem item;
        item.spirv =
            recompile_compute(code.data(), code.size(), table.get(), ComputeShaderConfig{});
        item.resources = table;
        item.launch.threads_x = threads;
        item.launch.local_x = threads;
        item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
        item.dispatch_index = order;
        item.command_order = order;
        return item;
    }
};

// v0 = address ; ds_read_b32 v1, v0 gds ; v0 = destination ; ds_write_b32 v0, v1 gds
std::vector<uint32_t> copy(uint32_t from, uint32_t to) {
    return {
        0x7e0002ffu, from,          // v_mov_b32 v0, from
        kRead,       0x01000000u,   // ds_read_b32 v1, v0 gds
        0x7e0002ffu, to,            // v_mov_b32 v0, to
        kWrite,      0x00000100u,   // ds_write_b32 v0, v1 gds
        0xbf810000u,                // s_endpgm
    };
}
}   // namespace

TEST(ComputeGdsRead, FixtureDecodesAsDescribed) {
    const auto code = copy(4, 8);
    std::vector<Rdna2Inst> ins;
    ASSERT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size());
    ASSERT_EQ(ins.size(), 5u);
    EXPECT_EQ(ins[1].fmt, Rdna2Format::DS);
    EXPECT_EQ(ins[1].opcode, 0x36u) << "ds_read_b32";
    EXPECT_TRUE(ins[1].ds_gds);
    EXPECT_EQ(ins[1].dst.value, 1);
    EXPECT_EQ(ins[1].src[0].value, 0);
    EXPECT_EQ(ins[3].fmt, Rdna2Format::DS);
    EXPECT_EQ(ins[3].opcode, 0x0du) << "ds_write_b32";
    EXPECT_TRUE(ins[3].ds_gds);
    // The instruction MOUSE: P.I. For Hire's program was refused at.
    const uint32_t mouse[] = {0xd8da0000u, 0x00000000u, 0xbf810000u};
    std::vector<Rdna2Inst> refused;
    ASSERT_EQ(rdna2_walk(mouse, 3, refused), 3u);
    EXPECT_EQ(refused[0].fmt, Rdna2Format::DS);
    EXPECT_EQ(refused[0].opcode, 0x36u);
    EXPECT_TRUE(refused[0].ds_gds);
}

TEST(ComputeGdsRead, ALaterDispatchReadsWhatAnEarlierOneStored) {
    Gds gds;
    // Dispatch 10 stores at byte 4; dispatch 20 reads byte 4 and stores what it read at byte 8.
    const std::vector<uint32_t> store{
        0x7e0002ffu, 0x00000004u, 0x7e0202ffu, 0xdeadbeefu, kWrite, 0x00000100u, 0xbf810000u,
    };
    const ComputeItem first = gds.dispatch(store, 1, 10);
    const ComputeItem second = gds.dispatch(copy(4, 8), 1, 20);
    ASSERT_FALSE(first.spirv.empty());
    ASSERT_FALSE(second.spirv.empty()) << "a GDS read must compile";
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({first, second}));
    EXPECT_EQ(gds.word(4), 0xdeadbeefu);
    EXPECT_EQ(gds.word(8), 0xdeadbeefu) << "the read returned what the earlier dispatch stored";
    EXPECT_EQ(gds.word(0), 0u);
    EXPECT_EQ(gds.word(12), 0u);
}

TEST(ComputeGdsRead, TheAddressWrapsAtTheShareAsTheStoreDoes) {
    Gds gds;
    gds.set(4, 0x0badf00du);
    // Byte address 0x10004 is byte 4 of the 64 KiB share.
    const ComputeItem item = gds.dispatch(copy(0x00010004u, 0x20), 1, 10);
    ASSERT_FALSE(item.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    EXPECT_EQ(gds.word(0x20), 0x0badf00du);
}

TEST(ComputeGdsRead, TheInstructionOffsetIsAddedToTheAddress) {
    Gds gds;
    gds.set(0x20, 0x11111111u);   // what the address register alone names
    gds.set(0x28, 0x600dcafeu);   // what address + offset names
    // v0 = 0x20 ; ds_read_b32 v1, v0 offset:8 gds ; v0 = 0x30 ; ds_write_b32 v0, v1 gds
    const std::vector<uint32_t> code{
        0x7e0002ffu, 0x00000020u, kRead | 0x0008u, 0x01000000u, 0x7e0002ffu,
        0x00000030u, kWrite,      0x00000100u,     0xbf810000u,
    };
    {
        std::vector<Rdna2Inst> ins;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size());
        ASSERT_EQ(ins[1].opcode, 0x36u);
        ASSERT_TRUE(ins[1].ds_gds);
        ASSERT_EQ(ins[1].literal, 8u);
    }
    const ComputeItem item = gds.dispatch(code, 1, 10);
    ASSERT_FALSE(item.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    EXPECT_EQ(gds.word(0x30), 0x600dcafeu);
}

TEST(ComputeGdsRead, AnInactiveLaneKeepsItsRegister) {
    Gds gds;
    gds.set(0, 0x22222222u);   // what an unpredicated read on the other lanes would return
    gds.set(4, 0xdeadbeefu);
    // Four lanes. Each lane's v0 becomes its own byte offset; v1 starts as 0x11111111 on every
    // lane; EXEC is narrowed to lane 0, which alone reads byte 4 into v1; then every lane stores
    // its v1 at 0x100 + its offset.
    const std::vector<uint32_t> code{
        0x34000082u,   // v_lshlrev_b32 v0, 2, v0          v0 = lane * 4
        0x7e0202ffu, 0x11111111u,   // v_mov_b32 v1, 0x11111111
        0x7da40080u,   // v_cmpx_eq_u32 0, v0              EXEC = lane 0
        0x7e0402ffu, 0x00000004u,   // v_mov_b32 v2, 4
        kRead,       0x01000002u,   // ds_read_b32 v1, v2 gds
        0xbefe04c1u,   // s_mov_b64 exec, -1
        0x4a0600ffu, 0x00000100u,   // v_add_nc_u32 v3, 0x100, v0
        kWrite,      0x00000103u,   // ds_write_b32 v3, v1 gds
        0xbf810000u,
    };
    {
        std::vector<Rdna2Inst> ins;
        ASSERT_EQ(rdna2_walk(code.data(), code.size(), ins), code.size());
        ASSERT_EQ(ins.size(), 9u);
        ASSERT_EQ(ins[4].fmt, Rdna2Format::DS);
        ASSERT_EQ(ins[4].opcode, 0x36u);
        ASSERT_EQ(ins[4].dst.value, 1);
        ASSERT_EQ(ins[4].src[0].value, 2);
        ASSERT_EQ(ins[6].fmt, Rdna2Format::VOP2);
        ASSERT_EQ(ins[6].dst.value, 3);
        ASSERT_EQ(ins[7].src[0].value, 3);
    }
    const ComputeItem item = gds.dispatch(code, 4, 10);
    ASSERT_FALSE(item.spirv.empty());
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    EXPECT_EQ(gds.word(0x100), 0xdeadbeefu) << "lane 0 read byte 4";
    for (uint32_t lane = 1; lane < 4; ++lane)
        EXPECT_EQ(gds.word(0x100 + lane * 4), 0x11111111u) << "lane " << lane;
}
