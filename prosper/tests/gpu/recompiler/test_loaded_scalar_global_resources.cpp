// An owned marker without the unique paired pointer equation must not resurrect a cached shader.
#include "gpu/recompiler/rdna2_loaded_scalar_global_resources.hpp"
#include "gpu/recompiler/compiler_resource_access.hpp"
#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <stdexcept>

using namespace prosper::gpu;
namespace {
const LoadedScalarGlobalRead kRead{28, 167, 10, 16, 9, 0, 16, 32};
struct Input {
    std::array<uint8_t, 8> parent{};
    std::array<uint8_t, 32> target{};
    ShaderResourceTable table;
    Input() {
        uint64_t pointer = 0x200000;
        std::memcpy(parent.data(), &pointer, sizeof(pointer));
        ShaderResource source;
        source.cls = ResourceClass::ConstantBuffer;
        source.format = DataFormat::Uint32;
        source.num_components = 1;
        source.fetch_pc = 28;
        source.gpu_addr = 0x100000;
        source.size = source.owned_nested_snapshot_bytes = 8;
        source.host_data = parent.data();
        source.host_data_size = parent.size();
        table.resources.push_back(source);
        source.fetch_pc = 167;
        source.gpu_addr = 0x200010;
        source.size = source.owned_nested_snapshot_bytes = 32;
        source.host_data = target.data();
        source.host_data_size = target.size();
        table.resources.push_back(source);
    }
};
} // namespace

TEST(LoadedScalarGlobalResources, CompleteUniquePairRetainsBothBindings) {
    Input input;
    input.table.resources[0].binding = 2;
    input.table.resources[1].binding = 3;
    const auto pair = owned_loaded_scalar_global_resources(input.table, kRead);
    ASSERT_TRUE(pair);
    EXPECT_EQ(pair->parent, &input.table.resources[0]);
    EXPECT_EQ(pair->target, &input.table.resources[1]);
}

TEST(LoadedScalarGlobalResources, OrphanParentAndOrphanTargetRefuse) {
    Input input;
    const auto target = input.table.resources[1];
    input.table.resources.pop_back();
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
    input.table.resources = {target};
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
}

TEST(LoadedScalarGlobalResources, ChangedLoadedPointerMustMatchTargetEquation) {
    Input input;
    uint64_t pointer = 0x300000;
    std::memcpy(input.parent.data(), &pointer, sizeof(pointer));
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
    input.table.resources[1].gpu_addr = pointer + kRead.window_offset;
    ASSERT_TRUE(owned_loaded_scalar_global_resources(input.table, kRead));
}

TEST(LoadedScalarGlobalResources, DuplicateOrDescriptorShapeCannotBorrowLaterValidOwner) {
    Input input;
    input.table.resources.push_back(input.table.resources[0]);
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
    input.table.resources.pop_back();
    input.table.resources[0].sgpr_base = 16;
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
    input.table.resources[0].sgpr_base = UINT32_MAX;
    input.table.resources[1].srt_offset = 0;
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
}

TEST(LoadedScalarGlobalResources, AbsentOrPartialBytesRefuseTheWholePair) {
    Input input;
    input.table.resources[0].host_data_size = 4;
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
    input.table.resources[0].host_data_size = 8;
    input.table.resources[1].host_data = nullptr;
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
}

TEST(LoadedScalarGlobalResources, RecordedCompilerBytesReplaceOpaqueHostPointers) {
    Input input;
    input.table.resources[0].host_data = reinterpret_cast<uint8_t*>(uintptr_t(1));
    input.table.resources[1].host_data = reinterpret_cast<uint8_t*>(uintptr_t(2));
    CompilerResourceScope recorded({
        {&input.table.resources[0], true, true, input.parent},
        {&input.table.resources[1], true, true, input.target}});
    ASSERT_TRUE(owned_loaded_scalar_global_resources(input.table, kRead));
}

TEST(LoadedScalarGlobalResources, OpaqueReplayBackingIsIncompleteRatherThanCurrentGuestMemory) {
    Input input;
    CompilerResourceScope recorded({
        {&input.table.resources[0], true, false, {}},
        {&input.table.resources[1], true, true, input.target}});
    EXPECT_THROW(owned_loaded_scalar_global_resources(input.table, kRead), std::runtime_error);
}

TEST(LoadedScalarGlobalResources, ChangedRecordedPointerCannotBorrowLiveParentBytes) {
    Input input;
    std::array<uint8_t, 8> changed{};
    const uint64_t pointer = 0x400000;
    std::memcpy(changed.data(), &pointer, sizeof(pointer));
    CompilerResourceScope recorded({
        {&input.table.resources[0], true, true, changed},
        {&input.table.resources[1], true, true, input.target}});
    EXPECT_FALSE(owned_loaded_scalar_global_resources(input.table, kRead));
}
