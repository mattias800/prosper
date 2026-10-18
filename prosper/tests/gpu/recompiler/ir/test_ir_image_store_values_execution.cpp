// MIMG store/load values on production Vulkan: every lane has different payloads; inactive
// texels and unused rows stay intact. Compare physical channels, never a checksum.
#include "gpu/recompiler/ir/image_store_values_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "shared/live/live_compute.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <system_error>
#include <utility>
#include <vector>
using namespace prosper::gpu;
using namespace prosper::test::ir;
namespace cases = prosper::test::ir::image_store_values;
namespace {
std::vector<uint32_t>& retain(std::vector<uint32_t> data) {
    static std::vector<std::unique_ptr<std::vector<uint32_t>>> owned;
    owned.push_back(std::make_unique<std::vector<uint32_t>>(std::move(data)));
    return *owned.back();
}
class IrImageStoreValuesExecution : public testing::Test {
protected:
    void SetUp() override {
        prosper::frontend::execute_live_compute_items({});
        const auto device = prosper::frontend::live_compute_storage_image_device();
        ASSERT_TRUE(device.initialized);
        if (!device.features.storage_image_capable())
            GTEST_SKIP() << "storage-image device contract";
    }
};
void execute(const cases::Arm& arm, uint32_t wave, uint32_t mask, uint32_t index) {
    auto& buffer = retain(cases::buffer_words(arm, mask));
    auto& image = retain(cases::image_words());
    const auto p = cases::program(arm, wave, reinterpret_cast<uint64_t>(buffer.data()),
                                  reinterpret_cast<uint64_t>(image.data()));
    ComputeItem item;
    item.code_addr = 0x5d54220000ull + uint64_t(index) * 0x100u;
    item.spirv = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                   {RecompileDiagnosticStage::Compute, item.code_addr});
    ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(item.code_addr);
    item.resources = std::make_shared<ShaderResourceTable>(p.table);
    item.user_sgprs = p.config.user_sgprs;
    item.dispatch_index = index;
    item.command_order = uint64_t(index) * 10u;
    item.launch.local_x = item.launch.threads_x = cases::kLanes;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    if (arm.load) {
        for (uint32_t c = 0; c < 4; ++c)
            for (uint32_t lane = 0; lane < cases::kLanes; ++lane)
                EXPECT_EQ(buffer[c * cases::kLanes + lane],
                          cases::expected_load(arm, mask, lane, c))
                    << "lane=" << lane << " component=" << c;
    } else {
        for (uint32_t pixel = 0; pixel < cases::kW * cases::kH; ++pixel)
            for (uint32_t c = 0; c < 4; ++c)
                EXPECT_EQ(image[pixel * 4 + c], cases::expected_store(arm, mask, pixel, c))
                    << "pixel=" << pixel << " component=" << c;
    }
}
TEST_F(IrImageStoreValuesExecution, EveryValueAndRoutingFormMatchesIndependentChannels) {
    uint32_t index = 1;
    for (uint32_t wave : {32u, 64u})
        for (const auto a : cases::arms()) {
            SCOPED_TRACE(testing::Message()
                         << "kind=" << int(a.kind) << " mask=" << a.mask
                         << " selectors=" << a.selectors << " load=" << a.load << " wave=" << wave);
            execute(a, wave, 0, index++);
        }
}
TEST_F(IrImageStoreValuesExecution, PartialAndEmptyExecKeepOldWordsAndGuardTexels) {
    uint32_t index = 1000;
    for (uint32_t wave : {32u, 64u})
        for (const auto a : cases::arms())
            for (uint32_t mask : {1u, 2u}) {
                SCOPED_TRACE(testing::Message()
                             << "kind=" << int(a.kind) << " mask=" << a.mask
                             << " selectors=" << a.selectors << " load=" << a.load
                             << " active=" << mask << " wave=" << wave);
                execute(a, wave, mask, index++);
            }
}
TEST(IrImageStoreValuesCases, ReservedAndDuplicateSelectorsStayRefused) {
    for (bool load : {false, true}) {
        const auto p = cases::program({cases::Kind::WordUint, 15, 0, load}, 64);
        EXPECT_FALSE(recompile_compute(p.code.data(), p.code.size(), &p.table, p.config).empty());
        auto reserved = p.table;
        reserved.resources.back().swizzle[0] = 2;
        EXPECT_TRUE(recompile_compute(p.code.data(), p.code.size(), &reserved, p.config).empty());
        if (!load) {
            auto duplicate = p.table;
            duplicate.resources.back().swizzle[0] = 5;
            EXPECT_TRUE(
                recompile_compute(p.code.data(), p.code.size(), &duplicate, p.config).empty());
        }
    }
}
TEST(IrImageStoreValuesWitness, ExportsEveryExistingForm) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable process launch configuration.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code ec;
        fs::remove_all(compare_witness_root("ir_image_store_values"), ec);
        ASSERT_FALSE(ec);
    }
    ASSERT_FALSE(fs::exists(compare_witness_root("ir_image_store_values")));
    const auto dump = compare_witness_dump("ir_image_store_values");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (uint32_t wave : {32u, 64u})
        for (const auto a : cases::arms()) {
            write_compare_witness_program(dump, 0x5d54220000ull + uint64_t(count) * 0x100u, count,
                                          cases::program(a, wave));
            ++count;
        }
    EXPECT_EQ(count, 318u);
}
} // namespace
