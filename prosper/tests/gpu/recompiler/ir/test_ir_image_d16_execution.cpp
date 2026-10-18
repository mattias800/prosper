// D16 packs selected components LSB first. Check both packed words and the following guard
// VGPRs independently; partial and empty EXEC must preserve each lane's old values.
#include "gpu/recompiler/ir/image_d16_cases.hpp"
#include "gpu/recompiler/ir/vector_compare_harness.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
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
namespace cases = prosper::test::ir::image_d16;
namespace {
std::vector<uint32_t>& retain(std::vector<uint32_t> words) {
    static std::vector<std::unique_ptr<std::vector<uint32_t>>> owned;
    owned.push_back(std::make_unique<std::vector<uint32_t>>(std::move(words)));
    return *owned.back();
}
class IrImageD16Execution : public testing::Test {
protected:
    void SetUp() override {
        prosper::frontend::execute_live_compute_items({});
        ASSERT_TRUE(prosper::frontend::live_compute_storage_image_device().initialized);
    }
};
void execute(const cases::Arm& arm, uint32_t wave, uint32_t mask, uint32_t index, bool d16 = true) {
    const auto in = cases::inputs(arm, mask);
    auto& buffer = retain(cases::buffer_words(in));
    auto& texture = retain(cases::texture_words());
    const auto p = cases::program(arm, wave, reinterpret_cast<uint64_t>(buffer.data()),
                                  reinterpret_cast<uint64_t>(texture.data()), d16);
    const uint64_t address = 0x5d54200000ull + uint64_t(index) * 0x100u;
    ComputeItem item;
    item.spirv = recompile_compute(p.code.data(), p.code.size(), &p.table, p.config,
                                   {RecompileDiagnosticStage::Compute, address});
    ASSERT_FALSE(item.spirv.empty()) << last_terminal_reject_reason(address);
    item.resources = std::make_shared<ShaderResourceTable>(p.table);
    item.user_sgprs = p.config.user_sgprs;
    item.code_addr = address;
    item.dispatch_index = index;
    item.command_order = uint64_t(index) * 10u;
    item.launch.local_x = item.launch.threads_x = cases::kLanes;
    item.launch.local_y = item.launch.local_z = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    ASSERT_TRUE(prosper::frontend::execute_live_compute_items({item}));
    for (uint32_t c = 0; c < 4; ++c)
        for (uint32_t lane = 0; lane < cases::kLanes; ++lane)
            EXPECT_EQ(buffer[c * cases::kLanes + lane],
                      (d16 ? cases::expected(arm, in, lane, c)
                           : cases::texel(cases::x(lane), cases::y(lane), c).f32))
                << "lane=" << lane << " result=" << c;
}
TEST_F(IrImageD16Execution, DwordControlPreservesRoundingInputsBeforePacking) {
    uint32_t index = 3000;
    for (bool nsa : {false, true})
        for (uint32_t wave : {32u, 64u}) execute({0, 15, nsa}, wave, 0, index++, false);
}
TEST_F(IrImageD16Execution, EveryMaskPacksHalvesAndPreservesFollowingRegisters) {
    uint32_t index = 1;
    for (uint32_t wave : {32u, 64u})
        for (const auto a : cases::arms()) {
            SCOPED_TRACE(testing::Message() << "op=" << a.opcode << " dmask=" << a.dmask
                                            << " nsa=" << a.nsa << " wave=" << wave);
            execute(a, wave, 0, index++);
        }
}
TEST_F(IrImageD16Execution, PartialAndEmptyExecPreserveEveryDestinationWord) {
    uint32_t index = 1000;
    for (uint32_t wave : {32u, 64u})
        for (const auto a : cases::arms())
            for (uint32_t mask : {1u, 2u}) {
                SCOPED_TRACE(testing::Message() << "op=" << a.opcode << " dmask=" << a.dmask
                                                << " nsa=" << a.nsa << " mask=" << mask);
                execute(a, wave, mask, index++);
            }
}
TEST(IrImageD16Cases, IntegerD16LoadsStayRefusedWithDwordControl) {
    for (bool nsa : {false, true}) {
        const cases::Arm a{0, 15, nsa};
        const auto good = cases::program(a, 64, 0, 0, false, true);
        EXPECT_FALSE(recompile_compute(good.code.data(), good.code.size(), &good.table, good.config)
                         .empty());
        const auto bad = cases::program(a, 64, 0, 0, true, true);
        EXPECT_TRUE(
            recompile_compute(bad.code.data(), bad.code.size(), &bad.table, bad.config).empty());
    }
}
TEST(IrImageD16Witness, ExportsEveryExistingForm) {
    namespace fs = std::filesystem;
    // NOLINTNEXTLINE(concurrency-mt-unsafe): immutable process launch configuration.
    if (!std::getenv("PROSPER_IR_WITNESS_DIR")) {
        std::error_code ec;
        fs::remove_all(compare_witness_root("ir_image_d16"), ec);
        ASSERT_FALSE(ec);
    }
    ASSERT_FALSE(fs::exists(compare_witness_root("ir_image_d16")));
    const auto dump = compare_witness_dump("ir_image_d16");
    ASSERT_TRUE(fs::create_directories(dump));
    size_t count = 0;
    for (uint32_t wave : {32u, 64u})
        for (const auto a : cases::arms()) {
            write_compare_witness_program(dump, 0x5d54200000ull + uint64_t(count) * 0x100u, count,
                                          cases::program(a, wave));
            ++count;
        }
    EXPECT_EQ(count, 136u);
}
} // namespace
