#include "bpermute_spirv_oracle.hpp"
#include "fixtures/fragment_packet_definedness_fixture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/fragment_packet_vgpr_requirements.hpp"
#include "gpu/recompiler/raster_quad_collector.hpp"
#include <gtest/gtest.h>
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace {
using namespace prosper::gpu;
namespace f = prosper::test::fragment_definedness;
namespace r = prosper::test::fragment_resource_packet;
void retain(const FragmentPacketProgram& p, const std::string& name) {
    // Diagnostic CPU-test-only SOURCE retention; unset creates no files and changes no lowering.
    const char* root = std::getenv("PROSPER_VGPR_DEFINEDNESS_SPV_DIRECTORY");
    if (!root || !*root) return;
    std::filesystem::create_directories(root);
    std::ofstream file(std::filesystem::path(root) / (name + ".spv"), std::ios::binary);
    file.write(reinterpret_cast<const char*>(p.spirv.data()),
               static_cast<std::streamsize>(p.spirv.size() * 4));
    file.close();
    EXPECT_TRUE(bool(file)) << name << ": actual SOURCE retention";
}
std::vector<uint32_t> evaluate(const FragmentPacketProgram& p, const std::string& name) {
    bpermute_oracle::Interpreter vm(p.spirv);
    auto words = vm.run_packet(p.input_words, p.output_words);
    EXPECT_TRUE(vm.error.empty()) << name << ": typed SOURCE " << vm.error;
    retain(p, name);
    return words;
}
std::vector<uint32_t> evaluate(const FragmentResourcePacketProgram& p, const std::string& name) {
    bpermute_oracle::Interpreter vm(p.packet.spirv);
    for (uint32_t image = 0; image < p.images.size(); ++image)
        for (const auto& mip : p.images[image].mips)
            vm.sampled_images[16 + image].push_back({mip.width, mip.height, mip.texels});
    auto words = vm.run_packet(p.packet.input_words, p.packet.output_words);
    EXPECT_TRUE(vm.error.empty()) << name << ": typed SOURCE " << vm.error;
    retain(p.packet, name);
    return words;
}
bool gap(const FragmentPacketPreparation& p, const char* reason) {
    return std::find(p.unmet.begin(), p.unmet.end(), reason) != p.unmet.end();
}
}   // namespace

TEST(FragmentPacketDefinedness, ActualSourceMaskBranchPeerAndRawOutputRails) {
    reset_float_controls_support_for_test();
    uint32_t attempts = 0;
    for (const auto& c : f::cases()) {
        SCOPED_TRACE(c.name);
        const auto p = recompile_fragment_packet(c.packet);
        ASSERT_FALSE(p.spirv.empty()) << p.rejection;
        ASSERT_NE(p.vgpr_status_offset, UINT32_MAX);
        ++attempts;
        const auto words = evaluate(p, c.name);
        const auto result = decode_fragment_packet(p, words, true);
        EXPECT_EQ(result.exports, c.expected) << "independent complete all64 raw sink";
        if (c.failure_kind) {
            EXPECT_FALSE(result.rejection.empty());
            EXPECT_TRUE(result.exports.empty());
            EXPECT_EQ(result.lane, c.failure_lane);
            EXPECT_EQ(result.pc, c.failure_pc);
            EXPECT_EQ(result.reg, c.failure_reg);
            EXPECT_EQ(result.kind, c.failure_kind);
        } else
            EXPECT_TRUE(result.rejection.empty()) << result.rejection;
    }
    EXPECT_EQ(attempts, f::kIntegerRailCount);
    std::fprintf(stderr, "[vgpr-definedness-source] integer_evaluation_attempts=%u expected=%u\n",
                 attempts, f::kIntegerRailCount);
}

TEST(FragmentPacketDefinedness, NumericExecMaskRequiresCompleteReachingWords) {
    for (uint32_t absent : {20u, 21u}) {
        auto input = f::numeric_exec_pair(0);
        std::erase_if(input.sgprs, [&](const auto& word) { return word.first == absent; });
        const auto p = recompile_fragment_packet(input);
        EXPECT_TRUE(p.spirv.empty());
        EXPECT_EQ(p.rejection, "packet-sgpr-read-before-definition");
    }
    // The same instruction stream with a genuine present-zero high word is executable above.
    // One conditional writer cannot replace a missing entry word on its skipped predecessor.
    auto join = f::numeric_exec_pair(1);
    std::vector<uint32_t> high_zero;
    f::fp::smov(high_zero, 21, 0);
    join.guest_code.insert(join.guest_code.begin() + 1, high_zero.begin(), high_zero.end());
    join.guest_code.insert(join.guest_code.begin() + 1, 0xbf840002u);
    std::erase_if(join.sgprs, [](const auto& word) { return word.first == 21; });
    for (bool scc : {false, true}) {
        join.scc = scc;
        const auto p = recompile_fragment_packet(join);
        EXPECT_TRUE(p.spirv.empty());
        EXPECT_EQ(p.rejection, "packet-sgpr-read-before-definition");
    }
    for (bool boundary : {false, true}) {
        SCOPED_TRACE(boundary ? "dispatcher reload" : "same emitted case");
        const auto p = recompile_fragment_packet(f::saved_mask_high_overwrite(boundary));
        EXPECT_TRUE(p.spirv.empty());
        EXPECT_NE(p.rejection.find("packet-exec-mask-source-words-unavailable"), std::string::npos)
            << p.rejection;
        EXPECT_NE(p.rejection.find(boundary ? "pc=4" : "pc=3"), std::string::npos) << p.rejection;
    }
}

TEST(FragmentPacketDefinedness, OwnedResourceChainUsesWriterOnlyAndInactiveObservedWords) {
    uint32_t attempts = 0;
    for (bool lod : {false, true})
        for (bool inactive : {false, true}) {
            const auto input = f::resource_chain(lod, inactive);
            const auto p = recompile_fragment_resource_packet(input);
            ASSERT_FALSE(p.packet.spirv.empty()) << p.packet.rejection;
            ASSERT_NE(p.packet.vgpr_status_offset, UINT32_MAX);
            ++attempts;
            const auto words =
                evaluate(p, "resource_" + std::to_string(lod) + "_" + std::to_string(inactive));
            const auto result =
                decode_fragment_resource_packet(p, words, true, input.device.device_identity);
            EXPECT_TRUE(result.rejection.empty()) << result.rejection;
            EXPECT_EQ(result.exports, r::expected_chain(input, lod))
                << "P1/P2/FP/SMEM/explicit image/raw EXP/peer sink";
        }
    EXPECT_EQ(attempts, 4u);
    std::fprintf(stderr, "[vgpr-definedness-source] resource_evaluation_attempts=%u expected=4\n",
                 attempts);
}

TEST(FragmentPacketDefinedness, ImplicitP2AndWideImageReadsCannotBorrowAllocatedZeros) {
    const auto check = [&](FragmentResourcePacket input, uint32_t pc, uint32_t reg,
                           const char* name) {
        const auto p = recompile_fragment_resource_packet(input);
        ASSERT_FALSE(p.packet.spirv.empty()) << p.packet.rejection;
        const auto words = evaluate(p, name);
        const auto result =
            decode_fragment_resource_packet(p, words, true, input.device.device_identity);
        EXPECT_TRUE(result.exports.empty());
        EXPECT_FALSE(result.rejection.empty());
        EXPECT_EQ(result.failure, FragmentPacketRuntimeFailure::UndefinedVgpr);
        EXPECT_EQ(result.pc, pc);
        EXPECT_EQ(result.vgpr, reg);
    };
    auto p2 = f::resource_chain(false, false);
    p2.invocation.guest_code[4] = 0x7e260280u;   // actual MOV v19,inline0 instead of P1 v10
    check(p2, 5, 10, "implicit_p2_old_destination_absent");
    auto wide = f::resource_chain(false, false);
    wide.invocation.guest_code[6] = wide.invocation.guest_code[7] = 0x7e260280u;
    // Keep the absent middle coordinate out of the preceding EXP so the MIMG's implicit
    // source range, not an earlier raw-export observation, is the actual first failing read.
    wide.invocation.guest_code[11] &= ~2u;
    check(wide, wide.images[0].pc, 11, "wide_image_middle_coordinate_absent");
    auto lod = f::resource_missing_lod();
    check(lod, lod.images[0].pc, 12, "explicit_lod_third_coordinate_absent");
}

TEST(FragmentPacketDefinedness, CacheBudgetCoversCompleteAlignedMapNodes) {
    const auto input = f::cases()[1].packet;
    const auto requirements = fragment_packet_vgpr_requirements(input.guest_code);
    ASSERT_TRUE(requirements.rejection.empty());
    ASSERT_EQ(requirements.reads.size(), 3u);
    using ReadMap = decltype(requirements.reads);
    // Independent lower budget: aligned WHOLE pair plus four pointer words, not separately
    // summed key/vector sizes. The former 60-byte estimate fails this 64-bit 64-byte floor.
    constexpr uint64_t alignment = alignof(ReadMap::value_type);
    constexpr uint64_t minimum_node =
        ((sizeof(ReadMap::value_type) + 4 * sizeof(void*) + alignment - 1) / alignment) * alignment;
    uint64_t minimum = sizeof(requirements) + requirements.rejection.capacity();
    for (const auto& [pc, accesses] : requirements.reads)
        minimum += minimum_node + accesses.capacity() * sizeof(FragmentPacketVgprAccess);
    EXPECT_GE(requirements.retained_bytes(), minimum);
    if constexpr (sizeof(void*) == 8)
        EXPECT_GT(minimum_node, sizeof(uint32_t) + sizeof(std::vector<FragmentPacketVgprAccess>) +
                                    4 * sizeof(void*));
}

TEST(FragmentPacketDefinedness, FirstFailureAcrossResourceAndValidityChannelsIsChronological) {
    auto input = f::resource_chain(false, false);
    input.invocation.vgprs[0].available_mask = 0;   // P1 first source failure at pc4
    input.invocation.sgprs[1].second ^= 1u;   // original s0 descriptor mismatch at earlier pc1
    const auto early = recompile_fragment_resource_packet(input);
    ASSERT_FALSE(early.packet.spirv.empty()) << early.packet.rejection;
    auto words = evaluate(early, "resource_failure_precedes_validity");
    auto result = decode_fragment_resource_packet(early, words, true, input.device.device_identity);
    EXPECT_TRUE(result.exports.empty());
    EXPECT_EQ(result.pc, 1u);
    EXPECT_EQ(result.failure, FragmentPacketRuntimeFailure::DescriptorMismatch);
    input.invocation.sgprs[1].second ^= 1u;
    input.invocation.sgprs[0].second ^=
        2u;   // M0 mismatch occurs at pc4, AFTER direct validity precheck
    const auto first = recompile_fragment_resource_packet(input);
    ASSERT_FALSE(first.packet.spirv.empty()) << first.packet.rejection;
    words = evaluate(first, "validity_precedes_same_pc_resource_failure");
    result = decode_fragment_resource_packet(first, words, true, input.device.device_identity);
    EXPECT_TRUE(result.exports.empty());
    EXPECT_EQ(result.pc, 4u);
    EXPECT_EQ(result.vgpr, 0u);
    EXPECT_EQ(result.failure, FragmentPacketRuntimeFailure::UndefinedVgpr);
}

TEST(FragmentPacketDefinedness, WholeStatusTailRequiredBeforeAnyExportPublication) {
    const auto input = f::cases().front().packet;
    const auto p = recompile_fragment_packet(input);
    ASSERT_FALSE(p.spirv.empty()) << p.rejection;
    const auto good = evaluate(p, "status_wire_controls");
    ASSERT_FALSE(decode_fragment_packet(p, good, true).exports.empty());
    GraphicsWaveStagePlan plan;
    plan.assembly = GraphicsWaveAssembly::FragmentPrimitiveQuadOrder;
    plan.packets = {input, input};
    plan.invocations.resize(2);
    GraphicsWaveOutputTransaction transaction;
    std::string transaction_refusal;
    ASSERT_TRUE(
        validate_graphics_wave_outputs(plan, {good, good}, transaction, transaction_refusal))
        << transaction_refusal;
    EXPECT_EQ(transaction.records.size(), 2u);
    const auto refused = [&](const auto& program, const auto& words, bool completed = true) {
        const auto result = decode_fragment_packet(program, words, completed);
        EXPECT_TRUE(result.exports.empty());
        EXPECT_FALSE(result.rejection.empty());
        return result;
    };
    const auto malformed_status = [&](const auto& words) {
        const auto result = refused(p, words);
        EXPECT_EQ(result.rejection, "packet-vgpr-status-record-invalid");
        EXPECT_FALSE(result.vgpr_status_validated);
    };
    refused(p, good, false);
    auto short_tail = good;
    short_tail.pop_back();
    refused(p, short_tail);
    auto bad = good;
    bad[p.vgpr_status_offset + 63 * 4] ^= 1u;
    malformed_status(bad);
    bad = good;
    bad[p.vgpr_status_offset + 63 * 4 + 1] = 0;
    malformed_status(bad);   // success cannot name a PC
    bad = good;
    bad[p.vgpr_status_offset + 63 * 4 + 3] = 99;
    malformed_status(bad);
    bad = good;
    const auto late = p.vgpr_status_offset + 63 * 4;
    bad[late + 1] = 1;
    bad[late + 2] = 1;
    bad[late + 3] = 4;   // exact real EXP site, late lane
    const auto failed = decode_fragment_packet(p, bad, true);
    EXPECT_TRUE(failed.exports.empty());
    EXPECT_TRUE(failed.vgpr_status_validated);
    EXPECT_EQ(failed.rejection, "packet-vgpr-raw-export-unavailable");
    EXPECT_EQ(failed.lane, 63u);
    EXPECT_EQ(failed.pc, 1u);
    EXPECT_EQ(failed.reg, 1u);
    EXPECT_EQ(failed.kind, 4u);
    EXPECT_FALSE(
        validate_graphics_wave_outputs(plan, {good, bad}, transaction, transaction_refusal));
    EXPECT_TRUE(transaction.records.empty()) << "failed last wave cannot publish the first";
    EXPECT_EQ(transaction_refusal, "packet-vgpr-raw-export-unavailable");
    ASSERT_TRUE(
        validate_graphics_wave_outputs(plan, {good, good}, transaction, transaction_refusal))
        << transaction_refusal;
    EXPECT_EQ(transaction.records.size(), 2u) << "fresh complete status restores the transaction";
    bad[late + 1] = 4095;
    malformed_status(bad);   // plausible range != actual original read PC
    bad[late + 1] = 1;
    bad[late + 2] = 2;
    malformed_status(bad);   // known PC != this register/read form
    auto bypass = p;
    bypass.vgpr_status_offset = UINT32_MAX;
    bypass.vgpr_failure_sites.clear();
    refused(bypass, good);   // actual module marker prevents metadata-only tail bypass
    auto invalid_module = p;
    invalid_module.spirv.resize(4);
    refused(invalid_module, good);
    invalid_module = p;
    invalid_module.spirv[0] ^= 1u;
    refused(invalid_module, good);
    auto failed_input =
        f::cases()[8].packet;   // first bad self-read followed by genuine good overwrite
    const auto sticky = recompile_fragment_packet(failed_input);
    ASSERT_FALSE(sticky.spirv.empty()) << sticky.rejection;
    const auto sticky_words = evaluate(sticky, "sticky_failure_after_good_writer");
    const auto result = decode_fragment_packet(sticky, sticky_words, true);
    EXPECT_TRUE(result.exports.empty());
    EXPECT_EQ(result.pc, 0u);
    EXPECT_EQ(result.kind, 1u);

    const auto resource_input = f::resource_chain(false, false);
    const auto resource = recompile_fragment_resource_packet(resource_input);
    ASSERT_FALSE(resource.packet.spirv.empty()) << resource.packet.rejection;
    auto corrupt = evaluate(resource, "resource_status_late_corruption_after_failure");
    const auto& read = resource.packet.vgpr_failure_sites.front();
    const auto first = resource.packet.vgpr_status_offset;
    corrupt[first + 1] = read.pc;
    corrupt[first + 2] = read.reg;
    corrupt[first + 3] = read.kind;
    corrupt[first + 63 * 4] ^= 1u;
    const auto refused_resource = decode_fragment_resource_packet(
        resource, corrupt, true, resource_input.device.device_identity);
    EXPECT_TRUE(refused_resource.exports.empty());
    EXPECT_EQ(refused_resource.rejection, "packet-vgpr-status-record-invalid");
    EXPECT_EQ(refused_resource.failure, FragmentPacketRuntimeFailure::None);
}

TEST(FragmentPacketDefinedness, ShippingPreparationConsumesExactCachedProgramRequirements) {
    auto code = f::cases().front().packet.guest_code;
    const auto analysis = acquire_shader_analysis(code.data(), code.size());
    auto in = std::make_shared<RasterQuadInputs>();
    in->source_fs =
        std::make_shared<const std::vector<uint32_t>>(std::initializer_list<uint32_t>{0x07230203u});
    in->raw_code = shader_analysis_owned_words(analysis);
    in->vgpr_requirements = shader_analysis_packet_vgpr_requirements(analysis);
    in->raw_matches_producing_source = true;
    const auto prepared = prepare_fragment_packet_inputs(in, true);
    ASSERT_TRUE(prepared->vgpr_requirements);
    EXPECT_EQ(prepared->vgpr_requirements.get(), in->vgpr_requirements.get());
    EXPECT_TRUE(prepared->vgpr_requirements->storage.test(1));
    EXPECT_FALSE(prepared->vgpr_requirements->possible_entry.any())
        << "writer-only v1 isn't a launch input";
    EXPECT_TRUE(gap(*prepared, "packet-entry-mask-abi-unproved"));
    EXPECT_TRUE(gap(*prepared, "packet-vgpr-runtime-read-validity-unproved"));
    EXPECT_FALSE(gap(*prepared, "packet-entry-vgpr-values-unproved"));
    EXPECT_FALSE(prepared->ready) << "no system/M0/helper/composition/attachment authority";
    const auto warm = acquire_shader_analysis(code.data(), code.size());
    EXPECT_EQ(shader_analysis_packet_vgpr_requirements(warm).get(), in->vgpr_requirements.get());
    code[0] = 0x7e020300u;   // original MOV v1,v0 at SAME source address, not the old scalar writer
    const auto changed = acquire_shader_analysis(code.data(), code.size());
    const auto next = shader_analysis_packet_vgpr_requirements(changed);
    ASSERT_TRUE(next);
    EXPECT_NE(next.get(), in->vgpr_requirements.get());
    EXPECT_TRUE(next->possible_entry.test(0));
    EXPECT_FALSE(prepared->vgpr_requirements->possible_entry.any())
        << "prior owned version remains immutable";
    auto substitute = std::make_shared<RasterQuadInputs>(*in);
    substitute->raw_code = shader_analysis_owned_words(changed);
    const auto refused = prepare_fragment_packet_inputs(substitute, true);
    EXPECT_FALSE(refused->vgpr_requirements);
    EXPECT_TRUE(gap(*refused, "packet-vgpr-program-requirements-unavailable"));
    substitute = std::make_shared<RasterQuadInputs>(*in);
    substitute->vgpr_requirements.reset();
    EXPECT_TRUE(gap(*prepare_fragment_packet_inputs(substitute, true),
                    "packet-vgpr-program-requirements-unavailable"));
    EXPECT_FALSE(prepare_fragment_packet_inputs(in, false)->vgpr_requirements);
}

TEST(FragmentPacketDefinedness, FullySuppliedLegacyRawAbiRemainsUnextended) {
    const auto input = prosper::test::fragment_packet::packet({});
    const auto p = recompile_fragment_packet(input);
    ASSERT_FALSE(p.spirv.empty()) << p.rejection;
    EXPECT_EQ(p.vgpr_status_offset, UINT32_MAX);
    EXPECT_TRUE(p.vgpr_failure_sites.empty());
    EXPECT_EQ(p.input_stride, input.vgprs.size() + 4);
    EXPECT_EQ(p.output_words.size(), 64 * 12u);
    const auto words = evaluate(p, "legacy_complete_columns");
    const auto decoded = decode_fragment_packet(p, words, true);
    EXPECT_TRUE(decoded.rejection.empty());
    EXPECT_EQ(decoded.exports, prosper::test::fragment_packet::expected({}, input));
}
