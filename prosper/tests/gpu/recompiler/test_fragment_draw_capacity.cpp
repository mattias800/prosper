// Separate GPU-capacity placement must neither authenticate guest entry values nor weaken WAT1.
// SOURCE execution uses genuinely hand-owned old fixtures, not a shipping launch reconstruction.
#include "fixtures/fragment_packet_wave_fixture.hpp"
#include "fixtures/fragment_draw_source.hpp"
#include "gpu/recompiler/fragment_draw_capacity.hpp"
#include "gpu/recompiler/fragment_draw_gpu.hpp"
#include "bpermute_spirv_oracle.hpp"
#include <gtest/gtest.h>

namespace {
using namespace prosper::gpu;
namespace fixture = prosper::test::fragment_packet_wave;
RasterQuadCollector collector(uint32_t count = 48) {
    RasterQuadCollector result;
    result.max_quads = count;
    result.lane_words = kRasterQuadLaneFixedWords;
    result.record_words = 4 * result.lane_words;
    return result;
}
struct Inputs {
    std::shared_ptr<const FragmentPacketKernel> capacity_kernel;
    std::shared_ptr<const FragmentDrawCapacity> capacity;
    FragmentPacketWaveBatch owned;
    std::vector<uint32_t> input, authority;
};
Inputs inputs() {
    Inputs result;
    const auto prototype = fixture::packet();
    result.capacity_kernel = std::make_shared<const FragmentPacketKernel>(
        recompile_fragment_packet_capacity_kernel(prototype));
    prosper::test::fragment_draw::retain_source(result.capacity_kernel->program.packet.spirv,
                                                "fixture_capacity");
    std::string rejection;
    result.capacity = fragment_draw_capacity(result.capacity_kernel, collector(), rejection);
    EXPECT_TRUE(result.capacity) << rejection;
    if (!result.capacity) return result;
    auto old_kernel =
        std::make_shared<const FragmentPacketKernel>(recompile_fragment_packet_kernel(prototype));
    prosper::test::fragment_draw::retain_source(old_kernel->program.packet.spirv, "fixture_legacy");
    std::vector<FragmentPacketWavePlacement> placements;
    for (uint32_t wave = 0; wave < 3; ++wave)
        placements.push_back(result.capacity->placement(wave));
    const std::vector<FragmentResourcePacket> waves{fixture::packet(0), fixture::packet(1),
                                                    fixture::packet(2)};
    result.owned = pack_fragment_packet_waves(old_kernel, waves, placements);
    EXPECT_TRUE(result.owned.rejection.empty()) << result.owned.rejection;
    result.input = result.owned.input_words;
    if (result.input.size() < kFragmentDrawHeaderWords) return result;
    const uint32_t header[kFragmentDrawHeaderWords] = {kFragmentDrawInputMagic,
                                                       3,
                                                       result.capacity->input_words(),
                                                       result.capacity->output_words(),
                                                       48,
                                                       3,
                                                       1,
                                                       1,
                                                       0,
                                                       0,
                                                       0,
                                                       0};
    std::copy(std::begin(header), std::end(header), result.input.begin());
    const auto& authority = result.capacity->authority();
    result.authority.assign(authority.begin(), authority.end());
    return result;
}
std::vector<uint32_t> execute(const Inputs& input, uint32_t wave,
                              const std::vector<uint32_t>& output) {
    bpermute_oracle::Interpreter vm(input.capacity_kernel->program.packet.spirv);
    auto result =
        vm.run_buffers(64, {{0, input.input}, {1, output}, {2, input.authority}}, 1, wave);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return result;
}
TEST(FragmentDrawCapacity, SeparateSourceDomainAndCheckedDisjointCapacity) {
    const auto data = inputs();
    ASSERT_TRUE(data.capacity);
    EXPECT_FALSE(data.owned.kernel->layout.gpu_capacity);
    EXPECT_TRUE(data.capacity_kernel->layout.gpu_capacity);
    EXPECT_EQ(data.capacity->max_quads(), 48u);
    EXPECT_EQ(data.capacity->max_waves(), 3u);
    EXPECT_EQ(data.capacity->input_words(), data.input.size());
    EXPECT_EQ(data.capacity->output_words(), data.owned.output_words.size());
    EXPECT_EQ(data.capacity_kernel->layout.input_words, data.owned.kernel->layout.input_words);
    EXPECT_EQ(data.capacity_kernel->layout.output_words, data.owned.kernel->layout.output_words);
    EXPECT_NE(data.capacity_kernel->program.packet.spirv, data.owned.kernel->program.packet.spirv);
    EXPECT_EQ(pack_fragment_packet_waves(data.capacity_kernel,
                                         std::vector<FragmentResourcePacket>{fixture::packet()})
                  .rejection,
              "packet-wave-kernel-unavailable");
    std::string rejection;
    EXPECT_FALSE(fragment_draw_capacity(data.owned.kernel, collector(), rejection));
    EXPECT_EQ(rejection, "fragment-draw-capacity-kernel-unavailable");
    EXPECT_FALSE(fragment_draw_capacity(data.capacity_kernel, collector(0), rejection));
    EXPECT_EQ(rejection, "fragment-draw-capacity-collector-invalid");
    EXPECT_FALSE(fragment_draw_capacity(data.capacity_kernel, collector(4097), rejection));
    auto excessive = std::make_shared<FragmentPacketKernel>(*data.capacity_kernel);
    excessive->layout.input_words = 16 * 1024 * 1024;
    EXPECT_FALSE(fragment_draw_capacity(excessive, collector(), rejection));
    EXPECT_EQ(rejection, "fragment-draw-capacity-storage-budget");
    const auto repeated_legacy = recompile_fragment_packet_kernel(fixture::packet());
    prosper::test::fragment_draw::retain_source(repeated_legacy.program.packet.spirv,
                                                "repeat_legacy");
    EXPECT_EQ(repeated_legacy.program.packet.spirv, data.owned.kernel->program.packet.spirv)
        << "default source remains unchanged";
}
TEST(FragmentDrawCapacity, SameOriginalSourceThreeIndependentWavesAndUniformRefusals) {
    auto data = inputs();
    ASSERT_TRUE(data.capacity);
    ASSERT_FALSE(data.capacity_kernel->program.packet.spirv.empty());
    auto output = data.owned.output_words;
    for (uint32_t wave : {2u, 0u, 1u}) output = execute(data, wave, output);
    const auto decoded = decode_fragment_packet_waves(data.owned, output, true,
                                                      fixture::packet().device.device_identity);
    ASSERT_TRUE(decoded.rejection.empty()) << decoded.rejection;
    ASSERT_EQ(decoded.exports.size(), 3u);
    for (uint32_t wave = 0; wave < 3; ++wave)
        EXPECT_EQ(decoded.exports[wave], fixture::expected(wave));
    const auto unchanged = data.owned.output_words;
    const auto good = data;
    for (const uint32_t word : {0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u}) {
        data = good;
        data.input[word] ^= 0x80000000u;
        EXPECT_EQ(execute(data, 2, unchanged), unchanged) << "mutable header word=" << word;
    }
    data = good;
    data.input.resize(kFragmentDrawHeaderWords - 1);
    EXPECT_EQ(execute(data, 2, unchanged), unchanged) << "actual input length before header loads";
    data = good;
    data.authority.pop_back();
    EXPECT_EQ(execute(data, 2, unchanged), unchanged) << "actual readonly-plane length";
    data = good;
    data.input[data.capacity->placement(2).input_base] = 0;
    EXPECT_EQ(execute(data, 2, unchanged), unchanged) << "late original wave tag";
    data = good;
    EXPECT_EQ(execute(data, 3, unchanged), unchanged) << "capacity is not actual launch count";
}

FragmentResourcePacket colors(uint32_t variant) {
    auto packet = prosper::test::fragment_resource_packet::base();
    auto& invocation = packet.invocation;
    invocation.vgprs.clear();
    invocation.sgprs.clear();
    for (uint32_t channel = 0; channel < 4; ++channel) {
        invocation.sgprs.emplace_back(channel, prosper::test::fragment_resource_packet::bits(
                                                   float(1 + variant * 4 + channel)));
        prosper::test::fragment_packet::vmov(invocation.guest_code, channel, channel);
    }
    prosper::test::fragment_packet::exp(invocation.guest_code, 15, 0x03020100u);
    invocation.guest_code.push_back(0xbf810000u);
    return packet;
}
struct Transaction {
    std::shared_ptr<const FragmentDrawCapacity> capacity;
    std::vector<uint32_t> input, output, source, authority, entry;
};
Transaction transaction() {
    Transaction result;
    auto kernel = std::make_shared<const FragmentPacketKernel>(
        recompile_fragment_packet_capacity_kernel(colors(0)));
    prosper::test::fragment_draw::retain_source(kernel->program.packet.spirv, "color_capacity");
    std::string rejection;
    result.capacity = fragment_draw_capacity(kernel, collector(), rejection);
    EXPECT_TRUE(result.capacity) << rejection;
    if (!result.capacity) return result;
    const auto& capacity = *result.capacity;
    auto legacy =
        std::make_shared<const FragmentPacketKernel>(recompile_fragment_packet_kernel(colors(0)));
    prosper::test::fragment_draw::retain_source(legacy->program.packet.spirv, "color_legacy");
    std::vector<FragmentPacketWavePlacement> places;
    for (uint32_t wave = 0; wave < 3; ++wave) places.push_back(capacity.placement(wave));
    const auto owned = pack_fragment_packet_waves(
        legacy, std::vector<FragmentResourcePacket>{colors(0), colors(1), colors(2)}, places);
    EXPECT_TRUE(owned.rejection.empty()) << owned.rejection;
    result.input = owned.input_words;
    result.output = owned.output_words;
    const uint32_t header[] = {kFragmentDrawInputMagic,
                               3,
                               capacity.input_words(),
                               capacity.output_words(),
                               48,
                               3,
                               1,
                               1,
                               0,
                               16,
                               12,
                               1};
    std::copy(std::begin(header), std::end(header), result.input.begin());
    result.authority.assign(capacity.authority().begin(), capacity.authority().end());
    result.source.resize(capacity.collector_words(), 0);
    const uint32_t collector_header[] = {48, 0, collector().record_words, kRasterQuadMagic};
    std::copy(std::begin(collector_header), std::end(collector_header), result.source.begin());
    for (uint32_t quad = 0; quad < 48; ++quad)
        for (uint32_t lane = 0; lane < 4; ++lane) {
            const uint32_t record[] = {0,
                                       1,
                                       1,
                                       1,
                                       0,
                                       prosper::test::fragment_resource_packet::bits(
                                           float((quad % 8) * 2 + (lane & 1)) + .5f),
                                       prosper::test::fragment_resource_packet::bits(
                                           float((quad / 8) * 2 + (lane >> 1)) + .5f),
                                       0,
                                       prosper::test::fragment_resource_packet::bits(1.0f)};
            std::copy(std::begin(record), std::end(record),
                      result.source.begin() + 4 + (quad * 4 + lane) * collector().lane_words);
        }
    result.entry.resize(4 + kernel->layout.input_words, 0);
    const uint32_t entry_header[] = {1, 16, 12, kFragmentDrawInputMagic};
    std::copy(std::begin(entry_header), std::end(entry_header), result.entry.begin());
    for (uint32_t scalar = 0; scalar < kernel->layout.sgprs.size(); ++scalar) {
        result.entry[4 + kernel->layout.scalar_offsets[scalar]] =
            colors(0).invocation.sgprs[scalar].second;
        result.entry[4 + kernel->layout.scalar_available_offsets[scalar]] = 1;
    }
    for (uint32_t wave : {2u, 0u, 1u}) {
        bpermute_oracle::Interpreter vm(kernel->program.packet.spirv);
        result.output = vm.run_buffers(
            64, {{0, result.input}, {1, result.output}, {2, result.authority}}, 1, wave);
        EXPECT_TRUE(vm.error.empty()) << vm.error;
    }
    return result;
}
std::vector<uint32_t> validate(const Transaction& data, const std::vector<uint32_t>& source) {
    bpermute_oracle::Interpreter vm(source);
    auto result = vm.run_buffers(1,
                                 {{0, data.input},
                                  {1, std::vector<uint32_t>(data.capacity->commit_words(), 0)},
                                  {2, data.authority},
                                  {3, data.output},
                                  {4, data.source}},
                                 1);
    EXPECT_TRUE(vm.error.empty()) << vm.error;
    return result;
}
TEST(FragmentDrawCapacity, AllWorkerAndPixelGateBeforeAnyAttachmentPublication) {
    const auto good = transaction();
    ASSERT_TRUE(good.capacity);
    const auto source = build_fragment_draw_validation(*good.capacity, collector());
    prosper::test::fragment_draw::retain_source(source, "validation");
    ASSERT_FALSE(source.empty());
    const auto accepted = validate(good, source);
    ASSERT_EQ(accepted.size(), good.capacity->commit_words());
    EXPECT_EQ(accepted[0], 1u);
    EXPECT_EQ(accepted[1], 0u);
    EXPECT_EQ(accepted[2], 48u);
    EXPECT_EQ(accepted[3], 1u);
    uint32_t indexed = 0;
    for (uint32_t slot = kFragmentDrawCommitHeaderWords; slot < accepted.size(); ++slot)
        indexed += accepted[slot] != 0;
    EXPECT_EQ(indexed, 192u);
    const auto& program = good.capacity->kernel()->program;
    const auto late = good.capacity->placement(2).output_base;
    for (const auto fault :
         {std::pair{late + 40 * 2, FragmentDrawFailure::WorkerCompletion},
          std::pair{late + kPacketWaveOutputPrefix + program.status_offset + 40 * 3,
                    FragmentDrawFailure::GuestRuntime},
          std::pair{late + kPacketWaveOutputPrefix + program.packet.vgpr_status_offset + 40 * 4,
                    FragmentDrawFailure::GuestRuntime},
          std::pair{late + kPacketWaveOutputPrefix + 40 * kFragmentPacketExportWords + 4,
                    FragmentDrawFailure::GuestExport}}) {
        auto broken = good;
        broken.output[fault.first] ^= 0x80000000u;
        const auto refused = validate(broken, source);
        EXPECT_EQ(refused[0], 0u) << "late original logical wave2/lane40";
        EXPECT_EQ(refused[1], uint32_t(fault.second));
    }
    auto duplicate = good;
    for (uint32_t field = 0; field < collector().record_words; ++field)
        duplicate.source[4 + 47 * collector().record_words + field] = good.source[4 + field];
    auto refused = validate(duplicate, source);
    EXPECT_EQ(refused[0], 0u);
    EXPECT_EQ(refused[1], uint32_t(FragmentDrawFailure::DuplicatePixel));
    for (uint32_t invalid : {0x7fc00001u, 0x7f800000u, 0xbf000000u,
                             prosper::test::fragment_resource_packet::bits(1.25f)}) {
        auto malformed = good;
        malformed.source[4 + (47 * 4 + 3) * collector().lane_words + 5] = invalid;
        refused = validate(malformed, source);
        EXPECT_EQ(refused[0], 0u);
        EXPECT_EQ(refused[1], uint32_t(FragmentDrawFailure::CollectionRecord));
    }
    auto short_output = good;
    short_output.output.pop_back();
    EXPECT_EQ(validate(short_output, source),
              std::vector<uint32_t>(good.capacity->commit_words(), 0));
    const auto replay = build_fragment_draw_replay(*good.capacity, collector());
    prosper::test::fragment_draw::retain_source(replay, "replay");
    EXPECT_FALSE(replay.empty());
    const auto late_collector = 4 + (47 * 4 + 3) * collector().lane_words;
    for (const auto fault : {std::pair{0u, FragmentDrawFailure::HelperEntryUnavailable},
                             std::pair{1u, FragmentDrawFailure::HelperEntryUnavailable},
                             std::pair{2u, FragmentDrawFailure::CollectionRecord},
                             std::pair{3u, FragmentDrawFailure::CollectionRecord}}) {
        auto changed_after_assembly = good;
        changed_after_assembly.source[late_collector + fault.first] ^= 1u;
        refused = validate(changed_after_assembly, source);
        EXPECT_EQ(refused[0], 0u) << "post-assembly host provenance field " << fault.first;
        EXPECT_EQ(refused[1], uint32_t(fault.second));
    }
    auto two_primitives = good;
    two_primitives.input[11] = 2; // in-bounds alternative identity, not merely a range error
    ASSERT_EQ(validate(two_primitives, source)[0], 1u);
    two_primitives.source[late_collector + 4] = 1;
    refused = validate(two_primitives, source);
    EXPECT_EQ(refused[0], 0u);
    EXPECT_EQ(refused[1], uint32_t(FragmentDrawFailure::CollectionRecord));
}
TEST(FragmentDrawCapacity, ImmutableCollectorShapeBeforeAnyShaderAddressGeneration) {
    const auto good = transaction();
    ASSERT_TRUE(good.capacity);
    const auto shape = collector();
    EXPECT_TRUE(good.capacity->matches_collector(shape));
    const auto count = build_fragment_draw_count(*good.capacity, shape);
    const auto assembly = build_fragment_draw_assembly(*good.capacity, shape);
    const auto validation = build_fragment_draw_validation(*good.capacity, shape);
    const auto replay = build_fragment_draw_replay(*good.capacity, shape);
    prosper::test::fragment_draw::retain_source(count, "count");
    prosper::test::fragment_draw::retain_source(assembly, "assembly");
    prosper::test::fragment_draw::retain_source(validation, "validation");
    prosper::test::fragment_draw::retain_source(replay, "replay");
    EXPECT_FALSE(count.empty());
    EXPECT_FALSE(assembly.empty());
    EXPECT_FALSE(validation.empty());
    EXPECT_FALSE(replay.empty());
    std::vector<RasterQuadCollector> wrong;
    wrong.push_back(shape);
    wrong.back().lane_words = 512;
    wrong.back().record_words = 2048;   // length-only guards cannot prove this changed stride safe
    wrong.push_back(shape);
    --wrong.back().max_quads;
    wrong.push_back(shape);
    ++wrong.back().record_words;
    wrong.push_back(shape);
    wrong.back().fields.push_back({RasterQuadFieldKind::Interpolant, 0, 0, 4});
    wrong.push_back(shape);
    wrong.back().rejection = "unavailable-collector-producer";
    for (const auto& changed : wrong) {
        EXPECT_FALSE(good.capacity->matches_collector(changed));
        EXPECT_TRUE(build_fragment_draw_count(*good.capacity, changed).empty());
        EXPECT_TRUE(build_fragment_draw_assembly(*good.capacity, changed).empty());
        EXPECT_TRUE(build_fragment_draw_validation(*good.capacity, changed).empty());
        EXPECT_TRUE(build_fragment_draw_replay(*good.capacity, changed).empty());
    }
    auto impossible = collector(1);
    impossible.lane_words = 512;
    impossible.record_words = 2048;
    std::string rejection;
    EXPECT_FALSE(fragment_draw_capacity(good.capacity->kernel(), impossible, rejection));
    EXPECT_EQ(rejection, "fragment-draw-capacity-collector-invalid");
    EXPECT_EQ(good.capacity->placement(UINT32_MAX).input_base, UINT32_MAX);
    EXPECT_EQ(good.capacity->placement(good.capacity->max_waves()).output_base, UINT32_MAX);
}
TEST(FragmentDrawCapacity, DeviceCountAndAssemblyKeepMaskAndScratchFactsAbsent) {
    const auto good = transaction();
    ASSERT_TRUE(good.capacity);
    const auto count_source = build_fragment_draw_count(*good.capacity, collector());
    const auto assembly_source = build_fragment_draw_assembly(*good.capacity, collector());
    prosper::test::fragment_draw::retain_source(count_source, "count");
    prosper::test::fragment_draw::retain_source(assembly_source, "assembly");
    ASSERT_FALSE(count_source.empty());
    ASSERT_FALSE(assembly_source.empty());
    bpermute_oracle::Interpreter counter(count_source);
    auto input = counter.run_buffers(1,
                                     {{0, good.source},
                                      {1, std::vector<uint32_t>(good.capacity->input_words(), 0)},
                                      {2, good.authority},
                                      {3, good.entry}},
                                     1);
    ASSERT_TRUE(counter.error.empty()) << counter.error;
    EXPECT_EQ(input[1], 3u);
    EXPECT_EQ(input[5], 3u);
    EXPECT_EQ(input[8], 0u);
    for (uint32_t wave : {2u, 0u, 1u}) {
        bpermute_oracle::Interpreter assembler(assembly_source);
        input = assembler.run_buffers(
            64, {{0, good.source}, {1, input}, {2, good.authority}, {3, good.entry}}, 1, wave);
        ASSERT_TRUE(assembler.error.empty()) << assembler.error;
    }
    const auto& kernel = *good.capacity->kernel();
    for (uint32_t wave = 0; wave < 3; ++wave) {
        const auto base = good.capacity->placement(wave).input_base;
        EXPECT_EQ(input[base], kPacketWaveInputMagic);
        EXPECT_EQ(input[base + 1], wave);
        for (uint32_t lane = 0; lane < 64; ++lane) {
            const auto row = base + 2 + lane * kernel.program.packet.input_stride;
            const auto state = 2 * kernel.layout.vgprs.size();
            for (uint32_t field = 0; field < state + 3; ++field)
                EXPECT_EQ(input[row + field], 0u) << "zero raw storage never grants availability";
            EXPECT_EQ(input[row + state + 3], 1u) << "separate nonhelper export eligibility";
        }
        for (uint32_t scalar = 0; scalar < kernel.layout.sgprs.size(); ++scalar) {
            EXPECT_EQ(input[base + 2 + kernel.layout.scalar_offsets[scalar]],
                      good.entry[4 + kernel.layout.scalar_offsets[scalar]]);
            EXPECT_EQ(input[base + 2 + kernel.layout.scalar_available_offsets[scalar]], 1u);
        }
    }
    auto helpers = good;
    helpers.source[4 + (47 * 4 + 3) * collector().lane_words] = 1;
    bpermute_oracle::Interpreter assembler(assembly_source);
    const auto refused = assembler.run_buffers(
        64, {{0, helpers.source}, {1, input}, {2, good.authority}, {3, good.entry}}, 1, 2);
    ASSERT_TRUE(assembler.error.empty()) << assembler.error;
    EXPECT_EQ(refused[8], uint32_t(FragmentDrawFailure::HelperEntryUnavailable));
    auto overflow = good.source;
    overflow[1] = 1;
    bpermute_oracle::Interpreter overflow_counter(count_source);
    const auto no_dispatch = overflow_counter.run_buffers(
        1, {{0, overflow}, {1, input}, {2, good.authority}, {3, good.entry}}, 1);
    ASSERT_TRUE(overflow_counter.error.empty()) << overflow_counter.error;
    EXPECT_EQ(no_dispatch[1], 0u);
    EXPECT_EQ(no_dispatch[5], 0u);
    EXPECT_EQ(no_dispatch[8], uint32_t(FragmentDrawFailure::CollectionOverflow));
}
}   // namespace
