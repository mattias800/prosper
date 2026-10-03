#include "gpu/execute/gpu_execute.hpp"
#include <gtest/gtest.h>
#include "gpu/execute/graphics_nested_wide_reader.hpp"
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/pm4/pm4_registers.hpp"
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/memory/guest_memory_topology.hpp"
#include "../recompiler/bpermute_spirv_oracle.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

TEST(NestedWideDataAdmission, Contract) {
    const auto fragment_commit = build_owned_fragment_export_commit(128u, {});
    ASSERT_FALSE(fragment_commit.empty());
    EXPECT_FALSE(fragment_spirv_uses_internal_gds(fragment_commit))
        << "completed raw fragment exports must not request a synthetic zero GDS binding";
    const std::vector<uint32_t> reserved_gds{
        0x07230203u,      0x00010000u, 0u,  8u, 0u,
        (4u << 16) | 71u, 1u,          34u, 1u,   // DescriptorSet1
        (4u << 16) | 71u, 1u,          33u, 0u,   // reserved Binding0
    };
    EXPECT_TRUE(fragment_spirv_uses_internal_gds(reserved_gds))
        << "real internal GDS identity remains reserved and detectable";
    register_builtin_hle();
    const auto allocate = Hle::lookup(nid_hash("sceKernelAllocateDirectMemory"));
    const auto map = Hle::lookup(nid_hash("sceKernelMapDirectMemory"));
    const auto unmap = Hle::lookup(nid_hash("sceKernelMunmap"));
    const auto release = Hle::lookup(nid_hash("sceKernelReleaseDirectMemory"));
    if (!allocate || !map || !unmap || !release) FAIL() << "legacy early exit";

    constexpr uint64_t page = 0x10000;
    uint64_t physical = 0, parent = 0, child = 0, output = 0, alias = 0;
    if (allocate(0, 0x200000000ull, 4 * page, page, 0,
                 reinterpret_cast<uint64_t>(&physical)) != 0 || !physical ||
        map(reinterpret_cast<uint64_t>(&parent), page, 3, 0, physical, page) != 0 ||
        map(reinterpret_cast<uint64_t>(&child), page, 3, 0, physical + page, page) != 0 ||
        map(reinterpret_cast<uint64_t>(&output), page, 3, 0, physical + 2 * page, page) != 0 ||
        map(reinterpret_cast<uint64_t>(&alias), page, 3, 0, physical + 2 * page, page) != 0)
        FAIL() << "legacy early exit";
    int failures = 0;
    const auto expect = [&](bool okay, const char* message) {
        if (!okay) {
            std::fprintf(stderr, "nested admission: %s\n", message);
            ++failures;
        }
    };

    const uint32_t shader[] = {
        0xf4080a00u, 0xfa000000u, // pc0: parent x4 s[40:43] from entry s[0:1]
        0xf4080b14u, 0xfa000010u, // pc2: child x4 s[44:47] from s[40:41]
        0x7e00022eu,              // pc4: numeric reader of s46
        0xf0200f08u, 0x00060004u, // pc5: 2D image_store
        0xbf810000u,
    };
    std::vector<Rdna2Inst> decoded;
    rdna2_walk(shader, std::size(shader), decoded);
    expect(rdna2_proven_raw_nested_wide_data_loads(decoded) ==
               std::vector<uint32_t>{2u},
           "production shader bytes prove one child at pc2");
    expect(decoded.size() == 5 && decoded[3].fmt == Rdna2Format::MIMG &&
               decoded[3].opcode == 0x08u && decoded[3].pc == 5u,
           "the positive fixture contains the intended image store");
    for (bool wide8 : {false, true}) {
        const uint32_t width = wide8 ? 8u : 4u;
        for (bool straddles_vcc : {false, true}) {
            const uint32_t destination = 106u - width + (straddles_vcc ? 2u : 0u);
            const uint32_t boundary_shader[] = {
                wide8 ? 0xf40c0a00u : 0xf4080a00u, 0xfa000000u,
                (wide8 ? 0xf40c0014u : 0xf4080014u) | (destination << 6u),
                0xfa000010u,
                0x7e000269u, // numeric s105 observer: last ordinary SGPR, never a Bool mask
                0xbf810000u,
            };
            std::vector<Rdna2Inst> boundary;
            rdna2_walk(boundary_shader, std::size(boundary_shader), boundary);
            expect(boundary.size() == 4u && boundary[1].dst.value == destination &&
                       boundary[2].src[0].kind == OperandKind::SGPR &&
                       boundary[2].src[0].value == 105 &&
                       rdna2_raw_wide_data_loads(boundary) ==
                           std::vector<uint32_t>({0u, 2u}),
                   "x4/x8 boundary fixture keeps the child numeric backing obligation");
            if (straddles_vcc) {
                expect(rdna2_proven_raw_nested_wide_data_loads(boundary).empty() &&
                           rdna2_owned_nested_wide_chains(boundary).empty(),
                       "x4/x8 writes through s106/s107 refuse nested owned admission");
            } else {
                expect(rdna2_proven_raw_nested_wide_data_loads(boundary) ==
                           std::vector<uint32_t>{2u} &&
                           rdna2_owned_nested_wide_chains(boundary) ==
                               std::vector<RawNestedWideChain>({{0u, 2u, width * 4u,
                                   width * 4u, 0u, 16u}}),
                       "x4/x8 ending at s105 retain the exact-PC parent/child proof");
            }
        }
    }
    expect(tiled_surface_bytes(64, 64, 27, 0, 1) > 64 * 64,
           "write footprint includes tile padding beyond the declared size");

    auto table_for = [&] {
        ShaderResourceTable table;
        ShaderResource source;
        source.cls = ResourceClass::ConstantBuffer;
        source.gpu_addr = parent;
        source.size = 64;
        source.fetch_pc = 0;
        table.resources.push_back(source);
        source.gpu_addr = child;
        source.fetch_pc = 2;
        table.resources.push_back(source);
        ShaderResource target;
        target.cls = ResourceClass::StorageImage;
        target.gpu_addr = output;
        target.fetch_pc = 5;
        target.width = target.height = 64;
        target.depth = target.sample_count = target.declared_mip_levels = 1;
        target.img_dim = 1;
        target.tile_mode = 27;
        target.format = DataFormat::Uint8;
        target.num_components = 1;
        target.size = 64 * 64;
        table.resources.push_back(target);
        return table;
    };
    SrtUse store_use;
    store_use.kind = 0;
    store_use.use_pc = 5;
    store_use.is_storage_image = true;
    store_use.descriptor_source_addr = parent + 0x100;
    const std::vector<SrtUse> uses{store_use};
    const uint32_t original_child_value = 0x12345678u;
    std::memcpy(reinterpret_cast<void*>(parent), &child, sizeof(child));
    std::memcpy(reinterpret_cast<void*>(child + 0x18), &original_child_value,
                sizeof(original_child_value));

    auto table = table_for();
    // The graphics contract uses physical ALLOCATION exclusion rather than the compute
    // fixture's precise linear write-footprint check above. Exercise the real HLE query.
    uint64_t isolated_physical = 0, isolated = 0;
    GuestDirectAllocation retained_origin;
    expect(allocate(0, 0x200000000ull, page, page, 0,
                    reinterpret_cast<uint64_t>(&isolated_physical)) == 0 &&
           map(reinterpret_cast<uint64_t>(&isolated), page, 3, 0,
               isolated_physical, page) == 0 && isolated,
           "map an independently allocated physical source beside the aliased allocation");
    if (isolated) {
        GuestMappingLease lease;
        retained_origin = guest_memory_direct_allocation(lease, isolated, page);
        const auto relation = [&](uint64_t source, uint64_t bytes,
                                  uint64_t producer, uint64_t minimum) {
            return guest_memory_direct_allocation_relation(lease, source, bytes, producer, minimum);
        };
        expect(relation(isolated + page - 4u, 4u, output, page) ==
                   GuestMemoryTopologyRelation::Disjoint &&
               relation(output, page, isolated + page - 1u, 1u) ==
                   GuestMemoryTopologyRelation::Disjoint,
               "distinct physical allocations admit exact inclusive last-byte endpoints");
        expect(relation(parent + page - 4u, 4u, alias + page - 1u, 1u) ==
                   GuestMemoryTopologyRelation::Overlap &&
               guest_memory_topology_relation(parent + page - 4u, 4u,
                   alias + page - 1u, 1u) == GuestMemoryTopologyRelation::Disjoint,
               "different mapping offsets still exclude the producer's whole physical allocation");
        expect(relation(isolated + page - 4u, 8u, output, 1u) ==
                   GuestMemoryTopologyRelation::Unknown &&
               relation(isolated, 4u, output + page - 1u, 2u) ==
                   GuestMemoryTopologyRelation::Unknown &&
               relation(isolated, 0u, output, 1u) == GuestMemoryTopologyRelation::Unknown &&
               relation(UINT64_MAX - 1u, 4u, output, 1u) == GuestMemoryTopologyRelation::Unknown &&
               relation(isolated, 4u, 0xdead00000000ull, 1u) ==
                   GuestMemoryTopologyRelation::Unknown,
               "mapping endpoints, overflow, empty extent and unknown producers refuse");
    }
    const auto retype = Hle::lookup(nid_hash("sceKernelMtypeprotect"));
    expect(retype && retype(output + 0x4000u, 0x4000u, 1u, 3u, 0, 0) == 0,
           "retype only a middle physical-allocation slice through the production HLE path");
    {
        GuestMappingLease lease;
        expect(guest_memory_direct_allocation_relation(lease, parent, 4u,
                   output + 0x4000u, 4u) == GuestMemoryTopologyRelation::Overlap,
               "typed ledger splits preserve complete original allocation ownership");
        expect(isolated && guest_memory_direct_allocation_relation(lease, isolated, 4u,
                   output + 0x4000u, 4u) == GuestMemoryTopologyRelation::Disjoint,
               "retype never coalesces adjacent distinct allocation births");
    }
    if (isolated) {
        expect(unmap(isolated, page, 0, 0, 0, 0) == 0 &&
               release(isolated_physical, page, 0, 0, 0, 0) == 0,
               "release the retained producer VA and physical allocation");
        isolated = 0;
        {
            GuestMappingLease lease;
            expect(guest_memory_retained_allocation_relation(lease, parent, 4u, retained_origin) ==
                       GuestMemoryTopologyRelation::Disjoint,
                   "unmapped old producer VA permits proved distinct physical backing");
        }
        uint64_t reused_physical = 0;
        expect(allocate(isolated_physical, isolated_physical + page, page, page, 0,
                    reinterpret_cast<uint64_t>(&reused_physical)) == 0 &&
               reused_physical == isolated_physical &&
               map(reinterpret_cast<uint64_t>(&isolated), page, 3, 0,
                   reused_physical, page) == 0 && isolated,
               "force a distinct allocation birth reusing the old physical interval");
        GuestMappingLease lease;
        const auto new_origin = guest_memory_direct_allocation(lease, isolated, 4u);
        expect(new_origin.identity && new_origin.identity != retained_origin.identity &&
               guest_memory_retained_allocation_relation(lease, isolated, 4u,
                   retained_origin) == GuestMemoryTopologyRelation::Overlap,
               "new allocation identity cannot erase an old retained physical overlap");
        expect(guest_memory_retained_allocation_relation(lease, parent, 4u, retained_origin) ==
                   GuestMemoryTopologyRelation::Disjoint,
               "physical reuse does not ban a different allocated physical interval");
    }
    // Exercise the production observation, not just the topology query. Its complete selector
    // interval includes the last x4/x8 word even when this invocation selects a smaller offset.
    // The authority provider below models the renderer's retained physical-origin exclusion;
    // current-draw attachment exclusion is independently performed by the observer itself.
    uint64_t wave_physical = 0, wave_source = 0, wave_alias = 0;
    uint64_t wave_output_physical = 0, wave_output = 0;
    expect(
        allocate(0, 0x200000000ull, page, page, 0, reinterpret_cast<uint64_t>(&wave_physical)) ==
                0 &&
            map(reinterpret_cast<uint64_t>(&wave_source), page, 3, 0, wave_physical, page) == 0 &&
            wave_source &&
            map(reinterpret_cast<uint64_t>(&wave_alias), page, 3, 0, wave_physical, page) == 0 &&
            wave_alias &&
            allocate(0, 0x200000000ull, page, page, 0,
                     reinterpret_cast<uint64_t>(&wave_output_physical)) == 0 &&
            map(reinterpret_cast<uint64_t>(&wave_output), page, 3, 0, wave_output_physical, page) ==
                0 &&
            wave_output,
        "map a real runtime window, physical alias and independently allocated full-page output");
    if (wave_source && wave_alias && wave_output) {
        // Explicitly fault/commit backing before asking the readable-window query. Querying a
        // lazy reservation is not permission for the observer to manufacture committed bytes.
        std::memset(reinterpret_cast<void*>(wave_source), 0x97, page);
        GuestDirectAllocation wave_origin;
        {
            GuestMappingLease lease;
            wave_origin = guest_memory_direct_allocation(lease, wave_source, page);
        }
        bool retain_wave_source = false;
        uint32_t authority_calls = 0;
        set_graphics_raw_source_authority([&](const GuestMappingLease& lease, uint64_t address,
                                              uint32_t bytes) {
            ++authority_calls;
            return !retain_wave_source ||
                   guest_memory_retained_allocation_relation(lease, address, bytes, wave_origin) ==
                       GuestMemoryTopologyRelation::Disjoint;
        });
        for (bool wide8 : {false, true}) {
            const uint32_t width = wide8 ? 8u : 4u;
            const std::vector<uint32_t> wave_code{
                0x7e280500u,   // READFIRSTLANE s20,v0
                0x87148f14u,   // AND s20,s20,15
                0x8f148414u,   // LSHL s20,s20,4: complete unsigned [0,240]
                (wide8 ? 0xf40c0200u : 0xf4080200u),
                0x28000010u,
                0x7e000200u | (8u + width - 1u),   // numeric highest component
                0xbf810000u,
            };
            const auto scalars_for = [](uint64_t address) {
                return std::vector<std::pair<uint32_t, uint32_t>>{{0u, uint32_t(address)},
                                                                  {1u, uint32_t(address >> 32u)}};
            };
            const uint32_t domain_bytes = 240u + width * 4u;
            const uint64_t base = wave_source + page - domain_bytes - 16u;
            auto* words = reinterpret_cast<uint32_t*>(base + 16u);
            for (uint32_t i = 0; i < domain_bytes / 4u; ++i) words[i] = 0x97000000u + i;
            GraphicsRawSnapshotContext complete;
            complete.producers_complete = true;
            complete.output_allocations.emplace_back(wave_output, page);
            std::vector<PacketRawWaveWindow> owners;
            std::string refusal;
            ASSERT_TRUE(observe_graphics_raw_wave_windows(wave_code, scalars_for(base), &complete,
                                                          owners, refusal))
                << refusal;
            ASSERT_EQ(owners.size(), 1u);
            expect(
                owners[0].load_pc == 3u && owners[0].guest_base == base &&
                    owners[0].guest_begin == base + 16u &&
                    owners[0].words.size() == domain_bytes / 4u &&
                    owners[0].words.back() == words[domain_bytes / 4u - 1u],
                "ordered real backing owns every certified byte through the allocation endpoint");
            const auto original_owner = owners;
            // The earlier retype split this output's tracked VA into multiple mappings. The
            // producer query requires its complete minimum span in one mapping; a physical birth
            // alone cannot substitute for that extent proof. Keep the conservative refusal.
            auto fragmented_output = complete;
            fragmented_output.output_allocations = {{output, page}};
            {
                GuestMappingLease lease;
                expect(guest_memory_direct_allocation_relation(lease, base + 16u, domain_bytes,
                                                               output, page) ==
                           GuestMemoryTopologyRelation::Unknown,
                       "fragmented producer mapping cannot prove a whole-page minimum extent");
            }
            expect(!observe_graphics_raw_wave_windows(wave_code, scalars_for(base),
                                                      &fragmented_output, owners, refusal) &&
                       owners.empty() && refusal == "wave-window-output-allocation-not-disjoint",
                   "unknown fragmented producer extent refuses before copying any window");
            auto incomplete = complete;
            incomplete.producers_complete = false;
            const uint32_t calls_before = authority_calls;
            expect(!observe_graphics_raw_wave_windows(wave_code, scalars_for(base), &incomplete,
                                                      owners, refusal) &&
                       owners.empty() && refusal == "wave-window-producer-incomplete" &&
                       authority_calls == calls_before,
                   "pending or failed producer refuses before source admission or any bytes");
            expect(!observe_graphics_raw_wave_windows(wave_code, scalars_for(base + 4u), &complete,
                                                      owners, refusal) &&
                       owners.empty() && refusal == "wave-window-readable-allocation-incomplete",
                   "one-word overflow rejects the whole x4/x8 selector domain without truncation");
            auto alias_output = complete;
            alias_output.output_allocations = {{wave_alias, page}};
            expect(!observe_graphics_raw_wave_windows(wave_code, scalars_for(base), &alias_output,
                                                      owners, refusal) &&
                       owners.empty() && refusal == "wave-window-output-allocation-not-disjoint",
                   "different output VA in the same physical allocation cannot supply a window");
            retain_wave_source = true;
            expect(!observe_graphics_raw_wave_windows(wave_code, scalars_for(base), &complete,
                                                      owners, refusal) &&
                       owners.empty() && refusal == "wave-window-current-source-unproved",
                   "retained producer physical origin refuses even with a disjoint current output");
            retain_wave_source = false;
            for (uint32_t i = 0; i < domain_bytes / 4u; ++i) words[i] ^= 0x01010101u;
            expect(
                observe_graphics_raw_wave_windows(wave_code, scalars_for(base), &complete, owners,
                                                  refusal) &&
                    owners.size() == 1u && original_owner.size() == 1u &&
                    owners[0].words != original_owner[0].words &&
                    original_owner[0].words.back() == (0x97000000u + domain_bytes / 4u - 1u),
                "fresh successful producer observes B while submitted owner A remains immutable");
            auto vertex_code = wave_code;
            vertex_code.pop_back();
            vertex_code.insert(vertex_code.end(),
                               {0x7e020280u, 0x7e040280u, 0x7e060280u, 0x7e0802f2u, 0xf80008cfu,
                                0x04030201u, 0xf8000a0fu, 0x00000000u,
                                0xbf810000u});   // actual POS0 and numeric PARAM0
            GraphicsWaveStagePlan stages;
            expect(prepare_owned_vertex_waves(vertex_code, scalars_for(base), {}, 64u, 7, 2u, {},
                                              &complete, stages, refusal) &&
                       stages.assembly == GraphicsWaveAssembly::VertexDrawOrder &&
                       stages.packets.size() == 2u && stages.invocations.size() == 2u &&
                       stages.invocations[0][31].vertex_index == 38u &&
                       stages.invocations[0][32].vertex_index == 39u &&
                       stages.invocations[0][63].vertex_index == 70u &&
                       stages.invocations[1][0].vertex_index == 7u &&
                       stages.invocations[1][0].instance_index == 1u &&
                       stages.packets[0].raw_windows[0].words ==
                           stages.packets[1].raw_windows[0].words,
                   "real stage plan owns distinct complete vertex/instance waves and one observed "
                   "numeric domain");
            // Independently different wave selectors: real InstanceIndex 0 and 1 select offsets
            // 0 and 16 from one retained observation. The numeric oracle executes emitted SOURCE,
            // not the certificate's abstract interval or a mirror of the export-record writer.
            auto instance_code = vertex_code;
            instance_code[0] = 0x7e280503u;   // READFIRSTLANE s20,v3 (actual InstanceIndex)
            expect(prepare_owned_vertex_waves(instance_code, scalars_for(base), {}, 64u, 0, 2u, {},
                                              &complete, stages, refusal),
                   "prepare two complete waves with independently different actual selectors");
            ASSERT_EQ(stages.packets.size(), 2u) << "numeric controls require BOTH admitted waves";
            ASSERT_EQ(stages.invocations.size(), 2u)
                << "numeric controls require BOTH identity rows";
            if (stages.packets.size() == 2u) {
                std::vector<std::vector<uint32_t>> outputs;
                for (uint32_t wave = 0; wave < 2u; ++wave) {
                    const auto program = recompile_fragment_packet(stages.packets[wave]);
                    bpermute_oracle::Interpreter vm(program.spirv);
                    auto actual = vm.run_packet(program.input_words, program.output_words);
                    std::vector<uint32_t> expected(64u * 24u);
                    for (uint32_t lane = 0; lane < 64u; ++lane) {
                        const uint32_t pos[] = {1u, 1u, 1u, 12u, 15u, 0u,
                                                1u, 0u, 0u, 0u,  0u,  0x3f800000u};
                        const uint32_t numeric = words[wave * 4u + width - 1u];
                        const uint32_t param[] = {1u, 1u, 1u,      32u,     15u,     0u,
                                                  1u, 0u, numeric, numeric, numeric, numeric};
                        std::copy_n(pos, 12u, expected.begin() + lane * 24u);
                        std::copy_n(param, 12u, expected.begin() + lane * 24u + 12u);
                    }
                    expect(vm.error.empty() && actual == expected,
                           "actual stage SOURCE selects each wave's own numeric offset and last "
                           "component");
                    outputs.push_back(std::move(actual));
                }
                GraphicsWaveOutputTransaction transaction;
                GraphicsVertexExportCommit commit;
                expect(validate_graphics_wave_outputs(stages, outputs, transaction, refusal) &&
                           prepare_owned_vertex_export_commit(stages, transaction, nullptr, commit,
                                                              refusal) &&
                           commit.vertices_per_instance == 64u && commit.instances == 2u &&
                           commit.record_words == 8u && commit.words.size() == 1024u &&
                           commit.words[4u] == words[width - 1u] &&
                           commit.words[64u * 8u + 4u] == words[4u + width - 1u] &&
                           !commit.shader.empty(),
                       "whole transaction commits distinct completed per-instance raw exports");
                // Capture consumes retained producing owners, never a later guest-memory read.
                // Both stages carry complete numeric windows, explicit FP profile and independent
                // physical PS launch observations; stored and raw replay share this one admission.
                auto captured_owner = std::make_shared<GraphicsOwnedWaveDraw>();
                captured_owner->vertex_pending = captured_owner->fragment_pending = true;
                captured_owner->vertex = stages;
                const FloatTransportConfig explicit_profile{
                    FloatTransportProfile::ExplicitNonFinite32};
                for (auto& packet : captured_owner->vertex.packets)
                    packet.float_transport = explicit_profile;
                auto fragment_code = wave_code;
                fragment_code.pop_back();
                fragment_code.insert(fragment_code.end(), {0xf800080fu, 0x00000000u, 0xbf810000u});
                captured_owner->fragment_code =
                    std::make_shared<const std::vector<uint32_t>>(fragment_code);
                captured_owner->fragment_scalars = scalars_for(base);
                captured_owner->fragment_windows = stages.packets.front().raw_windows;
                DrawItem capture_draw;
                capture_draw.owned_waves = captured_owner;
                capture_draw.vertex_count = capture_draw.raw_draw_count = 64u;
                capture_draw.instance_count = 2u;
                capture_draw.float_transport = explicit_profile;
                capture_draw.fragment_wave_config_available = true;
                capture_draw.ps_launch_rsrc1 = {true, 16u << 12u};
                capture_draw.ps_float_mode = {true, 16u};
                capture_draw.ps_float_flags = {true, false, false};
                capture_draw.ps_entry.observed = capture_draw.ps_entry.rsrc2_available = true;
                capture_draw.ps_entry.rsrc2 =
                    2u << prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT;
                capture_draw.ps_entry.user_data_available = 3u;
                capture_draw.ps_entry.user_data[0] = uint32_t(base);
                capture_draw.ps_entry.user_data[1] = uint32_t(base >> 32u);
                capture_draw.has_system_inputs = true;
                capture_draw.system_inputs.ena = capture_draw.system_inputs.addr = 1u << 8u;
                capture_draw.ps_raster_launch.input_ena_available = true;
                capture_draw.ps_raster_launch.input_addr_available = true;
                capture_draw.ps_raster_launch.input_ena = capture_draw.system_inputs.ena;
                capture_draw.ps_raster_launch.input_addr = capture_draw.system_inputs.addr;
                GpuCaptureMetadata metadata;
                metadata.width = metadata.height = 8u;
                GpuCaptureFile captured, decoded_capture;
                GpuReplayFrame replay;
                std::vector<uint8_t> wire;
                uint32_t unexpected_guest_reads = 0;
                const auto no_guest_reads = [&](uint64_t, uint8_t*, size_t) -> size_t {
                    ++unexpected_guest_reads;
                    return 0;
                };
                const bool capture_ok = capture_draw_items({capture_draw}, metadata, no_guest_reads,
                                                           captured, refusal) &&
                                        serialize_gpu_capture(captured, wire, refusal) &&
                                        deserialize_gpu_capture(wire, decoded_capture, refusal) &&
                                        materialize_gpu_replay(decoded_capture, replay, refusal);
                expect(capture_ok && unexpected_guest_reads == 0u &&
                           decoded_capture.format_version == 70u && replay.items.size() == 1u &&
                           replay.items[0].owned_waves &&
                           replay.items[0].float_transport == explicit_profile &&
                           replay.items[0].ps_entry == capture_draw.ps_entry &&
                           replay.items[0].ps_launch_rsrc1 == capture_draw.ps_launch_rsrc1 &&
                           replay.items[0].owned_waves->vertex.packets[1].guest_code ==
                               instance_code &&
                           replay.items[0].owned_waves->vertex.packets[1].raw_windows[0].words ==
                               stages.packets[1].raw_windows[0].words &&
                           *replay.items[0].owned_waves->fragment_code == fragment_code &&
                           replay.items[0].owned_waves->fragment_windows[0].words ==
                               captured_owner->fragment_windows[0].words,
                       "current full capture retains both producing stages, profile, entry facts "
                       "and complete immutable numeric bytes");
                if (capture_ok) {
                    // v70 intentionally encodes only the live VS subset. Nondefault fields used
                    // by the packet compiler must refuse, rather than disappear during decode.
                    for (uint32_t field = 0; field < 4u; ++field) {
                        auto unsupported = std::make_shared<GraphicsOwnedWaveDraw>(*captured_owner);
                        auto& packet = unsupported->vertex.packets[1];
                        if (field == 0u) packet.float_mode = {true, 16u};
                        if (field == 1u) packet.float_flags = {true, false, false};
                        if (field == 2u)
                            packet.quad_topology =
                                FragmentPacketQuadTopology::ConsecutiveLogicalQuads;
                        if (field == 3u) packet.float_mode = {true, 0u};
                        auto bad_subset = captured;
                        bad_subset.draws[0].owned_waves = std::move(unsupported);
                        std::vector<uint8_t> refused_bytes;
                        expect(
                            !serialize_gpu_capture(bad_subset, refused_bytes, refusal) &&
                                refusal == "owned logical-wave VS codec cannot retain "
                                           "mode/flags/topology wave=1 pc=3",
                            "writer cannot discard compiler-observable VS mode, flags or topology");
                    }
                    for (uint32_t column = 0; column < 2u; ++column) {
                        auto partial = std::make_shared<GraphicsOwnedWaveDraw>(*captured_owner);
                        partial->vertex.packets[1].vgprs[column].available_mask &=
                            ~(uint64_t(1) << 63u);
                        auto bad_subset = captured;
                        bad_subset.draws[0].owned_waves = std::move(partial);
                        std::vector<uint8_t> refused_bytes;
                        expect(!serialize_gpu_capture(bad_subset, refused_bytes, refusal),
                               "second-wave partial vertex/instance inputs cannot become full "
                               "CAP70 replay authority");
                    }
                    for (uint32_t i = 0; i < domain_bytes / 4u; ++i) words[i] ^= 0x02020202u;
                    std::vector<PacketRawWaveWindow> later;
                    expect(observe_graphics_raw_wave_windows(wave_code, scalars_for(base),
                                                             &complete, later, refusal) &&
                               later[0].words !=
                                   replay.items[0].owned_waves->fragment_windows[0].words &&
                               replay.items[0].owned_waves->fragment_windows[0].words ==
                                   captured_owner->fragment_windows[0].words,
                           "later producer B cannot mutate captured replay owner A");
                    // Semantic corruptions keep a well-formed file: the materializer must reject
                    // each before either stored-module use or raw packet compilation can execute.
                    const auto refuses = [&](GpuCaptureFile bad, const std::string& expected) {
                        GpuCaptureFile read;
                        GpuReplayFrame refused;
                        std::vector<uint8_t> bytes;
                        return serialize_gpu_capture(bad, bytes, refusal) &&
                               deserialize_gpu_capture(bytes, read, refusal) &&
                               !materialize_gpu_replay(read, refused, refusal) &&
                               refusal == expected;
                    };
                    auto bad = captured;
                    bad.draws[0].owned_waves.reset();
                    expect(refuses(bad, "logical-wave replay lacks original exact-PC window owners "
                                        "stage=vs pc=3"),
                           "removing owners cannot acquire stored or raw replay authority");
                    // A nonempty stored module cannot hide unavailable original analysis. Keep
                    // the other stage ordinary so it cannot supply an unrelated owner refusal.
                    const std::vector<uint32_t> ordinary_fs{0x7e000280u, 0x7e0202f2u, 0x7e040280u,
                                                            0x7e0602f2u, 0xf800180fu, 0x03020100u,
                                                            0xbf810000u};
                    auto stored_fallback = captured;
                    stored_fallback.draws[0].owned_waves.reset();
                    stored_fallback.draws[0].vs = commit.shader;
                    stored_fallback.draws[0].fs =
                        recompile_fragment(ordinary_fs.data(), ordinary_fs.size());
                    ASSERT_FALSE(stored_fallback.draws[0].vs.empty());
                    ASSERT_FALSE(stored_fallback.draws[0].fs.empty());
                    const auto set_raw = [&](GpuCaptureFile& file, bool vertex,
                                             const std::vector<uint32_t>& original) {
                        auto& raw =
                            file.raw_shader_versions[vertex ? file.draws[0].vs_raw_shader_index
                                                            : file.draws[0].fs_raw_shader_index];
                        raw.words = original;
                        raw.content_hash =
                            gpu_capture_hash(reinterpret_cast<const uint8_t*>(original.data()),
                                             original.size() * sizeof(uint32_t));
                    };
                    set_raw(stored_fallback, false, ordinary_fs);
                    for (uint32_t shape = 0; shape < 3u; ++shape) {
                        auto refused_original = instance_code;
                        if (shape == 0u) refused_original.erase(refused_original.begin() + 1u);
                        if (shape == 1u)
                            refused_original.insert(refused_original.end() - 1u, 513u, 0xbf800000u);
                        if (shape == 2u)
                            refused_original.insert(refused_original.end() - 1u, {0xd8000000u, 0u});
                        std::vector<Rdna2Inst> original_decoded;
                        rdna2_walk(refused_original.data(), refused_original.size(),
                                   original_decoded);
                        expect(rdna2_raw_wave_wide_certificates(original_decoded).empty() &&
                                   !rdna2_raw_wave_wide_data_loads(original_decoded).empty(),
                               "refused full-original admission retains an independent ownership "
                               "obligation");
                        auto fallback = stored_fallback;
                        set_raw(fallback, true, refused_original);
                        expect(refuses(fallback, "logical-wave replay lacks original exact-PC "
                                                 "window owners stage=vs pc=" +
                                                     std::to_string(shape == 0u ? 2u : 3u)),
                               "unbounded, over-budget and DS originals cannot acquire legacy "
                               "stored replay authority");
                    }
                    bool replay_framed = false;
                    const auto replay_stored = [&](const GpuCaptureFile& file, bool legacy,
                                                   GpuReplayFrame& result) {
                        replay_framed = false;
                        GpuCaptureFile read;
                        std::vector<uint8_t> bytes;
                        if (!serialize_gpu_capture(file, bytes, refusal)) return false;
                        if (legacy) {
                            // Genuine ownerless v69 prefix: remove the entire v70 count1/flags0
                            // tail before changing the version, rather than relabeling full bytes.
                            const std::array<uint8_t, 5> tail{1u, 0u, 0u, 0u, 0u};
                            if (bytes.size() < 17u ||
                                !std::equal(tail.begin(), tail.end(), bytes.end() - 5u))
                                return false;
                            bytes.resize(bytes.size() - 5u);
                            bytes[8] = 69u;
                            bytes[9] = bytes[10] = bytes[11] = 0u;
                        }
                        if (!deserialize_gpu_capture(bytes, read, refusal)) return false;
                        replay_framed = true;
                        return materialize_gpu_replay(read, result, refusal);
                    };
                    // READFIRST is present but its dead result has no numeric load consumer.
                    // Both stored stages are real modules compiled from these complete originals.
                    const std::vector<uint32_t> ordinary_vs{0x7e000280u, 0x7e280500u, 0x7e020280u,
                                                            0x7e040280u, 0x7e0602f2u, 0xf80008cfu,
                                                            0x03020100u, 0xbf810000u};
                    auto healthy = stored_fallback;
                    healthy.draws[0].vs = recompile_vertex(ordinary_vs.data(), ordinary_vs.size());
                    ASSERT_FALSE(healthy.draws[0].vs.empty());
                    set_raw(healthy, true, ordinary_vs);
                    std::vector<Rdna2Inst> healthy_decoded;
                    ASSERT_EQ(rdna2_walk(ordinary_vs.data(), ordinary_vs.size(), healthy_decoded),
                              ordinary_vs.size());
                    ASSERT_TRUE(healthy_decoded.back().is_end);
                    ASSERT_TRUE(rdna2_raw_wave_wide_data_loads(healthy_decoded).empty());
                    auto alternate_numeric = instance_code;
                    alternate_numeric.insert(alternate_numeric.begin() + 1u,
                                             {0xbf840001u, 0xbf810000u});
                    std::vector<Rdna2Inst> alternate_inventory;
                    uint32_t unavailable_pc = UINT32_MAX;
                    ASSERT_TRUE(rdna2_recompile_executable_instructions(
                        alternate_numeric.data(), alternate_numeric.size(), alternate_inventory,
                        unavailable_pc));
                    ASSERT_EQ(rdna2_raw_wave_wide_data_loads(alternate_inventory),
                              std::vector<uint32_t>{5u});
                    auto alternate_fallback = stored_fallback;
                    set_raw(alternate_fallback, true, alternate_numeric);

                    // Existing compiler-proven embedded-table shape. Its first two data words
                    // resemble unsupported encodings; they must remain data, never invented PCs.
                    const std::vector<uint32_t> table_vs{
                        0xb0020010u, 0xbe8303ffu, 0x10005004u, 0xbe801f00u, 0x800000ffu,
                        0x00000028u, 0x82010180u, 0x7e080280u, 0xe0381000u, 0x80000004u,
                        0xbf8c3f70u, 0xf80008cfu, 0x03020100u, 0xbf810000u, 0xffffffffu,
                        0xffffffffu, 0u,          0x3f800000u};
                    auto healthy_table = healthy;
                    healthy_table.draws[0].vs = recompile_vertex(table_vs.data(), table_vs.size());
                    ASSERT_FALSE(healthy_table.draws[0].vs.empty());
                    set_raw(healthy_table, true, table_vs);

                    const std::vector<uint32_t> prolog{0x7e000280u, 0xbe802006u};
                    auto healthy_chain = healthy;
                    const auto main_index = healthy_chain.draws[0].vs_raw_shader_index;
                    auto prolog_raw = healthy_chain.raw_shader_versions[main_index];
                    prolog_raw.words = prolog;
                    prolog_raw.content_hash =
                        gpu_capture_hash(reinterpret_cast<const uint8_t*>(prolog.data()),
                                         prolog.size() * sizeof(uint32_t));
                    healthy_chain.draws[0].vs_raw_shader_index =
                        uint32_t(healthy_chain.raw_shader_versions.size());
                    healthy_chain.raw_shader_versions.push_back(std::move(prolog_raw));
                    healthy_chain.draws[0].vs_chain_raw_shader_index = main_index;
                    healthy_chain.draws[0].vs = recompile_vertex_chain(
                        prolog.data(), prolog.size(), ordinary_vs.data(), ordinary_vs.size());
                    ASSERT_FALSE(healthy_chain.draws[0].vs.empty());
                    for (bool legacy : {false, true}) {
                        GpuReplayFrame ordinary;
                        ASSERT_TRUE(replay_stored(healthy, legacy, ordinary)) << refusal;
                        ASSERT_EQ(ordinary.items.size(), 1u);
                        expect(!ordinary.items[0].owned_waves &&
                                   ordinary.items[0].vs == healthy.draws[0].vs &&
                                   ordinary.items[0].fs == healthy.draws[0].fs,
                               "complete no-obligation original retains both healthy stored stages "
                               "in current and v69 replay");
                        for (const auto* positive : {&healthy_table, &healthy_chain}) {
                            GpuReplayFrame supported;
                            ASSERT_TRUE(replay_stored(*positive, legacy, supported)) << refusal;
                            ASSERT_EQ(supported.items.size(), 1u);
                            expect(!supported.items[0].owned_waves &&
                                       supported.items[0].vs == positive->draws[0].vs &&
                                       supported.items[0].fs == positive->draws[0].fs,
                                   "proved ordinary constant tables and linked prologs retain real "
                                   "stored stages");
                        }
                        GpuReplayFrame alternate_refused;
                        expect(!replay_stored(alternate_fallback, legacy, alternate_refused) &&
                                   alternate_refused.items.empty() &&
                                   refusal == "logical-wave replay lacks original exact-PC window "
                                              "owners stage=vs pc=5",
                               "numeric load in the second terminating arm cannot disappear behind "
                               "stored modules");
                        // Retaining an unreferenced prolog is a malformed capture, rejected by
                        // the codec before the semantic linked-stage proof can run.
                        auto orphaned_prolog = healthy_chain;
                        orphaned_prolog.draws[0].vs_raw_shader_index = UINT32_MAX;
                        std::vector<uint8_t> orphaned_bytes;
                        expect(
                            !serialize_gpu_capture(orphaned_prolog, orphaned_bytes, refusal) &&
                                orphaned_bytes.empty() && refusal == "raw shader is not referenced",
                            "orphaned raw prolog is refused before any capture bytes are emitted");
                        for (uint32_t broken = 0; broken < 5u; ++broken) {
                            auto unproved_chain = healthy_chain;
                            if (broken == 0u)
                                set_raw(unproved_chain, true, {0x7e000280u, 0xbe802008u});
                            if (broken == 1u) {
                                // Remove only the newly appended prolog record, preserving a
                                // well-framed main/FS capture with genuinely missing VS provenance.
                                ASSERT_EQ(unproved_chain.draws[0].vs_raw_shader_index,
                                          unproved_chain.raw_shader_versions.size() - 1u);
                                ASSERT_LT(unproved_chain.draws[0].vs_chain_raw_shader_index,
                                          unproved_chain.draws[0].vs_raw_shader_index);
                                ASSERT_LT(unproved_chain.draws[0].fs_raw_shader_index,
                                          unproved_chain.draws[0].vs_raw_shader_index);
                                unproved_chain.raw_shader_versions.pop_back();
                                unproved_chain.draws[0].vs_raw_shader_index = UINT32_MAX;
                            }
                            if (broken == 2u) {
                                unproved_chain.draws[0].vs_chain_raw_shader_index =
                                    uint32_t(unproved_chain.raw_shader_versions.size());
                                std::vector<uint8_t> invalid_index_bytes;
                                expect(
                                    !serialize_gpu_capture(unproved_chain, invalid_index_bytes,
                                                           refusal) &&
                                        invalid_index_bytes.empty() &&
                                        refusal == "realized draw references an invalid raw shader",
                                    "out-of-range chain index is refused before any capture bytes "
                                    "are emitted");
                                continue;
                            }
                            if (broken == 3u)
                                set_raw(unproved_chain, true,
                                        {0xbf820001u, 0x7e0002ffu, 0u, 0xbe802006u});
                            if (broken == 4u)
                                set_raw(unproved_chain, true, {0xbe802106u, 0xbe802006u});
                            GpuReplayFrame chain_refused;
                            expect(!replay_stored(unproved_chain, legacy, chain_refused) &&
                                       replay_framed && chain_refused.items.empty() &&
                                       refusal == "logical-wave replay linked vertex provenance "
                                                  "unavailable stage=vs",
                                   "declared linkage without the exact s[6:7] prolog and present "
                                   "originals grants no stored authority");
                        }
                        for (uint32_t transfer :
                             {0xbe802006u, 0xbe802106u, 0xbe802206u, 0xbb060001u, 0xbd800001u,
                              0xbe000001u, 0xbf920000u, 0xbf970001u, 0xbf980001u, 0xbf990001u,
                              0xbf9a0001u}) {
                            auto unproved_transfer = healthy;
                            set_raw(unproved_transfer, true, {transfer, 0xbf810000u});
                            GpuReplayFrame transfer_refused;
                            expect(!replay_stored(unproved_transfer, legacy, transfer_refused) &&
                                       transfer_refused.items.empty() &&
                                       refusal == "logical-wave replay original decode unavailable "
                                                  "stage=vs pc=0",
                                   "unproved indirect/call/subvector/trap/debug transfers cannot "
                                   "yield an empty safe inventory");
                        }
                        for (uint32_t malformed = 0; malformed < 4u; ++malformed) {
                            auto undecoded = malformed == 3u ? healthy : stored_fallback;
                            auto original = malformed == 3u ? ordinary_fs : instance_code;
                            const uint32_t stop_pc = malformed == 2u
                                                         ? uint32_t(original.size() - 1u)
                                                     : malformed == 1u ? 0u
                                                     : malformed == 3u ? 0u
                                                                       : 1u;
                            if (malformed == 2u)
                                original.back() = 0xffffffffu;
                            else
                                original.insert(original.begin() + stop_pc, 0xffffffffu);
                            set_raw(undecoded, malformed != 3u, original);
                            GpuReplayFrame refused;
                            expect(
                                !replay_stored(undecoded, legacy, refused) &&
                                    refused.items.empty() &&
                                    refusal ==
                                        "logical-wave replay original decode unavailable stage=" +
                                            std::string(malformed == 3u ? "fs" : "vs") +
                                            " pc=" + std::to_string(stop_pc),
                                "Unknown before/after READFIRST or at the terminal word cannot "
                                "authorize stored fallback");
                        }
                    }
                    const auto changed_owner = [&] {
                        return std::make_shared<GraphicsOwnedWaveDraw>(*captured_owner);
                    };
                    auto malformed = changed_owner();
                    malformed->fragment_windows[0].words.pop_back();
                    bad = captured;
                    bad.draws[0].owned_waves = malformed;
                    expect(refuses(bad, "wave-window-complete-owner-shape-invalid"),
                           "one missing highest numeric word refuses the whole captured selector "
                           "domain");
                    malformed = changed_owner();
                    ++malformed->fragment_windows[0].load_pc;
                    bad = captured;
                    bad.draws[0].owned_waves = malformed;
                    expect(refuses(bad, "wave-window-exact-code-or-base-unavailable"),
                           "a plausible window at a different PC cannot satisfy the original load");
                    bad = captured;
                    bad.draws[0].ps_entry.user_data[0] ^= 4u;
                    expect(refuses(bad, "owned fragment-wave replay entry observation disagrees"),
                           "physical launch observation and retained normalized entry must agree");
                    auto prefix32 = captured;
                    auto prefix_owner = changed_owner();
                    prefix32.draws[0].ps_entry.rsrc2 =
                        1u << prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_MSB_SHIFT;
                    prefix32.draws[0].ps_entry.user_data_available = UINT32_MAX;
                    for (uint32_t reg = 2u; reg < 32u; ++reg)
                        prefix_owner->fragment_scalars.emplace_back(reg, 0u);
                    prefix32.draws[0].owned_waves = prefix_owner;
                    std::vector<uint8_t> prefix_bytes;
                    GpuCaptureFile prefix_read;
                    GpuReplayFrame prefix_replay;
                    ASSERT_TRUE(serialize_gpu_capture(prefix32, prefix_bytes, refusal)) << refusal;
                    ASSERT_TRUE(deserialize_gpu_capture(prefix_bytes, prefix_read, refusal))
                        << refusal;
                    ASSERT_TRUE(materialize_gpu_replay(prefix_read, prefix_replay, refusal))
                        << refusal;
                    ASSERT_EQ(prefix_replay.items.size(), 1u);
                    ASSERT_TRUE(prefix_replay.items[0].owned_waves);
                    ASSERT_EQ(prefix_replay.items[0].owned_waves->fragment_scalars.size(), 32u);
                    EXPECT_EQ(prefix_replay.items[0].owned_waves->fragment_scalars.back(),
                              std::make_pair(31u, 0u));
                    for (uint32_t invalid_count : {33u, 63u}) {
                        bad = prefix32;
                        bad.draws[0].ps_entry.rsrc2 |=
                            (invalid_count & 31u)
                            << prosper::agc::Pm4::SPI_SHADER_PGM_RSRC2_PS_USER_SGPR_SHIFT;
                        expect(refuses(bad,
                                       "owned fragment-wave replay user prefix count out of range"),
                               "high-bit invalid PS counts refuse before prefix iteration or owner "
                               "publication");
                    }
                    bad = captured;
                    bad.draws[0].vertex_offset = 1;
                    expect(
                        refuses(bad, "owned vertex-wave replay invocation identity disagrees"),
                        "completed invocation plan cannot be replayed as a different indexed draw");
                    for (size_t end : {wire.size() - 1u, wire.size() - 2u}) {
                        auto truncated = wire;
                        truncated.resize(end);
                        expect(!deserialize_gpu_capture(truncated, decoded_capture, refusal),
                               "truncated v70 owned payload refuses without padding missing bytes");
                    }
                }
                auto failed = outputs;
                if (!failed.back().empty()) failed.back()[0] = 0u;
                expect(!validate_graphics_wave_outputs(stages, failed, transaction, refusal) &&
                           transaction.records.empty(),
                       "failed second wave publishes no partial first-wave transaction");
                failed = outputs;
                if (failed.back().size() > 3u) failed.back()[3] = 0u;
                expect(!validate_graphics_wave_outputs(stages, failed, transaction, refusal) &&
                           transaction.records.empty() &&
                           refusal == "graphics-wave-output-export-metadata-invalid",
                       "unmatched original EXP metadata cannot acquire output commit authority");
            }
            std::vector<uint32_t> indexed(64u);
            for (uint32_t i = 0; i < indexed.size(); ++i) indexed[i] = 127u - i;
            expect(prepare_owned_vertex_waves(vertex_code, scalars_for(base), indexed, 0u, -32, 1u,
                                              {}, &complete, stages, refusal) &&
                       stages.invocations[0][0].vertex_index == 95u &&
                       stages.invocations[0][63].vertex_index == 32u,
                   "indexed logical identities use the owned occurrence and actual signed base "
                   "vertex");
            expect(!prepare_owned_vertex_waves(vertex_code, scalars_for(base), {}, 63u, 0, 1u, {},
                                               &complete, stages, refusal) &&
                       stages.packets.empty() &&
                       refusal == "vertex-wave-complete-domain-unavailable",
                   "a partial final stage wave cannot be relabeled as a full64 input witness");
        }
        set_graphics_raw_source_authority({});
    }
    expect(!admit_compute_nested_wide_data(nullptr, decoded, uses, table) &&
               table.owned_host_data.empty(),
           "offline realization cannot claim live mapping ownership");
#if defined(__linux__) || defined(_WIN32)
    {
        GuestMappingLease lease;
        expect(admit_compute_nested_wide_data(&lease, decoded, uses, table),
               "disjoint complete direct ranges admit the real nested handoff");
        const ShaderResource* p = table.by_fetch_pc(0);
        const ShaderResource* c = table.by_fetch_pc(2);
        uint32_t captured = 0;
        if (c && c->host_data && c->host_data_size >= 0x1c)
            std::memcpy(&captured, c->host_data + 0x18, sizeof(captured));
        expect(p && p->host_data && c && c->host_data &&
                   table.owned_host_data.size() == 2 &&
                   captured == original_child_value,
               "parent and child carry independent owned current-byte snapshots");
    }
    const uint64_t replacement = alias;
    std::memcpy(reinterpret_cast<void*>(parent), &replacement, sizeof(replacement));
    expect(table.by_fetch_pc(0) && table.by_fetch_pc(0)->host_data &&
               std::memcmp(table.by_fetch_pc(0)->host_data, &child, sizeof(child)) == 0,
           "later parent mutation cannot change a submitted snapshot");
    auto stale_child = table_for();
    {
        GuestMappingLease lease;
        expect(!admit_compute_nested_wide_data(&lease, decoded, uses, stale_child) &&
                   stale_child.owned_host_data.empty() &&
                   !stale_child.by_fetch_pc(2)->host_data,
               "parent pointer changed after fold refuses a mixed child binding");
    }
    std::memcpy(reinterpret_cast<void*>(parent), &child, sizeof(child));
    auto aliased_child = table_for();
    aliased_child.resources[1].gpu_addr = alias;
    std::memcpy(reinterpret_cast<void*>(parent), &alias, sizeof(alias));
    {
        GuestMappingLease lease;
        expect(!admit_compute_nested_wide_data(&lease, decoded, uses, aliased_child),
               "same physical backing through another VA cannot supply numeric data");
    }
    std::memcpy(reinterpret_cast<void*>(parent), &child, sizeof(child));
    auto aliased_descriptor = table_for();
    auto bad_uses = uses;
    bad_uses[0].descriptor_source_addr = output + 0x5000;
    {
        GuestMappingLease lease;
        expect(!admit_compute_nested_wide_data(&lease, decoded, bad_uses,
                                                aliased_descriptor),
               "descriptor source in tiled padding is a writable alias");
    }
    auto unknown_write = table_for();
    unknown_write.resources.back().metadata_addr = output;
    {
        GuestMappingLease lease;
        expect(!admit_compute_nested_wide_data(&lease, decoded, uses, unknown_write),
               "metadata-bearing output has no proved complete write footprint");
    }
    expect(unmap(child, page, 0, 0, 0, 0) == 0,
           "child can be unmapped after dispatch lease release");
    auto unmapped = table_for();
    {
        GuestMappingLease lease;
        expect(!admit_compute_nested_wide_data(&lease, decoded, uses, unmapped),
               "unmapped child immediately loses admission");
    }
#else
    {
        GuestMappingLease lease;
        expect(!admit_compute_nested_wide_data(&lease, decoded, uses, table),
               "platform without lazy-fault proof remains fail closed");
    }
#endif
    GuestDirectAllocation split_origin;
    {
        GuestMappingLease lease;
        split_origin = guest_memory_direct_allocation(lease, output, 1u);
    }
    expect(release(physical + page, page, 0, 0, 0, 0) == 0,
           "partially release an allocation while a retained origin still owns its original bounds");
    uint64_t replacement_physical = 0, replacement_view = 0;
    expect(allocate(physical + page, physical + 2u * page, page, page, 0,
                    reinterpret_cast<uint64_t>(&replacement_physical)) == 0 &&
           replacement_physical == physical + page &&
           map(reinterpret_cast<uint64_t>(&replacement_view), page, 3u, 0u,
               replacement_physical, page) == 0 && replacement_view,
           "reuse only the released middle slice as a separate allocation birth");
    {
        GuestMappingLease lease;
        expect(!guest_memory_direct_allocation(lease, output, 1u).identity &&
               replacement_view && guest_memory_retained_allocation_relation(lease,
                   replacement_view, 4u, split_origin) == GuestMemoryTopologyRelation::Overlap,
               "partial release cannot shrink old producer bounds or reauthenticate a broken origin");
    }
    if (replacement_view) unmap(replacement_view, page, 0, 0, 0, 0);
    if (wave_alias) unmap(wave_alias, page, 0, 0, 0, 0);
    if (wave_source) unmap(wave_source, page, 0, 0, 0, 0);
    if (wave_physical) release(wave_physical, page, 0, 0, 0, 0);
    if (wave_output) unmap(wave_output, page, 0, 0, 0, 0);
    if (wave_output_physical) release(wave_output_physical, page, 0, 0, 0, 0);
    unmap(parent, page, 0, 0, 0, 0);
#if !defined(__linux__) && !defined(_WIN32)
    unmap(child, page, 0, 0, 0, 0);
#endif
    unmap(output, page, 0, 0, 0, 0);
    unmap(alias, page, 0, 0, 0, 0);
    if (isolated) unmap(isolated, page, 0, 0, 0, 0);
    if (isolated_physical) release(isolated_physical, page, 0, 0, 0, 0);
    release(physical, 4 * page, 0, 0, 0, 0);
    EXPECT_EQ(failures, 0);
}
