#include "gpu/execute/gpu_execute.hpp"
#include <gtest/gtest.h>
#include "gpu/recompiler/rdna2_decode.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "gpu/texture/tile.hpp"
#include "hle/dispatch/dispatch.hpp"
#include "hle/dispatch/nid.hpp"
#include "hle/memory/guest_memory_topology.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace prosper;
using namespace prosper::gpu;

TEST(NestedWideDataAdmission, Contract) {
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
    expect(!admit_compute_nested_wide_data(nullptr, decoded, uses, table) &&
               table.owned_host_data.empty(),
           "offline realization cannot claim live mapping ownership");
#if defined(__linux__)
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
    unmap(parent, page, 0, 0, 0, 0);
#if !defined(__linux__)
    unmap(child, page, 0, 0, 0, 0);
#endif
    unmap(output, page, 0, 0, 0, 0);
    unmap(alias, page, 0, 0, 0, 0);
    if (isolated) unmap(isolated, page, 0, 0, 0, 0);
    if (isolated_physical) release(isolated_physical, page, 0, 0, 0, 0);
    release(physical, 4 * page, 0, 0, 0, 0);
    EXPECT_EQ(failures, 0);
}
