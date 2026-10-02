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
    unmap(parent, page, 0, 0, 0, 0);
#if !defined(__linux__)
    unmap(child, page, 0, 0, 0, 0);
#endif
    unmap(output, page, 0, 0, 0, 0);
    unmap(alias, page, 0, 0, 0, 0);
    release(physical, 4 * page, 0, 0, 0, 0);
    EXPECT_EQ(failures, 0);
}
