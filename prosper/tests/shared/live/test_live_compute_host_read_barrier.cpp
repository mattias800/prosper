// test_live_compute_host_read_barrier — #3249: the live compute backend host-reads every dispatch
// result through a mapped pointer after a fence wait. A fence orders EXECUTION; it does not perform
// the availability operation that moves the shader's (or the image copy's) writes into the HOST
// domain. That needs a dependency with dstStageMask = VK_PIPELINE_STAGE_HOST_BIT and
// dstAccessMask = VK_ACCESS_HOST_READ_BIT.
//
// WHAT THIS TEST CANNOT DO, and why it is shaped the way it is.
//
// It cannot detect the defect from the dispatch's RESULT. Every allocation the compute backend maps
// is HOST_VISIBLE|HOST_COHERENT (both tiers of `host_memory_type` require it), and on this
// platform's driver the UNFIXED code returns byte-identical results — which is exactly why the gap
// survived. `live_compute_descriptor_array` and `game_compute_exec` both assert dispatch results
// and both stayed green with no barrier recorded anywhere. Synchronization validation cannot see it
// either: it does not observe a CPU read through a mapped pointer (#3248, measured on #3250).
//
// So the discriminator is structural: the barrier must be RECORDED into the dispatch command
// buffer. Arm 2 fails without the fix (delta 0, not 1); arm 3 is the negative control that stops a
// counter incremented anywhere on the compute path from satisfying arm 2 for the wrong reason.
#include "shared/live/live_compute.hpp"
#include "gpu/execute/host_read_barrier.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace prosper::gpu;

static int failures = 0;
#define CHECK(condition, message)                                                   \
    do {                                                                            \
        if (!(condition)) {                                                         \
            std::printf("  [FAIL] %s\n", message);                                  \
            ++failures;                                                             \
        } else {                                                                    \
            std::printf("  [ok]   %s\n", message);                                  \
        }                                                                           \
    } while (0)

static uint64_t barriers() {
    return prosper::gpu::backend_host_read_barrier_count().load(std::memory_order_relaxed);
}

static ShaderResource scalar_buffer(uint32_t binding, uint32_t sgpr_base, uint32_t* storage,
                                    uint32_t bytes) {
    ShaderResource resource;
    resource.cls = ResourceClass::ConstantBuffer;
    resource.format = DataFormat::Uint32;
    resource.num_components = 1;
    resource.binding = binding;
    resource.gpu_addr = reinterpret_cast<uint64_t>(storage);
    resource.size = bytes;
    resource.stride = sizeof(uint32_t);
    resource.sgpr_base = sgpr_base;
    resource.host_data = reinterpret_cast<uint8_t*>(storage);
    resource.host_data_size = bytes;
    return resource;
}

int main() {
    std::printf("== test_live_compute_host_read_barrier ==\n");

    // ---- Arm 1: the exact masks, with no device -----------------------------------------------
    // The helper is shared with the render backend (`host_read_barrier` pins it there too). What is
    // asserted here is that the COMPUTE source scope — a shader write, not a transfer write — does
    // not disturb the host half, since #3249's whole point is that the two producers differ.
    {
        VkBuffer buffer = reinterpret_cast<VkBuffer>(uintptr_t{0x3249});
        const HostReadBarrier shader_write = prosper::gpu::host_read_barrier_for(
            buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        CHECK(shader_write.barrier.srcAccessMask == VK_ACCESS_SHADER_WRITE_BIT &&
                  shader_write.src_stages == VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
              "a dispatch result names the compute shader write as its source scope");
        CHECK(shader_write.barrier.dstAccessMask == VK_ACCESS_HOST_READ_BIT &&
                  shader_write.dst_stages == VK_PIPELINE_STAGE_HOST_BIT,
              "the host half is exactly HOST_READ at the HOST stage");
        CHECK(shader_write.barrier.buffer == buffer && shader_write.barrier.offset == 0 &&
                  shader_write.barrier.size == VK_WHOLE_SIZE &&
                  shader_write.barrier.srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
                  shader_write.barrier.dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED,
              "whole-buffer scope with no queue-family ownership transfer");

        const HostReadBarrier transfer_write = prosper::gpu::host_read_barrier_for(buffer);
        CHECK(transfer_write.barrier.srcAccessMask == VK_ACCESS_TRANSFER_WRITE_BIT &&
                  transfer_write.src_stages == VK_PIPELINE_STAGE_TRANSFER_BIT,
              "a storage-image staging readback defaults to the transfer write scope");
        CHECK(transfer_write.barrier.dstAccessMask == shader_write.barrier.dstAccessMask &&
                  transfer_write.dst_stages == shader_write.dst_stages,
              "the host half does not change with the producer");
    }

    // ---- The fixture both execution arms share -------------------------------------------------
    uint32_t input = 0x5a5aa5a5u;
    uint32_t output = 0u;

    // load v1 = input[0]; store it through the writable output binding.
    const uint32_t write_code[] = {
        0xe0300000u, 0x80000100u, // buffer_load_dword v1, off, s[0:3]
        0xbf8c3f70u,              // s_waitcnt vmcnt(0)
        0xe0700000u, 0x80010100u, // buffer_store_dword v1, off, s[4:7]
        0xbf810000u,              // s_endpgm
    };
    // The same kernel with the store removed: one READ-ONLY binding and nothing the host reads back.
    const uint32_t read_only_code[] = {
        0xe0300000u, 0x80000100u, // buffer_load_dword v1, off, s[0:3]
        0xbf8c3f70u,              // s_waitcnt vmcnt(0)
        0xbf810000u,              // s_endpgm
    };

    auto build = [](const uint32_t* code, size_t words, ShaderResourceTable resources,
                    std::vector<uint32_t>& spirv, ComputeShaderConfig& config) {
        config.user_sgprs.resize(8);
        config.local_x = config.local_y = config.local_z = 1;
        spirv = recompile_compute(code, words, &resources, config);
    };

    ShaderResourceTable write_resources;
    write_resources.resources = {scalar_buffer(0, 0, &input, sizeof(input)),
                                 scalar_buffer(1, 4, &output, sizeof(output))};
    std::vector<uint32_t> write_spirv;
    ComputeShaderConfig write_config;
    build(write_code, std::size(write_code), write_resources, write_spirv, write_config);
    CHECK(!write_spirv.empty(), "the writable-result fixture recompiles");

    ShaderResourceTable read_resources;
    read_resources.resources = {scalar_buffer(0, 0, &input, sizeof(input))};
    std::vector<uint32_t> read_spirv;
    ComputeShaderConfig read_config;
    build(read_only_code, std::size(read_only_code), read_resources, read_spirv, read_config);
    CHECK(!read_spirv.empty(), "the read-only control fixture recompiles");
    if (write_spirv.empty() || read_spirv.empty()) {
        std::printf("== FAIL ==\n");
        return 1;
    }

    // The negative control is only a control if its binding really is reflected as read-only: a
    // dispatch that reflects NO descriptor at all would record no barrier for a reason that has
    // nothing to do with the fix.
    const DescriptorValidationReport read_report = validate_spirv_descriptor_interface(
        read_spirv, &read_resources, 0, SpirvShaderStage::Compute, false);
    const bool read_only_binding_present =
        read_report.ok() && !read_report.descriptors.empty() &&
        std::none_of(read_report.descriptors.begin(), read_report.descriptors.end(),
                     [](const SpirvDescriptorBinding& binding) { return binding.writable; });
    CHECK(read_only_binding_present,
          "the control reflects a storage-buffer binding and reflects it as READ-ONLY");

    const DescriptorValidationReport write_report = validate_spirv_descriptor_interface(
        write_spirv, &write_resources, 0, SpirvShaderStage::Compute, false);
    const size_t writable_bindings =
        static_cast<size_t>(std::count_if(
            write_report.descriptors.begin(), write_report.descriptors.end(),
            [](const SpirvDescriptorBinding& binding) { return binding.writable; }));
    CHECK(write_report.ok() && writable_bindings == 1,
          "the writable fixture reflects exactly one writable binding, so one barrier is expected");

    auto make_item = [](const std::vector<uint32_t>& spirv, const ComputeShaderConfig& config,
                        const ShaderResourceTable& resources) {
        ComputeItem item;
        item.spirv = spirv;
        item.user_sgprs = config.user_sgprs;
        item.resources = std::make_shared<ShaderResourceTable>(resources);
        item.launch.threads_x = item.launch.threads_y = item.launch.threads_z = 1;
        item.launch.local_x = item.launch.local_y = item.launch.local_z = 1;
        item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
        item.recompile_config = config;
        item.recompile_config_available = true;
        return item;
    };

    // ---- Arm 2: a real dispatch records the availability dependency ----------------------------
    ComputeItem write_item = make_item(write_spirv, write_config, write_resources);
    write_item.code_addr = 0x32490000u;
    const uint64_t before_write = barriers();
    const bool wrote = prosper::frontend::execute_live_compute_items({write_item});
    const uint64_t write_delta = barriers() - before_write;
    CHECK(wrote && output == input,
          "the production backend ran the dispatch and wrote the result back to guest memory");
    std::printf("  host-read barriers recorded by one writable-buffer dispatch: %llu\n",
                static_cast<unsigned long long>(write_delta));
    CHECK(write_delta == 1,
          "a writable dispatch result records exactly one host-read barrier before the host maps it");

    // ---- Arm 3: negative control ---------------------------------------------------------------
    ComputeItem read_item = make_item(read_spirv, read_config, read_resources);
    read_item.code_addr = 0x32490100u;
    const uint64_t before_read = barriers();
    const bool read = prosper::frontend::execute_live_compute_items({read_item});
    const uint64_t read_delta = barriers() - before_read;
    std::printf("  host-read barriers recorded by one read-only dispatch: %llu\n",
                static_cast<unsigned long long>(read_delta));
    CHECK(read, "the read-only control dispatch also executes");
    CHECK(read_delta == 0,
          "a dispatch the host never reads back records no host-read barrier");

    // ---- Arm 4: the OTHER producer — a storage image, whose staging buffer is written by
    // vkCmdCopyImageToBuffer rather than by the shader. Same helper, different source scope, and it
    // is the site the guest writeback reads to restore a texture into guest memory.
    {
        // v4 = x from the shell input; image_load from binding 4; image_store into binding 5.
        static const uint32_t image_copy[] = {
            0x7E080300u, 0xF0000F00u, 0x00000004u, 0xBF8C3F70u,
            0xF0200F00u, 0x00020004u, 0xBF810000u,
        };
        constexpr uint32_t texels = 64;
        std::vector<uint32_t> lane_index(texels);
        for (uint32_t i = 0; i < texels; ++i) lane_index[i] = i;
        std::vector<uint32_t> unused(4, 0);
        std::vector<uint8_t> source(texels * 4), destination(texels * 4, 0xee);
        for (uint32_t i = 0; i < texels * 4; ++i) source[i] = static_cast<uint8_t>(i * 37 + 5);

        ShaderResourceTable table;
        auto add_buffer = [&](uint32_t binding, void* data, uint32_t bytes) {
            ShaderResource resource{};
            resource.cls = ResourceClass::ConstantBuffer;
            resource.binding = binding;
            resource.gpu_addr = reinterpret_cast<uint64_t>(data);
            resource.size = bytes;
            table.resources.push_back(resource);
        };
        add_buffer(0, lane_index.data(), texels * sizeof(uint32_t));
        add_buffer(1, unused.data(), 16);
        add_buffer(2, unused.data(), 16);
        add_buffer(3, unused.data(), 16);
        auto add_image = [&](uint32_t binding, uint32_t sgpr, void* data, uint32_t bytes) {
            ShaderResource image{};
            image.cls = ResourceClass::StorageImage;
            image.img_dim = 0;                     // 1D
            image.binding = binding;
            image.sgpr_base = sgpr;
            image.format = DataFormat::Unorm8;
            image.num_components = 4;
            image.width = texels;
            image.height = 1;
            image.gpu_addr = reinterpret_cast<uint64_t>(data);
            image.size = bytes;
            table.resources.push_back(image);
        };
        add_image(4, 0, source.data(), texels * 4);
        add_image(5, 8, destination.data(), texels * 4);

        const std::vector<uint32_t> image_spirv =
            recompile_valu(image_copy, std::size(image_copy), 1, 0, &table);
        CHECK(!image_spirv.empty(), "the storage-image fixture recompiles");
        // Derive the expectation instead of hardcoding it. The VALU shell this fixture uses also
        // binds its own writable output buffer, so the dispatch has BOTH producers: one shader
        // write into a storage buffer and one transfer write into each storage image's staging
        // buffer. Both storage images are copied back regardless of shader writability, because the
        // guest writeback maps and reads both.
        const DescriptorValidationReport image_report = validate_spirv_descriptor_interface(
            image_spirv, &table, 0, SpirvShaderStage::Compute, false);
        const uint64_t expected_image_barriers = static_cast<uint64_t>(std::count_if(
            image_report.descriptors.begin(), image_report.descriptors.end(),
            [](const SpirvDescriptorBinding& binding) {
                return binding.kind == SpirvDescriptorKind::StorageImage ||
                       (binding.kind == SpirvDescriptorKind::StorageBuffer && binding.writable);
            }));
        CHECK(image_report.ok() && expected_image_barriers == 3,
              "the storage-image fixture reflects two storage images and one writable buffer");
        if (!image_spirv.empty()) {
            ComputeItem image_item;
            image_item.spirv = image_spirv;
            image_item.resources = std::make_shared<ShaderResourceTable>(table);
            image_item.launch.threads_x = texels;
            image_item.launch.local_x = 64;
            image_item.launch.local_y = image_item.launch.local_z = 1;
            image_item.launch.groups_x = 1;
            image_item.launch.groups_y = image_item.launch.groups_z = 1;
            image_item.code_addr = 0x32490200u;
            const uint64_t before_image = barriers();
            const bool copied = prosper::frontend::execute_live_compute_items({image_item});
            const uint64_t image_delta = barriers() - before_image;
            CHECK(copied && destination == source,
                  "the production backend ran the storage-image copy and wrote guest memory");
            std::printf("  host-read barriers recorded by a two-storage-image dispatch: %llu"
                        " (expected %llu)\n",
                        static_cast<unsigned long long>(image_delta),
                        static_cast<unsigned long long>(expected_image_barriers));
            CHECK(image_delta == expected_image_barriers,
                  "every staging readback and every writable buffer records its own barrier");
        }
    }

    std::printf(failures ? "== FAIL ==\n" : "== PASS ==\n");
    return failures ? 1 : 0;
}
