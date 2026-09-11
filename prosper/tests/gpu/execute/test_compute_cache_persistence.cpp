// #3425: the compute driver pipeline cache LOADED on launch and was never SAVED, because the only
// writer was ~VulkanComputeContext and prosper-app leaves through std::_Exit. Every launch then
// recompiled every pipeline from scratch forever (#3450 measured 821 ms of first-use compilation in
// one 5 s GTA V world window).
//
// Each mode below runs one real compute dispatch and then leaves through _Exit itself, so a
// destructor-only persistence path cannot make this test pass. The driver script chains the modes
// across SEPARATE PROCESSES, which is the only way to observe that bytes actually survived.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;

namespace {
constexpr uint32_t kBytes = 4096;
constexpr uint32_t kTail = 0x5a5a5a5au;
constexpr uint32_t kFill = 0x34250000u;
constexpr uint64_t kCodeAddress = 0x3425b00200ull;

// One invocation stores the four user-SGPR words through binding 0, which is enough to require a
// real VkPipeline and therefore a real driver compilation.
constexpr uint32_t kShader[] = {
    0x7e080280u,                                          // v4 = 0
    0x7e000204u, 0x7e020205u, 0x7e040206u, 0x7e060207u,   // v0..v3 = s4..s7
    0xe01c2000u, 0x80000004u,                             // buffer_store_dwordx4 v[0:3], v4, s[0:3]
    0xbf810000u,                                          // endpgm
};

int fail(const char* message) {
    std::fprintf(stderr, "[compute-cache-fixture] FAIL: %s\n", message);
    return 1;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: fixture save|load|cold|disabled\n");
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "save" && mode != "resave" && mode != "load" && mode != "cold" &&
        mode != "disabled")
        return 2;

    std::vector<uint32_t> guest(kBytes / sizeof(uint32_t), kTail);

    ShaderResourceTable resources;
    ShaderResource resource{};
    resource.cls = ResourceClass::ConstantBuffer;
    resource.format = DataFormat::Uint32;
    resource.num_components = 4;
    resource.binding = 0;
    resource.sgpr_base = 0;
    resource.stride = 16;
    resource.gpu_addr = reinterpret_cast<uint64_t>(guest.data());
    resource.size = kBytes;
    resources.resources.push_back(resource);

    ComputeShaderConfig config;
    config.local_x = 1;
    config.user_sgprs.resize(8);
    for (unsigned lane = 0; lane < 4; ++lane) config.user_sgprs[4 + lane] = kFill + lane;

    ComputeItem item;
    item.spirv = recompile_compute(kShader, std::size(kShader), &resources, config);
    if (item.spirv.empty()) return fail("the fixture kernel must recompile");
    item.user_sgprs = config.user_sgprs;
    item.code_addr = kCodeAddress;
    item.resources = std::make_shared<ShaderResourceTable>(resources);
    item.launch.threads_x = item.launch.threads_y = item.launch.threads_z = 1;
    item.launch.local_x = item.launch.local_y = item.launch.local_z = 1;
    item.launch.groups_x = item.launch.groups_y = item.launch.groups_z = 1;
    item.submit_no = 3425;

    if (!prosper::frontend::execute_live_compute_items({item}))
        return fail("the fixture dispatch must execute");
    for (unsigned lane = 0; lane < 4; ++lane)
        if (guest[lane] != kFill + lane)
            return fail("the fixture dispatch must really have run on the GPU");

    const auto status = prosper::frontend::live_compute_pipeline_cache_status();
    if (!status.context_live) return fail("a compute context must exist after a dispatch");

    if (mode == "disabled") {
        // The opt-out must not silently look like a successful save. A flush that returns true here
        // would mean the caller cannot distinguish "persistence off" from "persisted".
        if (status.persistence_configured) return fail("persistence must be off in this mode");
        if (prosper::frontend::flush_live_compute_pipeline_cache())
            return fail("a disabled cache must decline the flush rather than report a save");
        std::fflush(nullptr);
        std::_Exit(0);
    }

    if (!status.persistence_configured)
        return fail("an explicit cache path must configure persistence");

    if (mode == "save" || mode == "resave") {
        // "resave" is the concurrent arm: a peer process may already have written the file, so a
        // warm start is legal there. Only the serial first launch is required to be cold.
        if (mode == "save" && status.loaded_bytes)
            return fail("the save run must start from a cold cache");
        if (!prosper::frontend::flush_live_compute_pipeline_cache())
            return fail("the explicit shutdown snapshot must write the cache");
        // Idempotent: prosper-app may reach this path more than once across future teardown work,
        // and the destructor still calls it on the non-_Exit route.
        if (!prosper::frontend::flush_live_compute_pipeline_cache())
            return fail("a second snapshot must also succeed");
    } else if (mode == "load") {
        if (!status.loaded_bytes)
            return fail("the driver must have accepted bytes written by the previous process");
    } else if (mode == "cold") {
        if (status.loaded_bytes)
            return fail("a rejected cache file must leave the run cold rather than feed the driver");
    }

    std::fprintf(stderr, "[compute-cache-fixture] success mode=%s loaded=%llu\n", mode.c_str(),
                 static_cast<unsigned long long>(status.loaded_bytes));
    std::fflush(nullptr);
    // _Exit, exactly like prosper-app: destructor-only persistence must not be able to pass.
    std::_Exit(0);
}
