// #3407: optional result baselines must not churn a primary buffer working set that fits.
#include "gpu/capture/gpu_capture.hpp"
#include "gpu/execute/gpu_execute.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"
#include "shared/live/live_compute.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace prosper::gpu;
namespace {
constexpr uint32_t bytes = 2u << 20;
constexpr uint64_t code_address = 0x3407b00200ull;
constexpr uint32_t initial_fill = 0xb1b1b100u;
constexpr uint32_t changed_fill = 0xc2c2c200u;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
void env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
#else
    if (value) setenv(name, value, 1); else unsetenv(name);
#endif
}
std::string hex(uint64_t value) {
    char text[32];
    std::snprintf(text, sizeof(text), "0x%llx", static_cast<unsigned long long>(value));
    return text;
}
struct GuestBuffer {
    std::vector<uint32_t> data;
    uint32_t expected = initial_fill;
    explicit GuestBuffer(uint32_t tail, uint32_t length = bytes)
        : data(length / sizeof(uint32_t), tail) {}
    uint64_t address() const { return reinterpret_cast<uint64_t>(data.data()); }
    uint32_t size() const { return static_cast<uint32_t>(data.size() * sizeof(uint32_t)); }
    bool correct(uint32_t tail) const {
        for (unsigned i = 0; i < 4; ++i)
            if (data[i] != expected + i) return false;
        return std::all_of(data.begin() + 4, data.end(),
                           [&](uint32_t word) { return word == tail; });
    }
};
} // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "usage: fixture MODE CACHE_MIB [LARGE_BYTES]\n");
        return 2;
    }
    const std::string mode = argv[1];
    const bool large = mode == "requirements-large" || mode == "partial";
    if (mode != "requirements" && mode != "cycle" && mode != "pinned" && !large) return 2;
    if (large != (argc == 4)) return 2;
    char* end = nullptr;
    const auto cap_mib = std::strtoull(argv[2], &end, 10);
    if (!end || *end || cap_mib == 0 || cap_mib > 1024) return 2;
    uint32_t large_bytes = bytes;
    if (large) {
        const auto length = std::strtoull(argv[3], &end, 10);
        if (!end || *end || length < bytes || length > UINT32_MAX || (length & 15)) return 2;
        large_bytes = static_cast<uint32_t>(length);
    }
    env("PROSPER_COMPUTELOG", nullptr);
    env("PROSPER_COMPUTELOG_CODE", nullptr);
    env("PROSPER_NO_PERSISTENT_COMPUTE_BUFFERS", nullptr);
    env("PROSPER_NO_PERSISTENT_COMPUTE_BUFFER_RESULTS", nullptr);
    env("PROSPER_COMPUTE_BUFFER_CACHE_MB", argv[2]);
    env("PROSPER_COMPUTE_BUFFER_RESULT_MIN_MB", "1");
    env("PROSPER_COMPUTE_BUFFER_TIMING", "1");
    env("PROSPER_COMPUTE_TIMING_CAPTURE_ONLY", nullptr);
    env("PROSPER_NO_DISK_PIPELINE_CACHE", "1");
    // Guest pointers into ordinary host vectors keep this test independent of watch promotion.
    // No direct mapping is registered and no signal handler or watch policy is installed here.
    std::array<GuestBuffer, 3> buffers = {
        GuestBuffer(0xababababu), GuestBuffer(0xcdcdcdcdu), GuestBuffer(0xefefefefu, large_bytes)};
    constexpr uint32_t tails[] = {0xababababu, 0xcdcdcdcdu, 0xefefefefu};
    static constexpr uint32_t shader[] = {
        0x7e080280u, // v4 = 0: one invocation writes the first uvec4 through each descriptor.
        0x7e000204u, 0x7e020205u, 0x7e040206u, 0x7e060207u,
        0xe01c2000u, 0x80000004u,
        0x7e0a020cu, 0x7e0c020du, 0x7e0e020eu, 0x7e10020fu,
        0xe01c2000u, 0x80020504u,
        0x7e120214u, 0x7e140215u, 0x7e160216u, 0x7e180217u,
        0xe01c2000u, 0x80040904u,
        0xbf810000u,
    };
    ShaderResourceTable resources;
    for (unsigned binding = 0; binding < 3; ++binding) {
        ShaderResource resource{};
        resource.cls = ResourceClass::ConstantBuffer;
        resource.format = DataFormat::Uint32;
        resource.num_components = 4;
        resource.binding = binding;
        resource.sgpr_base = binding * 8;
        resource.stride = 16;
        resource.gpu_addr = buffers[0].address();
        resource.size = bytes;
        resources.resources.push_back(resource);
    }
    ComputeShaderConfig config;
    config.local_x = 1;
    config.user_sgprs.resize(24);
    for (unsigned binding = 0; binding < 3; ++binding)
        for (unsigned lane = 0; lane < 4; ++lane)
            config.user_sgprs[binding * 8 + 4 + lane] = initial_fill + lane;
    ComputeItem prototype;
    prototype.spirv = recompile_compute(shader, std::size(shader), &resources, config);
    if (prototype.spirv.empty()) return 3;
    prototype.user_sgprs = config.user_sgprs;
    prototype.code_addr = code_address;
    prototype.launch.threads_x = prototype.launch.threads_y = prototype.launch.threads_z = 1;
    prototype.launch.local_x = prototype.launch.local_y = prototype.launch.local_z = 1;
    prototype.launch.groups_x = prototype.launch.groups_y = prototype.launch.groups_z = 1;
    const auto hash = gpu_capture_hash(reinterpret_cast<const uint8_t*>(prototype.spirv.data()),
                                       prototype.spirv.size() * sizeof(uint32_t));
    env("PROSPER_COMPUTE_TIMING_CODE", hex(code_address).c_str());
    env("PROSPER_COMPUTE_TIMING_HASH", hex(hash).c_str());
    std::fprintf(stderr, "[buffer-residency-fixture] mode=%s code=%s hash=%s cap-mib=%llu\n",
                 mode.c_str(), hex(code_address).c_str(), hex(hash).c_str(), cap_mib);
    unsigned notifications = 0;
    set_guest_gpu_write_observer([&](uint64_t address, uint64_t length, const char*) {
        for (const auto& buffer : buffers)
            if (address == buffer.address() && length == buffer.size()) ++notifications;
    });
    auto run = [&](unsigned index, std::array<unsigned, 3> owners, uint32_t value) {
        ComputeItem item = prototype;
        auto bound = std::make_shared<ShaderResourceTable>(resources);
        std::array<bool, 3> touched{};
        for (unsigned binding = 0; binding < 3; ++binding) {
            auto& buffer = buffers[owners[binding]];
            bound->resources[binding].gpu_addr = buffer.address();
            bound->resources[binding].size = buffer.size();
            buffer.expected = value;
            touched[owners[binding]] = true;
            for (unsigned lane = 0; lane < 4; ++lane)
                item.user_sgprs[binding * 8 + 4 + lane] = value + lane;
        }
        item.resources = bound;
        item.submit_no = 4400 + index;
        item.dispatch_index = index;
        item.command_order = index * 10;
        const unsigned before = notifications;
        check(prosper::frontend::execute_live_compute_items({item}), "residency dispatch executes");
        for (unsigned owner = 0; owner < 3; ++owner)
            if (touched[owner])
                check(buffers[owner].correct(tails[owner]),
                      "shader output and full untouched tail survive cache pressure");
        check(notifications - before == std::count(touched.begin(), touched.end(), true),
              "publication is once per unique owner, including exact aliases");
    };
    auto single = [&](unsigned index, unsigned owner, uint32_t value = initial_fill) {
        run(index, {owner, owner, owner}, value);
    };
    single(1, mode == "requirements-large" ? 2 : 0);
    if (mode == "partial") {
        single(2, 1);
        // A is pinned by the first two bindings. B remains idle and reclaimable, but its
        // entire primary+baseline is insufficient to admit C. Refusal must leave B intact.
        run(3, {0, 0, 2}, initial_fill);
        single(4, 1);
    } else if (mode != "requirements" && mode != "requirements-large") {
        single(2, 1);
        unsigned index = 3;
        if (mode == "pinned") run(index++, {0, 1, 2}, initial_fill);
        single(index++, 2);
        single(index++, 0);
        single(index++, 1);
        single(index++, 2);
        single(index++, 0, changed_fill);
        single(index++, 0, changed_fill);
        buffers[1].data[0] ^= 0xffffffffu;
        single(index++, 1);
    }
    set_guest_gpu_write_observer({});
    if (!failures)
        std::fprintf(stderr, "[buffer-residency-fixture] success mode=%s\n", mode.c_str());
    return failures ? 1 : 0;
}
