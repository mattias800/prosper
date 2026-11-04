#include "fixtures/render_reconvergence_shader.hpp"
#include "gpu/recompiler/spirv/fragment_reconvergence.hpp"
#include <array>
#include <cstddef>
#include <cstdio>

namespace prosper::test {
bool render_reconvergence_supported(const std::vector<uint32_t>& module, bool device_enabled,
                                    uint64_t program, uint64_t identity) {
    const auto requirement = gpu::fragment_spirv_reconvergence(module);
    if (requirement == gpu::FragmentReconvergence::None ||
        (requirement == gpu::FragmentReconvergence::Required && device_enabled))
        return true;
    struct Refusal {
        uint64_t program = 0, identity = 0;
        gpu::FragmentReconvergence reason = gpu::FragmentReconvergence::None;
    };
    // Bounded diagnostic deduplication; no process-wide lock or shader walk on a cache hit.
    static thread_local std::array<Refusal, 64> reported{};
    static thread_local size_t next = 0;
    for (const auto& item : reported)
        if (item.program == program && item.identity == identity && item.reason == requirement)
            return false;
    reported[next++ % reported.size()] = {program, identity, requirement};
    std::fprintf(
        stderr,
        "[fragment-reconvergence-refused] program=0x%llx identity=%llu reason=%s "
        "shaderMaximalReconvergence=%d\n",
        static_cast<unsigned long long>(program), static_cast<unsigned long long>(identity),
        requirement == gpu::FragmentReconvergence::Malformed ? "malformed-module"
                                                             : "device-feature-not-enabled",
        static_cast<int>(device_enabled));
    return false;
}
} // namespace prosper::test
