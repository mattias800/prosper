// The dispatcher keeps direct entry descriptors until a real scalar write retires them.
#include "gpu/recompiler/rdna2_cfg_descriptor_sregs.hpp"
#include "gpu/resources/shader_resources.hpp"
#include <cstdint>
#include <set>
namespace prosper::gpu {
std::set<int> cfg_direct_descriptor_sregs(const ShaderResourceTable* rt) {
    // Direct user-data descriptors are deliberately absent from the initial scalar SSA map: their
    // identity lives in the resource table, and presence in sreg means shader code overwrote them.
    // The dispatcher persists every referenced SGPR in a Function variable, so recover that
    // distinction per block until the first static write; otherwise a direct V# such as UE4's s[8:11]
    // looks rewritten merely because the dispatcher loaded its zero-initialized backing variable.
    std::set<int> direct_descriptor_sregs;
    if (rt) {
        for (const auto& resource : rt->resources) {
            if (resource.srt_offset != 0xFFFFFFFFu || resource.sgpr_base == 0xFFFFFFFFu) continue;
            const uint32_t words = (resource.cls == ResourceClass::Texture ||
                                    resource.cls == ResourceClass::StorageImage) ? 8u : 4u;
            for (uint32_t word = 0; word < words; ++word)
                direct_descriptor_sregs.insert(static_cast<int>(resource.sgpr_base + word));
        }
    }

    return direct_descriptor_sregs;
}
} // namespace prosper::gpu
