#include "gpu/recompiler/compiler_resource_access.hpp"
#include <stdexcept>

namespace prosper::gpu {
namespace { thread_local CompilerResourceScope* active = nullptr; }
CompilerResourceScope::CompilerResourceScope(std::vector<CompilerResourceAccess> access)
    : previous_(active), access_(std::move(access)) { active = this; }
CompilerResourceScope::~CompilerResourceScope() { active = previous_; }
bool compiler_resource_scope_active() { return active != nullptr; }
void compiler_resource_forbid_guest_read() {
    if (active) throw std::runtime_error("INCOMPLETE: unrecorded compiler guest-memory read");
}
bool compiler_resource_has_host_data(const ShaderResource& resource) {
    if (!active) return resource.host_data != nullptr;
    for (const auto& input : active->access_) if (input.resource == &resource) return input.present;
    throw std::runtime_error("INCOMPLETE: unknown compiler resource presence");
}
const uint8_t* compiler_resource_data(const ShaderResource& resource, uint64_t minimum) {
    if (!active) return resource.host_data && minimum <= resource.host_data_size ? resource.host_data : nullptr;
    for (const auto& input : active->access_) if (input.resource == &resource) {
        if (!input.present) return nullptr;
        if (!input.owned || minimum > input.bytes.size())
            throw std::runtime_error("INCOMPLETE: opaque or out-of-bounds compiler byte read");
        return input.bytes.data();
    }
    throw std::runtime_error("INCOMPLETE: unknown compiler resource byte read");
}
} // namespace prosper::gpu
