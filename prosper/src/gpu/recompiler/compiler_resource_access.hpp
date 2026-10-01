#pragma once
#include "gpu/resources/shader_resources.hpp"
#include <span>
#include <vector>

namespace prosper::gpu {
struct CompilerResourceAccess {
    const ShaderResource* resource = nullptr;
    bool present = false, owned = false;
    std::span<const uint8_t> bytes;
};
// Opaque backing is a presence fact, NEVER a synthetic pointer/readable range. The compiler may
// query metadata independently, but every byte request must find an owned checked span.
class CompilerResourceScope {
public:
    explicit CompilerResourceScope(std::vector<CompilerResourceAccess> access);
    ~CompilerResourceScope();
    CompilerResourceScope(const CompilerResourceScope&) = delete;
    CompilerResourceScope& operator=(const CompilerResourceScope&) = delete;
private:
    friend bool compiler_resource_has_host_data(const ShaderResource&);
    friend const uint8_t* compiler_resource_data(const ShaderResource&, uint64_t);
    CompilerResourceScope* previous_ = nullptr;
    std::vector<CompilerResourceAccess> access_;
};
bool compiler_resource_has_host_data(const ShaderResource&);
const uint8_t* compiler_resource_data(const ShaderResource&, uint64_t minimum_bytes);
bool compiler_resource_scope_active();
// Call before any compiler fallback that could read a logical guest address or revalidate live
// memory. Offline SOURCE replay does not own those reads and must refuse rather than try them.
void compiler_resource_forbid_guest_read();
} // namespace prosper::gpu
