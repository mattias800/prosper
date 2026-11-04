#include "gpu/recompiler/rdna2_native_fragment_collectives.hpp"
#include "gpu/recompiler/rdna2_diagnostic_attempt.hpp"
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>
namespace prosper::gpu {
namespace {
struct Attempt {
    bool requested = false, repair = false;
};
thread_local Attempt* current = nullptr;
struct Scope {
    Attempt state;
    Attempt* previous = std::exchange(current, &state);
    ~Scope() { current = previous; }
};
}   // namespace
bool native_fragment_collective_retry_active() {
    return current && current->repair;
}
void request_native_fragment_collective_retry() {
    if (current) current->requested = true;
}
std::vector<uint32_t>
recompile_native_fragment_collectives(const std::function<std::vector<uint32_t>()>& compile) {
    if (current) return compile();
    Scope scope;
    RecompileDiagnosticAttempt original;
    auto module = compile();
    if (!module.empty() || !scope.state.requested) {
        original.commit();
        return module;
    }
    scope.state.repair = true;
    RecompileDiagnosticAttempt retry;
    module = compile();
    original.replace_with(retry);
    return module;
}
} // namespace prosper::gpu
