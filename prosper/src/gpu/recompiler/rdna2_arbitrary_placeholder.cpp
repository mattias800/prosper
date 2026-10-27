#include "gpu/recompiler/rdna2_arbitrary_placeholder.hpp"
#include "gpu/recompiler/rdna2_diagnostic_attempt.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"
#include "gpu/recompiler/rdna2_arbitrary_placeholder_observation.hpp"
#include <algorithm>
#include <utility>

namespace prosper::gpu {
namespace {
struct Attempt {
    bool allow = false, eligible = false;
};
thread_local Attempt* current = nullptr;
struct Scope {
    Attempt state;
    Attempt* previous = std::exchange(current, &state);
    ~Scope() { current = previous; }
};
} // namespace

bool placeholder_roots_are_unloaded(const PlaceholderRoots& roots) {
    return !roots.empty() && std::all_of(roots.begin(), roots.end(), [](const auto& root) {
        return root.origin == PlaceholderOrigin::Unloaded;
    });
}

bool arbitrary_placeholder_attempt_active() {
    return current != nullptr;
}

std::vector<uint32_t>
recompile_with_arbitrary_placeholders(const RecompileDiagnosticContext& diagnostic,
                                      const std::function<std::vector<uint32_t>()>& compile) {
    if (current) return compile();
    observe_arbitrary_placeholder_check();
    Scope scope;
    RecompileDiagnosticAttempt original;
    auto module = compile();
    if (!module.empty() || !scope.state.eligible) {
        original.commit();
        return module;
    }
    scope.state.allow = true;
    RecompileDiagnosticAttempt retry;
    module = compile();
    if (module.empty()) {
        retry.discard();
        original.commit();
        return module;
    }
    original.replace_with(retry);
    const auto reads = arbitrary_placeholder_reads(module);
    observe_arbitrary_placeholder_admission(diagnostic.program_address, reads.fresh_mask,
                                            reads.loop_carry);
    return module;
}

bool admit_arbitrary_placeholder_read(SpirvCompute& b, uint32_t pc, ArbitraryPlaceholderRead kind,
                                      const PlaceholderRoots& roots) {
    // Owned packets retain their independent genuine-entry/runtime-validity contract.
    if (!current || b.is_fragment_packet() || !placeholder_roots_are_unloaded(roots)) return false;
    current->eligible = true;
    if (!current->allow) return false;
    std::string marker = kind == ArbitraryPlaceholderRead::FreshMask
                             ? "Prosper.ArbitraryPlaceholderRead=fresh-mask:"
                             : "Prosper.ArbitraryPlaceholderRead=loop-carry:";
    marker += std::to_string(pc);
    std::vector<uint32_t> words;
    b.pstr(words, marker.c_str());
    SpirvCompute::putv(b.debug, Op_ModuleProcessed, words);
    return true;
}
} // namespace prosper::gpu
