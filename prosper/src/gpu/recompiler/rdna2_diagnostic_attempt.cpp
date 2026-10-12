#include "gpu/recompiler/rdna2_diagnostic_attempt.hpp"
#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/recompiler/rdna2_to_spirv_internal.hpp"

#include <utility>
#include <optional>
#include <vector>
#include <cstdint>
#include <cassert>

namespace prosper::gpu {
namespace {
thread_local std::vector<DeferredRecompileDiagnostic>* pending = nullptr;
}

RecompileDiagnosticAttempt::RecompileDiagnosticAttempt()
    : previous_(std::exchange(pending, &records_)) {}

RecompileDiagnosticAttempt::~RecompileDiagnosticAttempt() {
    discard();
}

void RecompileDiagnosticAttempt::discard() {
    if (!active_) return;
    assert(pending == &records_);
    pending = previous_;
    active_ = false;
    records_.clear();
}

void RecompileDiagnosticAttempt::commit() {
    if (!active_) return;
    assert(pending == &records_);
    pending = previous_;
    active_ = false;
    for (const auto& record : records_)
        if (record.terminal_only)
            record_terminal_reject_reason(record.context.program_address, record.tag.c_str(),
                                          record.payload.c_str());
        else
            log_recompile_diagnostic(record.context, record.tag.c_str(),
                                     record.role ? record.role->c_str() : nullptr, "%s",
                                     record.payload.c_str());
    records_.clear();
}

bool defer_recompile_diagnostic(const RecompileDiagnosticContext& context, const char* tag,
                                const char* role, const char* payload) {
    if (!pending) return false;
    pending->push_back({context, tag ? tag : "",
                        role ? std::optional<std::string>(role) : std::nullopt,
                        payload ? payload : ""});
    return true;
}

bool defer_terminal_reject_reason(uint64_t address, const char* tag, const char* payload) {
    if (!pending) return false;
    pending->push_back({{RecompileDiagnosticStage::Standalone, address},
                        tag ? tag : "",
                        std::nullopt,
                        payload ? payload : "",
                        true});
    return true;
}
} // namespace prosper::gpu
