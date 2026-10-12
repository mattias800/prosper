#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"

#include <string>
#include <cstdint>
#include <optional>
#include <vector>

namespace prosper::gpu {
struct DeferredRecompileDiagnostic {
    RecompileDiagnosticContext context;
    std::string tag;
    std::optional<std::string> role;
    std::string payload;
    bool terminal_only = false;
};

// Compiler retries must not publish a discarded attempt as a final refusal. Keep both the
// address-indexed reason and capture/log effects private until the caller selects an attempt.
// Nested commits join the enclosing transaction; storage is local to the compiling thread.
// Complete nested attempts in LIFO order. A TerminalRejectCapture used to observe the selected
// attempt must surround its commit; a capture opened and taken inside an uncommitted attempt
// observes no records yet. Discard suppresses new records, never clears older address history.
class RecompileDiagnosticAttempt {
public:
    RecompileDiagnosticAttempt();
    ~RecompileDiagnosticAttempt();
    RecompileDiagnosticAttempt(const RecompileDiagnosticAttempt&) = delete;
    RecompileDiagnosticAttempt& operator=(const RecompileDiagnosticAttempt&) = delete;
    void commit();
    void discard();

private:
    std::vector<DeferredRecompileDiagnostic> records_;
    std::vector<DeferredRecompileDiagnostic>* previous_ = nullptr;
    bool active_ = true;
};

bool defer_recompile_diagnostic(const RecompileDiagnosticContext&, const char* tag,
                                const char* role, const char* payload);
bool defer_terminal_reject_reason(uint64_t program_address, const char* tag, const char* payload);
} // namespace prosper::gpu
