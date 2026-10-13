// The process's guest-backed crt0 arguments exist before module initializers and remain valid for
// the process lifetime. The primary entry reuses this same stack/vector, rather than rebuilding it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace prosper {

struct ProcessEntryContext {
    void* stack = nullptr;
    size_t stack_bytes = 0;
    uint64_t stack_pointer = 0;
    uint32_t argc = 0;
    uint64_t argv = 0;
};

// Prepare once, before any initializer/primary guest entry. Repeated calls retain the initial
// process data, as the real kernel getters retain their startup cells. No host argv pointers escape.
bool prepare_process_entry(const std::vector<std::string>& arguments,
                           const char* configured_environment, std::string* error = nullptr);

// Uses the ordinary configured or exact LoadExec argument source, and the existing guest env
// spelling. Boot-only; guest execution must be skipped on failure.
bool prepare_configured_process_entry(std::string* error = nullptr);

// Empty before preparation; immutable after publication. The backing mapping is never reclaimed
// while guest workers can retain pointers. Native process replacement reclaims it with the image.
ProcessEntryContext process_entry_context();

}   // namespace prosper
