#pragma once
#include <cstdint>
#include <span>
#include <string>

namespace prosper::diagnostics {
void scan_host_stack_candidates(std::span<const uint64_t> stack_words,
                                char (&host_returns)[288],
                                bool (*guest_trace_page_executable)(uintptr_t),
                                std::string (*describe_code_address)(uint64_t));
} // namespace prosper::diagnostics
