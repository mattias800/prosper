#include "diagnostics/native_host_wait.hpp"
#include <cstdio>

namespace prosper::diagnostics {
void scan_host_stack_candidates(std::span<const uint64_t> stack_words,
                                char (&host_returns)[288],
                                bool (*guest_trace_page_executable)(uintptr_t),
                                std::string (*describe_code_address)(uint64_t)) {
    size_t host_returns_used = 0;
    unsigned host_return_count = 0;
    for (size_t i = 0; i < stack_words.size() && host_return_count < 6; ++i) {
        const uint64_t candidate = stack_words[i];
        if (candidate < 0x10000) continue;
        if (!guest_trace_page_executable((uintptr_t)candidate)) continue;
        const std::string described = describe_code_address(candidate);
        if (described.rfind("prosper+", 0) != 0) continue;   // ours only; DLLs are noise here
        bool duplicate = false;
        for (size_t j = 0; j < i; ++j) duplicate |= stack_words[j] == candidate;
        if (duplicate) continue;
        const int appended = std::snprintf(
            host_returns + host_returns_used, sizeof(host_returns) - host_returns_used,
            host_return_count ? ",%s" : "%s", described.c_str());
        if (appended <= 0 || (size_t)appended >= sizeof(host_returns) - host_returns_used) break;
        host_returns_used += (size_t)appended;
        ++host_return_count;
    }
}
} // namespace prosper::diagnostics
