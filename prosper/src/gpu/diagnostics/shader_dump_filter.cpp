#include "gpu/diagnostics/shader_dump_filter.hpp"

#include "gpu/diagnostics/diag_ratelimit.hpp"
#include "gpu/diagnostics/watch_list.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace prosper::gpu {

ShaderDumpProgramFilter::ConfigureResult ShaderDumpProgramFilter::configure(const char* spec) {
    std::lock_guard lock(mutex_);
    addresses_.clear();
    withheld_ = 0;
    if (!spec || !*spec) return ConfigureResult::Unset;
    std::vector<uint64_t> parsed;
    if (!parse_hex_watch_list(spec, parsed)) return ConfigureResult::Malformed;
    std::sort(parsed.begin(), parsed.end());
    parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());
    addresses_ = std::move(parsed);
    return ConfigureResult::Armed;
}

bool ShaderDumpProgramFilter::armed() const {
    std::lock_guard lock(mutex_);
    return !addresses_.empty();
}

std::size_t ShaderDumpProgramFilter::size() const {
    std::lock_guard lock(mutex_);
    return addresses_.size();
}

bool ShaderDumpProgramFilter::allows(uint64_t program_address, uint64_t chain_address) const {
    std::lock_guard lock(mutex_);
    if (addresses_.empty()) return true;
    const auto listed = [this](uint64_t address) {
        return address != 0 &&
               std::binary_search(addresses_.begin(), addresses_.end(), address);
    };
    return listed(program_address) || listed(chain_address);
}

ShaderDumpProgramFilter::Withheld ShaderDumpProgramFilter::note_withheld() {
    std::lock_guard lock(mutex_);
    const uint64_t ordinal = ++withheld_;
    return {ordinal, diag_should_print(ordinal, 1)};
}

uint64_t ShaderDumpProgramFilter::withheld_total() const {
    std::lock_guard lock(mutex_);
    return withheld_;
}

ShaderDumpProgramFilter& shader_dump_program_filter() {
    static ShaderDumpProgramFilter filter;
    static std::mutex sync_mutex;
    // Sentinel distinct from every possible environment value, so the first call always configures
    // — including when the variable is unset, which must not be confused with "already synced".
    static std::string current = std::string("\x01unset-sentinel");
    static bool have_current = false;

    const char* spec = std::getenv("PROSPER_SHADER_DUMP_PROGRAM");
    const std::string observed = spec ? std::string(spec) : std::string();
    std::lock_guard lock(sync_mutex);
    if (have_current && observed == current) return filter;
    current = observed;
    have_current = true;

    switch (filter.configure(spec)) {
    case ShaderDumpProgramFilter::ConfigureResult::Unset:
        break;
    case ShaderDumpProgramFilter::ConfigureResult::Armed:
        std::fprintf(stderr,
                     "[shader-dump] PROSPER_SHADER_DUMP_PROGRAM=%s -> armed on %zu program "
                     "address(es); ONLY shaders at those addresses are dumped\n",
                     observed.c_str(), filter.size());
        break;
    case ShaderDumpProgramFilter::ConfigureResult::Malformed:
        // Loud, and it says which way it failed open. See the header: an empty dump directory would
        // read as "that program never compiled".
        std::fprintf(stderr,
                     "[shader-dump] PROSPER_SHADER_DUMP_PROGRAM='%s' is MALFORMED (expects "
                     "0xADDR[,0xADDR...]: hex only, explicit 0x, no zero, no trailing comma) -> "
                     "NOT armed; EVERY successful shader will be dumped\n",
                     observed.c_str());
        break;
    }
    return filter;
}

}  // namespace prosper::gpu
