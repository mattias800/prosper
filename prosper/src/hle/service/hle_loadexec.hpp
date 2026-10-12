#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace prosper {
struct LoadExecRequest {
    std::string dump_root;
    std::string executable;
    std::string host_executable;
    std::vector<std::string> argv;
};

// A successful backend replaces the process and never returns. A returned value is an error;
// returning zero is treated as failure, so a frontend cannot resume the abandoned old program.
using LoadExecBackend = std::function<uint64_t(const LoadExecRequest&)>;
void set_loadexec_backend(LoadExecBackend backend);
void configure_loadexec_app0(const std::string& dump_root);
void register_loadexec_hle();
} // namespace prosper
