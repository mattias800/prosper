#pragma once

#include <string>
#include <vector>

namespace prosper {

// Explicit process-entry selection, distinct from auto-discovery of dependent PRXs. Only guest
// executables contained in this app0 may become the main image; the dependent-module policy is
// unchanged. A failure returns an empty path and never forwards a guest path to the host.
std::string resolve_guest_executable(const std::string& dump_root, const std::string& guest_path,
                                     std::string* error);

// Exact argv supplied by a process replacement. Ordinary boots retain PROSPER_GUEST_ARGS.
void set_guest_entry_arguments(const std::vector<std::string>& arguments);
std::vector<std::string> guest_entry_arguments(const char* configured_arguments = nullptr);

} // namespace prosper
