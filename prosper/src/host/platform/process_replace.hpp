#pragma once

#include <string>
#include <vector>

namespace prosper::host {
// Native errno result. Preflight leaves the running process unchanged. Replacement never returns
// on success; a returned errno means the OS refused the exec. Callers must decide whether their
// old process can still resume (a frontend that already drained its guest cannot).
int process_replace_preflight(const std::string& executable);
// Linux separates this thread's descriptor table from still-running old workers, then marks
// nonstdio entries CLOEXEC. Returns ENOTSUP on hosts with a different native replacement API.
int process_replace_isolate_descriptors();
int replace_process(const std::vector<std::string>& argv);
} // namespace prosper::host
