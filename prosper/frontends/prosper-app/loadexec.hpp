#pragma once
#include "hle/service/hle_loadexec.hpp"
#include <string>
#include <vector>

namespace prosper::frontend {
// Install the process-replacement backend before the guest starts. The caller runs replacement
// only after its ordinary guest/GPU shutdown; the abandoned LoadExec caller parks until then.
void install_loadexec_frontend(int argc, char** argv, const std::string& host_executable);
LoadExecBackend make_loadexec_frontend_backend(int argc, char** argv,
                                               const std::string& host_executable);
bool parse_loadexec_option(int& i, int argc, char** argv, std::string* error);
std::string selected_guest_executable();
std::vector<std::string> loadexec_host_arguments(int argc, char** argv);
void finish_loadexec_replacement();
} // namespace prosper::frontend
