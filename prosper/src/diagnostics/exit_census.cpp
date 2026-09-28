#include "diagnostics/exit_census.hpp"

#include "diagnostics/exit_reports.hpp"

#include <cstdlib>
#include <utility>

namespace prosper::diagnostics {

void register_census(const char* disable_env, std::function<bool()> report) {
    register_exit_report([disable_env, report = std::move(report)] {
        // Read the environment at REPORT time, not registration time. A census may register
        // during static initialisation, before anything has established what the run is.
        if (disable_env && std::getenv(disable_env)) return;
        (void)report();
    });
}

}  // namespace prosper::diagnostics
