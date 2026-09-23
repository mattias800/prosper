#include "shared/diagnostics/kena_menu_trace_policy.hpp"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string_view>

int main() {
    using prosper::frontend::parse_kena_menu_trace;
    using prosper::frontend::kena_menu_trace_callback;
    using prosper::frontend::kena_menu_gpu_source;
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
    };

    check(!parse_kena_menu_trace(nullptr).armed, "unset is disabled");
    check(parse_kena_menu_trace("ms:21000").armed &&
          parse_kena_menu_trace("ms:21000").after_ms == 21000,
          "a bounded menu window activates");
    check(parse_kena_menu_trace("ms:120000").armed, "latest permitted deadline activates");
    const auto spec = parse_kena_menu_trace("ms:21000");
    check(!kena_menu_trace_callback(spec, 20999, false, true), "early callback refuses");
    check(!kena_menu_trace_callback(spec, 21000, false, false), "intermediate span refuses");
    check(kena_menu_trace_callback(spec, 21000, false, true), "first final menu callback activates");
    check(!kena_menu_trace_callback(spec, 21001, true, true), "later callbacks refuse");
    for (const char* invalid : {"", "21000", "ms:", "ms:-1", "ms:1x", "ms:120001",
                                "ms:18446744073709551616"})
        check(!parse_kena_menu_trace(invalid).armed, "malformed or late deadline refuses");

    const auto gpu_new = kena_menu_gpu_source(true, true, 0x1000, 91, 17);
    check(std::string_view(gpu_new.kind) == "gpu-front" && gpu_new.address == 0x1000 &&
          gpu_new.flip == 91 && gpu_new.publication_id == 17,
          "successful GPU blit carries front address, exact flip, and publication id");
    const auto gpu_reused = kena_menu_gpu_source(true, false, 0x1000, 91, 17);
    check(std::string_view(gpu_reused.kind) == "gpu-front-same-flip" &&
          gpu_reused.flip == 91 && gpu_reused.publication_id == 17,
          "same-flip GPU suppression names its prior publication");
    const auto gpu_refused = kena_menu_gpu_source(false, true, 0x1000, 91, 17);
    check(std::string_view(gpu_refused.kind) == "none" && gpu_refused.publication_id == 0,
          "failed GPU blit cannot claim a visible source");
    const auto gpu_unknown = kena_menu_gpu_source(true, true, 0x1000, 91, 0);
    check(std::string_view(gpu_unknown.kind) == "gpu-source-unknown" &&
          gpu_unknown.address == 0 && gpu_unknown.flip == 91,
          "a GPU publication without identity cannot reuse a stale CPU candidate");
    check(std::string_view(kena_menu_gpu_source(true, true, 0, 91, 17).kind) ==
              "gpu-source-unknown",
          "a GPU publication without a front address is explicit unknown");

    std::printf("kena_menu_trace_policy: %d failure(s)\n", failures);
    return failures ? 1 : 0;
}
