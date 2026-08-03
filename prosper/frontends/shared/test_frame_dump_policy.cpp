#include "frame_dump_policy.hpp"

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>

using prosper::frontend::kFrameDumpsByDefault;
using prosper::frontend::frame_dump_request_allowed;
using prosper::frontend::register_live_renderer_from_environment;

namespace {

struct Registration {
    unsigned calls = 0;
    std::string frame_dir;
    bool dump_bmps = true;
};

bool check(const char* name,
           std::initializer_list<std::pair<const char*, const char*>> variables,
           const char* expected_dir, bool expected_dump) {
    Registration observed;
    auto lookup = [&](const char* key) -> const char* {
        for (const auto& [name, value] : variables)
            if (std::strcmp(name, key) == 0) return value;
        return nullptr;
    };
    register_live_renderer_from_environment(
        lookup,
        [&](const std::string& frame_dir, bool dump_bmps) {
            ++observed.calls;
            observed.frame_dir = frame_dir;
            observed.dump_bmps = dump_bmps;
        });
    if (observed.calls != 1 || observed.frame_dir != expected_dir ||
        observed.dump_bmps != expected_dump) {
        std::fprintf(stderr,
                     "%s: calls=%u dir=%s dump=%d, expected calls=1 dir=%s dump=%d\n",
                     name, observed.calls, observed.frame_dir.c_str(),
                     (int)observed.dump_bmps, expected_dir, (int)expected_dump);
        return false;
    }
    return true;
}

} // namespace

int main() {
    static_assert(!kFrameDumpsByDefault,
                  "the shared renderer must not dump periodic BMPs by default");
    static_assert(frame_dump_request_allowed(true, nullptr));
    static_assert(!frame_dump_request_allowed(true, "1"));

    bool ok = true;
    ok &= check("no variables", {}, ".", false);
    ok &= check("output directory only", {{"PROSPER_FRAME_DIR", "frames"}}, "frames", false);

    constexpr const char* opt_ins[] = {
        "PROSPER_FRAME_DUMPS",
        "PROSPER_DUMP_CONTENT",
        "PROSPER_FRAME_DUMP_FIRST",
        "PROSPER_FRAME_DUMP_EVERY",
    };
    for (const char* opt_in : opt_ins) {
        const std::string enabled_name = std::string(opt_in) + " enables";
        ok &= check(enabled_name.c_str(), {{opt_in, "1"}}, ".", true);

        const std::string disabled_name = std::string(opt_in) + " overridden by disable";
        ok &= check(disabled_name.c_str(),
                    {{opt_in, "1"}, {"PROSPER_NO_FRAME_DUMPS", "1"}}, ".", false);
    }
    ok &= check("disable alone", {{"PROSPER_NO_FRAME_DUMPS", "1"}}, ".", false);
    return ok ? 0 : 1;
}
