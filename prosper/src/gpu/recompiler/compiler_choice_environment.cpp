// The one place the recompiler's switch-type choices meet the process environment (ADR 0006, spec
// GPU-1). Callers read it HERE and pass the result in as a request; the emitters read the request.
// Until every entry point supplies its own choices, an unsupplied choice still lands here, so this
// file is the whole remaining surface and the recompiler-include/getenv ratchet can watch it shrink.
// Environment edits must be serialized with sampling/compilation. Live callers use their launch
// environment; isolated tests change these switches only between compiler invocations.
#include "gpu/recompiler/compiler_choices.hpp"

#include <cstdlib>

namespace prosper::gpu {
namespace {
// Per-process once, as the emitters' own function-local statics always were.
int64_t forced_array_layer_from_environment() {
    static const int64_t v = [] {
        // NOLINTNEXTLINE(concurrency-mt-unsafe): launch-environment read; no concurrent edits.
        const char* e = std::getenv("PROSPER_FORCE_LAYER");
        if (!e || !*e) return int64_t{-1};
        char* end = nullptr;
        const long n = std::strtol(e, &end, 10);
        if (!end || *end || n < 0 || n > 65535) return int64_t{-1};
        return int64_t{n};
    }();
    return v;
}
int64_t vcc_scalar_merge_from_environment() {
    // NOLINTNEXTLINE(concurrency-mt-unsafe): launch-environment read; no concurrent edits.
    static const bool v = std::getenv("PROSPER_VCC_SCALAR_DATA_MERGE") != nullptr;
    return v ? 1 : 0;
}
}   // namespace

int64_t environment_compiler_choice(CompilerChoice choice) {
    switch (choice) {
        case CompilerChoice::ForcedArrayLayer: return forced_array_layer_from_environment();
        case CompilerChoice::VccScalarMerge: return vcc_scalar_merge_from_environment();
        // Read at every use, never cached: a test arms PROSPER_MIMG_SOFT between compiles.
        // NOLINTNEXTLINE(concurrency-mt-unsafe): edits occur only between compiler invocations.
        case CompilerChoice::MimgSoft: return std::getenv("PROSPER_MIMG_SOFT") != nullptr ? 1 : 0;
        case CompilerChoice::NormalizeSamplerCoordinates:
            // NOLINTNEXTLINE(concurrency-mt-unsafe): edits occur only between compiler invocations.
            return std::getenv("PROSPER_NO_UNNORMALIZED_COORD_NORMALIZE") == nullptr ? 1 : 0;
        default:
            return 0;   // not a switch-type choice; the lambda-free form is only used for the four above
    }
}

CompilerChoiceSet compiler_choices_from_environment() {
    CompilerChoiceSet set;
    for (const auto choice :
         {CompilerChoice::ForcedArrayLayer, CompilerChoice::VccScalarMerge,
          CompilerChoice::MimgSoft, CompilerChoice::NormalizeSamplerCoordinates})
        set.set(choice, environment_compiler_choice(choice));
    return set;
}
}   // namespace prosper::gpu
