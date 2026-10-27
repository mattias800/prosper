// Stable refusal identity excludes the separately printed placeholder-root envelope.
#pragma once
#include <string_view>

namespace prosper::gpu {
// Returns a borrowed prefix; it never changes the complete terminal diagnostic.
std::string_view recompile_refusal_identity(std::string_view reason);
} // namespace prosper::gpu
