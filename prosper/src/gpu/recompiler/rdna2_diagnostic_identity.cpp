#include "gpu/recompiler/rdna2_diagnostic_identity.hpp"
#include <string_view>

namespace prosper::gpu {
std::string_view recompile_refusal_identity(std::string_view reason) {
    const auto at = reason.rfind(" placeholder-roots=[");
    if (at == std::string_view::npos || !reason.ends_with(']')) return reason;
    const auto identity = reason.substr(0, at);
    if (identity.find("fabricated-fresh-mask-source") != std::string_view::npos ||
        identity.find("is fabricated on the back edge") != std::string_view::npos)
        return identity;
    return reason;
}
} // namespace prosper::gpu
