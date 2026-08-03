#pragma once

namespace prosper {

// Revision compiled into this build. "unknown" is returned when the source tree is not a Git
// checkout or Git was unavailable while building.
const char* embedded_build_revision() noexcept;

} // namespace prosper
