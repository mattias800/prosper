#pragma once

namespace prosper {

// Revision compiled into this build. "unknown" is returned when the source tree is not a Git
// checkout or Git was unavailable while building.
const char* embedded_build_revision() noexcept;
// Actual source bytes and toolchain/configuration hashed at build time. Unlike HEAD, this changes
// for dirty/untracked compiled source edits and never samples a later runtime working directory.
const char* embedded_build_source_identity() noexcept;

} // namespace prosper
