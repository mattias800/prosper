// refused_shader_dump.hpp -- keep the evidence of every refused shader, by default (#4273).
//
// A refused shader drops draws or skips a dispatch. The per-shader reject lines appear only under
// PROSPER_DBG and the raw code is written only under PROSPER_SHADER_DUMP, so by the time anyone notices
// missing content the run that showed it has left nothing behind. This records each DISTINCT refused
// guest program once per run: its raw code, plus one index line naming the stage, address, size and
// first unsupported instruction. Bounded (kMaxPrograms) and deduplicated by code hash, so a title
// that refuses one shader every frame writes one file. The directory is created lazily under
// PROSPER_CAPTURE_DIR (default: the working directory, like the F9 grab), so a clean run creates
// nothing. A diagnostic: it changes nothing the guest sees. PROSPER_NO_REFUSED_SHADER_DUMP turns it
// off.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace prosper::gpu {

inline constexpr size_t kRefusedShaderDumpMaxPrograms = 64;

// `stage` is a short tag ("vs", "ps", "cs"); `code` is the guest program, `dwords` its proven-readable
// length. `detail` is appended to the index line verbatim (sizes, draw order, reject reason).
// Returns true when this call wrote a new program.
bool note_refused_shader(const char* stage, uint64_t address, const uint32_t* code, size_t dwords,
                         const std::string& detail);

// Test hooks: where the current run writes, and resetting the per-run state with a new root.
std::string refused_shader_dump_directory();
void reset_refused_shader_dump_for_test(const std::string& root);

}  // namespace prosper::gpu
