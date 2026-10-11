// refused_shader_inputs.hpp -- the live compile inputs beside a refused-shader dump (#4808).
//
// The refused-shader dump (refused_shader_dump.hpp) keeps a refused program's code, but the
// recompiler's decision also depends on the launch state and the resolved descriptor table, so an
// offline recompile of the bare code stops at a different instruction than the live one did (150
// of 200 compute programs in one census). This records, once per dumped program, exactly what the
// live compute recompile was given: the complete ComputeShaderConfig (user-data SGPR values and
// launch shape included), the complete ShaderResourceTable metadata (every resolved T#/V#/S#
// field), and the bytes of the constant/vertex buffers the compiler may read, bounded by
// kRefusedShaderInputsMaxPayloadBytes. `shader_inspect --inputs <file>` replays it.
//
// The resource table travels in the `.prfc` compile-case codec (fragment_compile_case.hpp), which
// already owns the bounded, pointer-free encoding of a table and its backing; here it is only a
// carrier and is always marked INCOMPLETE, so it can never be replayed as a fragment case.
//
// Compute schema 1 does not record compiler-choice environment switches (offline uses the same defaults),
// the trip-bound diagnostic (disarmed by default), and guest memory the recompiler reads directly
// rather than through the table. Replay runs under a CompilerResourceScope, so such a read reports
// itself as INCOMPLETE instead of reading the host process's memory.
//
// Graphics schemas 2/3 additionally retain actual compiler selectors/spans/config/choice reads;
// schema 3 separates full terminal text from the resource carrier's smaller text ABI. See
// refused_graphics_inputs.hpp. Compute schema 4 appends the observed packet/register launch state;
// legacy schema 1 remains readable with that state explicitly unknown.
//
// A diagnostic: written only beside a newly dumped refused program, never read by the live path.
// Guest buffer bytes stay local, beside the dump; they are never part of the repository.
#pragma once

#include "gpu/recompiler/rdna2_to_spirv.hpp"
#include "gpu/resources/shader_resources.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

// Guest buffer bytes kept per sidecar. Above it the table is still recorded in full, but buffer
// contents become opaque and a replay that needs them reports INCOMPLETE.
inline constexpr uint64_t kRefusedShaderInputsMaxPayloadBytes = 256ull * 1024ull;
inline constexpr uint32_t kRefusedShaderInputsSchema = 4;

struct RefusedGraphicsInputs;
struct RefusedShaderInputs {
    std::string stage;   // "cs", "vs", "ps"
    std::shared_ptr<const RefusedGraphicsInputs> graphics;   // graphics schemas 2/3
    uint64_t program_address = 0;
    std::string live_reject;           // the recompiler's terminal reject, "" if none recorded
    std::string incomplete;            // why inputs are partial, "" when complete
    bool payload_omitted = false;      // buffer bytes exceeded the cap and were left out
    ComputeShaderConfig compute;
    bool has_resources = false;
    ShaderResourceTable resources;     // backing pointers re-pointed into owned storage
    // Per resource, whether its host backing was present live and whether its bytes were kept.
    // A replay serves exactly these to the compiler (CompilerResourceScope).
    std::vector<uint8_t> backing_present, backing_owned;
};

// The descriptor-level identity of one input set, for counting how many distinct sets a refused
// program was seen with. Covers the config and each resource's resolved descriptor fields, not
// buffer contents.
uint64_t refused_compute_inputs_identity(const ShaderResourceTable* table,
                                         const ComputeShaderConfig& config);

// Captures and writes `path` atomically enough for a diagnostic (write then rename). Never throws.
bool write_refused_compute_inputs(const std::filesystem::path& path, uint64_t program_address,
                                  const ShaderResourceTable* table,
                                  const ComputeShaderConfig& config,
                                  const std::string& live_reject);

// `<stem>.bin` -> `<stem>.inputs`. A dump written before #4808 has no such file; that is not an
// error, the program simply has no recorded inputs.
std::filesystem::path refused_shader_inputs_path(const std::filesystem::path& program);
// Throws std::runtime_error on a malformed, truncated, oversized or unknown-schema file.
RefusedShaderInputs decode_refused_shader_inputs(const std::vector<uint8_t>& bytes);
RefusedShaderInputs read_refused_shader_inputs(const std::filesystem::path& path);

struct RefusedShaderReplay {
    bool produced = false;             // the recompile returned a module
    std::string reject;                // the terminal reject this replay recorded
    std::string incomplete;            // set when the replay could not run on recorded inputs
};

// Recompiles the recompile span of `code` (rdna2_recompile_code_span, as the live compile key
// does) with the recorded inputs. With `use_inputs` false it recompiles with a
// default config and no table -- the historical shader_inspect behaviour -- for comparison.
RefusedShaderReplay replay_refused_shader_inputs(const std::vector<uint32_t>& code,
                                                 const RefusedShaderInputs& inputs,
                                                 bool use_inputs = true);

}  // namespace prosper::gpu
