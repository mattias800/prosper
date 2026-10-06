// refused_shader_dump.hpp -- keep the evidence of every refused shader, by default (#4273).
//
// A refused shader drops draws or skips a dispatch. The per-shader reject lines appear only under
// PROSPER_DBG and the raw code is written only under PROSPER_SHADER_DUMP, so by the time anyone notices
// missing content the run that showed it has left nothing behind. This records each DISTINCT refused
// guest program once per run: its raw code, plus one index line naming the stage, address, size and
// first unsupported instruction. Bounded (kMaxPrograms) and deduplicated by code hash, so a title
// that refuses one shader every frame writes one file. A draw the owned-wave gate refuses is
// recorded too, although nothing was recompiled: its index line ends in `refusal=<reason>`, and
// that field, not first_bad_*, is why the draw was lost (#4555). The [refused-shader] line on
// stderr carries the same field. The directory is created
// lazily under PROSPER_CAPTURE_DIR (default: the working directory, like the F9 grab), so a clean
// run creates nothing. A diagnostic: it changes nothing the guest sees.
// PROSPER_NO_REFUSED_SHADER_DUMP turns it off.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace prosper::gpu {

inline constexpr size_t kRefusedShaderDumpMaxPrograms = 64;

// Observation state belongs to the existing immutable original-code owner, not an address map.
// Its lifetime/byte budget is the producer's; a replacement owner starts unobserved. The packed
// epoch/stage bits also let test-only run resets re-arm a retained owner without stale observations.
struct RefusedShaderMemo {
    mutable std::atomic<uint64_t> epoch_stages{0};
};
static_assert(sizeof(RefusedShaderMemo) == sizeof(uint64_t));
static_assert(std::atomic<uint64_t>::is_always_lock_free);

struct RefusedShaderSource {
    std::shared_ptr<const std::vector<uint32_t>> words;
    const RefusedShaderMemo* memo = nullptr;   // owned by the same control block as words
};

bool note_refused_shader(const char* stage, uint64_t address, const RefusedShaderSource& source,
                         const std::string& detail);
// An owner-version/stage probe, never an address/first-word shortcut. Does not mark an observation.
bool refused_shader_already_noted(const char* stage, const RefusedShaderSource& source);
bool note_refused_compute_shader(uint64_t address, const RefusedShaderSource& source,
                                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);

// `stage` is a short tag ("vs", "ps", "cs"); `code` is the guest program, `dwords` its proven-readable
// length. `detail` is appended to the index line verbatim (sizes, draw order, reject reason).
// Returns true when this call wrote a new program.
bool note_refused_shader(const char* stage, uint64_t address, const uint32_t* code, size_t dwords,
                         const std::string& detail);

// Stops default diagnostic work, not shader execution/rejection or explicit unbounded dumping.
bool refused_shader_dump_full();

struct RefusedShaderDumpStats {
    size_t content_records = 0;   // all process-owned dedup metadata; no address/alias table
    uint64_t hash_evaluations = 0;
    uint64_t hashed_dwords = 0;
};
RefusedShaderDumpStats refused_shader_dump_stats();

// Test hooks: where the current run writes, and a quiescent reset with a new root. Production never
// resets the observation epoch while workers run.
std::string refused_shader_dump_directory();
void reset_refused_shader_dump_for_test(const std::string& root);

}  // namespace prosper::gpu
