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
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace prosper::gpu {

inline constexpr size_t kRefusedShaderDumpMaxPrograms = 64;
// A dispatch the live compute backend declined (#4808) is dumped under its own budget. Those are
// the largest skip category and every one recompiled cleanly, so letting them share the 64 slots
// above would let a title with many declined programs hide the recompile refusals this dump
// exists for.
inline constexpr size_t kBackendDeclinedDumpMaxPrograms = 24;

enum class RefusedShaderBudget { Refusal, BackendDeclined };

// Observation state belongs to the existing immutable original-code owner, not an address map.
// Its lifetime/byte budget is the producer's; a replacement owner starts unobserved. The packed
// epoch/stage bits also let test-only run resets re-arm a retained owner without stale observations.
struct RefusedShaderInputCount;
struct RefusedShaderMemo {
    mutable std::atomic<uint64_t> epoch_stages{0};
    mutable std::atomic<uint64_t> code_hash{0};
    // Only input capture uses this owner-local lock. Warm observations never take DumpState's
    // process-global lock; the record is shared only by owners of identical code at this stage.
    mutable std::mutex input_mutex;
    mutable uint64_t input_epoch = 0;
    mutable std::array<std::shared_ptr<RefusedShaderInputCount>, 4> input_counts;
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);

struct RefusedShaderSource {
    std::shared_ptr<const std::vector<uint32_t>> words;
    const RefusedShaderMemo* memo = nullptr;   // owned by the same control block as words
};

// The live inputs a refusal was made with (#4808, refused_shader_inputs.hpp). `write` is called
// once, beside a NEWLY dumped program, with that program's path minus ".bin", and writes the
// sidecar there; `identity` names the input set for the per-program distinct-set counter.
struct RefusedShaderInputSet {
    uint64_t identity = 0;
    std::function<bool(const std::string& stem)> write;
    // Called only while counting is enabled and unsaturated. Metadata can change per dispatch,
    // so hashing may not be cached by a table/config pointer or address.
    std::function<uint64_t()> identify;
};
inline constexpr size_t kRefusedShaderMaxInputSetsCounted = 256;

bool note_refused_shader(const char* stage, uint64_t address, const RefusedShaderSource& source,
                         const std::string& detail,
                         RefusedShaderBudget budget = RefusedShaderBudget::Refusal,
                         const RefusedShaderInputSet* inputs = nullptr);
// An owner-version/stage probe, never an address/first-word shortcut. Does not mark an observation.
bool refused_shader_already_noted(const char* stage, const RefusedShaderSource& source);
// Before owning cold compiler inputs: missing sidecars and saturated counters need no observation.
bool refused_shader_inputs_needed(const char* stage, const RefusedShaderSource& source);
bool note_refused_shader_inputs(const char* stage, uint64_t address, const RefusedShaderSource&,
                                const std::string& detail, const RefusedShaderInputSet&);
bool note_refused_compute_shader(uint64_t address, const RefusedShaderSource& source,
                                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z,
                                 const RefusedShaderInputSet* inputs = nullptr);
// `stage` is a short tag ("vs", "ps", "cs"); `code` is the guest program, `dwords` its proven-readable
// length. `detail` is appended to the index line verbatim (sizes, draw order, reject reason).
// Returns true when this call wrote a new program.
bool note_refused_shader(const char* stage, uint64_t address, const uint32_t* code, size_t dwords,
                         const std::string& detail,
                         RefusedShaderBudget budget = RefusedShaderBudget::Refusal,
                         const RefusedShaderInputSet* inputs = nullptr);

// Stops default diagnostic work, not shader execution/rejection or explicit unbounded dumping.
bool refused_shader_dump_full();
// The same for the backend-declined budget, which fills independently of the refusal budget.
bool backend_declined_dump_full();
// Advances when the dump is reset, so a per-thread cache of what was already tried can tell.
uint64_t refused_shader_dump_epoch();

struct RefusedShaderDumpStats {
    size_t content_records = 0;   // all process-owned dedup metadata; no address/alias table
    uint64_t hash_evaluations = 0;
    uint64_t input_set_global_locks = 0;
    uint64_t input_identity_evaluations = 0;
    uint64_t hashed_dwords = 0;
    uint64_t graphics_source_copy_dwords = 0;   // diagnostic ownership attempts, never cache work
};
RefusedShaderDumpStats refused_shader_dump_stats();
void note_refused_graphics_source_copy(size_t dwords);

// Test hooks: where the current run writes, and a quiescent reset with a new root. Production never
// resets the observation epoch while workers run.
std::string refused_shader_dump_directory();
void reset_refused_shader_dump_for_test(const std::string& root);

}  // namespace prosper::gpu
