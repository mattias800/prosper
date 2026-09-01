#pragma once
// `PROSPER_SHADER_DUMP_PROGRAM` — restrict `PROSPER_SHADER_DUMP_SUCCESS` to named guest programs.
//
// `PROSPER_SHADER_DUMP_SUCCESS=DIR` writes every successfully recompiled shader, and its filenames
// used to carry only content hashes. So once an investigation had identified a program BY ADDRESS —
// which is how `PROSPER_SKIP_DRAW_PROGRAM`, `PROSPER_COMPUTE_SKIP_PROGRAM`,
// `PROSPER_DRAW_PROGRAM_CENSUS` and the `[buf-op]` / `[mubuf-unresolved]` diagnostics all name
// programs — there was no way to ask the dump for it, and the only route left was hash-matching a
// directory by hand. Two unrelated investigations hit that wall independently (#3196, and the
// comment in `live_compute.cpp` beside `PROSPER_COMPUTELOG_RAW`).
//
// The fix has two halves and needs both: the address now appears in the filename, and this selector
// is what makes that usable on a title that compiles hundreds of programs — a 4K run should not
// write thousands of modules when one is wanted.
//
// Contract, mirroring `PROSPER_SKIP_DRAW_PROGRAM` and `PROSPER_COMPUTE_SKIP_PROGRAM` deliberately:
//   * Default OFF. Unset, `allows()` is true for everything and the dump behaves exactly as it did.
//   * Impossible to arm by accident. The spec goes through the STRICT hex parser in
//     `watch_list.hpp`, so a bare decimal, a stray comma, trailing junk, an overflow or a zero
//     address arms NOTHING. `strtoull(spec, &end, 0)` is the trap this avoids: base 0 reads an
//     unprefixed token as DECIMAL, and the run then reports itself armed on an address nobody asked
//     about.
//   * It reports itself, once, naming the variable and the count. A log from a filtered run can
//     never later be read as a full one.
//
// ONE DELIBERATE ASYMMETRY WITH THE SKIP SELECTORS, because the failure directions are opposite.
// A malformed *skip* spec must decline nothing; a malformed *dump* spec must not dump nothing.
// Withholding every module would leave an empty directory, and an empty directory reads as "that
// program was never compiled" — a false negative dressed as evidence, which is the failure mode the
// diagnostics folder's own AGENTS.md warns about. So a malformed spec leaves the filter DISARMED,
// says so loudly, and the run dumps everything: the address is in the filename now, so nothing is
// lost, only volume is.
//
// TWO LIMITS A READER OF THE RESULT CANNOT SEE IN THE OUTPUT.
//   1. **An address names a PROGRAM, not a variant.** One program recompiled against different
//      resource tables yields different SPIR-V, and each variant is dumped under the same address
//      with a different hash. Several files for one address is the expected shape, not a bug.
//   2. **Only programs that recompile SUCCESSFULLY reach the dump.** A program the recompiler
//      rejected is written by `PROSPER_SHADER_DUMP` instead, so naming its address here produces
//      nothing. Silence is not proof the address was wrong — check the arming line first.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace prosper::gpu {

class ShaderDumpProgramFilter {
public:
    enum class ConfigureResult { Unset, Armed, Malformed };

    // Parse `spec` (the raw environment value; null or empty means "unset"). Arms only on a
    // completely valid list — never partially. Pure apart from the member state, so it is
    // unit-testable with no GPU and no environment.
    ConfigureResult configure(const char* spec);

    // All state reads take the lock: `configure()` can re-run when a test changes the environment,
    // and the dump path itself is reached from the parallel draw-realization workers.
    bool armed() const;
    std::size_t size() const;

    // True when this program may be dumped: always, when disarmed. `chain_address` is the NGG main
    // continuation of a vertex chain, 0 when there is none; naming EITHER half selects the pair,
    // because a caller reading a census does not always know which half it is holding.
    bool allows(uint64_t program_address, uint64_t chain_address = 0) const;

    // Counted only while armed. The 1-based ordinal of this withholding, and whether it should be
    // printed under `diag_ratelimit.hpp`'s contract (first, then powers of two).
    struct Withheld { uint64_t ordinal = 0; bool print = false; };
    Withheld note_withheld();
    uint64_t withheld_total() const;

private:
    std::vector<uint64_t> addresses_;
    mutable std::mutex mutex_;
    uint64_t withheld_ = 0;
};

// The process-wide instance, synchronized from `PROSPER_SHADER_DUMP_PROGRAM`. It re-reads the
// environment only when the value has actually changed, so the ordinary run parses once and prints
// its arming line once; a test that sets the variable still gets a correctly re-armed filter, which
// a `static` one-shot would not give it.
ShaderDumpProgramFilter& shader_dump_program_filter();

}  // namespace prosper::gpu
