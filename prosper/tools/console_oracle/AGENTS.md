# tools/console_oracle

An **optional cross-check, measured by a human with their own console**: it records what a real PS5
returns for a table of system-library calls, so prosper's HLE can be compared with a measurement.
Read-only observation of your own console.

It is never a required step for adding or changing HLE. Disassembly, live guest captures and tests
come first and stay sufficient. Nothing in CI or in the replay needs a console: the committed goldens
are plain test inputs, and a mismatch is always resolvable without hardware (fix the HLE, or list the
case in `known_gaps.tsv` with a reason). If your own evidence contradicts a golden, do not edit it:
list the case in `known_gaps.tsv` with that evidence and flag it for a human to re-measure.

- `payload/lib_oracle.c` + `payload/Makefile` -- the payload. Built with the ps5-payload-sdk (not by
  CMake), it `dlopen`s the named libraries, calls each function with the arguments in the cases file
  and prints one result line per call. The argument grammar and the result format are documented at the
  top of `lib_oracle.c`; that comment is the specification.
- `run_oracle.py` -- the driver. Embeds a cases file into the payload, builds it, sends it to the
  console's ELF loader, parses what came back and writes the golden file. Run it from Linux or WSL
  (it needs `make`, the installed SDK and a network path to the console). `--check` verifies an
  existing golden against its cases file without a console.
- `test_run_oracle.py` -- pytest for the driver's pure parts and for the committed data.

The data lives in `tests/data/console_oracle/`: `<family>.cases.tsv` (what to call),
`<family>.golden.tsv` (what the console returned) and `known_gaps.tsv` (cases where prosper is known to
differ, each with a reason). `tests/hle/test_console_oracle_replay.cpp` replays every golden through
prosper's HLE and fails on any mismatch that is not a listed gap; a listed gap that now matches also
fails, so the list only shrinks.

## Adding a case

Adding a case needs a measurement, so it is a step for a person with a console. Without one, change the
HLE the usual way (disassembly, captures, a normal test) and leave the case for them.

1. Add a line to a `<family>.cases.tsv` (`id`, `lib`, `func`, `args`, optional `expect`).
2. A person with a console re-measures:
   `python3 prosper/tools/console_oracle/run_oracle.py --cases <file> --host <console> --sdk <sdk>`.
3. Run `test_console_oracle_replay`. A mismatch is either a prosper defect (fix it, or list it in
   `known_gaps.tsv` with a reason) or a case that measures something volatile (give it `expect=ret`
   or `none`; the driver then stores a placeholder instead of the volatile bytes).

## Traps

- **Volatile results are not goldens.** Time, random bytes, process ids and device identifiers differ
  per run or per console. Compare only the return code (`ret`) or nothing (`none`), and never commit a
  value that identifies a console or a user.
- **Pointers do not compare across address spaces.** Use `retoff:<k>` and `outptr:<k>` so a returned or
  written pointer is compared as an offset into one of the call's own buffers.
- **A function's `int` return is only 32 bits.** The default comparison masks to the low 32 bits;
  add `r64` for `size_t`, `long` and pointers.
- **A fault or hang in the called function.** A fault (SIGSEGV etc.) is caught and recorded as
  `fault:<signal>`; a hang is not recovered and shows up as a missing `# done` line.
- **String arguments are inputs only.** An `s:` buffer is never reported, so a function that writes
  into a string (`strtok`, in-place transforms) needs an `in:` or `out:` buffer to be compared.
- **Out of scope:** floating point and variadic functions (the `printf` family), and prosper handlers
  registered through the guest-ABI or typed paths, which `HleFn` cannot call. The replay reports those as
  "registered, but not callable through HleFn" instead of "not implemented".
- **Do not `dlclose` a system library.** It fails on this console, so the payload never does.
- **Goldens are measurements, not Sony code.** Commit call results only -- no firmware, no game data,
  nothing that came out of a module's code or data sections.
