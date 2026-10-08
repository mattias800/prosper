# tools/console_oracle

Measures what a **real PS5** returns for a table of system-library calls, so prosper's HLE can be
checked against the console instead of against a guess. Read-only observation of your own console.

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

1. Add a line to a `<family>.cases.tsv` (`id`, `lib`, `func`, `args`, optional `expect`).
2. Re-measure: `python3 prosper/tools/console_oracle/run_oracle.py --cases <file> --host <console> --sdk <sdk>`.
3. Run `test_console_oracle_replay`. A mismatch is either a prosper defect (fix it, or list it in
   `known_gaps.tsv` with a reason) or a case that measures something volatile (give it `expect=ret`
   or `none`).

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
- **Do not `dlclose` a system library.** It fails on this console, so the payload never does.
- **Goldens are measurements, not Sony code.** Commit call results only -- no firmware, no game data,
  nothing that came out of a module's code or data sections.
