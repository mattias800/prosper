# tools/console_oracle

An **optional cross-check, measured by a human with their own console**: it records what a real PS5
returns for a table of system-library calls, so prosper's HLE can be compared with a measurement.
Read-only observation of your own console.

It is never a required step for adding or changing HLE. Disassembly, live guest captures and tests
come first and stay sufficient. Nothing in CI or in the replay needs a console: the committed goldens
are plain test inputs, and a mismatch is always resolvable without hardware (fix the HLE, or list the
case in `known_gaps.tsv` with an action and a reason). If your own evidence contradicts a golden, do not
edit it: list the case in `known_gaps.tsv` as `triage` with that evidence and flag it for a human to
re-measure.

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
   `python3 prosper/tools/console_oracle/run_oracle.py --family <name> --host <console> --sdk <sdk>`.
   The family is a bare name: the tool only ever reads and writes `tests/data/console_oracle/`.
3. Run `test_console_oracle_replay`. A mismatch is either a prosper defect (fix it, or list it in
   `known_gaps.tsv` with an action and a reason) or a case that measures something volatile (give it `expect=ret`
   or `none`; the driver then stores a placeholder instead of the volatile bytes).

## What a known gap says

`known_gaps.tsv` is `id`, `action`, `reason`, so the list also answers "does this need fixing?":

- `fix` -- prosper is wrong here and should change. `fix:#4757` says where it is tracked.
- `keep` -- prosper differs on purpose and stays as it is; the reason says why (a safe direction, a
  deliberate floor). `keep:#4757` links the discussion.
- `triage` -- nobody has decided yet. This is also what to use when your own evidence contradicts a golden.

The replay rejects a row with no action, or an action outside that set. A gap that starts matching still
fails the replay, so fixing one means deleting its row.

## Command-buffer builders (`dcb:<N>`)

An Agc builder (`sceAgcDcbDrawIndexAuto`, `sceAgcAcbCopyData`, ...) takes a pointer to a command-buffer
descriptor, not to raw dwords, so a plain `out:` buffer cannot stand in for it. `dcb:<N>` builds the
descriptor over a fresh N-dword buffer pre-filled with 0xAB; the payload and the replay build it
identically. The reported value is `<dwords written>:<those dwords in hex>`, and `retoff:<k>` is relative
to the command dwords, so a builder's returned packet pointer compares as an offset. A builder that does
not fit leaves the cursor where it was and reports `0:`.

`gen_builder_cases.py` generates the `agcbuild` family: every exported Dcb/Cb/Acb builder, called with
three argument patterns whose values are easy to recognise in the emitted words, which recovers each
builder's argument mapping by observation (the signatures are not published). It only ever calls
builders, which encode their arguments into the descriptor's own buffer; it excludes the `*Patch*`
functions, which store through a pointer into a packet built earlier.

## Probe families and the resumable runner

A family named `probe_*`, or one with a row in `tests/data/console_oracle/probe_baseline.tsv`, is a
measured inventory rather than a list of individually reviewed gaps. Its differences are not listed one
by one in `known_gaps.tsv`; the replay counts them and fails if the count differs from the baseline, so
it can only go down (and the baseline is lowered with it). A mismatch printed as `[probe-gap]` is a lead.

Baselined today: `agcbuild`, `kernelx`, `nethttp`, `posix`. Five `kernelx` cases are deliberately absent:
using an event flag or equeue after deleting it aborts the replay process in prosper (#4779). Put them back
with the fix.

`expect=default0` compares an UNREGISTERED function against the dispatcher's return-0 default instead of
reporting "not implemented": what is left is exactly the false-success class (#2081), a function the
console answers with an error where prosper answers 0.

`run_oracle.py --resume N --prune-unmeasured` is for a bulk run. If the payload dies or hangs, the case it
stopped on is recorded as `hang` and the rest run again in a fresh payload (up to N times); only cases the
console ran to completion stay in the cases and golden files, and the others are listed in
`<family>.unmeasured.tsv`. A hang that never returns relies on `--timeout`.

What is never called, however it is generated: anything that changes console state (save data, settings,
reboot, install), anything that returns a device or account identifier, anything that talks to a network
endpoint, and `sceKernelDebugRaiseException`. Blind probing of every export is not the goal: a call with
garbage arguments mostly measures argument validation, not what a title relies on.

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
