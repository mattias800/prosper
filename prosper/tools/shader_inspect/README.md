# Shader inspect

`shader_inspect` decodes one raw RDNA2 shader. It is the fast offline path for mapping shader control
flow before changing the recompiler.

```bash
cmake --build build-linux -j8 --target shader_inspect
./build-linux/shader_inspect /tmp/shaders/exec_ps_7f123400.bin

# Run the complete stage translator too; PROSPER_DBG identifies a contextual rejection PC.
PROSPER_DBG=1 ./build-linux/shader_inspect /tmp/shaders/exec_vs_7f123400.bin --stage vertex
PROSPER_DBG=1 ./build-linux/shader_inspect /tmp/shaders/exec_cs_7f123400.bin --stage compute
```

`PROSPER_SHADER_DUMP=DIR` writes raw shaders that fail recompilation. To inspect a shader that succeeds,
run the title or replay with `PROSPER_SHADER_DUMP_SUCCESS=DIR`. Each unique successful graphics or compute shader
produces a pair such as:

```text
success_ps_at_00000005008f1400_35956264829da0c6_62ad8faab4566b7a.bin
success_ps_at_00000005008f1400_35956264829da0c6_62ad8faab4566b7a.spv
success_cs_at_0000000413dc6700_6f6a5fdd6e9f8d73_2339aa565126d182.bin
success_cs_at_0000000413dc6700_6f6a5fdd6e9f8d73_2339aa565126d182.spv
```

The field after the stage tag is the **guest code address** (#3196), so a program you identified by
address is one glob away — `ls DIR/success_*_at_00000005008efd00_*` — and
`PROSPER_SHADER_DUMP_PROGRAM=0x5008efd00` narrows the run to it so a 4K boot does not write thousands
of modules. A chained vertex program also writes a `_main_<main-addr>_<main-hash>.bin` holding its NGG
continuation, and either half's address selects the pair. See `tools/AGENTS.md` for the filter's full
contract, including why a malformed spec dumps everything rather than nothing.

For a large live compute program that reaches the backend but fails during Vulkan pipeline creation,
filter the backend trace by guest code address and dump only that exact translated module:

```sh
PROSPER_COMPUTELOG_CODE=0x50057b800 \
PROSPER_COMPUTELOG_SPIRV=~/prosper-shaders/target.spv \
  ./build-linux/prosper-app --dump <DUMP_ROOT>/PPSA21564-app0
spirv-val --target-env vulkan1.2 ~/prosper-shaders/target.spv
```

The destination directory must already exist. As with raw shader dumps, the resulting SPIR-V is
title-derived diagnostic data: keep it local and do not commit it.

`PROSPER_COMPUTELOG_RAW=PATH` is the other half of that pair, and it is the one `shader_inspect`
consumes: it writes the **guest RDNA2 bytes** of the same traced program, so "what did the guest
write" and "what did we emit" can be compared for one address in a single run.

```sh
PROSPER_COMPUTELOG_CODE=0x413dc6700 \
PROSPER_COMPUTELOG_RAW=~/work/target.bin \
PROSPER_COMPUTELOG_SPIRV=~/work/target.spv \
  ./build-linux/prosper-app <DUMP_ROOT>/<TITLE_ID>-app0
./build-linux/shader_inspect ~/work/target.bin        # guest ISA, with resolved branch targets
```

Reach for it whenever the question is about **control flow** — where the loops are, what a loop's
exits are, whether a branch is backward. `PROSPER_SHADER_DUMP_SUCCESS` also writes raw `.bin` files,
and since #3196 those filenames carry the code address too, so either route recovers a program by
address; this one is still the right choice when you want *one* live traced program and its emitted
SPIR-V written to exact paths you name. The program length is re-derived by decoding to the program's
own terminator rather than trusted, and a run prints what it wrote:

```text
[compute]   traced raw program=0x413dc6700 dwords=903 instructions=781 window=262144 path=... result=written
```

The first hash identifies the translated SPIR-V and the second identifies the exact raw RDNA2 stream.
Pass the `.bin` file to `shader_inspect`; use the adjacent `.spv` for SPIR-V disassembly or validation.
The dump directory is created automatically. `vs`, `ps`, and `cs` identify vertex, pixel, and compute stages.

Each instruction row includes its dword PC, encoded length, format/opcode, exact raw words, decoded
operands, and, for SOPP branches, the signed immediate and resolved target PC. The header also prints
the stage-agnostic `recompile_coverage` result. That coverage is intentionally compute-safe; a VCC
control-flow shape may remain listed there even when the vertex/fragment structurizer accepts it.
`--stage vertex|fragment|compute` additionally runs the complete stage translator, which exposes contextual
resource and structured-control-flow failures that per-instruction coverage cannot decide.

## `--mimg-sites`: the MIMG census, for tools that must not carry their own decoder

```bash
./build-linux/shader_inspect exec_ps_7f123400.bin --mimg-sites
mimg-site pc=0042 op=0x20 dim=5
mimg-site pc=0061 op=0x00 dim=1
mimg-sites-end dwords=903 consumed=903 instructions=781 sites=2 endpgm=1
```

One line per image instruction — its dword PC, its 8-bit opcode with the split MSB already
reassembled, and its `DIM` — and nothing else: no coverage, no stage recompile, no resource-table
reasoning. The default listing is untouched; this replaces it, and combining it with `--stage` is a
usage error so the exit code stays unambiguous.

**Why it exists.** `tools/shader_conformance/scan.py` needs to know where the image instructions are
in a raw dump, and to do that it needs real instruction lengths — otherwise an operand or a trailing
literal dword whose top six bits alias the MIMG encoding reads as an instruction. It used to carry
its own Python port of `rdna2_decode_one`'s format and length dispatch for exactly that (#3040). The
port was correct, and that was never the problem: a copy of a decoding rule is the one nobody
updates when the decoder learns a new literal-forcing opcode or a wider NSA field, and it fails by
silently returning a plausible instruction stream. This flag deletes the copy (#3184).

`mimg-sites-end` is a **completion sentinel, not decoration**. A consumer must be able to tell "this
shader contains no image instruction" — the answer that certifies a clean conformance scan — from
"this process printed nothing because it died", and those two must never look alike. `scan.py`
refuses any census that arrives without it, and cross-checks its `sites=` count against the number
of `mimg-site` lines it parsed — so a line the consumer's parser stops matching is loud rather than
a quietly smaller census.

Exit is `0` whenever the file was read and walked; `2` for usage or I/O errors, including an empty
input or one that is not a whole number of dwords. A stream that stops short of `s_endpgm` is
reported as `endpgm=0` rather than as a failure — a census over a truncated dump is still an exact
census of what could be decoded.

## `--wave-reasons`: which shaders need the guest Wave64, and why

On any NVIDIA host the entire supported subgroup range is `32..32` -- no extension, driver or
device yields a 64-wide subgroup -- so a fragment module that requires the guest's Wave64 is
dropped. On *Grand Theft Auto V* that removes most of the world's lighting (#3464).

```sh
shader_inspect <raw-rdna2.bin> --wave-reasons
wave-reasons required-subgroup-size=64 reasons=0x2 names=wave-any reason-set-admissible=1 \
    internal-gds=0 subgroup-features=0x1
wave-reasons gate-undecided=title-allowlist,host-subgroup-size-control,host-subgroup-features ...
wave-reasons-end file=... input=rdna2 dwords=12 recompiled=1 table_dependent=0 endpgm=1
```

### `reason-set-admissible` is NOT `admitted`, and the difference is most of the answer

The renderer's gate is **five conjuncts** (`tests/fixtures/render_runner.h:7128-7133`). The
reason-set equality at `:7140` -- the one this field mirrors -- is only the innermost. The
decisive one is `bd.allow_native_fragment_vote_width`, which **defaults false** (`:565`) and is
set in exactly one place, from `title_id == "PPSA04263"` (`live_renderer.cpp:1216`, assigned
`:7591`). Its own comment says so: *"Tests, replay and all other titles leave this false."*

**So outside GTA V, every shader requiring more than 32 lanes is dropped regardless of its
reason set.** An earlier version of this tool called the reason-set test `native-wave32-admitted`
and was therefore wrong in the one direction that mattered: it under-reported the loss #3464
exists to size. Its *Blasphemous 2* validation run reported `0 dropped` where the true answer
was `2` -- and it went unquestioned because it agreed with the expectation.

The two conjuncts a module CAN decide are reported beside it (`internal-gds`,
`subgroup-features`); the three it cannot are named on their own `gate-undecided=` line, every
time, so no consumer can read admission into the row by omission. For a per-title verdict use
`capture_wave_census.py --title`.

A module that needs no particular width reports `size=0 reasons=absent reason-set-admissible=0`.
`reasons=0x0` is never printed: a module either carries the marker or does not, and `absent` is
not none. A module that fails to recompile prints **no census row at all** -- only the sentinel,
with `recompiled=0`.

A SPIR-V input reports `recompiled=n/a` (nothing was recompiled -- the module was read) plus
`spirv-walk=ok|truncated`. That last field exists because a corrupt module reports `size=0
reasons=absent`, byte-identical to a genuine wave-free shader, so without it an unreadable dump
lands in the population that says #3464 does not affect it.

`wave_reason_census.py` tallies the mode over a directory and prints the reason-set histogram
plus the names of the dropped shaders. It exists because the *runtime* skip log answers this
only for whatever a route happened to reach, once per distinct shader, at whatever scene depth
the run stopped: two runtime counts taken that way, 43 and 86, differed by run length alone and
read as a regression until they were matched by frame.

### Measured limit: over a raw dump this mode is mostly blind, and says so

Run against a 3,453-shader dump of GTA V's pixel-shader database (2026-09-08):

```
files=3453  errors=0  unrecompiled(no resource table)=3430  analysed=23
```

**99.3% contributed no reason data.** A real pixel shader samples a texture, and the section
below explains why a table-less tool can never lower one. The 23 that were analysed are the
shaders that happen not to touch memory, which is not a random sample of anything -- a
texture-sampling lighting shader is exactly the kind that is missing.

This is reported rather than hidden: an unrecompilable shader is counted in its own column and
is **never** tallied as `requires nothing`, and the tally prints the caveat both above and below
its numbers. A census that silently equated "could not be analysed" with "needs no wide wave"
would have reported that GTA V has essentially no Wave64 problem, which is the opposite of the
truth.

### The way out: read the module gpu_replay compiled

`--wave-reasons` accepts **either** a raw RDNA2 stream **or** a SPIR-V module, detected by the
SPIR-V magic rather than by a flag or a file extension -- the magic is the format's own
self-identification, so a renamed or mislabelled file cannot be read as the wrong kind.

`gpu_replay` has the real resource table, and `--dump-shader DRAW:fs PATH` already writes the
recompiled module. So the census over memory-using shaders is a two-step pipeline with no new
analysis in it:

```sh
gpu_replay --dump-shader 18:fs fs.spv <capture>.prgcap   # real descriptors, real compile
shader_inspect fs.spv --wave-reasons                     # reads the module's own markers
```

The reasons then come from a real-table compile while this tool remains the single place that
formats them. A SPIR-V input reports `input=spirv` in the sentinel and leaves `table_dependent`
and `endpgm` at zero -- those describe an RDNA2 walk that did not happen.


## `--stage` cannot prove a shader is unsupported (#1571)

**`shader_inspect` has no resource table, and a table-less stage rejection is NOT evidence of a shader
defect.** A raw dump carries instructions but no descriptors, so the tool can never build a
`ShaderResourceTable`. The recompiler then refuses — by design — to lower anything that resolves a
V#/T#/S# through that table: `MIMG`, `MUBUF` and `MTBUF` in every stage, and additionally `SMEM` in the
**vertex and fragment** stages, which gate scalar memory on `allow_smem = (rt != nullptr)`. Compute
passes `allow_smem = true`, so constant-buffer loads are fine there — but compute image and buffer ops
still need the table.

The gates live in `emit_alu` in `src/gpu/recompiler/rdna2_to_spirv.cpp` (MIMG rejects on `!allow_smem || !rt`,
SMEM on `!allow_smem`); the per-stage `allow_smem` values are set by `recompile_fragment_impl` and
`recompile_vertex_impl` (`rt != nullptr`) versus `recompile_compute` (`true`). Those functions are named
rather than cited by line because line numbers drift — grep the names.

When such an instruction is present the tool now reports:

```text
stage-recompile stage=fragment status=undetermined-no-resource-table spirv_dwords=0 table_dependent=17
stage-recompile NOTE: TOOL LIMITATION, NOT A SHADER DEFECT. ...
```

Treat `undetermined-no-resource-table` as "no verdict", never as a missing opcode. This mattered: on a
corpus of 114 shaders that had provably recompiled and rendered live, the pre-fix tool called **109 of
them `rejected`** — 33/35 compute, 27/29 fragment, 49/50 vertex. Agents acted on those false leads.

**And a reject that IS reported here can still name a different PC than the live one.** The missing
table is not only a missing descriptor set: with no launch state there are no seeded scalar inputs
either, so the Wave64 MUST dataflow starts with an empty `scalar_words` and `scalar_scc = false`, and
every proof that depends on a user-data SGPR fails offline that would hold live. Worked example
(#2790): Sonic Frontiers' `0x2005a0ca00` writes the same VCC pair twice, at pc64 (`886afd6b`) and
pc76 (`886a6bfd`) — **12 dwords / 9 instructions apart** in the same program (`pc` is a dword offset:
`rdna2_walk` does `i.pc = pc; pc += i.len_dwords`). It rejects **live at pc76** and **offline at
pc64**, because offline the *first* of the two already lacks its scalar sources. The offline PC was a
real reject of a real defect, and it was still the wrong PC to reason from: the two sites carry
different words and have different causes. Use the offline run to iterate quickly once you know the shape, and
take the PC itself from the live `[compute] skip unsupported program …` line.

**You can now supply the launch shape, and for any Wave64 MUST question you should.** Three
environment variables seed the `ComputeShaderConfig` the tool otherwise leaves empty:

```bash
PROSPER_SHADER_INSPECT_USER_SGPRS=14 \
PROSPER_SHADER_INSPECT_TGID=xy \
PROSPER_SHADER_INSPECT_LOCAL=16x3x1 \
PROSPER_DBG=1 ./build/shader_inspect exec_cs_2005717e00.bin --stage compute
```

`_USER_SGPRS=N` gives the launch `N` user SGPRs (`s0..sN-1`), `_TGID` enables the workgroup-id
registers that follow them (any subset of `xyz`), and `_LOCAL=XxYxZ` sets the workgroup dimensions —
the census line `local=16x3` is where those come from. Without them every SGPR the guest reads as
launch input is absent from the initial `RegState`, so `scalar_words` starts empty and an ordinary
`s_add_i32 vcc_lo, s14, 1` has a non-scalar source. Worked example (#2790): Sonic Frontiers'
`0x2005717e00` declines offline at **pc96** with a default config and at **pc481** — the live PC, 385
dwords later, and a completely different instruction — once `_USER_SGPRS=14 _TGID=xy` is supplied.
The first is a phantom the instrument manufactured. Bracket the value if you do not know it: the
right one is the smallest that makes the offline PC agree with the live `[compute] skip …` line.

**For a table-accurate verdict, use `gpu_replay`**, which has the real descriptors from a capture:

```bash
gpu_replay <capture>.prgcap --inspect-only                     # per-draw realize/fail status
gpu_replay <capture>.prgcap --dump-failed-shader FAILURE:STAGE out.bin   # a genuinely failed stage
```

Exit codes: `0` recompiled, `1` genuine defect (truncated stream, or a rejection attributable to the
shader), `2` usage/IO error, `3` undetermined because no resource table could be supplied. `3` is
deliberately non-zero — a table-less run is never reported as a pass.
When a shader contains the fully proven bounded scalar `s_setpc_b64` jump-table idiom, the header also
prints its constant-buffer selector, adjustment/clamp, complete target list, merge PC, and owning span.

The input is bounded to 16 MiB and decoding stops at the first `s_endpgm` or unknown instruction.
Raw and SPIR-V dumps contain title-derived code, are local-only, and must not be committed.
