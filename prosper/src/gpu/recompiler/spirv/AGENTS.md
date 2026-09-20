# `spirv` — what the recompiler emits, one file per RDNA2 facility

The back half of RDNA2→SPIR-V. `../rdna2_decode` turns guest bytes into instructions and
`../rdna2_emit_*` decides *when* to emit; everything here is *how* a particular facility of the
hardware is expressed in SPIR-V. Each file is one facility, so the question "how does prosper
translate an image sample / an LDS atomic / a wave ballot" is answered by opening one file whose
name you can guess.

| file | the RDNA2 facility it emits for |
| --- | --- |
| `image` | **MIMG** — sample, fetch, gather, LOD query, resinfo, image atomics, and the coordinate/tap setup they share |
| `buffer` | **SMEM / MUBUF** — constant and storage buffer access, binding resolution, coherence marking |
| `lds_gds` | **DS** and **GDS** — shared and global data share, including their atomics |
| `wave` | subgroup and wave-level operations — ballot, votes, `mbcnt`, DPP, swizzle, lane and invocation ids |
| `shader_io` | **EXP** plus stage interface — exports, fragment interpolation, vertex and fragment inputs/outputs |
| `control_flow` | phi placement and patching, switches, barriers, `discard`, invocation guards |
| `numeric` | packing, unpacking, conversion, 64-bit arithmetic and the integer helpers the ALU emitters call |
| `guest_memory` | guest scratch, pointer relocation and descriptor capture — the parts that reach into guest address space |
| `declarations` | the SPIR-V declaration pool: types, constants, variables, capabilities |

## What is NOT here

`SpirvCompute` itself — the class, its fields and the small accessors — stays in
`../rdna2_to_spirv_internal.hpp`. These files hold its method *definitions*, which is why every one
of them is `SpirvCompute::` qualified and none declares anything.

The **decision** to emit lives in the `rdna2_emit_*` files. If you are asking "why did this
instruction produce nothing", that is a question for the emitters or the decoder, not for this
folder. If you are asking "what does prosper emit for `image_sample_lz_offset_2d`", it is here.

## Adding to it

A new method goes in the file for the facility it serves, not the file whose helpers it happens to
call. When a new facility arrives, it gets a new file rather than being added to the closest one —
these names are the only architecture documentation that cannot go stale, because the compiler
enforces the contents.

`tools/refactor/outline_methods.py` performs the move and routes by name; its `--group` patterns
are how a method lands here, and it **refuses** a method matching no group rather than filling a
default bucket. A file called `misc` would undo the point of the folder.
