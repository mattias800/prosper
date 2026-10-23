# `spirv` — what the recompiler emits, one file per RDNA2 facility

The back half of RDNA2→SPIR-V. `../rdna2_decode` turns guest bytes into instructions and
`../rdna2_emit_*` decides *when* to emit; everything here is *how* a particular facility of the
hardware is expressed in SPIR-V. Each file is one facility, so the question "how does prosper
translate an image sample / an LDS atomic / a wave ballot" is answered by opening one file whose
name you can guess.

| file | the RDNA2 facility it emits for |
| --- | --- |
| `image` / `texture_types` | **MIMG** — sampled-image types and texture declarations, sample, fetch, gather, LOD query, resinfo, image atomics, and shared coordinate/tap setup |
| `storage_image_access` | storage-image declarations, reads and predicated writes |
| `image_compare_manual` | manual 2D comparison sampling, including address selection and nearest/linear tap mixing |
| `buffer` | **SMEM / MUBUF** — constant and storage buffer access, binding resolution, coherence marking |
| `buffer_access` | buffer element selection and ordinary load/store emission |
| `buffer_declarations` | **SMEM / MUBUF** — storage-buffer Block types, table-contract checks and descriptor binding declarations |
| `lds_gds` / `lds_gds_access` | **DS** and **GDS** — shared and global data share, including atomics, storage declarations, predicated stores and scratch-wave append |
| `wave` | subgroup and wave-level operations — ballot, votes, `mbcnt`, DPP, swizzle, lane and invocation ids |
| `wave_lane_ops` | native subgroup row shift/XOR and 32-lane BPERMUTE emission |
| `shader_io` | **EXP** plus stage interface — exports, fragment interpolation, vertex and fragment inputs/outputs |
| `geometry_setup` | interpolation geometry entry, interface declarations and input loading |
| `geometry_vertices` | interpolation geometry parameter planes, rectangle selection and vertex emission |
| `stage_entry` | fragment and vertex entry declarations and function setup |
| `compute_entry` | compute module entry, storage interfaces and guest invocation coordinates |
| `control_flow` | phi placement and patching, switches, barriers, `discard`, invocation guards |
| `numeric` | packing, unpacking, conversion, 64-bit arithmetic and the integer helpers the ALU emitters call |
| `guest_memory` | guest scratch, pointer relocation and descriptor capture — the parts that reach into guest address space |
| `declarations` | the SPIR-V declaration pool: types, constants, variables, capabilities |

## What is NOT here

`SpirvCompute` itself — the class, its fields and small accessors — is exposed by
`../rdna2_to_spirv_internal.hpp` through the adjacent `rdna2_spirv_compute*.inl` class sections.
The sections preserve member order and are included only at that class scope. These source files
hold method *definitions*, each `SpirvCompute::` qualified.
Direct section includes redirect through the umbrella; declaration/member scope guards keep
the original payload at its original class site.

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
