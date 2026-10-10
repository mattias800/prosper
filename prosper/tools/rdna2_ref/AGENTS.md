# Independent RDNA2 Reference Interpreter (tools/rdna2_ref)

## Purpose and Independence Charter
This directory contains an **independent reference interpreter and decoder** for the AMD RDNA 2 instruction set architecture (focusing on the scalar ALU, control flow, EXEC mask, scalar memory, and integer VALU lane-op subsets).

### Hard Independence Rules
1. **Source of Truth**: Instruction semantics and encoding formats are written strictly from the **AMD "RDNA 2" Instruction Set Architecture Reference Guide (document 70648)**. No semantics or decoder tables are taken from `prosper/src/gpu/recompiler/` or other emulators.
2. **Recompiler Isolation**: Contributors must not consult `prosper/src/gpu/recompiler/` when writing or extending semantics or decoder tables. The recompiler may only be inspected after reference implementation verification to analyze differential mismatches.
3. **Third Independent Reference**: Decoding tables and instruction boundaries are cross-checked against LLVM's disassembler (`llvm-mc` / `llvm-objdump -d` with `-triple=amdgcn-amd-amdhsa -mcpu=gfx1030`). Disagreements are settled by consulting document 70648.
4. **Honest Coverage & No Skip Policy**: Any instruction not modeled in the subset stops execution and reports `unsupported:<mnemonic>`. Instructions are never silently skipped.
5. **Corpora Are Local-Only**: Shader dumps, binaries, firmware images, and title-derived shaders are local-only and must never be committed to git, nor may any hashes, game names, or host drive paths be checked in.

## Modules
- `decode.py`: Independent RDNA2 instruction decoder and complete opcode tables from ISA chapter 13.
- `machine.py`: Wave state (Wave64/Wave32, SGPRs, VGPRs, EXEC, VCC, SCC, memory) and instruction semantics.
- `llvm_check.py`: Decode cross-check harness using the LLVM AMDGPU disassembler (native `llvm-mc`/`llvm-objdump` on PATH, else WSL Ubuntu).
- `corpus.py`: Parsers and program carvers for local firmware and census shader stores.
- `cli.py`: Command-line tool for disassembling, running, and generating coverage/cross-check reports.
