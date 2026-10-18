# `ir/effects` — instruction facts by encoding family

The per-opcode tables included by `../../rdna2_instruction_effects.cpp` live here, split by
encoding family. They remain one translation unit with internal linkage. Each `.inc` is a
source fragment, included after the shared row constructors inside the anonymous namespace;
it is not a standalone header.

These are architectural reads, writes and widths derived from AMD RDNA2 ISA 70648 and checked
with llvm-mc. They describe instructions, never definedness, descriptor provenance, admission or
emission. Consumers ask the public effects interface; they keep no private opcode lists.
