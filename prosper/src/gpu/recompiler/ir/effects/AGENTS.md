# `ir/effects` — instruction facts by encoding family

The per-opcode tables included by `../../rdna2_instruction_effects.cpp` live here, split by
encoding family. They remain one translation unit with internal linkage. Each `.inc` is a
source fragment, included after the shared row constructors inside the anonymous namespace;
it is not a standalone header.

These are architectural reads, writes and widths derived from AMD RDNA2 ISA 70648 and checked
with llvm-mc. They describe instructions, never definedness, descriptor provenance, admission or
emission. Consumers ask the public effects interface; they keep no private opcode lists.

`scalar_result` and `memory` are step 8a facts with no production consumer. Legacy query fields
and their Known/Valu domains stay unchanged. Memory layouts describe control-dependent spans:
a zero fixed count is not a claim of no access when the layout or data rule supplies the count.
Resolve those rules with the instruction's flags before using a span. LLVM-only encodings not
specified by ISA 70648 have Unmodeled rows, never invented zero effects or Reserved rows.
