# `pm4` — the guest command stream

Decodes the PM4 packet stream the guest submits, and maintains the register state it writes.

- `pm4_decode` — packet-level decode: types, opcodes, payload extents.
- `pm4_registers` — the register namespace and offsets the packets address.
- `command_processor` — walks a submission, applies register writes, and emits the draws and
  dispatches the rest of the stack consumes.
- `pm4_memory` — integer `ATOMIC_MEM` effects and raw-span `COND_EXEC` folding. Unknown producer
  authority refuses before reading a conditional predicate.
- `reg_watch_selector` — pure `parse_reg_watch` selector parsing; live reporting stays in the
  command processor.
- `ordered_wait_diagnostics` — names for memory-effect classes and checked overlay outcomes;
  classification stays in the command processor.
- `vgt_shader_stages.hpp` — decodes VGT_SHADER_STAGES_EN's routing fields (GS_EN, PRIMGEN_EN, the
  wave-32 bits) and names a draw's NGG shape (`merged-gs`, `ngg-vs`, `legacy`).
- `cond_indirect_buffer` — what a Jump-style segment does when the command processor reaches it:
  sceAgcCbBranch's conditional packet resolved to its then/else Jump, and the fail-visible guards
  (size, depth, mapped) a segment passes before it runs. Split out because `command_processor.cpp`
  is past the ratchet's line cap.
- `pending_write_snapshot` — nonblocking observations of that processor's completion queue for
  bounded performance capture; it does not alter visibility, execute writes or wait for them.
- `wait_regmem_sample` — retains the exact value used by a wait predicate so later diagnostic
  reads cannot be presented as that decision's sample; queue policy stays in the processor.

This is the **entry point of the whole stack**: everything downstream is a consequence of what is
decoded here, so a decode error does not look like a decode error — it looks like a missing draw, a
wrong extent, or a resource bound from the wrong address.

**A builder's dword count is an ABI contract with the guest's own reservations.** See
`docs/gpu/AGC_PACKET_SIZES.md` before changing any packet's size; getting it wrong desynchronises the
stream rather than failing locally.
