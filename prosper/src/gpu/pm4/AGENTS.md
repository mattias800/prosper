# `pm4` — the guest command stream

Decodes the PM4 packet stream the guest submits, and maintains the register state it writes.

- `pm4_decode` — packet-level decode: types, opcodes, payload extents.
- `pm4_registers` — the register namespace and offsets the packets address.
- `command_processor` — walks a submission, applies register writes, and emits the draws and
  dispatches the rest of the stack consumes.

This is the **entry point of the whole stack**: everything downstream is a consequence of what is
decoded here, so a decode error does not look like a decode error — it looks like a missing draw, a
wrong extent, or a resource bound from the wrong address.

**A builder's dword count is an ABI contract with the guest's own reservations.** See
`docs/AGC_PACKET_SIZES.md` before changing any packet's size; getting it wrong desynchronises the
stream rather than failing locally.
