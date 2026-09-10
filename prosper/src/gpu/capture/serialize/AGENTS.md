# `capture/serialize` — the `.prgcap` / `.prgbundle` file format

Turning a collected capture into bytes and back. Its sibling `capture/` decides *what* to record;
this folder decides only how that is written down and read again, and the boundary is the version
contract: a change here is a change to a format that already-written captures are in.

- `capture_codecs` — the `Writer`/`Reader` byte cursors, and one `write_*`/`read_*` pair per field
  group (pipeline, colour target, scissor, logic op, resource, table). A header because both entry
  points use them, and the only place a field's on-disk shape is spelled.
- `capture_serialize` — `serialize_gpu_capture`.
- `capture_deserialize` — `deserialize_gpu_capture`, plus the legacy-alias restoration older
  versions need.

**The two entry points are one contract read from opposite ends, so they change together.** A field
written and not read is silent data loss; read and not written is a parse that succeeds on garbage.
Neither shows up as a build failure, and both survive a round-trip test that only exercises the
version it was written for.

**A version bump is append-only.** `kVersion` lives in `capture/gpu_capture_internal.hpp` with the
per-version notes; older captures must keep loading, which is why the reader carries restoration
paths the writer has no counterpart for. Adding a field means writing it at the end and reading it
behind a version check — never reordering, because the cursors are positional and a reordered field
mis-parses every prior capture without erroring.

The size limits (`kMax*`, also in `gpu_capture_internal.hpp`) are the bound on untrusted input:
these files parse bytes that came off disk, and every count read from a header is checked against
one before it is used to size an allocation.
