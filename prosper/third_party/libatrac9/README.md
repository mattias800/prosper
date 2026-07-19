# LibAtrac9 (vendored)

ATRAC9 audio decoder, vendored **verbatim** from **LibAtrac9** by Alex Barney
(https://github.com/Thealexbarney/LibAtrac9), the `C/src` tree.

- **License:** MIT (see `LICENSE`). Copyright (c) 2018 Alex Barney.
- **Why vendored, not re-derived:** ATRAC9 is a large, self-contained, well-defined codec (not a Sony
  HLE interface). LibAtrac9 is permissively licensed and is itself a clean-room reverse-engineering of
  the format — it contains no Sony code, firmware, or keys. Vendoring it with attribution is a
  deliberate project-owner decision (a documented exception to the charter's "re-derive external
  implementations" rule, which targets other emulators' HLE, not general codec libraries).
- **Prosper glue lives elsewhere:** the RIFF/config parsing and the bridge into the NGS2 sampler voice
  path are prosper's own code (`src/hle/...`). This directory is the unmodified upstream decoder.

Do not edit files under `src/` — keep them byte-identical to upstream so the vendored copy is auditable
and updatable. Prosper-specific behavior belongs in the glue layer.
