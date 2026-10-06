---
kind: adr
status: proposed
date: 2026-10-05
---

# ADR 0003: Split guest semantics out of host/ into a guest/ layer

## Context

`src/host/` holds two different things: OS services ("how does this OS do X") and guest semantics
running on the host -- the System V / Microsoft x64 bridge (`host/abi`), the guest TLS block
(`host/tls`), guest fault interpretation (`host/fault`), guest memory query/copy/watch
(`host/memory`) and image mapping and execution (`host/image`). The second group needs
`hle/dispatch`, which is where most of today's baselined `host` -> `hle` include inversions come
from (`docs/architecture/ARCHITECTURE_TARGET_TREE.md` § Layer order).

Engines that implement guest-visible coprocessor or kernel behaviour have no layer of their own, so
they live inside whichever HLE library first needed them. APR is the recorded example: its command
buffer model, executor and completion dialects are spread over `hle/memory`, `hle/fs` and
`hle/kernel`. Fiber titles show the cost of not having one owner for scheduling state: three
defects of one shape, each assuming the host thread entering an HLE call is the one that returns
(#3615, #3638, #3623; `docs/games/UNCHARTED_STATUS.md`).

## Decision

1. Create `src/guest/` and move `host/{abi,tls,fault,memory,image}` into it with
   `tools/refactor/move_module.py`, one move-only PR (target-tree moves 4 and 5).
2. Insert `guest` into `LAYER_ORDER` between `host` and `gpu`; `host` becomes the OS seam only.
3. Move `ImportSlot` and the TLS layout type below `hle` so `loader` and `guest` stop including
   `hle/dispatch` (move 6).
4. Engines shared by several libraries live once in `guest/`; `hle/` keeps NID handlers, argument
   decoding and return codes (spec `HLE-1`, `LAY-4`). APR is the first extraction (move 12), after
   the seam migration of its platform calls.
5. Every engine ships a replay test that feeds a recorded command stream and asserts what the
   guest observes, on both hosts (target-tree Standard 3). `CONFIDENCE: LOW` on how practical
   this is for large streams: it has not been tried.

## Consequences

Most `host` -> `hle` inversions disappear by relocation rather than redesign. The include-path
rewrite touches every file that includes the moved headers, so the move PR conflicts with any
in-flight lane touching them; land it in a quiet window and announce it. Behaviour changes the move
enables come in later PRs, never in the move.

## Alternatives considered

- Keep one `host/` and enforce sub-folder rules: rejected, the layer order is the thing that keeps
  the two groups apart and it works at layer granularity.
- Put engines in `hle/` under a shared subfolder: rejected, `hle` is the top layer, so nothing
  below it (`gpu`) could reach an engine without an inversion.

## Approval

Requires the project owner's acceptance; `ARCHITECTURE_TARGET_TREE.md` and the charter both say
the `guest/` extraction is not authorized until then. Accepting it unblocks target-tree moves 4,
5, 6 and 12.
