# `scripts/sonic-frontiers-PPSA03831/`

Input routes for **Sonic Frontiers (`PPSA03831`)**, Hedgehog Engine 2 / Needle. One `.pad` file per
destination, driven with `PROSPER_PAD_SCRIPT=@<path>`.

The directory name carries the title id on purpose. `scripts/sonic/` is a *different* title
(*Sonic Origins*, `PPSA05325`) and `scripts/sonic-crossworlds/` a third; a franchise directory is a
convention here, never a contract, so check the id before reusing a route.

## What belongs here

Routes, and only routes — no logs, captures or manifests (those go under `$HOME`, never `/tmp` or
`/var/tmp`). A route file is expected to carry a header explaining *why* each window exists, because
on this title almost every window encodes a measured fact about the guest that is invisible from the
route text alone: how many pages the boot notice queue has, which buttons advance it, where the
main-menu cursor starts. A route without that header is unmaintainable the first time the guest's
behaviour shifts.

## Two facts that bind every route in here

- **Anchor on flips (`f`), not seconds.** This title runs about 30 flips/s on a CPU-only
  `boot_trace` arm and about 3 flips/s on a live 3840x2160 renderer arm — a tenfold spread — so a
  wall-clock route that works on one is meaningless on the other. `f` is the guest's own presented-
  frame counter and transfers between them unchanged.
- **Save state selects the route.** `PROSPER_SAVE0` is the mounted `/savedata0` root, and Frontiers
  reads what an earlier run left there. The boot notice queue in particular is only the measured
  twelve pages on an isolated first-boot area. Pass `PROSPER_SAVE0=<private dir>` per run.

Route findings, the measured A/Bs behind them and the `## Ruled out` table live in
`prosper/docs/SONIC_FRONTIERS_STATUS.md`; the tracker is issue #1891.
