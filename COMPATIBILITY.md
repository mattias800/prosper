# Game compatibility

prosper is an experimental compatibility layer, not a general-purpose game runner. The results below
describe specific, user-supplied PS5 dumps tested primarily on Linux. A milestone means that the
documented route is reproducible; it does **not** mean the entire game is playable or free of bugs.
Different title revisions may behave differently.

Last updated: 2026-07-24

## Summary

| Title | Title ID | Engine | Tested milestone |
| --- | --- | --- | --- |
| *The Messenger* | `PPSA24651` | Unity / IL2CPP | ✅ Complete first level renders at native 1920×1080 |
| *Dead Cells* | `PPSA15552` | Custom | ✅ Controllable Prisoners' Quarters scene renders in full color |
| *Blasphemous 2* | `PPSA13579` | Unity | ✅ Opening route reaches and renders the first playable room |
| *Evergate* | `PPSA01885` | Unity | ✅ Reaches and renders the first tutorial-room gameplay |
| Terminator (2D)&nbsp;¹ | `PPSA25872` | Unity / IL2CPP | ✅ Main menu and attract-mode gameplay reached (user-verified) |
| *Blue Prince* | `PPSA25009` | Unity | 🚧 Opening gameplay scene ("Day One" at the manor) renders; known lighting/texture defects |
| *Grand Theft Auto V* | `PPSA04263` | RAGE | 🚧 Title and first-run/language menus render; STORY/ONLINE main menu behind loading not yet reached |
| *Dragon Quest VII Reimagined* | `PPSA17942` | Unreal Engine 4 | 🚧 Interactive title and save-slot flow; later content load is blocked |
| Additional Unity/IL2CPP target | `PPSA02664` | Unity / IL2CPP | 🔬 Exercised, with no published gameplay milestone |

¹ Exact retail game name pending confirmation.

## The Messenger — `PPSA24651`

<p align="center">
  <img src="assets/screenshots/messenger-title.png" width="380" alt="The Messenger — title screen">
  <img src="assets/screenshots/messenger.png" width="380" alt="The Messenger — first level gameplay">
</p>

The primary Unity 2022 / IL2CPP target boots through the engine, accepts scripted gamepad input, and
renders the intro, title, menus, save list, dialogue, player, terrain, water, structures, and
foreground composition through the complete first level at native 1920×1080. A full-resolution
scripted sequence was checked against PS5 hardware.

Reusable input routes are documented in
[`prosper/scripts/messenger/README.md`](prosper/scripts/messenger/README.md).

## Dead Cells — `PPSA15552`

<p align="center">
  <img src="assets/screenshots/dead-cells-title.png" width="380" alt="Dead Cells — main menu">
  <img src="assets/screenshots/dead-cells.png" width="380" alt="Dead Cells — Prisoners' Quarters gameplay">
</p>

A deterministic input route passes the splash and menus and reaches the controllable Prisoners'
Quarters scene. The current renderer produces the colored atlas, lighting, player, terrain, effects,
tutorial prompt, and HUD.

Graphics and compute execute in retained PM4 order (#584), guest compute writes invalidate
overlapping depth-cache state (#611), and the remaining lighting loops recompile through a narrowly
proved wave-uniform form (#615). The final grayscale-world defect was MRT selection: shaders export
MRT3 through MRT0, while the single-attachment backend previously used the first export. MRT0 now
feeds color attachment 0 (#626). Capture v8 preserves the complete color/depth checkpoint and
source-image oracle for fast standalone regression replay (#569).

See the exact route, current state, and regression workflow in
[`prosper/docs/DEAD_CELLS_STATUS.md`](prosper/docs/DEAD_CELLS_STATUS.md).

## Blasphemous 2 — `PPSA13579`

<p align="center">
  <img src="assets/screenshots/blasphemous2-title.png" width="380" alt="Blasphemous 2 — opening cinematic">
  <img src="assets/screenshots/blasphemous2.png" width="380" alt="Blasphemous 2 — first playable room">
</p>

The FMOD plugin path, AGC marker contract, two-pass HTTP URI parser, and guest-thread return boundary
are implemented. A poll-safe scripted route traverses the EULA and opening cinematic into a complete
native 1920×1080 first room with the player, HUD, world layers, lighting, and interaction UI.

The last black-world defect was PS5 primitive type 7: it is a RectList used by transparent fullscreen
clears, while prosper treated it as three points. The observed procedural form now runs its fourth
corner and renders as a Vulkan triangle strip (#654).

The reproducible route and capture recipe are in
[`prosper/scripts/blasphemous2/README.md`](prosper/scripts/blasphemous2/README.md).

## Evergate — `PPSA01885`

<p align="center">
  <img src="assets/screenshots/evergate-title.png" width="380" alt="Evergate — title screen">
  <img src="assets/screenshots/evergate.png" width="380" alt="Evergate — first tutorial room gameplay">
</p>

This Stone Lantern 2D puzzle-platformer boots through its intro and title, navigates the main menu with
scripted input, and reaches the first tutorial room: the masked spirit *Ki* on snow-capped platforms
among glowing bulb-plants, with the *"HOLD [X] TO JUMP HIGHER"* prompt. A crash when a keyboard key was
pressed at the menu — the guest's IME event handler ran on the host thread-local (`%fs`) base and read a
garbage per-thread allocator pointer — was fixed by restoring the caller's guest `%fs` around the callback
(#1288). A deterministic content-guard snapshot (`evergate-gameplay`) locks the tutorial-room milestone
against regressions.

The reproducible route is in
[`prosper/scripts/evergate/README.md`](prosper/scripts/evergate/README.md).

## Blue Prince — `PPSA25009`

This Unity target boots to its menu, renders the full opening cinematic, completes the first-room
load, and renders the "Day One" opening gameplay scene: the Mt. Holly manor exterior, gardens, and
the player character on the approach path, captured over a 600-second scripted route with zero
render stalls. Two loading-phase performance defects were removed to get here — unbounded
register-file growth from incrementally patched register arrays (#1266) and per-draw-reference
full-buffer content hashing (#1269). Known open defects: lit surfaces render blown-out (#1271) and
small foliage textures render as pixel noise (#1272), so no verified-visuals milestone is claimed.
Boot remains intermittent (#1178).

## Dragon Quest VII Reimagined — `PPSA17942`

This Unreal Engine 4 target reaches an interactive title screen, navigates the save-slot menu, and
starts the subsequent content-load flow. Honest offline and signed-out service behavior removed the
earlier permanent loading-screen gate. The next documented blocker is a MallocBinned3 heap-corruption
failure during menu-driven content loading, so no gameplay milestone is claimed here.

The investigation and current blocker are documented in
[`prosper/docs/DOLL_LOADING_PROGRESSION.md`](prosper/docs/DOLL_LOADING_PROGRESSION.md).

## Grand Theft Auto V — `PPSA04263`

Rockstar's RAGE engine boots through the intro and reaches the title and the first-run/language-selection
menus, which render cleanly (owner-confirmed). Reaching this required a series of service and ABI fixes
spanning the async-compute submit contract, the APR file/write primitives, and guest-`%fs` restoration on
fault-skip. The STORY/ONLINE main menu behind the title loading spinner is not yet reached (#1160), and the
in-game loading path (the legal-notice text pass) is a documented blocker, so no gameplay milestone is
claimed.

## Terminator (2D) — `PPSA25872`

This 2D action title reaches its main menu and attract-mode gameplay, verified and recorded by the
project owner, with a menu-reach snapshot guard in place. The title ID is confirmed; the exact retail
name is pending confirmation.

## Requirements and scope

- Game files, keys, and copyrighted Sony code are not included. You must supply your own
  legally obtained dump.
- prosper currently requires unencrypted module segments and does not contain console decryption
  keys or circumvent Sony cryptography.
- Linux is the primary runtime target. Windows support is secondary and does not currently imply
  equivalent title validation.
- An unlisted title has no published compatibility result; the table is not an allowlist.
