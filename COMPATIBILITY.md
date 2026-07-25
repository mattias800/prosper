# Game compatibility

prosper is an experimental compatibility layer, not a general-purpose game runner. The results below
describe specific, user-supplied PS5 dumps tested primarily on Linux. A milestone means that the
documented route is reproducible; it does **not** mean the entire game is playable or free of bugs.
Different title revisions may behave differently.

Last updated: 2026-07-25

## Summary

| Title | Title ID | Engine | Tested milestone |
| --- | --- | --- | --- |
| *The Messenger* | `PPSA24651` | Unity / IL2CPP | ✅ Complete first level renders at native 1920×1080 |
| *Dead Cells* | `PPSA15552` | Custom | ✅ Controllable Prisoners' Quarters scene renders in full color |
| *Blasphemous 2* | `PPSA13579` | Unity | ✅ Opening route reaches and renders the first playable room |
| *Evergate* | `PPSA01885` | Unity | ✅ Reaches and renders the first tutorial-room gameplay |
| *GRIS* | `PPSA09804` | Unity / IL2CPP | ✅ Native 1920×1080 title reached; opening route and audio verified |
| *Space Adventure Cobra — The Awakening* | `PPSA17337` | Unity / IL2CPP | ✅ Native 1920×1080 title and audio verified |
| *Sonic Origins*&nbsp;² | `PPSA05325` | Hedgehog Engine | 🔬 Frontend loop reached; supplied update-only dump lacks its base title assets |
| Terminator (2D)&nbsp;¹ | `PPSA25872` | Unity / IL2CPP | ✅ Main menu and attract-mode gameplay reached (user-verified) |
| *Blue Prince* | `PPSA25009` | Unity | 🚧 Opening gameplay scene ("Day One" at the manor) renders; known lighting/texture defects |
| *Grand Theft Auto V* | `PPSA04263` | RAGE | 🚧 Title and STORY/ONLINE main menu render; known UI and composition defects remain |
| *Dragon Quest VII Reimagined* | `PPSA17942` | Unreal Engine 4 | 🚧 Interactive title and save-slot flow; later content load is blocked |
| Additional Unity/IL2CPP target | `PPSA02664` | Unity / IL2CPP | 🔬 Exercised, with no published gameplay milestone |

¹ Exact retail game name pending confirmation.

² No compatibility milestone is claimed for the incomplete Sonic dump. A merged base+update image is
required before its title, gameplay, or audio can be evaluated. The guest also consumes an authentic
Sonic 1 activity launch, but still needs the same base UI assets before entering the classic runtime.

## The Messenger — `PPSA24651`

<p align="center">
  <img src="assets/screenshots/messenger-title.png" alt="The Messenger — title screen">
</p>
<p align="center">
  <img src="assets/screenshots/messenger.png" alt="The Messenger — first level gameplay">
</p>

The primary Unity 2022 / IL2CPP target boots through the engine, accepts scripted gamepad input, and
renders the intro, title, menus, save list, dialogue, player, terrain, water, structures, and
foreground composition through the complete first level at native 1920×1080. A full-resolution
scripted sequence was checked against PS5 hardware.

Reusable input routes are documented in
[`prosper/scripts/messenger/README.md`](prosper/scripts/messenger/README.md).

## Dead Cells — `PPSA15552`

<p align="center">
  <img src="assets/screenshots/dead-cells-title.png" alt="Dead Cells — main menu">
</p>
<p align="center">
  <img src="assets/screenshots/dead-cells.png" alt="Dead Cells — Prisoners' Quarters gameplay">
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
  <img src="assets/screenshots/blasphemous2-title.png" alt="Blasphemous 2 — title screen">
</p>
<p align="center">
  <img src="assets/screenshots/blasphemous2.png" alt="Blasphemous 2 — first playable room">
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
  <img src="assets/screenshots/evergate-title.png" alt="Evergate — title screen">
</p>
<p align="center">
  <img src="assets/screenshots/evergate.png" alt="Evergate — first tutorial room gameplay">
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

## GRIS — `PPSA09804`

<p align="center">
  <img src="prosper/docs/screenshots/issue-1356-gris-title.png" alt="GRIS — New Game title screen">
</p>

The Unity/IL2CPP title reaches its animated **NEW GAME** screen at native 1920×1080 and accepts the
scripted default selection into the opening sequence. Its Wwise path produces sustained non-zero
PCM; the objective repetition check reports clean audio (`rms=0.1800`, no duplicated grains).

## Space Adventure Cobra — The Awakening — `PPSA17337`

<p align="center">
  <img src="prosper/docs/screenshots/issue-1356-space-adventure-cobra-title.png" alt="Space Adventure Cobra — The Awakening title screen">
</p>

This Unity/IL2CPP title boots through its opening flow and renders the complete title composition at
native 1920×1080. Its captured output also passes the objective repetition check (`rms=0.0436`, no
duplicated grains). Cobra imports an SDK-revision alias of `sceAgcCreateInterpolantMapping`; routing
that alias to the real builder initializes all 32 advertised Cx records instead of exposing stale
stack entries as register writes. The command processor therefore preserves every valid register
write and rejects only offsets outside its register window.

## Sonic Origins — `PPSA05325`

The supplied 02.002.000 directory is an update image targeting 02.001.000, rather than a merged
base+update app. Live file tracing shows that its only unresolved startup requests are
`raw/ui/ui_startup.pac` and `raw/ui/rpl_texture/ui_title_nocopy.dds`; both are absent from the dump.
The game otherwise initializes its renderer, connected pad, CRI sound banks, and AudioOut2 pump, but
correctly remains in a black startup loop and emits silence without those title assets. No screenshot,
audio, or compatibility success is claimed until a complete dump is available.

The update declares PS5 `launchActivity` support and contains the classic RSDK files. An exact
`TITLE_SONIC_1_CLASSIC` Game Intent is received by the guest and its `activityId` property is consumed,
but Sonic still requests both missing UI files before opening `raw/retro/Sonic1u.rsdk`. This rules out
the platform activity route as a way around the incomplete content while preserving truthful default
no-intent behavior.

Routes, capture commands, audio evidence, and the Sonic audit are recorded in
[`prosper/docs/GRIS_SONIC_COBRA_BRINGUP.md`](prosper/docs/GRIS_SONIC_COBRA_BRINGUP.md).

## Blue Prince — `PPSA25009`

<p align="center">
  <img src="assets/screenshots/blue-prince-title.png" alt="Blue Prince — title screen">
</p>

This Unity target boots to its menu, renders the full opening cinematic, completes the first-room
load, and renders the "Day One" opening gameplay scene: the Mt. Holly manor exterior, gardens, and
the player character on the approach path, captured over a 600-second scripted route with zero
render stalls. Two loading-phase performance defects were removed to get here — unbounded
register-file growth from incrementally patched register arrays (#1266) and per-draw-reference
full-buffer content hashing (#1269). Known open defects: lit surfaces render blown-out (#1271) and
small foliage textures render as pixel noise (#1272), so no verified-visuals milestone is claimed.
Boot remains intermittent (#1178).

## Dragon Quest VII Reimagined — `PPSA17942`

<p align="center">
  <img src="assets/screenshots/dragon-quest-vii-title.png" alt="Dragon Quest VII Reimagined — title screen">
</p>

This Unreal Engine 4 target renders its interactive title screen at native 3840×2160, navigates the
save-slot menu, and starts the subsequent content-load flow. The current title capture still exposes a
localization key in place of the localized logo text. Honest offline and signed-out service behavior
removed the earlier permanent loading-screen gate. The next documented blocker is a MallocBinned3
heap-corruption failure during menu-driven content loading, so no gameplay milestone is claimed here.

The investigation and current blocker are documented in
[`prosper/docs/DOLL_LOADING_PROGRESSION.md`](prosper/docs/DOLL_LOADING_PROGRESSION.md).

## Grand Theft Auto V — `PPSA04263`

<p align="center">
  <img src="assets/screenshots/gta5-title.png" alt="Grand Theft Auto V — title screen with a known missing Continue-button icon">
</p>
<p align="center">
  <img src="assets/screenshots/gta5-main-menu.png" alt="Grand Theft Auto V — STORY and ONLINE main menu with known rendering defects">
</p>

Rockstar's RAGE engine boots through the intro and reaches the title and the first-run/language-selection
menus, then continues to the STORY/ONLINE main menu. Reaching this required a series of service and ABI
fixes spanning the async-compute submit contract, the APR file/write primitives, and guest-`%fs`
restoration on fault-skip. These direct Linux `prosper-app` captures document the current renderer state,
not visual correctness: the title omits the X button icon beside **Continue**, and the main menu has
substantial UI and composition glitches. The later in-game loading path remains blocked at the legal-notice
text pass, so no gameplay milestone is claimed.

## Terminator (2D) — `PPSA25872`

<p align="center">
  <img src="assets/screenshots/terminator-title.png" alt="Terminator 2D: NO FATE — main menu">
</p>
<p align="center">
  <img src="assets/screenshots/terminator.png" alt="Terminator 2D: NO FATE — attract-mode gameplay">
</p>

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
