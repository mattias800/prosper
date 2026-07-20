# Blasphemous 2 Routes

Reusable `PROSPER_PAD_SCRIPT` routes for Blasphemous 2 (`PPSA13579`). Run tools
from `prosper/` so relative script paths resolve.

## Loading a save (preferred for gameplay measurement)

`load-save-first-station.pad` boots, leaves the title, opens "Pilgrimage", and loads an existing
save at the first Prie Dieu, reaching real gameplay in under two minutes. Prefer it over
`reach-first-gameplay.pad` for anything that measures or guards GAMEPLAY: the exploratory route
needs roughly 380 s and every press is a guess against cinematic and tutorial pacing, so a build
that loads at a different speed silently desynchronises the whole route and measures the wrong
scene.

It requires save data to exist. A reference copy of the first-station save is kept in
`saves/savedata0-backup-2026-07-20/` (`slot1/segmentSlot` plus `slot0/segmentGlobal`); restore it
into the `PROSPER_SAVE0` root (default `/tmp/prosper-savedata0`) before running if that root has
been cleared. Note the save root is SHARED across titles, not per-game.

The route presses Cross once per five seconds rather than at three fixed offsets. The wall time at
which each screen becomes ready swings widely with the rendering policy -- the same route reaches
the title within seconds under `prosper-app` but needs about a minute under full-resolution
`screenshot` capture -- so it repeats instead of guessing. Surplus presses land in gameplay as
jumps, which is harmless and keeps both arms of an A/B identical.

## Current Route

`reach-first-gameplay.pad` is an exploratory Cross route anchored to the
first pad poll. It passes the studio logos, title, and EULA. With the local URI
parser from #642, it also passes the title's telemetry calls and loads gameplay
scenes and assets. The normal-return pthread/TLS cleanup crash is fixed by #644,
and native-resolution captures render the opening cinematic and first gameplay. Its one-second holds
are intentionally longer than normal taps: synchronous native-resolution rendering
can reduce this title to two or three pad polls per second, so short wall-clock
windows are not reliable. Set `PROSPER_PAD_SCRIPT_LOG=1` to record the scripted
states that were actually observed by the game.
After 200 seconds the route switches from menu Cross presses to the mandatory
bile-flask tutorial's Triangle/Circle exchange. It repeats the close/action pair
because the help page becomes ready at different wall times under sparse versus
full-resolution rendering, including a late fallback through 302 seconds. The
route then holds Right and acknowledges the movement tutorial with long Down and
Down+Cross windows before continuing right with periodic jumps. The reviewed
evidence window is 380-479 seconds, after those recovery and movement inputs.
The awakening glyph is the PlayStation Triangle symbol; Cross alone only jumps,
while omitting Circle leaves the title parked on the help page.

The first-room world used to remain black behind valid HUD pixels because the
transparent UI-target clear uses PS5 primitive type 7. It is a RectList: the guest
submits three procedural vertices and hardware supplies the fourth rectangle
corner. Treating 7 as an unknown point list cleared only three pixels; treating it
as an ordinary three-vertex triangle exposed exactly half the screen. The renderer
now invokes procedural vertex 3 and uses a four-vertex triangle strip (#654). This
is deliberately limited to the observed non-indexed, no-vertex-buffer form; a
future vertex-buffer-backed RectList still needs general post-VS corner synthesis.

Capture one full-resolution PNG per second with:

```bash
PROSPER_GUEST_FS=1 \
PROSPER_GUEST_ARGS=-force-gfx-direct \
PROSPER_PAD_SCRIPT=@scripts/blasphemous2/reach-first-gameplay.pad \
./build-linux/screenshot /path/PPSA13579-app0 \
  --render-every 30 --render-every-for-seconds 230 \
  --seconds 1 --count 480 --timeout 570 --out shots
```

At full resolution, rendering every draw-carrying submit reduces this route to
roughly two or three game frames per second. The validated Blasphemous 2 recipe
samples every 30th complete submit during the long opening while still writing one
normal 1920x1080 PNG per second. After 230 seconds it renders every submit so
temporal world layers are rebuilt before the first playable room. The
manifest records the renderer policy and reports source and pixel progression
separately. The route-aware guard retains moving 1920x1080 first-area gameplay
after the late tutorial fallback; the historical #654 validation produced 420
requested PNGs with 315 pixel-distinct samples.

For the routine renderer regression gate, run:

```bash
python3 tools/snapshot/snapshot.py check blasphemous2-gameplay
```

That guard starts from an isolated fresh save, replays this route, and evaluates
multiple changing frames only after the first-playable-room movement begins. It
uses a coarse content threshold rather than exact hashes so subtle pixel changes
do not fail the check. Any new or changed threshold must first be generated with
`snapshot.py verify blasphemous2-gameplay` and accepted only after every retained
image from both runs is inspected; see `tools/snapshot/README.md`.

The title's optional `libfmodstudio.prx` and `libfmod.prx` must be prelinked as
described by #638/#640. A run that stops before the first studio logo has not
reached the #641 marker fix; a run that faults at guest `eboot+0x11f79d0` has not
picked up the #642 URI parser.

## Title-screen route (audio observation)

`title-screen-idle.pad` presses Cross once to dismiss the intro, then sends **no further input** so the
game rests on the title screen and its ambience/music soundtrack can start (it begins a few seconds
after the title appears). The `reach-first-gameplay.pad` route deliberately presses Cross every three
seconds and drives straight past this state, so the title soundtrack never plays under it — use the
idle route whenever you are observing title/menu audio.

Audio is verifiable without a human listening: `PROSPER_AUDIO_DUMP=<path>` records the final mixed
output, so a run can be asserted on level/coverage.

```bash
PROSPER_GUEST_FS=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_RENDER=1 \
PROSPER_VULKAN_LIB=libvulkan.so.1 PROSPER_AUDIO_DUMP=/tmp/t.raw \
PROSPER_PAD_SCRIPT=@scripts/blasphemous2/title-screen-idle.pad \
  ./build-linux/prosper-app --dump <PPSA13579-app0>
# measure per-second peak/coverage of /tmp/t.raw.port16.raw
```

A healthy title screen holds a sustained non-zero level (coverage >80%). As of 2026-07-20 it does not:
frame rate starves FMOD's per-frame update so streaming voices cannot be fed, leaving long silences
(#1080, blocked on the performance issue #1082).
