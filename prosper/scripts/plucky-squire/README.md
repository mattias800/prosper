# The Plucky Squire routes

Run the realtime fresh-save route from the repository root with:

```sh
scripts/plucky-squire/run-first-gameplay.sh /path/to/PPSA15319-app0
```

The launcher renders every submit at native scale and drives
`reach-first-gameplay.pad`. It prints the new run directory before starting. Set
`PROSPER_ARTIFACT_ROOT` to choose the parent directory, or pass a second argument
that names a new, nonexistent run directory.

Plucky can use two independent PS5 save paths. `PROSPER_SAVE0` backs files mounted
under guest `/savedata0`; `PROSPER_SAVEDATA_DIR` backs SaveDataMemory slot blobs.
They are cached independently on first use, so setting one does not configure the
other. The launcher always creates separate fresh `save0/` and `savedata/` roots
before the guest starts. Reuse an old run only for an intentional continuation;
use a new run when comparing progression or renderer behavior.

The longer `reach-first-gameplay.pad` route is intended for synchronous realtime
rendering. `capture-first-gameplay.pad` is the accelerated trace/capture variant;
it is not a timing-equivalent visual route.

## `reach-gameplay-reads.pad` — the same ladder on the guest's own axis

This title's guest tick rate spans **147x** between its menus (~25 polls/s) and the chapter-one intro
cutscene (~0.17 polls/s), so no single wall-clock cadence serves both: the 5 s pulses that pace the
menus are worth ~125 guest polls at the title screen and one or two once the 3D world is up.

`reach-gameplay-reads.pad` writes the same ladder with **pad-read anchors** (`p<start>-<end>:button`),
which are counted in guest polls rather than seconds. It is therefore invariant to host speed and to
the sampling cadence — the same file drives a default run and a `PROSPER_RENDER_EVERY` run with no
retiming. Cross is the title's own `UI_Select` *and* `Jump` (read out of its
`InputSettingsKeyMappings` save), so surplus presses are harmless in gameplay.

Prefer it over `reach-first-gameplay.pad` for anything past the menus. Background and the throughput
wall it exists to survive: `prosper/docs/PLUCKY_SQUIRE_STATUS.md`, issues #2839 and #2837.
