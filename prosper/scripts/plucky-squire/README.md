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
