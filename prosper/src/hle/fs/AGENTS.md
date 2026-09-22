# `fs` — the guest's filesystem and its save data

Two subjects that share a root: what the guest reads and writes through file descriptors, and the
save data the platform layers on top of it.

- `hle_file` — the POSIX and `sceKernel*` file surface, with `/app0` path translation. The largest
  thing here and still per-platform in one file; see the note below.
- `save_paths` — where a title's saves live on the host, per title id (#2734).
- `save_param` — the save's parameter block, the metadata a title attaches to a slot (#2786).
- `save_capacity` — a save's allocation (the blocks its creating mount asked for) and how much of it
  its files use, for `sceSaveDataGetMountInfo` (#3654). The accounting rule and how sure each part is
  are in its header; the records live in a dot-directory beside the saves, never inside one.
- `savedata` — libSceSaveData and libSceSaveDataDialog: mounts, transaction resources, the
  save-data memory API, the two umount entry points, and SetParam/GetParam. Split out of
  `hle/service/hle_service.cpp` in #3735, into the folder that already held the two helpers above.

## What belongs here

A Sony library whose subject is a file or a save. libScePlayGo is **not** here even though it
answers questions about installed content — it is about *distribution*, not storage, and it stays
with app content in `hle/service/`.

## The thing to know before editing `hle_file.cpp`

It is ~4,500 lines of which roughly a fifth is behind an inactive `#if` on any one platform, so an
AST tool reading it on Linux is answering about four fifths of the file. Splitting it per platform
(`*_linux.cpp` / `*_win.cpp` beside a shared header) is tracked separately; until then, check which
arm you are reading.

## Registration

Each library file owns a `register_<lib>_hle()` that `register_builtin_hle()` calls, declared in
`hle/dispatch/dispatch.hpp`. That keeps every handler `static` to the file implementing it.

**A test that calls `register_service_hle()` directly will not see these NIDs.** Two did, and both
had to be corrected when the split landed — `test_psml_unimplemented` and
`test_savedata_event_drain`. Prefer `register_builtin_hle()` in a new test unless you specifically
mean one library.
