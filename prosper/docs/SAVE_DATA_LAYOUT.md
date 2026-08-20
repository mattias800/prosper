# Where prosper keeps save data

Status: current. Landed with [#2734](https://github.com/mattias800/prosper/issues/2734).

prosper serves two independent PS5 save APIs, and they are **separate stores** — confusing them has
cost real time before (`GAME_COMPAT_ORCHESTRATION.md` instrument trap 124).

| Guest API | Host layout | Root override |
| --- | --- | --- |
| `/savedata0` file mount (`sceSaveDataMount3`) | `<root>/<TITLE_ID>/<guest dirName>/…` | `PROSPER_SAVE0` |
| SaveDataMemory (`sceSaveDataMemorySet2` / `Get2`) | `<root>/<TITLE_ID>/savemem_<userId>_<slotId>.bin` | `PROSPER_SAVEDATA_DIR` |

`<TITLE_ID>` is the running application's `titleId` from `sce_sys/param.json`, validated as `PPSA`
plus five digits. It is derived in exactly one place — the single param.json parse in
`src/hle/service/hle_addcontent.cpp`, published as `AppParamDeclaration::title_id` — so the
add-content authorization check and the save namespace cannot disagree about which title is running.
An application with no usable declaration (an eboot-only dump has no `param.json`) gets the explicit
placeholder `_unknown-title`, which no valid title id can spell; that case is announced on stderr.

Default roots, when the override is unset:

| Platform | `/savedata0` | SaveDataMemory |
| --- | --- | --- |
| Linux / macOS | `$XDG_DATA_HOME/prosper/savedata0`, else `~/.local/share/prosper/savedata0` | `…/prosper/savedata-mem` |
| Windows | `%APPDATA%\prosper\savedata0` | `%APPDATA%\prosper\savedata-mem` |

Saves are user **data**, so the default follows the platform's per-user data location — the sibling
of the per-user *config* location `prosper-app` already uses for its settings file, and the same
shape the rest of the project uses for host paths (an environment override in front of a sensible
default). If neither variable that names such a location is set, prosper falls back to the historic
`/tmp` path and says so, because on the Linux development box that is a RAM-backed tmpfs with a
per-user quota shared by every concurrent agent, and a save that quietly does not survive a reboot is
worse than one that announces where it went.

## `PROSPER_SAVE0` and `PROSPER_SAVEDATA_DIR` are ROOTS

Both used to name the flat directory saves were written directly into. They now name the root that
the per-title directories are created **under**, and there is deliberately **no way to opt out of the
title component**.

The alternative — keeping them as an absolute override that bypasses namespacing — was rejected
because roughly fifteen documented run recipes in `docs/` and `scripts/` set these variables, and
those recipes are what agents and users copy. An escape-hatch spelling would have left the collision
reachable in precisely the configurations people actually use, which is the whole defect.

`tools/snapshot`'s `savedata_policy: "fresh"` keeps working unchanged. It now means only "start from
an empty save state" rather than doubling as collision avoidance.

## Why this exists: two titles, one save

Before #2734 both roots were a single flat host directory shared by every title on the machine, and
the key below the root was chosen by the **guest**:

* the `/savedata0` per-save subdirectory is the guest's own `dirName`, and UE4 titles routinely pick
  generic ones — `OptionSettings`, `GameUserSettingsini`, `Profile`, `Global`, `Inputini`;
* SaveDataMemory is keyed by `(userId, slotId)`, which is `1, 0` for essentially every Unity title.

Neither key is unique across titles, so two installed titles read and overwrote each other's saves.
One developer machine had accumulated **34 slot directories from many titles in one directory**.

*Little Nightmares III* (`PPSA05143`) read an `OptionSettings` slot a **different** Unreal title had
written, correctly concluded it was not its own save format, **deleted it**, and held on a modal
reading `Your options save has corrupted and has been deleted` that an unattended run cannot dismiss.
That is real data loss caused by prosper and reported to the player in the game's own words, so it
reads as the title's defect rather than ours. The A/B is in
[#2734](https://github.com/mattias800/prosper/issues/2734): identical binary and route, the only
difference being a private empty save root, reaches the title screen.

It is also the charter's local-inventory rule pointed the other way. The save that got read was
genuinely present; it just belonged to a different application.

## Existing saves are NOT migrated

Save data written before #2734 stays exactly where it is, in the flat root, untouched and invisible
to every title. On the first save operation of a process, prosper prints a one-time report naming the
directory, the count, an example entry, and the move to make by hand.

**This is deliberate, and both of the obvious automatic migrations are worse than doing nothing.**

## Ruled out

| Rejected approach | Why it is wrong |
| --- | --- |
| **Move-on-first-use** — the first time title *T* mounts slot *S*, move `<root>/<S>` to `<root>/<T>/<S>`. | First-boot-wins attribution: whichever title happens to launch first *takes* `OptionSettings`, possibly another title's. Destructive and irreversible — the same data-loss bug wearing a fix's clothes, and strictly worse than leaving the data alone. |
| **Read-fallback** — if `<root>/<T>/<S>` is absent, read `<root>/<S>`. | This *is* #2734. It is exactly what let *Little Nightmares III* read a foreign `OptionSettings` and delete it. A fix that re-creates the reported failure on the first launch after upgrading is not a fix. |
| **Attribute by directory name** — map known slot names back to titles. | Only a minority of the observed names carry any title identity (`TheMessenger`, `SifuProfile`, `SyberiaAutoSave`, `GreakMainData`, `PathlessConfig`). The majority are generic (`OptionSettings`, `Profile`, `Global`, `Settings`, `Inputini`, `GameUserSettingsini`, `SavedSettings`, `Saves`, `storage`, `app`, `net`, `MetaData`, …) and are precisely the ones that collided. A partial map would silently mis-attribute the rest. |
| **Attribute by mtime** | The writer's mtime is overwritten by any later reader-then-writer, and the issue reporter's own investigation lost the evidence exactly this way before they could look. |
| **Attribute from the dump directory name** (`PPSA05143-app0`) | A second source of truth for the title id, derived from the developer's private directory layout rather than from the application. `param.json` is the declaration; the directory name is a coincidence. |

Nothing on disk records which title wrote a flat slot directory. Since prosper cannot know, it does
not guess — it preserves and reports. To restore an old save, move it into the owning title's
directory yourself:

```text
<root>/OptionSettings   ->   <root>/<TITLE_ID>/OptionSettings
```

## Related

* [#2734](https://github.com/mattias800/prosper/issues/2734) — the defect, the A/B, and the frame.
* `src/hle/fs/save_paths.hpp` — the contract, in code.
* `tests/hle/test_savedata_title_namespace.cpp` — the regression guard: two titles over one guest
  save name, in every arm.
* `GAME_COMPAT_ORCHESTRATION.md` instrument trap 124 — the two roots are different stores, and an
  arm that isolates only one of them is not isolated.
