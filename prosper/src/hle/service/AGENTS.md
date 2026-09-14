# `src/hle/service` — the PS5 *system services*: what the console knows about itself

The libraries a title asks about its **environment** rather than about hardware: save data (the
`/savedata0` mount, its parameter block, SaveDataMemory, the transaction resources and the save
dialog), add-content and entitlements, user service and system service, PlayGo, NP and trophies,
application parameters, and the platform UI shims beside them. If the question a guest is asking is
*"what is installed here, who is signed in, what does this application own, where do its saves
live"*, the answer is written in this folder.

The boundary against the siblings is the **kind of question**, not the library's name. `src/hle/fs`
owns the filesystem itself — the mount table, path translation, the bytes on disk — and this folder
owns the *policy* on top of it: which mount point is real, when a save exists, what a title is
allowed to be told about one. `src/hle/kernel`, `graphics`, `audio`, `input`, `net`, `memory`,
`video` and `sync` each own their own device or primitive.

**One thing the layout does not say for itself:** `hle_service.cpp` is ~6,000 lines and is also the
place a good many *non*-service NIDs happen to be registered — Videodec, AvPlayer, Ime, Ssl, Share
among them — while their real implementations live in the sibling that owns them (the video backend
is `src/hle/video`, not here). So finding a NID registered in this file is not evidence that its
subject belongs to this folder, and a new handler with a real home elsewhere should be written
there rather than appended here. Treat that as debt to unwind incidentally, not as the pattern.

## The rule this folder exists to hold

**Every answer must be derivable from bytes that are actually on this machine.** That is the whole
character of the code here, and it cuts in both directions — which is the part that gets missed:

- **Never manufacture a positive answer.** "Did this person buy X?" is not derivable from anything
  local, and no amount of engineering makes it so. A handler that returns *owned* or *entitled*
  unconditionally to get a title past a check is the emulator performing a circumvention itself, no
  matter how the code is phrased. Record the cap and stop.
- **Never under-report what IS here, either.** A stub answering a local-inventory question with 0
  silently removes content the user has, and the title then behaves *correctly* on a wrong answer —
  so its own UI explains the problem in its own words and the defect reads as a licensing wall
  rather than as ours. That failure is as real as the first one; it just fails in the direction
  nobody files.

The test that separates the two is whether the bytes exist. `param.json`'s `contentId`,
`applicationCategoryType` and `applicationDrmType` state this application's SKU offline, so
reimplement that faithfully — the way Wine answers `GetVersionEx`. A purchase history does not
exist offline, so it is a cap, not a feature to derive.

**And answer the same question the same way through every library that exposes it.** Sony surfaces
SKU and add-content state through more than one library; a divergence between two paths is a defect
even when each looks defensible alone (`sceAppContentAppParamGetInt` said *full* while
`sceNpEntitlementAccessGetSkuFlag` fell to the return-0 stub, and one title asked through the NP
one — #1873). Put the derivation in one place both callers reach.

## The failure mode that keeps recurring here: SCE_OK by omission

The dispatcher's default for an unregistered NID is `return 0`, and **0 is `SCE_OK` for almost every
contract in this folder**. So *not implementing* a call is not neutral — it actively reports success
for an operation that never happened, and leaves every out-parameter holding whatever the guest's
own stack held. That has cost this project several multi-session investigations: an unregistered
`sceSaveDataTransferringMountPs4` was a four-session black screen (#2023); `SetParam`/`GetParam`
told a title its metadata was stored when nothing was (#2786); an unregistered font call let a title
divide by an untouched stack slot (#2951).

The same shape reaches registered handlers that answer without looking at their arguments. Before
adding or editing one, ask what it can say *no* to. A call that cannot fail is nearly always a call
that is not reading something it was given — `sceSaveDataGetMountInfo` reported a live mount's
capacity for a null mount point, "/savedata1", and an unmounted "/savedata0" alike, because it never
read the mount point at all (#3653). Validate before touching an out-parameter, and leave the
caller's buffer untouched on an error return: that is this folder's existing convention, followed by
Mount3's NOT_FOUND path and by `GetParam`'s refusals.

## Where the numbers come from

The `SAVE_DATA_ERR_*`-style constants and struct offsets in here are reverse-engineered, and each
carries a `CONFIDENCE: HIGH/MED/LOW` beside it saying how well. Take the labels seriously in both
directions — several are pinned by guest disassembly across independent titles (0x809F0018's
"still in flight" is corroborated by five call sites in four dumps) and several are inherited PS4
values nothing local can confirm. `tools/re/nid_gate_scan.py` answers the question that usually
matters before changing one: *do any local titles branch on this return value, and can they tell one
error code from another?* Run it before raising a confidence label on the strength of a guess.
