# src/loader — one address space out of many modules

This folder decides **what gets loaded, where it lands, and what every import resolves to**. It takes
a list of `LinkInput`s (the eboot plus its dependent and optional PRXs), builds each module's image
at a fixed base, and produces the global export table plus one `ImportSlot` per unresolved import —
either another module's export (a real cross-module call) or an HLE stub slot. It is host-agnostic:
mapping the images and installing the stubs is `src/host/image`'s job, and the HLE handlers behind
the slots are `src/hle`'s.

The boundary worth stating: **`src/host/image/boot_program.cpp` decides which modules are
*candidates*; this folder decides what those candidates *mean* together.** Policy about the
candidate list that is more than a filename — anything needing imports, exports or link order to
answer — belongs here as a pure function over parsed data, so it can be tested without a game dump.
`support_modules.hpp` is the worked example.

Two properties of the link that surprise people, both load-bearing:

- **The global export table is first-definition-wins and silent.** Two modules exporting the same
  NID both link, both initialize, and every import aliases to whichever came first in the list. That
  is why `LinkInput::skip_on_export_collision` exists (a title shipping release *and* logging builds
  of the same library), and why deduplicating on filename would not work — the names genuinely
  differ.
- **Linking a module runs its `module_start`, which is guest code.** Adding a module to the
  candidate list is therefore not a free "make its exports available"; it can block the boot
  forever. `LinkInput::only_if_imported` exists because a module preloaded for one title deadlocked
  an unrelated title that merely shipped the same file.

Init functions run in **reverse list order**, so a dependency must be placed *after* its dependent —
`libc.prx` goes last precisely so it initializes first. Get this wrong and the symptom is a module
touching another module's uninitialized state, arbitrarily far from the ordering that caused it.
