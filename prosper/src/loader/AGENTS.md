# src/loader — one address space out of many modules

This folder decides **what gets loaded, where it lands, and what every import resolves to**. It takes
a list of `LinkInput`s (the eboot plus its dependent and optional PRXs), builds each module's image
at a fixed base, and produces the global export table plus one `ImportSlot` per unresolved import —
another module's export (a real cross-module call), an HLE code stub, or a slot in the writable
import-data aperture. It is host-agnostic: mapping the images and installing the apertures is
`src/host/image`'s job, and the HLE handlers behind the stub slots are `src/hle`'s.

The boundary worth stating: **`src/host/image/boot_program.cpp` decides which modules are
*candidates*; this folder decides what those candidates *mean* together.** Policy about the
candidate list that is more than a filename — anything needing imports, exports or link order to
answer — belongs here as a pure function over parsed data, so it can be tested without a game dump.
`support_modules.hpp` is the worked example.

**An unresolved import goes to one of TWO apertures, and the ELF symbol type is what chooses.** A
function import is bound to an executable stub that calls its handler; a `STT_OBJECT` import is
bound to a writable, non-executable, zero-filled slot, because it names a *variable* the guest
dereferences and may store through. Until #3529 there was one aperture and every import went to it,
so a data import made the guest read prosper's own machine code as its value — and, the serious
half, made a guest store rewrite an executable stub, silently and with arbitrary delay. This is why
`Symbol`/`Import` carry `elf_type` at all: nothing else in an undefined symbol entry distinguishes
the two cases (shndx 0, value 0 and size 0 for both).

The distinction is not a corner case, and these figures are the tool's own — `nid_census
--data-only` over the 60 local dump roots, whose default scope is already the loader's link set:

```text
scope: 2284 distinct imported NIDs over 338 module(s) read, 0 unreadable
424 DATA binding(s) (ELF STT_OBJECT) unresolved by any sibling module, over 60 of 60 input(s);
a further 1566 were satisfied cross-module

f7uOxY9mM1U  __stack_chk_guard   libkernel            60 of 60 titles
djxxOmW6-aw  __progname          libkernel            60 of 60 titles
ZT4ODD2Ts9o  (unnamed)           libSceLibcInternal   51 titles
GAtITrgxKDE  (unnamed)           libSceNet             1 title
```

Most data imports resolve cross-module and never reach a slot — 1566 of them. These four are the ones
that do, and every title in the corpus has at least two.

**Quote link-set figures, not disk figures, and take them from the tool.** Two earlier revisions of
this paragraph got it wrong the same way, the second while correcting the first. "596 shipped
modules" came from passing *module paths* to `nid_census` individually, where each `.prx` becomes its
own "title" and cross-module exclusion degenerates — a recursive population of 808 modules the linker
never links. The replacement then called the link set "roughly 303 modules", which is a **binding**
count written where a module count goes. Both times the per-title counts were right, which is
precisely what let the number underneath them travel unchecked.

Two properties of the link that surprise people, both load-bearing:

- **The global export table is first-definition-wins and silent.** Two modules exporting the same
  NID both link, both initialize, and every import aliases to whichever came first in the list. That
  is why `LinkInput::skip_on_export_collision` exists (a title shipping release *and* logging builds
  of the same library), and why deduplicating on filename would not work — the names genuinely
  differ.
- **Linking a module runs its `module_start`, which is guest code.** Adding a module to the
  candidate list is therefore not a free "make its exports available"; it can block the boot
  forever. `LinkInput::only_if_imported` exists because a module preloaded for one title deadlocked
  an unrelated title that merely shipped the same file. **Changing that list changes every title
  that ships the file** — census the dumps before and after, and write the number down; the first
  use of the flag turned out to affect two titles where the author had checked one.

Init functions run in **reverse list order**, so a dependency must be placed *after* its dependent —
`libc.prx` goes last precisely so it initializes first. Get this wrong and the symptom is a module
touching another module's uninitialized state, arbitrarily far from the ordering that caused it.
