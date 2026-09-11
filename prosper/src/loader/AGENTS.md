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

The distinction is not a corner case. Across the 60 local dumps, **all 60** import
`__stack_chk_guard` and 53 import an unnamed `libSceLibcInternal` object; over the set the loader
actually links that is **417 unresolved OBJECT bindings across exactly four distinct NIDs**. Most
data imports resolve cross-module and never reach a slot at all — these four are the ones that do.

**Quote link-set figures here, not disk figures.** An earlier version of this paragraph said
`__stack_chk_guard` "appears in 596 shipped modules", taken from a recursive scan of every module
file present in every dump (808 modules). `boot_link_inputs` links roughly 303 of them, so that
figure described a population the linker never sees and overstated the bindings by about 2.4x. The
per-title counts were correct in both scopes, which is what made the binding count easy to repeat
unchecked.

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
