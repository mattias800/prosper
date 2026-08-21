# src/host/image — turning a game dump into a running process image

This folder owns the step between "a directory of files on disk" and "guest code mapped and
entered". It decides **which files in a dump are modules**, in what order they link, where each one
is based, and how the host actually maps them. Everything here is dump-layout knowledge and
per-platform mapping; nothing here knows about Sony API semantics (that is `src/hle/`) or about the
relocatable-image format itself (that is `src/self/` and `src/loader/`).

The boundary against `src/loader/` is worth stating because the two are easy to confuse:
`src/loader/` links a set of modules it is *handed* and knows nothing about dumps. This folder is
what produces that set — it is where `<dump>/Media/Plugins/…` becomes a `LinkInput`.

## What lives here

- **`boot_program.*`** — the boot sequence. `boot_link_inputs()` is the single definition of the
  loader's real link set, and is deliberately public so tools (`nid_census`) report on the same set
  the loader uses rather than re-deriving it and silently disagreeing (#2199). Note it *prints* while
  it works; that output is part of the loader's behaviour.
- **`module_path_policy.*`** — the reject-by-default allowlist of dump locations a module may be
  linked from. Read its header before touching module discovery: some dumps ship third-party
  replacements of Sony libraries, and this is what keeps them from being linked and shadowing
  prosper's own implementations.
- **`exec_image_{linux,win}.cpp`** behind `exec_image.hpp` — the per-platform mapping substrate.
  Platform divergence belongs *here*, not in `boot_program.cpp`.
- **`runtime_module_load.*`** — the guest asking for a module after boot, as opposed to the fixed
  preload list. True runtime PRX loading is not implemented (#639), which is why the preload list
  carries so many optional entries: they exist so a first P/Invoke resolves instead of hanging.

## Things a newcomer learns the hard way

**Dump layout is not consistent between titles, and casing is not either.** The same Unity module
ships as `Il2cppUserAssemblies.prx` and `Il2CppUserAssemblies.prx` depending on the title, so paths
are case-corrected against the real directory entry (`resolve_host_path_case`, #1006). On a
case-insensitive host the wrong casing "works" and hides the bug; on Linux it silently drops the
module and the guest null-jumps.

**Order in the link list is reverse init order.** A module's init function runs *after* those listed
below it, so a dependency goes later in the list, not earlier. Several comments in
`boot_program.cpp` restate this at each site because getting it backwards produces a
plausible-looking list that initializes FMOD's studio layer before its core.

**Adding a module to the discovery path is a security-relevant change, not just a convenience.**
`discover_extra_plugin_modules()` links whatever a dump ships in one directory (#1609). Widening
that — recursing, accepting another extension, scanning another directory — changes which
third-party code prosper will execute on behalf of the guest. `module_path_policy` is the guard, and
`tests/host/image/test_module_path_policy.cpp` is what makes it real; extend both together.
