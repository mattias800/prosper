# Cross-engine check — Unreal Engine 4 title (PPSA17942)

**Date:** 2026-07-05. **Purpose:** a second, unrelated engine is the real test of whether prosper is a
genuine compat layer or a Unity-specific shim. A UE4 4.27 title (`PPSA17942`, uses the CRIWARE UE4
plugin; C++/no IL2CPP) was added alongside the Unity/IL2CPP first target (`PPSA24651`, The Messenger).
This records what the check found — **without** committing to a full UE4 bring-up (a separate project).

## Result summary

| Layer | Generalizes? | Evidence |
|---|---|---|
| **SELF/ELF loader** (`src/self/module.cpp`) | ✅ **Yes** | `imgdump` parses the 183 MB UE4 eboot cleanly → 189 MB flat image, `entry=0x80`, sane vaddrs. The SELF header is structurally identical to The Messenger's from offset 4 on (`0001 0112 0101 0000 6005 100…`); only the first-4-byte magic differs (`5414f5ee` vs `4f153d1d`) yet the loader handles it. The eboot's `needed_files` (DT_NEEDED) are extracted normally: `libSceAgc`, `libSceAgcCore`, `libSceAcm`, … |
| **Boot harness / module loading** (`tools/boot_trace`) | ⚠️ **No — Unity-hardcoded** | `boot_trace.cpp:40-42` hardcodes exactly three module paths — `Media/Modules/Il2cppUserAssemblies.prx`, `Media/Modules/PS5Util.prx`, `sce_module/libc.prx` — which are Messenger-specific. Booting UE4 fails immediately: `load …/Media/Modules/Il2cppUserAssemblies.prx: cannot open file`. |
| **Shader tooling** (`shader_histo` / recompiler) | ⚠️ **N/A for UE4** | `shader_histo` finds **0** shaders: Unity/IL2CPP embeds RDNA2 shader ELFs (`EM_AMDGPU`) in the eboot rodata; **UE4 ships shaders in its `.pak`/asset files** (`doll/content/`), a different container the eboot scan can't see. |
| **Runtime HLE** (libc/libkernel/services) | ❓ **Untested** | Blocked behind the module-loading gap; UE4's C++ runtime (no IL2CPP) will exercise a different API set. |

## The "Proton vs. shim" verdict

The **core is genuinely general** — the SELF/ELF loader parses an unrelated engine's executable and its
Sony-library dependencies with no title-specific code. What's *not* yet general is **scaffolding around
it**: the boot harness hardcodes the Messenger's module list, and the shader tooling assumes Unity's
"shaders-embedded-in-eboot" packaging. Those are the natural next generalizations, not deep rewrites.

## Concrete, tractable next steps (each independent; none is "full UE4 bring-up")

1. **Generic module discovery (highest-value, tractable).** The loader *already* exposes the eboot's
   `needed_files` / `needed_mods` (`module.hpp:76-78`). Replace `boot_trace`'s hardcoded 3-module list
   with: iterate `needed_files`, resolve each to a path (search `sce_module/`, `Media/Modules/`,
   app-root), assign module bases sequentially (instead of the hardcoded `IL2CPP`/`PS5UTIL`/`LIBC`
   constants), then link. **Risk:** this path is load-bearing for the *working* Messenger boot — do it
   as a focused, regression-tested change (assert The Messenger still boots to the GfxDevice wall) and
   keep the base-address assignment deterministic. Benefits both titles + any future one.
2. **UE4 shader extraction (larger).** UE4 stores compiled shaders in its shader library / `.pak`
   assets, not the eboot. Testing the recompiler on UE4 shaders needs a UE4 shader-library/pak reader to
   pull the RDNA2 blobs; only then does `shader_histo`/`recompile_coverage` apply.
3. **UE4 runtime bring-up (a project).** Once modules load, UE4's C++ runtime will hit a different set of
   unimplemented Sony APIs; this is a second-title bring-up comparable in size to the Unity one.

## Practical note

Unlike The Messenger, most PS5 titles' SELF segments are encrypted (need console keys) and cannot load
at all. This UE4 title *does* load (its header parses), which is the fortunate precondition that makes
it a viable second target — the module-discovery step (1) is the gate to actually exercising it.
