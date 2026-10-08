---
kind: adr
status: proposed
date: 2026-10-08
---

# ADR 0036: Link allowlisted local-compute system libraries from the user's own firmware dump

## Context

prosper reimplements every Sony system library as HLE (`src/hle/`), one function at a time, from
disassembly and live captures. That is the right default and this ADR keeps it. It is also the
largest standing cost in the project: a title's imports fall to the dispatcher's return-0 default
until somebody implements them, and a library that is pure computation (a JSON parser, a font
rasteriser, a PNG decoder) is reimplemented from scratch even though the real, working code sits in
the same firmware the user already owns.

**There is precedent for running real Sony code that came from the user's own files.** prosper
already links the genuine `libc.prx` from a title's `sce_module/` (`src/host/image/boot_program.cpp:268`)
and an optional `libSceNpCppWebApi.prx` (`:267`), and `module_path_policy.hpp` permits a `libSce*`
module only from that directory. What is new here is the *source* (the system-software dump, not the
title's own directory) and the *selection* (a committed allowlist, not "whatever the title bundles").

**Measurements, 2026-10-08, on a user-supplied firmware dump (537 modules).** Taken with `nid_census`
in single-module mode, so every import was read by prosper's own `Module::load`, and the registration
state is prosper's live HLE registry. The scripts are ad hoc and not committed; migration step 1
lands them as a tool.

- All 537 modules parse with `Module::load`. 180 are managed `.dll.sprx` assemblies (out of scope);
  357 are native. Four of those (`libkernel*`, `libSceLibcInternal*`) are the syscall/libc boundary
  and stay HLE, leaving **353 candidate libraries exporting 47,042 symbols**.
- **No candidate links on today's HLE alone: 0 of 353.** The cause is concentrated, not diffuse. 869
  distinct imports are exported by no firmware library and registered by no prosper handler; the
  most-blocking are libc/C++ runtime (`_ZNSt8__sce_v2...` exceptions and mutexes, `sceLibcMspace*`,
  `abort`, `fprintf`), kernel/posix (`sceKernelGetCompiledSdkVersion`, `sceKernelGetAppInfo`, `mmap`,
  `ioctl`) and a few markers. Of the 300 most-blocking: 155 libc/C++ runtime, 66 kernel/posix, 79 other.
- Adding the top-N of those would resolve this many libraries (cumulative): 2 -> 21 libs / 401
  exports; 20 -> 34 / 857; 80 -> 54 / 1,208; 300 -> 73 / 2,363; 600 -> 160 / 8,322; all 869 -> 353 /
  47,042. The first two entries are a marker and a data symbol, and the linker binds a data import to
  a zero-filled slot rather than a handler (`src/loader/linker.cpp:155-159`), so that first step
  overstates the HLE work; the rest of the curve is real functions.
- **Resolvable is not the same as useful.** Splitting the 353 by a mechanical rule (a library is
  *service/device-bound* if it imports from `libSceIpmi`, uses `ioctl`, is on the licensing list below,
  or depends on a library that does), **72 libraries (5,005 exports) compute locally** -- Folly,
  FreeType, ICU, the font stack, Json, Json2, Xml, Jxr, Png/Jpeg/WebP, HarfBuzz, fontconfig, Rtc, Ult,
  Fiber -- and **281 (42,037 exports) are clients of a system service or device** (Np*, Http, Ssl,
  audio/video, pad, save data). The rule is a heuristic; a library it calls local can still make a
  direct syscall.
- **Black Flag** (`PPSA28183`): of 146 unregistered imports, **20 come from local libraries** (`libSceJson2`
  9, `libSceJson` 6, `libSceAgc` 5), **85 from service/device/policy libraries** (`libScePsml` 17,
  `libSceSsl` 7, `libSceTextToSpeech2` 7, `libSceNpEntitlementAccess` 6, ...) and **41 from no firmware
  library at all**, so they stay HLE whatever this ADR decides.

So the real prize is bounded: running real service-client libraries only moves the boundary, because
the service half (the shell, the PSN stack, the audio driver) does not exist off-console and would
still be HLE; the libraries that pay for themselves are the ones that compute locally.

## Decision

1. **LLE is an optional, per-library, allowlisted capability. HLE stays the default and the
   fallback.** prosper may link a *named* system library from the user's firmware dump when it is on a
   committed allowlist. With no firmware dump configured, nothing changes: the same HLE and the same
   return-0 behaviour as today, and the boot log names every allowlisted library it did not link.
2. **What is eligible.** A library enters the allowlist only when (a) the mechanical rule above finds no
   IPMI-client, `ioctl` or licensing dependency in its transitive closure, (b) it is not graphics,
   video-out or other renderer-core code, and (c) a title in the corpus actually imports it. The
   census output that justifies each entry is reviewed in the PR that adds it. The pilot is `libSceJson2`
   and `libSceJson`; the font, image and XML families follow by title demand.
3. **What stays HLE, always.** `libkernel*` and `libSceLibcInternal*` (the syscall boundary), the
   graphics path (`libSceAgc*`, `libSceGnmDriver*`, `libSceVideoOut*` -- the census calls `libSceAgc`
   "local" because it computes packets in user space, but it is prosper's renderer core), and the
   licensing libraries `libSceNpEntitlementAccess`, `libSceAppContent`, `libSceGameUpdate` and
   `libSceAmpr`. Those answer ownership questions from the local inventory
   (`src/hle/service/hle_addcontent.cpp`); running Sony's own implementation of them would be a
   different thing from deriving the answer, and the charter draws that line.
4. **Precedence is the linker's, and it is explicit.** `linker.cpp:153` binds an import to any linked
   module's export before it considers a handler, so linking a library replaces its HLE for every NID it
   exports, registered or not. Enabling a library is therefore a visible, per-library act, and the HLE
   implementation stays in the tree as the fallback and as the control arm of an A/B.
5. **The path policy gains a second root, reject-by-default, not switchable.** `module_path_policy`
   keeps `sce_module/` as is and permits the firmware root only for the allowlisted file names. There
   is no environment variable or flag; widening the allowlist is a source change with a reviewer. The
   `fakelib/` rejection is unchanged, and `tests/host/image/test_module_path_policy.cpp` gains arms that
   redden if the firmware root becomes a general one. `tools/dump_hygiene.py` learns the same rule.
6. **The firmware dump is the user's, supplied by configuration, never in the repository.** It is
   located the way the game dump is. prosper includes no Sony bytes, and the allowlist is file names, so
   nothing in the tree reproduces firmware content.
7. **HLE-vs-LLE is an oracle.** Where both exist, the same call can be made through each and the results
   compared; that is a differential test on pure functions that needs no console and no circular
   reasoning. A mismatch is a finding against the HLE until shown otherwise.
8. **No spec rule is added.** This ADR is a policy-and-loader change; a rule may follow once the pilot
   shows what an instrument can enforce.

## Consequences

- The HLE surface shrinks where it is cheapest to shrink: the local libraries (about 5,000 exports
  across 72) stop being reimplemented one function at a time, and correctness comes from running the
  real code.
- Most of the unlock comes from filling the shared base (libc/C++ runtime, a few kernel queries),
  which benefits HLE-only titles equally; the 80 most-blocking functions are worth doing whatever
  happens to this ADR.
- Real libraries expose prosper's own libkernel and libc HLE to code that exercises them harder than a
  game does. That is a feature (new bugs found against a known-good caller) and a cost (new bugs).
- A new prerequisite appears for the LLE path: the user's firmware dump, whose version may not match
  the title's SDK. HLE remains for anyone without one.
- Boot gains module loads (TLS, `init_array` order, `module_start`), and each linked library is one more
  thing a fault can land in; the boot log must attribute addresses to it.
- Enforced by the path-policy tests above, a committed tool that reproduces the census, and review of
  every allowlist change.

## Alternatives considered

- **Stay HLE-only (status quo).** Correct and simple, but it keeps paying the per-function cost for
  libraries whose real implementation is available and pure. Retained as the default and the fallback.
- **Link all 353 libraries.** Rejected: 281 are clients of a service or device that does not exist
  off-console, so linking them relocates the boundary without removing the HLE, and four are licensing
  libraries the charter requires prosper to answer itself.
- **Vendor the open-source cores instead** (FreeType, ICU, HarfBuzz, fontconfig, libwebp, Brotli, Folly
  appear among the local libraries), under the charter's exception for permissively licensed standalone
  libraries. Viable where a library's export surface equals the upstream API, which is **not verified**
  here; the Sony-named wrappers (`libSceFont`, `libSceJson2`) would still need HLE. A candidate
  complement per library, not a replacement.
- **Require a full firmware package.** Rejected: far more than the allowlist needs, and it changes who
  can run prosper at all.

## Migration order

1. Land the census as a committed tool (single-module `nid_census` over a firmware dump, the fixpoint
   and the LOCAL/service split) with a pytest, so the numbers above are reproducible.
2. The shared base, by blocking count: the 20 then 80 most-blocking libc/C++ runtime and kernel/posix
   functions. Useful on their own, and the precondition for step 4.
3. The path-policy second root and its tests, red without the change.
4. Pilot: `libSceJson2` and `libSceJson` behind explicit allowlist entries. Success is Black Flag's 15
   Json imports resolving, an HLE-vs-LLE differential test on the parser, and an unchanged boot for a
   user with no firmware dump.
5. Wave 2 by title demand: the font, image and XML families, each entry justified by its census output.
6. Settle the default (on when a firmware dump is configured, or opt-in) in an issue and delete any
   selector.

## Open questions

- **Firmware vs SDK version skew.** Whether to pin an allowlisted file by size or hash, and how a
  library from newer firmware behaves for a title built against an older SDK.
- **Initialisation.** How a firmware library's `module_start`, TLS and `init_array` order interact with
  the boot order `boot_program.cpp` already fixes for `libc.prx` (which is loaded last so its
  `init_array` runs first).
- **Whether the "local" heuristic is sound.** A library that reaches the kernel by a direct `syscall`
  instruction rather than an `ioctl`/IPMI import would be classified local; step 1's tool should look
  for that, and the pilot libraries should be read for it before they are allowlisted.
- **Does LLE of an allowlisted library change a title that already works?** By decision 4 it replaces
  the HLE for every NID the library exports, so each addition needs the cross-title check the charter
  already asks of shared changes.

## Approval

The project owner accepts or rejects this ADR. Acceptance unblocks the path-policy change and the
pilot; it does not authorise linking any library outside the allowlist, and it does not touch the
licensing libraries, which stay HLE by the charter.
