---
kind: adr
status: proposed
date: 2026-10-08
---

# ADR 0036: Link allowlisted local-compute system libraries from the user's own firmware dump

## Context

prosper provides its own implementation (HLE, `src/hle/`) of the Sony system libraries it supports,
one function at a time, from disassembly and live captures. That is the right default and this ADR
keeps it. It is also the largest standing cost in the project: a title's imports fall to the
dispatcher's return-0 default until somebody implements them, and a library that is pure computation
(a JSON parser, a character-set converter) is reimplemented from scratch although a working copy sits
in firmware the user has a licensed copy of.

**There is precedent for running real Sony code that came from the user's own files.** prosper already
links the genuine `libc.prx` from a title's `sce_module/` (`src/host/image/boot_program.cpp:268`) and an
optional `libSceNpCppWebApi.prx` (`:267`), loaded last so its `init_array` runs first (`:362`), and
`module_path_policy.hpp` permits a `libSce*` module only from that directory. What is new here is the
*source* (a system-software dump, not the title's own directory) and the *selection* (a committed
allowlist, not whatever the title bundles). It is nonetheless a departure from "reimplement published
interfaces from scratch, the way Wine reimplements Win32"; Wine's native-DLL override is the nearest
precedent, and whether to follow it is the owner's call (Decisions for the owner, 2).

### Plaintext, and what prosper will not do

prosper contains no decryption of any kind, and this ADR adds none. Firmware system modules are
distributed to consoles encrypted; the only input this ADR accepts is a module that is **already
plaintext**. How a user comes to hold a plaintext firmware dump happens outside prosper, and is
listed as a decision for the owner below, because the charter's plaintext measurement (eboots at
6.4-6.8 bits/byte against 8.000 for an untouched console original) was taken on game executables and
does not cover system software.

Measured on the dump used here (a user-supplied dump; 2026-10-08): all **569** `.sprx` files begin with
the ELF magic; whole-file Shannon entropy is min 0.21, median 4.45, max 6.71 bits/byte with none at or
above 7.5; the executable segment's is min 4.32, median 6.23, max 6.86. The same function on three
title eboots gives 6.03, 6.18 and 6.60. So this dump is plaintext by the same test the charter
applies to eboots. Nothing in `src/self/` currently refuses a module that is *not* plaintext: an
encrypted file would be parsed as garbage, which is why Decision 5 adds a gate.

### Measurements

Taken 2026-10-08 with `nid_census` in single-module mode, so every import was read by prosper's own
`Module::load`, and registration is prosper's live HLE registry **as built from `origin/main`
`debe081ed`** (the HLE-dependent figures below move with every HLE PR). The dump carries no version
file, so its firmware version is **not recorded**; it is identified by a fingerprint, the SHA-256 of
its sorted `"<relative path> <size>"` listing: `b76dd2cc2ef2f20af588698d5a247546c6b183b270023df2fc8a3604538f55a0`
(569 modules, 528,768,912 bytes, 537 distinct names -- the census stages one copy per name). The
scripts are ad hoc and not committed; migration step 1 lands them as a tool.

- All 537 distinct modules parse with `Module::load`. 180 are managed `.dll.sprx` assemblies (out of
  scope); 357 are native, of which 4 (`libkernel*`, `libSceLibcInternal*`) stay HLE, leaving **353
  candidate libraries exporting 47,042 symbols**.
- **No candidate links on today's HLE alone: 0 of 353.** The cause is concentrated: 869 distinct imports
  are exported by no firmware library and registered by no prosper handler. Of the 300 most blocking,
  155 are libc/C++ runtime (`_ZNSt8__sce_v2...`, `sceLibcMspace*`, `abort`, `fprintf`), 66 kernel/posix
  (`sceKernelGetCompiledSdkVersion`, `sceKernelGetAppInfo`, `mmap`, `ioctl`) and 79 other.
- Adding the top-N would resolve (cumulative): 2 -> 21 libs / 401 exports; 20 -> 34 / 857; 80 -> 54 /
  1,208; 300 -> 73 / 2,363; 600 -> 160 / 8,322; all 869 -> 353 / 47,042. The first two are a marker and a
  data symbol, and the linker binds a data import to a zero-filled slot rather than a handler
  (`src/loader/linker.cpp:155-159`), so that first step overstates the HLE work.
- **Resolvable is not useful.** Splitting by a mechanical rule (a library is *service/device-bound* if it
  imports from `libSceIpmi`, uses `ioctl`, is a renderer-core or store/licensing library (below), or
  depends on one that is), **71 libraries (4,759 exports) compute locally** -- Folly, FreeType, ICU,
  the font stack, Json, Json2, Xml, Jxr, Png/Jpeg/WebP, HarfBuzz, fontconfig, Rtc, Ult, Fiber -- and the
  rest are clients of a service or device. The rule is a heuristic: a library it calls local can still
  make a direct syscall.
- **Demand is much narrower than the pool.** Across four title dumps (Black Flag `PPSA28183`, two more
  from separate drives -- one UE4, one Unity -- and *Dreaming Sarah* `PPSA02929`), 724 imports are
  unregistered. **52 (7%) come from a local library; 135 (19%) from no firmware library at all; 537 (74%)
  from service/device/policy/renderer libraries.** Only **3 of the 71** local libraries are imported at
  all: `libSceJson2` (3 titles, 46 imports), `libSceJson` (3 titles, 17), `libSceCesCs` (1 title, 6). The
  other 68 -- including every font, image, XML and ICU library -- have no unregistered import in any of
  the four titles. By providing library the service-bound demand is renderer core 421, licensing 106,
  service-dependent 95, IPMI client 60, device 38 (attributions, so an import several libraries export
  counts for each).
- **Export collisions.** 0 of the 71 local libraries export a NID that `libkernel*` or
  `libSceLibcInternal*` exports. 7 export NIDs prosper registers today: `libSceJson2` (37), `libSceRtc`
  (23), `libSceJson` (17), `libSceSysmodule` (10), `libSceUlt` (5), `libSceFiber` (4), `libScePngDec`
  (1). That is a lower bound (registration was read off imports seen in the firmware).

So the honest size of Wave 1 is small: three libraries and about 70 imports across three titles. The
pool is the ceiling, the demand is what is worth doing, and the largest single bucket of unregistered
imports -- the renderer core -- is HLE work this ADR leaves exactly where it is.

## Decision

1. **LLE is an optional, per-library, allowlisted capability. HLE stays the default and the fallback.**
   prosper may link a *named* system library from the user's firmware dump when it is on a committed
   allowlist. With no firmware directory supplied nothing changes: the same HLE and return-0 behaviour as
   today, and the boot log names every allowlisted library it did not link.
2. **What is eligible.** A library enters the allowlist only when **all** hold: (a) the mechanical rule
   finds no IPMI-client, `ioctl` or exclusion-list dependency in its transitive closure; (b) it is not
   renderer-core, video-out or other graphics code; (c) a title in the corpus actually imports it; (d) its
   export set has an **empty intersection with the always-HLE NIDs** (below), checked by the census tool and
   asserted again by the linker at boot; (e) its file passes the plaintext gate (Decision 5). The census
   output justifying each entry is reviewed in the PR that adds it. The pilot is `libSceJson2`,
   `libSceJson` and `libSceCesCs`. A service-bound library becomes eligible only when the service it talks
   to exists in prosper at the IPMI boundary; that service is its own PR, and its client libraries are
   allowlisted in the same review.
3. **What stays HLE, always, and how that is enforced.** The always-HLE set is: `libkernel*` and
   `libSceLibcInternal*` (the syscall boundary); the graphics path (`libSceAgc*`, `libSceGnmDriver*`,
   `libSceVideoOut*` -- prosper's renderer core, which the census would otherwise call "local"); and the
   libraries that answer or mediate **ownership, entitlement, store or authorisation**: the four the charter
   names (`libSceAppContent`, `libSceNpEntitlementAccess`, `libSceGameUpdate`, `libSceAmpr`) plus
   `libSceNpCommerce` and `libSceNpAuth`. (`libSceAmpr` is on the charter's list because stand-ins for it
   exist, not because it answers ownership questions; the list is defined by function, so the exclusion does
   not depend on the heuristic.) Ownership queries are answered from the local inventory
   (`src/hle/service/hle_addcontent.cpp`); running Sony's implementation would be a different act from
   deriving the answer. **Enforcement:** the linker's export table is first-wins by NID alone
   (`src/loader/linker.cpp:124-136`), so an allowlisted library exporting a NID in the always-HLE set would
   silently shadow it for every module. The link step therefore **refuses** a firmware library whose exports
   intersect that set, naming the NIDs, and the existing alias report (#1635) is treated as an error for
   firmware libraries.
4. **Precedence.** `linker.cpp:153` binds an import to any linked module's export before it considers a
   handler, so linking a library replaces its HLE for every NID it exports; for the pilot that is 54
   registered handlers (`libSceJson2` 37, `libSceJson` 17), which is also what makes the differential
   oracle (Decision 7) meaningful. When a title's own `sce_module/` ships the same library, **the title's
   copy wins** -- it is what the title was built against -- and the firmware root is consulted only for an
   allowlisted library the title does not ship, so the outcome does not depend on link order.
5. **The path policy gains a second root and a plaintext gate, reject-by-default.** `module_path_policy` is
   dump-root-relative (`classify_module_path(dump_root, path)`), so this is a new entry point for a firmware
   root, not a list edit: it permits only the allowlisted file names, and **refuses, loudly and by name, any
   file that is not plaintext** -- no ELF magic (for example a SELF wrapper), a failed parse, or whole-file
   entropy at or above 7.5 bits/byte -- instead of attempting to load it. prosper decrypts nothing. The
   `fakelib/` rejection is unchanged. `tests/host/image/test_module_path_policy.cpp` gains arms, all on
   synthetic ELF fixtures so they run in CI: an allowlisted plaintext ELF is accepted; a non-allowlisted name,
   a random-byte file with the ELF magic, and a non-ELF file are refused; an export collision is refused;
   removing the guard reddens them. `tools/dump_hygiene.py` learns the same rule.
6. **The firmware directory is supplied by a command-line option, `--firmware-dir <path>`,** on the same
   footing as the positional game-dump argument -- no environment variable (so no new raw `getenv`), no
   config file, and nothing in the repository. It is a **guest-behaviour selector** in the charter's
   taxonomy: it decides whether the second root exists, so whether allowlisted libraries replace their HLE.
   It is **off by default** (no argument, no change). This ADR does not claim it can be deleted: the option
   names the *location of an input*, and what runs is fixed by the source-level allowlist and the plaintext
   gate, which stay unswitchable. Whether that classification satisfies the charter's "no environment
   variable to switch it off" for the policy, and whether the default is "on when the directory is
   supplied" or per-library opt-in, is for the owner (Decisions for the owner, 3).
7. **HLE-vs-LLE is an oracle.** Where both exist the same call can go through each and the results be
   compared: a differential test on pure functions that needs no console. A mismatch is a finding against
   the HLE until shown otherwise. Those tests need the user's firmware, so **CI reports them `(Skipped)`**,
   as it does for non-Messenger dumps today; a green CI run says the path-policy and plaintext-gate arms ran,
   not that the oracle did, and the PR that adds an oracle test says which it is.
8. **No Sony bytes enter the tree.** The allowlist is file names; the fingerprint above hashes a listing,
   not content. No spec rule is added.

## Consequences

- The HLE surface shrinks where it is cheapest to shrink, but measured demand says the first step is
  modest: three libraries and about 70 imports across three of four titles. The case for continuing past
  the pilot rests on titles that need more, which the census will show.
- Most of the unlock comes from the shared base (libc/C++ runtime, a few kernel queries), which helps
  HLE-only titles equally; the 80 most blocking functions are worth doing whatever happens to this ADR.
- Real libraries exercise prosper's libkernel and libc HLE harder than a game does: new bugs found against
  a known-good caller, and new bugs.
- A new prerequisite appears for the LLE path: the user's plaintext firmware dump, whose version may not
  match a title's SDK. HLE remains for anyone without one.
- Boot gains module loads (TLS, `init_array` order, `module_start`); the boot log must attribute addresses
  to a firmware library.
- Enforced by the path-policy and plaintext-gate tests above (which run in CI), the linker's collision
  refusal, a committed census tool, and review of every allowlist change.

## Alternatives considered

- **Stay HLE-only (status quo).** Correct and simple; keeps paying the per-function cost for libraries whose
  real implementation is available and pure. Retained as the default and the fallback.
- **Link all 353 libraries now.** Rejected as a first step, not as an end state. 281 are clients of a system
  service or device, and 84 import the IPMI client. With no service on the other side such a library blocks
  or errors on its first call, where today's HLE answers at once with a tailored value (the sign-in wait that
  stalled a title for good, #3784, is that failure class), so linking them regresses titles that work. The
  excluded libraries are excluded by the charter or because prosper's HLE is the translation layer for the
  hardware they talk to. The end state is the Switch-emulator model, real libraries over services
  implemented at the IPC boundary; it is reached one service at a time (Decision 2, Migration step 6),
  because prosper has no IPMI service side and the wire formats are not documented.
- **Vendor the open-source cores instead** (FreeType, ICU, HarfBuzz, fontconfig, libwebp, Brotli, Folly appear
  among the local libraries), under the charter's exception for permissively licensed standalone libraries.
  Viable where a library's export surface equals the upstream API, which is **not verified** here; the
  Sony-named wrappers (`libSceFont`, `libSceJson2`) would still need HLE. A candidate complement per
  library, not a replacement. Given measured demand, none of these has a user among four titles today.
- **Require a full firmware package.** Rejected: far more than the allowlist needs, and it changes who can
  run prosper at all.

## Migration order

1. Land the census as a committed tool (single-module `nid_census` over a firmware dump, the fixpoint, the
   LOCAL/service split, the always-HLE collision check, the plaintext/entropy report, and the dump
   fingerprint) with a pytest, so every number above is reproducible and tied to a prosper commit.
2. The shared base, by blocking count: the 20 then 80 most blocking libc/C++ runtime and kernel/posix
   functions. Useful on their own, and the precondition for step 4.
3. The path-policy firmware root, the plaintext gate, the collision refusal and `--firmware-dir`, with the
   synthetic-fixture tests, red without the change.
4. Pilot: `libSceJson2`, `libSceJson`, `libSceCesCs`. Success is Black Flag's Json imports resolving, an
   HLE-vs-LLE differential test on the parser (skipped in CI, said so), and an unchanged boot for a user
   without a firmware directory.
5. Wave 2 only where a title's demand appears; today no other local library has any.
6. Wave 3, per service: take the service a title needs (user service and save data are the likely first),
   implement it at the IPMI boundary from live traces, then allowlist the client libraries that talk to it.
7. Settle the default (on when `--firmware-dir` is supplied, or per-library opt-in) in an issue.

## Open questions

- **Firmware vs SDK version skew.** Whether to pin an allowlisted file by size or hash, and how a library
  from newer firmware behaves for a title built against an older SDK. The dump here has no version file.
- **Initialisation.** How a firmware library's `module_start`, TLS and `init_array` order interact with the
  boot order `boot_program.cpp` already fixes around `libc.prx`.
- **Whether the "local" heuristic is sound.** A library that reaches the kernel by a direct `syscall`
  instruction, not an `ioctl`/IPMI import, would be classified local; step 1's tool should look for that and
  the pilot libraries should be read for it first.
- **Does enabling a library change a title that already works?** By Decision 4 it replaces the HLE for every
  NID the library exports, so each addition needs the cross-title check the charter asks of shared changes.

## Decisions for the owner

1. **Provenance.** Is a user-supplied, already-plaintext firmware dump acceptable input? Game dumps are
   accepted on the same terms, but firmware is system software rather than the title the user bought, and
   the charter's plaintext measurement does not cover it. How such a dump is obtained is outside prosper
   and this ADR does not describe it.
2. **Project identity.** Running genuine Sony system code at runtime departs from "reimplement from scratch,
   the way Wine reimplements Win32". Wine's native-DLL override is the closest precedent; the title-bundled
   `libc.prx` is an existing, narrower one.
3. **The selector.** Is `--firmware-dir` an acceptable guest-behaviour selector given the policy's "no
   environment variable to switch it off", and should LLE be on whenever it is supplied or opt-in per library?
4. **The exclusion boundary.** Are the charter's four libraries, `libSceNpCommerce`/`libSceNpAuth`, the
   renderer core and the mechanical IPMI/`ioctl` rule enough, or should more be named?
5. **Oracle authority.** May a firmware library's output serve as an oracle that HLE is corrected against,
   given that it cannot run in CI?

## Approval

The project owner accepts or rejects this ADR. Acceptance unblocks the path-policy change and the pilot;
it does not authorise linking any library outside the allowlist, nor any decryption, and it does not touch
the always-HLE set, which stays HLE by the charter.
