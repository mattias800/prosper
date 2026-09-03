# `src/hle/net` — Sony networking libraries

Reimplementations of the PS5 networking libraries that prosper answers itself. Today that is
**libSceHttp** (`hle_http.cpp`): the URI helper family and the library/template id lifecycle.

## The policy this folder exists to hold

**prosper has no network, and acquiring one is not the plan.** That makes the interesting question
per entry point *"is this answerable from local state?"* rather than *"can we connect?"* — and the
two halves of the answer are both obligations:

- **Anything computable offline is implemented, not stubbed.** The URI helpers are ordinary string
  code; there is no network anywhere in parsing or rebuilding a URI, so leaving them to the
  dispatcher is a false success rather than a limitation. Same for id lifecycles: a template really
  is allocated locally, so it gets a real id from a real table.
- **Anything that genuinely needs a network answer must FAIL, and fail loudly.** The dispatcher's
  unregistered default is `0`, which is `SCE_OK` — so an unregistered send path reports that a
  request nobody sent succeeded, and a response getter reports success while writing nothing to its
  out-parameters. That is the specific shape that crashes callers (#2894), and it is worse than a
  reported error, because the guest's own error handling never runs.

Registering a NID can therefore be worse than leaving it unregistered, and the deciding question is
what the guest does with the *value*. An **id**-returning contract must never answer `0` — that is a
plausible-looking handle the guest will carry into later calls — and must never answer a
zero-extended error either; sign-extend so `int32` and `int64` reads both see it as negative. An
**SCE_OK-or-error** contract keeps the 32-bit form the library returns in `eax`. Both conventions
live in `hle_http.cpp` with comments saying which is which.

## Where the evidence comes from

Contracts here are read off the shipped modules rather than guessed. `<DUMP_ROOT>/sprx/` carries
`libSceHttp.sprx` and `libSceHttp2.sprx` as **plain ELF** — nothing is decrypted to read them — so
argument shapes, per-component length caps, emission order and error constants can be derived
directly, and a claim in this folder should cite the offset it came from. Two facts worth not
re-deriving: **libSceHttp2's error facility is `0x817b____`, not v1's `0x8043____`** (the low code
bytes are shared, so the v1 constants are not reusable across the two), and on a connect failure the
library propagates the **raw libSceNet error**, encoded `0x80410100 | BSD errno`. See #2894.

## The boundary against its siblings

This folder is smaller than "prosper's networking" — several networking surfaces are answered
elsewhere, and looking for them here is the mistake to avoid:

- **`sceHttp2Init`, `sceNetCtlGetState` and the NetCtl/NP service surface live in
  `../service/hle_service.cpp`**, not here. That is why `libSceHttp2` looks absent from this folder
  while one of its entry points is already registered.
- Answers must not contradict each other across that boundary. `sceNetCtlGetState` already reports
  `SCE_NET_CTL_ERROR_NOT_CONNECTED`, so an HTTP layer that reported a successful request would be
  telling the same guest two incompatible things.

New Sony networking libraries that prosper answers itself belong here, one file per library.
