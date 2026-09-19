# `np` — the online account, network and trophy libraries

One honest, consistent answer to "is this console online and signed in?", which is **no**.

- `np` — libSceNpManager (sign-in state and its callbacks), libSceNpTrophy2, libSceNetCtl,
  libSceNpUniversalDataSystem, libSceShare / libSceGameLiveStreaming. Split out of
  `hle/service/hle_service.cpp` in #3735.

The libraries are together because they answer one question between them and must not disagree: a
title that reads SIGNED_OUT from NpManager and then gets a connected state from NetCtl takes a path
no real console produces. NetCtl is here rather than in `hle/net/` for the same reason — it shares
the guest-callback delivery discipline with the Np state callbacks, and `hle/net/` is about HTTP.

## What is deliberately NOT here

- **libSceNpEntitlementAccess and libSceGameUpdate** stay with app content in `hle/service/`. They
  answer whether add-on content is owned, from the **same local inventory** `hle_addcontent.cpp`
  reads. The charter requires the same question to be answered the same way through every library
  exposing it — #1873 is the case where `sceAppContentAppParamGetInt` and
  `sceNpEntitlementAccessGetSkuFlag` disagreed and the answer depended on which library the guest
  asked through. Keeping them in one file is what stops that reappearing.
- **sceNpGameIntent** stays with the system service. The intent is *delivered* through the
  system-service event queue — `s_sysservice_receiveevent` reads `g_gameintent_*` — so the two are
  one mechanism, not two libraries that happen to share a prefix.

Both boundaries are about behaviour, not convenience: each was measured as real coupling before it
was drawn.

## The guest-callback discipline

Several entry points here invoke a guest callback, which on Linux must run on the **guest `%fs`**.
That is what the `PROSPER_ASM_TRAMPOLINE` pairs and `CbGuestFsScope` exist for, and it is why those
registrations sit inside `#ifndef _WIN32` with a different handler on the other arm. A registration
moved out of its conditional would leave one platform with the NID unbound.

## Registration

`register_np_hle()`, declared in `hle/dispatch/dispatch.hpp` and called by `register_builtin_hle()`.
Every handler stays `static` to this file, including its own `s_np_ok` success stub — that per-file
copy is why the split needed nothing promoted.

A test calling `register_service_hle()` directly will not see these NIDs; prefer
`register_builtin_hle()`.
