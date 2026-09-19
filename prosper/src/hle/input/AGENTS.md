# `input` — the guest's input devices and text entry

The Sony libraries a title calls to read a controller or to ask the player for text.

- `hle_pad` — libScePad, backed by a real host backend (evdev/SDL3) and driveable by deterministic
  `.pad` routes for title progression.
- `ime_input` — the shared queue a host frontend pushes key events into. `ime.cpp` drains it; the
  frontends fill it. It is a header so both sides agree on the event shape without either owning it.
- `ime` — libSceIme (the on-screen keyboard API) and libSceImeDialog (the text-entry dialog). Split
  out of `hle/service/hle_service.cpp` in #3735. Both report an honest "no physical keyboard"
  state; the dialog auto-completes because there is no UI to show.

## What belongs here

A Sony library whose subject is the player's input. libSceMouse is still in
`hle/service/hle_service.cpp` — it is a handful of lines reporting a device that exists and never
moves, and it should come here when there is a second reason to touch it.

## Registration

Each library file owns a `register_<lib>_hle()` that `register_builtin_hle()` calls, declared in
`hle/dispatch/dispatch.hpp`. That keeps every handler `static` to the file implementing it, which is
what lets these be separate translation units.
