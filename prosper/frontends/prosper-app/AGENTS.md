# Desktop frontend

`main.cpp` owns the SDL window, Vulkan presentation and application lifecycle.
Library UI, settings, gamepad input and FPS overlays live alongside it; guest APIs and
shared rendering implementations belong in their existing core/shared directories.
Stop frontend workers and release SDL-dependent objects before calling SDL_Quit.
Keep performance displays explicit about guest flips, presentations and fresh render work.

`snapshot_capture.{hpp,cpp}` implements explicit `--snapshot-capture` wall-clock PNG/JSONL
evidence through the normal CPU publication presentation path. Retain pixels only after a
successful host presentation, with origin and identity from that same immutable lease. Preserve
stale sampling and the requested deadline/count; unavailable or failed frames must never become
zero-filled success. It shares pristine image classification and persistence with
`frontends/shared/capture`. Tests live in `tests/frontend/test_visible_snapshot_capture.cpp`.

RenderDoc's `PROSPER_RENDERDOC_AT_FRAME` uses a host-present ordinal. For a scripted route, use
`PROSPER_RENDERDOC_AT_PAD_FLIP=<positive ordinal>` to aim at the guest pad-flip ordinal established
by the route's first pad poll. It is a one-shot guest-input axis; unavailable pad state leaves it
pending, and it still captures through the renderer Vulkan instance and guest-present closure.
