# Dear ImGui (vendored)

Immediate-mode GUI, vendored **verbatim** from **Dear ImGui** by Omar Cornut
(https://github.com/ocornut/imgui), release **v1.91.9b**.

- **License:** MIT (see `LICENSE`). Copyright (c) 2014-2025 Omar Cornut.
- **Why vendored, not re-derived:** the game library view (#1471) needs text layout, a font atlas, and
  widget behaviour. That is a large, general-purpose, well-defined problem with nothing PS5-specific
  about it — re-deriving a font rasterizer and layout engine to draw a grid of covers would be
  disproportionate. Dear ImGui is permissively licensed and contains no Sony code, firmware, or keys.
  This is the charter's "vendoring permissively-licensed standalone libraries" exception, and the
  project owner chose ImGui explicitly over hand-rolling a software text renderer.
- **Frontend only.** Only `prosper-app` links this. `prosper_core` does not, and must not: the
  dependency arrow in `docs/FRONTEND_APP.md` points one way, so deleting `frontends/` still leaves CI
  unaffected.

## What is here, and what is not

`src/` is the core library and `backends/` the two platform backends this frontend uses:

| | |
|---|---|
| `src/` | `imgui.cpp`, `imgui_draw.cpp`, `imgui_tables.cpp`, `imgui_widgets.cpp` plus their headers and the bundled `imstb_*` helpers |
| `backends/` | `imgui_impl_sdl3.*` and `imgui_impl_vulkan.*` — the app already owns an SDL3 window and a Vulkan device, so ImGui renders into that existing context rather than creating a second one |

Deliberately **not** vendored: `imgui_demo.cpp`, `examples/`, `docs/`, and the other backends. They are
not built, and leaving them out keeps the vendored tree to what is actually compiled. Every file that is
here is byte-identical to the upstream release.

Do not edit anything under `src/` or `backends/` — keep them byte-identical to upstream so the copy is
auditable and updatable. prosper's own UI code (the library grid, the settings view, the texture upload
for cover art) lives in `frontends/prosper-app/`.

## Updating

Download the release tarball, replace the files listed above, and check `diff -r` against upstream is
empty. `imconfig.h` is unmodified: prosper's build passes its configuration through compile
definitions instead, so an update never has to be merged by hand.
