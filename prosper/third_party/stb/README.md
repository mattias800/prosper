# stb single-header libraries (vendored)

Two headers live here and they have **different** scopes — read the one you are about to use.

---

# stb_image (vendored)

Single-header image decoder, vendored **verbatim** from **stb** by Sean Barrett
(https://github.com/nothings/stb), `stb_image.h` **v2.30**, commit
`f0569113c93ad095470c54bf34a17b36646bbbb5`.

- **License:** dual public-domain / MIT — the author's choice of either (see `LICENSE`, extracted
  verbatim from the end of the header, where stb ships it).
- **Why vendored, not re-derived:** cover art is `sce_sys/icon0.png`, and nothing in the tree decodes
  PNG — SDL3 has no built-in image loader and Dear ImGui does not decode. Writing a PNG decoder
  (inflate, all five filter types, colour-type and bit-depth handling, interlacing) to draw a library
  grid would be disproportionate, and getting it subtly wrong is easy. stb_image is permissively
  licensed, self-contained, and contains no Sony code, firmware, or keys — the charter's
  "vendoring permissively-licensed standalone libraries" exception, approved with Dear ImGui for #1471.
- **Frontend only.** Only `prosper-app` links this; `prosper_core` does not and must not.

## Scope of use

Used **only** to decode `icon0.png` from a game dump's `sce_sys/` for display in the library view. It is
not on the guest path: nothing a title does reaches this decoder, and no guest-controlled bytes are fed
to it. Texture decode for the *emulated* GPU is prosper's own code in `src/gpu/`.

The header is compiled into exactly one translation unit in `frontends/prosper-app/` (the one that
defines `STB_IMAGE_IMPLEMENTATION`), with only the PNG decoder enabled — `STBI_ONLY_PNG` — so the
JPEG/BMP/TGA/GIF/HDR/PIC/PNM paths are not built at all. That is both smaller and a narrower surface
than pulling in every format for a file we already know is PNG.

Do not edit `stb_image.h` — keep it byte-identical to upstream so the copy is auditable and updatable.
prosper's glue (reading the file, uploading the pixels as a Vulkan texture) lives in
`frontends/prosper-app/`.

---

# stb_truetype (vendored)

Single-header TrueType/OpenType rasterizer, vendored **verbatim** from **stb** by Sean Barrett
(https://github.com/nothings/stb), `stb_truetype.h` **v1.26**, commit
`f0569113c93ad095470c54bf34a17b36646bbbb5`, SHA-256
`ecd30b05e0dd4fea3a13c26810dd9e1992dc379049482c393d5a19e6b5090aab`.

- **License:** dual public-domain / MIT, the same dual grant as `stb_image.h` (see `LICENSE`).
- **Why vendored, not re-derived:** `libSceFont`'s `Ft` (FreeType) editions are a glyph rasterizer.
  Re-deriving TrueType outline parsing, hinting-free scan conversion, `cmap` lookup and `hmtx`
  metrics would be disproportionate, and it is exactly the "codec-shaped standalone problem" the
  charter's vendoring exception names. It contains no Sony code, firmware, or keys.
- **The alternative was FreeType**, and it was rejected on build surface, not on quality: FreeType is
  an external shared dependency that would have to be found, version-gated and shipped on Linux,
  Windows/MinGW and macOS alike. This header adds a file and no dependency.

## Scope of use — this one IS on the guest path

Unlike `stb_image.h` above, `stb_truetype.h` is compiled into **`prosper_core`**, in the single
translation unit `src/hle/util/hle_font.cpp` that defines `STB_TRUETYPE_IMPLEMENTATION`. It is
reached by guest calls to `libSceFont`, and the bytes it parses come from the game dump — a title
hands prosper its own font file through `sceFontOpenFontMemory`.

**Upstream states, at the top of the header, "NO SECURITY GUARANTEE — DO NOT USE THIS ON UNTRUSTED
FONT FILES".** That warning is recorded here rather than glossed, and the reason it is acceptable in
this position is a property of prosper's threat model, not of the parser: the font bytes come from
the same game dump whose **x86-64 code prosper already executes natively on the host**. A dump that
could be trusted to run its own code but not to supply its own font would be a strange threat model.
prosper is not a browser and does not open fonts from the network or from other users.

What prosper's own glue must still do, and does in `hle_font.cpp`:

- **Copy the guest's font bytes into a host-owned buffer at open time** rather than retaining a
  pointer into guest memory. The guest may unmap or reuse that range, and a rasterizer holding a
  dangling pointer would fault far from the cause.
- **Bound the accepted size** (`kMaxFontBytes`, 64 MiB) *before* the copy. The length is a
  guest-supplied `uint32`, and prosper cannot see how large the guest's buffer actually is, so an
  oversized one is not a big allocation — it is a read of that many bytes **from guest memory**,
  i.e. a host SIGSEGV that no `catch (...)` can turn back into an error return. Review caught this
  bullet promising a bound the code did not yet have; do not let it drift back apart.
- **Refuse a blob `stbtt_InitFont` rejects**, with an error return — never a success return over an
  unparsed font.

Do not edit `stb_truetype.h` — keep it byte-identical to upstream so the copy is auditable and
updatable. prosper's glue (the libSceFont ABI, surfaces, scissors, metrics) is project code in
`src/hle/util/hle_font.cpp`.
