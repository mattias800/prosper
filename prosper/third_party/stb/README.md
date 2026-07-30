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
