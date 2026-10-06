# `tests/data` — binary test fixtures

Small checked-in inputs that tests load by path (`fixtures/test_data.h` resolves this folder). They
are guest shader code, register images and symbol tables, never textures, audio or other game
content, and each one is the smallest thing that reproduces what its test asserts.

When you add one, add a line below saying where it came from, so the next reader can tell a capture
from a hand-built file without opening the test.

- `ngg_merged_es_prolog.bin` (102 dwords), `ngg_merged_gs_main.bin` (413 dwords): the ES prolog
  and GS main of a merged ES+GS NGG program, the 32-slice 3D LUT producer captured from *Kena:
  Bridge of Spirits* (`PPSA01802`, program `0x5009440000`). Linked, they are the guest code of the
  #3135 subgroup-shell execution tests. Byte-identical to the `strip_layer_*.bin` fixtures of #3857.
