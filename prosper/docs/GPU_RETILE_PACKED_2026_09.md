# Packed native storage-image GPU retile

Tracking: [#3407](https://github.com/mattias800/prosper/issues/3407).

## Scope and evidence

The existing GPU tiler accepted word-sized ordinary images and mode-24 native two-byte arrays.
Ordinary R16_UINT and R8/RG8_UNORM images still tiled on the CPU even when their Vulkan transfer
already contained exact native guest pixels. This extension groups adjacent texels into full 32-bit
words and uses the same submission, linear baseline and host-publication contract as existing retile.

The retained GTA Performance-route F8 from source `b785d0cd35dc2129775ae50a5fcb8173320d1bec`
(buffer retention disabled) identifies the following CPU layout work:

| Native format / mode | Writeback records | CPU layout ms | Repeated guest bytes |
| --- | ---: | ---: | ---: |
| R8_UNORM / 27 | 30 | 19.495 | 117,964,800 |
| RG8_UNORM / 27 | 30 | 14.935 | 235,929,600 |
| R16_UINT / 24 | 31 | 17.241 | 97,517,568 |

This is 51.671 ms of targeted CPU work in a 5.0185-second window, not a prediction of net savings.
Packing already takes zero reported milliseconds for these native inputs. The F8 reports 373.427 ms
of total compute writeback; adding GPU work and retaining its output can offset CPU savings. Current
before/after game results must therefore include the enclosing dispatch and presentation intervals.
Freshly rendered frames are not distinguished by these captures. The source, binary and original
capture are identified in [the preceding measurement guide](RENDERER_BUFFER_RESIDENCY_2026_09.md).

## Ownership and layout contract

- Mode 24 admits two-byte texels; mode 27 admits one- or two-byte texels. The selected pipe equation
  must map its low address bits exactly to the horizontal group and exclude those x bits from every
  higher address bit. Each invocation owns a complete aligned output word; no read-modify-write,
  8/16-bit storage capability, atomics or vendor-specific feature is needed.
- Ordinary 2D images, one-layer views and existing exact native multilayer images use independent
  planes. A row must contain complete words. Partial rows, non-native interchange, selected mips,
  ambiguous metadata/pitches and unsupported equations retain CPU layout. Native shader declarations
  are unchanged; RG8/R16F multilayer interchange remains the existing CPU path.
- Padded dispatch and byte bounds are checked before allocation. Out-of-image output words are zeroed
  without reading the source. The linear result remains separate for exact comparison; the tiled
  buffer is owned through submission completion and receives its own host-read barrier.
- Allocation failure retains CPU fallback. Device loss remains a sticky failure; it is not hidden as
  successful fallback. The ordinary guest write-watch and publication sequence is unchanged.

`PROSPER_NO_GPU_RETILE_PACKED_EXTENSION=1` restores the previous packed scope (mode-24 two-byte
multilayer images only), while leaving word/volume GPU tiling enabled. It is a same-binary comparison
control. `PROSPER_NO_GPU_RETILE=1` still disables all GPU retile. Buffer retention remains opt-in.

## Verification

`test_gpu_retile --subword` exercises actual native R8_UNORM, R8_UINT, RG8_UNORM, R16_UINT and R16F
storage-image copies through the production compute backend. It checks exact guest bytes, poisoned
padding, destination guards, three partial updates crossing packed-word boundaries, ordinary and
one-layer array views, incomplete-word fallback, actual GPU recording and host barriers. The existing
multilayer Uint16 cases cover both modes; RG8/R16F arrays retain explicit raw-interchange controls.
Seven separate processes cover `PROSPER_RX_PIPES=1,2,4,8,16,32,64`. Existing allocation/mapping failure
and device-loss controls remain, including new ordinary packed-image cases.

Focused build and 30/30 retile tests passed. With the extension deliberately disabled, the subword
fixture fails 144 per-dispatch and 48 aggregate GPU-recording assertions while every byte-correctness
assertion passes. A collapsed equation bit is separately rejected by the permutation guard.

```sh
cmake --build prosper/build-linux -j6
ctest --test-dir prosper/build-linux --no-tests=error --output-on-failure
python3 prosper/tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux \
  --probe test_gpu_retile --sync --allowlist /dev/null \
  --ctest-arg=-R '--ctest-arg=^gpu_retile' --ctest-arg=--no-tests=error --ctest-arg=-j1
```

## Ruled out

- Horizontal packing does not apply to every sub-word mode: mode-24 byte texels put y0 in address
  bit1. The equation proof refuses them, and real runtime cases preserve correct CPU fallback. #3407.
- Reduced CPU layout alone is not an end-to-end speedup. The enclosing GPU/dispatch and presentation
  costs remain the acceptance measurements; the previous buffer-cache experiment remains closed.
