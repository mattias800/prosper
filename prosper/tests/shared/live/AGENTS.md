# Live backend integration tests

These tests drive the registered graphics/compute backends, their shared Vulkan device, resource
ownership and capture interfaces. Metadata-only arms may seed the real caches without submitting
GPU commands; distinguish those state assertions from rendered pixel or synchronization evidence.
Pure frontend policy tests belong in `frontends/shared/tests`. `tests/fixtures/render_runner.h`
is also the shipping graphics backend, and its synchronization rules apply here.

`test_raster_quad_collection` uses owned AGC registration and the real realization/registered
renderer route. Its CPU arm checks producing SOURCE/raw lifetime without creating a device; its
GPU arm observes only private pre-depth/pre-guest-PS raster-input records, including helper versus
covered status and per-primitive overlap. The CAS budget controls discard whole incomplete batches.
No observed varying is labeled MUST-defined guest state, and neither append order nor the supplied
fullscreen fixture demonstrates PS5 Wave64 raster packing or ordered attachment commit.
The realized smooth and explicit-parameter programs export an owned per-vertex varying; all covered
Interpolant/Parameter/SystemInterpolation record words have an independent clip-vertex/barycentric
oracle. Helper values remain raw observations without numeric guest authority. The separate
presence-zero empty-scissor case must refuse rather than collect different effective raster state.

`test_graphics_raw_wide_upload` registers owned synthetic AGC programs, realizes their real graphics
tables, and checks numeric raw x4/x8 uploads and descriptor-only consumers in both stages. Its
`--cpu-only` arm checks registration, backing, reflection and pre-backend refusal without creating
a device. Default execution observes pixels, including hosted bytes and the memory-fed selector's
realization-owned word. The registry owners remain mapped for process lifetime: AGC interprets
pointer fields below 4 GiB as relative offsets, so a low-address static fixture is not valid input.
The descriptor-to-fresh-mask arms reuse the descriptor SGPR pair for an independent compare and
exact Bool condition, retaining a real consumer table and both nonempty stages before pixel checks.

`test_draw_resource_status` seeds explicit reflection-memo contracts and calls the real resource
builder. It checks rejection reasons, both cache-cold MSAA skips, later bindings/stages, and
constructor-population versus successful census/timing deltas. Its hosted-byte checks are metadata
materialization evidence, with no rendered-pixel, Vulkan-device, validation-layer or completion
claim. Keep its same-thread context objects alive because two native TLS aliases bind once.

`test_render_buffer_capture --direct-validation` keeps the actual cold/changed/warm packed-texture
pixels and F8 records, then checks publication of direct comparison counts and the old failed-prefix
charges. The scratch-copy CTest variant retains the same pixels/cache contract and requires no
new direct observations. Its Linux `--direct-validation-watch` arm uses real mapping, watch
promotion, a CPU fault and dirty query: unchanged-watch reuse and watch-only refusal add no direct
samples, while pre-promotion comparisons provide a positive population control. Environment modes
are fixed before startup. These assertions do not imply physical comparison traffic or GPU timing.

## Ruled out

- Complementing the expected PS float with `1 - value` leaves `0.5` unchanged. The #3987
  raw-wide negative observer produced 47 pixel failures and one passing fresh-mask arm, despite
  all 48 production pixel checks passing. Flip expected red bit `0x80` instead: every byte changes
  by 128, beyond the pixel observer's one-byte tolerance. Preserve the failed calibration.

- A reused persistent output is not a fresh black background. The raw-wide upload fixture's
  translated triangles leave old pixels untouched when the backend correctly loads that target.
  Use distinct private output identities and observe every row pixel for each isolated draw.
- A low-address static AGC program owner does not exercise absolute guest pointers: fields below
  4 GiB are interpreted as relative offsets. Assert patched PGM/header metadata before using a
  failed shader realization as a refusal control.
- An EXEC-writing CMPX does not isolate a VS VCC lifetime refusal. Use the decoded non-EXEC
  VCC compare with a known VGPR setup in the focused graphics backing test.
