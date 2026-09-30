# Live backend integration tests

These tests drive the registered graphics/compute backends, their shared Vulkan device, resource
ownership and capture interfaces. Metadata-only arms may seed the real caches without submitting
GPU commands; distinguish those state assertions from rendered pixel or synchronization evidence.
Pure frontend policy tests belong in `frontends/shared/tests`. `tests/fixtures/render_runner.h`
is also the shipping graphics backend, and its synchronization rules apply here.

`test_graphics_raw_wide_upload` registers owned synthetic AGC programs, realizes their real graphics
tables, and checks numeric raw x4/x8 uploads and descriptor-only consumers in both stages. Its
`--cpu-only` arm checks registration, backing, reflection and pre-backend refusal without creating
a device. Default execution observes pixels, including hosted bytes and the memory-fed selector's
realization-owned word. The registry owners remain mapped for process lifetime: AGC interprets
pointer fields below 4 GiB as relative offsets, so a low-address static fixture is not valid input.

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
