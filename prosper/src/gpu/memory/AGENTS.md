# `memory` — how much device memory a renderer cache may hold

Pure residency *policy*: functions that take a device's heap figures (size, the driver's
`VK_EXT_memory_budget` budget and usage, prosper's own per-heap count) and answer "how many bytes may
this cache keep resident?". Nothing here allocates, frees or calls Vulkan, so the policy is testable
with plain numbers (`tests/gpu/memory/`) and the renderer owns only the plumbing — querying the
driver, choosing the heap, applying the answer (`tests/fixtures/render_runner.h`).

Boundary: *counting* what prosper holds is `../diagnostics/gpu_memory_budget` (observation, on by
default, changes no pixel); *deciding* what a cache may hold is here, and it does change behaviour.
Keep the two apart so the instrument stays a pure observer.

`texture_cache_budget` is the persistent texture-image cache's budget (#3873). Its header states the
whole formula and why unified-memory devices are capped at the pre-#3873 rule; read it before
changing a constant, and change the unit test's hand-computed expectations with it.

`memory_type_select` is the other policy here: WHICH Vulkan memory type an allocation takes (#3888).
Anything only the GPU touches (render targets, depth, sampled/storage images, GPU-only buffers)
prefers a `DEVICE_LOCAL` type and falls back to any compatible one; memory the CPU maps keeps its
explicit flags. Do not add another "first compatible type" loop beside a renderer allocation: that
rule puts images in system memory on a driver that lists a host type first, silently. It is
header-only because its inputs are Vulkan structs; it still makes no Vulkan call.
