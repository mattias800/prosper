# `tools/vkval` — the Vulkan validation guard

Runs prosper's ctest suite under `VK_LAYER_KHRONOS_validation` and fails when a validation message
appears that is not written down in `allowlist.txt`.

## Why

#1690 was a `VK_IMAGE_VIEW_TYPE_3D` view bound to a descriptor whose SPIR-V module declared
`OpTypeImage ... 2D`. That is undefined behaviour, and three conformant drivers resolved it three
ways — two returned the seeded texel, one returned zero. It presented as a Mesa *version*
regression, was attributed to an upstream driver defect, cost `game_compute_exec` its place on CI
(484 executed assertions), and was only settled by disassembling the SPIR-V by hand.

The validation layer names that defect at the dispatch, on the first run:

```
VUID-vkCmdDispatch-viewType-07752
vkCmdDispatch(): the storage image descriptor [VkDescriptorSet ..., Set 0, Binding 5, Index 0]
VkImageViewType is VK_IMAGE_VIEW_TYPE_3D but the OpTypeImage has (Dim = 2D) and (Arrayed = 0).
Either fix in shader or update the VkImageViewType to VK_IMAGE_VIEW_TYPE_2D.
```

That is the difference the guard buys.

## Running it

```bash
# gate (what CI runs): fail on any message id not in allowlist.txt, or any test that fails
python3 tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux

# measure only: print the grouped findings, never fail
python3 tools/vkval/vk_validation_scan.py --build-dir prosper/build-linux --report-only

# re-parse a log you already have
python3 tools/vkval/vk_validation_scan.py --log prosper/build-linux/Testing/Temporary/LastTest.log
```

The layer package is `vulkan-validationlayers` on Debian/Ubuntu and `vulkan-validation-layers` on
Fedora. **Build in an environment that actually has Vulkan** — a configure that quietly loses it
drops every Vulkan execution test and the scan then observes nothing (#1675). On this project's
Bazzite box that means the `ps5ys` distrobox, not the host.

## The instrument trap this tool is built around

A validation layer that fails to load produces *exactly* the same output as a clean run: nothing.
So does a correctly loaded layer whose `report_flags` are misconfigured — the layer's default is
`error` only, and a probe that asks for best-practices warnings with the wrong setting key prints
nothing and reads as success. Both were hit while writing this.

`vk_validation_scan.py` therefore refuses to report a clean verdict until it has watched the loader
say `Insert instance layer "VK_LAYER_KHRONOS_validation"` (via `VK_LOADER_DEBUG=layer`) on a probe
binary. If the layer is missing, the scan fails with an install hint rather than passing.

That check alone was not enough, and the way it failed is worth keeping. **The two layer versions
in play frame the same message differently:**

```
layers 1.4.341 (Fedora 44)    Validation Error: [ VUID-x ] | MessageID = 0x...
                              <message text on following lines>

layers 1.3.275 (Ubuntu 24.04, VUID-x(ERROR / SPEC): msgNum: N - Validation Error: [ VUID-x ] ...
                the CI runner)  <everything on one line>
```

The first revision of this parser anchored its pattern to the start of a line. It read all 51
messages on 1.4.341 and **0 of 187** on 1.3.275 — the layer loaded, the probe passed, the scan
reported a clean suite, and the CI gate would have been permanently green while observing nothing.
It was caught only by running the positive control (below) on the CI environment rather than the
development one.

So the scan has a second instrument check, and it is deliberately per-entry rather than
all-or-nothing: **every allow-list entry marked `required` must still produce messages.** If one
stops appearing, exactly two things can have happened — the defect was fixed (delete the line, in
the same change) or the scan stopped seeing what it used to. Both are failures until someone says
which.

Checking each id separately is what catches a *partial* break. An all-absent check alone passes a
VUID rename, or a framing change that affects some message shapes and not others, while silently
halving what the guard can see. The two ids that genuinely cannot appear everywhere are marked
`environment-dependent` and say why on their reason line, so the exemption is written down rather
than assumed:

* `VUID-vkCmdDispatch-maintenance4-08602` is a newer check than validation layers 1.3.275, which is
  what Ubuntu 24.04 (the CI runner) carries;
* `VUID-VkMappedMemoryRange-size-01389` cannot trigger on lavapipe at all, whose
  `nonCoherentAtomSize` is 1 — it was only seen on RADV.

Deleting the last allow-list entry — the intended end state — switches the check off.

`test_vk_validation_scan.py` (ctest `vkval_scan_logic`) pins both layer framings verbatim, pins the
`VUID-vkCmdDispatch-viewType-07752` line that motivated all of this, and asserts that the test's own
output contains nothing the scan would read as a validation message. That last one is not paranoia:
ctest captures this test's stdout into the same log the scan parses, so a test that prints sample
messages makes the guard read its own tail — measured, it did.

Smaller closures in the same spirit: a **missing** allow-list file is a hard error rather than an
empty one, and ctest runs with `--no-tests=error`, so "build directory with nothing registered"
cannot report success either.

## `allowlist.txt`

```
<message id> | <required|environment-dependent> | <tests, or *> | <reason, with its issue>
```

Every message id observed when the guard was switched on is listed there. A line the scanner cannot
fully parse is a hard error — the point of the file is that pre-existing findings are *visibly
deferred*, never silently tolerated. Fixing a defect means deleting its line.

**The test list is not decoration.** Several VUIDs are catch-alls —
`VUID-VkShaderModuleCreateInfo-pCode-08737` is VVL's identity for *any* `spirv-val` error at
`vkCreateShaderModule` — so an id-only ledger would defer every future SPIR-V validity defect
anywhere in the suite behind one known emitter bug. An allow-listed id arriving from a test the
ledger does not record fails the scan: a deferral is scoped to where it was measured.

## Cost

Measured on the real GitHub runner (`actions/runs/30719139452`, Linux job): ctest reports
**30.46 s** without the layer and **31.08 s** with it, 168/168 passing either way, so the scan step
costs **31 s** of wall clock on a job that spends minutes compiling. Cheap enough to run on every PR
rather than on a schedule.

The same suite in `podman run --rm ubuntu:24.04` on a faster machine takes 17.0-24.0 s plain and
33.2-34.0 s under the layer, i.e. the layer's *relative* cost is larger the faster the box. Neither
figure is close to needing a dedicated job.

## Reproducing the CI environment locally

```bash
podman run --rm -v "$PWD:/work" ubuntu:24.04     # mesa-vulkan-drivers 25.2.8, vulkan-validationlayers 1.3.275.0
```

Both packages come from the Ubuntu archive at job time rather than from a pin, so check
`apt-cache policy` if the observed set ever shifts under you.

## What this guard does NOT cover

**Its self-validation is borrowed from the defect backlog, and that backlog is meant to shrink.**
The `required` check proves the parser still reads this layer version's output *because* known
defects are still firing. As #1710-#1717 are fixed and their lines deleted, that proof weakens; once
only `environment-dependent` entries remain, a run on the CI driver observes nothing legitimately,
and a parser that quietly stopped matching would again be invisible. Whoever deletes the last
`required` entry is switching that off, and should replace it first — the durable form is a positive
control built into the probe (provoke one known violation and require the scanner to *parse* it),
which proves the parser rather than only the loader. Tracked in **#1725**.

`#1704` also fixed a Vulkan-teardown-from-a-static-destructor defect in
`frontends/shared/live_compute.cpp` that the layer exposed. **CI would not catch a regression of
it.** That crash reproduces on validation layers 1.4.341; the runner carries 1.3.275, where the
suite passes with the defect present. Reverting the lifetime change would leave this step green.
Running the guard on a recent layer version locally is what covers it — one more reason not to treat
a green CI run of this step as "no Vulkan misuse anywhere".
