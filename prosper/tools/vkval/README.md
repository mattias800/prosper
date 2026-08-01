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

So the scan has a second, blunter instrument check: **if the allow-list is non-empty and *zero*
messages were observed, that is a failure.** Nine known defects spread over eight tests do not all
fall silent at once; a parser that stopped matching, a log read from the wrong place, or a build
without the Vulkan tests all produce exactly that shape. Deleting the last allow-list entry — the
intended end state — switches the check off.

`test_vk_validation_scan.py` pins both layer framings verbatim, and asserts that the test's own
output contains nothing the scan would read as a validation message (ctest captures it into the
same log, so a test that prints sample messages makes the guard read its own tail — measured, it
did).

The same rule applies to changes to the tool itself: `test_vk_validation_scan.py` (ctest
`vkval_scan_logic`) pins that the parser really matches the layer's message framing, including the
`VUID-vkCmdDispatch-viewType-07752` line that motivated all of this.

## `allowlist.txt`

Every message id observed when the guard was switched on is listed there with a one-line reason and
an open issue. A bare id with no reason is a hard error — the point of the file is that pre-existing
findings are *visibly deferred*, never silently tolerated. Fixing a defect means deleting its line.

An allow-listed id that produces no messages in a run is reported but does not fail, because the
observed set is driver-dependent in both directions:

* `VUID-vkCmdDispatch-maintenance4-08602` is a newer check than validation layers 1.3.275, which is
  what Ubuntu 24.04 (the CI runner) carries;
* `VUID-VkMappedMemoryRange-size-01389` cannot trigger on lavapipe at all, whose
  `nonCoherentAtomSize` is 1 — it was only seen on RADV.

So run the guard on hardware occasionally even though CI runs it on lavapipe; they see different
subsets of the same set of defects.

## Cost

Measured on the CI runner's exact configuration (Ubuntu 24.04, Mesa 25.2.8 lavapipe, reproduced with
`podman run --rm ubuntu:24.04`), over two runs of the same image: the full ctest suite takes
**17.0-24.0 s** without the layer and **33.2-34.0 s** with it, everything passing either way. So the
guard roughly doubles a test phase that is already a small fraction of a job spending minutes
compiling — cheap enough to run on every PR rather than on a schedule.

## Reproducing the CI environment locally

```bash
podman run --rm -v "$PWD:/work" ubuntu:24.04     # mesa-vulkan-drivers 25.2.8, vulkan-validationlayers 1.3.275.0
```

Both packages come from the Ubuntu archive at job time rather than from a pin, so check
`apt-cache policy` if the observed set ever shifts under you.
