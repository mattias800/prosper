# Graphics pipeline cache persistence

The renderer shares one device-lifetime `VkPipelineCache` across graphics pipeline creation.
Disk persistence is opt-in, as for the compute cache:

- `PROSPER_DISK_PIPELINE_CACHE=1` selects the platform cache directory.
- `PROSPER_GRAPHICS_PIPELINE_CACHE_PATH=<FILE>` selects an explicit graphics cache file and
  also opts in. An empty path disables graphics persistence.
- `PROSPER_NO_DISK_PIPELINE_CACHE=1` overrides both and disables disk access.

Graphics and compute files are separate. Graphics filenames include vendor/device identity,
driver version and pipeline-cache UUID. The graphics file wraps the original driver data with
a versioned length/checksum envelope and records driver version and process pointer width.
Damaged, truncated or incompatible files become cache misses. Only the original driver bytes
are passed to Vulkan. A driver returning an error for cached creation is retried with empty data;
failure to create even an empty cache retains uncached graphics pipeline creation.

These checks do **not** establish that a driver can safely load every blob it produced. Repeated
GTA V runs previously crashed in NVIDIA's cache-load path; the cause remains unresolved. That is
why loading remains opt-in, including on NVIDIA. No experimental Vulkan feature is required.

`prosper-app` explicitly requests a snapshot before its deliberate `_Exit`. The initialized
renderer context is published atomically and lives for the process lifetime. The snapshot uses
a dedicated mutex shared with graphics pipeline creation, independent of the whole-pass resource
lock. It waits up to one second to acquire that mutex; a compilation still holding it then skips
the save. Queue draining alone does not serialize pipeline compilation. Disk I/O occurs
after the snapshot releases the mutex. Guest-triggered abrupt exits may still bypass this path.

Each writer reserves a unique sibling temporary directory, closes the complete file, then
replaces the destination atomically. A failed replacement preserves the existing destination.
Concurrent writers produce complete files; the last successful replacement wins. This is a
rebuildable cache, not a promise of durability after power loss.

The `pipeline_cache_file` test covers identity checks, concurrent writers and failed replacement.
`graphics_cache_persistence` runs separate processes: render/save/`_Exit`, load/render, disabled
loading, corrupted/truncated data and driver rejection with an empty-cache retry. It observes
the actual Vulkan creation argument, so a cached file alone cannot make loading pass. It also
checks that a held renderer mutex permits saving while a held compilation mutex prevents it.
Both tests require Python for their
temporary-directory/process orchestration; the Vulkan test additionally requires a device.

Successful persistence proves cross-launch reuse is available. It does not establish that
pipeline compilation causes a particular game's stutter or promises an FPS improvement.

References: [#3378](https://github.com/mattias800/prosper/issues/3378),
[Vulkan cache input](https://docs.vulkan.org/refpages/latest/refpages/source/VkPipelineCacheCreateInfo.html),
[cache extraction](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetPipelineCacheData.html).
