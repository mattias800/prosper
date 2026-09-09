# `shared/rtt` — render-target binding, extent and pixel policies

Shared render-to-texture rules belong here: multiple-target binding and extent selection,
renderer/consumer shape compatibility, CPU/GPU representation authority, and native-format CPU
pixel filling and nearest-neighbor scaling. Snapshot materialization may retain immutable CPU
owners, but these helpers do not acquire Vulkan resources or establish completion or freshness.

Live publication, invalidation, image imports, leases and synchronization belong in `shared/live`
and its renderer/backend integration. A matching extent or byte layout alone does not authorize
borrowing a live image. Generic format decoding belongs in `shared/texture`. CPU byte-oracle and
policy tests belong in `shared/tests`; device ownership and ordered producer/consumer behavior
need the corresponding live GPU integration tests.
