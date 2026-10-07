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

Renderer-claimed VOLUMES follow one rule per colour slot: `volume_producer_shape.hpp` derives the
guest footprint a volume slot's producer pass claims from its proven native layout, for every slot
of a layered pass (#4643), and `volume_publication_source.hpp` is what a claimed volume hands to
its guest publication (#4625).
