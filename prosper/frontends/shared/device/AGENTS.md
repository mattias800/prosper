# Shared Vulkan device policy

Loader/device requirements and selection shared by live frontends and execution tests.
Pipeline-cache files belong here: they store opaque driver data, not guest shaders or game assets.
Cache compatibility checks reject damaged or mismatched files; they cannot guarantee a driver
accepts its own data safely. Keep disk loading opt-in while the recorded NVIDIA failure remains.

Feature acquisition that more than one device path needs belongs here rather than beside whichever
device was written first: the renderer's device, the compute backend's own device and the adopt path
must answer the same question the same way. `image_robustness.hpp` is that for the recompiler's
out-of-bounds image-read contract, and `storage_image_contract.hpp` holds the Vulkan-free decision so
the refusing branch is testable without a driver -- no device here lacks the feature, so a test drawn
from a real device can only exercise the accepting one.
