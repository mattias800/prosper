# Shared Vulkan device policy

Loader/device requirements and selection shared by live frontends and execution tests.
Pipeline-cache files belong here: they store opaque driver data, not guest shaders or game assets.
Cache compatibility checks reject damaged or mismatched files; they cannot guarantee a driver
accepts its own data safely. Keep disk loading opt-in while the recorded NVIDIA failure remains.
