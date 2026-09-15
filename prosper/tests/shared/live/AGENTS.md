# Live backend integration tests

These tests drive the registered graphics/compute backends, their shared Vulkan device, resource
ownership and capture interfaces. Metadata-only arms may seed the real caches without submitting
GPU commands; distinguish those state assertions from rendered pixel or synchronization evidence.
Pure frontend policy tests belong in `frontends/shared/tests`. `tests/fixtures/render_runner.h`
is also the shipping graphics backend, and its synchronization rules apply here.
