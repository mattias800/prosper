# `frontends/shared/present/` — getting a finished guest frame onto the window

Everything between "the renderer has the flipped front buffer" and "prosper-app shows it", shared by
every frontend. Two paths exist and every file here serves one of them or the choice between them:

- **GPU present** (#1270). The renderer copies the front buffer into a small pool of scanout slots on
  the shared device (`present_blit`), and prosper-app blits a slot to its swapchain. No CPU round
  trip. Only prosper-app turns it on (`set_gpu_present_active`); headless tools never do.
- **The CPU fallback.** The renderer publishes CPU pixels, a render target's readback or the guest's
  own display buffer de-swizzled (`guest_scanout_present`), and the frontend uploads them. It is the
  only path headless, and the path prosper-app falls back to whenever GPU present declines a flip.

When a final render span declines GPU present, the renderer names the reason (`GpuPresentOutcome` in
`present_blit_policy.hpp`); the `present-path-fallback` perf alarm reports it as `declines=`. A
display buffer written by a **compute** dispatch rather than a draw has no render target, so
`compute_scanout` keeps a GPU mirror of the dispatch's own result for it (#3915), used only when the
renderer holds no entry at all at the flipped address: a pixel-less tombstone there means the CPU
path keeps the previous frame, and the mirror must too.

Decisions live in small header-only policy files (`*_policy.hpp`, `present_extent.hpp`,
`guest_scanout_present.hpp`, the pure parts of `compute_scanout.hpp`) so they can be unit-tested
without a device; the Vulkan mechanism sits beside them in the `.cpp` files. Keep that split: most
defects on this path have been a predicate that nobody could test without a 4K title run.
