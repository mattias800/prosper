# Render-state tests

CPU extraction and pipeline contracts live beside separately gated Vulkan rendering tests.
`test_raster_launch_facts` folds actual PM4 register writes and draw snapshots: raw observed
launch words are not initialized guest registers or permission to synthesize Wave64 helpers.
Keep absent inputs distinct from explicit zeroes; use independent literal register/value oracles.
