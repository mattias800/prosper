# Frontend capture contracts

Shared pristine-frame classification, source/guest outcome tracking and JSONL evidence for
presented captures. Capturing pixels does not authenticate composited rendering or guest state.
The screenshot tool and normal desktop app share this contract; route windows, likeness oracles
and reviewed baselines belong to tools/snapshot. Compute metrics before any HUD annotation.

`capture_manifest.{hpp,cpp}` owns normalization, content metrics, stale classification and JSONL
records. `capture_png.{hpp,cpp}` owns the shared PNG writer and CRC. Keep these contracts identical
between screenshot and app producers; a CPU publication capture is not native GPU visual proof.
