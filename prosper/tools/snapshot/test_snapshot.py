#!/usr/bin/env python3
import importlib.util
import json
import os
import tempfile
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("prosper_snapshot", os.path.join(HERE, "snapshot.py"))
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


class SnapshotContentTests(unittest.TestCase):
    def test_content_window_requires_multiple_frames_and_progression(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = [
                (1, 1, "a", "raw_scanout", 0, 0, 100, 2),
                (5, 2, "b", "composited", 0, 0, 100, 2),
                (6, 2, "c", "composited", 1, 0, 100, 2),
                (7, 2, "c", "composited", 1, 0, 101, 2),
                (8, 1, "d", "composited", 0xffffffffffffffff, 0xffffffffffffffff, 0, 0),
            ]
            events = [{"type": "run"}]
            for index, (elapsed, colors, crc, source, ahash, dhash, luma, nonblack) in enumerate(samples):
                path = os.path.join(tmp, f"shot_{index:02d}.png")
                with open(path, "wb") as f:
                    f.write(b"png evidence")
                events.append({
                    "type": "sample", "elapsed_seconds": elapsed, "source": source,
                    "png": path, "width": 2, "height": 1,
                    "distinct_rgb_colors": colors, "pixel_crc32": crc,
                    "average_hash": f"{ahash:016x}",
                    "difference_hash": f"{dhash:016x}",
                    "luma16x9": f"{luma:02x}" * (16 * 9),
                    "nonblack_rgb_pixels": nonblack,
                })
            events.append({
                "type": "summary", "exit_code": 0,
                "saved": len(samples), "requested": len(samples),
            })
            with open(os.path.join(tmp, "capture.jsonl"), "w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

            entry = {
                "min_colors": 2, "min_qualifying_frames": 3, "min_pixel_changes": 2,
                "capture_after_seconds": 4, "capture_before_seconds": 9, "dims": [2, 1],
                "structural_references": [
                    {"luma16x9": "64" * (16 * 9)},
                ],
                "min_structural_similarity": 0.99, "min_structural_matches": 3,
                "min_nonblack_ratio": 0.5, "min_nonblack_matches": 3,
                "min_content_match_ratio": 0.75,
            }
            summary = SNAPSHOT.analyze_content_manifest(tmp, entry)
            ok, failures = SNAPSHOT.content_result(entry, summary)

            self.assertTrue(ok, failures)
            self.assertEqual(4, len(summary["records"]))
            self.assertEqual(3, len(summary["qualifying"]))
            self.assertEqual(2, summary["pixel_changes"])
            self.assertEqual(3, summary["structural_matches"])
            self.assertEqual(3, summary["nonblack_matches"])
            self.assertEqual(4, len(SNAPSHOT.choose_review_records(summary)))
            metadata_path = os.path.join(tmp, "evidence.json")
            SNAPSHOT.save_content_metadata(metadata_path, summary)
            with open(metadata_path) as f:
                metadata = json.load(f)
            self.assertEqual(4, len(metadata["records"]))
            self.assertEqual("composited", metadata["records"][0]["source"])

    def test_content_manifest_rejects_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "capture.jsonl"), "w") as f:
                f.write(json.dumps({
                    "type": "summary", "exit_code": 1, "saved": 2, "requested": 3,
                }) + "\n")
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                SNAPSHOT.analyze_content_manifest(tmp, {"min_colors": 1})

    def test_content_result_reports_each_failed_contract(self):
        summary = {
            "qualifying": [{}], "pixel_changes": 0, "dimensions_consistent": True,
            "dims": (2, 1),
        }
        entry = {
            "min_qualifying_frames": 2, "min_pixel_changes": 1, "dims": [3, 1],
        }
        ok, failures = SNAPSHOT.content_result(entry, summary)
        self.assertFalse(ok)
        self.assertEqual(3, len(failures))

    def test_entry_env_resolves_route_and_uses_fresh_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = os.path.join(tmp, "route.pad")
            with open(route, "w") as f:
                f.write("1-2:cross\n")
            env = {}
            SNAPSHOT.apply_entry_env(env, {
                "pad_script": route, "savedata_policy": "fresh", "env": {"EXTRA": "yes"},
            }, tmp)
            self.assertEqual("@" + route, env["PROSPER_PAD_SCRIPT"])
            self.assertEqual("1", env["PROSPER_PAD_SCRIPT_LOG"])
            self.assertEqual("yes", env["EXTRA"])
            self.assertTrue(os.path.isdir(env["PROSPER_SAVEDATA_DIR"]))

    def test_structural_similarity_tolerates_small_changes_but_rejects_missing_layer(self):
        reference = (100,) * (16 * 9)
        subtle_change = (101,) * (16 * 9)
        missing_half = (0,) * (16 * 4) + (100,) * (16 * 5)
        self.assertGreater(SNAPSHOT.structural_similarity(reference, subtle_change), 0.99)
        self.assertLess(SNAPSHOT.structural_similarity(reference, missing_half), 0.1)

    def test_candidate_fingerprint_ignores_baseline_and_review_only(self):
        base = {"name": "x", "frame": 7, "hash": "old", "dims": [1, 1], "review": "old"}
        changed_baseline = dict(base, hash="new", dims=[2, 2], review="new",
                                structural_references=[{"luma16x9": "00" * (16 * 9)}])
        changed_contract = dict(base, frame=8)
        self.assertEqual(SNAPSHOT.entry_fingerprint(base), SNAPSHOT.entry_fingerprint(changed_baseline))
        self.assertNotEqual(SNAPSHOT.entry_fingerprint(base), SNAPSHOT.entry_fingerprint(changed_contract))

    def test_content_candidate_requires_complete_review_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_review = SNAPSHOT.REVIEW_DIR
            SNAPSHOT.REVIEW_DIR = tmp
            try:
                entry = {"name": "gameplay", "min_colors": 10,
                         "min_structural_similarity": 0.85}
                records = []
                for run in (1, 2):
                    run_records = []
                    for index in range(2):
                        path = os.path.join(tmp, "gameplay", f"run{run}_{index:02d}.png")
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "wb") as f:
                            f.write(b"reviewed image")
                        run_records.append({
                            "path": path, "elapsed": run * 10 + index, "colors": 20,
                            "dims": (4, 3), "average_hash": run * 10 + index,
                            "difference_hash": run * 20 + index,
                            "luma16x9": (run * 10 + index,) * (16 * 9),
                            "nonblack_ratio": 0.8,
                        })
                    records.append({
                        "records": run_records, "qualifying": run_records,
                        "richest": run_records[0], "dims": (4, 3),
                    })
                SNAPSHOT.save_content_candidate(entry, records)
                SNAPSHOT.approve_content_candidate(entry)
                self.assertEqual(4, len(entry["structural_references"]))
                self.assertEqual(0.4, entry["min_nonblack_ratio"])
                self.assertIn("4 composited images", entry["review"])

                os.remove(os.path.join(tmp, "gameplay", "run2_01.png"))
                entry.pop("structural_references")
                entry["review"] = "pending"
                with self.assertRaisesRegex(RuntimeError, "incomplete"):
                    SNAPSHOT.approve_content_candidate(entry)
            finally:
                SNAPSHOT.REVIEW_DIR = old_review


if __name__ == "__main__":
    unittest.main()
