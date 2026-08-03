#!/usr/bin/env python3
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("prosper_snapshot", os.path.join(HERE, "snapshot.py"))
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


class SnapshotContentTests(unittest.TestCase):
    def test_snapshot_lock_serializes_independent_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock_path = os.path.join(tmp, "snapshot.lock")
            child_script = """
import importlib.util
import json
import os

spec = importlib.util.spec_from_file_location("snapshot_child", {snapshot_path})
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)
print("attempting", flush=True)
with snapshot.snapshot_run_lock("verify", ["child"], {lock_path}):
    print("acquired", flush=True)
""".format(
                snapshot_path=json.dumps(os.path.join(HERE, "snapshot.py")),
                lock_path=json.dumps(lock_path),
            )
            child = None
            try:
                with SNAPSHOT.snapshot_run_lock("check", ["parent"], lock_path):
                    child = subprocess.Popen(
                        [sys.executable, "-c", child_script],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                    )
                    self.assertEqual("attempting", child.stdout.readline().strip())
                    time.sleep(0.25)
                    self.assertIsNone(child.poll(), "second snapshot process bypassed the lock")

                # Continue through the same TextIOWrapper used by readline() above. Mixing that
                # buffered read with communicate() can lose a line that readline() prefetched
                # from the pipe under scheduler load (#1428).
                child.wait(timeout=5)
                output = child.stdout.read()
                self.assertEqual(0, child.returncode, output)
                self.assertIn("[snapshot-lock] waiting for pid", output)
                self.assertIn("[snapshot-lock] acquired after", output)
                self.assertTrue(output.rstrip().endswith("acquired"), output)
            finally:
                if child is not None:
                    if child.poll() is None:
                        child.kill()
                        child.wait(timeout=5)
                    if child.stdout is not None:
                        child.stdout.close()

    @unittest.skipUnless(sys.platform.startswith("linux"), "PR_SET_PDEATHSIG is Linux-specific")
    def test_snapshot_child_dies_when_harness_is_killed(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid_path = os.path.join(tmp, "child.pid")
            harness_script = """
import importlib.util
import os
import subprocess
import sys
import time

spec = importlib.util.spec_from_file_location("snapshot_harness", {snapshot_path})
snapshot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snapshot)
child = snapshot._spawn_snapshot_process(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    dict(os.environ), subprocess.DEVNULL)
with open({pid_path}, "w") as pid_file:
    pid_file.write(str(child.pid))
    pid_file.flush()
while True:
    time.sleep(1)
""".format(
                snapshot_path=json.dumps(os.path.join(HERE, "snapshot.py")),
                pid_path=json.dumps(pid_path),
            )
            harness = subprocess.Popen([sys.executable, "-c", harness_script])
            child_pid = None
            try:
                deadline = time.time() + 10
                while time.time() < deadline and not os.path.exists(pid_path):
                    if harness.poll() is not None:
                        self.fail(f"snapshot harness exited early with code {harness.returncode}")
                    time.sleep(0.05)
                self.assertTrue(os.path.exists(pid_path), "snapshot harness did not publish child PID")
                with open(pid_path) as pid_file:
                    child_pid = int(pid_file.read())
                self.assertTrue(self._process_is_running(child_pid))

                # Tolerate the harness already being gone: an absent harness still satisfies the
                # property under test (the child must not outlive it). Guards a benign TOCTOU race
                # with the harness's own exit that otherwise surfaces as a spurious ERROR (#1069).
                try:
                    os.kill(harness.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                harness.wait(timeout=5)
                deadline = time.time() + 10
                while time.time() < deadline and self._process_is_running(child_pid):
                    time.sleep(0.05)
                self.assertFalse(
                    self._process_is_running(child_pid),
                    "snapshot child survived the harness's uncatchable termination",
                )
            finally:
                # Cleanup only. A ProcessLookupError here means the target exited between the check and
                # the signal (a benign TOCTOU race with the process's own death) — that is the desired
                # end state, so tolerate it rather than turning a passing run into a spurious ERROR
                # (#1069). _process_is_running already tolerates a vanished PID via FileNotFoundError.
                if harness.poll() is None:
                    try:
                        harness.kill()
                        harness.wait(timeout=5)
                    except ProcessLookupError:
                        pass
                if child_pid is not None and self._process_is_running(child_pid):
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    @staticmethod
    def _process_is_running(pid):
        # A process that exits while we inspect /proc surfaces as one of three benign vanish signals,
        # all meaning "not running" (#1069): FileNotFoundError if /proc/<pid> is already gone at open;
        # ProcessLookupError if the process exits between open and read() (this was the actual CI flake —
        # the original code caught only FileNotFoundError); and an empty/partial stat (IndexError) if it
        # is torn down mid-read. Any of these = the PID is not a live, non-zombie process.
        try:
            with open(f"/proc/{pid}/stat") as stat_file:
                state = stat_file.read().split()[2]
            return state != "Z"
        except (FileNotFoundError, ProcessLookupError, IndexError):
            return False

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

    def test_consecutive_mode_rejects_enough_total_but_interspersed_matches(self):
        entry = {
            "min_qualifying_frames": 3,
            "min_structural_matches": 3,
            "min_nonblack_ratio": 0.5,
            "min_nonblack_matches": 3,
            "min_consecutive_content_matches": 3,
            "structural_references": [{"luma16x9": "00" * (16 * 9)}],
        }
        summary = {
            # Every aggregate floor passes. Only adjacency distinguishes the defect shape:
            # three intended frames separated by intro/attract frames are not one plateau.
            "records": [{}, {}, {}, {}, {}, {}],
            "qualifying": [{}, {}, {}],
            "pixel_changes": 5,
            "dimensions_consistent": True,
            "dims": (2, 1),
            "structural_matches": 3,
            "nonblack_matches": 6,
            "longest_content_run": [{}],
        }
        ok, failures = SNAPSHOT.content_result(entry, summary)
        self.assertFalse(ok)
        self.assertEqual(["consecutive content matches 1 < 3"], failures)

        summary["longest_content_run"] = [{}, {}, {}]
        ok, failures = SNAPSHOT.content_result(entry, summary)
        self.assertTrue(ok, failures)

    def test_consecutive_mode_scopes_pixel_progression_to_the_content_run(self):
        entry = {
            "min_qualifying_frames": 3,
            "min_pixel_changes": 2,
            "min_structural_matches": 3,
            "min_nonblack_ratio": 0.5,
            "min_nonblack_matches": 3,
            "min_consecutive_content_matches": 3,
            "structural_references": [{"luma16x9": "00" * (16 * 9)}],
        }
        summary = {
            # Intro animation satisfies the aggregate progression count, but the intended
            # content plateau is frozen. The unrelated phase must not make this guard pass.
            "records": [{}, {}, {}, {}, {}, {}],
            "qualifying": [{}, {}, {}],
            "pixel_changes": 5,
            "dimensions_consistent": True,
            "dims": (2, 1),
            "structural_matches": 3,
            "nonblack_matches": 3,
            "longest_content_run": [
                {"hash": "frozen"}, {"hash": "frozen"}, {"hash": "frozen"},
            ],
        }
        ok, failures = SNAPSHOT.content_result(entry, summary)
        self.assertFalse(ok)
        self.assertEqual(["consecutive content pixel changes 0 < 2"], failures)

        summary["longest_content_run"] = [
            {"hash": "a"}, {"hash": "b"}, {"hash": "c"},
        ]
        ok, failures = SNAPSHOT.content_result(entry, summary)
        self.assertTrue(ok, failures)

    def test_consecutive_mode_and_ratio_are_mutually_exclusive(self):
        entry = {
            "min_content_match_ratio": 0.75,
            "min_consecutive_content_matches": 3,
            "structural_references": [{"luma16x9": "00" * (16 * 9)}],
        }
        expected = (
            "min_content_match_ratio and min_consecutive_content_matches are mutually exclusive")
        with self.assertRaisesRegex(ValueError, expected):
            SNAPSHOT.validate_content_entry(entry)
        ok, failures = SNAPSHOT.content_result(entry, {})
        self.assertFalse(ok)
        self.assertEqual([expected], failures)

    def test_consecutive_mode_requires_existing_reference_seed(self):
        entry = {"min_consecutive_content_matches": 3}
        with self.assertRaisesRegex(
                ValueError, "requires existing structural_references to seed plateau"):
            SNAPSHOT.validate_content_entry(entry)

    def test_missing_sample_breaks_consecutive_content_run(self):
        signature = tuple([0] * (16 * 9))
        entry = {
            "min_colors": 300,
            "dims": [2, 1],
            "min_nonblack_ratio": 0.5,
            "min_structural_similarity": 0.85,
            "structural_references": [{"luma16x9": "00" * (16 * 9)}],
        }
        records = [
            {
                "index": index, "colors": 335, "dims": (2, 1),
                "nonblack_ratio": 1.0, "luma16x9": signature,
            }
            for index in (4, 5, 7, 8)
        ]
        self.assertEqual(2, len(SNAPSHOT.longest_content_run(entry, records)))

    def test_default_ratio_mode_still_uses_the_whole_window(self):
        entry = {
            "min_qualifying_frames": 2,
            "min_structural_matches": 2,
            "min_nonblack_ratio": 0.5,
            "min_nonblack_matches": 2,
            "min_content_match_ratio": 0.75,
            "structural_references": [{"luma16x9": "00" * (16 * 9)}],
        }
        summary = {
            "records": [{} for _ in range(8)],
            "qualifying": [{}, {}],
            "pixel_changes": 0,
            "dimensions_consistent": True,
            "dims": (2, 1),
            "structural_matches": 5,
            "nonblack_matches": 5,
        }
        ok, failures = SNAPSHOT.content_result(entry, summary)
        self.assertFalse(ok)
        self.assertEqual([
            "structural matches 5 < 6 (minimum SSIM 0.85)",
            "non-black coverage matches 5 < 6 (minimum ratio 0.5)",
        ], failures)

    def test_cross_run_streak_uses_source_only_references_and_preserves_order(self):
        menu = tuple([0] * 72 + [255] * 72)
        negative = tuple([255] * 72 + [0] * 72)
        entry = {
            "min_colors": 300,
            "dims": [2, 1],
            "min_nonblack_ratio": 0.5,
            "min_structural_similarity": 0.85,
        }

        def record(index, signature):
            return {
                "index": index,
                "path": f"frame-{index}.png", "filename": f"frame-{index}.png",
                "elapsed": float(index), "colors": 335, "dims": (2, 1),
                "nonblack_ratio": 1.0, "luma16x9": signature,
            }

        source_run = [record(index, menu) for index in range(3)]
        source = {
            "records": source_run, "longest_content_run": source_run,
            "richest": source_run[0],
        }
        target_run = [record(10, negative)] + [record(index, menu) for index in range(11, 14)]
        target = {"records": target_run}
        self.assertEqual(3, SNAPSHOT.cross_run_content_streak(entry, source, target))
        self.assertEqual(
            [11, 12, 13],
            [item["index"] for item in SNAPSHOT.cross_run_content_run(entry, source, target)],
        )

        interspersed = [
            record(20, menu), record(21, negative), record(22, menu),
            record(23, negative), record(24, menu),
        ]
        self.assertEqual(
            1, SNAPSHOT.cross_run_content_streak(entry, source, {"records": interspersed}))

    def test_cross_run_plateau_excludes_transition_admitted_by_seed(self):
        menu = tuple([0] * 72 + [255] * 72)
        transition = tuple([255] * 72 + [0] * 72)
        entry = {
            "min_colors": 300,
            "dims": [2, 1],
            "min_nonblack_ratio": 0.5,
            "min_structural_similarity": 0.85,
            # The old reviewed seed contains both phases, so it alone admits the transition.
            "structural_references": [
                {"luma16x9": bytes(menu).hex()},
                {"luma16x9": bytes(transition).hex()},
            ],
        }

        def record(index, signature):
            return {
                "index": index,
                "path": f"frame-{index}.png", "filename": f"frame-{index}.png",
                "elapsed": float(index), "colors": 335, "dims": (2, 1),
                "nonblack_ratio": 1.0, "luma16x9": signature,
            }

        run_a_records = [record(index, menu) for index in range(4)]
        run_b_records = [record(10, transition)] + [
            record(index, menu) for index in range(11, 14)
        ]
        run_a = {
            "records": run_a_records,
            "longest_content_run": SNAPSHOT.longest_content_run(entry, run_a_records),
            "richest": run_a_records[0],
        }
        run_b = {
            "records": run_b_records,
            "longest_content_run": SNAPSHOT.longest_content_run(entry, run_b_records),
            "richest": run_b_records[0],
        }
        self.assertEqual([10, 11, 12, 13], [
            item["index"] for item in run_b["longest_content_run"]])
        self.assertEqual([11, 12, 13], [
            item["index"] for item in SNAPSHOT.cross_run_content_run(entry, run_a, run_b)])

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
            # BOTH save roots must be redirected into the run's temp dir, or a "fresh" guard
            # inherits the developer's real saves: PROSPER_SAVEDATA_DIR covers SaveDataMemory (the
            # Unity titles) and PROSPER_SAVE0 covers the /savedata0 file mount (Blasphemous 2).
            # Leaving either on its shared /tmp default lets existing progress carry a blind route
            # past the content it guards, so the guard can pass without rendering that scene (#1102).
            self.assertTrue(os.path.isdir(env["PROSPER_SAVEDATA_DIR"]))
            self.assertTrue(os.path.isdir(env["PROSPER_SAVE0"]))
            self.assertTrue(env["PROSPER_SAVEDATA_DIR"].startswith(tmp))
            self.assertTrue(env["PROSPER_SAVE0"].startswith(tmp))
            self.assertNotEqual(env["PROSPER_SAVEDATA_DIR"], env["PROSPER_SAVE0"])

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
