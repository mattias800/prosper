#!/usr/bin/env python3
import contextlib
import difflib
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone


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

    def test_guest_fault_is_named_in_the_exit_status_message(self):
        # The frontend's own `[shot] guest fault:` line goes to the run log, which is DEVNULL unless
        # the caller asked for one. Without this the operator sees a bare exit code for a run whose
        # guest crashed -- the same unreadable result as #2007 itself.
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "capture.jsonl")
            with open(manifest, "w") as f:
                f.write(json.dumps({
                    "type": "summary", "exit_code": 1, "saved": 25, "requested": 25,
                    "guest_state": "faulted", "guest_kind": 2,
                    "guest_detail": "SIGSEGV at addr=(nil)  rip=0x410024055 (image+0x24055)",
                    "guest_fault_rip": "0x410024055", "status": "GUEST-FAULT",
                }) + "\n")
            suffix = SNAPSHOT._guest_failure_suffix(manifest)
            self.assertIn("primary thread faulted", suffix)
            self.assertIn("image+0x24055", suffix)

            # An intact guest adds nothing, so the suffix cannot be reporting a fault for every
            # failing run -- the assertion above would pass either way without this arm.
            with open(manifest, "w") as f:
                f.write(json.dumps({
                    "type": "summary", "exit_code": 1, "saved": 2, "requested": 3,
                    "guest_state": "running", "status": "FAILED",
                }) + "\n")
            self.assertEqual("", SNAPSHOT._guest_failure_suffix(manifest))
            # A manifest an older binary wrote, or none at all, must not raise.
            with open(manifest, "w") as f:
                f.write(json.dumps({"type": "summary", "exit_code": 1}) + "\n")
            self.assertEqual("", SNAPSHOT._guest_failure_suffix(manifest))
            self.assertEqual("", SNAPSHOT._guest_failure_suffix(os.path.join(tmp, "absent.jsonl")))

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


class BaselineVerificationMetadataTests(unittest.TestCase):
    """`verified_at` / `verified_by`: when a baseline was last established, and by whom.

    Each arm asserts a BEHAVIOUR that would change if the feature were reverted or inverted, not
    that a key exists. The default-is-agent arm and the check-never-stamps arm are the two that
    matter: between them they are the reason a `verified_by: human` in the manifest may be believed.
    """

    def setUp(self):
        self._saved_review_dir = SNAPSHOT.REVIEW_DIR
        self._saved_manifest = SNAPSHOT.MANIFEST
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self._restore)
        SNAPSHOT.REVIEW_DIR = os.path.join(self.tmp.name, "review")
        SNAPSHOT.MANIFEST = os.path.join(self.tmp.name, "snapshots.json")

    def _restore(self):
        SNAPSHOT.REVIEW_DIR = self._saved_review_dir
        SNAPSHOT.MANIFEST = self._saved_manifest

    ISO_UTC = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"

    def _content_entry(self, name="gameplay"):
        return {"name": name, "dump": "PPSAXXXXX-app0", "min_colors": 10,
                "min_structural_similarity": 0.85}

    def _stage_content_candidate(self, entry):
        """Build the review evidence `approve_content_candidate` insists on, with no capture."""
        summaries = []
        for run in (1, 2):
            run_records = []
            for index in range(2):
                path = os.path.join(SNAPSHOT.REVIEW_DIR, entry["name"], f"run{run}_{index:02d}.png")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "wb") as image:
                    image.write(b"reviewed image")
                run_records.append({
                    "path": path, "elapsed": run * 10 + index, "colors": 20, "dims": (4, 3),
                    "average_hash": run * 10 + index, "difference_hash": run * 20 + index,
                    "luma16x9": (run * 10 + index,) * (16 * 9), "nonblack_ratio": 0.8,
                })
            summaries.append({"records": run_records, "qualifying": run_records,
                              "richest": run_records[0], "dims": (4, 3)})
        SNAPSHOT.save_content_candidate(entry, summaries)

    def _stage_exact_candidate(self, entry):
        first = os.path.join(self.tmp.name, "first.bmp")
        second = os.path.join(self.tmp.name, "second.bmp")
        for path in (first, second):
            with open(path, "wb") as image:
                image.write(b"exact frame")
        SNAPSHOT.save_exact_candidate(entry, first, second, "d" * 64, (4, 3))

    # --- the default, and the assertion that overrides it -----------------------------------------

    def test_content_approval_defaults_to_agent_verification(self):
        entry = self._content_entry()
        self._stage_content_candidate(entry)
        SNAPSHOT.approve_content_candidate(entry)
        self.assertEqual("agent", entry["verified_by"])
        self.assertRegex(entry["verified_at"], self.ISO_UTC)

    def test_exact_approval_defaults_to_agent_verification(self):
        entry = {"name": "exact", "dump": "PPSAXXXXX-app0", "frame": 7}
        self._stage_exact_candidate(entry)
        SNAPSHOT.approve_exact_candidate(entry)
        self.assertEqual("agent", entry["verified_by"])
        self.assertRegex(entry["verified_at"], self.ISO_UTC)

    def test_human_verification_requires_the_explicit_assertion(self):
        # The whole point of the field is that `human` is worth more than `agent`, so it must be
        # unreachable except by saying so. Nothing about the environment may produce it.
        for variables in ({}, {"USER": "mattias"}, {"CI": ""}, {"TERM": "xterm-256color"},
                          {"PROSPER_SNAPSHOT_VERIFIED_BY": "human"}):
            with self.subTest(env=variables):
                saved = {key: os.environ.get(key) for key in variables}
                os.environ.update(variables)
                try:
                    entry = self._content_entry()
                    self._stage_content_candidate(entry)
                    SNAPSHOT.approve_content_candidate(entry)
                    self.assertEqual("agent", entry["verified_by"])
                finally:
                    for key, value in saved.items():
                        if value is None:
                            os.environ.pop(key, None)
                        else:
                            os.environ[key] = value

        entry = self._content_entry()
        self._stage_content_candidate(entry)
        SNAPSHOT.approve_content_candidate(entry, SNAPSHOT.VERIFIED_BY_HUMAN)
        self.assertEqual("human", entry["verified_by"])

    def test_resolve_verified_by_is_agent_unless_the_option_says_human(self):
        self.assertEqual("agent", SNAPSHOT.resolve_verified_by({}))
        self.assertEqual("agent", SNAPSHOT.resolve_verified_by({"--reviewed": True}))
        self.assertEqual("agent", SNAPSHOT.resolve_verified_by({"--verified-by": "agent"}))
        self.assertEqual("human", SNAPSHOT.resolve_verified_by({"--verified-by": "human"}))
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
            SNAPSHOT.resolve_verified_by({"--verified-by": "Human"})
        self.assertEqual(2, refused.exception.code)

    def test_stamp_verification_refuses_an_unrecognised_actor(self):
        for actor in ("HUMAN", "person", "", None, True):
            with self.subTest(actor=actor), self.assertRaises(ValueError):
                SNAPSHOT.stamp_verification({"name": "x"}, actor)

    # --- checking is not verifying ----------------------------------------------------------------

    def test_check_never_stamps_a_verification(self):
        # Conflating check with verify is the failure this feature exists to prevent: a routine
        # regression run must not be able to refresh a date nobody re-reviewed. Two arms, both
        # reachable without a GPU: the early no-baseline refusal, and the full loop with a baseline
        # whose capture then fails.
        def refuse_to_save(_manifest):
            raise AssertionError("check wrote the manifest")

        saved_save, saved_fail_dir = SNAPSHOT.save_manifest, SNAPSHOT.FAIL_DIR
        saved_boot_trace = os.environ.get("PROSPER_BOOT_TRACE")
        SNAPSHOT.save_manifest = refuse_to_save
        SNAPSHOT.FAIL_DIR = os.path.join(self.tmp.name, "failures")
        os.environ["PROSPER_BOOT_TRACE"] = os.path.join(self.tmp.name, "no-such-boot-trace")
        try:
            no_baseline = {"name": "unbaselined", "dump": "PPSAXXXXX-app0", "frame": 3}
            with_baseline = {"name": "baselined", "dump": "PPSAXXXXX-app0", "frame": 3,
                             "hash": "a" * 64, "review": "inspected by hand"}
            manifest = {"snapshots": [no_baseline, with_baseline]}
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                rc = SNAPSHOT.cmd_check(manifest, [], {})
            self.assertEqual(1, rc, "both arms must reach a failure, not a pass")
            for entry in (no_baseline, with_baseline):
                self.assertNotIn("verified_at", entry)
                self.assertNotIn("verified_by", entry)
        finally:
            SNAPSHOT.save_manifest, SNAPSHOT.FAIL_DIR = saved_save, saved_fail_dir
            if saved_boot_trace is None:
                os.environ.pop("PROSPER_BOOT_TRACE", None)
            else:
                os.environ["PROSPER_BOOT_TRACE"] = saved_boot_trace

    def test_update_writes_the_stamp_that_check_withholds(self):
        # The positive half of the arm above: the same manifest, the same helpers, and the field
        # DOES appear — so "check leaves it absent" is a fact about check, not about the fixture.
        entry = self._content_entry()
        self._stage_content_candidate(entry)
        manifest = {"snapshots": [entry]}
        with contextlib.redirect_stdout(io.StringIO()):
            rc = SNAPSHOT.cmd_update(manifest, ["gameplay"],
                                     {"--reviewed": True, "--verified-by": "human"})
        self.assertEqual(0, rc)
        self.assertEqual("human", entry["verified_by"])
        self.assertRegex(entry["verified_at"], self.ISO_UTC)
        with open(SNAPSHOT.MANIFEST, encoding="utf-8") as written:
            self.assertEqual(manifest, json.load(written))

    def test_unnamed_update_is_refused_rather_than_stamping_every_guard(self):
        entries = [self._content_entry("one"), self._content_entry("two")]
        with self.assertRaises(SystemExit) as refused, contextlib.redirect_stderr(io.StringIO()):
            SNAPSHOT.cmd_update({"snapshots": entries}, [], {"--reviewed": True})
        self.assertEqual(2, refused.exception.code)
        for entry in entries:
            self.assertNotIn("verified_at", entry)

    # --- no invented history ----------------------------------------------------------------------

    def test_repository_manifest_round_trips_byte_for_byte(self):
        # A rewrite that reflowed or reordered the other entries would bury every future baseline
        # change in noise, so the read/write pair must be the identity on an untouched manifest.
        real_manifest = os.path.join(HERE, "snapshots.json")
        with open(real_manifest, "rb") as source:
            original = source.read()
        SNAPSHOT.MANIFEST = real_manifest
        manifest = SNAPSHOT.load_manifest()
        SNAPSHOT.MANIFEST = os.path.join(self.tmp.name, "roundtrip.json")
        SNAPSHOT.save_manifest(manifest)
        with open(SNAPSHOT.MANIFEST, "rb") as written:
            self.assertEqual(original, written.read())

    def test_stamping_one_guard_leaves_every_other_entry_untouched(self):
        real_manifest = os.path.join(HERE, "snapshots.json")
        SNAPSHOT.MANIFEST = real_manifest
        manifest = SNAPSHOT.load_manifest()
        before = [dict(entry) for entry in manifest["snapshots"]]
        target = manifest["snapshots"][0]["name"]
        SNAPSHOT.stamp_verification(manifest["snapshots"][0], SNAPSHOT.VERIFIED_BY_HUMAN)

        SNAPSHOT.MANIFEST = os.path.join(self.tmp.name, "stamped.json")
        SNAPSHOT.save_manifest(manifest)
        reloaded = SNAPSHOT.load_manifest()["snapshots"]
        self.assertEqual(len(before), len(reloaded))
        for original, current in zip(before, reloaded):
            if current["name"] == target:
                self.assertEqual("human", current["verified_by"])
                self.assertEqual(set(original) | {"verified_at", "verified_by"}, set(current))
                continue
            self.assertEqual(original, current)
            self.assertNotIn("verified_at", current)

        # And the on-disk diff really is a two-line append, not a reflow of the whole file. The
        # only line that can legitimately DISAPPEAR is the stamped entry's former last key, which
        # comes back verbatim with the JSON comma the append requires.
        with open(real_manifest, encoding="utf-8") as source:
            old_lines = source.read().splitlines()
        with open(SNAPSHOT.MANIFEST, encoding="utf-8") as written:
            new_lines = written.read().splitlines()
        delta = [line for line in difflib.unified_diff(old_lines, new_lines, n=0)
                 if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))]
        removed = [line[1:] for line in delta if line.startswith("-")]
        added = [line[1:] for line in delta if line.startswith("+")]
        self.assertEqual(1, len(removed), delta)
        self.assertEqual([removed[0] + ","], added[:1], delta)
        self.assertRegex(added[1], r'^ +"verified_at": "\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",$')
        self.assertRegex(added[2], r'^ +"verified_by": "human"$')
        self.assertEqual(3, len(added), delta)

    def test_no_shipped_entry_carries_a_verification_stamp(self):
        # The 17 entries in the repository predate the fields. Absence is the honest record of
        # "never recorded"; a date inferred from `review` prose or git history would read as fact.
        SNAPSHOT.MANIFEST = os.path.join(HERE, "snapshots.json")
        stamped = [entry["name"] for entry in SNAPSHOT.load_manifest()["snapshots"]
                   if "verified_at" in entry or "verified_by" in entry]
        self.assertEqual([], stamped, "existing baselines must not be backfilled")

    def test_fingerprint_ignores_verification_metadata(self):
        # Provenance is not contract. If it were hashed, `update NAME --reviewed` run twice would
        # refuse the second time with "snapshot configuration changed after verify".
        base = {"name": "x", "frame": 7}
        stamped = SNAPSHOT.stamp_verification(dict(base), SNAPSHOT.VERIFIED_BY_HUMAN)
        self.assertEqual(SNAPSHOT.entry_fingerprint(base), SNAPSHOT.entry_fingerprint(stamped))
        self.assertNotEqual(SNAPSHOT.entry_fingerprint(base),
                            SNAPSHOT.entry_fingerprint(dict(base, frame=8)))

    def test_a_second_approval_is_not_refused_as_a_configuration_change(self):
        entry = self._content_entry()
        self._stage_content_candidate(entry)
        SNAPSHOT.approve_content_candidate(entry)
        first_stamp = entry["verified_at"]
        SNAPSHOT.approve_content_candidate(entry, SNAPSHOT.VERIFIED_BY_HUMAN)
        self.assertEqual("human", entry["verified_by"])
        self.assertGreaterEqual(entry["verified_at"], first_stamp)

    # --- visibility ---------------------------------------------------------------------------------

    def test_list_distinguishes_never_verified_from_a_recent_human_verification(self):
        now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
        never = {"name": "never-verified", "dump": "PPSA00001-app0", "min_colors": 10}
        fresh = SNAPSHOT.stamp_verification(
            {"name": "fresh-human", "dump": "PPSA00002-app0", "min_colors": 10},
            SNAPSHOT.VERIFIED_BY_HUMAN, now)
        stale = SNAPSHOT.stamp_verification(
            {"name": "stale-agent", "dump": "PPSA00003-app0", "min_colors": 10},
            SNAPSHOT.VERIFIED_BY_AGENT, now - timedelta(days=48, hours=3))

        rendered = {entry["name"]: SNAPSHOT.format_verification(entry, now)
                    for entry in (never, fresh, stale)}
        self.assertEqual("NEVER VERIFIED", rendered["never-verified"])
        self.assertNotIn("NEVER", rendered["fresh-human"])
        self.assertIn("2026-09-02", rendered["fresh-human"])
        self.assertIn("human", rendered["fresh-human"])
        self.assertIn("0d", rendered["fresh-human"])
        self.assertIn("agent", rendered["stale-agent"])
        self.assertIn("48d", rendered["stale-agent"])
        self.assertEqual(3, len(set(rendered.values())), rendered)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            SNAPSHOT.cmd_list({"snapshots": [never, fresh, stale]}, [], {})
        printed = out.getvalue()
        self.assertIn("NEVER VERIFIED", printed)
        self.assertIn("3 guards: 1 never verified, 1 agent-verified, 1 human-verified", printed)
        never_line = next(l for l in printed.splitlines() if l.startswith("  never-verified"))
        human_line = next(l for l in printed.splitlines() if l.startswith("  fresh-human"))
        self.assertNotEqual(never_line[28:50], human_line[28:50])

    def test_an_unreadable_stamp_is_reported_rather_than_shown_as_a_date(self):
        for broken in ("yesterday", "2026-13-45T00:00:00Z", "", 20260902, None):
            with self.subTest(stamp=broken):
                self.assertIsNone(SNAPSHOT.verification_age_days(broken))
        rendered = SNAPSHOT.format_verification({"verified_at": "yesterday", "verified_by": "human"})
        self.assertIn("UNPARSEABLE", rendered)

    def test_verification_age_is_measured_in_whole_days_from_the_stamp(self):
        now = datetime(2026, 9, 2, 12, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(0, SNAPSHOT.verification_age_days("2026-09-02T00:00:00Z", now))
        self.assertEqual(32, SNAPSHOT.verification_age_days("2026-08-01T00:00:00Z", now))
        # A non-UTC offset must be normalised, not compared as wall-clock text.
        self.assertEqual(0, SNAPSHOT.verification_age_days("2026-09-02T14:00:00+02:00", now))

    def test_utc_timestamp_is_sortable_iso_8601_in_utc(self):
        stamp = SNAPSHOT.utc_timestamp(datetime(2026, 9, 2, 14, 30, 5,
                                                tzinfo=timezone(timedelta(hours=2))))
        self.assertEqual("2026-09-02T12:30:05Z", stamp)
        self.assertRegex(SNAPSHOT.utc_timestamp(), self.ISO_UTC)
        self.assertLess("2026-08-31T23:59:59Z", stamp)          # plain string sort is chronological

    # --- the option parser this feature added -------------------------------------------------------

    def test_an_unrecognised_option_is_refused_not_read_as_a_guard_name(self):
        # Guard names are positional, so before this a mistyped flag became a name -- and for
        # `list`, which ignored its arguments entirely, it vanished without a word.
        for command, tokens in (("update", ["--reviewd"]),
                                ("update", ["--verified-by=human", "--force"]),
                                ("check", ["--reviewed"]),
                                ("verify", ["--verified-by=human"]),
                                ("list", ["--verified-by=human"])):
            with self.subTest(command=command, tokens=tokens):
                with self.assertRaises(SystemExit) as refused, \
                        contextlib.redirect_stderr(io.StringIO()) as err:
                    SNAPSHOT.split_options(command, tokens)
                self.assertEqual(2, refused.exception.code)
                self.assertIn("unknown option", err.getvalue())

    def test_malformed_option_values_are_refused_in_both_directions(self):
        for tokens, expected in ((["--verified-by"], "needs a value"),
                                 (["--reviewed=yes"], "takes no value")):
            with self.subTest(tokens=tokens):
                with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as err:
                    SNAPSHOT.split_options("update", tokens)
                self.assertIn(expected, err.getvalue())

    def test_guard_names_and_options_are_separated_in_any_order(self):
        options, names = SNAPSHOT.split_options(
            "update", ["gris-gameplay", "--reviewed", "worms-armageddon-gameplay",
                       "--verified-by=human"])
        self.assertEqual({"--reviewed": True, "--verified-by": "human"}, options)
        self.assertEqual(["gris-gameplay", "worms-armageddon-gameplay"], names)


if __name__ == "__main__":
    unittest.main()
