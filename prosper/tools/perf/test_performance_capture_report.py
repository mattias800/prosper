#!/usr/bin/env python3

import unittest

from performance_capture_report import CaptureError, summarize, validate_capture


def capture(post=None, renderer=None, compute=None, dropped=(0, 0)):
    post = post or []
    renderer = renderer or []
    compute = compute or []
    records = [{
        "type": "header", "format": "prosper-performance-capture", "version": 1,
        "title_id": "TEST00001", "title": "Test Title", "revision": "abc123",
    }]
    records.extend({"type": "sample", "phase": "post", **sample} for sample in post)
    records.extend({"type": "renderer", **record} for record in renderer)
    records.extend({"type": "compute", **record} for record in compute)
    records.append({
        "type": "footer", "complete": True,
        "pre_samples": 0, "post_samples": len(post),
        "renderer_records": len(renderer), "compute_records": len(compute),
        "renderer_dropped": dropped[0], "compute_dropped": dropped[1],
    })
    return records


SAMPLES = [
    {"t_ns": 0, "process_cpu_ns": 10, "rss_bytes": 100,
     "guest_presents": 0, "rendered_frames": 0, "host_presented_frames": 0},
    {"t_ns": 1_000_000_000, "process_cpu_ns": 1_000_000_010, "rss_bytes": 200,
     "guest_presents": 60, "rendered_frames": 10, "host_presented_frames": 9},
]


class PerformanceCaptureReportTests(unittest.TestCase):
    def test_gpu_device_classification(self):
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 100, "gpu_device_ms": 60, "gpu_wait_ms": 65,
            "gpu_timestamp_samples": 2,
            "build_resources_ms": 5, "setup_resources_ms": 5, "readback_ms": 2,
        }]))
        self.assertEqual(summary["classification"], "gpu-device")
        self.assertIn("60%", summary["reason"])
        self.assertTrue(summary["gpu_timestamps_available"])
        self.assertEqual(summary["components"]["gpu-wait-overhead"], 5)

    def test_gpu_wait_stays_unsplit_when_device_timestamps_are_unavailable(self):
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 100, "gpu_device_ms": 0, "gpu_wait_ms": 65,
            "gpu_timestamp_samples": 0,
        }]))
        self.assertEqual(summary["classification"], "gpu-wait")
        self.assertFalse(summary["gpu_timestamps_available"])
        self.assertEqual(summary["components"]["gpu-wait"], 65)
        self.assertIsNone(summary["components"]["gpu-device"])
        self.assertIsNone(summary["components"]["gpu-wait-overhead"])

    def test_renderer_resource_classification(self):
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 100, "gpu_device_ms": 10, "gpu_wait_ms": 12,
            "build_resources_ms": 25, "setup_resources_ms": 25, "readback_ms": 2,
        }]))
        self.assertEqual(summary["classification"], "renderer-resource")

    def test_compute_classification(self):
        summary = summarize(capture(SAMPLES,
            renderer=[{"total_ms": 20}], compute=[{"total_ms": 80}]))
        self.assertEqual(summary["classification"], "compute")

    def test_cpu_outside_renderer_classification(self):
        summary = summarize(capture(SAMPLES, renderer=[{"total_ms": 100}]))
        self.assertEqual(summary["classification"], "cpu-outside-renderer")
        self.assertAlmostEqual(summary["cpu_cores"], 1.0, places=3)

    def test_inconclusive_without_post_population(self):
        summary = summarize(capture())
        self.assertEqual(summary["classification"], "inconclusive")
        self.assertIsNone(summary["seconds"])

    def test_pacing_gap_is_evidence_not_cause(self):
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 100, "gpu_device_ms": 60, "gpu_timestamp_samples": 1,
        }]))
        self.assertIn("does not assign its cause", summary["pacing_note"])

    def test_dropped_counts_are_reported_not_added_to_population(self):
        summary = summarize(capture(SAMPLES, renderer=[{"total_ms": 1}], dropped=(7, 9)))
        self.assertEqual(summary["counts"]["renderer"], 1)
        self.assertEqual(summary["truncation"], "detail truncated: renderer dropped 7, compute dropped 9")

    def test_footer_must_match_actual_population(self):
        records = capture(SAMPLES)
        records[-1]["post_samples"] = 99
        with self.assertRaisesRegex(CaptureError, "post_samples"):
            validate_capture(records)

    def test_incomplete_file_is_rejected(self):
        records = capture(SAMPLES)
        records.pop()
        with self.assertRaisesRegex(CaptureError, "incomplete"):
            validate_capture(records)


if __name__ == "__main__":
    unittest.main(verbosity=2)
