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

    def test_compute_programs_group_by_stable_hash_and_keep_unknown_explicit(self):
        summary = summarize(capture(SAMPLES, compute=[
            {"total_ms": 10, "dispatches": 1, "program_addr": 0x1000,
             "program_hash": 0xABC},
            {"total_ms": 20, "dispatches": 2, "program_addr": 0x2000,
             "program_hash": 0xABC},
            {"total_ms": 5, "dispatches": 1, "program_addr": 0x3000,
             "program_hash": 0xDEF},
            {"total_ms": 7, "dispatches": 3, "program_addr": None,
             "program_hash": None},
        ]))
        programs = summary["compute_programs"]
        self.assertEqual(programs["group_count"], 2)
        self.assertEqual(programs["unknown_records"], 1)
        self.assertEqual(programs["unknown_dispatches"], 3)
        self.assertEqual(programs["unknown_total_ms"], 7)
        dominant = programs["groups"][0]
        self.assertEqual(dominant["program_hash"], "0x0000000000000abc")
        self.assertEqual(dominant["addresses"], [
            "0x0000000000001000", "0x0000000000002000"])
        self.assertEqual(dominant["records"], 2)
        self.assertEqual(dominant["dispatches"], 3)
        self.assertEqual(dominant["total_ms"], 30)
        self.assertEqual(dominant["mean_ms"], 15)
        self.assertEqual(dominant["max_ms"], 20)

    def test_cpu_outside_renderer_classification(self):
        summary = summarize(capture(SAMPLES, renderer=[{"total_ms": 100}]))
        self.assertEqual(summary["classification"], "cpu-outside-renderer")
        self.assertAlmostEqual(summary["cpu_cores"], 1.0, places=3)

    def test_inconclusive_without_post_population(self):
        summary = summarize(capture())
        self.assertEqual(summary["classification"], "inconclusive")
        self.assertIsNone(summary["seconds"])

    # #3152/#3153. A readback verdict means two opposite things, and the capture can tell them
    # apart: `rendered_frame_counter` returns nullopt exactly when GPU present was adopted, so a
    # null rendered-frame population IS the GPU-present signal.
    READBACK_RENDERER = [{
        "total_ms": 100, "gpu_device_ms": 5, "gpu_wait_ms": 6,
        "build_resources_ms": 8, "setup_resources_ms": 7, "readback_ms": 60,
    }]

    def test_readback_without_gpu_present_is_called_out_as_the_harness(self):
        # No GPU present -> the frontend copied every scanout frame to the CPU. Optimising that is
        # optimising the measuring apparatus, which is exactly what the offscreen capture of The
        # Forgotten City invited.
        no_gpu_present = [dict(s) for s in SAMPLES]          # rendered_frames present => NOT adopted
        summary = summarize(capture(no_gpu_present, renderer=self.READBACK_RENDERER))
        self.assertEqual(summary["classification"], "readback")
        self.assertIsNotNone(summary["readback_note"])
        self.assertIn("harness, not the title", summary["readback_note"])

    def test_readback_with_gpu_present_is_called_out_as_real_work(self):
        # GPU present adopted (rendered_frames unavailable) -> scanout readback is skipped, so this
        # readback is genuine (ordered-DMA authoritative copies, storage writeback). The note must
        # push TOWARDS investigating it; a blanket "probably your harness" would teach readers to
        # dismiss a real signal.
        gpu_present = [{k: v for k, v in s.items() if k != "rendered_frames"} for s in SAMPLES]
        summary = summarize(capture(gpu_present, renderer=self.READBACK_RENDERER))
        self.assertEqual(summary["classification"], "readback")
        self.assertIsNotNone(summary["readback_note"])
        self.assertIn("real", summary["readback_note"])
        self.assertNotIn("harness, not the title", summary["readback_note"])

    def test_degenerate_sample_population_does_not_flip_an_offscreen_capture(self):
        # THE REGRESSION. The discriminator used to be `rendered_fps is not None`, but that derived
        # rate is also None for a population with fewer than two samples, a non-increasing t_ns, or a
        # zero window. An OFFSCREEN capture -- whose samples DO carry a real rendered_frames counter,
        # so GPU present was not adopted -- then took the GPU-present branch and was told its
        # readback was real work, sending the reader to optimise the harness. Key on the field.
        # Asserted, not guarded: `if classification == "readback"` would make these arms silently
        # vacuous if a future change classified a degenerate capture differently, and a vacuous arm
        # on the exact regression it was written for is worse than no arm.
        one_sample = [dict(SAMPLES[0])]
        summary = summarize(capture(one_sample, renderer=self.READBACK_RENDERER))
        self.assertEqual(summary["classification"], "readback")
        self.assertIn("harness, not the title", summary["readback_note"])

        frozen_clock = [dict(SAMPLES[0]), dict(SAMPLES[0])]     # equal t_ns => rate is None
        summary = summarize(capture(frozen_clock, renderer=self.READBACK_RENDERER))
        self.assertEqual(summary["classification"], "readback")
        self.assertIn("harness, not the title", summary["readback_note"])

    def test_mixed_gpu_present_population_says_it_cannot_tell(self):
        # Adopted and then lost mid-capture. Neither branch is honest, so the note must decline
        # rather than pick one -- a wrong confident answer here is the failure being prevented.
        mixed = [dict(SAMPLES[0]), {k: v for k, v in SAMPLES[1].items() if k != "rendered_frames"}]
        summary = summarize(capture(mixed, renderer=self.READBACK_RENDERER))
        self.assertEqual(summary["classification"], "readback")
        self.assertIn("cannot say", summary["readback_note"])

    def test_explicit_null_rendered_frames_is_the_real_wire_shape(self):
        # `write_optional` serializes an adopted GPU present as the KEY PRESENT WITH A NULL VALUE,
        # not as an absent key. The other GPU-present arm omits the key, so without this one the
        # discriminator is never tested against the shape a real capture actually carries.
        wire = [{**s, "rendered_frames": None} for s in SAMPLES]
        summary = summarize(capture(wire, renderer=self.READBACK_RENDERER))
        self.assertEqual(summary["classification"], "readback")
        self.assertIn("real", summary["readback_note"])
        self.assertNotIn("harness, not the title", summary["readback_note"])

    def test_no_post_samples_cannot_determine_gpu_present(self):
        # The one uncovered line in the helper. With no post population there is nothing to read the
        # field from, so the honest answer is the third state rather than either branch.
        summary = summarize(capture([], renderer=self.READBACK_RENDERER))
        if summary["classification"] == "readback":
            self.assertIn("cannot say", summary["readback_note"])
        else:
            self.assertIsNone(summary["readback_note"])

    def test_non_readback_verdict_carries_no_readback_note(self):
        # The note is specific to the readback verdict; on every report it would be noise.
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 100, "gpu_device_ms": 10, "gpu_wait_ms": 12,
            "build_resources_ms": 25, "setup_resources_ms": 25, "readback_ms": 2,
        }]))
        self.assertEqual(summary["classification"], "renderer-resource")
        self.assertIsNone(summary["readback_note"])

    def test_material_readback_is_called_out_even_when_not_the_verdict(self):
        # The Dragon Quest shape: readback is a large minority of measured work and loses the
        # classification, so a ranking- or verdict-gated note stays silent on exactly the number
        # a reader would chase. 300 of 1000ms is 30%.
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 1000, "gpu_device_ms": 100, "gpu_wait_ms": 100,
            "build_resources_ms": 250, "setup_resources_ms": 250, "readback_ms": 300,
        }]))
        self.assertNotEqual(summary["classification"], "readback")
        self.assertIsNotNone(summary["readback_note"])
        self.assertIn("NOT adopted", summary["readback_note"])

    def test_pacing_gap_is_evidence_not_cause(self):
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 100, "gpu_device_ms": 60, "gpu_timestamp_samples": 1,
        }]))
        self.assertIn("does not assign its cause", summary["pacing_note"])

    def test_direct_present_unavailable_render_counter_declines_pacing_inference(self):
        direct_present = [dict(sample, rendered_frames=None) for sample in SAMPLES]
        summary = summarize(capture(direct_present, renderer=[{"total_ms": 1}]))
        self.assertEqual(summary["rates"]["guest_fps"], 60)
        self.assertEqual(summary["rates"]["host_fps"], 9)
        self.assertIsNone(summary["rates"]["rendered_fps"])
        self.assertIsNone(summary["pacing_note"])

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
