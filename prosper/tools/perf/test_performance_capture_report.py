#!/usr/bin/env python3

import unittest

import contextlib
import io

from performance_capture_report import (CaptureError, CLASSIFICATION_EVIDENCE_SHARE,
                                        READBACK_NOTE_MIN_SHARE, print_summary, summarize,
                                        validate_capture)
from performance_capture_report import summarize_handoffs


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
    def test_pending_queue_observations_keep_signed_deadlines_and_missing_samples(self):
        queue = dict(queued=123, active_submits=2, inflight_batches=1, front_item_age_ns=456,
                     release_delay_ns=-789, scope_begins=12, scope_ends=10, deadline_resets=3)
        records = capture([dict(SAMPLES[0], pending_writes=queue), SAMPLES[1]])
        records[0].update(present_handoffs_enabled=True, post_window_ns=2_000_000_000)
        records[-1].update(present_records=0, present_dropped=0)
        observed = summarize_handoffs(records)['pending_queue_samples']
        self.assertEqual(observed[0]['pending_writes'], queue)
        self.assertIsNone(observed[1]['pending_writes'])
        for bad in (dict(queue, queued=-1), dict(queue, scope_ends=11),
                    dict(queue, front_item_age_ns=None), dict(queue, inflight_batches=True), 'bad'):
            records[1]['pending_writes'] = bad
            with self.assertRaises(CaptureError): summarize_handoffs(records)

    def test_admission_waits_are_optional_but_complete_and_consistent(self):
        queue = dict(queued=2, active_submits=0, inflight_batches=1, front_item_age_ns=5,
                     release_delay_ns=-2, scope_begins=10, scope_ends=10, deadline_resets=10,
                     admission_waiters=1, admission_wait_count=3, admission_wait_ns=900,
                     admission_wait_max_ns=500, admission_retired_wait_count=2,
                     admission_retired_wait_ns=700)
        records = capture([dict(SAMPLES[0], pending_writes=queue), SAMPLES[1]])
        records[0].update(present_handoffs_enabled=True, post_window_ns=2_000_000_000)
        records[-1].update(present_records=0, present_dropped=0)
        self.assertEqual(summarize_handoffs(records)['pending_queue_samples'][0]['pending_writes'], queue)
        partial = dict(queue); del partial['admission_wait_ns']
        for bad in (partial, dict(queue, admission_waiters=True), dict(queue, admission_wait_ns=-1),
                    dict(queue, admission_wait_max_ns=901), dict(queue, admission_retired_wait_count=4),
                    dict(queue, admission_retired_wait_ns=901), dict(queue, admission_wait_count=0),
                    dict(queue, admission_retired_wait_count=0)):
            records[1]['pending_writes'] = bad
            with self.assertRaises(CaptureError): summarize_handoffs(records)

    def test_handoffs_keep_boundaries_and_cpu_namespace_separate(self):
        records = capture(SAMPLES)
        records[0].update(present_handoffs_enabled=True, post_window_ns=1000)
        def row(event, identity, source=7, kind="gpu", result=0, begin=None):
            return dict(type="present-handoff", event=event, source_kind=kind, t_ns=100,
                        begin_ns=begin, publication_id=identity, source_seq=source,
                        other_seq=0, slot=0, result=result)
        rows = [row("published", 1), row("superseded", 1),
                row("published", 2), row("acquired", 2, begin=-100), row("gpu-shown", 2),
                row("published", 3), row("gpu-attempt-result", 3, result=1),
                row("published", 4), # window ends before a terminal observation
                row("gpu-shown", 5), # publication precedes the observed window
                row("cpu-acquired", 2, source=99, kind="cpu"),
                row("cpu-shown", 2, source=99, kind="cpu")]
        rows[1]['other_seq'] = 2
        records[-1].update(present_records=len(rows), present_dropped=0)
        records[-1:-1] = rows
        summary = summarize_handoffs(records)
        self.assertEqual(summary['published_outcomes'],
                         {'superseded':1, 'gpu-shown':1, 'skipped':1, 'unresolved-in-window':1})
        self.assertFalse(summary['gpu_publications'][-1]['publication_observed'])
        self.assertAlmostEqual(summary['waits_ms']['acquired']['total'], 0.0002)
        self.assertEqual(summary['event_counts']['cpu-shown'], 1)
        # Keep footer counts correct so these exercise semantics, not the count guard.
        rows[1]['event'] = 'future-replacement-name'
        with self.assertRaises(CaptureError): summarize_handoffs(records)
        rows[1]['event'] = 'superseded'
        rows[-1]['event'] = 'cpu-fallback-needed' # GPU selection metadata, never a CPU handoff
        with self.assertRaises(CaptureError): summarize_handoffs(records)
        rows[-1]['event'] = 'cpu-shown'
        rows[0]['publication_id'] = 0
        with self.assertRaises(CaptureError): summarize_handoffs(records)
        rows[0]['publication_id'] = 1
        rows[1]['other_seq'] = 1
        with self.assertRaises(CaptureError): summarize_handoffs(records)
        rows[1]['other_seq'] = 2
        # Removing a production event must fail visibly, rather than inventing a lost frame.
        records.pop(-2)
        with self.assertRaises(CaptureError): summarize_handoffs(records)

    def test_handoffs_refuse_unavailable_overflow_and_conflicting_outcomes(self):
        with self.assertRaises(CaptureError): summarize_handoffs(capture(SAMPLES))
        records = capture(SAMPLES)
        records[0].update(present_handoffs_enabled=True, post_window_ns=1000)
        records[-1].update(present_records=0, present_dropped=1)
        with self.assertRaises(CaptureError): summarize_handoffs(records)
        records[-1].update(present_records=2, present_dropped=0)
        rows = [dict(type="present-handoff", event=e, source_kind="gpu", t_ns=100,
                     begin_ns=None, publication_id=1, source_seq=7, other_seq=0, slot=0, result=0)
                for e in ("gpu-shown", "superseded")]
        rows[1]['other_seq'] = 2
        records[-1:-1] = rows
        with self.assertRaises(CaptureError): summarize_handoffs(records)

    def test_texture_witness_identity_tracks_selected_row_and_missing_metadata(self):
        row = {"frontend_tex_rtt_ms": 0, "frontend_texture_ms": 2,
               "frontend_tex_other_slowest_ms": 2, "frontend_tex_other_class": 4,
               "frontend_tex_other_img_dim": 1, "frontend_tex_other_depth_compare": False,
               "frontend_tex_other_host_backed": False}
        smaller = dict(row, frontend_tex_other_slowest_ms=1, frontend_tex_other_class=2)
        for selected, expected_class in [(row, 4),
                                         (dict(row, frontend_tex_other_class=0), 0),
                                         (dict(row, frontend_tex_other_class=999), 999),
                                         ({k: v for k, v in row.items() if k not in
                                           ('frontend_tex_other_class', 'frontend_tex_other_img_dim',
                                            'frontend_tex_other_depth_compare',
                                            'frontend_tex_other_host_backed')}, None)]:
            with self.subTest(resource_class=expected_class):
                summary = summarize(capture(SAMPLES, renderer=[smaller, selected]))
                witness = summary['resource_breakdown']['tex_other_witness']
                self.assertEqual(witness['resource_class'], expected_class)
                self.assertEqual(witness['img_dim'], None if expected_class is None else 1)
                self.assertEqual(witness['host_backed'], None if expected_class is None else False)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    print_summary(summary)
                label = {4: 'StorageImage', 0: 'ConstantBuffer', 999: '999', None: 'UNAVAILABLE'}[expected_class]
                self.assertIn('class=' + label, output.getvalue())
                self.assertIn('host_backed=' + ('UNAVAILABLE' if expected_class is None else '0'),
                              output.getvalue())

    def test_range_sharing_totals_missing_fields_and_signed_savings(self):
        row = {"res_texture_ms": 0, "res_buffer_ms": 10, "res_buffer_copy_ms": 3,
               "res_descriptor_ms": 0, "buffer_range_uploads": 2, "buffer_range_bindings": 1,
               "buffer_range_upload_bytes": 2**53 + 7, "buffer_range_bound_bytes": 4096,
               "res_buffer_range_plan_ms": 1.25}
        full = summarize(capture(SAMPLES, renderer=[row, row]))["resource_breakdown"]
        self.assertTrue(full["buffer_range_sharing_available"])
        ranges = full["buffer_range_sharing"]
        self.assertEqual(ranges["buffer_range_upload_bytes"], 2**54 + 14)
        self.assertEqual(ranges["avoided_copy_bytes"], 8192 - (2**54 + 14))
        self.assertEqual(ranges["res_buffer_range_plan_ms"], 2.5)
        for field in ("buffer_range_bindings", "res_buffer_range_plan_ms"):
            partial = dict(row)
            del partial[field]
            mixed = summarize(capture(SAMPLES, renderer=[row, partial]))["resource_breakdown"]
            self.assertFalse(mixed["buffer_range_sharing_available"])
            self.assertNotIn("buffer_range_sharing", mixed)

    def test_resident_buffer_partition_and_partial_population(self):
        row = {"res_texture_ms": 0, "res_buffer_ms": 20, "res_descriptor_ms": 0,
               "res_buffer_copy_ms": 2, "res_buffer_create_ms": 1,
               "res_buffer_index_find_ms": 1, "res_buffer_index_insert_ms": 1,
               "res_buffer_hash_ms": 1, "res_buffer_resident_ms": 9,
               "buffer_upload_bytes": 2**40 + 7, "buffer_resident_hits": 3,
               "buffer_resident_compared_bytes": 8192, "buffer_resident_reused_bytes": 4096,
               "buffer_resident_admitted_bytes": 4096, "buffer_resident_refreshed_bytes": 8192,
               "buffer_resident_declined_bytes": 0,
               "buffer_resident_ineligible_bytes": 0}
        full = summarize(capture(SAMPLES, renderer=[row, row]))["resource_breakdown"]
        self.assertTrue(full["buffer_residency_available"])
        self.assertEqual(full["res_buffer_other"], 10)
        self.assertEqual(full["buffer_residency"]["buffer_upload_bytes"], 2**41 + 14)
        self.assertFalse(full["buffer_watch_available"])
        watched_row = dict(row, buffer_resident_watched_bytes=2**40 + 9, res_buffer_watch_ms=3)
        watched = summarize(capture(SAMPLES, renderer=[watched_row, watched_row]))["resource_breakdown"]
        self.assertTrue(watched["buffer_watch_available"])
        self.assertEqual(watched["buffer_watch"]["buffer_resident_watched_bytes"], 2**41 + 18)
        self.assertEqual(watched["buffer_watch"]["res_buffer_watch_ms"], 6)
        self.assertEqual(watched["res_buffer_other"], 10)  # watch time is already inside resident
        partial_watch = dict(watched_row)
        del partial_watch["res_buffer_watch_ms"]
        mixed_watch = summarize(capture(SAMPLES, renderer=[watched_row, partial_watch]))["resource_breakdown"]
        self.assertTrue(mixed_watch["buffer_residency_available"])
        self.assertFalse(mixed_watch["buffer_watch_available"])
        self.assertNotIn("buffer_watch", mixed_watch)
        self.assertEqual(mixed_watch["res_buffer_other"], 10)
        # One missing field cannot silently become zero, or grant a partial subtraction.
        missing = dict(row)
        del missing["buffer_resident_hits"]
        mixed = summarize(capture(SAMPLES, renderer=[row, missing]))["resource_breakdown"]
        self.assertFalse(mixed["buffer_residency_available"])
        self.assertNotIn("buffer_residency", mixed)
        self.assertEqual(mixed["res_buffer_other"], 28)

    def test_texture_snapshot_totals_preserve_bytes_and_nested_timing(self):
        row = {"total_ms": 100, "build_resources_ms": 80, "frontend_texture_ms": 60,
               "frontend_tex_rtt_ms": 20, "frontend_tex_persist_invalid_ms": 30,
               "frontend_tex_source_snapshot_copied_bytes": 2**53 + 7,
               "frontend_tex_source_snapshot_transferred_bytes": 1024,
               "frontend_tex_source_snapshot_handoff_ms": 1.25}
        second = dict(row, frontend_tex_source_snapshot_copied_bytes=9,
                      frontend_tex_source_snapshot_transferred_bytes=2048,
                      frontend_tex_source_snapshot_handoff_ms=2.5)
        summary = summarize(capture(SAMPLES, renderer=[row, second]))
        breakdown = summary["resource_breakdown"]
        self.assertTrue(breakdown["texture_source_snapshot_available"])
        self.assertEqual(breakdown["texture_source_snapshot"], {
            "copied_bytes": 2**53 + 16, "transferred_bytes": 3072, "handoff_ms": 3.75})
        self.assertIsInstance(breakdown["texture_source_snapshot"]["copied_bytes"], int)
        without = [{key: value for key, value in record.items()
                    if not key.startswith("frontend_tex_source_snapshot_")}
                   for record in (row, second)]
        prior = summarize(capture(SAMPLES, renderer=without))
        # The child must neither change classification nor be subtracted again from the residual.
        self.assertEqual(summary["components"], prior["components"])
        self.assertEqual(summary["classification"], prior["classification"])
        self.assertEqual(breakdown["tex_other"], 20)
        self.assertEqual(breakdown["tex_other"], prior["resource_breakdown"]["tex_other"])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summary)
        lines = [line for line in output.getvalue().splitlines()
                 if "texture source snapshot" in line]
        self.assertEqual(len(lines), 1)
        self.assertIn("included in frontend texture", lines[0])
        self.assertIn("handoff=3.750ms", lines[0])
        self.assertIn(f"copied={2**53 + 16}B", lines[0])
        self.assertIn("transferred=3072B", lines[0])
        self.assertIn("excludes guest reads and GPU uploads", lines[0])

    def test_gpu_detile_population_preserves_missing_and_integer_bytes(self):
        row = {"frontend_gpu_detile_preparations": 5,
               "frontend_gpu_detile_2d_preparations": 3,
               "frontend_gpu_detile_source_bytes": 2**53 + 17}
        result = summarize(capture(SAMPLES, renderer=[row, row]))["resource_breakdown"]
        self.assertTrue(result["gpu_detile_available"])
        self.assertEqual(result["gpu_detile"], {
            "preparations": 10, "2d_preparations": 6, "source_bytes": 2**54 + 34})
        self.assertIsInstance(result["gpu_detile"]["source_bytes"], int)
        zero = dict.fromkeys(row, 0)
        result = summarize(capture(SAMPLES, renderer=[zero]))["resource_breakdown"]
        self.assertTrue(result["gpu_detile_available"])
        self.assertEqual(result["gpu_detile"]["preparations"], 0)
        for missing in row:
            partial = row.copy()
            del partial[missing]
            result = summarize(capture(SAMPLES, renderer=[row, partial]))["resource_breakdown"]
            self.assertFalse(result["gpu_detile_available"])
            self.assertNotIn("gpu_detile", result)
        result = summarize(capture(SAMPLES, renderer=[{}]))["resource_breakdown"]
        self.assertFalse(result["gpu_detile_available"])

    def test_texture_snapshot_zero_and_missing_population_are_distinct(self):
        fields = ("frontend_tex_source_snapshot_copied_bytes",
                  "frontend_tex_source_snapshot_transferred_bytes",
                  "frontend_tex_source_snapshot_handoff_ms")
        zero = dict.fromkeys(fields, 0)
        summary = summarize(capture(SAMPLES, renderer=[zero, zero]))
        self.assertTrue(summary["resource_breakdown"]["texture_source_snapshot_available"])
        self.assertEqual(summary["resource_breakdown"]["texture_source_snapshot"], {
            "copied_bytes": 0, "transferred_bytes": 0, "handoff_ms": 0})
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summary)
        self.assertIn("handoff=0.000ms copied=0B transferred=0B", output.getvalue())
        self.assertNotIn("texture source snapshot: UNAVAILABLE", output.getvalue())
        for missing in fields:
            with self.subTest(missing=missing):
                partial = dict(zero)
                del partial[missing]
                mixed = summarize(capture(SAMPLES, renderer=[zero, partial]))
                self.assertFalse(mixed["resource_breakdown"]["texture_source_snapshot_available"])
                self.assertNotIn("texture_source_snapshot", mixed["resource_breakdown"])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    print_summary(mixed)
                self.assertIn("texture source snapshot: UNAVAILABLE", output.getvalue())
                self.assertNotIn("handoff=", output.getvalue())
        legacy = summarize(capture(SAMPLES, renderer=[{}]))["resource_breakdown"]
        self.assertFalse(legacy["texture_source_snapshot_available"])
        self.assertNotIn("texture_source_snapshot", legacy)
        self.assertIsNone(summarize(capture(SAMPLES))["resource_breakdown"])

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
        # The record carries a timestamp inside the sample window, as every real capture's does --
        # `performance_capture.cpp` stamps `monotonic_ns` on every renderer and compute record. The
        # fixture previously omitted it, which since #3678 means no cohort can be formed and the
        # coverage ratio is withheld, so the verdict this test names could not be reached.
        summary = summarize(capture(SAMPLES, renderer=[{"total_ms": 100, "t_ns": 500_000_000}]))
        self.assertEqual(summary["classification"], "cpu-outside-renderer")
        self.assertAlmostEqual(summary["cpu_cores"], 1.0, places=3)
        self.assertEqual(summary["detail_outside"], 0)
        self.assertAlmostEqual(summary["coverage"], 0.1, places=6)

    # ---- #3678: the counter-rate window and the detail population are distinct observations ----

    def test_detail_records_outside_the_sample_window_are_reported_and_excluded_from_coverage(self):
        # Two records of equal cost, one completing inside the 1 s sample window and one 2 s after
        # its end. `measured_total` keeps both -- it is the population actually retained -- but the
        # coverage ratio against the window may only use the one that completed inside it.
        summary = summarize(capture(SAMPLES, renderer=[
            {"total_ms": 100, "t_ns": 500_000_000},
            {"total_ms": 100, "t_ns": 3_000_000_000},
        ]))
        self.assertEqual(summary["graphics_total_ms"], 200)
        self.assertEqual(summary["detail_outside"], 1)
        self.assertEqual(summary["windowed_total_ms"], 100)
        # 100 ms of 1000 ms, NOT 200/1000 -- mixing the populations would have doubled it.
        self.assertAlmostEqual(summary["coverage"], 0.1, places=6)

    def test_detail_span_is_reported_separately_from_the_sample_window(self):
        summary = summarize(capture(SAMPLES, renderer=[
            {"total_ms": 1, "t_ns": 0},
            {"total_ms": 1, "t_ns": 4_000_000_000},
        ]))
        self.assertAlmostEqual(summary["seconds"], 1.0, places=6)
        self.assertAlmostEqual(summary["detail_seconds"], 4.0, places=6)

    def test_timestampless_details_withhold_coverage_and_the_verdict_it_would_drive(self):
        # The deliberately mismatched fixture the issue asks for: a population that cannot be
        # aligned to the sampled window must not yield a ratio that looks aligned. Withholding it
        # also withholds `cpu-outside-renderer`, which is the point -- that verdict must not rest on
        # a number formed across two populations.
        summary = summarize(capture(SAMPLES, renderer=[{"total_ms": 100}]))
        self.assertIsNone(summary["coverage"])
        self.assertIsNone(summary["detail_outside"])
        self.assertNotEqual(summary["classification"], "cpu-outside-renderer")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summary)
        self.assertIn("window coverage: unavailable", output.getvalue())

    def test_unavailable_lines_name_the_actual_cause_not_a_plausible_one(self):
        # Review of #3701 found the first version printing "records carry no timestamps" for every
        # state that left the span unset -- including a capture that retained no records at all,
        # which is what F8 on a hung title produces. A report whose explanation is wrong is the same
        # defect this change exists to remove, so each cause is asserted by TEXT.
        empty = summarize(capture(SAMPLES))
        self.assertIsNone(empty["detail_seconds"])
        self.assertEqual(empty["detail_status"], "no renderer or compute records were retained")
        printed = self._printed(empty)
        self.assertIn("no renderer or compute records were retained", printed)
        self.assertNotIn("carry no timestamps", printed)

        untimed = summarize(capture(SAMPLES, renderer=[{"total_ms": 1}]))
        self.assertEqual(untimed["detail_status"], "records carry no timestamps")

        single = summarize(capture(SAMPLES, renderer=[{"total_ms": 1, "t_ns": 500_000_000}]))
        self.assertIsNone(single["detail_seconds"])
        self.assertIn("share one timestamp", single["detail_status"])
        # ...and the coverage line beside it must NOT deny timestamps, because this record has one.
        self.assertIsNotNone(single["coverage"])

        # Two DISTINCT in-window timestamps plus one bare record. The two distinct values are what
        # make `detail_seconds` computable and so make the "every record completed inside" note
        # REACHABLE -- with a single timestamp the branch is unreachable regardless of the gate, and
        # the assertion below would have passed before the fix as well as after it.
        partial = summarize(capture(SAMPLES, renderer=[
            {"total_ms": 1, "t_ns": 100_000_000},
            {"total_ms": 1, "t_ns": 900_000_000},
            {"total_ms": 1}]))
        self.assertIn("only 2 of 3 records carry timestamps", partial["detail_status"])
        # A population that cannot be partitioned must not claim every record landed inside.
        self.assertIsNone(partial["detail_outside"])
        self.assertNotIn("every record completed inside", self._printed(partial))

    def test_coverage_unavailable_distinguishes_no_window_from_no_alignment(self):
        # `coverage` is also None when there is no post-sample window at all -- a property of the
        # SAMPLES, not of the records. The first version blamed the records either way.
        no_window = summarize(capture(SAMPLES[:1], renderer=[{"total_ms": 1, "t_ns": 0}]))
        self.assertIsNone(no_window["coverage"])
        self.assertIn("no post-sample window", no_window["coverage_status"])
        self.assertNotIn("carry no timestamps", self._printed(no_window))

    def test_windowed_total_is_withheld_rather_than_reported_unclipped(self):
        # Publishing the whole population under a key named "windowed" would reintroduce the exact
        # mix-up removed from the printed report.
        untimed = summarize(capture(SAMPLES, renderer=[{"total_ms": 100}]))
        self.assertIsNone(untimed["coverage"])
        self.assertIsNone(untimed["windowed_total_ms"])

    def test_counter_delta_rejects_a_decreasing_counter(self):
        # These are monotonic counters; a decrease means the pair is unusable, and printing
        # "unavailable (-4 over 5.02 s)" would be worse than printing nothing.
        backwards = [
            {"t_ns": 0, "process_cpu_ns": 0, "guest_presents": 60,
             "rendered_frames": 10, "host_presented_frames": 9},
            {"t_ns": 1_000_000_000, "process_cpu_ns": 10, "guest_presents": 56,
             "rendered_frames": 10, "host_presented_frames": 9},
        ]
        summary = summarize(capture(backwards))
        self.assertIsNone(summary["rate_events"]["guest_fps"])
        self.assertNotIn("-4 over", self._printed(summary))

    def test_rates_expose_their_numerator_so_one_event_is_visible(self):
        # #3678's own example: 36 events against 35 read as 7.16884 and 7.34651 per second, a gap
        # smaller than one event's contribution. The count has to be on the page.
        summary = summarize(capture(SAMPLES))
        self.assertEqual(summary["rate_events"]["guest_fps"], 60)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_summary(summary)
        self.assertIn("(60 over 1.00 s)", output.getvalue())

    def test_producer_lineage_reports_rates_and_known_delivery_coverage(self):
        lineage = [
            dict(SAMPLES[0], producer_publications=10, producer_publications_known=8,
                 producer_delivered_new=4, producer_delivered_repeat=3,
                 producer_delivered_unknown=1),
            dict(SAMPLES[1], producer_publications=30, producer_publications_known=24,
                 producer_delivered_new=13, producer_delivered_repeat=9,
                 producer_delivered_unknown=3),
        ]
        summary = summarize(capture(lineage))
        observed = summary["producer_lineage"]
        self.assertEqual(observed["events"], {
            "producer_publications": 20, "producer_publications_known": 16,
            "producer_delivered_new": 9, "producer_delivered_repeat": 6,
            "producer_delivered_unknown": 2})
        self.assertEqual(observed["rates"]["producer_delivered_new"], 9.0)
        self.assertEqual(observed["known_deliveries"], 15)
        self.assertEqual(observed["classified_deliveries"], 17)
        self.assertAlmostEqual(observed["delivery_coverage"], 15 / 17)
        self.assertAlmostEqual(observed["publication_coverage"], 0.8)
        printed = self._printed(summary)
        self.assertIn("producer lineage rates: publications=20.00/s (20 over 1.00 s)", printed)
        self.assertIn("producer delivery lineage coverage: 88% (15 known deliveries of 17", printed)

    def test_old_capture_keeps_producer_lineage_unavailable(self):
        # No producer fields is the expected shape for a capture written before lineage existed.
        summary = summarize(capture(SAMPLES))
        observed = summary["producer_lineage"]
        self.assertIsNone(observed["delivery_coverage"])
        self.assertIn("delivery lineage counters are unavailable", observed["delivery_coverage_status"])
        self.assertIn("producer delivery lineage coverage: unavailable", self._printed(summary))

    def test_producer_lineage_requires_complete_delivery_classification(self):
        partial = [dict(SAMPLES[0], producer_delivered_new=1, producer_delivered_repeat=2),
                   dict(SAMPLES[1], producer_delivered_new=4, producer_delivered_repeat=6)]
        observed = summarize(capture(partial))["producer_lineage"]
        self.assertIsNone(observed["delivery_coverage"])
        self.assertIsNone(observed["known_deliveries"])

    def test_a_late_first_post_sample_displaces_only_the_rate_window(self):
        # The rate window starts at the FIRST post sample, so a late first sample DISPLACES it --
        # the span is 1.00 s either way -- while leaving the detail population where it was. The two
        # must move independently, which is the property under test; an earlier name said
        # "shortens", which this fixture does not do.
        late = [
            {"t_ns": 2_000_000_000, "process_cpu_ns": 0, "guest_presents": 0,
             "rendered_frames": 0, "host_presented_frames": 0},
            {"t_ns": 3_000_000_000, "process_cpu_ns": 10, "guest_presents": 30,
             "rendered_frames": 5, "host_presented_frames": 5},
        ]
        summary = summarize(capture(late, renderer=[{"total_ms": 10, "t_ns": 0}]))
        self.assertAlmostEqual(summary["seconds"], 1.0, places=6)
        # The record predates the window entirely; it is retained but cannot count toward coverage.
        self.assertEqual(summary["detail_outside"], 1)
        self.assertEqual(summary["windowed_total_ms"], 0)
        self.assertAlmostEqual(summary["coverage"], 0.0, places=6)

    def test_single_post_sample_leaves_both_spans_and_coverage_unavailable(self):
        summary = summarize(capture(SAMPLES[:1], renderer=[{"total_ms": 10, "t_ns": 0}]))
        self.assertIsNone(summary["seconds"])
        self.assertIsNone(summary["coverage"])

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

    def test_every_readback_note_is_well_formed_prose(self):
        # A structural guard, added because the previous commit shipped a visible break that every
        # existing arm was blind to: the tails became full sentences while one branch stayed a
        # trailing clause, rendering "...through a real window. before acting on this verdict."
        # `assertIn` on a substring cannot see a break OUTSIDE that substring, so no amount of
        # content assertions would have caught it -- this checks the joins instead of the content.
        # The FULL product of reachable joins: three GPU-present states (each selecting a different
        # note body) x three verdicts (each selecting a different tail). Enumerated as a product
        # rather than hand-listed because a hand-listed set is exactly how an arm named "every" ends
        # up excluding the one case that fails it -- which a first version of this arm did, missing
        # the `adopted is True` body, the only reachable note that lacked a terminating period.
        posts = {
            "not-adopted": SAMPLES,                                            # harness readback
            "adopted": [{**s, "rendered_frames": None} for s in SAMPLES],      # real readback
            "cannot-say": [],                                                  # no post population
        }
        verdicts = {
            "readback": (self.READBACK_RENDERER, []),
            "compute": ([{"total_ms": 1500, "readback_ms": 1400}], [{"total_ms": 3500.0}]),
            "inconclusive": ([{"total_ms": 2000, "readback_ms": 977.7}], [{"total_ms": 1134.0}]),
        }
        cases = {f"{pn}/{vn}": capture(post, renderer=r, compute=c)
                 for pn, post in posts.items() for vn, (r, c) in verdicts.items()}
        self.assertEqual(len(cases), 9)
        for name, records in cases.items():
            note = summarize(records)["readback_note"]
            self.assertIsNotNone(note, name)
            self.assertNotRegex(note, r"\.\s+[a-z]", f"{name}: lowercase clause after a period")
            self.assertNotRegex(note, r"\s\.", f"{name}: space before a period")
            self.assertNotRegex(note, r"\.\.", f"{name}: doubled period")
            self.assertTrue(note.endswith("."), f"{name}: note does not end in a period")

    def test_no_post_samples_cannot_determine_gpu_present(self):
        # The one uncovered line in the helper. With no post population there is nothing to read the
        # field from, so the honest answer is the third state rather than either branch.
        #
        # The note assertion is UNCONDITIONAL: this fixture's readback share clears the trigger, so
        # the note fires whatever the verdict turns out to be. Only the classification is left
        # tolerant, because that is the part this arm does not care about. The previous shape had a
        # defensive `else` asserting the note is None -- which became both unreachable and false
        # once the trigger stopped depending on the verdict, i.e. a branch that would have silently
        # stopped testing anything.
        summary = summarize(capture([], renderer=self.READBACK_RENDERER))
        self.assertIn("cannot say", summary["readback_note"])

    def test_large_sub_threshold_readback_warns_even_though_it_lost_classification(self):
        # The gap this closes, and the shape of the capture that motivated it (Dragon Quest VII):
        # readback 977.7 ms against compute 1134.0 ms. Neither reaches the 40% evidence bar, so
        # the verdict is "inconclusive" -- and the old trigger, which fired only on a readback
        # VERDICT, printed the second-largest number on the page with nothing attached.
        #
        # Note the arm would also pass under a "is readback the max component" trigger only if
        # readback won, which it does NOT here (977.7 < 1134.0). That is deliberate: ranking was
        # the first attempt at this fix and is silent on exactly this capture.
        summary = summarize(capture(
            SAMPLES,
            renderer=[{"total_ms": 2000, "readback_ms": 977.7}],
            compute=[{"total_ms": 1134.0}]))
        # Neither component clears 40% of the 3134 ms measured total: compute is 36%, readback 31%.
        self.assertEqual(summary["classification"], "inconclusive")
        self.assertIn("harness, not the title", summary["readback_note"])

    def test_readback_note_threshold_is_a_share_not_a_ranking(self):
        # Holds verdict and GPU-present state constant so the only variable is readback's SHARE.
        #
        # It does NOT pin the lower bound, and an earlier version of this comment wrongly claimed it
        # did ("no test would notice" a drift to any-nonzero). Measured per-arm: a drift to
        # any-nonzero is caught by the PRE-EXISTING 2% arm below, and this 1% assertion passes at
        # every threshold down to 0.011. What binds above is this arm's 20% assertion. Keeping the
        # 1% half is still right -- it is the only arm that varies share alone -- but the reason is
        # isolation, not coverage of the quiet end.
        def note_for(readback_ms):
            return summarize(capture(
                SAMPLES,
                renderer=[{"total_ms": 1000, "readback_ms": readback_ms}],
                compute=[{"total_ms": 1000.0}]))["readback_note"]

        self.assertIsNone(note_for(20.0))        # 1% of measured work -- rounding, stays silent
        self.assertIsNotNone(note_for(400.0))    # 20% -- a reader could mistake it for the answer

    def test_readback_note_threshold_stays_under_the_evidence_bar(self):
        # The INEQUALITY is the invariant, and it is all this asserts. The note exists to cover the
        # gap under the classification bar, so a threshold at or above the bar would make it
        # unreachable in exactly the cases it is for.
        #
        # It deliberately does NOT assert the one-quarter ratio. A previous version did, on the
        # grounds that a future change to the bar would then "carry the threshold along" -- but the
        # bar was a bare inline literal, so `0.40 / 4` in the test was a second disconnected copy
        # and the assert only ever pinned `== 0.1` in longer notation. The mutation offered as proof
        # (moving the threshold) was drawn from the same source as the claim and could not have
        # failed: it showed the assert fires when the CONSTANT moves, never in doubt, not when the
        # BAR moves, which was the claim. One quarter is a chosen fraction, not a derived one.
        self.assertGreater(READBACK_NOTE_MIN_SHARE, 0)
        self.assertLess(READBACK_NOTE_MIN_SHARE, CLASSIFICATION_EVIDENCE_SHARE)

    def test_classification_evidence_bar_is_pinned(self):
        # Nothing pinned this at all: moving the tool's primary classification threshold from 0.40
        # to 0.50 passed the entire suite.
        #
        # The two halves below are COMPLEMENTARY, not redundant, and the measured split is not what
        # "both arms fail if it moves" would suggest -- swept with the literal assert neutralised,
        # the behavioural pair alone tolerates (0.375, 0.4444]:
        #
        #   bar        0.375   0.376   0.42   0.4444   0.445   0.50
        #   behavioural  FAIL    pass   pass     pass    FAIL   FAIL
        #
        # So inside that window only the literal catches a value change -- while reverting the USE
        # SITE to an inline literal (decoupling the constant from the code it governs) is caught
        # only by the behavioural pair, which is the failure the whole finding was about. Each
        # covers what the other cannot.
        self.assertAlmostEqual(CLASSIFICATION_EVIDENCE_SHARE, 0.40)
        # Totals are scaled so measured work clears 40% of the 1000 ms wall window in SAMPLES --
        # otherwise both arms land in `cpu-outside-renderer` and neither exercises the bar at all,
        # which is what a first draft of this test did.
        just_under = summarize(capture(SAMPLES, renderer=[{"total_ms": 1000}],
                                       compute=[{"total_ms": 600}]))         # 37.5% -- under
        self.assertEqual(just_under["classification"], "inconclusive")
        just_over = summarize(capture(SAMPLES, renderer=[{"total_ms": 1000}],
                                      compute=[{"total_ms": 800}]))          # 44.4% -- over
        self.assertEqual(just_over["classification"], "compute")

    def test_readback_note_tail_does_not_disown_a_verdict_readback_did_not_decide(self):
        # A decisive compute verdict with a material readback used to close "...before acting on
        # this verdict" -- telling the reader to discard a conclusion the readback had no part in.
        # readback 1400 / 5000 measured = 28%, comfortably over the 10% threshold rather than the
        # 2 points of headroom a first draft had -- a fixture that only just clears the trigger
        # fails for a threshold reason under any threshold mutation, which is not what it tests.
        decisive = summarize(capture(
            SAMPLES,
            renderer=[{"total_ms": 1500, "readback_ms": 1400}],
            compute=[{"total_ms": 3500.0}]))
        self.assertEqual(decisive["classification"], "compute")          # 70%, decisive
        self.assertIsNotNone(decisive["readback_note"])
        # The live discriminator, not the retired phrase: `assertNotIn` on a string that no
        # longer exists anywhere in the module cannot fail, so it asserts nothing.
        self.assertNotIn("nothing else here to act on", decisive["readback_note"])
        self.assertIn("does not depend on it", decisive["readback_note"])

        # ...while a readback VERDICT says the strongest thing of the five, because there the
        # readback is not a caveat on the verdict -- it IS the verdict.
        owned = summarize(capture(SAMPLES, renderer=[self.READBACK_RENDERER[0]]))
        self.assertEqual(owned["classification"], "readback")
        self.assertIn("nothing else here to act on", owned["readback_note"])
        self.assertNotIn("does not depend on it", owned["readback_note"])

    def test_readback_note_does_not_vouch_for_a_verdict_the_readback_can_flip(self):
        # The case the previous version got WRONG, and it is this change's own motivating capture:
        # readback 977.7 against compute 1134.0 with a 2000 ms renderer total is "inconclusive"
        # only because nothing reaches the bar. Remove the readback and compute wins outright at
        # 53% -- so "the verdict above does not depend on it" would be a false statement printed on
        # the exact capture this feature exists for, inviting trust in a verdict the harness made.
        contaminated = summarize(capture(
            SAMPLES,
            renderer=[{"total_ms": 2000, "readback_ms": 977.7}],
            compute=[{"total_ms": 1134.0}]))
        self.assertEqual(contaminated["classification"], "inconclusive")
        self.assertIn("may itself be an artefact", contaminated["readback_note"])
        self.assertNotIn("does not depend on it", contaminated["readback_note"])

        # And the flip is real, not asserted: drop the readback and the verdict changes.
        without = summarize(capture(
            SAMPLES,
            renderer=[{"total_ms": 2000 - 977.7}],
            compute=[{"total_ms": 1134.0}]))
        self.assertEqual(without["classification"], "compute")

    def test_non_readback_verdict_carries_no_readback_note(self):
        # The note is specific to the readback verdict; on every report it would be noise.
        summary = summarize(capture(SAMPLES, renderer=[{
            "total_ms": 100, "gpu_device_ms": 10, "gpu_wait_ms": 12,
            "build_resources_ms": 25, "setup_resources_ms": 25, "readback_ms": 2,
        }]))
        self.assertEqual(summary["classification"], "renderer-resource")
        self.assertIsNone(summary["readback_note"])

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

    def _printed(self, summary):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_summary(summary)
        return buffer.getvalue()

    def test_committed_bytes_are_reported_when_the_capture_carries_them(self):
        # Windows records the commit charge alongside the working set (#3448), because the two
        # diverge by gigabytes under prosper's host-side caches and only the second one tracks them.
        samples = [dict(sample, private_bytes=value)
                   for sample, value in zip(SAMPLES, (3 * 2**20, 5 * 2**20))]
        summary = summarize(capture(samples))
        self.assertEqual((summary["private_min"], summary["private_max"]),
                         (3 * 2**20, 5 * 2**20))
        self.assertIn("private/committed: 3.0..5.0 MiB", self._printed(summary))

    def test_capture_without_committed_bytes_still_reads_cleanly(self):
        # The backward-compatibility arm. SAMPLES carries no `private_bytes` key at all, which is
        # the exact shape of every .prperf written before the field existed: it must validate, it
        # must summarize, and it must say unavailable rather than inventing a number.
        records = capture(SAMPLES)
        validate_capture(records)
        summary = summarize(records)
        self.assertIsNone(summary["private_min"])
        self.assertIsNone(summary["private_max"])
        self.assertIn("private/committed: unavailable on this platform/run",
                      self._printed(summary))

    def test_zero_committed_bytes_is_a_measurement_not_an_absence(self):
        # A zero and an absence are different claims, and a truth test on the population would
        # merge them -- reporting a measured zero as "unavailable on this platform/run".
        summary = summarize(capture([dict(sample, private_bytes=0) for sample in SAMPLES]))
        self.assertEqual(summary["private_min"], 0)
        self.assertIn("private/committed: 0.0..0.0 MiB", self._printed(summary))

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


# The record shape that made #3447 invisible: almost all of the cost is in `pipeline_ms` and
# `gpu_storage_copy_ms`, both of which the report captured and never printed. The five timers it
# did print account for 6 ms of a 300 ms dispatch.
HIDDEN_COST_COMPUTE = [{
    "program_addr": 0x1000, "program_hash": 0xABCD, "dispatches": 1,
    "total_ms": 300.0,
    "setup_ms": 2.0, "pipeline_ms": 220.0, "dispatch_wait_ms": 70.0,
    "writeback_ms": 3.0, "cleanup_ms": 1.0,
    "gpu_device_ms": 65.0, "gpu_pre_ms": 1.0, "gpu_shader_ms": 4.0,
    "gpu_storage_copy_ms": 55.0, "gpu_compare_ms": 5.0, "gpu_restore_ms": 0.0,
    "gpu_timestamp_samples": 1,
}]



# A dispatch that broke during setup, in the shape #3461 describes: `phase_writeback` never
# advanced, so `cleanup_ms` books the whole item and `pipeline_ms` books the compensating
# negative. The stored values are self-consistent -- they telescope to the item duration --
# so the capture is right and only the RENDERING can go wrong.
BROKEN_DISPATCH_COMPUTE = [{
    "program_addr": 0x2000, "program_hash": 0xBEEF, "dispatches": 1,
    "total_ms": 12.0,
    "setup_ms": 10.0, "pipeline_ms": -10.0, "dispatch_wait_ms": 0.0,
    "writeback_ms": 0.0, "cleanup_ms": 12.0,
    "gpu_device_ms": 0.0, "gpu_timestamp_samples": 0,
}]


class ComputeDecompositionTests(unittest.TestCase):
    def _render(self, compute):
        summary = summarize(capture(SAMPLES, compute=compute))
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print_summary(summary)
        return summary, buffer.getvalue()

    def _group_line(self, text):
        """The per-program line alone.

        Every assertion about the group breakdown must run against THIS, not against the whole
        report: the summary table prints the same field names, so a substring match on the full
        text is satisfied by the summary and never reaches the group line. That is why reverting
        the entire group-line change once left this suite green.
        """
        lines = [line for line in text.splitlines() if line.strip().startswith("0x")
                 and "records=" in line]
        self.assertEqual(len(lines), 1, f"expected exactly one group line in:\n{text}")
        return lines[0]

    def test_every_captured_timer_is_reported(self):
        # COVERAGE arm, deliberately whole-text: it asserts the summary carries every field and
        # that each name reaches the output SOMEWHERE. It does not distinguish the summary from
        # the group line -- either satisfies it -- so it survives a mutation of either surface.
        # The surface-specific pinning is done by the summary arms and the _group_line arms;
        # this one exists so a newly added timer that reaches neither surface still fails.
        from performance_capture_report import COMPUTE_CPU_PHASES, COMPUTE_GPU_BRACKETS
        summary, text = self._render(HIDDEN_COST_COMPUTE)
        for field in COMPUTE_CPU_PHASES:
            self.assertIn(field, summary["compute_cpu_phases"], field)
        for field in COMPUTE_GPU_BRACKETS:
            self.assertIn(field, summary["compute_gpu_brackets"], field)
        self.assertEqual(summary["compute_cpu_phases"]["pipeline_ms"], 220.0)
        self.assertEqual(summary["compute_gpu_brackets"]["gpu_storage_copy_ms"], 55.0)
        self.assertIn("pipeline=220.0ms", text)
        self.assertIn("storage-copy=55.0ms", text)

    def test_storage_children_remain_nested_and_signed(self):
        records = [dict(HIDDEN_COST_COMPUTE[0], gpu_image_transfer_ms=20.0, gpu_retile_ms=36.0)]
        summary, text = self._render(records)
        self.assertEqual(summary["compute_storage_children"], {
            "gpu_image_transfer_ms": 20.0, "gpu_retile_ms": 36.0, "other_storage_ms": -1.0})
        self.assertEqual(sum(summary["compute_gpu_brackets"].values()), 65.0)
        self.assertIn("children of storage-copy (already included)", text)
        self.assertIn("other-storage=-1.0ms", text)

    def test_missing_storage_children_are_unavailable(self):
        summary, text = self._render(HIDDEN_COST_COMPUTE)
        self.assertIsNone(summary["compute_storage_children"])
        self.assertNotIn("children of storage-copy", text)
        complete = dict(HIDDEN_COST_COMPUTE[0], gpu_image_transfer_ms=0.0, gpu_retile_ms=0.0)
        self.assertIsNone(self._render([complete, HIDDEN_COST_COMPUTE[0]])[0]["compute_storage_children"])
        self.assertEqual(self._render([complete])[0]["compute_storage_children"]["gpu_retile_ms"], 0.0)

    def test_dominant_timer_reaches_the_printed_output(self):
        # The point of the fix: the largest cost must be visible without parsing the file by hand.
        _, text = self._render(HIDDEN_COST_COMPUTE)
        self.assertIn("(73.3%)", text)   # pipeline_ms, 220 of 300
        self.assertIn("(84.6%)", text)   # gpu_storage_copy_ms, 55 of 65 device

    def test_remainder_is_named_rather_than_left_to_subtraction(self):
        # 300 total against 296 of named CPU phases: the 4 ms must be stated, not inferred.
        _, text = self._render(HIDDEN_COST_COMPUTE)
        self.assertIn("unattributed=4.0ms", text)

    def test_gpu_brackets_are_scaled_against_device_time_not_total(self):
        # The brackets sit inside dispatch_wait, so they partition gpu_device_ms. Scaling them
        # against total_ms would understate the storage copy by 4.6x here and invite summing two
        # sibling decompositions into one bogus 100%.
        _, text = self._render(HIDDEN_COST_COMPUTE)
        self.assertIn("GPU brackets of 65.0ms device", text)
        self.assertNotIn("storage-copy=55.0ms (18.3%)", text)

    def test_shader_share_of_device_time_is_stated(self):
        # The headline this whole change exists to surface.
        _, text = self._render(HIDDEN_COST_COMPUTE)
        self.assertIn("compute shader is 6.2% of the device time", text)

    def test_group_line_carries_the_dropped_timers(self):
        # Asserted on the group line ALONE. The old report printed five fields here and the
        # two holding 92% of this dispatch were absent; a whole-text match cannot see that,
        # because the summary table above prints the same names.
        _, text = self._render(HIDDEN_COST_COMPUTE)
        line = self._group_line(text)
        self.assertIn("pipeline=220.0ms", line)
        self.assertIn("storage-copy=55.0ms", line)
        self.assertIn("cleanup=1.0ms", line)
        self.assertIn("unattributed=4.0ms", line)

    def test_group_line_separates_the_two_scopes(self):
        # Printed flat, the eleven fields sum to 430ms against the 300ms total on the same line,
        # because the GPU brackets are inside dev which is inside wait. The line must say so
        # rather than leaving a reader to add them up and conclude the tool cannot count.
        _, text = self._render(HIDDEN_COST_COMPUTE)
        line = self._group_line(text)
        self.assertIn("cpu ", line)
        self.assertIn("gpu dev=65.0ms (inside wait)", line)
        self.assertIn("of which", line)
        cpu_section = line.split("[", 1)[1].split(" | ", 1)[0]
        self.assertNotIn("storage-copy", cpu_section)
        self.assertNotIn("dev=", cpu_section)

    def test_group_line_cpu_fields_account_for_its_own_total(self):
        # The invariant the labelling exists to protect: the cpu section, remainder included,
        # sums to the total printed on the same line.
        _, text = self._render(HIDDEN_COST_COMPUTE)
        line = self._group_line(text)
        cpu_section = line.split("[", 1)[1].split(" | ", 1)[0]
        values = [float(part.split("=")[1].removesuffix("ms"))
                  for part in cpu_section.split() if "=" in part]
        self.assertAlmostEqual(sum(values), 300.0, places=1)

    def test_negative_remainder_is_shown_signed_on_both_surfaces(self):
        # A record whose named phases exceed its own total. NO CURRENT PRODUCER PATH BUILDS ONE,
        # and the reason is worth stating so nobody re-derives it: the five CPU phases are
        # consecutive differences over start->setup->pipeline->dispatch->writeback->cleanup
        # (live_compute.cpp:11639-11643), and phase_milliseconds (:11635) is a plain signed
        # difference with no clamping, so they telescope to cleanup-start for ANY ordering of the
        # markers -- degenerate ones included. A break leaves a marker behind and yields a
        # negative PHASE offset by a compensating positive one; the SUM is unchanged. Since
        # total_ms spans at least that window, the remainder cannot go negative today.
        #
        # An earlier version of this comment claimed a break before phase_pipeline produced one.
        # It does not, and the claim reached here from a review citation that skipped the
        # telescoping step -- which is why it is corrected rather than quietly dropped.
        #
        # The arm stays because the THRESHOLD is what is under test, not that mechanism: a
        # one-sided threshold would silently hide a negative remainder if the telescoping ever
        # broke (a clamp, or a phase measured outside the start..cleanup window), and a
        # remainder that does not partition its total is precisely what this report exists to
        # surface. It also matches the signed convention used for every other remainder here.
        over = [dict(HIDDEN_COST_COMPUTE[0])]
        over[0]["total_ms"] = 290.0          # named CPU phases sum to 296.0
        _, text = self._render(over)
        self.assertIn("unattributed=-6.0ms", self._group_line(text))
        cpu_summary = next(line for line in text.splitlines()
                           if line.strip().startswith("CPU phases of"))
        self.assertIn("unattributed=-6.0ms", cpu_summary)

    def test_negative_phase_values_are_printed_not_suppressed(self):
        # A `value < 0.05` threshold treats a negative as an absence. That hid the compensating
        # term of a signed pair, so the visible parts summed to 183% of their own total, while
        # the `unattributed` row -- computed from the full set, hidden term included -- came to
        # exactly zero and printed nothing. The one mechanism that would have flagged the
        # inconsistency was silenced by the very value it should have reported, which is the
        # same shape as the defect this whole change exists to fix.
        _, text = self._render(BROKEN_DISPATCH_COMPUTE)
        summary = next(line for line in text.splitlines()
                       if line.strip().startswith("CPU phases of"))
        self.assertIn("pipeline=-10.0ms", summary)
        self.assertIn("pipeline=-10.0ms", self._group_line(text))
        # And with it visible, the printed parts account for the whole rather than 183% of it.
        #
        # Summed in MILLISECONDS, not in the rendered percentages. Percentages are each rounded
        # to one decimal before being printed, so their sum lands on 100.0 only when the errors
        # happen to cancel: this fixture does (its suppressed phases are exactly 0.0 and its
        # +/-0.033 errors are equal and opposite), while HIDDEN_COST_COMPUTE sums to 99.9 with
        # nothing suppressed at all. A tolerance wide enough for both would no longer detect the
        # 183% this arm exists to catch, so the check is done on the underlying values, as
        # test_group_line_cpu_fields_account_for_its_own_total already does.
        values = [float(part.split("=")[1].split("ms")[0])
                  for part in summary.split() if "=" in part and "ms" in part]
        self.assertAlmostEqual(sum(values), 12.0, places=3)

    def test_absent_gpu_timestamps_do_not_invent_a_bracket_table(self):
        from performance_capture_report import COMPUTE_GPU_BRACKETS
        no_gpu = [dict(HIDDEN_COST_COMPUTE[0])]
        for field in ("gpu_device_ms", *COMPUTE_GPU_BRACKETS):
            no_gpu[0][field] = 0.0
        no_gpu[0]["gpu_timestamp_samples"] = 0
        _, text = self._render(no_gpu)
        self.assertIn("CPU phases of 300.0ms total", text)
        self.assertNotIn("GPU brackets of", text)
        # ...and the header must not advertise a section this capture cannot supply.
        self.assertNotIn("GPU brackets partition", text)

    def test_launch_shapes_show_direct_and_resolved_indirect_dimensions(self):
        record = dict(HIDDEN_COST_COMPUTE[0], dispatches=9,
                      dispatch_shapes=[
                          {"groups": [4, 5, 6], "local": [8, 2, 1],
                           "dispatches": 3, "indirect": False},
                          {"groups": [7, 1, 1], "local": [64, 1, 1],
                           "dispatches": 4, "indirect": True},
                      ], dispatch_shape_overflow=2)
        summary, text = self._render([record])
        shapes = summary["compute_launch_shapes"]
        self.assertTrue(shapes["available"])
        self.assertEqual((shapes["direct_dispatches"],
                          shapes["resolved_indirect_dispatches"],
                          shapes["unknown_dispatches"]), (3, 4, 2))
        self.assertIn("direct groups=4x5x6 local=8x2x1 dispatches=3", text)
        self.assertIn("resolved-indirect groups=7x1x1 local=64x1x1 dispatches=4", text)
        self.assertIn("unknown dispatches were omitted", text)

    def test_old_capture_leaves_launch_dimensions_explicitly_unavailable(self):
        _, text = self._render(HIDDEN_COST_COMPUTE)
        self.assertIn("compute launch shapes: UNAVAILABLE (capture predates launch-shape records)", text)

    def test_launch_shape_population_cannot_exceed_dispatch_count(self):
        malformed = dict(HIDDEN_COST_COMPUTE[0], dispatches=1,
                         dispatch_shapes=[{"groups": [1, 1, 1], "local": [1, 1, 1],
                                           "dispatches": 2, "indirect": True}],
                         dispatch_shape_overflow=0)
        with self.assertRaisesRegex(CaptureError, "does not match batch dispatches"):
            summarize(capture(SAMPLES, compute=[malformed]))

    def test_launch_shape_population_cannot_omit_dispatches(self):
        malformed = dict(HIDDEN_COST_COMPUTE[0], dispatches=3,
                         dispatch_shapes=[{"groups": [1, 1, 1], "local": [1, 1, 1],
                                           "dispatches": 2, "indirect": False}],
                         dispatch_shape_overflow=0)
        with self.assertRaisesRegex(CaptureError, "does not match batch dispatches"):
            summarize(capture(SAMPLES, compute=[malformed]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
