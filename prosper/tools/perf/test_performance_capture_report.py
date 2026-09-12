#!/usr/bin/env python3

import unittest

import contextlib
import io

from performance_capture_report import (CaptureError, CLASSIFICATION_EVIDENCE_SHARE,
                                        READBACK_NOTE_MIN_SHARE, print_summary, summarize,
                                        validate_capture)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
