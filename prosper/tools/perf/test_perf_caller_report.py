#!/usr/bin/env python3
"""Known stack populations constrain missing-caller and period accounting."""
from pathlib import Path
import tempfile
import unittest

import perf_caller_report as tool

APP = "/capture/prosper-app"
LIBC = "/usr/lib/libc.so.6"


def frame(symbol, dso=LIBC, ip="7f001000"):
    return f"\t{ip} {symbol} ({dso})\n"


def sample(period, frames, event="u", pid=10, tid=11, comm="render thread"):
    return f"{comm} {pid}/{tid} 12.000001: {period} cpu-cycles:{event}:\n{frames}\n"


class CallerReport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "stacks.txt"

    def run_report(self, text, records=None):
        self.path.write_text(text)
        raw = None
        if records is not None:
            raw = self.path.with_name("records.txt")
            raw.write_text(records)
        return tool.report(self.path, {APP}, {LIBC}, raw)

    def test_weighted_recovery_and_missing_callers(self):
        text = sample(10, frame("__memmove_avx512_unaligned_erms"))
        text += sample(30, frame("__memcpy_avx512_unaligned_erms") + frame("upload()", APP, "401000"))
        text += sample(60, frame("render()", APP, "402000"))
        result = self.run_report(text)
        event = result["events"]["cpu-cycles:u"]["aggregate"]
        copy = event["families"]["copy"]
        self.assertEqual(event["total"], {"samples": 3, "period": 100})
        self.assertEqual(copy["total"]["samples"], 2)
        self.assertEqual(copy["leaf_only"]["percent_family_periods"], 25)
        self.assertEqual(copy["named_app_ancestor"]["percent_family_periods"], 75)
        self.assertEqual(copy["named_app_ancestor"]["percent_event_periods"], 30)
        self.assertEqual(copy["callers"][0]["symbol"], "upload()")
        self.assertEqual(copy["callers"][0]["period"], 30)
        self.assertIsNone(result["observed_record_loss"])

    def test_exact_dso_and_named_ancestor(self):
        text = sample(4, frame("memcpy", "/different/libc.so.6") + frame("bad()", APP))
        text += sample(5, frame("memcpy") + frame("imposter()", "/other/prosper-app"))
        text += sample(6, frame("memcpy") + frame("[unknown]", APP))
        copy = self.run_report(text)["events"]["cpu-cycles:u"]["aggregate"]["families"]["copy"]
        self.assertEqual(copy["total"]["period"], 11)
        self.assertEqual(copy["no_named_app_ancestor"]["samples"], 2)
        self.assertEqual(copy["named_app_ancestor"]["samples"], 0)

    def test_unresolved_and_suspicious_chains_stay_visible(self):
        text = sample(7, frame("malloc") + frame("[unknown]", "[unknown]", "91") + frame("owner()", APP))
        row = self.run_report(text)["events"]["cpu-cycles:u"]["aggregate"]["families"]["allocation"]
        self.assertEqual(row["named_app_ancestor"]["period"], 7)
        self.assertEqual(row["unresolved_before_app"]["period"], 7)
        self.assertEqual(row["suspicious_low_ip"]["period"], 7)
        self.assertEqual(row["leaf_only"]["period"], 0)

    def test_low_app_address_does_not_count_as_recovered(self):
        text = sample(9, frame("malloc") + frame("owner()", APP, "91"))
        row = self.run_report(text)["events"]["cpu-cycles:u"]["aggregate"]["families"]["allocation"]
        self.assertEqual(row["named_app_ancestor"]["period"], 0)
        self.assertEqual(row["suspicious_low_ip"]["period"], 9)

    def test_thread_and_event_denominators(self):
        text = sample(100, frame("memcmp"), tid=11)
        text += sample(300, frame("memset"), tid=12)
        text += sample(2000, frame("memcmp"), event="k", tid=11)
        events = self.run_report(text)["events"]
        self.assertEqual(events["cpu-cycles:u"]["aggregate"]["total"]["period"], 400)
        self.assertEqual(events["cpu-cycles:k"]["aggregate"]["total"]["period"], 2000)
        threads = events["cpu-cycles:u"]["threads"]
        self.assertEqual([(r["tid"], r["total"]["period"]) for r in threads], [(11, 100), (12, 300)])
        self.assertEqual(threads[0]["families"]["compare"]["total"]["percent_event_periods"], 100)
        self.assertEqual(events["cpu-cycles:u"]["aggregate"]["families"]["compare"]["total"]["percent_event_periods"], 25)

    def test_family_scope_and_zero_denominator(self):
        known = [("_int_malloc", "allocation"), ("__libc_malloc2", "allocation"), ("__libc_calloc+0x20", "allocation"),
                 ("__GI___libc_free", "free"), ("_int_free_chunk", "free"),
                 ("__memcmp_evex_movbe", "compare"), ("__memset_avx512", "fill")]
        for symbol, expected in known:
            with self.subTest(symbol=symbol):
                self.assertEqual(tool.family(dict(symbol=symbol, dso=LIBC), {LIBC}), expected)
        self.assertIsNone(tool.family(dict(symbol="application_memcpy", dso=LIBC), {LIBC}))
        result = self.run_report(sample(1, frame("memcpy")))["events"]["cpu-cycles:u"]["aggregate"]
        self.assertEqual(result["families"]["copy"]["total"]["percent_event_periods"], 100)
        self.assertIsNone(result["families"]["free"]["total"]["percent_family_periods"])

    def test_selector_visibility_and_disjointness(self):
        result = self.run_report(sample(2, frame("memcpy")))
        self.assertEqual(result["unobserved_selectors"], [APP])
        self.assertEqual(result["selector_matches"][LIBC], {"frames": 1, "named_frames": 1})
        with self.assertRaisesRegex(ValueError, "disjoint"):
            tool.report(self.path, {APP, LIBC}, {LIBC})

    def test_zero_weight_event_refuses(self):
        with self.assertRaisesRegex(ValueError, "zero aggregate period"):
            self.run_report(sample(0, frame("memcpy")))

    def test_empty_stack_is_counted_not_clean(self):
        row = self.run_report(sample(123, ""))["events"]["cpu-cycles:u"]["aggregate"]
        self.assertEqual(row["empty"], {"samples": 1, "period": 123})

    def test_lost_markers_are_visible(self):
        result = self.run_report(sample(4, frame("memcpy")),
            "12 0xa [0x40]: PERF_RECORD_SAMPLE(IP, 0x2): 10/11\n"
            "PERF_RECORD_LOST 8\nPERF_RECORD_LOST_SAMPLES 3\n")
        self.assertEqual(result["observed_record_loss"], {"sample_record_markers": 1, "lost_record_markers": 2})

    def test_raw_sample_count_mismatch_refuses(self):
        for raw in ["", "PERF_RECORD_LOST 12\n", "PERF_RECORD_SAMPLE(IP, 0x2)\n" * 2]:
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "count mismatch"):
                self.run_report(sample(4, frame("memcpy")), raw)

    def test_parser_refuses_void_or_unsupported_data(self):
        for text in ["", "\n", frame("memcpy"), "warning: truncated export\n",
                     sample(4, frame("memcpy")).replace("cpu-cycles:u", "cpu-clock:u"),
                     sample(4, frame("memcpy")) + "unparsed trailing record\n"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.run_report(text)


if __name__ == "__main__":
    unittest.main()
