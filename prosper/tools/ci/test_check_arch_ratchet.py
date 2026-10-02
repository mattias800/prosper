"""Tests for check_arch_ratchet: one hand-built positive per rule, the lexer, verdicts, CLI.

Every rule gets a positive instance written here by hand rather than drawn from the repository,
so a matcher that quietly stops matching fails a test instead of reporting a clean tree forever.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_arch_ratchet as car  # noqa: E402

SLUGS = car.SELF_SLUGS


def values(tree):
    return car.scan_values(tree, SLUGS)


def kinds(tree, rows):
    return [p.kind for p in car.compare(car.scan(tree, SLUGS), rows)]


class OnePositivePerRule(unittest.TestCase):
    """Each arm is the only instance of its rule; it must be found, and found as NEW."""

    CASES = {
        "title-id": (
            {"prosper/src/hle/x.cpp": 'if (id == "PPSA12345") quirk();\n'},
            "title-id|prosper/src/hle/x.cpp",
            1,
        ),
        "title-id (frontends, CUSA)": (
            {"prosper/frontends/app/x.cpp": "auto t = CUSA00042;\n"},
            "title-id|prosper/frontends/app/x.cpp",
            1,
        ),
        "title-dir": (
            {"prosper/src/gpu/sonic_frontiers/a.cpp": "int a;\nint b;\nint c;\n"},
            "title-dir|prosper/src/gpu/sonic_frontiers",
            3,
        ),
        "title-dir (title-id basename)": (
            {"prosper/src/gpu/ppsa03831/a.hpp": "int a;\n"},
            "title-dir|prosper/src/gpu/ppsa03831",
            1,
        ),
        "getenv": (
            {"prosper/src/a.cpp": 'const char* v = std::getenv("PROSPER_X");\n'},
            "getenv|prosper/src/a.cpp",
            1,
        ),
        "getenv (render_runner.h)": (
            {car.RENDER_RUNNER: 'if (getenv("PROSPER_Y")) {}\n'},
            f"getenv|{car.RENDER_RUNNER}",
            1,
        ),
        "blocking-sync": (
            {"prosper/frontends/shared/a.cpp": "vkDeviceWaitIdle(dev);\n"},
            "blocking-sync|prosper/frontends/shared/a.cpp|vkDeviceWaitIdle",
            1,
        ),
        "blocking-sync (ALL_COMMANDS)": (
            {"prosper/src/a.cpp": "auto s = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;\n"},
            "blocking-sync|prosper/src/a.cpp|VK_PIPELINE_STAGE_ALL_COMMANDS_BIT",
            1,
        ),
        "file-size": (
            {"prosper/frontends/shared/big.cpp": car.BIG},
            "file-size|prosper/frontends/shared/big.cpp",
            car.FILE_SIZE_THRESHOLD + 1,
        ),
        "test-dep": (
            {"prosper/frontends/shared/a.cpp": "using X = prosper :: test :: Thing;\n"},
            "test-dep|prosper/frontends/shared/a.cpp",
            1,
        ),
    }

    def test_each_rule_fires_alone_and_reports_new(self):
        for label, (tree, key, value) in self.CASES.items():
            with self.subTest(label):
                self.assertEqual({key: value}, values(tree))
                self.assertEqual(["new"], kinds(tree, {}))

    def test_every_rule_has_a_positive_arm(self):
        covered = {car.rule_of(key) for _tree, key, _v in self.CASES.values()}
        self.assertEqual(set(car.RULES), covered)

    def test_selftest_tree_covers_every_rule(self):
        self.assertEqual(car.POSITIVE_KEYS, values(car.POSITIVE_TREE))
        self.assertEqual(set(car.RULES), {car.rule_of(k) for k in car.POSITIVE_KEYS})


class Negatives(unittest.TestCase):
    """Where each pattern must NOT count."""

    def test_comments_strings_lookalikes_and_other_roots(self):
        self.assertEqual({}, values(car.NEGATIVE_TREE))

    def test_file_at_threshold_is_not_large(self):
        tree = {"prosper/src/a.cpp": "int x;\n" * car.FILE_SIZE_THRESHOLD}
        self.assertEqual({}, values(tree))

    def test_title_dir_needs_a_directory_not_a_filename(self):
        self.assertEqual({}, values({"prosper/src/gpu/gta5.cpp": "int a;\n"}))

    def test_title_dir_only_under_src(self):
        self.assertEqual({}, values({"prosper/frontends/gta5/a.cpp": "int a;\n"}))

    def test_title_id_in_comment_is_allowed(self):
        tree = {"prosper/src/a.cpp": "// measured on PPSA24651's first level\nint a;\n"}
        self.assertEqual({}, values(tree))


class Lexer(unittest.TestCase):
    """The comment stripper, checked on the constructs that break naive ones."""

    def test_positions_and_newlines_preserved(self):
        text = 'a; // x\n/* y\n z */ b; "s//t"\n'
        code, bare = car.lex(text)
        self.assertEqual(len(text), len(code))
        self.assertEqual(len(text), len(bare))
        self.assertEqual(text.count("\n"), code.count("\n"))
        self.assertIn('"s//t"', code)
        self.assertNotIn("x", code)
        self.assertNotIn("s//t", bare)

    def test_code_after_tricky_literals_is_seen(self):
        for label, prefix in {
            "char quote": "char q = '\"';",
            "digit separator": "int n = 0x1'0000;",
            "escaped quote": 'auto s = "a\\"b";',
            "raw string with comment text": 'auto r = R"d(/* // ")d";',
            "u8 raw string": 'auto r = u8R"(x)";',
            "block comment": "/* a */",
        }.items():
            with self.subTest(label):
                tree = {"prosper/src/a.cpp": prefix + ' getenv("X");\n'}
                self.assertEqual({"getenv|prosper/src/a.cpp": 1}, values(tree))

    def test_line_numbers_are_real(self):
        tree = {"prosper/src/a.cpp": '/* x\n y */\nint a;\ngetenv("X");\n'}
        self.assertEqual([4], car.scan(tree, SLUGS)["getenv|prosper/src/a.cpp"].lines)


class Verdicts(unittest.TestCase):
    """The four failure kinds, and the cap band."""

    def setUp(self):
        self.found = car.scan(car.POSITIVE_TREE, SLUGS)
        self.rows = {k: car.Row(k, f.value) for k, f in self.found.items()}

    def test_equal_is_clean(self):
        self.assertEqual([], car.compare(self.found, self.rows))

    def test_stale_baseline_row_fails(self):
        rows = dict(self.rows, **{"getenv|prosper/src/gone.cpp": car.Row("getenv|x", 3)})
        self.assertEqual(["stale"], [p.kind for p in car.compare(self.found, rows)])

    def test_decrease_requires_baseline_update(self):
        rows = dict(self.rows)
        rows["getenv|prosper/src/e.cpp"] = car.Row("getenv|prosper/src/e.cpp", 5)
        problems = car.compare(self.found, rows)
        self.assertEqual(["decrease"], [p.kind for p in problems])
        self.assertEqual(2, problems[0].current)

    def test_increase_fails(self):
        rows = dict(self.rows)
        rows["getenv|prosper/src/e.cpp"] = car.Row("getenv|prosper/src/e.cpp", 1)
        self.assertEqual(["increase"], [p.kind for p in car.compare(self.found, rows)])

    def test_cap_band(self):
        key = "file-size|prosper/tests/fixtures/big.h"
        size = car.FILE_SIZE_THRESHOLD + 1
        inside = int(size / (1 - car.SHRINK_MARGIN))  # file is within 2% of this cap
        outside = inside + 200
        for cap, want in ((size - 1, ["increase"]), (inside, []), (outside, ["decrease"])):
            with self.subTest(cap=cap):
                rows = dict(self.rows, **{key: car.Row(key, cap)})
                self.assertEqual(want, [p.kind for p in car.compare(self.found, rows)])

    def test_update_lowers_and_deletes_but_never_raises_or_adds(self):
        rows = dict(self.rows)
        rows["getenv|prosper/src/e.cpp"] = car.Row("getenv|prosper/src/e.cpp", 5, "kept note")
        rows["getenv|prosper/src/gone.cpp"] = car.Row("getenv|prosper/src/gone.cpp", 2)
        rows["test-dep|prosper/frontends/g.cpp"] = car.Row("test-dep|prosper/frontends/g.cpp", 0)
        del rows["title-id|prosper/src/a.cpp"]
        repaired = car.apply_repairs(rows, car.compare(self.found, rows))
        self.assertEqual(2, repaired["getenv|prosper/src/e.cpp"].value)
        self.assertEqual("kept note", repaired["getenv|prosper/src/e.cpp"].note)
        self.assertNotIn("getenv|prosper/src/gone.cpp", repaired)
        self.assertEqual(0, repaired["test-dep|prosper/frontends/g.cpp"].value)  # not raised
        self.assertNotIn("title-id|prosper/src/a.cpp", repaired)  # not added
        self.assertEqual(["new", "increase"], [p.kind for p in car.compare(self.found, repaired)])


class BaselineFormat(unittest.TestCase):
    """Parsing: round trip, and refusals that map to exit 2."""

    def test_round_trip(self):
        text = "# head\n\ngetenv|a.cpp 3  # why\ntitle-id|b.cpp 1\n"
        header, rows = car.parse_baseline(text)
        self.assertEqual(["# head"], header)
        self.assertEqual("why", rows["getenv|a.cpp"].note)
        self.assertEqual(
            car.parse_baseline(car.format_baseline(header, rows.values())), (header, rows)
        )

    def test_refusals(self):
        for bad in ("getenv|a 1 2\n", "nonsense|a 1\n", "getenv|a 1\ngetenv|a 2\n", "getenv|a 0\n"):
            with self.subTest(bad=bad), self.assertRaises(car.EvaluationError):
                car.parse_baseline(bad)


class Cli(unittest.TestCase):
    """The whole gate against a throwaway git repository: exit 0, 1 and 2."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for path, text in car.POSITIVE_TREE.items():
            self.write(path, text)
        for n in range(car.MIN_SLUGS):
            self.write(f"prosper/scripts/title{n}/route.pad", "x\n")
        self.write("prosper/scripts/gta5-PPSA04263/route.pad", "x\n")
        self.git("init", "-q")
        self.git("add", "-A")
        self.baseline = self.root / "baseline.txt"
        rows = [car.Row(k, v) for k, v in car.POSITIVE_KEYS.items()]
        self.baseline.write_text(car.format_baseline(["# test"], rows), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def git(self, *args):
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def gate(self, *extra):
        return car.main(["--root", str(self.root), "--baseline", str(self.baseline), *extra])

    def test_clean_tree_passes(self):
        self.assertEqual(car.EXIT_OK, self.gate())

    def test_new_instance_fails(self):
        self.write("prosper/src/new.cpp", 'auto v = getenv("X");\n')
        self.git("add", "-A")
        self.assertEqual(car.EXIT_VIOLATION, self.gate())

    def test_untracked_file_is_not_scanned(self):
        self.write("prosper/src/untracked.cpp", 'auto v = getenv("X");\n')
        self.assertEqual(car.EXIT_OK, self.gate())

    def test_update_repairs_a_decrease_then_passes(self):
        self.write("prosper/src/e.cpp", 'auto v = getenv("X");\n')
        self.assertEqual(car.EXIT_VIOLATION, self.gate())
        self.assertEqual(car.EXIT_OK, self.gate("--update"))
        self.assertIn("getenv|prosper/src/e.cpp 1\n", self.baseline.read_text(encoding="utf-8"))
        self.assertEqual(car.EXIT_OK, self.gate())

    def test_unevaluable_is_two_not_zero(self):
        self.assertEqual(car.EXIT_UNEVALUATED, car.main(["--root", str(self.root / "nowhere")]))
        self.baseline.write_text("getenv|a not-a-number\n", encoding="utf-8")
        self.assertEqual(car.EXIT_UNEVALUATED, self.gate())

    def test_too_few_slugs_is_unevaluable(self):
        self.git("rm", "-q", "--cached", "-r", "prosper/scripts")
        self.assertEqual(car.EXIT_UNEVALUATED, self.gate())


class RealTree(unittest.TestCase):
    """The committed baseline describes the committed tree, and the selftest passes."""

    def test_selftest(self):
        self.assertEqual(car.EXIT_OK, car.selftest())

    def test_repository_matches_baseline(self):
        self.assertEqual(car.EXIT_OK, car.main(["--root", str(HERE.parents[2])]))


if __name__ == "__main__":
    unittest.main()
