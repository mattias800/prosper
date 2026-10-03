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
        "platform-ifdef": (
            {"prosper/src/hle/sync/a.cpp": "#if defined(__linux__)\n#endif\n"},
            "platform-ifdef|prosper/src/hle/sync/a.cpp",
            1,
        ),
        "layer-include": (
            {"prosper/src/host/image/a.cpp": '#include "hle/dispatch/dispatch.hpp"\n'},
            "layer-include|prosper/src/host/image/a.cpp|hle",
            1,
        ),
        "fixture-include": (
            {"prosper/frontends/shared/live/a.cpp": '#include "fixtures/render_runner.h"\n'},
            "fixture-include|prosper/frontends/shared/live/a.cpp",
            1,
        ),
        "vk-object": (
            {"prosper/src/gpu/a.cpp": "vkCreateDescriptorPool(d, &i, nullptr, &p);\n"},
            "vk-object|prosper/src/gpu/a.cpp|vkCreateDescriptorPool",
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
        self.baseline = self.root / car.CANONICAL_BASELINE
        self.baseline.parent.mkdir(parents=True, exist_ok=True)
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


class PathContainment(unittest.TestCase):
    """Every path argument must resolve inside --root; --update writes only the canonical file.

    Addresses the SonarCloud path-injection finding on --update's baseline write (#4199).
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "repo"
        self.canonical = self.root / car.CANONICAL_BASELINE
        self.canonical.parent.mkdir(parents=True)
        self.canonical.write_text("# t\n", encoding="utf-8")
        self.outside = Path(self.tmp.name) / "outside.txt"
        self.outside.write_text("# t\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, baseline, update=False):
        with self.assertRaises(car.EvaluationError):
            car.contained_paths(str(self.root), baseline, update)

    def test_relative_traversal_is_refused(self):
        self.refused("../outside.txt")
        self.refused("../outside.txt", update=True)

    def test_absolute_path_outside_is_refused(self):
        self.refused(str(self.outside))

    def test_symlink_escaping_the_root_is_refused(self):
        link = self.root / "escape.txt"
        try:
            link.symlink_to(self.outside)
        except (OSError, NotImplementedError):
            self.skipTest("this host does not allow creating symlinks")
        self.refused("escape.txt")

    def test_update_writes_only_the_canonical_file(self):
        other = self.root / "other.txt"
        other.write_text("# t\n", encoding="utf-8")
        self.refused("other.txt", update=True)
        _root, baseline = car.contained_paths(str(self.root), "other.txt", False)
        self.assertEqual(baseline, other.resolve())

    def test_canonical_path_is_accepted(self):
        for arg in (None, car.CANONICAL_BASELINE, str(self.canonical)):
            _root, baseline = car.contained_paths(str(self.root), arg, True)
            self.assertEqual(baseline, self.canonical.resolve())

    def test_main_maps_refusal_to_two_and_writes_nothing(self):
        rc = car.main(["--root", str(self.root), "--baseline", "../outside.txt", "--update"])
        self.assertEqual(car.EXIT_UNEVALUATED, rc)
        self.assertEqual("# t\n", self.outside.read_text(encoding="utf-8"))

    def test_option_like_base_is_refused(self):
        rc = car.main(["--root", str(self.root), "--base=--output=x"])
        self.assertEqual(car.EXIT_UNEVALUATED, rc)


class StructuralRules(unittest.TestCase):
    """Hand-built positive and negative instances of the structural rules."""

    def test_platform_ifdef_counts_directives_outside_host_only(self):
        body = "#if defined(_MSC_VER) || defined(__MINGW32__)\n#endif\n#ifndef __APPLE__\n#endif\n"
        files = {"prosper/src/loader/a.cpp": body, "prosper/src/host/memory/b.cpp": body}
        self.assertEqual({"platform-ifdef|prosper/src/loader/a.cpp": 2}, car.scan_values(files))

    def test_platform_ifdef_ignores_comments_and_unrelated_macros(self):
        body = "// #ifdef _WIN32\n#ifdef NDEBUG\n#endif\n#define X _WIN32\n"
        self.assertEqual({}, car.scan_values({"prosper/src/gpu/a.cpp": body}))

    def test_layer_include_direction(self):
        files = {
            "prosper/src/hle/a.cpp": '#include "host/x.hpp"\n#include "gpu/y.hpp"\n',
            "prosper/src/self/b.cpp": '#include "loader/z.hpp"\n#include "self/w.hpp"\n',
            "prosper/src/gpu/c.cpp": '#include <hle/q.hpp>\n#include "hle/r.hpp"\n',
        }
        want = {
            "layer-include|prosper/src/self/b.cpp|loader": 1,
            "layer-include|prosper/src/gpu/c.cpp|hle": 2,
        }
        self.assertEqual(want, car.scan_values(files))

    def test_layer_include_resolves_parent_relative_paths(self):
        files = {
            "prosper/src/diagnostics/d.cpp": '#include "../../frontends/shared/p.hpp"\n',
            "prosper/src/diagnostics/core/e.cpp": '#include "../boot_phase_log.hpp"\n',
            "prosper/src/loader/f.hpp": (
                '#include "../self/module.hpp"\n#include "../build_revision.hpp"\n'
            ),
        }
        want = {"layer-include|prosper/src/diagnostics/d.cpp|frontends": 1}
        self.assertEqual(want, car.scan_values(files))

    def test_layer_include_never_into_tests(self):
        files = {"prosper/src/host/t.cpp": '#include "../../tests/fixtures/x.h"\n'}
        self.assertEqual({"layer-include|prosper/src/host/t.cpp|tests": 1}, car.scan_values(files))

    def test_fixture_include_only_from_frontends(self):
        inc = '#include "fixtures/render_runner.h"\n'
        files = {"prosper/frontends/a.cpp": inc, "prosper/tests/host/b.cpp": inc}
        self.assertEqual({"fixture-include|prosper/frontends/a.cpp": 1}, car.scan_values(files))

    def test_vk_object_counts_calls_not_types_or_strings(self):
        body = (
            "vkAllocateMemory(d, &i, nullptr, &m);\n"
            "PFN_vkAllocateMemory p;\n"
            'log("vkCreateFence(");\n'
        )
        want = {"vk-object|prosper/src/gpu/a.cpp|vkAllocateMemory": 1}
        self.assertEqual(want, car.scan_values({"prosper/src/gpu/a.cpp": body}))


class Delta(unittest.TestCase):
    """Delta mode against a throwaway repository: a base commit, then a change on top of it.

    The base commits a baseline that is already STALE for e.cpp (its row is below the file's
    real count) -- the state main is in whenever a PR grows a capped file. Full mode fails on it;
    delta mode must not, unless the change is the one that raised it.
    """

    E_CPP = "prosper/src/e.cpp"
    STALE_KEY = f"getenv|{E_CPP}"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for path, text in car.POSITIVE_TREE.items():
            self.write(path, text)
        for n in range(car.MIN_SLUGS):
            self.write(f"prosper/scripts/title{n}/route.pad", "x\n")
        self.write("prosper/scripts/gta5-PPSA04263/route.pad", "x\n")
        self.baseline = self.root / car.CANONICAL_BASELINE
        self.baseline.parent.mkdir(parents=True, exist_ok=True)
        rows = {k: car.Row(k, v) for k, v in car.POSITIVE_KEYS.items()}
        rows[self.STALE_KEY] = car.Row(self.STALE_KEY, 1)  # e.cpp really has 2
        self.write_rows(rows)
        self.git("init", "-q")
        self.git("add", "-A")
        self.git("commit", "-q", "-m", "base")
        self.git("branch", "base")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, text):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def append(self, path, text):
        self.write(path, (self.root / path).read_text(encoding="utf-8") + text)

    def write_rows(self, rows):
        self.baseline.write_text(car.format_baseline(["# t"], rows.values()), encoding="utf-8")

    def raise_row(self, key, value):
        _header, rows = car.parse_baseline(self.baseline.read_text(encoding="utf-8"))
        rows[key] = car.Row(key, value, "justified")
        self.write_rows(rows)

    def git(self, *args):
        cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args]
        subprocess.run(cmd, cwd=self.root, check=True, capture_output=True)

    def delta(self, base="base"):
        return car.main(
            ["--root", str(self.root), "--baseline", str(self.baseline), "--base", base]
        )

    def full(self):
        return car.main(["--root", str(self.root), "--baseline", str(self.baseline)])

    def test_unrelated_preexisting_over_row_file_does_not_fail(self):
        self.append("prosper/src/a.cpp", "int unrelated;\n")
        self.git("commit", "-qam", "unrelated")
        self.assertEqual(car.EXIT_VIOLATION, self.full())  # the stale row is real...
        self.assertEqual(car.EXIT_OK, self.delta())  # ...and not this change's doing

    def test_each_kind_of_rise_in_a_changed_file_fails(self):
        cases = {
            "getenv": ("prosper/src/a.cpp", 'auto g = getenv("NEW");\n'),
            "title-id": ("prosper/src/e.cpp", 'auto t = "PPSA99999";\n'),
            "blocking-sync": ("prosper/src/f.cpp", "vkWaitForFences(d, 1, &f, 1, 0);\n"),
            "file-size": ("prosper/tests/fixtures/big.h", "int y;\n"),
            "title-dir": ("prosper/src/gpu/gta5/c.cpp", "int c;\n"),
        }
        for label, (path, text) in cases.items():
            with self.subTest(label):
                self.git("reset", "-q", "--hard", "base")
                self.append(path, text)  # unstaged: delta mode must see the working tree
                self.assertEqual(car.EXIT_VIOLATION, self.delta())

    def test_a_raised_row_in_the_same_change_passes(self):
        self.append("prosper/src/f.cpp", "vkWaitForFences(d, 1, &f, 1, 0);\n")
        self.assertEqual(car.EXIT_VIOLATION, self.delta())
        self.raise_row("blocking-sync|prosper/src/f.cpp|vkWaitForFences", 2)
        self.assertEqual(car.EXIT_OK, self.delta())

    def test_raising_an_already_stale_row_needs_to_cover_the_new_count(self):
        self.append(self.E_CPP, 'getenv("Z");\n')  # 2 -> 3 against a stale row of 1
        self.assertEqual(car.EXIT_VIOLATION, self.delta())
        self.raise_row(self.STALE_KEY, 2)  # covers the merge base, not the new count
        self.assertEqual(car.EXIT_VIOLATION, self.delta())
        self.raise_row(self.STALE_KEY, 3)
        self.assertEqual(car.EXIT_OK, self.delta())

    def test_new_untracked_file_is_compared_against_zero(self):
        self.write("prosper/src/brand_new.cpp", 'auto v = getenv("X");\n')
        self.assertEqual(car.EXIT_VIOLATION, self.delta())

    def test_committed_change_is_seen(self):
        self.append("prosper/src/a.cpp", 'auto g = getenv("NEW");\n')
        self.git("commit", "-qam", "grow")
        self.assertEqual(car.EXIT_VIOLATION, self.delta())

    def test_decrease_and_deletion_never_fail(self):
        self.write(self.E_CPP, "int none;\n")
        (self.root / "prosper/src/f.cpp").unlink()
        self.assertEqual(car.EXIT_OK, self.delta())

    def test_branch_behind_its_base_is_not_blamed_for_the_base(self):
        # main moves on and removes a getenv from e.cpp (fixing its stale row); this branch, cut
        # before that, changes a.cpp only. Diffed against main's TIP, e.cpp would read as 1 -> 2
        # over a row of 1; against the merge base it is unchanged, which is the truth.
        self.git("checkout", "-q", "-b", "main2")
        self.write(self.E_CPP, 'auto v = std::getenv("X");\n')
        self.git("commit", "-qam", "main shrinks e.cpp")
        self.git("checkout", "-q", "-b", "topic", "base")
        self.append("prosper/src/a.cpp", "int other;\n")
        self.git("commit", "-qam", "topic")
        self.assertEqual(car.EXIT_OK, self.delta("main2"))

    def test_base_resolution_peels_tags_and_rejects_non_commits(self):
        # Resolved without the `<ref>^{commit}` spelling, which reaches git mangled on the
        # Windows/MSYS route (every --base then read as "not a commit"): an annotated tag must
        # still peel to its commit, and a tree must still be refused.
        self.git("tag", "-a", "-m", "t", "reltag", "base")
        self.assertEqual(car.EXIT_OK, self.delta("reltag"))
        self.assertEqual(car.EXIT_UNEVALUATED, self.delta("base:prosper"))

    def test_unreachable_base_is_two(self):
        self.assertEqual(car.EXIT_UNEVALUATED, self.delta("no-such-ref"))
        self.assertEqual(car.EXIT_UNEVALUATED, self.delta("0" * 40))


class RealTree(unittest.TestCase):
    """The checker evaluates the committed tree, and this change raised nothing past its row.

    Full mode is asserted only to EVALUATE (0 or 1, never 2): whether main's baseline is
    currently stale is a fact about main, not about the change under test, and asserting it
    here would redden every unrelated PR the moment another one grows a capped file.
    """

    def test_selftest(self):
        self.assertEqual(car.EXIT_OK, car.selftest())

    def test_full_mode_evaluates_the_repository(self):
        rc = car.main(["--root", str(HERE.parents[2])])
        self.assertIn(rc, (car.EXIT_OK, car.EXIT_VIOLATION))

    def test_delta_mode_against_origin_main(self):
        root = HERE.parents[2]
        probe = subprocess.run(
            ["git", "rev-list", "-n", "1", "origin/main", "--"],
            cwd=root,
            capture_output=True,
        )
        if probe.returncode != 0:
            self.skipTest("origin/main is not available in this checkout")
        self.assertEqual(car.EXIT_OK, car.main(["--root", str(root), "--base", "origin/main"]))


if __name__ == "__main__":
    unittest.main()
