# tools/ci — checks about the process, not about the emulator

Everything here answers a question about **how a change is being made**, never about whether the
emulator is correct. Nothing in this folder loads a module, decodes a packet or touches a GPU. If a
check needs a dump, a Vulkan device or a guest, it does not belong here — it belongs in `tests/`.

The boundary against its siblings: `tools/docs/` validates the *content* of Markdown (numbered
tables, trap citations, cross-references); this folder validates the *mechanics* around a change —
the shape of a contribution, whether a test gate was actually armed, whether a tool still prints its
usage, whether a PR is safe to merge.

## What lives here

- **`check_contribution_shape.py`** — the contribution-shape CI job.
- **`check_python_quality.py`** — the `python-quality` CI job: a ratchet over the Python a PR
  *changes*. Added files must be ruff-clean, ruff-format-clean and carry a module docstring; modified
  files may not gain ruff findings; an added tool under `prosper/tools/` needs a test in the same PR.
  Config lives in the root `pyproject.toml` (uv, `uv.lock`); see the Python rule below.
- **`check_pytest_policy.py`** + `pytest_legacy_allowlist.txt` — enforces that new Python test files
  (`test_*.py` under `prosper/`) use pytest conventions (functions/asserts/fixtures, not legacy
  `unittest.TestCase` subclasses or script-only uncollected files). The allowlist baseline records
  pre-existing files and only shrinks. `--base REF` must resolve to an immutable commit; a failed
  lookup exits 2. Initial rollout explicitly uses `--bootstrap-allowlist`, which permits only
  already non-compliant sources in that base tree, never newly added exemptions.
- **`check_gtest_policy.py`** + `gtest_legacy_allowlist.txt` — enforces that new C++ tests use GoogleTest
  rather than a hand-rolled `main()`. The allowlist baseline records pre-existing files and only shrinks.
- **`check_cpp_lint.py`** — the C++ half of the same idea: root `.clang-format` and `.clang-tidy`,
  enforced on the lines a change **adds or modifies**, against the merge base. It runs last in the
  Linux CI job because clang-tidy needs that job's `compile_commands.json`. See *C++ lint* below.
- **`check_arch_ratchet.py`** + `arch_ratchet_baseline.txt` — a ratchet over costs every title
  pays: title ids and title-named directories in shared code, raw `getenv` reads, blocking GPU syncs
  (`vkWaitForFences`, `*WaitIdle`, `ALL_COMMANDS` barriers), files over 5,000 lines, and
  `prosper::test::` in the shipping frontend; and five structural ones: host-platform `#if`s
  outside `src/host` (`platform-ifdef`, seam: `docs/architecture/HOST_PLATFORM_SEAM.md`), `HLE(*_stub)` platform-arm
  handlers (`platform-stub`, a name-based proxy), includes against the
  layer order (`layer-include`, `LAYER_ORDER` in the checker; `docs/architecture/ARCHITECTURE_TARGET_TREE.md`),
  frontends including `tests/fixtures/` (`fixture-include`), and Vulkan object-creation call sites
  (`vk-object`). The practice -> rule table, including what is not mechanically checkable yet
  (giant functions: owned by the clang-tidy PR's `readability-function-size`), is in
  `docs/architecture/ARCHITECTURE_TARGET_TREE.md` § Bad practices. Path arguments are resolved and must stay
  inside `--root`, and `--update` writes only `prosper/tools/ci/arch_ratchet_baseline.txt`. Per-file counts may go down, never up; a count that
  fell must have its row lowered (`--update` lowers and deletes, never raises). Raising a row is a
  reviewed baseline edit with a `# note` — for a new sync, naming the guest-visible result it
  delivers. **Two modes:** `--base REF` (delta — what CI gates on) looks only at files changed
  since the merge base of HEAD and REF, working tree included, and fails only when *this* change
  raised a count past its row; the plain full-tree run is the baseline-maintenance report and is
  informational in CI, because main's baseline goes stale whenever a PR grows a capped file and
  that must not redden every unrelated PR. It is not a refactoring plan; it only stops the numbers
  getting worse while one happens.
- **`check_ctest_gate.py`** — finds callers that run `ctest` without `--no-tests=error`. Plain
  `ctest` exits 0 when it finds no tests, so "nothing ran" and "everything passed" share a status.
- **`check_usage_text.py`** — finds tools whose usage block stopped being a docstring, so
  `print(__doc__)` prints the literal `None` and the caller is told nothing.
- **`pr_merge_gate.py`** — is this PR safe to merge right now? Counts checks by bucket from
  `gh --json` and refuses when the PR's recorded head is no longer the branch tip.
- **`pr_body.py`** — set a PR body and PROVE it took, or read one back and prove it matches a file.
  `gh pr edit --body-file` has been seen returning rc=1 on a GraphQL projects-deprecation error
  *without applying the edit* (#2918), so `set` writes over REST and then re-reads the live body;
  the verdict is always the read-back, never the write's exit code.
- **`ratchet_hook.py`** — the Claude Code PreToolUse hook (`.claude/settings.json`, Bash and
  PowerShell) that runs `check_arch_ratchet.py --base origin/main` (delta mode: only what this
  checkout changed since its merge base) before `git commit` / `git push`. The rollout is
  **warn-only**: a violation is shown as a `systemMessage` and the command proceeds;
  `BLOCK_ON_VIOLATION` is the single switch that makes it block. A missing checker, a missing
  `origin/main`, an exit 2 or a missing interpreter fails open with a visible `ratchet UNVERIFIED`
  message. CI remains the gate.

## Secret scan (`.github/workflows/gitleaks.yml`, `.github/gitleaks.toml`)

Gitleaks reads every commit a PR adds (`base..head`), every push to `main`, and the full history
weekly. The binary is a pinned release checked against a sha256 (no third-party action, no token);
findings are printed redacted. The config extends the default ruleset and allowlists only measured
false positives, scoped to one rule via `targetRules` (baseline 2026-10-02: 10 findings in 4,040
commits, all identifiers containing "key"; NID hashes, shader hex and fixtures produced none, so they
are not exempted). To add an entry, run the scan, open the finding unredacted **locally**, prove it is
not a credential, then add a narrow `paths`/`regexes` line with that evidence beside it. A real secret is
never allowlisted: rotate it and tell the owner privately. Run locally with
`gitleaks git . --config .github/gitleaks.toml --redact` (add `--log-opts "origin/main..HEAD"` for a PR
range). A clean zero is only meaningful once a hand-built fake token is seen to fail the same command.

## Python rule (applies to every Python file in the repository)

Python behavior changes need a meaningful regression that **fails without the change**, reusing
applicable pytest or stdlib `unittest`/`--selftest` cases. Register host-side cases in ctest.
Mechanical changes use relevant checks and state their scope and execution limits; they need no
test mirroring the implementation. Every new `.py` file opens with a purpose docstring (ruff `D100`,
enforced on added files). Follow the task's execution authority and resource coordination.

Measured baseline, 2026-10-02: 211 tracked `.py` files, 1,520 ruff findings under the config in
`pyproject.toml`, 209 files that `ruff format` would rewrite. That is why the gate is a ratchet and
not a sweep: **do not mass-reformat**, and do not reformat a file you only touch for a fix. Lowering
a file's findings is welcome; raising them fails CI. `pytest` collects only the files listed in `testpaths` (`pyproject.toml`): of 77 `test_*.py`, 12 collect
tests, 40 are script-style (ctest runs them as programs; pytest collects nothing) and 25 fail at import,
so ctest stays the source of truth. When you convert a test to unittest/pytest style, add it to
`testpaths`. Set up with `uv sync --group dev`, run with
`uv run --group dev pytest` and `uv run --group dev ruff check <file>`.

## C++ lint (`check_cpp_lint.py`, `.clang-format`, `.clang-tidy`)

The tree predates both configs: at the 2026-10-02 census `.clang-format` would rewrite ~30% of the
lines under `prosper/src` and `prosper/frontends`, and the candidate tidy set found 7,196 issues.
So both are judged on changed lines only. **Never run clang-format over a whole file**, and never
mass-fix tidy findings in a PR about something else. Each config file records its census counts
and the reason for every disabled check. Re-run the census (clang-tidy over the whole compile
database, counting unique file:line:check) before you turn a check on. Tool versions are
exact-pinned in the `cpp-lint` uv group, so bumping them is its own PR. `misc-include-cleaner`
only warns until the header-cost work settles umbrella headers. Suppress a finding with
`// NOLINT(check-name): reason`: no bare `NOLINT`, and the total count should go down over time.

    uv sync --group cpp-lint
    uv run --no-sync python prosper/tools/ci/check_cpp_lint.py --build prosper/build-linux
    uv run --no-sync git-clang-format "$(git merge-base origin/main HEAD)"   # apply format fixes

On a MinGW build, add `--extra-arg=--target=x86_64-w64-mingw32`, because clang otherwise assumes
MSVC and cannot compile the TU. `--no-tidy` checks formatting with no build at all.
`--check-db` is the positive control. The gate skips any file clang cannot compile, so it proves
that a real TU from the database does compile.

## The property they share, and why it dictates how they are tested

**Every check here fails silently when it breaks.** A matcher that stops matching reports a clean
tree forever. A gate that mis-parses reports green. None of these produce an error when they go
wrong — they produce a *reassuring answer*, which is worse than an error because it stops the reader
looking.

So each carries self-tests that pin its own behaviour rather than relying on the repository
happening to contain a violation, and each is registered in ctest. When you add a check here, add
the arm that would redden if its matcher quietly widened. `test_pr_merge_gate.py` also carries the
harder half for a boolean gate: because nearly every arm is a refusal, and a gate that refuses
*everything* satisfies all of them, it leads with a positive control and asserts on the refusal
*reason* rather than on the boolean.

## Reading back is the only proof a remote write happened

`pr_body.py` is in this folder rather than beside the other `gh` helpers because it is the same
species as the checks above: the thing it guards against is a *reassuring answer*. `gh pr edit`
exiting 0 is what everybody reads, and on #2910 it was wrong while the corrected body — four
blocking review findings, all prose — sat unpublished.

Two rules generalise past this one tool.

**A remote write is not verified by its own exit code.** That applies to `gh pr edit`, and equally
to the `--delete-branch` flag the charter already tells you to confirm with `git ls-remote`. Where
a write matters, read the state back through a different call than the one that wrote it.

**Do not fail on a non-zero exit code that the read-back contradicts.** `pr_body.py` exits 0 when
the write errored and the body is nonetheless correct, because a gate that cries wolf on GraphQL
noise is a gate people learn to pass with `|| true`. The verdict has to be about the live state,
in both directions.

## A note on gates that decide something irreversible

`pr_merge_gate.py` exists because a PR merged red (#3234) when a shell gate split `gh pr checks`'s
tab-separated output on whitespace and read the second word of `Windows MinGW` as a status — a
reading that looked entirely plausible. Its head-vs-tip rule is a precaution rather than a scar:
#3259 originally cited #3243 as a second incident and that was **wrong on the dates**, which is
itself the lesson worth keeping — the claim was withdrawn only because a reviewer dated the commits
against `mergedAt` instead of trusting a frozen-looking PR record. When a check here decides
something that cannot be undone by rerunning it, prefer being wrong in the direction of refusing:
exit non-zero on anything unrecognised, treat an empty result as void rather than clean, and use a
distinct exit status for "could not evaluate" so it can never be confused with "evaluated, and the
answer is no".

## Verifying a prose correction: normalise whitespace before grepping

A phrase you are removing from documentation will often be **hard-wrapped across a line break**, and
`grep` is line-oriented, so it reports zero and you believe the phrase is gone. This is not
hypothetical: correcting one wrong claim across this PR's files took three review rounds, and a
*different* file survived each round for exactly this reason. `grep 'checks describe'` returned
nothing while the phrase sat in `CLAUDE.md`, split after "the head the checks".

    # what actually answers "is this phrase still anywhere?"
    for f in $(git diff --name-only origin/main...HEAD); do
        n=$(tr '\n' ' ' < "$f" | tr -s ' ' | grep -o 'the phrase' | wc -l)
        [ "$n" -gt 0 ] && echo "STILL PRESENT in $f ($n)"
    done

The same applies to any claim, figure or citation you are retracting. A single-line grep is fine for
code, where the thing you are looking for rarely wraps; it is close to useless for prose in this
repository, which wraps at about 100 columns.
