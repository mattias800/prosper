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
- **`check_ctest_gate.py`** — finds callers that run `ctest` without `--no-tests=error`. Plain
  `ctest` exits 0 when it finds no tests, so "nothing ran" and "everything passed" share a status.
- **`check_usage_text.py`** — finds tools whose usage block stopped being a docstring, so
  `print(__doc__)` prints the literal `None` and the caller is told nothing.
- **`pr_merge_gate.py`** — is this PR safe to merge right now? Counts checks by bucket from
  `gh --json` and refuses when the PR's recorded head is no longer the branch tip.

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
