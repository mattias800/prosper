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

`pr_merge_gate.py` exists because two PRs merged wrongly on one day (#3259), and neither cause was
carelessness — a check name containing a space was read as a status, and CI reported green for a
commit the branch had moved past. Both readings looked plausible. When a check here decides
something that cannot be undone by rerunning it, prefer being wrong in the direction of refusing:
exit non-zero on anything unrecognised, treat an empty result as void rather than clean, and use a
distinct exit status for "could not evaluate" so it can never be confused with "evaluated, and the
answer is no".
