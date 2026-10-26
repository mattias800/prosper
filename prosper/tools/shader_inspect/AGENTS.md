# shader_inspect — offline compiler replay and corpus measurements

`shader_inspect` decodes shader bytes, recompiles synthetic inputs or recorded sidecars and
reports the emitted module and refusal. Corpus helpers preserve input identities, compare
verdicts and accepted module hashes, and group changes for review. They do not run games.

`placeholder_origin_census.py` reads a survey's recorded origin details. It counts each program
once per origin and each distinct root separately, so mixed-origin groups overlap visibly.
Missing diagnostics stay unclassified; the tool cannot turn them into unloaded words or change
the compiler's admission policy.
