# Unrequested color-readback synchronization (2026-09-07)

Issue [#2283](https://github.com/mattias800/prosper/issues/2283) has two distinct parts. The
unused color copies were already removed. Their former readback predicate still forced a queue
submit and fence wait even when the caller explicitly declined all CPU color results.

The flush decision now uses the actual readback request. Standalone calls, explicit batch ends,
and storage-image writeback still submit and wait. The live caller's existing union of bound MRT
slots decides whether color pixels are wanted; this change does not infer that from color-0's
format, color-0's address, a black image, or a draw write mask.

Pending commands retain transient attachments, descriptors, uploads and command pools through
`BackendSubmissionBatch`. Persistent depth/stencil state is published speculatively so later
commands in the same batch can LOAD it. Existing failure callbacks revoke validity if that batch
is discarded or fails. The implementation changes neither that ownership nor the publication
contract; it allows additional passes to use it.

## Regression guards

`multidraw_render` now records a stencil producer with no color target and no requested color
pixels. It verifies zero submits/waits, then checks that a persistent stencil consumer produces
the expected green pixels with both commands behind one fence. Its discard arm first observes
the speculative cache entry, then verifies layout/depth/stencil validity is revoked and a
persistent consumer clears it instead of observing a write that never executed.

Restoring the old flush predicate fails the pending-batch and combined-submit assertions.
Removing the depth/stencil failure callback fails the explicit revocation assertion. Review
caught an initial discard test whose consumer used a transient attachment; that test could not
establish cache invalidation and was corrected before final verification.

`texture_sample_render` verifies that no-readback standalone and explicit batch-end calls still
complete, and that a storage-image atomic writes the expected guest value with color readback
explicitly disabled. Existing offscreen pixel consumers and sparse-MRT coverage remain enabled.

## Blue Prince measurement

The native Linux/RADV baseline used a fresh save, `PROSPER_IME_AUTOKEY=1`, direct graphics mode,
immediate presentation, timing and non-deferred pass logs, an F8 capture after 300 seconds, and
an F9 capture after 310 seconds, outside the performance window. No compiler or other GPU workload
ran during measurement. This is the opening cutscene, not a Day One gameplay measurement.

The retained baseline executable contains the production source subsequently merged in #3420
(`241a59342`); its embedded revision still names `e1ed278ce`, from before those working-tree
changes were committed. The private run manifest records that distinction and the binary hash.

Across 360 seconds, all 27,605 logged no-color-base passes had `raw_fmt=0`, `writes=0` and
`bytes=0`. Later timing windows averaged 9.72–10.28 readback-driven waits per callback out of
15.68–16.32 total submits, with no `no_batch` or storage-writeback contribution. The domain is
present, and the already-removed copies are not the remaining work.

The baseline F8 contains 20 pre-trigger and 21 post-trigger samples, 46 renderer records and 708
compute records with zero drops. Over the full sampled interval it measured 4.48 guest flips/s,
4.38 host presents/s and 1.82 process CPU cores. Renderer detail records cover only the five-second
post-trigger window; their 759 callbacks total 3,886.69 ms, including 218.92 ms GPU waits.
These scopes must not be mixed into a single frame-time budget.

The captured cutscene also confirms the user's separately reported recent regression: world
geometry appears in front of the FMV. The user identified recent GTA work as a possible lead;
that causal link has not been established. It is present before this flush change and is not
claimed fixed here. Review also noted a pre-existing DRAW_ISO empty-output indexing defect;
that diagnostic issue is outside the flush change.
