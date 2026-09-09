# `frontends/shared/perf` — bounded interactive performance capture

Owns F8's process-sample ring, bounded detailed renderer/compute records, completed capture writer,
and shared timing gates. The app schedules samples and capture completion; the live backends supply
measurements from actual work. Resource admission, cache ownership and synchronization belong to
those backends, not this recorder. Capture fields are additive; preserve existing presentation
counter semantics and represent unavailable fresh-render counts as unavailable rather than zero.

Keep default capture overhead bounded and separate frontend resource preparation from backend
setup. Timer leaves are nested in their enclosing buckets, not extra critical-path costs. Buffer
resident comparison bytes mean requested spans, not bytes actually read by an early-exiting compare;
reused/admitted/refreshed/declined/ineligible bytes are payload observations, not cache occupancy or
physical memory accounting. Refreshes copy changed contents only while the cache solely owns the
allocation; they are neither unchanged hits nor new admissions. Ineligible bytes count shareable
unique payloads of at least 4 KiB rejected by the whole-pass gate, including explicit controls, not
only shader-proof failures. `buffer_upload_bytes` counts all actual CPU upload copy spans, including
arena/pool, transient and resident copies and work whose later admission fails. Resident validation,
admission and refresh time is separate from ordinary arena/pool copy time; transient copies remain
within creation time. Live callbacks accumulate backend-call records
across ordered graphics spans before publishing one semantic-submit record; CPU serialization tests
here cannot substitute for a real backend-to-recorder integration guard in the live test family.
