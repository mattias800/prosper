# SDL audio frontend

This optional frontend adapts the core AudioSink interface to one SDL stream per output.
Keep SDL device ownership, stream lifetime, host pacing and playback diagnostics here;
guest ABI decoding and independently owned output/channel PCM belong in the HLE.
Diagnostics must distinguish queued input, SDL consumption demand and backend XRUNs.
Callbacks run under SDL's stream lock: never take the sink mutex or log from them.
The dummy-device tests cover lifecycle; unbound stream tests can verify consumption without
depending on a physical device or scheduler timing.

`PROSPER_AUDIO_LIFECYCLE=1` adds successful submission and stream-retirement observations.
Use it with `PROSPER_AUDIO_DEMAND=1` for per-generation callback totals. Final demand snapshots
are taken after normal stream destruction has ended callbacks and before meter reuse; emit logs
after releasing the sink lock. Periodic observations cover streams still open when the frontend
takes `_Exit`. SDL callback times and HLE steady-clock times have different origins: use the
reported clock brackets when comparing them. These observations do not establish hardware XRUNs
or whether demand after a guest port retires was audible.
