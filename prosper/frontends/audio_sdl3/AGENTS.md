# SDL audio frontend

This optional frontend adapts the core AudioSink interface to one SDL stream per output.
Keep SDL device ownership, stream lifetime, host pacing and playback diagnostics here;
guest ABI decoding and independently owned output/channel PCM belong in the HLE.
Diagnostics must distinguish queued input, SDL consumption demand and backend XRUNs.
Callbacks run under SDL's stream lock: never take the sink mutex or log from them.
The dummy-device tests cover lifecycle; unbound stream tests can verify consumption without
depending on a physical device or scheduler timing.
