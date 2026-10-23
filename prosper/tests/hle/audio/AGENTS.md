# Guest audio contract tests

These fixtures call the real HLE dispatch boundary with owned guest buffers and a recording sink.
They check handle generations, port/context lifetimes, publication failures, and independent PCM
ownership through scratch-buffer reuse. SDL device scheduling and callback synchronization belong
to `frontends/audio_sdl3/` tests; a recording sink cannot establish those behaviors.

The lifecycle log verifier runs separate enabled and disabled processes and checks teardown
observations against actual delivered PCM. Missing observations are a failed measurement, not zero
activity. Keep guest handles and sink generations distinct when checking recycled slots.

The existing pure audio-helper fixtures retain the format, channel-volume, queue-slot, sample-count,
stamp-verdict and stereo-fold contracts declared in `audio.hpp`; their implementations live in
`src/hle/audio/audio_pcm_contracts.cpp`. They require no guest execution or host audio device.
