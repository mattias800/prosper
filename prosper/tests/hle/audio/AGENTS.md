# Guest audio contract tests

These fixtures call the real HLE dispatch boundary with owned guest buffers and a recording sink.
They check handle generations, port/context lifetimes, publication failures, and independent PCM
ownership through scratch-buffer reuse. SDL device scheduling and callback synchronization belong
to `frontends/audio_sdl3/` tests; a recording sink cannot establish those behaviors.

The lifecycle log verifier runs separate enabled and disabled processes and checks teardown
observations against actual delivered PCM. Missing observations are a failed measurement, not zero
activity. Keep guest handles and sink generations distinct when checking recycled slots.
