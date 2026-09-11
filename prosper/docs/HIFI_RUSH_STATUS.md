# Hi-Fi RUSH (`PPSA17168`) — status

Tracker: [#2891](https://github.com/mattias800/prosper/issues/2891). Engine: Unreal Engine 4, IoStore
packaging. Route: [`scripts/hifi-rush/`](../../scripts/hifi-rush/README.md).

**The recorded milestone is rung 2 — the complete 4K title screen on a default launch, 2026-08-21 on
`b033a96e`. Current `main` does not reach it.** A default launch now produces a single flat white 4K
frame and holds it for the whole run.

## READ THIS BEFORE FORMING A HYPOTHESIS ABOUT THE WHITE FRAME

The white frame is `distinct_rgb_colors == 1`, `nonblack_rgb_pixels == 8294400`, RGBA CRC32
`96a57e22` — **byte-identical to signature (a) in
[#2932](https://github.com/mattias800/prosper/issues/2932)**, the cross-title "wrong composite"
issue. On this title it is **not** a composite-selection defect, and starting at
`select_present_source` wastes the session.

What actually happens, measured 2026-09-11 on `main` `5bea42f8`:

1. The guest boots cleanly (`[bootphase] +281.1ms BOOT_COMPLETE`) and submits real GPU work.
2. Somewhere in the first ~20 s the GPU takes a **GPUVM permission fault** and RADV **hard-recovers**
   the context. The Vulkan device is lost, permanently.
3. Every subsequent graphics submission returns `VK_ERROR_DEVICE_LOST`; one 165 s run reached
   `[backend] GRAPHICS submission failed: submit=-4 … occurrence=257`.
4. With no submission able to complete, no pass produces a present source, so the present layer
   correctly declines to publish and re-serves the retained frame — which is the title's own white
   boot splash, the last thing published before the loss.

```text
radv/amdgpu: The CS has been cancelled because the context is lost. This context is guilty of a hard recovery.
radv: GPUVM fault detected at address 0x80018096f000.
GCVM_L2_PROTECTION_FAULT_STATUS: 0x601031
	 CLIENT_ID: (TCP) 0x8
	 MORE_FAULTS: 1
	 WALKER_ERROR: 0
	 PERMISSION_FAULTS: 3
	 MAPPING_ERROR: 0
	 RW: 0

[rtt] PRESENT SOURCE EXTENT MISMATCH #1: no pass produced a 3840x2160 (33177600-byte) present source
      — px_front=none px_vo=none px_last=none, offered 0 bytes; serving the retained frame instead
      (published so far: fresh=713 retained=1)
```

`fresh` stops at **713** and never increments again; `retained` climbs to 128 by the end of a 165 s
run. The manifest reports `distinct_frames: 1` for the entire run. **The ordering is device loss
first, frozen composite second**, and the present layer is behaving exactly as #1986 designed it to.

`CLIENT_ID: (TCP)` with `RW: 0` and `MAPPING_ERROR: 0` is a **vector-memory read of a page that is
mapped but not readable by that shader** — an out-of-bounds or wrongly-based read from a shader, not
a missing allocation.

## The same fault is filed on a second UE4 title

[#3340](https://github.com/mattias800/prosper/issues/3340) records *Little Nightmares III*
(`PPSA05143`) taking a field-for-field identical fault dump about two minutes into a default launch,
also losing its title screen. Only `GCVM_L2_PROTECTION_FAULT_STATUS` differs (`0x601031` here,
`0x401031` there). Whether the two share a cause is **open** — same signature is not same cause —
but they are worth treating as one investigation.

**This title settles #3340's own open question about whether the fault is new.** That issue declines
to call it a regression, on the reasonable grounds that `PPSA05143`'s title screen sat at t≈110 s
against a collapse at t≈115-145 s — a five-second window that a marginally slower boot would miss
with no change to the fault. That reading cannot cover `PPSA17168`: tracker #2891 records a title
screen **held from t≈224 s to the end of a 315 s run**, plus runs of **715 s and 1168 s** all ending
`guest=running status=ok`. A permanent device loss at t≈20 s is not compatible with any of those.

## Reproduction

```bash
mkdir -p <SCRATCH>/s0 <SCRATCH>/sm          # empty: a first boot, per instrument trap 234
SDL_VIDEODRIVER=offscreen PROSPER_VULKAN_LIB=libvulkan.so.1 \
PROSPER_RENDER=1 PROSPER_GUEST_ARGS=-force-gfx-direct PROSPER_BOOTPHASE=1 \
PROSPER_SAVE0=<SCRATCH>/s0 PROSPER_SAVEDATA_DIR=<SCRATCH>/sm \
timeout -s KILL 340 ./screenshot <DUMP_ROOT>/PPSA17168-app0 \
    --seconds 11 --count 14 --timeout 300 --out <OUT_DIR>
```

The fault lands between the first and second sample, so a ~150 s bound is ample — this title is the
cheaper of the two subjects for anyone working #3340. Keep the sampling interval **prime** (instrument
trap 211) and remember `tools/screenshot --timeout` cannot bound a boot (trap 214), so the outer
`timeout` is load-bearing.

Discriminator for any arm here, in order of strength:

```bash
grep -ac 'GPUVM fault'              run.log     # 0 = the device survived
grep -ao 'occurrence=[0-9]*'        run.log | sort -t= -k2 -n | tail -1
grep -a  'PRESENT SOURCE EXTENT MISMATCH' run.log | tail -1   # fresh= stops moving at the loss
```

Do **not** use the screenshot count or `source-distinct` as the discriminator: the retained frame is
re-served as a fresh source publication, so both keep climbing through a dead device.

## Ruled out

- **A stale save directory at the shared default location (instrument trap 234) is NOT the cause.**
  `~/.local/share/prosper/savedata0/PPSA17168/Profile/ue4savegame.dpx.sav` does exist, dated
  2026-08-21, written by the routed run that produced #2891's §1/§2 evidence — so the trap was live
  and was the leading hypothesis. A matched first-boot arm with isolated empty `PROSPER_SAVE0` /
  `PROSPER_SAVEDATA_DIR` reproduces the flat white identically: **15 of 15 samples**
  `source=composited`, `distinct_rgb_colors=1`, `nonblack_rgb_pixels=8294400`, one `distinct_frames`
  for the whole run. Save isolation makes no difference on this title. (2026-09-11, `5bea42f8`; #2891.)
- **The present layer is not choosing a wrong composite.** `px_front=none px_vo=none px_last=none`
  with `offered 0 bytes` means there was no candidate of any extent to choose between — the device is
  lost and nothing rendered. Both #2932 signatures and every lever filed against them
  (`PROSPER_CB_EFC_NO_COLOR`, `PROSPER_LEGACY_CB_DISABLE_MASK`, `PROSPER_NO_COMPUTE_RTT_MIRROR`,
  `PROSPER_RTT_NOSEED`, `PROSPER_NO_UNORM_RTT_VALUE_REUSE`, `PROSPER_NO_NORMALIZED_UNORM_RTT_BIND`)
  act on a decision this title never reaches. (2026-09-11.)

### Void, not falsified

These arms produced a null that says nothing, and are recorded so the null is not later quoted as a
negative result.

- **`PROSPER_NO_COMPUTE=1` produced zero GPUVM faults — VOID.** The arm rendered **zero** frames in
  320 s where the matched default arm produced 15 samples, so it never reached the state that faults.
  A null from an arm that cannot express the case tests the discriminator, not the domain.
- **The compute program named on the device-loss line is not an attribution.** Two runs named two
  different programs (`0x3010440000`, `0x3010360000`), and in the 2026-09-11 run **six graphics
  submissions reported the loss before the compute line did**. Instrument trap 170 and the comment at
  the reporting site in `tests/fixtures/render_runner.h` both say a loss at submit time names the
  submission that *observes* it. `PROSPER_COMPUTE_SKIP_PROGRAM` on one of them merely moved the name
  to the other (#2891).
- **Running a container-built `screenshot` on the HOST is not a cheaper arm.** Every shared library
  resolves (`ldd`: 0 not-found), and the process still wedges before printing a single line: outside
  the `ps5ys` distrobox it maps **every** Mesa ICD present on the host — `libvulkan_asahi`,
  `broadcom`, `dzn`, `freedreno`, `intel`, `intel_hasvk`, `lvp`, `nouveau`, `panfrost` — and does not
  reach `BOOT_COMPLETE`. Any measurement taken that way would also be against whichever ICD
  `select_vulkan_device` happened to pick, not RADV. Run inside the container.

## Open

1. **Which submission faults.** Unattributed on both this title and `PPSA05143`. `RADV_DEBUG=hang`
   for a hang report naming the shader is the indicated instrument and has not been run here.
2. **Which change introduced it.** The window is 2026-08-21 (`b033a96e`) to now. No bisect has been
   run; a build is ~25 minutes on this box. Note that other lanes' worktrees often hold prebuilt
   `screenshot` binaries at intermediate commits, which makes a coarse first cut free — survey with
   `git worktree list` and check each tree's `prosper/build-linux/screenshot`.
3. Everything in #2891's own blocker list (the language-wizard dialog's missing selected-label text,
   the flat-yellow post-wizard screen, the black opaque shading in the rooftop scene) is **downstream
   of this** and cannot be worked until the device survives.
