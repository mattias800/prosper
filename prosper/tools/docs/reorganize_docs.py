#!/usr/bin/env python3
"""reorganize_docs.py -- one-shot: move prosper/docs/*.md|json into topic folders and rewrite citations.

Moves with `git mv`, then rewrites (1) every `docs/NAME` path citation repo-wide, (2) relative
markdown links inside moved/unmoved docs. Prose mentions of a bare filename are left alone; names
are unique so they stay greppable.
"""

import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)
DOCS = ROOT / "prosper" / "docs"

GROUPS = {
    "process": "VERIFICATION DEBUGGING_WORKFLOWS DIAGNOSTICS_ROADMAP DIAGNOSTIC_GATE_AUDIT BUG_HUNT_BACKLOG GAME_COMPAT_ORCHESTRATION SESSION_START",
    "architecture": "ARCHITECTURE ARCHITECTURE_TARGET_TREE HOST_PLATFORM_SEAM REFACTOR_PLAN_2026_09",
    "gpu": "GRAPHICS GRAPHICS_PIPELINES_AND_RTT_2026_09 GRAPHICS_PIPELINE_CACHE GPU_EXECUTOR_DESIGN GFX10_SW_4KB_S_TILING GFX10_SW_64KB_TILING GPU_RETILE_PACKED_2026_09 MIP_TAIL_LAYOUT_2026_09 AGC_IMPL_PLAN AGC_PACKET_SIZES AGC_TRACE RESOURCE_BINDING RECOMPILER_REMAINING FLAT_LOAD_DESIGN COMPUTE_BUFFER_IDLE_RESIDENCY_2026_09 COMPUTE_HOSTED_WRITEBACK_2026_09 RENDERER_ARCHITECTURE_COMPARISON_2026_09_26 RENDERER_ARCHITECTURE_GAPS_2026_09_25 RENDERER_BUFFER_RANGE_SHARING_2026_09 RENDERER_BUFFER_RESIDENCY_2026_09 SHADER_KEY_PREPARATION_2026_09 TEXTURE_SOURCE_SNAPSHOT_2026_09 RESOURCE_PREPARATION_2026_09 FP16_GPU_PREPARATION_2026_09 UNREQUESTED_COLOR_FLUSH_2026_09 VULKAN_RUNTIME NEXT_STEP_VERTEX_FETCH RENDER_LOOP",
    "performance": "PERFORMANCE_ROADMAP_HANDOFF_2026_09_23 RENDERER_PERFORMANCE_2026_07 GPU_PROFILING_EXTERNAL THREAD_WAIT_PROFILING",
    "subsystems": "AUDIO VIDEODEC2 SAVE_DATA_LAYOUT INPUT_REPLAY GUEST_INPUT_SAFETY FRONTEND_APP",
    "platforms": "PORTING WINDOWS_PORT_HANDOFF WINDOWS_RELEASE LINUX_RELEASE",
    "engines": "CROSS_ENGINE_UE4 UE4_APR_IOSTORE_BRINGUP UNITY_SHADER_FORMAT",
    "games/messenger": "FINDINGS REAL_FRAMES_FINDINGS CUTSCENE_GC_ABORT CUTSCENE_PROGRESSION CUTSCENE_RENDER DESER_STALE_CACHE_BLOCK GFXDEVICE_BRINGUP_PROBLEM MESSENGER_BLACK_RENDER",
    "games": "ASTROBOT_LINUX_HANDOFF_2026_07_19 DEAD_CELLS_PROGRESSION_MATRIX DOLL_LOADING_PROGRESSION DOLL_POSTPROCESS_HANDOFF EVERGATE_PERFORMANCE_HANDOFF_2026_07 GRIS_SONIC_COBRA_BRINGUP NEVER_BOOTED_SURVEY_2026_08 PPSA02664_BLACK_WORLD YAKUZA_JUDGMENT_BRINGUP",
}
mapping = {}
for folder, names in GROUPS.items():
    for n in names.split():
        for ext in (".md", ".json"):
            if (DOCS / (n + ext)).exists():
                mapping[n + ext] = folder
for p in DOCS.glob("*_STATUS.md"):
    mapping[p.name] = "games"
mapping["AC_BLACK_FLAG_STATUS.md"] = "games"
left = sorted(p.name for p in DOCS.iterdir() if p.is_file() and p.name not in mapping)
print("stay in docs/:", left)

dry = "--dry-run" in sys.argv
old2new = {DOCS / n: DOCS / f / n for n, f in mapping.items()}
if dry:
    sys.exit(0)
for o, n in old2new.items():
    n.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "mv", str(o), str(n)], cwd=ROOT)
new2old = {v: k for k, v in old2new.items()}

files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
path_re = re.compile(r"(?<![\w/.-])((?:prosper/)?docs/)((?:[A-Za-z0-9_-]+)\.(?:md|json))\b")
link_re = re.compile(r"(\]\()([^)#\s]+)((?:#[^)\s]*)?\))")
changed = 0
for rel in files:
    if not rel or rel.endswith((".webp", ".png", ".bmp", ".gz", ".bin")):
        continue
    f = ROOT / rel
    if not f.is_file():
        continue
    try:
        text = f.read_bytes().decode("utf-8")
    except Exception:
        continue
    orig = text

    def sub_path(m):
        n = m.group(2)
        return m.group(1) + mapping[n] + "/" + n if n in mapping else m.group(0)

    text = path_re.sub(sub_path, text)
    if f.suffix == ".md" and DOCS in f.parents:
        old_here = new2old.get(f, f)

        def sub_link(m, old_here=old_here, f=f):
            tgt = m.group(2)
            if re.match(r"[a-z]+:", tgt) or tgt.startswith("/"):
                return m.group(0)
            old_tgt = (old_here.parent / tgt).resolve()
            new_tgt = old2new.get(old_tgt, old_tgt)
            if new_tgt == old_tgt and old_here == f:
                return m.group(0)
            r = os.path.relpath(new_tgt, f.parent).replace("\\", "/")
            return m.group(1) + r + m.group(3)

        text = link_re.sub(sub_link, text)
    if text != orig:
        f.write_bytes(text.encode("utf-8"))
        changed += 1
print("moved", len(old2new), "rewrote", changed, "files")
