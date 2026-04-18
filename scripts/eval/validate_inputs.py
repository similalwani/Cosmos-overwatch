#!/usr/bin/env python3
"""
Pre-inference validation: seed videos, annotations, and spec JSONs.

  python3 scripts/eval/validate_inputs.py
"""
import sys
import json
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SEQ_NAMES, FIRE_NAMES, PREP_DIR, OUTPUT_FRAMES, DST_FPS

BASE = Path(__file__).parent.parent.parent
SPECS_DIR = BASE / "configs" / "specs"


def validate_videos():
    print("\n" + "=" * 60)
    print("SEED VIDEOS")
    print("=" * 60)

    issues = []
    for name in SEQ_NAMES + FIRE_NAMES:
        path = PREP_DIR / f"{name}.mp4"
        if not path.exists():
            issues.append(f"MISSING: {name}.mp4")
            continue

        probe = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(path),
        ], capture_output=True, text=True)

        try:
            n = int(probe.stdout.strip())
            mb = path.stat().st_size / 1e6
            if n == OUTPUT_FRAMES:
                print(f"  ✓  {name}  ({n} frames, {mb:.1f} MB)")
            else:
                issues.append(f"FRAME MISMATCH: {name} has {n}, expected {OUTPUT_FRAMES}")
        except ValueError:
            issues.append(f"FFPROBE FAILED: {name}")

    if issues:
        print("\n  Issues:")
        for i in issues:
            print(f"    {i}")

    return len(issues) == 0


def validate_annotations():
    print("\n" + "=" * 60)
    print("ANNOTATIONS")
    print("=" * 60)

    ann_dir = PREP_DIR / "annotations"
    issues = []

    for name in SEQ_NAMES + FIRE_NAMES:
        path = ann_dir / f"{name}.txt"
        if not path.exists():
            issues.append(f"MISSING: {name}.txt")
            continue
        lines = [l for l in path.read_text().strip().split("\n") if l]
        status = "✓" if lines else "⚠"
        print(f"  {status}  {name}  ({len(lines)} annotations)")

    if issues:
        print("\n  Issues:")
        for i in issues:
            print(f"    {i}")

    return len(issues) == 0


def validate_specs():
    print("\n" + "=" * 60)
    print("INFERENCE SPECS")
    print("=" * 60)

    spec_files = sorted(SPECS_DIR.glob("*.json"))
    issues = []

    if len(spec_files) != 48:
        issues.append(f"expected 48 specs, found {len(spec_files)}")

    for f in spec_files:
        try:
            spec = json.loads(f.read_text())
            missing = [k for k in ["prompt", "video_path", "output_dir", "guidance", "edge", "depth"] if k not in spec]
            if missing:
                issues.append(f"{f.name}: missing fields {missing}")
                continue

            vid = Path(spec["video_path"])
            if vid.is_absolute():
                issues.append(f"{f.name}: video_path should be relative")
            elif not (BASE / vid).resolve().exists():
                issues.append(f"{f.name}: video not found")

            if "seed" in spec:
                issues.append(f"{f.name}: unexpected 'seed' field")
            if spec["edge"].get("control_weight") != 0.5:
                issues.append(f"{f.name}: edge weight != 0.5")
            if spec["depth"].get("control_weight") != 0.5:
                issues.append(f"{f.name}: depth weight != 0.5")

        except json.JSONDecodeError:
            issues.append(f"{f.name}: invalid JSON")

    if not issues and len(spec_files) == 48:
        uav = sum(1 for f in spec_files if "uav" in f.name)
        fire = len(spec_files) - uav
        print(f"  ✓  48 specs valid  (VisDrone: {uav}, Fire: {fire})")
    elif issues:
        print("  Issues:")
        for i in issues:
            print(f"    {i}")

    return len(issues) == 0


def main():
    all_names = SEQ_NAMES + FIRE_NAMES
    total_gb = sum((PREP_DIR / f"{n}.mp4").stat().st_size for n in all_names if (PREP_DIR / f"{n}.mp4").exists()) / 1e9

    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    results = {
        "videos": validate_videos(),
        "annotations": validate_annotations(),
        "specs": validate_specs(),
    }

    print(f"""
\n{"=" * 60}
DEPLOYMENT READINESS
{"=" * 60}
  Videos : {len(all_names)} ({len(SEQ_NAMES)} VisDrone + {len(FIRE_NAMES)} fire), {total_gb:.2f} GB
  Format : 93 frames @ {DST_FPS}fps, 1280x720
  Specs  : 48 JSONs (36 VisDrone x 4 shifts, 12 fire x 3 shifts)
  Est.   : ~7-8 min/clip on A100-80GB, ~6-7 hr total
""")

    print("=" * 60)
    if all(results.values()):
        print("✓  ALL CHECKS PASSED - READY FOR INFERENCE")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"⚠  ISSUES IN: {', '.join(failed)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
