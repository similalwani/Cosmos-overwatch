#!/usr/bin/env python3
"""Pre-Lambda validation: check all inputs, specs, and readiness.
Usage: python3 scripts/validate_lambda.py
"""
import json
import subprocess
from pathlib import Path
from config import SEQ_NAMES, FIRE_NAMES, PREP_DIR, OUTPUT_FRAMES, DST_FPS, DURATION_SEC

BASE = Path(__file__).parent.parent
SPECS_DIR = BASE / "configs" / "specs"


def validate_videos():
    """Verify all seed videos exist and contain exactly OUTPUT_FRAMES frames."""
    print("\n" + "="*60)
    print("VALIDATE SEED VIDEOS")
    print("="*60)

    all_names = SEQ_NAMES + FIRE_NAMES
    issues = []

    for name in all_names:
        vid_path = PREP_DIR / f"{name}.mp4"

        if not vid_path.exists():
            issues.append(f"MISSING: {name}.mp4")
            continue

        probe = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(vid_path),
        ], capture_output=True, text=True)

        try:
            frame_count = int(probe.stdout.strip())
            if frame_count == OUTPUT_FRAMES:
                size_mb = vid_path.stat().st_size / 1_000_000
                print(f"  ✓  {name}  ({frame_count} frames, {size_mb:.1f} MB)")
            else:
                issues.append(f"FRAME MISMATCH: {name} has {frame_count}, expected {OUTPUT_FRAMES}")
        except ValueError:
            issues.append(f"FFPROBE FAILED: {name}")

    if issues:
        print("\n⚠  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def validate_annotations():
    """Check all annotation files exist."""
    print("\n" + "="*60)
    print("VALIDATE ANNOTATIONS")
    print("="*60)

    all_names = SEQ_NAMES + FIRE_NAMES
    ann_dir = PREP_DIR / "annotations"
    issues = []

    for name in all_names:
        ann_file = ann_dir / f"{name}.txt"

        if not ann_file.exists():
            issues.append(f"MISSING: {name}.txt")
            continue

        try:
            lines = ann_file.read_text().strip().split('\n')
            line_count = len([l for l in lines if l])
            status = "✓" if line_count > 0 else "⚠"
            print(f"  {status}  {name}  ({line_count} annotations)")
        except Exception as e:
            issues.append(f"READ FAILED: {name}.txt - {e}")

    if issues:
        print("\n⚠  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def validate_specs():
    """Validate all 48 spec JSONs: required fields, relative paths, control weights."""
    print("\n" + "="*60)
    print("VALIDATE INFERENCE SPECS")
    print("="*60)

    spec_files = sorted(SPECS_DIR.glob("*.json"))
    issues = []

    if len(spec_files) != 48:
        issues.append(f"SPEC COUNT MISMATCH: found {len(spec_files)}, expected 48")

    for spec_file in spec_files:
        try:
            spec = json.loads(spec_file.read_text())

            required = ["prompt", "video_path", "output_dir", "guidance", "edge", "depth"]
            missing = [f for f in required if f not in spec]
            if missing:
                issues.append(f"SPEC MISSING FIELDS: {spec_file.name} - {missing}")
                continue

            # video_path must be relative so specs are portable between Mac and Lambda
            vid_path_raw = Path(spec["video_path"])
            if vid_path_raw.is_absolute():
                issues.append(f"SPEC PATH SHOULD BE RELATIVE: {spec_file.name}")
            else:
                resolved = (BASE / vid_path_raw).resolve()
                if not resolved.exists():
                    issues.append(f"SPEC VIDEO NOT FOUND: {spec_file.name} → {resolved}")

            if "seed" in spec:
                issues.append(f"SPEC HAS 'seed' (not in Cosmos schema): {spec_file.name}")

            if spec["edge"].get("control_weight") != 0.5:
                issues.append(f"EDGE WEIGHT WRONG: {spec_file.name}")
            if spec["depth"].get("control_weight") != 0.5:
                issues.append(f"DEPTH WEIGHT WRONG: {spec_file.name}")

        except json.JSONDecodeError as e:
            issues.append(f"INVALID JSON: {spec_file.name}")
        except Exception as e:
            issues.append(f"SPEC ERROR: {spec_file.name}")

    if len(spec_files) == 48 and not issues:
        print(f"  ✓  All 48 specs valid")
        visdrone = len([f for f in spec_files if 'uav' in f.name])
        fire = len([f for f in spec_files if 'fire' in f.name or 'wildfire' in f.name])
        print(f"     VisDrone: {visdrone} | Fire: {fire}")

    if issues:
        print("\n⚠  ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")

    return len(issues) == 0


def print_summary():
    """Deployment readiness summary."""
    print("\n" + "="*60)
    print("LAMBDA DEPLOYMENT READINESS")
    print("="*60)

    all_names = SEQ_NAMES + FIRE_NAMES
    total_size = sum((PREP_DIR / f"{name}.mp4").stat().st_size for name in all_names) / 1e9

    print(f"""
Input data:
  - Videos: {len(all_names)} files ({len(SEQ_NAMES)} VisDrone + {len(FIRE_NAMES)} Fire)
  - Size: {total_size:.2f} GB
  - Format: 93 frames @ {DST_FPS}fps, 1280×720

Specs:
  - Total: 48 JSON files
  - VisDrone: 36 (9×4 shifts)
  - Fire: 12 (4×3 shifts)

Estimated Lambda runtime:
  - Single spec: ~2-3 min on A100-80GB
  - All 48: ~1.6-2.4 hours sequential

Next: inference_runner.py will process all specs and generate outputs
""")


def main():
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)

    results = {
        "videos": validate_videos(),
        "annotations": validate_annotations(),
        "specs": validate_specs(),
    }

    print_summary()

    print("\n" + "="*60)
    if all(results.values()):
        print("✓  ALL CHECKS PASSED — READY FOR LAMBDA")
    else:
        print("⚠  ISSUES DETECTED")
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
    print("="*60 + "\n")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit(main())
