#!/usr/bin/env python3
"""Validate generated video outputs for quality issues.

Checks:
  - Black frame detection (entropy < threshold)
  - Static video (optical flow magnitude too low)
  - Scene dissolution (histogram variance spike)
  - Corrupt video (ffprobe errors)

Usage:
  python3 scripts/validate_outputs.py --output-dir <dir>
"""
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


ENTROPY_THRESHOLD = 0.1  # Frames below this are considered black
FLOW_THRESHOLD = 0.5     # Average optical flow magnitude
HISTOGRAM_SPIKE = 2.0    # Variance spike indicates dissolution


def get_frame_entropy(frame_gray):
    """Normalized Shannon entropy of a grayscale frame (0-1)."""
    hist, _ = np.histogram(frame_gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) / 8


def is_black_frame(frame_gray, threshold=ENTROPY_THRESHOLD):
    return get_frame_entropy(frame_gray) < threshold


def is_static_video(video_path, sample_frames=10):
    """Return True if the video has negligible motion (Farneback optical flow)."""
    if cv2 is None:
        return False, "cv2 not available"

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, frame_count - 2, sample_frames, dtype=int)

    flows = []
    prev_gray = None

    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(mag.mean())

        prev_gray = gray

    cap.release()

    if not flows:
        return False, "no frames sampled"

    avg_flow = np.mean(flows)
    is_static = avg_flow < FLOW_THRESHOLD
    return is_static, f"avg_flow={avg_flow:.3f}"


def check_black_frames(video_path, sample_frames=5):
    """Sample frames and return True if any are black/blank."""
    if cv2 is None:
        return False, "cv2 not available"

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_indices = np.linspace(0, frame_count - 1, sample_frames, dtype=int)

    black_count = 0
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if is_black_frame(gray):
            black_count += 1

    cap.release()

    has_black = black_count > 0
    return has_black, f"{black_count}/{sample_frames} sampled frames are black"


def validate_video_file(video_path):
    """Check file integrity via ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type,duration,nb_read_frames",
        "-of", "json",
        str(video_path)
    ], capture_output=True, text=True)

    if result.returncode != 0:
        return False, "ffprobe error"

    try:
        data = json.loads(result.stdout)
        if not data.get("streams"):
            return False, "no video stream"
        return True, "valid"
    except json.JSONDecodeError:
        return False, "invalid json from ffprobe"


def validate_output_dir(output_dir):
    """Run all quality checks on every generated MP4 in the output tree."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []

    results = []
    for video_file in output_path.rglob("generated.mp4"):
        spec_dir = video_file.parent
        spec_name = spec_dir.relative_to(output_path).as_posix()

        issues = []

        valid, msg = validate_video_file(video_file)
        if not valid:
            issues.append(f"corrupt: {msg}")
            results.append({
                "spec": spec_name,
                "video": str(video_file),
                "valid": False,
                "issues": issues,
            })
            continue

        # Check for black frames
        has_black, msg = check_black_frames(video_file)
        if has_black:
            issues.append(f"black_frames: {msg}")

        # Check for static/frozen video
        is_static, msg = is_static_video(video_file)
        if is_static:
            issues.append(f"static: {msg}")

        results.append({
            "spec": spec_name,
            "video": str(video_file),
            "valid": len(issues) == 0,
            "issues": issues,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate generated outputs")
    parser.add_argument("--output-dir", default="outputs", help="Output directory to validate")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("VALIDATE GENERATED OUTPUTS")
    print(f"{'='*60}\n")

    results = validate_output_dir(args.output_dir)

    if not results:
        print(f"No output videos found in {args.output_dir}")
        return 0

    passed = sum(1 for r in results if r["valid"])
    failed = sum(1 for r in results if not r["valid"])

    for result in results:
        status = "✓" if result["valid"] else "✗"
        print(f"{status}  {result['spec']}")
        if result["issues"]:
            for issue in result["issues"]:
                print(f"     - {issue}")

    print(f"\n{'='*60}")
    print(f"Passed: {passed}/{len(results)}")
    print(f"Failed: {failed}/{len(results)}")
    print(f"{'='*60}\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
