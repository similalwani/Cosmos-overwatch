#!/usr/bin/env python3
"""
Post-inference QC: black frames, static video, file corruption.

  python3 scripts/eval/validate_outputs.py --output-dir outputs
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

ENTROPY_THRESHOLD = 0.1
FLOW_THRESHOLD = 0.5


def entropy(frame_gray):
    hist, _ = np.histogram(frame_gray, bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) / 8


def check_black_frames(path, n=5):
    if cv2 is None:
        return False, "cv2 not available"
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    black = 0
    for idx in np.linspace(0, total - 1, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and entropy(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) < ENTROPY_THRESHOLD:
            black += 1
    cap.release()
    return black > 0, f"{black}/{n} frames black"


def check_static(path, n=10):
    if cv2 is None:
        return False, "cv2 not available"
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    flows, prev = [], None
    for idx in np.linspace(0, total - 2, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2).mean())
        prev = gray
    cap.release()
    if not flows:
        return False, "no frames sampled"
    avg = np.mean(flows)
    return avg < FLOW_THRESHOLD, f"avg_flow={avg:.3f}"


def check_corrupt(path):
    r = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_type,duration",
        "-of", "json", str(path),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        return False, "ffprobe error"
    try:
        data = json.loads(r.stdout)
        if not data.get("streams"):
            return False, "no video stream"
        return True, "ok"
    except json.JSONDecodeError:
        return False, "bad ffprobe output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    print(f"\n{'='*60}")
    print("VALIDATE GENERATED OUTPUTS")
    print(f"{'='*60}\n")

    videos = [p for p in output_path.rglob("*.mp4") if "control" not in p.name]
    if not videos:
        print(f"No output videos found in {args.output_dir}")
        return

    ok_count = 0
    for video in sorted(videos):
        label = video.relative_to(output_path).as_posix()
        issues = []

        valid, msg = check_corrupt(video)
        if not valid:
            print(f"  ✗  {label}  ({msg})")
            continue

        has_black, msg = check_black_frames(video)
        if has_black:
            issues.append(msg)

        is_static, msg = check_static(video)
        if is_static:
            issues.append(msg)

        if issues:
            print(f"  ⚠  {label}")
            for issue in issues:
                print(f"       {issue}")
        else:
            print(f"  ✓  {label}")
            ok_count += 1

    print(f"\n{ok_count}/{len(videos)} passed\n")


if __name__ == "__main__":
    main()
