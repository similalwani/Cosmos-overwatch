#!/usr/bin/env python3
"""
Prepare seed videos and annotations for Cosmos Transfer2.5 inference.

  python scripts/data/dataset_prep.py                    # both sources
  python scripts/data/dataset_prep.py --sources visdrone
  python scripts/data/dataset_prep.py --sources fire
"""
import sys
import subprocess
import argparse
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    SEQ_NAMES, FIRE_NAMES, SEQ_DIR, ANN_DIR, FIRE_RAW_DIR,
    PREP_DIR, OUTPUT_FRAMES, SRC_FPS, DST_FPS, DURATION_SEC,
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

COCO_TO_VISDRONE = {
    0:  1,   # person      -> pedestrian
    1:  3,   # bicycle     -> bicycle
    2:  4,   # car         -> car
    3:  10,  # motorcycle  -> motor
    5:  9,   # bus         -> bus
    7:  6,   # truck       -> truck
}


def encode_video(input_flags, out_path):
    r = subprocess.run([
        "ffmpeg", "-y",
        *input_flags,
        "-t", str(DURATION_SEC),
        "-r", "16",
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an", str(out_path),
    ], capture_output=True, text=True)
    return r


def frame_count(path):
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(path),
    ], capture_output=True, text=True)
    s = probe.stdout.strip()
    return int(s) if s.isdigit() else -1


def prep_visdrone_videos():
    print("\n" + "=" * 60)
    print("VISDRONE VIDEO PREP")
    print("=" * 60)

    sizes = []
    for name in SEQ_NAMES:
        seq_path = SEQ_DIR / name
        out_path = PREP_DIR / f"{name}.mp4"

        if not seq_path.exists():
            print(f"  MISSING  {name}")
            continue

        r = encode_video(["-framerate", "30", "-i", str(seq_path / "%07d.jpg")], out_path)
        if r.returncode != 0:
            print(f"  ERROR  {name}\n{r.stderr[-300:]}")
            continue

        n = frame_count(out_path)
        mb = out_path.stat().st_size / 1e6
        sizes.append(mb)
        status = "✓" if n == OUTPUT_FRAMES else f"⚠ {n} frames (expected {OUTPUT_FRAMES})"
        print(f"  {status}  {name}.mp4  [{mb:.1f} MB]")

    print(f"\n{len(sizes)}/{len(SEQ_NAMES)} ready - {sum(sizes):.1f} MB total")


def propagate_visdrone_annotations():
    print("\n" + "=" * 60)
    print("VISDRONE ANNOTATION PROPAGATION")
    print("=" * 60)

    out_dir = PREP_DIR / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)

    def src_frame(n):
        return round((n - 1) * SRC_FPS / DST_FPS) + 1

    for name in SEQ_NAMES:
        ann_file = ANN_DIR / f"{name}.txt"
        if not ann_file.exists():
            print(f"  MISSING  {name}")
            continue

        frame_map = {}
        for line in ann_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_map.setdefault(int(parts[0]), []).append(parts)

        out_lines = []
        empty = 0
        for n in range(1, OUTPUT_FRAMES + 1):
            rows = frame_map.get(src_frame(n), [])
            if not rows:
                empty += 1
            for parts in rows:
                out_lines.append(",".join([str(n)] + parts[1:]))

        (out_dir / f"{name}.txt").write_text("\n".join(out_lines) + "\n")
        suffix = f"  ({empty} empty frames)" if empty else ""
        print(f"  ✓  {name}  {len(out_lines)} annotations{suffix}")

    print(f"\nWritten to {out_dir}/")


def prep_fire_videos():
    print("\n" + "=" * 60)
    print("FIRE VIDEO PREP")
    print("=" * 60)

    sizes = []
    for name in FIRE_NAMES:
        in_path = FIRE_RAW_DIR / f"{name}.mp4"
        out_path = PREP_DIR / f"{name}.mp4"

        if not in_path.exists():
            print(f"  MISSING  {name}.mp4")
            continue

        r = encode_video(["-i", str(in_path)], out_path)
        if r.returncode != 0:
            print(f"  ERROR  {name}\n{r.stderr[-300:]}")
            continue

        n = frame_count(out_path)
        mb = out_path.stat().st_size / 1e6
        sizes.append(mb)
        status = "✓" if n == OUTPUT_FRAMES else f"⚠ {n} frames (expected {OUTPUT_FRAMES})"
        print(f"  {status}  {name}.mp4  [{mb:.1f} MB]")

    print(f"\n{len(sizes)}/{len(FIRE_NAMES)} ready - {sum(sizes):.1f} MB total")


def annotate_fire_videos():
    print("\n" + "=" * 60)
    print("FIRE VIDEO AUTO-ANNOTATION (YOLOv8)")
    print("=" * 60)

    if YOLO is None:
        print("  ultralytics not installed: pip install ultralytics")
        return

    out_dir = PREP_DIR / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")

    for name in FIRE_NAMES:
        vid_path = PREP_DIR / f"{name}.mp4"
        if not vid_path.exists():
            print(f"  MISSING  {name}.mp4")
            continue

        print(f"  -> {name} ...", end="", flush=True)
        cap = cv2.VideoCapture(str(vid_path))
        frame_num, out_lines = 0, []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1
            tid = 1
            for result in model(frame, verbose=False):
                for box in result.boxes:
                    cls, conf = int(box.cls[0]), float(box.conf[0])
                    if cls not in COCO_TO_VISDRONE:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    out_lines.append(
                        f"{frame_num},{tid},{x1},{y1},{x2-x1},{y2-y1},{conf:.2f},{COCO_TO_VISDRONE[cls]},0,0"
                    )
                    tid += 1

        cap.release()
        (out_dir / f"{name}.txt").write_text("\n".join(out_lines) + "\n" if out_lines else "")
        print(f"  {len(out_lines)} detections / {frame_num} frames")

    print(f"\nWritten to {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", choices=["visdrone", "fire", "all"], default="all")
    args = parser.parse_args()

    PREP_DIR.mkdir(parents=True, exist_ok=True)

    if args.sources in ("visdrone", "all"):
        prep_visdrone_videos()
        propagate_visdrone_annotations()

    if args.sources in ("fire", "all"):
        prep_fire_videos()
        annotate_fire_videos()


if __name__ == "__main__":
    main()
