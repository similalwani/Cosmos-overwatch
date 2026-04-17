#!/usr/bin/env python3
"""Unified dataset preparation for seed videos.

Handles two sources:
  1. VisDrone sequences (frame sequences + ground-truth annotations)
     → Transfer2.5 MP4s + remapped frame indices
  2. Fire raw MP4s (no ground-truth)
     → Transfer2.5 MP4s + YOLOv8 auto-labels

Output: seed_videos_prepped/ with MP4s and annotations/ with VisDrone-format .txt files.

Usage:
  python dataset_prep.py                 # both sources
  python dataset_prep.py --sources visdrone fire
"""
import subprocess
import argparse
from pathlib import Path
from config import (
    SEQ_NAMES, FIRE_NAMES, SEQ_DIR, ANN_DIR, FIRE_RAW_DIR,
    PREP_DIR, OUTPUT_FRAMES, SRC_FPS, DST_FPS, DURATION_SEC
)

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def prep_visdrone_videos():
    """Convert VisDrone frame sequences to Transfer2.5-compatible MP4s (93 frames @ 16fps, 1280×720)."""
    print("\n" + "="*60)
    print("VISDRONE VIDEO PREP")
    print("="*60)

    results = []
    for name in SEQ_NAMES:
        seq_path = SEQ_DIR / name
        out_path = PREP_DIR / f"{name}.mp4"

        if not seq_path.exists():
            print(f"  MISSING  {name}")
            continue

        r = subprocess.run([
            "ffmpeg", "-y",
            "-framerate", "30",
            "-i", str(seq_path / "%07d.jpg"),
            "-t", str(DURATION_SEC),
            "-r", "16",
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            str(out_path),
        ], capture_output=True, text=True)

        if r.returncode != 0:
            print(f"  ERROR  {name}\n{r.stderr[-300:]}")
            continue

        probe = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(out_path),
        ], capture_output=True, text=True)

        frame_count = int(probe.stdout.strip()) if probe.stdout.strip().isdigit() else -1
        size_mb = out_path.stat().st_size / 1_000_000
        results.append(size_mb)
        status = "✓" if frame_count == OUTPUT_FRAMES else f"⚠ {frame_count} frames (expected {OUTPUT_FRAMES})"
        print(f"  {status}  {name}.mp4  [{size_mb:.1f} MB]")

    total_mb = sum(results)
    print(f"\n{len(results)}/{len(SEQ_NAMES)} VisDrone videos ready - {total_mb:.1f} MB total")


def propagate_visdrone_annotations():
    """Remap VisDrone annotation frame indices from 30fps source to 16fps output."""
    print("\n" + "="*60)
    print("VISDRONE ANNOTATION PROPAGATION")
    print("="*60)

    OUT_DIR = PREP_DIR / "annotations"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def src_frame(n: int) -> int:
        """Map 1-indexed output frame n → 1-indexed original frame at 30fps."""
        return round((n - 1) * SRC_FPS / DST_FPS) + 1

    for name in SEQ_NAMES:
        ann_file = ANN_DIR / f"{name}.txt"
        if not ann_file.exists():
            print(f"  MISSING annotation  {name}")
            continue

        frame_map: dict[int, list[list[str]]] = {}
        for line in ann_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_map.setdefault(int(parts[0]), []).append(parts)

        out_lines: list[str] = []
        empty_frames = 0

        for n in range(1, OUTPUT_FRAMES + 1):
            rows = frame_map.get(src_frame(n), [])
            if not rows:
                empty_frames += 1
            for parts in rows:
                out_lines.append(",".join([str(n)] + parts[1:]))

        (OUT_DIR / f"{name}.txt").write_text("\n".join(out_lines) + "\n")

        total = len(out_lines)
        print(f"  ✓  {name}  →  {total} annotations across {OUTPUT_FRAMES} frames"
              + (f"  ({empty_frames} empty frames)" if empty_frames else ""))

    print(f"\nAnnotations written to {OUT_DIR}/")


def prep_fire_videos():
    """Convert fire raw MP4s to Transfer2.5-compatible MP4s (93 frames @ 16fps, 1280×720)."""
    print("\n" + "="*60)
    print("FIRE VIDEO PREP")
    print("="*60)

    results = []
    for name in FIRE_NAMES:
        in_path = FIRE_RAW_DIR / f"{name}.mp4"
        out_path = PREP_DIR / f"{name}.mp4"

        if not in_path.exists():
            print(f"  MISSING  {name}.mp4")
            continue

        r = subprocess.run([
            "ffmpeg", "-y",
            "-i", str(in_path),
            "-t", str(DURATION_SEC),
            "-r", "16",
            "-vf", "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-an",
            str(out_path),
        ], capture_output=True, text=True)

        if r.returncode != 0:
            print(f"  ERROR  {name}\n{r.stderr[-300:]}")
            continue

        probe = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", str(out_path),
        ], capture_output=True, text=True)

        frame_count = int(probe.stdout.strip()) if probe.stdout.strip().isdigit() else -1
        size_mb = out_path.stat().st_size / 1_000_000
        results.append(size_mb)
        status = "✓" if frame_count == OUTPUT_FRAMES else f"⚠ {frame_count} frames (expected {OUTPUT_FRAMES})"
        print(f"  {status}  {name}.mp4  [{size_mb:.1f} MB]")

    total_mb = sum(results)
    print(f"\n{len(results)}/{len(FIRE_NAMES)} fire videos ready - {total_mb:.1f} MB total")


def annotate_fire_videos():
    """Auto-label fire videos with YOLOv8 and write VisDrone-format annotation files."""
    print("\n" + "="*60)
    print("FIRE VIDEO AUTO-ANNOTATION (YOLOv8)")
    print("="*60)

    if YOLO is None:
        print("  ERROR: ultralytics not installed. Run: pip install ultralytics")
        return

    # COCO class IDs → VisDrone category IDs
    coco_to_visdrone = {
        0: 1,   # person → pedestrian
        1: 3,   # bicycle → bicycle
        2: 4,   # car → car
        3: 10,  # motorcycle → motor
        5: 9,   # bus → bus
        7: 6,   # truck → truck
    }

    OUT_DIR = PREP_DIR / "annotations"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")  # downloads on first run (~6MB)

    for name in FIRE_NAMES:
        vid_path = PREP_DIR / f"{name}.mp4"
        if not vid_path.exists():
            print(f"  MISSING  {name}.mp4 (run video prep first)")
            continue

        print(f"  → {name}  ", end="", flush=True)

        import cv2
        cap = cv2.VideoCapture(str(vid_path))
        frame_num = 0
        out_lines = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1

            results = model(frame, verbose=False)

            track_id = 1
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if class_id not in coco_to_visdrone:
                        continue
                    visdrone_cat = coco_to_visdrone[class_id]
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    # VisDrone format: frame,track_id,x,y,w,h,score,category,truncation,occlusion
                    out_lines.append(f"{frame_num},{track_id},{x1},{y1},{bbox_w},{bbox_h},{conf:.2f},{visdrone_cat},0,0")
                    track_id += 1

        cap.release()

        # Write annotation file
        (OUT_DIR / f"{name}.txt").write_text("\n".join(out_lines) + "\n" if out_lines else "")
        print(f"✓  {len(out_lines)} detections across {frame_num} frames")

    print(f"\nAnnotations written to {OUT_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Unified dataset prep")
    parser.add_argument(
        "--sources",
        choices=["visdrone", "fire", "all"],
        default="all",
        help="Which sources to process"
    )
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
