#!/usr/bin/env python3
"""
Run YOLOv11 detection on all generated output clips and write VisDrone-format
annotation files alongside them.

Runs two models per frame:
  - yolo11l.pt  : standard COCO detection (people, cars, trucks, etc.)
  - yolo11-fire : fire/smoke detection (category ID 11 in output annotations)

Output: outputs/<clip>/<condition>/annotations.txt
Format: frame,track_id,x,y,w,h,score,category_id,truncation,occlusion

Usage:
  python3 scripts/eval/annotate_outputs.py
  python3 scripts/eval/annotate_outputs.py --clip uav0000288_00001_v --condition heavy_rain
"""
import sys
import argparse
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

BASE = Path(__file__).parent.parent.parent

COCO_TO_VISDRONE = {
    0:  1,   # person      -> pedestrian
    1:  3,   # bicycle     -> bicycle
    2:  4,   # car         -> car
    3:  10,  # motorcycle  -> motor
    5:  9,   # bus         -> bus
    7:  6,   # truck       -> truck
}
FIRE_CAT = 11
CONF = 0.25


def annotate_video(video_path, coco_model, fire_model):
    cap = cv2.VideoCapture(str(video_path))
    frame_num = 0
    lines = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_num += 1
        tid = 1

        for r in coco_model(frame, verbose=False):
            for box in r.boxes:
                cls, conf = int(box.cls[0]), float(box.conf[0])
                if cls not in COCO_TO_VISDRONE or conf < CONF:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                lines.append(f"{frame_num},{tid},{x1},{y1},{x2-x1},{y2-y1},{conf:.2f},{COCO_TO_VISDRONE[cls]},0,0")
                tid += 1

        for r in fire_model(frame, verbose=False):
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < CONF:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                lines.append(f"{frame_num},{tid},{x1},{y1},{x2-x1},{y2-y1},{conf:.2f},{FIRE_CAT},0,0")
                tid += 1

    cap.release()
    return lines, frame_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(BASE / "outputs"))
    parser.add_argument("--clip", help="only process this clip name")
    parser.add_argument("--condition", help="only process this condition")
    args = parser.parse_args()

    output_root = Path(args.output_dir)

    print("\n" + "=" * 60)
    print("ANNOTATE OUTPUT VIDEOS")
    print("=" * 60)

    print("Loading models...")
    coco_model = YOLO("yolo11l.pt")
    try:
        fire_model = YOLO("yolo11-fire.pt")
    except Exception:
        print("  yolo11-fire.pt not found, using coco model for fire detection")
        fire_model = coco_model

    # collect all condition-level dirs
    clip_dirs = []
    for clip_dir in sorted(output_root.iterdir()):
        if not clip_dir.is_dir() or clip_dir.name.startswith("."):
            continue
        if args.clip and clip_dir.name != args.clip:
            continue
        for cond_dir in sorted(clip_dir.iterdir()):
            if not cond_dir.is_dir():
                continue
            if args.condition and cond_dir.name != args.condition:
                continue
            clip_dirs.append(cond_dir)

    print(f"{len(clip_dirs)} clips to process\n")

    for cond_dir in clip_dirs:
        label = f"{cond_dir.parent.name}/{cond_dir.name}"
        ann_path = cond_dir / "annotations.txt"

        if ann_path.exists():
            print(f"  -  {label}  (already done)")
            continue

        videos = [p for p in cond_dir.glob("*.mp4") if "control" not in p.name]
        if not videos:
            print(f"  ✗  {label}  no video found")
            continue

        print(f"  →  {label} ...", end="", flush=True)
        lines, n_frames = annotate_video(videos[0], coco_model, fire_model)
        ann_path.write_text("\n".join(lines) + "\n" if lines else "")
        print(f"  {len(lines)} detections / {n_frames} frames")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
