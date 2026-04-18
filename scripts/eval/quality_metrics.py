#!/usr/bin/env python3
"""
CLIP / LPIPS / SSIM metrics for each (seed, output) pair.

  python3 scripts/eval/quality_metrics.py
  python3 scripts/eval/quality_metrics.py --output-dir outputs --device cuda

Results saved to outputs/quality_metrics.json.
"""
import sys
import ssl
import json
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

BASE = Path(__file__).parent.parent.parent
OUTPUTS_DIR = BASE / "outputs"
SPECS_DIR = BASE / "configs" / "specs"

SAMPLE_FRAMES = 8


def load_frames(path, n=SAMPLE_FRAMES):
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def clip_score(frames, prompt, model, processor, device):
    import torch
    from PIL import Image

    inputs = processor(text=[prompt], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        text_feat = model.get_text_features(**inputs)
        if not isinstance(text_feat, torch.Tensor):
            text_feat = text_feat.pooler_output
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    scores = []
    for frame in frames:
        inp = processor(images=Image.fromarray(frame), return_tensors="pt").to(device)
        with torch.no_grad():
            img_feat = model.get_image_features(**inp)
            if not isinstance(img_feat, torch.Tensor):
                img_feat = img_feat.pooler_output
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        scores.append((img_feat @ text_feat.T).item())
    return float(np.mean(scores))


def lpips_score(seed_frames, out_frames, lpips_fn, device):
    import torch

    def to_tensor(f):
        return (torch.from_numpy(f).float().permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(0).to(device)

    scores = []
    for s, o in zip(seed_frames, out_frames):
        with torch.no_grad():
            scores.append(lpips_fn(to_tensor(s), to_tensor(o)).item())
    return float(np.mean(scores))


def ssim_score(seed_frames, out_frames):
    import cv2
    from skimage.metrics import structural_similarity as ssim

    scores = []
    for s, o in zip(seed_frames, out_frames):
        if s.shape != o.shape:
            o = cv2.resize(o, (s.shape[1], s.shape[0]))
        scores.append(ssim(s, o, channel_axis=2, data_range=255))
    return float(np.mean(scores))


def load_models(device):
    from transformers import CLIPModel, CLIPProcessor
    import lpips

    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    return clip_model, clip_proc, lpips_fn


def find_output_video(output_dir, stem):
    candidates = [p for p in Path(output_dir).glob("*.mp4") if "control" not in p.name]
    for c in candidates:
        if c.stem == stem:
            return c
    return candidates[0] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR))
    parser.add_argument("--specs-dir", default=str(SPECS_DIR))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        import torch
        import lpips          # noqa: F401
        from transformers import CLIPModel  # noqa: F401
        from skimage.metrics import structural_similarity  # noqa: F401
    except ImportError as e:
        print(f"Missing dep: {e}")
        print("pip install torch torchvision transformers lpips scikit-image opencv-python pillow")
        return

    clip_model, clip_proc, lpips_fn = load_models(args.device)

    spec_map = {
        f.stem: json.loads(f.read_text())
        for f in Path(args.specs_dir).glob("*.json")
    }

    output_root = Path(args.output_dir)
    seed_cache = {}
    results = []

    print(f"\n{'='*70}")
    print(f"{'SPEC':<45} {'CLIP':>6} {'LPIPS':>6} {'SSIM':>6}")
    print(f"{'='*70}")

    for spec_name, spec in sorted(spec_map.items()):
        seed_path = (BASE / spec["video_path"]).resolve()

        out_subdir = (
            output_root / Path(spec["output_dir"]).relative_to("outputs")
            if spec["output_dir"].startswith("outputs")
            else Path(spec["output_dir"])
        )
        out_video = find_output_video(out_subdir, Path(spec["video_path"]).stem)

        if not out_video or not out_video.exists():
            print(f"  {'MISSING':<68} {spec_name}")
            continue
        if not seed_path.exists():
            print(f"  {'NO SEED':<68} {spec_name}")
            continue

        if str(seed_path) not in seed_cache:
            seed_cache[str(seed_path)] = load_frames(seed_path)
        seed_frames = seed_cache[str(seed_path)]
        out_frames = load_frames(out_video)

        if not seed_frames or not out_frames:
            print(f"  {'LOAD FAILED':<68} {spec_name}")
            continue

        n = min(len(seed_frames), len(out_frames))
        sf, of = seed_frames[:n], out_frames[:n]

        c = clip_score(of, spec["prompt"], clip_model, clip_proc, args.device)
        l = lpips_score(sf, of, lpips_fn, args.device)
        s = ssim_score(sf, of)

        results.append({"spec": spec_name, "clip_score": round(c, 4), "lpips": round(l, 4), "ssim": round(s, 4)})
        print(f"  {spec_name:<45} {c:>6.3f} {l:>6.3f} {s:>6.3f}")

    if not results:
        print("No outputs found.")
        return

    clips = [r["clip_score"] for r in results]
    lpipss = [r["lpips"] for r in results]
    ssims = [r["ssim"] for r in results]

    print(f"{'='*70}")
    print(f"  {'MEAN':<45} {np.mean(clips):>6.3f} {np.mean(lpipss):>6.3f} {np.mean(ssims):>6.3f}")
    print(f"  {'STD':<45} {np.std(clips):>6.3f} {np.std(lpipss):>6.3f} {np.std(ssims):>6.3f}")
    print(f"{'='*70}\n")

    summary = {
        "n_clips": len(results),
        "mean": {"clip_score": round(float(np.mean(clips)), 4), "lpips": round(float(np.mean(lpipss)), 4), "ssim": round(float(np.mean(ssims)), 4)},
        "std":  {"clip_score": round(float(np.std(clips)), 4),  "lpips": round(float(np.std(lpipss)), 4),  "ssim": round(float(np.std(ssims)), 4)},
        "per_clip": results,
    }
    out_json = output_root / "quality_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved to {out_json}")


if __name__ == "__main__":
    main()
