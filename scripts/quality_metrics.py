#!/usr/bin/env python3
"""Per-clip quality metrics for Cosmos Transfer2.5 outputs.

Computes three metrics for each (seed, output) pair:
  - CLIP score     : prompt-visual alignment (higher = output matches the condition prompt)
  - LPIPS          : perceptual distance from seed (higher = more domain shift applied)
  - SSIM           : structural similarity to seed (higher = structure better preserved)

Results are written to outputs/quality_metrics.json and printed as a table.

Usage:
  pip install torch torchvision transformers lpips scikit-image opencv-python
  python3 scripts/quality_metrics.py
  python3 scripts/quality_metrics.py --output-dir outputs --specs-dir configs/specs
"""
import json
import argparse
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
OUTPUTS_DIR = BASE / "outputs"
SPECS_DIR = BASE / "configs" / "specs"

# Sample this many frames per video for all metrics (balances speed vs accuracy)
SAMPLE_FRAMES = 8


def load_frames(video_path, n=SAMPLE_FRAMES):
    """Extract n evenly-spaced frames from a video. Returns list of HxWx3 uint8 arrays."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def compute_clip_score(frames, prompt, model, processor, device):
    """Mean CLIP cosine similarity between sampled frames and the condition prompt."""
    import torch
    from PIL import Image
    texts = processor(text=[prompt], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        text_out = model.get_text_features(**texts)
        text_features = text_out if isinstance(text_out, torch.Tensor) else text_out.pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = []
    for frame in frames:
        img = Image.fromarray(frame)
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_out = model.get_image_features(**inputs)
            img_features = img_out if isinstance(img_out, torch.Tensor) else img_out.pooler_output
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        scores.append((img_features @ text_features.T).item())
    return float(np.mean(scores))


def compute_lpips(seed_frames, out_frames, lpips_fn, device):
    """Mean LPIPS perceptual distance between seed and output frames."""
    import torch

    def to_tensor(frame):
        t = torch.from_numpy(frame).float().permute(2, 0, 1) / 127.5 - 1.0
        return t.unsqueeze(0).to(device)

    scores = []
    for s, o in zip(seed_frames, out_frames):
        with torch.no_grad():
            d = lpips_fn(to_tensor(s), to_tensor(o))
        scores.append(d.item())
    return float(np.mean(scores))


def compute_ssim(seed_frames, out_frames):
    """Mean SSIM between seed and output frames."""
    from skimage.metrics import structural_similarity as ssim
    import cv2
    scores = []
    for s, o in zip(seed_frames, out_frames):
        # Resize output to seed resolution if they differ
        if s.shape != o.shape:
            o = cv2.resize(o, (s.shape[1], s.shape[0]))
        score = ssim(s, o, channel_axis=2, data_range=255)
        scores.append(score)
    return float(np.mean(scores))


def load_models(device):
    """Load CLIP and LPIPS models."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    from transformers import CLIPModel, CLIPProcessor
    import lpips

    print("Loading CLIP (openai/clip-vit-base-patch32)...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    print("Loading LPIPS (AlexNet)...")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    return clip_model, clip_processor, lpips_fn


def get_spec_map(specs_dir):
    """Return {spec_name: spec_dict} for all specs."""
    spec_map = {}
    for f in Path(specs_dir).glob("*.json"):
        spec = json.loads(f.read_text())
        spec_map[f.stem] = spec
    return spec_map


def find_output_video(output_dir, video_stem):
    """Find the main generated MP4 (not control videos) in an output directory."""
    candidates = [
        p for p in Path(output_dir).glob("*.mp4")
        if "control" not in p.name
    ]
    if not candidates:
        return None
    # Prefer exact stem match
    for c in candidates:
        if c.stem == video_stem:
            return c
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description="Compute CLIP / LPIPS / SSIM for Cosmos outputs")
    parser.add_argument("--output-dir", default=str(OUTPUTS_DIR))
    parser.add_argument("--specs-dir", default=str(SPECS_DIR))
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    try:
        import torch
        import lpips  # noqa: F401
        from transformers import CLIPModel  # noqa: F401
        from skimage.metrics import structural_similarity  # noqa: F401
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install torch torchvision transformers lpips scikit-image opencv-python pillow")
        return 1

    device = args.device
    clip_model, clip_processor, lpips_fn = load_models(device)
    spec_map = get_spec_map(args.specs_dir)
    output_root = Path(args.output_dir)

    results = []
    seed_cache = {}  # avoid re-loading the same seed video

    print(f"\n{'='*70}")
    print(f"{'SPEC':<45} {'CLIP':>6} {'LPIPS':>6} {'SSIM':>6}")
    print(f"{'='*70}")

    for spec_name, spec in sorted(spec_map.items()):
        seed_path = (BASE / spec["video_path"]).resolve()
        out_subdir = output_root / Path(spec["output_dir"]).relative_to("outputs") \
            if spec["output_dir"].startswith("outputs") \
            else Path(spec["output_dir"])

        video_stem = Path(spec["video_path"]).stem
        out_video = find_output_video(out_subdir, video_stem)

        if not out_video or not out_video.exists():
            print(f"  {'MISSING':<43} {spec_name}")
            continue
        if not seed_path.exists():
            print(f"  {'NO SEED':<43} {spec_name}")
            continue

        if str(seed_path) not in seed_cache:
            seed_cache[str(seed_path)] = load_frames(seed_path)
        seed_frames = seed_cache[str(seed_path)]
        out_frames = load_frames(out_video)

        if not seed_frames or not out_frames:
            print(f"  {'FRAME LOAD FAILED':<43} {spec_name}")
            continue

        n = min(len(seed_frames), len(out_frames))
        seed_frames = seed_frames[:n]
        out_frames = out_frames[:n]

        clip_score = compute_clip_score(out_frames, spec["prompt"], clip_model, clip_processor, device)
        lpips_score = compute_lpips(seed_frames, out_frames, lpips_fn, device)
        ssim_score = compute_ssim(seed_frames, out_frames)

        results.append({
            "spec": spec_name,
            "seed": str(seed_path),
            "output": str(out_video),
            "clip_score": round(clip_score, 4),
            "lpips": round(lpips_score, 4),
            "ssim": round(ssim_score, 4),
        })

        print(f"  {spec_name:<43} {clip_score:>6.3f} {lpips_score:>6.3f} {ssim_score:>6.3f}")

    if not results:
        print("No outputs found.")
        return 1

    # Summary
    clips = [r["clip_score"] for r in results]
    lpipss = [r["lpips"] for r in results]
    ssims = [r["ssim"] for r in results]

    print(f"{'='*70}")
    print(f"  {'MEAN':<43} {np.mean(clips):>6.3f} {np.mean(lpipss):>6.3f} {np.mean(ssims):>6.3f}")
    print(f"  {'STD':<43} {np.std(clips):>6.3f} {np.std(lpipss):>6.3f} {np.std(ssims):>6.3f}")
    print(f"{'='*70}\n")

    out_json = output_root / "quality_metrics.json"
    summary = {
        "n_clips": len(results),
        "mean": {"clip_score": round(float(np.mean(clips)), 4),
                 "lpips": round(float(np.mean(lpipss)), 4),
                 "ssim": round(float(np.mean(ssims)), 4)},
        "std":  {"clip_score": round(float(np.std(clips)), 4),
                 "lpips": round(float(np.std(lpipss)), 4),
                 "ssim": round(float(np.std(ssims)), 4)},
        "per_clip": results,
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Results saved to {out_json}")
    return 0


if __name__ == "__main__":
    exit(main())
