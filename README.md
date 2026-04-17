# Overwatch: Synthetic Adverse-Condition Data for Drone Perception

Overwatch is a synthetic data generation pipeline that domain-shifts aerial drone footage into adverse visual conditions - rain, fog, thermal, fire/night - using [NVIDIA Cosmos Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5). The outputs are MOT17-compatible video datasets for training drone-based perception models on conditions underrepresented in real-world collections.

---

## Overview

| | |
|---|---|
| **Model** | Cosmos-Transfer2.5-2B |
| **Parameters** | 2.36B |
| **Control** | Edge + Depth (dual-controlnet, weight 0.5 each) |
| **Resolution** | 1280×720 (720p) |
| **Clip length** | 93 frames @ 16 fps (5.8 s) |
| **Guidance** | CFG = 5 |
| **Inference steps** | 35 |
| **GPU** | NVIDIA A100-SXM4 80GB (65.4 GB VRAM required) |
| **Inference time** | ~7-8 min / clip on A100-80GB |

---

## Dataset

**Seed videos (13 total):**

| Source | Count | Scene type |
|--------|-------|-----------|
| [VisDrone-VID](http://aiskyeye.org/) train set | 9 | Urban aerial traffic (intersection, roundabout, expressway, residential) |
| [Pexels](https://www.pexels.com/) | 4 | Aerial fire and smoke scenes |

**Domain shifts:**

| Condition | Applied to | Total clips |
|-----------|-----------|-------------|
| Heavy rain | VisDrone + Fire | 13 |
| Dense fog | VisDrone | 9 |
| Dusk / golden hour | VisDrone | 9 |
| Thermal (LWIR) | VisDrone + Fire | 13 |
| Night fire | Fire | 4 |
| **Total** | | **48** |

**Annotations:** 47,683 ground-truth bounding boxes (VisDrone format) remapped from 30 fps source to 16 fps output. Fire clips annotated with YOLOv8 auto-labels.

---

## Model Performance

Cosmos-Transfer2.5-2B official inference times (single GPU, segmentation-control configuration):

| GPU | Inference time |
|-----|---------------|
| NVIDIA B200 | 286 s |
| NVIDIA H100 NVL | 719 s |
| NVIDIA H100 PCIe | 870 s |
| NVIDIA A100-80GB | ~450 s (measured, dual-control) |
| NVIDIA H20 | 2327 s |

> Transfer2.5-2B is **3.5× smaller** than Cosmos-Transfer1 while delivering higher structural fidelity and robust long-horizon generation ([Cosmos paper](https://arxiv.org/abs/2511.00062)).

**Overwatch quality metrics** (CLIP / LPIPS / SSIM across completed clips - run `quality_metrics.py` to reproduce):

| Metric | Meaning | Target |
|--------|---------|--------|
| CLIP score | Prompt-visual alignment | Higher is better |
| LPIPS | Perceptual distance from seed | Higher = more domain shift applied |
| SSIM | Structural similarity to seed | Higher = structure better preserved |

> Full per-clip results in [`outputs/quality_metrics.json`](outputs/quality_metrics.json) after running `quality_metrics.py`.

---

## Repository Structure

```
.
├── scripts/
│   ├── config.py               # Shared constants (sequences, frame rate, paths)
│   ├── dataset_prep.py         # VisDrone + fire video prep and annotation
│   ├── generate_specs.py       # Generate 48 inference spec JSONs
│   ├── validate_inputs.py      # Pre-flight validation before cloud inference
│   ├── inference_runner.py     # Inference orchestrator (invokes Cosmos CLI)
│   ├── validate_outputs.py     # Post-inference QC (black frames, static, corrupt)
│   ├── quality_metrics.py      # CLIP / LPIPS / SSIM metrics per clip
│   ├── setup_cloud.sh         # One-shot cloud instance setup
│   └── requirements.txt        # Python deps for this repo
├── seed_videos_prepped/
│   ├── *.mp4                   # 13 seed videos (gitignored)
│   └── annotations/*.txt       # VisDrone-format ground truth
├── configs/specs/              # 48 inference spec JSONs
├── outputs/                    # Generated videos (gitignored)
│   └── quality_metrics.json    # Metrics output
├── data/
│   └── sources.md              # Dataset sources, URLs, licenses
└── review.html                 # Browser-based side-by-side viewer
```

---

## Quickstart

### 1. Validate locally (Mac)

```bash
python3 scripts/validate_inputs.py
# Expected: ✓ ALL CHECKS PASSED
```

### 2. Deploy to cloud GPU

```bash
# Upload repo to /workspace/cosmos_overwatch
rsync -avz seed_videos_prepped/ configs/ scripts/ \
    root@<host>:/workspace/cosmos_overwatch/

# On the instance - install everything (~10-15 min, one-time)
export HF_TOKEN=hf_xxx    # Must have accepted NVIDIA Cosmos-Transfer2.5 license
bash scripts/setup_cloud.sh
```

### 3. Run inference

```bash
export COSMOS_DIR=/workspace/cosmos-transfer2.5

# Smoke test (single clip, ~8 min)
python3 scripts/inference_runner.py --spec-name uav0000288_00001_v_heavy_rain

# Full batch (48 clips, ~6-7 hr on A100-80GB)
python3 scripts/inference_runner.py

# Resume after interruption
python3 scripts/inference_runner.py --resume
```

### 4. Evaluate outputs

```bash
# Post-inference QC (black frames, static video, file corruption)
python3 scripts/validate_outputs.py --output-dir outputs

# Quality metrics (CLIP / LPIPS / SSIM) - run on Mac after rsync pull
pip install torch torchvision transformers lpips scikit-image opencv-python pillow
python3 scripts/quality_metrics.py
```

### 5. Review visually

Open [`review.html`](review.html) in a browser for a side-by-side seed vs. output viewer.

---

## Reproducibility

- Inference specs are stored as JSON under `configs/specs/` and committed to the repo.
- All `video_path` fields are relative to the repo root, making specs portable between local and cloud environments.
- `inference_runner.py` resolves paths at run time and logs timing + status to `outputs/inference_log.txt`.
- Checkpoint-based resume: `--resume` picks up from the last completed spec.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Cosmos inference script not found` | `export COSMOS_DIR=/workspace/cosmos-transfer2.5` |
| HuggingFace 401 / weight download fails | Accept license at [huggingface.co/nvidia/Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B), then `hf auth login` |
| `CUDA out of memory` | One spec at a time only; use `--resume` if interrupted |
| `ffprobe not found` | `sudo apt-get install ffmpeg` |
| `validate_inputs.py` reports absolute paths | Regenerate specs: `python3 scripts/generate_specs.py` |
| Flash-attn version mismatch | Inside cosmos dir: `uv sync --extra=cu128 --reinstall` |

---

## Data Sources & Licenses

| Dataset | License | Link |
|---------|---------|------|
| VisDrone-VID | Academic use | [aiskyeye.org](http://aiskyeye.org/) |
| Pexels fire clips | Free commercial use | [pexels.com](https://www.pexels.com/) |
| Cosmos-Transfer2.5-2B weights | [NVIDIA Open Model License](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) | [HuggingFace](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) |

See [`data/sources.md`](data/sources.md) for full attribution and download instructions.

---

## References

- Agarwal et al., [Cosmos: A Foundation World Model for Physical AI](https://arxiv.org/abs/2511.00062), arXiv 2025
- [Cosmos Transfer2.5 GitHub](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
- [Cosmos Transfer2.5 Docs](https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html)
- [VisDrone Dataset](http://aiskyeye.org/)
