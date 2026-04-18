#!/usr/bin/env python3
"""
Cosmos Transfer2.5 inference runner.

  python3 scripts/inference/inference_runner.py                        # all specs
  python3 scripts/inference/inference_runner.py --spec-name <name>     # single spec
  python3 scripts/inference/inference_runner.py --resume               # resume from checkpoint
  python3 scripts/inference/inference_runner.py --list                 # list specs
  python3 scripts/inference/inference_runner.py --dry-run              # print commands only

Requires:
  export COSMOS_DIR=/workspace/cosmos-transfer2.5
  uv sync --extra=cu128  (run inside COSMOS_DIR)
  hf auth login          (nvidia/Cosmos-Transfer2.5-2B license accepted)
"""
import json
import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent.parent.resolve()
SPECS_DIR = BASE / "configs" / "specs"
OUTPUTS_DIR = BASE / "outputs"
LOG_FILE = OUTPUTS_DIR / "inference_log.txt"
CHECKPOINT_FILE = OUTPUTS_DIR / ".inference_checkpoint"

COSMOS_DIR = Path(os.environ.get("COSMOS_DIR", "/workspace/cosmos-transfer2.5"))
COSMOS_INFERENCE = COSMOS_DIR / "examples" / "inference.py"


def get_specs():
    return {f.stem: f for f in sorted(SPECS_DIR.glob("*.json"))}


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for key in ["prompt", "video_path", "output_dir", "guidance", "edge", "depth"]:
        if key not in spec:
            raise ValueError(f"missing key in spec: {key}")
    return spec


def resolve_spec(spec):
    vid = Path(spec["video_path"])
    if not vid.is_absolute():
        vid = (BASE / vid).resolve()
    if not vid.exists():
        raise FileNotFoundError(f"video not found: {vid}")

    out = Path(spec["output_dir"])
    if not out.is_absolute():
        out = (BASE / out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    return {**spec, "video_path": str(vid), "output_dir": str(out)}, out


def run_inference(spec, output_dir, dry_run=False):
    if not dry_run and not COSMOS_INFERENCE.exists():
        raise FileNotFoundError(
            f"Cosmos not found at {COSMOS_INFERENCE} - set $COSMOS_DIR or re-run setup_cloud.sh"
        )

    cosmos_json = {
        "prompt": spec["prompt"],
        "video_path": spec["video_path"],
        "guidance": spec.get("guidance", 3),
        "name": Path(spec["video_path"]).stem,
    }
    for key in ("edge", "depth"):
        if key in spec:
            cosmos_json[key] = spec[key]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, dir=output_dir) as tf:
        json.dump(cosmos_json, tf, indent=2)
        tmp = tf.name

    venv_python = COSMOS_DIR / ".venv" / "bin" / "python"
    cmd = [str(venv_python), str(COSMOS_INFERENCE), "-i", tmp, "-o", str(output_dir), "--guidance", "5"]
    print(f"  CMD: {' '.join(cmd)}")

    if dry_run:
        return {"status": "dry_run"}

    t0 = datetime.now()
    result = subprocess.run(cmd, cwd=str(COSMOS_DIR))
    elapsed = (datetime.now() - t0).total_seconds()

    if result.returncode != 0:
        return {"status": "failed", "returncode": result.returncode, "elapsed_s": elapsed}
    return {"status": "ok", "elapsed_s": elapsed}


def check_output(output_dir):
    mp4s = list(Path(output_dir).glob("*.mp4"))
    if not mp4s:
        return False, "no mp4 produced"
    total = sum(p.stat().st_size for p in mp4s)
    if total < 10_000:
        return False, f"output too small ({total}B)"
    return True, f"{len(mp4s)} mp4s, {total/1e6:.1f} MB"


def log(spec_name, result, ok, msg):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] {spec_name}\n")
        f.write(f"  run:    {json.dumps(result)}\n")
        f.write(f"  output: ok={ok} {msg}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-name")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    specs = get_specs()

    if args.list:
        print(f"\n{len(specs)} specs:\n")
        for name in specs:
            print(f"  {name}")
        return

    if args.spec_name:
        if args.spec_name not in specs:
            print(f"spec not found: {args.spec_name}")
            return
        to_run = [args.spec_name]
    elif args.resume and CHECKPOINT_FILE.exists():
        last = CHECKPOINT_FILE.read_text().strip()
        all_names = list(specs)
        if last in specs:
            idx = all_names.index(last)
            to_run = all_names[idx + 1:]
            print(f"Resuming after {last} ({len(to_run)} remaining)")
        else:
            to_run = all_names
    else:
        to_run = list(specs)

    print(f"\n{'='*60}")
    print("COSMOS TRANSFER2.5 INFERENCE")
    print(f"{'='*60}")
    print(f"Specs:   {len(to_run)}")
    print(f"Cosmos:  {COSMOS_DIR}")
    print(f"Output:  {OUTPUTS_DIR}")
    print(f"Dry run: {args.dry_run}\n")

    for i, name in enumerate(to_run, 1):
        print(f"\n[{i}/{len(to_run)}] {name}")
        try:
            spec = load_spec(specs[name])
            resolved, out_dir = resolve_spec(spec)
            result = run_inference(resolved, out_dir, dry_run=args.dry_run)
            ok, msg = (True, "dry_run") if args.dry_run else check_output(out_dir)
            log(name, result, ok, msg)
            CHECKPOINT_FILE.write_text(name)
            print(f"  {'✓' if ok else '✗'}  {msg}")
        except Exception as e:
            log(name, {"status": "error"}, False, str(e))
            print(f"  ✗  {e}")

    print(f"\nLog: {LOG_FILE}\n")


if __name__ == "__main__":
    main()
