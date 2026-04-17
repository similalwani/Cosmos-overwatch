#!/usr/bin/env python3
"""Cosmos Transfer2.5 inference runner.

Loads each spec, resolves paths, writes a Cosmos-compatible JSON to a temp
file, then invokes the Cosmos Transfer2.5 CLI. Logs timings and supports
checkpoint-based resume.

Usage (on Lambda, after setup_lambda.sh):
  python3 inference_runner.py                     # run all specs
  python3 inference_runner.py --spec-name <name>  # single spec
  python3 inference_runner.py --resume            # resume from checkpoint
  python3 inference_runner.py --list              # list available specs
  python3 inference_runner.py --dry-run           # print commands only

Requires:
  - cosmos-transfer2.5 cloned at $COSMOS_DIR (default: /workspace/cosmos-transfer2.5)
  - `uv sync --extra=cu128` completed inside that repo
  - HuggingFace login with access to nvidia/Cosmos-Transfer2.5-2B
"""
import json
import os
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

BASE = Path(__file__).parent.parent.resolve()
SPECS_DIR = BASE / "configs" / "specs"
OUTPUTS_DIR = BASE / "outputs"
LOG_FILE = OUTPUTS_DIR / "inference_log.txt"
CHECKPOINT_FILE = OUTPUTS_DIR / ".inference_checkpoint"

# Location of the cloned cosmos-transfer2.5 repo on Lambda
COSMOS_DIR = Path(os.environ.get("COSMOS_DIR", "/workspace/cosmos-transfer2.5"))
COSMOS_INFERENCE = COSMOS_DIR / "examples" / "inference.py"


def get_available_specs():
    specs = sorted(SPECS_DIR.glob("*.json"))
    return {f.stem: f for f in specs}


def load_spec(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)
    required = ["prompt", "video_path", "output_dir", "guidance", "edge", "depth"]
    for key in required:
        if key not in spec:
            raise ValueError(f"Missing required key: {key}")
    return spec


def resolve_spec(spec):
    """Resolve relative paths in spec to absolute paths; create output dir."""
    resolved = dict(spec)
    vid = Path(spec["video_path"])
    if not vid.is_absolute():
        vid = (BASE / vid).resolve()
    if not vid.exists():
        raise FileNotFoundError(f"Video not found: {vid}")
    resolved["video_path"] = str(vid)

    out = Path(spec["output_dir"])
    if not out.is_absolute():
        out = (BASE / out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    resolved["output_dir"] = str(out)
    return resolved, out


def run_cosmos_inference(resolved_spec, output_dir, dry_run=False):
    """Write resolved spec to temp file and invoke the Cosmos CLI."""
    if not dry_run and not COSMOS_INFERENCE.exists():
        raise FileNotFoundError(
            f"Cosmos inference script not found at {COSMOS_INFERENCE}. "
            f"Set $COSMOS_DIR or re-run setup_lambda.sh."
        )

    # Cosmos CLI accepts a subset of fields; output_dir is a CLI flag, not JSON.
    cosmos_json = {
        "prompt": resolved_spec["prompt"],
        "video_path": resolved_spec["video_path"],
        "guidance": resolved_spec.get("guidance", 3),
    }
    if "edge" in resolved_spec:
        cosmos_json["edge"] = resolved_spec["edge"]
    if "depth" in resolved_spec:
        cosmos_json["depth"] = resolved_spec["depth"]
    cosmos_json["name"] = Path(resolved_spec["video_path"]).stem

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, dir=output_dir
    ) as tf:
        json.dump(cosmos_json, tf, indent=2)
        temp_spec_path = tf.name

    venv_python = COSMOS_DIR / ".venv" / "bin" / "python"
    cmd = [
        str(venv_python), str(COSMOS_INFERENCE),
        "-i", temp_spec_path,
        "-o", str(output_dir),
        "--guidance", "5",
    ]

    print(f"  CMD: {' '.join(cmd)}")

    if dry_run:
        return {"status": "dry_run", "cmd": cmd}

    start = datetime.now()
    result = subprocess.run(cmd, cwd=str(COSMOS_DIR))
    elapsed = (datetime.now() - start).total_seconds()

    if result.returncode != 0:
        return {"status": "failed", "returncode": result.returncode, "elapsed_s": elapsed}
    return {"status": "ok", "elapsed_s": elapsed}


def validate_output(output_dir):
    """Check that at least one non-trivial MP4 was produced."""
    output_dir = Path(output_dir)
    mp4s = list(output_dir.glob("*.mp4"))
    if not mp4s:
        return {"valid": False, "reason": "no mp4 output produced"}
    total_size = sum(p.stat().st_size for p in mp4s)
    if total_size < 10_000:
        return {"valid": False, "reason": f"output too small ({total_size}B)"}
    return {
        "valid": True,
        "mp4_count": len(mp4s),
        "total_size_mb": round(total_size / 1e6, 2),
    }


def write_log(spec_name, result, validation):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] {spec_name}\n")
        f.write(f"  run:      {json.dumps(result)}\n")
        f.write(f"  validate: {json.dumps(validation)}\n")


def save_checkpoint(spec_name):
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_FILE.write_text(spec_name)


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return CHECKPOINT_FILE.read_text().strip()
    return None


def main():
    parser = argparse.ArgumentParser(description="Cosmos Transfer2.5 inference runner")
    parser.add_argument("--spec-name", help="Run single spec by name (no .json)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--list", action="store_true", help="List available specs")
    parser.add_argument("--dry-run", action="store_true", help="Print command, don't execute")
    args = parser.parse_args()

    specs = get_available_specs()

    if args.list:
        print(f"\nAvailable specs ({len(specs)}):\n")
        for name in sorted(specs.keys()):
            print(f"  {name}")
        return 0

    if args.spec_name:
        if args.spec_name not in specs:
            print(f"ERROR: spec '{args.spec_name}' not found")
            return 1
        spec_names = [args.spec_name]
    elif args.resume:
        last = load_checkpoint()
        if last and last in specs:
            idx = list(specs.keys()).index(last)
            spec_names = list(specs.keys())[idx + 1:]
            if not spec_names:
                print("No remaining specs — all done")
                return 0
            print(f"Resuming after {last} ({len(spec_names)} remaining)")
        else:
            spec_names = list(specs.keys())
            print("No checkpoint, starting from beginning")
    else:
        spec_names = list(specs.keys())

    print(f"\n{'='*60}")
    print("COSMOS TRANSFER2.5 INFERENCE RUNNER")
    print(f"{'='*60}")
    print(f"Specs to process: {len(spec_names)}")
    print(f"Cosmos dir:       {COSMOS_DIR}")
    print(f"Output directory: {OUTPUTS_DIR}")
    print(f"Log file:         {LOG_FILE}")
    print(f"Dry run:          {args.dry_run}\n")

    passed = 0
    failed = 0

    for i, spec_name in enumerate(spec_names, 1):
        print(f"\n[{i}/{len(spec_names)}] {spec_name}")
        try:
            spec = load_spec(specs[spec_name])
            resolved, out_dir = resolve_spec(spec)
            result = run_cosmos_inference(resolved, out_dir, dry_run=args.dry_run)
            validation = (
                {"valid": True, "reason": "dry_run"}
                if args.dry_run
                else validate_output(out_dir)
            )
            write_log(spec_name, result, validation)
            save_checkpoint(spec_name)

            if validation["valid"]:
                passed += 1
                print(f"  ✓ PASSED  {result}")
            else:
                failed += 1
                print(f"  ✗ FAILED  {validation['reason']}")
        except Exception as e:
            failed += 1
            result = {"status": "error", "error": str(e)}
            validation = {"valid": False, "reason": str(e)}
            write_log(spec_name, result, validation)
            print(f"  ✗ ERROR  {e}")

    print(f"\n{'='*60}")
    print(f"Passed: {passed}/{len(spec_names)}")
    print(f"Failed: {failed}/{len(spec_names)}")
    print(f"Log:    {LOG_FILE}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
