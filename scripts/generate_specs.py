#!/usr/bin/env python3
"""Generate Cosmos Transfer2.5 inference specs (seed video × domain shift).

Output: configs/specs/<stem>_<shift>.json
Usage:  python scripts/generate_specs.py
"""
import json
from pathlib import Path
from config import SEQ_NAMES, FIRE_NAMES, PREP_DIR

SPECS_DIR = PREP_DIR.parent / "configs" / "specs"
SPECS_DIR.mkdir(parents=True, exist_ok=True)

VISDRONE_SHIFTS = {
    "heavy_rain": (
        "An aerial drone view during heavy rainfall. Dense rain streaks fall "
        "diagonally across the frame. Water accumulates on paved surfaces, "
        "creating reflective puddles that mirror ambient light. The sky is "
        "uniformly dark grey with thick cloud cover. Visibility is reduced to "
        "approximately 200 meters. All surfaces appear wet with specular "
        "highlights from diffused overcast lighting. The camera is mounted on "
        "a drone looking downward at approximately 45 degrees, capturing the "
        "scene with natural motion and slight vibration."
    ),
    "dense_fog": (
        "An aerial drone view in dense fog conditions. Atmospheric scattering "
        "reduces visibility to under 100 meters. Objects in the foreground are "
        "clearly visible but fade progressively into white-grey haze with distance. "
        "Lighting is flat and diffused with no directional shadows. Surfaces "
        "appear desaturated. Fine water droplets suspended in air create a soft "
        "volumetric effect. The camera captures the scene from above, looking "
        "down at the terrain through the fog layer."
    ),
    "dusk_golden": (
        "An aerial drone view during golden hour, approximately 20 minutes before "
        "sunset. Warm amber and orange light arrives at a low angle, casting long "
        "shadows across the ground. The sky transitions from deep blue overhead to "
        "warm orange-pink near the horizon. Surfaces facing the sun have strong "
        "warm highlights while shadowed areas show cool blue tones. The contrast "
        "ratio is high. The drone camera captures the scene with rich cinematic "
        "color depth and natural lens flare."
    ),
    "thermal": (
        "An aerial thermal infrared view of the scene. Warm objects and people glow "
        "bright white and yellow against cooler blue-grey backgrounds. Heat signatures "
        "create distinct thermal blooms. Surfaces with different thermal properties "
        "show clear contrast: asphalt and roads appear cool grey, buildings show "
        "warm edges, vegetation appears cool. The palette is monochromatic with "
        "emphasis on intensity rather than color. Heat shimmer distorts air above "
        "warm surfaces. The overall tone is desaturated with a greenish-grey cast "
        "typical of LWIR approximation."
    ),
}

FIRE_SHIFTS = {
    "night_fire": (
        "An aerial drone view at night with an active fire as the primary light source. "
        "The scene is mostly dark with deep shadows. Fire glows warm orange and red, "
        "illuminating nearby surfaces with a flickering warm cast. Smoke is backlit by "
        "the flames, creating dark silhouettes against the glow. Ambient night sky shows "
        "faint stars or glow from distant city lights. The contrast between the bright "
        "fire and surrounding darkness is extreme. The drone camera captures the dynamic "
        "lighting and slight motion of flames and smoke."
    ),
    "heavy_rain": (
        "An aerial drone view of a fire scene during heavy rainfall. Dense rain streaks "
        "fall through the scene. Water accumulates on surfaces. The fire's glow is "
        "diffused through rain, creating a hazy orange-reddish atmosphere. Smoke mixes "
        "with rain mist. Water reflections mirror the firelight. The contrast between "
        "warm fire light and cool wet surfaces creates an emergency atmosphere. Visibility "
        "is reduced by both rain and smoke."
    ),
    "thermal": (
        "An aerial thermal infrared view of a fire scene. The fire zone appears as "
        "brilliant white and yellow heat blooms, extremely intense and saturated. Smoke "
        "appears as grey haze with cooler blue tones. Surrounding terrain shows blue-grey. "
        "Objects with absorbed heat show warm signatures. The thermal palette emphasizes "
        "the extreme heat differential. Hottest zones (active fire) are pure white, "
        "cooling outward through yellow, orange, red, and then to cool blues and greys."
    ),
}


def generate_specs():
    """Generate all 48 specs: VisDrone (9×4) + Fire (4×3).

    Paths are stored relative to repo root so the same specs work on Mac
    (validation) and Lambda (inference). inference_runner.py resolves them
    at run time.
    """
    specs_created = 0

    for name in SEQ_NAMES:
        vid_path = PREP_DIR / f"{name}.mp4"
        if not vid_path.exists():
            print(f"  WARNING: missing {name}.mp4")
            continue

        rel_video = f"seed_videos_prepped/{name}.mp4"

        for shift, prompt in VISDRONE_SHIFTS.items():
            spec = {
                "prompt": prompt,
                "video_path": rel_video,
                "output_dir": f"outputs/{name}/{shift}",
                "guidance": 3,
                "edge": {"control_weight": 0.5},
                "depth": {"control_weight": 0.5},
            }
            spec_path = SPECS_DIR / f"{name}_{shift}.json"
            with open(spec_path, "w") as f:
                json.dump(spec, f, indent=2)
            specs_created += 1

    for name in FIRE_NAMES:
        vid_path = PREP_DIR / f"{name}.mp4"
        if not vid_path.exists():
            print(f"  WARNING: missing {name}.mp4")
            continue

        rel_video = f"seed_videos_prepped/{name}.mp4"

        for shift, prompt in FIRE_SHIFTS.items():
            spec = {
                "prompt": prompt,
                "video_path": rel_video,
                "output_dir": f"outputs/{name}/{shift}",
                "guidance": 3,
                "edge": {"control_weight": 0.5},
                "depth": {"control_weight": 0.5},
            }
            spec_path = SPECS_DIR / f"{name}_{shift}.json"
            with open(spec_path, "w") as f:
                json.dump(spec, f, indent=2)
            specs_created += 1

    return specs_created


def main():
    print("\n" + "="*60)
    print("GENERATE INFERENCE SPECS")
    print("="*60)

    specs_created = generate_specs()

    spec_files = sorted(SPECS_DIR.glob("*.json"))
    print(f"\n✓  Created {specs_created} specs")
    print(f"Location: {SPECS_DIR}/")
    print(f"\nSample specs:")
    for f in spec_files[:5]:
        print(f"  - {f.name}")
    if len(spec_files) > 5:
        print(f"  ... and {len(spec_files) - 5} more")


if __name__ == "__main__":
    main()
