"""
Quick crapification for a SINGLE video (one scene/angle, ~200 frames).

Pipeline:
    clean (data/scene_xxx/images/angle)
  -> fog (depth-based)
  -> rain masks
  -> composite
  -> droplets (GPU, persistent or not)

This is a lighter, per-video version of test_crapification_pipeline.py
intended to be much faster when you only care about one sequence.

Default run (no CLI args):
    scene_004 / front-forward
    fog = medium, rain = medium, droplets = medium, persistent = True
"""
import gc
import os
import sys
import shutil
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch

# =======================
# PATH CONFIG
# =======================

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

DATA_DIR = BASE / "data"
OUTPUT_BASE = BASE / "data_after_crapification_single"
TEXTURE_DIR = BASE / "rain-rendering" / "rainstreakdb"

# Import stages
from crapification.stage_fog import run_fog_stage
from crapification.stage_rain_masks import run_rain_mask_stage
from crapification.stage_composite import run_composite_stage
from crapification.stage_droplets import run_droplet_stage
from crapification.generate_depth import generate_depth_for_scene

# =======================
# INTENSITY PARAMETERS
# =======================

FOG_PARAMS = {
    "none":   {"fog_density": 0.0,  "airlight": 255},
    "light":  {"fog_density": 0.02, "airlight": 240},
    "medium": {"fog_density": 0.06, "airlight": 230},
    "heavy":  {"fog_density": 0.12, "airlight": 220},
}

RAIN_PARAMS = {
    "none":   {"density": 0,    "min_length": 0,  "max_length": 0},
    "light":  {"density": 1000, "min_length": 8,  "max_length": 20},
    "medium": {"density": 2500, "min_length": 8,  "max_length": 35},
    "heavy":  {"density": 4000, "min_length": 15, "max_length": 50},
}


# =======================
# UTIL
# =======================

def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def ensure_depth_exists(scene_name: str, angle: str) -> bool:
    """
    Make sure depth maps exist for the given scene/angle.
    If not, call generate_depth_for_scene.
    """
    depth_dir = DATA_DIR / scene_name / "depth" / angle
    img_dir = DATA_DIR / scene_name / "images" / angle

    if not img_dir.exists():
        print(f"⚠️  Images not found: {img_dir}")
        return False

    if depth_dir.exists():
        depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    else:
        depth_files = []
    print("depth length:", len(depth_files))

    img_files = [
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(depth_files) == len(img_files) and len(depth_files) > 0:
        print(f"✓ Depth exists ({len(depth_files)} files)")
        return True

    print("🔍 Generating depth...")
    success = generate_depth_for_scene(scene_name, angle, str(BASE))
    return success


# =======================
# CORE PER-VIDEO PIPELINE
# =======================

def crapify_single_video(
    scene_name: str,
    angle: str,
    fog_intensity: str = "heavy",
    rain_intensity: str = "heavy",
    droplet_intensity: str = "heavy",
    persistent_droplets: bool = True,
):
    """
    Crapify a single scene/angle sequence (~200 frames).

    This assumes:
        - clean frames in: data/scene_xxx/images/angle
        - depth maps in:   data/scene_xxx/depth/angle  (or they will be generated)

    Output:
        data_after_crapification_single/scene_xxx/angle
    """
    assert fog_intensity in FOG_PARAMS, f"Invalid fog intensity: {fog_intensity}"
    assert rain_intensity in RAIN_PARAMS, f"Invalid rain intensity: {rain_intensity}"

    scene_dir = DATA_DIR / scene_name
    img_dir = scene_dir / "images" / angle
    depth_dir = scene_dir / "depth" / angle

    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    if not ensure_depth_exists(scene_name, angle):
        raise RuntimeError("Depth generation failed")

    # Output directories
    output_dir = OUTPUT_BASE / scene_name / angle
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary dirs for intermediate stages – all under OUTPUT_BASE to keep IO local
    temp_root = OUTPUT_BASE / "_temp" / f"{scene_name}_{angle}"
    clean_dir = img_dir            # use original clean images directly
    fog_dir = temp_root / "01_fog"
    rain_mask_dir = temp_root / "02_rain_masks"
    rain_dir = temp_root / "03_rain"

    fog_dir.mkdir(parents=True, exist_ok=True)
    rain_mask_dir.mkdir(parents=True, exist_ok=True)
    rain_dir.mkdir(parents=True, exist_ok=True)

    # Count images
    num_images = sum(
        1 for f in os.listdir(clean_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if num_images == 0:
        raise RuntimeError(f"No images found in {clean_dir}")

    print("\n" + "=" * 60)
    print(f"Crapifying single video: {scene_name}/{angle}")
    print("=" * 60)
    print(f"Frames: {num_images}")
    print(f"Fog:      {fog_intensity}")
    print(f"Rain:     {rain_intensity}")
    print(f"Droplets: {droplet_intensity} (persistent={persistent_droplets})")
    print("=" * 60 + "\n")

    start = datetime.now()

    # 1) Fog (depth-based)
    fog_params = FOG_PARAMS[fog_intensity]
    print("[1] Fog...")
    run_fog_stage(
        img_dir=str(clean_dir),
        depth_dir=str(depth_dir),
        output_dir=str(fog_dir),
        fog_density=fog_params["fog_density"],
        airlight=fog_params["airlight"],
    )
    free_gpu_memory()

    # 2) Rain masks
    rain_params = RAIN_PARAMS[rain_intensity]
    print("[2] Rain masks...")
    run_rain_mask_stage(
        depth_dir=str(depth_dir),
        texture_dir=str(TEXTURE_DIR),
        output_dir=str(rain_mask_dir),
        rain_density=rain_params["density"],
        min_length=rain_params["min_length"],
        max_length=rain_params["max_length"],
    )
    free_gpu_memory()

    # 3) Composite fog + rain
    print("[3] Composite...")
    run_composite_stage(
        fog_dir=str(fog_dir),
        rain_dir=str(rain_mask_dir),
        output_dir=str(rain_dir),
        rain_brightness=0.4,
    )
    free_gpu_memory()

    # 4) Droplets (GPU, persistent)
    print("[4] Droplets (GPU)...")
    run_droplet_stage(
        input_dir=str(rain_dir),
        output_dir=str(output_dir),
        mask_dir=None,
        seed=hash(f"{scene_name}_{angle}") % 10000,
        intensity=droplet_intensity,
        use_gpu=True,          # <- GPU acceleration
        persistent=persistent_droplets,
    )
    free_gpu_memory()

    duration = datetime.now() - start
    print("\n" + "=" * 60)
    print("DONE – Single video crapified")
    print("=" * 60)
    print(f"Scene:   {scene_name}")
    print(f"Angle:   {angle}")
    print(f"Frames:  {num_images}")
    print(f"Time:    {duration}")
    print(f"Output:  {output_dir}")
    print("=" * 60 + "\n")

    # Cleanup temp
    try:
        shutil.rmtree(temp_root)
    except Exception:
        pass


# =======================
# CLI
# =======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick crapification for a single scene/angle (~200 frames)."
    )

    # Defaults so you can run with NO ARGS
    parser.add_argument(
        "--scene",
        type=str,
        default="scene_004",
        help="Scene name, e.g. scene_004 (default: scene_004)",
    )
    parser.add_argument(
        "--angle",
        type=str,
        default="front-forward",
        help="Angle, e.g. front-forward / left-forward / ... (default: front-forward)",
    )

    parser.add_argument(
        "--fog",
        type=str,
        default="medium",
        choices=list(FOG_PARAMS.keys()),
        help="Fog intensity (default: medium)",
    )
    parser.add_argument(
        "--rain",
        type=str,
        default="medium",
        choices=list(RAIN_PARAMS.keys()),
        help="Rain intensity (default: medium)",
    )
    parser.add_argument(
        "--droplets",
        type=str,
        default="medium",
        help="Droplet intensity (string passed to droplet stage, default: medium)",
    )

    parser.add_argument(
        "--no-persistent",
        action="store_true",
        help="Use non-persistent droplets instead of persistent",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    crapify_single_video(
        scene_name=args.scene,
        angle=args.angle,
        fog_intensity=args.fog,
        rain_intensity=args.rain,
        droplet_intensity=args.droplets,
        persistent_droplets=not args.no_persistent,
    )


if __name__ == "__main__":
    main()
