"""
Quick crapification for a SINGLE video (one scene/angle, ~200 frames).

Now supports crapifying only a persistent n×n crop in the frame.

Pipeline:
    clean (data/scene_xxx/images/angle)
  -> crop (same crop location for all frames)
  -> fog (depth-based)
  -> rain masks
  -> composite
  -> droplets (GPU, persistent or not)
  -> paste crop back into full frame

Default run (no CLI args):
    scene_004 / front-forward
    fog = medium, rain = medium, droplets = medium, persistent = True
    crop_size = 512, crop location = random but fixed per video
"""
import gc
import os
import sys
import shutil
import argparse
import time
import random
from pathlib import Path
from datetime import datetime

import cv2
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


def get_image_file_list(dir_path: Path):
    return sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


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
    crop_size: int = 512,
    crop_x: int | None = None,
    crop_y: int | None = None,
):
    """
    Crapify a single scene/angle sequence (~200 frames), but only inside
    a persistent n×n crop across all frames.

    Output:
        data_after_crapification_single/scene_xxx/angle
        where each frame is crop_size×crop_size (the crapified crop only).
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

    # Final output dir: cropped crapified frames
    output_dir = OUTPUT_BASE / scene_name / angle
    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporary dirs for intermediate stages – all under OUTPUT_BASE/_temp
    temp_root = OUTPUT_BASE / "_temp" / f"{scene_name}_{angle}"
    crop_img_dir = temp_root / "00_crop" / "images"
    crop_depth_dir = temp_root / "00_crop" / "depth"
    fog_dir = temp_root / "01_fog"
    rain_mask_dir = temp_root / "02_rain_masks"
    rain_dir = temp_root / "03_rain"

    crop_img_dir.mkdir(parents=True, exist_ok=True)
    crop_depth_dir.mkdir(parents=True, exist_ok=True)
    fog_dir.mkdir(parents=True, exist_ok=True)
    rain_mask_dir.mkdir(parents=True, exist_ok=True)
    rain_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if len(img_files) == 0:
        raise RuntimeError(f"No images found in {img_dir}")
    num_images = len(img_files)

    # Determine persistent crop location
    first_img_path = img_dir / img_files[0]
    first_img = cv2.imread(str(first_img_path))
    if first_img is None:
        raise RuntimeError(f"Failed to read first image: {first_img_path}")
    h, w = first_img.shape[:2]

    if crop_size <= 0 or crop_size > min(h, w):
        raise ValueError(
            f"Invalid crop_size={crop_size}. Must be >0 and <= min(H,W)={min(h,w)}"
        )

    if crop_x is None or crop_y is None:
        max_x = w - crop_size
        max_y = h - crop_size
        if max_x < 0 or max_y < 0:
            raise ValueError(
                f"crop_size={crop_size} too large for frame size {w}x{h}"
            )
        crop_x = random.randint(0, max_x)
        crop_y = random.randint(0, max_y)
    else:
        if not (0 <= crop_x <= w - crop_size and 0 <= crop_y <= h - crop_size):
            raise ValueError(
                f"Invalid crop_x/crop_y: ({crop_x}, {crop_y}) for frame size {w}x{h} and crop_size={crop_size}"
            )

    print("\n" + "=" * 60)
    print(f"Crapifying single video (CROPS ONLY): {scene_name}/{angle}")
    print("=" * 60)
    print(f"Frames:      {num_images}")
    print(f"Frame size:  {w}x{h}")
    print(f"Crop size:   {crop_size}x{crop_size}")
    print(f"Crop origin: (x={crop_x}, y={crop_y})  [persistent across frames]")
    print(f"Fog:         {fog_intensity}")
    print(f"Rain:        {rain_intensity}")
    print(f"Droplets:    {droplet_intensity} (persistent={persistent_droplets})")
    print("=" * 60 + "\n")

    # Build depth file list (assumes same names)
    depth_files = sorted(
        f for f in os.listdir(depth_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if len(depth_files) != num_images:
        print(f"⚠ Depth/image count mismatch: images={num_images}, depth={len(depth_files)}")
    depth_set = set(depth_files)

    # 0) Create cropped image + depth sequences
    print("[0] Cropping images and depth...")
    for fname in img_files:
        img_path = img_dir / fname
        depth_path = depth_dir / fname

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        if fname in depth_set:
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise RuntimeError(f"Failed to read depth: {depth_path}")
        else:
            raise RuntimeError(f"Depth file missing for image {fname} in {depth_dir}")

        img_crop = img[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
        depth_crop = depth[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

        cv2.imwrite(str(crop_img_dir / fname), img_crop)
        cv2.imwrite(str(crop_depth_dir / fname), depth_crop)

    free_gpu_memory()
    start = datetime.now()

    # 1) Fog on crops
    fog_params = FOG_PARAMS[fog_intensity]
    print("[1] Fog on crops...")
    run_fog_stage(
        img_dir=str(crop_img_dir),
        depth_dir=str(crop_depth_dir),
        output_dir=str(fog_dir),
        fog_density=fog_params["fog_density"],
        airlight=fog_params["airlight"],
    )
    free_gpu_memory()

    # 2) Rain masks on crops
    rain_params = RAIN_PARAMS[rain_intensity]
    print("[2] Rain masks on crops...")
    run_rain_mask_stage(
        depth_dir=str(crop_depth_dir),
        texture_dir=str(TEXTURE_DIR),
        output_dir=str(rain_mask_dir),
        rain_density=rain_params["density"],
        min_length=rain_params["min_length"],
        max_length=rain_params["max_length"],
    )
    free_gpu_memory()

    # 3) Composite on crops
    print("[3] Composite on crops...")
    run_composite_stage(
        fog_dir=str(fog_dir),
        rain_dir=str(rain_mask_dir),
        output_dir=str(rain_dir),
        rain_brightness=0.4,
    )
    free_gpu_memory()

    # 4) Droplets on crops – write directly to final output_dir
    print("[4] Droplets (GPU) on crops...")
    run_droplet_stage(
        input_dir=str(rain_dir),
        output_dir=str(output_dir),   # <-- final cropped outputs
        mask_dir=None,
        seed=hash(f"{scene_name}_{angle}") % 10000,
        intensity=droplet_intensity,
        use_gpu=True,
        persistent=persistent_droplets,
    )
    free_gpu_memory()

    duration = datetime.now() - start
    print("\n" + "=" * 60)
    print("DONE – Single video CROPS crapified")
    print("=" * 60)
    print(f"Scene:        {scene_name}")
    print(f"Angle:        {angle}")
    print(f"Frames:       {num_images}")
    print(f"Crop size:    {crop_size}x{crop_size}")
    print(f"Crop origin:  (x={crop_x}, y={crop_y})")
    print(f"Time:         {duration}")
    print(f"Output (crops only): {output_dir}")
    print("=" * 60 + "\n")

    # Cleanup temp (but keep final crops)
    try:
        shutil.rmtree(temp_root)
    except Exception:
        pass


# =======================
# CLI
# =======================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Quick crapification for a single scene/angle (~200 frames) on a persistent n×n crop."
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

    # New crop arguments
    parser.add_argument(
        "--crop-size",
        type=int,
        default=512,
        help="Square crop size in pixels (n×n). Default: 512",
    )
    parser.add_argument(
        "--crop-x",
        type=int,
        default=None,
        help="Left coordinate of crop (x). If not given, chosen randomly.",
    )
    parser.add_argument(
        "--crop-y",
        type=int,
        default=None,
        help="Top coordinate of crop (y). If not given, chosen randomly.",
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
        crop_size=args.crop_size,
        crop_x=args.crop_x,
        crop_y=args.crop_y,
    )


if __name__ == "__main__":
    main()
