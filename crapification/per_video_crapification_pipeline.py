# crapification/per_video_crapification_pipeline.py
"""
Realistic test set pipeline with PERSISTENT droplets
Processes only test scenes with persistent droplets for realistic evaluation
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Get BASE directory (parent of crapification/)
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

# Import from crapification
from crapification.stages_crapification.stage_fog import run_fog_stage
from crapification.stages_crapification.stage_rain_masks import run_rain_mask_stage
from crapification.stages_crapification.stage_composite import run_composite_stage
from crapification.stages_crapification.stage_droplets import run_droplet_stage
from crapification.stages_crapification.generate_depth import generate_depth_for_scene
from crapification.helpers.scene_configurations import load_configurations

# =======================
# PATH CONFIG
# =======================

DATA_DIR = BASE / "data" / "data_original"
OUTPUT_BASE = BASE / "data" / "data_crapified_test"

# Configs inside crapification folder
CONFIG_FILE = Path(__file__).parent / "helpers" / "scene_intensity_configs.json"
SPLIT_FILE = Path(__file__).parent / "helpers" / "scene_split.json"
SPLIT_GROUP = "test"

TEXTURE_DIR = BASE / "crapification" / "rain-rendering" / "rainstreakdb"

ANGLES = [
    'front-forward',
    'left-backward',
    'left-forward',
    'right-backward',
    'right-forward'
]


# =======================
# TEST SCENE SELECTION
# =======================

def get_scene_numbers(split_group):
    """Load test scenes from scene_split.json"""
    if not SPLIT_FILE.exists():
        print(f"❌ Split file not found: {SPLIT_FILE}")
        print("   Run determine_split.py first!")
        return []

    with open(SPLIT_FILE, 'r') as f:
        split_info = json.load(f)

    return split_info[split_group]


# =======================
# INTENSITY PARAMETERS
# =======================

FOG_PARAMS = {
    'none': {'fog_density': 0.0, 'airlight': 255},
    'light': {'fog_density': 0.02, 'airlight': 240},
    'medium': {'fog_density': 0.06, 'airlight': 230},
    'heavy': {'fog_density': 0.12, 'airlight': 220}
}

RAIN_PARAMS = {
    'none': {'density': 0, 'min_length': 0, 'max_length': 0},
    'light': {'density': 1000, 'min_length': 8, 'max_length': 20},
    'medium': {'density': 2500, 'min_length': 8, 'max_length': 35},
    'heavy': {'density': 4000, 'min_length': 15, 'max_length': 50}
}


# =======================
# UTIL
# =======================

def copy_clean_images(src, dst):
    Path(dst).mkdir(parents=True, exist_ok=True)
    count = 0
    for f in os.listdir(src):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            shutil.copy2(
                os.path.join(src, f),
                os.path.join(dst, f)
            )
            count += 1
    return count


def ensure_depth_exists(scene_name, angle):
    depth_dir = DATA_DIR / scene_name / "depth" / angle
    img_dir = DATA_DIR / scene_name / "images" / angle

    if not img_dir.exists():
        print(f"  ⚠️  Images not found: {img_dir}")
        return False

    if depth_dir.exists():
        depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        depth_files = []

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(depth_files) == len(img_files) and len(depth_files) > 0:
        print(f"  ✓ Depth exists ({len(depth_files)} files)")
        return True

    print(f"  🔍 Generating depth...")
    success = generate_depth_for_scene(scene_name, angle, str(BASE))
    return success


# =======================
# PROCESS TEST SCENE
# =======================

def process_test_scene_angle(scene_name, angle, temp_dir, output_dir, intensity_config):
    """Process test scene with PERSISTENT droplets"""
    scene_dir = DATA_DIR / scene_name
    img_dir = scene_dir / "images" / angle
    depth_dir = scene_dir / "depth" / angle

    if not img_dir.exists():
        print(f"  ⚠️  Images not found")
        return 0

    if not ensure_depth_exists(scene_name, angle):
        print(f"  ❌ Depth generation failed")
        return 0

    # Temp directories
    clean_dir = temp_dir / "00_clean"
    fog_dir = temp_dir / "01_fog"
    rain_mask_dir = temp_dir / "02_rain_masks"
    rain_dir = temp_dir / "03_rain"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fog_intensity = intensity_config['fog']
    rain_intensity = intensity_config['rain']
    droplet_intensity = intensity_config['droplets']

    print(f"  🎨 Config: Fog={fog_intensity}, Rain={rain_intensity}, Droplets={droplet_intensity} (PERSISTENT)")

    try:
        # 0️⃣ Copy clean
        print("  [0] Copying clean...")
        num_images = copy_clean_images(img_dir, clean_dir)
        if num_images == 0:
            return 0

        # 1️⃣ Fog (depth-based, varies per frame - same as training)
        fog_params = FOG_PARAMS[fog_intensity]
        print(f"  [1] Fog ({fog_intensity})...")
        run_fog_stage(
            img_dir=str(clean_dir),
            depth_dir=str(depth_dir),
            output_dir=str(fog_dir),
            fog_density=fog_params['fog_density'],
            airlight=fog_params['airlight']
        )

        # 2️⃣ Rain masks (same as training)
        rain_params = RAIN_PARAMS[rain_intensity]
        print(f"  [2] Rain masks ({rain_intensity})...")
        run_rain_mask_stage(
            depth_dir=str(depth_dir),
            texture_dir=str(TEXTURE_DIR),
            output_dir=str(rain_mask_dir),
            rain_density=rain_params['density'],
            min_length=rain_params['min_length'],
            max_length=rain_params['max_length']
        )

        # 3️⃣ Composite
        print("  [3] Compositing...")
        run_composite_stage(
            fog_dir=str(fog_dir),
            rain_dir=str(rain_mask_dir),
            output_dir=str(rain_dir),
            rain_brightness=0.4
        )

        # 4️⃣ PERSISTENT droplets (KEY DIFFERENCE!)
        print(f"  [4] PERSISTENT droplets ({droplet_intensity})...")
        run_droplet_stage(
            input_dir=str(rain_dir),
            output_dir=str(output_dir),
            mask_dir=None,
            seed=hash(f"{scene_name}_{angle}") % 10000,
            intensity=droplet_intensity,
            use_gpu=True,
            persistent=True  # ← PERSISTENT for test set!
        )

        print(f"  ✓ Processed {num_images} images with persistent droplets")
        return num_images

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


# =======================
# MAIN
# =======================

def main():
    print("\n" + "=" * 60)
    print("REALISTIC TEST SET PIPELINE")
    print("WITH PERSISTENT DROPLETS")
    print("=" * 60 + "\n")

    # Get test scenes
    scenes = get_scene_numbers(SPLIT_GROUP)

    if not scenes:
        print("❌ No scenes found!")
        return

    print(f"📋 scenes (80/10/10 split):")
    print(f"   Scenes: {scenes}")
    print(f"   Count: {len(scenes)} scenes")
    print(f"   Angles: {len(ANGLES)}")
    print(f"   Total: {len(scenes) * len(ANGLES)} combinations\n")

    # Load intensity configs
    if not CONFIG_FILE.exists():
        print(f"❌ Configuration file not found: {CONFIG_FILE}")
        print("   Run per_frame_crapification_pipeline.py first.")
        return

    scene_configs = load_configurations(str(CONFIG_FILE))

    start_time = datetime.now()
    total_images = 0
    completed = 0
    total_combinations = len(scenes) * len(ANGLES)

    temp_base = BASE / "temp_test_pipeline"

    # Process test scenes
    for scene_num in scenes:
        scene_name = f"scene_{scene_num:03d}"
        intensity_config = scene_configs[scene_num]

        for angle in ANGLES:
            print(f"\n{'=' * 60}")
            print(f"Processing: {scene_name}/{angle} (TEST - PERSISTENT)")
            print(f"{'=' * 60}")

            temp_dir = temp_base / f"{scene_name}_{angle}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            output_dir = OUTPUT_BASE / scene_name / angle

            num_images = process_test_scene_angle(
                scene_name, angle, temp_dir, output_dir, intensity_config
            )

            if num_images > 0:
                total_images += num_images
                completed += 1

            try:
                shutil.rmtree(temp_dir)
            except:
                pass

            percent = (completed / total_combinations) * 100
            print(f"\n📊 Progress: {completed}/{total_combinations} ({percent:.1f}%)")

    # Cleanup
    try:
        shutil.rmtree(temp_base)
    except:
        pass

    duration = datetime.now() - start_time

    print(f"\n{'=' * 60}")
    print("TEST SET COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Time: {duration}")
    print(f"Images: {total_images}")
    print(f"Completed: {completed}/{total_combinations}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"\nTest set created with PERSISTENT droplets")
    print(f"Ready for model evaluation")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()