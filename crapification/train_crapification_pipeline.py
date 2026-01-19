# crapification/train_crapification_pipeline.py
"""
Training data crapification pipeline
Processes only train+val scenes (excludes test scenes)
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Get BASE directory (parent of crapification/)
BASE = Path(__file__).parent.parent  # Go up one level from crapification/
sys.path.insert(0, str(BASE))

# Now import from crapification
from crapification.stage_fog import run_fog_stage
from crapification.stage_rain_masks import run_rain_mask_stage
from crapification.stage_composite import run_composite_stage
from crapification.stage_droplets import run_droplet_stage
from crapification.generate_depth import generate_depth_for_scene
from crapification.scene_configurations import (
    generate_scene_configurations,
    load_configurations,
    save_configurations,
    print_configuration_summary
)

# =======================
# PATH CONFIG
# =======================

DATA_DIR = BASE / "data"
OUTPUT_BASE = BASE / "data_after_crapification"

# Configs inside crapification folder
CONFIG_FILE = Path(__file__).parent / "scene_intensity_configs.json"
SPLIT_FILE = Path(__file__).parent / "scene_split.json"
PROGRESS_FILE = Path(__file__).parent / "pipeline_progress.json"

TEXTURE_DIR = BASE / "rain-rendering" / "3rdparty" / "rainstreakdb"

# Camera angles
ANGLES = [
    'front-forward',
    'left-backward',
    'left-forward',
    'right-backward',
    'right-forward'
]

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

DROPLET_PARAMS = {
    'light': {'n_large': 10, 'n_medium': 18},
    'medium': {'n_large': 16, 'n_medium': 28},
    'heavy': {'n_large': 28, 'n_medium': 45},
    'extreme': {'n_large': 40, 'n_medium': 60}
}


# =======================
# GET TRAIN+VAL SCENES
# =======================

def get_train_val_scenes():
    """Load train+val scenes from scene_split.json"""
    if SPLIT_FILE.exists():
        print(f"📋 Loading scene split from: {SPLIT_FILE}")
        with open(SPLIT_FILE, 'r') as f:
            split_info = json.load(f)

        train_scenes = split_info['train']
        val_scenes = split_info['val']
        train_val_scenes = sorted(train_scenes + val_scenes)

        print(f"   Train scenes: {len(train_scenes)}")
        print(f"   Val scenes: {len(val_scenes)}")
        print(f"   Total to process: {len(train_val_scenes)}")
        print(f"   Test scenes (EXCLUDED): {len(split_info['test'])}")

        return train_val_scenes

    else:
        print("⚠️  scene_split.json not found!")
        print("   Run determine_split.py first!")
        return []


# =======================
# PROGRESS TRACKING
# =======================

class ProgressTracker:
    def __init__(self):
        self.progress_file = PROGRESS_FILE
        self.progress = self.load_progress()

    def load_progress(self):
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed': [], 'last_scene': None, 'last_angle': None}

    def save_progress(self):
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def is_completed(self, scene, angle):
        key = f"{scene}_{angle}"
        return key in self.progress['completed']

    def mark_completed(self, scene, angle):
        key = f"{scene}_{angle}"
        if key not in self.progress['completed']:
            self.progress['completed'].append(key)
        self.progress['last_scene'] = scene
        self.progress['last_angle'] = angle
        self.save_progress()

    def get_status(self):
        return len(self.progress['completed'])


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
        depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith('.png')]
    else:
        depth_files = []

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(depth_files) == len(img_files) and len(depth_files) > 0:
        print(f"  ✓ Depth exists ({len(depth_files)} files)")
        return True

    print(f"  🔍 Depth missing - generating...")
    success = generate_depth_for_scene(scene_name, angle, str(BASE))
    return success


# =======================
# PROCESS SINGLE SCENE
# =======================

def process_scene_angle(scene_name, angle, temp_dir, output_dir, intensity_config):
    """Process a single scene+angle with random droplets"""
    scene_dir = DATA_DIR / scene_name
    img_dir = scene_dir / "images" / angle
    depth_dir = scene_dir / "depth" / angle

    if not img_dir.exists():
        print(f"  ⚠️  Images not found: {img_dir}")
        return 0

    if not ensure_depth_exists(scene_name, angle):
        print(f"  ❌ Failed to generate depth maps")
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

    print(f"  🎨 Config: Fog={fog_intensity}, Rain={rain_intensity}, Droplets={droplet_intensity} (RANDOM)")

    try:
        # 0️⃣ Copy clean
        print("  [0] Copying clean...")
        num_images = copy_clean_images(img_dir, clean_dir)
        if num_images == 0:
            return 0

        # 1️⃣ Fog
        fog_params = FOG_PARAMS[fog_intensity]
        print(f"  [1] Fog ({fog_intensity})...")
        run_fog_stage(
            img_dir=str(clean_dir),
            depth_dir=str(depth_dir),
            output_dir=str(fog_dir),
            fog_density=fog_params['fog_density'],
            airlight=fog_params['airlight']
        )

        # 2️⃣ Rain masks
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

        # 4️⃣ Random droplets
        print(f"  [4] Droplets ({droplet_intensity}, RANDOM)...")
        run_droplet_stage(
            input_dir=str(rain_dir),
            output_dir=str(output_dir),
            mask_dir=None,
            seed=hash(f"{scene_name}_{angle}") % 10000,
            intensity=droplet_intensity,
            use_gpu=True,
            persistent=False
        )

        print(f"  ✓ Processed {num_images} images")
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
    print("TRAINING DATA CRAPIFICATION PIPELINE")
    print("(TRAIN + VAL SCENES ONLY)")
    print("=" * 60 + "\n")

    train_val_scenes = get_train_val_scenes()

    if not train_val_scenes:
        print("❌ No scenes to process!")
        return

    total_combinations = len(train_val_scenes) * len(ANGLES)

    # Load configs
    if CONFIG_FILE.exists():
        print("\n📋 Loading scene configurations...")
        scene_configs = load_configurations(str(CONFIG_FILE))
    else:
        print("\n🎲 Generating scene configurations...")
        scene_configs = generate_scene_configurations(num_scenes=101, seed=42)
        save_configurations(scene_configs, str(CONFIG_FILE))

    print_configuration_summary(scene_configs)

    tracker = ProgressTracker()

    print(f"\nScenes to process: {len(train_val_scenes)} (train+val)")
    print(f"Angles per scene: {len(ANGLES)}")
    print(f"Total combinations: {total_combinations}")
    print(f"Already completed: {tracker.get_status()}")
    print(f"\n{'=' * 60}\n")

    start_time = datetime.now()
    total_images = 0

    temp_base = BASE / "temp_pipeline"

    for scene_num in train_val_scenes:
        scene_name = f"scene_{scene_num:03d}"
        intensity_config = scene_configs[scene_num]

        for angle in ANGLES:
            if tracker.is_completed(scene_name, angle):
                print(f"✓ Skipping {scene_name}/{angle} (completed)")
                continue

            print(f"\n{'=' * 60}")
            print(f"Processing: {scene_name}/{angle} (TRAIN/VAL)")
            print(f"{'=' * 60}")

            temp_dir = temp_base / f"{scene_name}_{angle}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            output_dir = OUTPUT_BASE / scene_name / angle

            num_images = process_scene_angle(
                scene_name, angle, temp_dir, output_dir, intensity_config
            )

            if num_images > 0:
                total_images += num_images
                tracker.mark_completed(scene_name, angle)

            try:
                shutil.rmtree(temp_dir)
            except:
                pass

            completed = tracker.get_status()
            percent = (completed / total_combinations) * 100
            print(f"\n📊 Progress: {completed}/{total_combinations} ({percent:.1f}%)")

    try:
        shutil.rmtree(temp_base)
    except:
        pass

    duration = datetime.now() - start_time

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Time: {duration}")
    print(f"Images: {total_images}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()