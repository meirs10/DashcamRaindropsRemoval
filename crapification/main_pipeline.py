import os
import json
import shutil
from pathlib import Path
from datetime import datetime

from crapification.stage_fog import run_fog_stage
from crapification.stage_rain_masks import run_rain_mask_stage
from crapification.stage_composite import run_composite_stage
from crapification.stage_droplets import run_droplet_stage
from generate_depth import generate_depth_for_scene
from scene_configurations import generate_scene_configurations, load_configurations, save_configurations, \
    print_configuration_summary

# =======================
# PATH CONFIG
# =======================

BASE = r"D:\Pycharm Projects\DashcamRaindropsRemoval"
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_BASE = os.path.join(BASE, "data_after_crapification")
CONFIG_FILE = os.path.join(BASE, "scene_intensity_configs.json")

TEXTURE_DIR = os.path.join(
    BASE, "rain-rendering", "3rdparty", "rainstreakdb"
)

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
# PROGRESS TRACKING
# =======================

class ProgressTracker:
    def __init__(self, progress_file='pipeline_progress.json'):
        self.progress_file = os.path.join(BASE, progress_file)
        self.progress = self.load_progress()

    def load_progress(self):
        if os.path.exists(self.progress_file):
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
    depth_dir = os.path.join(DATA_DIR, scene_name, "depth", angle)
    img_dir = os.path.join(DATA_DIR, scene_name, "images", angle)

    if not os.path.exists(img_dir):
        print(f"  ⚠️  Images not found: {img_dir}")
        return False

    if os.path.exists(depth_dir):
        depth_files = [f for f in os.listdir(depth_dir) if f.lower().endswith('.png')]
    else:
        depth_files = []

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if len(depth_files) == len(img_files) and len(depth_files) > 0:
        print(f"  ✓ Depth exists ({len(depth_files)} files)")
        return True

    print(f"  🔍 Depth missing or incomplete - generating...")
    success = generate_depth_for_scene(scene_name, angle, BASE)

    return success


# =======================
# PROCESS SINGLE SCENE + ANGLE
# =======================

def process_scene_angle(scene_name, angle, temp_dir, output_dir, intensity_config):
    """
    Process a single scene+angle with specified intensity configuration
    """
    scene_dir = os.path.join(DATA_DIR, scene_name)
    img_dir = os.path.join(scene_dir, "images", angle)
    depth_dir = os.path.join(scene_dir, "depth", angle)

    if not os.path.exists(img_dir):
        print(f"  ⚠️  Images not found: {img_dir}")
        return 0

    # Ensure depth exists
    if not ensure_depth_exists(scene_name, angle):
        print(f"  ❌ Failed to generate/find depth maps - skipping")
        return 0

    # Create temp directories
    clean_dir = os.path.join(temp_dir, "00_clean")
    fog_dir = os.path.join(temp_dir, "01_fog")
    rain_mask_dir = os.path.join(temp_dir, "02_rain_masks")
    rain_dir = os.path.join(temp_dir, "03_rain")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get intensity parameters
    fog_intensity = intensity_config['fog']
    rain_intensity = intensity_config['rain']
    droplet_intensity = intensity_config['droplets']

    print(f"  🎨 Config: Fog={fog_intensity}, Rain={rain_intensity}, Droplets={droplet_intensity}")

    try:
        # 0️⃣ Copy clean images
        print("  [0] Copying clean images...")
        num_images = copy_clean_images(img_dir, clean_dir)
        if num_images == 0:
            print("  ⚠️  No images found")
            return 0

        # 1️⃣ Fog (with intensity)
        fog_params = FOG_PARAMS[fog_intensity]
        print(f"  [1] Running fog stage ({fog_intensity})...")
        run_fog_stage(
            img_dir=clean_dir,
            depth_dir=depth_dir,
            output_dir=fog_dir,
            fog_density=fog_params['fog_density'],
            airlight=fog_params['airlight']
        )

        # 2️⃣ Rain mask generation (with intensity)
        rain_params = RAIN_PARAMS[rain_intensity]
        print(f"  [2] Generating rain masks ({rain_intensity})...")
        run_rain_mask_stage(
            depth_dir=depth_dir,
            texture_dir=TEXTURE_DIR,
            output_dir=rain_mask_dir,
            rain_density=rain_params['density'],
            min_length=rain_params['min_length'],
            max_length=rain_params['max_length']
        )

        # 3️⃣ Rain compositing
        print("  [3] Compositing rain...")
        run_composite_stage(
            fog_dir=fog_dir,
            rain_dir=rain_mask_dir,
            output_dir=rain_dir,
            rain_brightness=0.4
        )

        # 4️⃣ Camera droplets (with intensity)
        droplet_params = DROPLET_PARAMS[droplet_intensity]
        print(f"  [4] Adding camera droplets ({droplet_intensity})...")
        run_droplet_stage(
            input_dir=rain_dir,
            output_dir=output_dir,
            mask_dir=None,
            seed=hash(f"{scene_name}_{angle}") % 10000,
            intensity=droplet_intensity,  # Just pass the intensity string
            use_gpu=True
        )

        print(f"  ✓ Processed {num_images} images")
        return num_images

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0


# =======================
# MAIN PIPELINE
# =======================

def main():
    print("\n" + "=" * 60)
    print("DASHCAM RAIN + DROPLET DATA PIPELINE")
    print("WITH DIVERSE INTENSITY CONFIGURATIONS")
    print("=" * 60 + "\n")

    NUM_SCENES = 101
    total_combinations = NUM_SCENES * len(ANGLES)

    # Load or generate scene configurations
    if os.path.exists(CONFIG_FILE):
        print("📋 Loading existing scene configurations...")
        scene_configs = load_configurations(CONFIG_FILE)
    else:
        print("🎲 Generating diverse scene configurations...")
        scene_configs = generate_scene_configurations(num_scenes=NUM_SCENES, seed=42)
        save_configurations(scene_configs, CONFIG_FILE)

    print_configuration_summary(scene_configs)

    # Initialize progress tracker
    tracker = ProgressTracker()

    print(f"Scenes to process: {NUM_SCENES}")
    print(f"Angles per scene: {len(ANGLES)}")
    print(f"Total combinations: {total_combinations}")
    print(f"Already completed: {tracker.get_status()}")
    print(f"\n{'=' * 60}\n")

    start_time = datetime.now()
    total_images = 0

    temp_base = os.path.join(BASE, "temp_pipeline")

    # Process each scene
    for scene_num in range(1, NUM_SCENES + 1):
        scene_name = f"scene_{scene_num:03d}"

        # Get intensity configuration for this scene
        intensity_config = scene_configs[scene_num]

        # Process each angle
        for angle in ANGLES:
            if tracker.is_completed(scene_name, angle):
                print(f"✓ Skipping {scene_name}/{angle} (already completed)")
                continue

            print(f"\n{'=' * 60}")
            print(f"Processing: {scene_name}/{angle}")
            print(f"{'=' * 60}")

            temp_dir = os.path.join(temp_base, f"{scene_name}_{angle}")
            Path(temp_dir).mkdir(parents=True, exist_ok=True)

            output_dir = os.path.join(OUTPUT_BASE, scene_name, angle)

            num_images = process_scene_angle(
                scene_name=scene_name,
                angle=angle,
                temp_dir=temp_dir,
                output_dir=output_dir,
                intensity_config=intensity_config
            )

            if num_images > 0:
                total_images += num_images
                tracker.mark_completed(scene_name, angle)
                print(f"✓ Completed {scene_name}/{angle}")

            try:
                shutil.rmtree(temp_dir)
            except:
                pass

            completed = tracker.get_status()
            percent = (completed / total_combinations) * 100
            print(f"\n📊 Overall Progress: {completed}/{total_combinations} ({percent:.1f}%)")
            print(f"   Total images processed: {total_images}")

    try:
        shutil.rmtree(temp_base)
    except:
        pass

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Total time: {duration}")
    print(f"Total images processed: {total_images}")
    print(f"Completed combinations: {tracker.get_status()}/{total_combinations}")
    print(f"Output directory: {OUTPUT_BASE}")
    print(f"\nReady for U-Net training.\n")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()