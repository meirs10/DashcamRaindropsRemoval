import os
import cv2
from pathlib import Path
import subprocess

# =======================
# PATH CONFIG
# =======================

BASE = r"D:\Pycharm Projects\DashcamRaindropsRemoval"
DATA_DIR = os.path.join(BASE, "data")
OUTPUT_DIR = os.path.join(BASE, "data_after_crapification")
VIDEO_OUTPUT_DIR = os.path.join(BASE, "videos")

ANGLES = [
    'front-forward',
    'left-backward',
    'left-forward',
    'right-backward',
    'right-forward'
]


def get_user_choice():
    """Get user input for video generation"""
    print("\n" + "=" * 60)
    print("VIDEO GENERATOR - 10 FPS")
    print("=" * 60 + "\n")

    # 1. Source selection
    print("1️⃣  Select source:")
    print("   [1] Clean images")
    print("   [2] Crapified images")
    source_choice = input("\nEnter choice (1 or 2): ").strip()

    if source_choice == '1':
        source = 'clean'
    elif source_choice == '2':
        source = 'crapified'
    else:
        print("❌ Invalid choice. Defaulting to clean.")
        source = 'clean'

    # 2. Scene selection
    print(f"\n2️⃣  Enter scene number (1-101):")
    scene_num = input("Scene number: ").strip()

    try:
        scene_num = int(scene_num)
        if not 1 <= scene_num <= 101:
            print("❌ Invalid scene number. Must be 1-101.")
            return None
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return None

    scene_name = f"scene_{scene_num:03d}"

    # 3. Angle selection
    print(f"\n3️⃣  Select camera angle:")
    for i, angle in enumerate(ANGLES, 1):
        print(f"   [{i}] {angle}")

    angle_choice = input("\nEnter choice (1-5): ").strip()

    try:
        angle_idx = int(angle_choice) - 1
        if not 0 <= angle_idx < len(ANGLES):
            print("❌ Invalid choice.")
            return None
        angle = ANGLES[angle_idx]
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return None

    return {
        'source': source,
        'scene_name': scene_name,
        'scene_num': scene_num,
        'angle': angle
    }


def create_video_opencv(image_dir, output_path, fps=10):
    """Create video using OpenCV"""
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return False

    print(f"\n📸 Found {len(image_files)} images")

    # Read first image to get dimensions
    first_img_path = os.path.join(image_dir, image_files[0])
    first_img = cv2.imread(first_img_path)

    if first_img is None:
        print(f"❌ Could not read image: {first_img_path}")
        return False

    height, width = first_img.shape[:2]
    print(f"📐 Resolution: {width}x{height}")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\n🎬 Creating video at {fps} FPS...")

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(image_dir, img_file)
        frame = cv2.imread(img_path)

        if frame is None:
            print(f"⚠️  Skipping corrupted image: {img_file}")
            continue

        out.write(frame)

        if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
            print(f"   Progress: {i + 1}/{len(image_files)}")

    out.release()
    print(f"\n✅ Video saved: {output_path}")

    # Get video duration
    duration = len(image_files) / fps
    print(f"⏱️  Duration: {duration:.2f} seconds ({len(image_files)} frames @ {fps} FPS)")

    return True


def create_video_ffmpeg(image_dir, output_path, fps=10):
    """Create video using FFmpeg (better quality, if available)"""
    # Get all image files
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        print(f"❌ No images found in {image_dir}")
        return False

    print(f"\n📸 Found {len(image_files)} images")

    # Create a temporary file list for FFmpeg
    temp_list = os.path.join(image_dir, "temp_filelist.txt")

    with open(temp_list, 'w') as f:
        for img_file in image_files:
            # FFmpeg concat format
            f.write(f"file '{img_file}'\n")
            f.write(f"duration {1 / fps}\n")

    # FFmpeg command
    cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', temp_list,
        '-vsync', 'vfr',
        '-pix_fmt', 'yuv420p',
        '-vcodec', 'libx264',
        '-crf', '18',
        '-y',
        output_path
    ]

    print(f"\n🎬 Creating video with FFmpeg at {fps} FPS...")

    try:
        result = subprocess.run(cmd, cwd=image_dir, capture_output=True, text=True)

        if result.returncode == 0:
            os.remove(temp_list)
            print(f"\n✅ Video saved: {output_path}")

            duration = len(image_files) / fps
            print(f"⏱️  Duration: {duration:.2f} seconds ({len(image_files)} frames @ {fps} FPS)")
            return True
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
            os.remove(temp_list)
            return False

    except FileNotFoundError:
        print("⚠️  FFmpeg not found. Falling back to OpenCV...")
        os.remove(temp_list)
        return None


def main():
    # Get user choices
    choices = get_user_choice()

    if not choices:
        return

    source = choices['source']
    scene_name = choices['scene_name']
    scene_num = choices['scene_num']
    angle = choices['angle']

    # Determine source directory
    if source == 'clean':
        image_dir = os.path.join(DATA_DIR, scene_name, "images", angle)
    else:  # crapified
        image_dir = os.path.join(OUTPUT_DIR, scene_name, angle)

    # Check if directory exists
    if not os.path.exists(image_dir):
        print(f"\n❌ Directory not found: {image_dir}")
        return

    # Create output directory
    Path(VIDEO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Output filename
    output_filename = f"{scene_name}_{angle}_{source}.mp4"
    output_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)

    print("\n" + "=" * 60)
    print("GENERATING VIDEO")
    print("=" * 60)
    print(f"Source:    {source}")
    print(f"Scene:     {scene_name}")
    print(f"Angle:     {angle}")
    print(f"Input:     {image_dir}")
    print(f"Output:    {output_path}")
    print("=" * 60)

    # Try FFmpeg first, fall back to OpenCV
    result = create_video_ffmpeg(image_dir, output_path, fps=10)

    if result is None:  # FFmpeg not available
        result = create_video_opencv(image_dir, output_path, fps=10)

    if result:
        print(f"\n🎉 Done! Video saved to: {VIDEO_OUTPUT_DIR}")
    else:
        print(f"\n❌ Failed to create video")


if __name__ == "__main__":
    main()