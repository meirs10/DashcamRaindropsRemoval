import torch
import cv2
import os
import glob
import numpy as np
from pathlib import Path


def generate_depth_for_scene(scene_name, angle, base_dir):
    """
    Generate depth maps for a specific scene and camera angle.

    Args:
        scene_name: e.g., "scene_001"
        angle: e.g., "front-forward"
        base_dir: Base project directory

    Returns:
        True if successful, False otherwise
    """
    base_path = Path(base_dir)

    # Define paths
    input_folder = base_path / "data_original" / scene_name / "images" / angle
    output_folder = base_path / "data_original" / scene_name / "depth" / angle

    # Check if input exists
    if not input_folder.exists():
        print(f"  ⚠️  Images not found: {input_folder}")
        return False

    # Check if depth already exists
    existing_depth = list(output_folder.glob("*.png")) if output_folder.exists() else []
    existing_images = list(input_folder.glob("*.png")) + list(input_folder.glob("*.jpg")) + list(
        input_folder.glob("*.jpeg"))

    if len(existing_depth) == len(existing_images) and len(existing_depth) > 0:
        print(f"  ✓ Depth already exists ({len(existing_depth)} files)")
        return True

    print(f"  📊 Generating depth maps for {scene_name}/{angle}...")

    try:
        # Load MiDaS model
        print("    Loading MiDaS model...")
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()

        if torch.cuda.is_available():
            print(f"    🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("    Using CPU")

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        transform = midas_transforms.small_transform

        # Create output directory
        output_folder.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_files = sorted(
            list(input_folder.glob("*.png")) +
            list(input_folder.glob("*.jpg")) +
            list(input_folder.glob("*.jpeg"))
        )

        if len(image_files) == 0:
            print(f"  ⚠️  No images found in {input_folder}")
            return False

        print(f"    Processing {len(image_files)} images...")

        # Process each image
        for idx, img_path in enumerate(image_files):
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"    ⚠️  Could not read: {img_path.name}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Predict depth
            input_batch = transform(img_rgb).to(device)

            with torch.no_grad():
                prediction = midas(input_batch)

                # Resize to original resolution
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Normalize to 0-255
            depth_map = prediction.cpu().numpy()
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
            depth_img = (depth_norm * 255).astype(np.uint8)

            # Save with same filename
            save_path = output_folder / img_path.name
            cv2.imwrite(str(save_path), depth_img)

            if (idx + 1) % 20 == 0 or (idx + 1) == len(image_files):
                print(f"    Depth: {idx + 1}/{len(image_files)}")

        print(f"  ✓ Depth generation complete ({len(image_files)} files)")
        return True

    except Exception as e:
        print(f"  ❌ Error generating depth: {str(e)}")
        return False


def main():
    """Standalone mode - generate depth for specific scene"""
    project_root = Path(r"/")

    print("\n=== DEPTH MAP GENERATION ===\n")

    # You can modify these for standalone use
    scene_name = "scene_001"
    angle = "front-forward"

    success = generate_depth_for_scene(scene_name, angle, project_root)

    if success:
        print(f"\n✓ Depth maps saved to: data_original/{scene_name}/depth/{angle}\n")
    else:
        print("\n❌ Depth generation failed\n")


if __name__ == "__main__":
    main()