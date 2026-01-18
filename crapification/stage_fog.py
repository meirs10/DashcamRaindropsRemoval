import cv2
import numpy as np
import os
from pathlib import Path
import torch


def add_atmosphere_gpu(img, depth, fog_density=0.06, airlight=230, device='cuda'):
    """GPU-accelerated atmospheric fog effect using PyTorch"""
    # Convert to PyTorch tensors [H, W, C]
    img_tensor = torch.from_numpy(img).float().to(device)
    depth_tensor = torch.from_numpy(depth).float().to(device)

    # Normalize depth
    d_norm = depth_tensor / 255.0

    # Calculate transmission: exp(-fog_density * (d_norm * 10.0))
    transmission = torch.exp(-fog_density * (d_norm * 10.0))

    # Expand transmission to 3 channels [H, W, 3]
    t_map = transmission.unsqueeze(-1).repeat(1, 1, 3)

    # Calculate: img * t_map + airlight * (1.0 - t_map)
    fogged = img_tensor * t_map + airlight * (1.0 - t_map)

    # Clamp and convert back
    result = torch.clamp(fogged, 0, 255)
    return result.cpu().numpy().astype(np.uint8)


def add_atmosphere_cpu(img, depth, fog_density=0.06, airlight=230):
    """Fallback CPU version"""
    d_norm = depth.astype(float) / 255.0
    transmission = np.exp(-fog_density * (d_norm * 10.0))
    t_map = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    fogged = img.astype(float) * t_map + airlight * (1.0 - t_map)
    return np.clip(fogged, 0, 255).astype(np.uint8)


def run_fog_stage(img_dir, depth_dir, output_dir,
                  fog_density=0.06, airlight=230, use_gpu=True):
    """
    Apply depth-based fog to images.
    Only saves final fogged images.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(img_dir))

    # Check GPU availability
    gpu_available = False
    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        gpu_available = True
        device = 'cuda'
        print(f"    🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    Using CPU")

    print(f"    Applying fog ({len(files)} frames)")

    for i, f in enumerate(files):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img = cv2.imread(os.path.join(img_dir, f))
        if img is None:
            continue

        depth_path = os.path.join(depth_dir, os.path.splitext(f)[0] + ".png")
        if not os.path.exists(depth_path):
            depth_path = os.path.join(depth_dir, f)

        if os.path.exists(depth_path):
            depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

            if gpu_available:
                result = add_atmosphere_gpu(img, depth, fog_density, airlight, device)
            else:
                result = add_atmosphere_cpu(img, depth, fog_density, airlight)
        else:
            result = img

        # ONLY save final fogged image
        cv2.imwrite(os.path.join(output_dir, f), result)

        if (i + 1) % 20 == 0:
            print(f"    Fog: {i + 1}/{len(files)}")

    print("    ✓ Fog stage complete.")