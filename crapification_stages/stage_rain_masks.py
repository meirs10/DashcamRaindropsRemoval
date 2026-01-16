import cv2
import numpy as np
import os
import random
from pathlib import Path
import math
import torch
import torch.nn.functional as F


def load_textures(texture_dir, limit=200):
    textures = []

    for r, _, fs in os.walk(texture_dir):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img = cv2.imread(os.path.join(r, f), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                textures.append(img)

    if not textures:
        raise RuntimeError("No rain textures found")

    random.shuffle(textures)
    return textures[:limit]


def generate_rain_streaks_cpu(shape, depth, rain_density, min_length, max_length,
                              min_thickness=1, max_thickness=3, angle_deg=None):
    """Generate rain streaks on CPU"""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)

    if angle_deg is None:
        angle_deg = random.uniform(-12, 12)

    angle = math.radians(angle_deg)
    dx = math.sin(angle)
    dy = math.cos(angle)

    for _ in range(rain_density):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        d = depth[y, x] / 255.0

        opacity = np.clip(1.0 - d, 0.15, 1.0)
        length = int(min_length + (1.0 - d) * (max_length - min_length))
        thickness = int(min_thickness + (1.0 - d) * (max_thickness - min_thickness))

        x2 = int(x + dx * length)
        y2 = int(y + dy * length)

        if x2 < 0 or x2 >= w or y2 < 0 or y2 >= h:
            continue

        cv2.line(mask, (x, y), (x2, y2), color=opacity,
                 thickness=thickness, lineType=cv2.LINE_AA)

    return mask


def apply_texture_gpu(mask, tex_resized, texture_strength=0.35, device='cuda'):
    """Apply texture modulation using PyTorch GPU"""
    mask_tensor = torch.from_numpy(mask).float().to(device)
    tex_tensor = torch.from_numpy(tex_resized).float().to(device)

    tex_norm = tex_tensor / 255.0
    mask_norm = mask_tensor / 255.0

    modulated = mask_norm * (1.0 - texture_strength + texture_strength * tex_norm)

    result = torch.clamp(modulated * 255.0, 0, 255)
    return result.cpu().numpy().astype(np.uint8)


def apply_texture_cpu(mask, tex_resized, texture_strength=0.35):
    """CPU fallback"""
    tex = tex_resized.astype(np.float32) / 255.0
    mask = mask * (1.0 - texture_strength + texture_strength * tex)
    return np.clip(mask * 255, 0, 255).astype(np.uint8)


def gaussian_blur_torch(img_tensor, kernel_size):
    """Gaussian blur using PyTorch"""
    sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    x = torch.arange(kernel_size, device=img_tensor.device, dtype=torch.float32)
    x = x - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()

    padding = kernel_size // 2

    img = img_tensor.unsqueeze(0).unsqueeze(0)

    blurred = F.conv2d(img, kernel.view(1, 1, 1, -1), padding=(0, padding))
    blurred = F.conv2d(blurred, kernel.view(1, 1, -1, 1), padding=(padding, 0))

    return blurred.squeeze(0).squeeze(0)


def generate_rain_mask_gpu(shape, depth, textures, rain_density, min_length, max_length, device='cuda'):
    """Generate rain mask with GPU acceleration"""
    h, w = shape

    # Generate streaks on CPU
    mask = generate_rain_streaks_cpu(shape, depth, rain_density, min_length, max_length)
    mask = (mask * 255).astype(np.uint8)

    # Apply texture on GPU
    tex = random.choice(textures)
    tex_resized = cv2.resize(tex, (w, h))
    mask = apply_texture_gpu(mask, tex_resized, device=device)

    return mask


def generate_rain_mask_cpu(shape, depth, textures, rain_density, min_length, max_length):
    """CPU-only version"""
    h, w = shape
    mask = generate_rain_streaks_cpu(shape, depth, rain_density, min_length, max_length)

    tex = random.choice(textures)
    tex_resized = cv2.resize(tex, (w, h))
    mask = apply_texture_cpu(mask, tex_resized)

    return mask


def run_rain_mask_stage(depth_dir, texture_dir, output_dir,
                        rain_density=2500, min_length=8, max_length=35,
                        use_gpu=True):
    """
    Generate rain masks based on depth with configurable intensity.

    Args:
        rain_density: Number of rain streaks (0 for none, 1000-4000+ typical)
        min_length: Minimum streak length in pixels
        max_length: Maximum streak length in pixels
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # If rain density is 0, skip rain generation
    if rain_density == 0:
        print("    ⊘ Skipping rain (density=0)")
        # Create empty masks for consistency
        files = sorted(os.listdir(depth_dir))
        for f in files:
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            depth = cv2.imread(os.path.join(depth_dir, f), cv2.IMREAD_GRAYSCALE)
            if depth is None:
                continue
            empty_mask = np.zeros_like(depth)
            out_name = os.path.splitext(f)[0] + ".png"
            cv2.imwrite(os.path.join(output_dir, out_name), empty_mask)
        return

    textures = load_textures(texture_dir)

    # Check GPU
    gpu_available = False
    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        gpu_available = True
        device = 'cuda'
        print(f"    🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    Using CPU")

    files = sorted(os.listdir(depth_dir))
    print(f"    Generating rain masks ({len(files)} frames, density={rain_density})")

    for i, f in enumerate(files):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        depth = cv2.imread(os.path.join(depth_dir, f), cv2.IMREAD_GRAYSCALE)
        if depth is None:
            continue

        if gpu_available:
            mask = generate_rain_mask_gpu(depth.shape, depth, textures,
                                          rain_density, min_length, max_length, device)
        else:
            mask = generate_rain_mask_cpu(depth.shape, depth, textures,
                                          rain_density, min_length, max_length)

        out_name = os.path.splitext(f)[0] + ".png"
        cv2.imwrite(os.path.join(output_dir, out_name), mask)

        if (i + 1) % 20 == 0:
            print(f"    Rain masks: {i + 1}/{len(files)}")

    print("    ✓ Rain mask stage complete.")