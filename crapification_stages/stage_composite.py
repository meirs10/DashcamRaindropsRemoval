import cv2
import numpy as np
import os
from pathlib import Path
import torch


def composite_gpu(base, mask, rain_brightness=1.3, device='cuda'):
    """GPU-accelerated compositing using PyTorch"""
    # Convert to tensors [H, W, C]
    base_tensor = torch.from_numpy(base).float().to(device)
    mask_tensor = torch.from_numpy(mask).float().to(device)

    # Convert grayscale mask to BGR
    rain_tensor = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)

    # Normalize
    base_norm = base_tensor / 255.0
    rain_norm = (rain_tensor / 255.0) * rain_brightness

    # Scene-aware lighting
    luma = (base_tensor[:, :, 2] * 0.299 +
            base_tensor[:, :, 1] * 0.587 +
            base_tensor[:, :, 0] * 0.114) / 255.0

    luma_weight = 0.4 + 0.6 * luma
    luma_3ch = luma_weight.unsqueeze(-1).repeat(1, 1, 3)
    rain_weighted = rain_norm * luma_3ch

    # Screen blend
    final = 1.0 - (1.0 - base_norm) * (1.0 - rain_weighted)

    result = torch.clamp(final * 255.0, 0, 255)
    return result.cpu().numpy().astype(np.uint8)


def composite_cpu(base, mask, rain_brightness=1.3):
    """CPU fallback"""
    rain = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    base_f = base.astype(float) / 255.0
    rain_f = (rain.astype(float) / 255.0) * rain_brightness

    luma = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    rain_f *= (0.4 + 0.6 * luma[..., None])

    final = 1.0 - (1.0 - base_f) * (1.0 - rain_f)
    return (final * 255).astype(np.uint8)


def run_composite_stage(fog_dir, rain_dir, output_dir,
                        rain_brightness=1.3, use_gpu=True):
    """
    Composite rain onto fogged images.
    Only saves final composited images.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check GPU
    gpu_available = False
    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        gpu_available = True
        device = 'cuda'
        print(f"    🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("    Using CPU")

    files = sorted(os.listdir(fog_dir))
    print(f"    Compositing rain ({len(files)} frames)")

    for i, f in enumerate(files):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        base = cv2.imread(os.path.join(fog_dir, f))
        if base is None:
            continue

        mask_path = os.path.join(rain_dir, os.path.splitext(f)[0] + ".png")
        if not os.path.exists(mask_path):
            # No rain mask, just copy fogged image
            cv2.imwrite(os.path.join(output_dir, f), base)
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape[:2] != base.shape[:2]:
            mask = cv2.resize(mask, (base.shape[1], base.shape[0]))

        if gpu_available:
            result = composite_gpu(base, mask, rain_brightness, device)
        else:
            result = composite_cpu(base, mask, rain_brightness)

        # ONLY save the final composited image
        cv2.imwrite(os.path.join(output_dir, f), result)

        if (i + 1) % 20 == 0:
            print(f"    Composited: {i + 1}/{len(files)}")

    print("    ✓ Composite stage complete.")