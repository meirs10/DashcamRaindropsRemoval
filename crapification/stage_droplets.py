import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def gaussian_blur_torch(img_tensor, kernel_size, sigma=0):
    """Fast Gaussian blur using PyTorch (GPU-accelerated)"""
    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    x = torch.arange(kernel_size, device=img_tensor.device, dtype=torch.float32)
    x = x - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()

    padding = kernel_size // 2
    blurred = F.conv2d(
        img_tensor.unsqueeze(0),
        kernel.view(1, 1, 1, -1).repeat(img_tensor.shape[0], 1, 1, 1),
        padding=(0, padding),
        groups=img_tensor.shape[0]
    )
    blurred = F.conv2d(
        blurred,
        kernel.view(1, 1, -1, 1).repeat(img_tensor.shape[0], 1, 1, 1),
        padding=(padding, 0),
        groups=img_tensor.shape[0]
    )

    return blurred.squeeze(0)


def add_camera_sensor_water_gpu(
        img,
        device='cuda',
        seed=None,
        n_large_bokeh=22,
        n_medium_bokeh=35
):
    """GPU-accelerated water effects - NO RINGS, just clean bokeh"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # If no droplets, return original
    if n_large_bokeh == 0 and n_medium_bokeh == 0:
        return img

    h, w = img.shape[:2]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)

    blur_extreme = gaussian_blur_torch(img_tensor, 91)
    blur_heavy = gaussian_blur_torch(img_tensor, 71)

    result = img_tensor.clone()

    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # Large bokeh
    for _ in range(n_large_bokeh):
        cx = np.random.randint(int(w * 0.05), int(w * 0.95))
        cy = np.random.randint(int(h * 0.05), int(h * 0.95))
        radius = np.random.uniform(80, 180)
        opacity = np.random.uniform(0.85, 1.0)

        dist = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        sigma_factor = np.random.uniform(0.35, 0.45)
        circle = torch.exp(-(dist ** 2) / (2 * (radius * sigma_factor) ** 2))
        circle = torch.clamp(circle, 0, 1) * opacity

        circle_3ch = circle.unsqueeze(0)
        brightness = np.random.uniform(1.25, 1.55)
        result = result * (1 - circle_3ch) + blur_extreme * brightness * circle_3ch

    # Medium bokeh
    for _ in range(n_medium_bokeh):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        radius = np.random.uniform(40, 95)
        opacity = np.random.uniform(0.8, 1.0)

        dist = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        sigma_factor = np.random.uniform(0.32, 0.42)
        circle = torch.exp(-(dist ** 2) / (2 * (radius * sigma_factor) ** 2))
        circle = torch.clamp(circle, 0, 1) * opacity

        circle_3ch = circle.unsqueeze(0)
        brightness = np.random.uniform(1.2, 1.4)
        result = result * (1 - circle_3ch) + blur_heavy * brightness * circle_3ch

    result = torch.clamp(result, 0, 255)
    result_np = result.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    return result_np


def run_droplet_stage(
        input_dir,
        output_dir,
        mask_dir=None,
        seed=None,
        intensity='heavy',
        use_gpu=True
):
    """
    Fast camera sensor water with optional GPU acceleration.
    Only saves final crapified images (masks optional).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check GPU availability
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
        print(f"    🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("    Using CPU")

    files = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )

    configs = {
        'light': {
            'n_large_bokeh': 10,
            'n_medium_bokeh': 18
        },
        'medium': {
            'n_large_bokeh': 16,
            'n_medium_bokeh': 28
        },
        'heavy': {
            'n_large_bokeh': 28,
            'n_medium_bokeh': 45
        },
        'extreme': {
            'n_large_bokeh': 40,
            'n_medium_bokeh': 60
        }
    }

    config = configs.get(intensity, configs['heavy'])

    print(f"    Applying camera sensor water - {intensity} ({len(files)} frames)")

    base_seed = np.random.randint(0, 10_000) if seed is None else seed

    for i, fname in enumerate(files):
        img_path = os.path.join(input_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        var = {k: max(0, v + np.random.randint(-3, 4)) for k, v in config.items()}

        result = add_camera_sensor_water_gpu(
            img, device=device, seed=base_seed + i // 4, **var
        )

        cv2.imwrite(
            os.path.join(output_dir, fname),
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"    Water: {i + 1}/{len(files)}")

    print("    ✓ Sensor water complete.")


if __name__ == "__main__":
    run_droplet_stage(
        input_dir="path/to/clear/images",
        output_dir="path/to/output/water",
        mask_dir=None,
        seed=42,
        intensity='heavy',
        use_gpu=True
    )