# crapification/stage_droplets.py

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path


def gaussian_blur_torch(img_tensor, kernel_size, sigma=0):
    """
    Fast Gaussian blur using PyTorch (GPU-accelerated).

    Supports:
        - img_tensor: [C, H, W]
        - img_tensor: [B, C, H, W]
    """
    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    if img_tensor.dim() == 3:
        img = img_tensor.unsqueeze(0)  # [1, C, H, W]
    else:
        img = img_tensor  # [B, C, H, W]

    b, c, h, w = img.shape

    x = torch.arange(kernel_size, device=img.device, dtype=torch.float32)
    x = x - kernel_size // 2
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()  # [K]

    padding = kernel_size // 2

    # Depthwise separable blur (horizontal then vertical)
    kernel_x = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)  # [C,1,1,K]
    kernel_y = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)  # [C,1,K,1]

    # Horizontal
    blurred = F.conv2d(img, kernel_x, padding=(0, padding), groups=c)
    # Vertical
    blurred = F.conv2d(blurred, kernel_y, padding=(padding, 0), groups=c)

    if img_tensor.dim() == 3:
        return blurred.squeeze(0)  # [C, H, W]
    return blurred  # [B, C, H, W]


def add_camera_sensor_water_gpu(
        img,
        device='cuda',
        seed=None,
        n_large_bokeh=22,
        n_medium_bokeh=35,
        droplet_positions=None  # Pre-defined STATIC positions
):
    """
    Original single-frame GPU-accelerated water effects.

    Args:
        droplet_positions: If provided, use these EXACT positions (no randomness)
                          Format: list of (cx, cy, radius, opacity, sigma_factor, brightness, is_large)
    """
    if seed is not None and droplet_positions is None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if n_large_bokeh == 0 and n_medium_bokeh == 0 and droplet_positions is None:
        return img

    h, w = img.shape[:2]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device)  # [3,H,W]

    blur_extreme = gaussian_blur_torch(img_tensor, 91)
    blur_heavy = gaussian_blur_torch(img_tensor, 71)

    result = img_tensor.clone()

    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # If positions provided, use EXACT same positions
    if droplet_positions is not None:
        for (cx, cy, radius, opacity, sigma_factor, brightness, is_large) in droplet_positions:
            dist = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            circle = torch.exp(-(dist ** 2) / (2 * (radius * sigma_factor) ** 2))
            circle = torch.clamp(circle, 0, 1) * opacity
            circle_3ch = circle.unsqueeze(0)  # [1,H,W]

            blur_to_use = blur_extreme if is_large else blur_heavy  # [3,H,W]
            result = result * (1 - circle_3ch) + blur_to_use * brightness * circle_3ch

    else:
        # Random positions (training mode)
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


def generate_static_droplet_positions(h, w, n_large, n_medium, seed):
    """
    Generate STATIC droplet positions that DON'T CHANGE across frames.

    Returns:
        List of (cx, cy, radius, opacity, sigma_factor, brightness, is_large)
        These positions are used for ALL frames - droplets stay in exact same spot!
    """
    np.random.seed(seed)

    positions = []

    # Large droplets - FIXED positions
    for _ in range(n_large):
        positions.append((
            float(np.random.randint(int(w * 0.05), int(w * 0.95))),  # cx - FIXED
            float(np.random.randint(int(h * 0.05), int(h * 0.95))),  # cy - FIXED
            float(np.random.uniform(80, 180)),  # radius - FIXED
            float(np.random.uniform(0.85, 1.0)),  # opacity - FIXED
            float(np.random.uniform(0.35, 0.45)),  # sigma - FIXED
            float(np.random.uniform(1.25, 1.55)),  # brightness - FIXED
            True  # is_large
        ))

    # Medium droplets - FIXED positions
    for _ in range(n_medium):
        positions.append((
            float(np.random.randint(0, w)),  # cx - FIXED
            float(np.random.randint(0, h)),  # cy - FIXED
            float(np.random.uniform(40, 95)),  # radius - FIXED
            float(np.random.uniform(0.8, 1.0)),  # opacity - FIXED
            float(np.random.uniform(0.32, 0.42)),  # sigma - FIXED
            float(np.random.uniform(1.2, 1.4)),  # brightness - FIXED
            False  # is_large
        ))

    return positions


def _apply_droplets_batch_persistent(img_batch_np, static_positions, device):
    """
    Batched droplet application for PERSISTENT droplets (same positions every frame).

    img_batch_np: [B, H, W, 3] RGB uint8
    static_positions: list of (cx, cy, radius, opacity, sigma_factor, brightness, is_large)
    """
    # [B,H,W,3] -> [B,3,H,W]
    img_batch = torch.from_numpy(img_batch_np).permute(0, 3, 1, 2).float().to(device)  # [B,3,H,W]

    b, c, h, w = img_batch.shape

    # Compute heavy blurs for the batch
    blur_extreme = gaussian_blur_torch(img_batch, 91)  # [B,3,H,W]
    blur_heavy = gaussian_blur_torch(img_batch, 71)    # [B,3,H,W]

    result = img_batch.clone()

    # Shared grids for all frames
    y_grid, x_grid = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )  # [H,W]

    for (cx, cy, radius, opacity, sigma_factor, brightness, is_large) in static_positions:
        dist = torch.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)  # [H,W]
        circle = torch.exp(-(dist ** 2) / (2 * (radius * sigma_factor) ** 2))
        circle = torch.clamp(circle, 0., 1.) * opacity  # [H,W]

        # Broadcast to [B,3,H,W]
        circle_b = circle.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        blur_to_use = blur_extreme if is_large else blur_heavy  # [B,3,H,W]
        result = result * (1 - circle_b) + blur_to_use * brightness * circle_b

    result = torch.clamp(result, 0, 255)
    result_np = result.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # [B,H,W,3]

    return result_np


def run_droplet_stage(
        input_dir,
        output_dir,
        mask_dir=None,
        seed=None,
        intensity='heavy',
        use_gpu=True,
        persistent=False,
        batch_size=68,  # NEW: batch size for batched persistent mode
):
    """
    Apply camera sensor water droplets.

    Args:
        persistent (bool): If True, droplets stay in EXACT same positions across ALL frames
                          If False, droplets randomize per frame (training mode)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check GPU
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

    if not files:
        print("    ⚠️ No images found for droplet stage.")
        return

    configs = {
        'light': {'n_large_bokeh': 10, 'n_medium_bokeh': 18},
        'medium': {'n_large_bokeh': 16, 'n_medium_bokeh': 28},
        'heavy': {'n_large_bokeh': 28, 'n_medium_bokeh': 45},
        'extreme': {'n_large_bokeh': 40, 'n_medium_bokeh': 60}
    }

    config = configs.get(intensity, configs['heavy'])

    mode_str = "PERSISTENT (STATIC)" if persistent else "RANDOM"
    print(f"    Applying droplets - {intensity} - {mode_str} ({len(files)} frames)")

    base_seed = np.random.randint(0, 10_000) if seed is None else seed

    # ============================
    # PERSISTENT + GPU: BATCHED
    # ============================
    if persistent and device == 'cuda':
        # Get dimensions from first image
        first_img = cv2.imread(os.path.join(input_dir, files[0]))
        h, w = first_img.shape[:2]

        print(f"    Generating STATIC droplet positions...")
        static_positions = generate_static_droplet_positions(
            h, w,
            config['n_large_bokeh'],
            config['n_medium_bokeh'],
            base_seed
        )
        print(f"    ✓ Generated {len(static_positions)} droplets (will stay in same spots)")

        total = len(files)
        processed = 0

        for start in range(0, total, batch_size):
            batch_files = files[start:start + batch_size]
            img_batch_list = []

            # Load RGB images
            for fname in batch_files:
                img_path = os.path.join(input_dir, fname)
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    img_batch_list.append(None)
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_batch_list.append(img_rgb)

            # Filter out any failed loads
            valid_indices = [i for i, im in enumerate(img_batch_list) if im is not None]
            if not valid_indices:
                continue

            imgs_np = np.stack([img_batch_list[i] for i in valid_indices], axis=0)  # [B_valid,H,W,3]

            # Apply droplets in batch
            result_batch = _apply_droplets_batch_persistent(imgs_np, static_positions, device=device)  # [B_valid,H,W,3]

            # Write back results
            v_idx = 0
            for local_idx, fname in enumerate(batch_files):
                if img_batch_list[local_idx] is None:
                    continue

                result_rgb = result_batch[v_idx]
                v_idx += 1
                out_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(output_dir, fname), out_bgr)
                processed += 1

                if processed % 10 == 0 or processed == total:
                    print(f"    Droplets (GPU batched): {processed}/{total}")

        print(f"    ✓ Droplets complete ({mode_str})")
        return

    # ============================
    # ORIGINAL PER-FRAME LOGIC
    # (CPU OR NON-PERSISTENT)
    # ============================
    if persistent:
        # Persistent but CPU: generate same positions and apply per-frame
        first_img = cv2.imread(os.path.join(input_dir, files[0]))
        h, w = first_img.shape[:2]
        print(f"    Generating STATIC droplet positions...")
        static_positions = generate_static_droplet_positions(
            h, w,
            config['n_large_bokeh'],
            config['n_medium_bokeh'],
            base_seed
        )
        print(f"    ✓ Generated {len(static_positions)} droplets (will stay in same spots)")
    else:
        static_positions = None

    # Process ALL frames one-by-one (random or CPU)
    for i, fname in enumerate(files):
        img_path = os.path.join(input_dir, fname)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if persistent:
            result = add_camera_sensor_water_gpu(
                img,
                device=device,
                seed=None,
                droplet_positions=static_positions  # SAME positions for all frames
            )
        else:
            # Random per frame (training mode)
            var = {k: max(0, v + np.random.randint(-3, 4)) for k, v in config.items()}
            result = add_camera_sensor_water_gpu(
                img,
                device=device,
                seed=base_seed + i // 4,
                **var
            )

        cv2.imwrite(
            os.path.join(output_dir, fname),
            cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        )

        if (i + 1) % 10 == 0 or (i + 1) == len(files):
            print(f"    Droplets: {i + 1}/{len(files)}")

    print(f"    ✓ Droplets complete ({mode_str})")


if __name__ == "__main__":
    run_droplet_stage(
        input_dir="path/to/input",
        output_dir="path/to/output",
        seed=42,
        intensity='heavy',
        use_gpu=True,
        persistent=True  # ← STATIC droplets, batched on GPU
    )
