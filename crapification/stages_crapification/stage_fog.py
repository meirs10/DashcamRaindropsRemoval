import cv2
import numpy as np
import os
from pathlib import Path
import torch


def add_atmosphere_cpu(img, depth, fog_density=0.06, airlight=230):
    """CPU version (same behavior as before)"""
    d_norm = depth.astype(float) / 255.0
    transmission = np.exp(-fog_density * (d_norm * 10.0))
    t_map = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)
    fogged = img.astype(float) * t_map + airlight * (1.0 - t_map)
    return np.clip(fogged, 0, 255).astype(np.uint8)


def run_fog_stage(
    img_dir,
    depth_dir,
    output_dir,
    fog_density=0.06,
    airlight=230,
    use_gpu=True,
    batch_size=100,     # new: how many frames to process at once on GPU
):
    """
    Apply depth-based fog to images.

    - If GPU is available and use_gpu=True:
        * Uses batched PyTorch ops on GPU (B x H x W x 3 tensors)
    - Otherwise:
        * Falls back to CPU implementation (original behavior)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(img_dir))
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("    ⚠️ No images found for fog stage.")
        return

    # Check GPU availability
    gpu_available = False
    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        gpu_available = True
        device = 'cuda'
        print(f"    🚀 Using GPU for fog: {torch.cuda.get_device_name(0)}")
    else:
        print("    Using CPU for fog")

    print(f"    Applying fog ({len(image_files)} frames)")

    # ========================
    # CPU PATH (original style)
    # ========================
    if not gpu_available:
        for i, f in enumerate(image_files):
            img_path = os.path.join(img_dir, f)
            img = cv2.imread(img_path)
            if img is None:
                continue

            depth_path = os.path.join(depth_dir, os.path.splitext(f)[0] + ".png")
            if not os.path.exists(depth_path):
                depth_path = os.path.join(depth_dir, f)

            if os.path.exists(depth_path):
                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                result = add_atmosphere_cpu(img, depth, fog_density, airlight)
            else:
                # no depth -> keep original
                result = img

            cv2.imwrite(os.path.join(output_dir, f), result)

            if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
                print(f"    Fog (CPU): {i + 1}/{len(image_files)}")

        print("    ✓ Fog stage complete.")
        return

    # ========================
    # GPU PATH (BATCHED)
    # ========================
    total = len(image_files)
    processed = 0

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_files = image_files[start:start + batch_size]

            imgs = []
            depths = []
            valid_names = []

            # Load batch into CPU memory
            for f in batch_files:
                img_path = os.path.join(img_dir, f)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                depth_path = os.path.join(depth_dir, os.path.splitext(f)[0] + ".png")
                if not os.path.exists(depth_path):
                    depth_path = os.path.join(depth_dir, f)

                if not os.path.exists(depth_path):
                    # No depth -> just save original later on CPU
                    # (to keep logic simple, handle it immediately)
                    cv2.imwrite(os.path.join(output_dir, f), img)
                    processed += 1
                    continue

                depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                if depth is None:
                    cv2.imwrite(os.path.join(output_dir, f), img)
                    processed += 1
                    continue

                # Assume all frames same size in a scene
                imgs.append(img)
                depths.append(depth)
                valid_names.append(f)

            if not imgs:
                continue

            # Stack into [B, H, W, C] and [B, H, W]
            img_batch_np = np.stack(imgs, axis=0).astype(np.float32)  # [B, H, W, 3]
            depth_batch_np = np.stack(depths, axis=0).astype(np.float32)  # [B, H, W]

            # Move to GPU
            img_batch = torch.from_numpy(img_batch_np).to(device, non_blocking=True)
            depth_batch = torch.from_numpy(depth_batch_np).to(device, non_blocking=True)

            # Normalize depth: [0, 1]
            d_norm = depth_batch / 255.0  # [B, H, W]

            # transmission = exp(-fog_density * (d_norm * 10.0))
            transmission = torch.exp(-fog_density * (d_norm * 10.0))  # [B, H, W]

            # Expand to [B, H, W, 3]
            t_map = transmission.unsqueeze(-1)  # [B, H, W, 1]

            # fogged = img * t_map + airlight * (1 - t_map)
            fogged = img_batch * t_map + airlight * (1.0 - t_map)

            fogged = torch.clamp(fogged, 0.0, 255.0).byte().cpu().numpy()  # [B, H, W, 3]

            # Save results
            for out_img, name in zip(fogged, valid_names):
                cv2.imwrite(os.path.join(output_dir, name), out_img)
                processed += 1

            if processed % 20 == 0 or processed == total:
                print(f"    Fog (GPU): {processed}/{total}")

    print("    ✓ Fog stage complete.")
