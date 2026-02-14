import cv2
import numpy as np
import os
from pathlib import Path
import torch


def composite_gpu(base, mask, rain_brightness=1.3, device='cuda'):
    """GPU-accelerated compositing using PyTorch (single frame)."""
    # Convert to tensors [H, W, C]
    base_tensor = torch.from_numpy(base).float().to(device)
    mask_tensor = torch.from_numpy(mask).float().to(device)

    # Convert grayscale mask to BGR
    rain_tensor = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)

    # Normalize
    base_norm = base_tensor / 255.0
    rain_norm = (rain_tensor / 255.0) * rain_brightness

    # Scene-aware lighting (BGR -> luma)
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
    """CPU fallback (single frame)."""
    rain = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    base_f = base.astype(float) / 255.0
    rain_f = (rain.astype(float) / 255.0) * rain_brightness

    luma = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    rain_f *= (0.4 + 0.6 * luma[..., None])

    final = 1.0 - (1.0 - base_f) * (1.0 - rain_f)
    return (final * 255).astype(np.uint8)


def run_composite_stage(
    fog_dir,
    rain_dir,
    output_dir,
    rain_brightness=1.3,
    use_gpu=True,
    batch_size=68,  # NEW: batched GPU compositing
):
    """
    Composite rain onto fogged images.
    Only saves final composited images.

    GPU mode:
      - Loads fog + mask for a batch of frames
      - Runs compositing in one batched pass on GPU
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Check GPU
    gpu_available = False
    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        gpu_available = True
        device = 'cuda'
        print(f"    🚀 Using GPU for composite: {torch.cuda.get_device_name(0)}")
    else:
        print("    Using CPU for composite")

    files = sorted(os.listdir(fog_dir))
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print("    ⚠️ No images found for composite stage.")
        return

    print(f"    Compositing rain ({len(image_files)} frames)")

    # ========================
    # CPU PATH (original style)
    # ========================
    if not gpu_available:
        for i, f in enumerate(image_files):
            base = cv2.imread(os.path.join(fog_dir, f))
            if base is None:
                continue

            mask_path = os.path.join(rain_dir, os.path.splitext(f)[0] + ".png")
            if not os.path.exists(mask_path):
                # No rain mask, just copy fogged image
                cv2.imwrite(os.path.join(output_dir, f), base)
                continue

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                cv2.imwrite(os.path.join(output_dir, f), base)
                continue

            if mask.shape[:2] != base.shape[:2]:
                mask = cv2.resize(mask, (base.shape[1], base.shape[0]))

            result = composite_cpu(base, mask, rain_brightness)

            cv2.imwrite(os.path.join(output_dir, f), result)

            if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
                print(f"    Composited (CPU): {i + 1}/{len(image_files)}")

        print("    ✓ Composite stage complete.")
        return

    # ========================
    # GPU PATH (BATCHED)
    # ========================
    total = len(image_files)
    processed = 0

    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch_files = image_files[start:start + batch_size]

            base_list = []
            mask_list = []
            out_names = []

            # Load a batch into CPU memory
            for f in batch_files:
                base_path = os.path.join(fog_dir, f)
                base = cv2.imread(base_path)
                if base is None:
                    continue

                mask_path = os.path.join(rain_dir, os.path.splitext(f)[0] + ".png")
                if not os.path.exists(mask_path):
                    # No rain mask, just copy fogged image immediately
                    cv2.imwrite(os.path.join(output_dir, f), base)
                    processed += 1
                    continue

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    cv2.imwrite(os.path.join(output_dir, f), base)
                    processed += 1
                    continue

                if mask.shape[:2] != base.shape[:2]:
                    mask = cv2.resize(mask, (base.shape[1], base.shape[0]))

                base_list.append(base.astype(np.float32))
                mask_list.append(mask.astype(np.float32))
                out_names.append(f)

            if not base_list:
                continue

            # Stack to [B, H, W, 3] and [B, H, W]
            base_batch_np = np.stack(base_list, axis=0)   # [B,H,W,3]
            mask_batch_np = np.stack(mask_list, axis=0)   # [B,H,W]

            # Move to GPU
            base_batch = torch.from_numpy(base_batch_np).to(device, non_blocking=True)
            mask_batch = torch.from_numpy(mask_batch_np).to(device, non_blocking=True)

            # Normalize
            base_norm = base_batch / 255.0                          # [B,H,W,3]
            rain_norm = (mask_batch / 255.0) * rain_brightness      # [B,H,W]

            # Grayscale luma from BGR
            # base_batch is [B,H,W,3] in BGR order
            b = base_batch[..., 0]
            g = base_batch[..., 1]
            r = base_batch[..., 2]
            luma = (r * 0.299 + g * 0.587 + b * 0.114) / 255.0      # [B,H,W]

            luma_weight = 0.4 + 0.6 * luma                          # [B,H,W]
            luma_3ch = luma_weight.unsqueeze(-1)                     # [B,H,W,1]

            # Expand rain to 3 channels and weight by luma
            rain_3ch = rain_norm.unsqueeze(-1)                       # [B,H,W,1]
            rain_weighted = rain_3ch * luma_3ch                      # [B,H,W,1]
            rain_weighted = rain_weighted.expand_as(base_norm)       # [B,H,W,3]

            # Screen blend
            final = 1.0 - (1.0 - base_norm) * (1.0 - rain_weighted)  # [B,H,W,3]

            out_batch = torch.clamp(final * 255.0, 0.0, 255.0).byte().cpu().numpy()

            # Save results
            for out_img, name in zip(out_batch, out_names):
                cv2.imwrite(os.path.join(output_dir, name), out_img)
                processed += 1

            if processed % 20 == 0 or processed == total:
                print(f"    Composited (GPU): {processed}/{total}")

    print("    ✓ Composite stage complete.")
