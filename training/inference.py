# inference.py
"""
Tiled inference / visualization for Stage 1 (full-res frames).

- Loads best_stage1.pth.
- For each 1080p frame: split into 12 overlapping 512x512 crops (3x4), NO padding.
- Run inference on the 12 tiles as a batch (B=12, T=1).
- Reconstruct full-res output by overlap-averaging.
- Save:
  1) Per-frame PNG: rainy (top) | output (mid) | clean (bottom), stacked vertically.
  2) Full video at 10fps with same stacking.
- Print average + max inference time per frame.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "training"))

from model import MobileNetV3UNetConvLSTMVideo

CHECKPOINT_PATH = BASE / "checkpoints" / "best_stage1.pth"

# Example scene/angle to visualize
RAINY_DIR = BASE / "data_after_crapification" / "scene_001" / "front-forward"
CLEAN_DIR = BASE / "data" / "scene_001" / "images" / "front-forward"

OUTPUT_DIR = BASE / "test_results" / "scene_001_comparison_stage1_tiled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 10
TILE = 512
ROWS, COLS = 3, 4  # 3x4 = 12 tiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_frame_fullres(path: Path) -> torch.Tensor:
    """Read image as float32 RGB in [0,1], return (C,H,W) tensor on CPU."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # (C,H,W)


def get_tile_coords(h: int, w: int, tile: int = 512, rows: int = 3, cols: int = 4):
    """
    Compute (y0,x0) for an overlapping rows x cols grid, covering full image
    without padding. Uses rounded linspace so last tile aligns to bottom/right.
    """
    if h < tile or w < tile:
        raise ValueError(f"Frame too small for tile={tile}: got {w}x{h}")

    xs = np.linspace(0, w - tile, cols).round().astype(int).tolist()
    ys = np.linspace(0, h - tile, rows).round().astype(int).tolist()

    coords = []
    for y in ys:
        for x in xs:
            coords.append((y, x))
    return coords


def tiles_to_full(tiles: torch.Tensor, coords, h: int, w: int, tile: int = 512) -> torch.Tensor:
    """
    Reconstruct full-res (3,H,W) from tile outputs (N,3,tile,tile) by overlap averaging.
    """
    # Accumulate on same device as tiles
    acc = torch.zeros((3, h, w), device=tiles.device, dtype=tiles.dtype)
    wgt = torch.zeros((1, h, w), device=tiles.device, dtype=tiles.dtype)

    for i, (y0, x0) in enumerate(coords):
        acc[:, y0:y0 + tile, x0:x0 + tile] += tiles[i]
        wgt[:, y0:y0 + tile, x0:x0 + tile] += 1.0

    acc = acc / wgt.clamp_min(1.0)
    return acc


def chw_to_bgr_uint8(chw: torch.Tensor) -> np.ndarray:
    """(C,H,W) float in [0,1] -> BGR uint8 HxWx3 for OpenCV."""
    hwc = chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    hwc_u8 = (hwc * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(hwc_u8, cv2.COLOR_RGB2BGR)


def stack_triplet_vertical(rainy_bgr: np.ndarray, out_bgr: np.ndarray, clean_bgr: np.ndarray) -> np.ndarray:
    """Stack 3 images vertically (same W,H)."""
    return np.concatenate([rainy_bgr, out_bgr, clean_bgr], axis=0)


def put_labels(stacked_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Put labels on the stacked image:
    Top block: RAINY INPUT, middle: MODEL OUTPUT, bottom: GROUND TRUTH.
    Each block is height h, width w.
    """
    img = stacked_bgr
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2

    # Top
    cv2.putText(img, "RAINY INPUT", (20, 40), font, scale, (0, 0, 255), thickness)
    # Middle
    cv2.putText(img, "MODEL OUTPUT", (20, h + 40), font, scale, (0, 255, 0), thickness)
    # Bottom
    cv2.putText(img, "GROUND TRUTH", (20, 2 * h + 40), font, scale, (255, 0, 0), thickness)

    return img


def main():
    # ----- Load checkpoint -----
    print("\nLoading Stage-1 checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.6f})\n"
    )

    # ----- Load some frames -----
    rainy_files = sorted(RAINY_DIR.glob("*.jpeg"))
    clean_files = sorted(CLEAN_DIR.glob("*.jpeg"))
    n = min(len(rainy_files), len(clean_files))
    rainy_files, clean_files = rainy_files[:n], clean_files[:n]
    if not rainy_files or not clean_files:
        raise RuntimeError("No frames found in the specified directories.")
    n = min(len(rainy_files), len(clean_files))
    rainy_files, clean_files = rainy_files[:n], clean_files[:n]

    print(f"Using {n} frames for visualization.\n")

    # Read one frame to set video writer shape
    rainy0 = load_frame_fullres(rainy_files[0])
    c, h, w = rainy0.shape
    if (h, w) != (1080, 1920):
        print(f"[Note] Frame is {w}x{h}, not exactly 1920x1080. Using its native size.\n")

    coords = get_tile_coords(h, w, tile=TILE, rows=ROWS, cols=COLS)
    assert len(coords) == ROWS * COLS

    # ----- Video writer (stacked vertically: 3H x W) -----
    video_path = OUTPUT_DIR / "comparison_stacked_10fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (w, h * 3))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter (mp4v). Try a different codec/container.")

    # ----- Inference loop -----
    times_ms = []

    print("Running tiled inference...")
    for i, (rp, cp) in enumerate(zip(rainy_files, clean_files)):
        rainy_chw = load_frame_fullres(rp)  # CPU
        clean_chw = load_frame_fullres(cp)  # CPU

        if rainy_chw.shape != clean_chw.shape:
            raise ValueError(f"Shape mismatch rainy vs clean at frame {i}: {rainy_chw.shape} vs {clean_chw.shape}")

        # Build tile batch (N,3,512,512) on CPU then move once
        tiles = []
        for (y0, x0) in coords:
            tiles.append(rainy_chw[:, y0:y0 + TILE, x0:x0 + TILE])
        tiles = torch.stack(tiles, dim=0)  # (N,3,512,512)

        # Model expects (B,T,C,H,W). We'll do T=1.
        inp = tiles.unsqueeze(1).to(device)  # (N,1,3,512,512)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out_tiles = model(inp)  # (N,1,3,512,512)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        times_ms.append(dt_ms)

        out_tiles = out_tiles.squeeze(1)  # (N,3,512,512)

        # Reconstruct full-res output (3,H,W) on device, then bring to CPU
        out_full = tiles_to_full(out_tiles, coords, h=h, w=w, tile=TILE).clamp(0, 1).cpu()

        # ----- Save per-frame stacked image + write video -----
        rainy_bgr = chw_to_bgr_uint8(rainy_chw)
        out_bgr = chw_to_bgr_uint8(out_full)
        clean_bgr = chw_to_bgr_uint8(clean_chw)

        stacked = stack_triplet_vertical(rainy_bgr, out_bgr, clean_bgr)
        stacked = put_labels(stacked, w=w, h=h)

        out_path = OUTPUT_DIR / f"comparison_frame_{i:04d}_stacked.png"
        cv2.imwrite(str(out_path), stacked)
        writer.write(stacked)

        print(f"Frame {i+1}/{n}: {dt_ms:.2f} ms")

    writer.release()
    print("\n✓ Inference complete.")
    print(f"✓ Saved {n} images to: {OUTPUT_DIR}")
    print(f"✓ Saved video to: {video_path}")

    # ----- Timing stats -----
    avg_ms = float(np.mean(times_ms)) if times_ms else 0.0
    max_ms = float(np.max(times_ms)) if times_ms else 0.0
    print("\n" + "=" * 60)
    print(f"Inference time per frame (12 tiles batch): avg={avg_ms:.2f} ms | max={max_ms:.2f} ms")
    print(f"Stage-1 best val loss (combined): {checkpoint['val_loss']:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
