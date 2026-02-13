# overlapping_inference.py
"""
Tiled inference with overlap + feathered blending (NO padding).

- Splits each frame into 3x4 overlapping 512x512 tiles (linspace positions).
- Runs inference on the 12 tiles as a batch (B=12, T=1).
- Reconstructs full-res output with weighted blending:
  pixels nearer a tile center get higher weight (2D Hann window).
- Saves stacked comparison images + video, prints timing.
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

CHECKPOINT_PATH = BASE / "checkpoints" / "latest_with_attention.pth"

RAINY_DIR = BASE / "data_after_crapification_per_video" / "scene_004" / "front-forward"
CLEAN_DIR = BASE / "data" / "scene_004" / "images" / "front-forward"

OUTPUT_DIR = BASE / "test_results" / "scene_004_inference_attention"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 10
TILE = 512
ROWS, COLS = 3, 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_frame_fullres(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # (C,H,W)


def get_tile_coords(h: int, w: int, tile: int = 512, rows: int = 3, cols: int = 4):
    """
    Overlapping rows x cols grid, covers full image without padding.
    Uses rounded linspace so last tile aligns to bottom/right.
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


def make_hann_mask(tile: int, device, dtype, eps: float = 1e-6) -> torch.Tensor:
    """
    (1, tile, tile) weight mask: high at center, low near edges.
    """
    w1 = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    w2 = torch.outer(w1, w1).clamp_min(eps)  # avoid exact zeros
    return w2.unsqueeze(0)  # (1,tile,tile)


def tiles_to_full_weighted(tiles: torch.Tensor, coords, h: int, w: int, tile: int = 512) -> torch.Tensor:
    """
    Weighted overlap-add.
    tiles: (N,3,tile,tile)
    """
    acc = torch.zeros((3, h, w), device=tiles.device, dtype=tiles.dtype)
    wgt = torch.zeros((1, h, w), device=tiles.device, dtype=tiles.dtype)

    mask = make_hann_mask(tile, device=tiles.device, dtype=tiles.dtype)  # (1,tile,tile)

    for i, (y0, x0) in enumerate(coords):
        acc[:, y0:y0 + tile, x0:x0 + tile] += tiles[i] * mask
        wgt[:, y0:y0 + tile, x0:x0 + tile] += mask

    return acc / wgt.clamp_min(1e-6)


def chw_to_bgr_uint8(chw: torch.Tensor) -> np.ndarray:
    hwc = chw.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    hwc_u8 = (hwc * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(hwc_u8, cv2.COLOR_RGB2BGR)


def stack_triplet_vertical(rainy_bgr: np.ndarray, out_bgr: np.ndarray, clean_bgr: np.ndarray) -> np.ndarray:
    return np.concatenate([rainy_bgr, out_bgr, clean_bgr], axis=0)


def put_labels(stacked_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    img = stacked_bgr
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.0
    thickness = 2
    cv2.putText(img, "RAINY INPUT", (20, 40), font, scale, (0, 0, 255), thickness)
    cv2.putText(img, "MODEL OUTPUT", (20, h + 40), font, scale, (0, 255, 0), thickness)
    cv2.putText(img, "GROUND TRUTH", (20, 2 * h + 40), font, scale, (255, 0, 0), thickness)
    return img


def main():
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

    print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.6f})\n")

    rainy_files = sorted(RAINY_DIR.glob("*.jpeg"))
    clean_files = sorted(CLEAN_DIR.glob("*.jpeg"))
    n = min(len(rainy_files), len(clean_files))
    rainy_files, clean_files = rainy_files[:n], clean_files[:n]
    if n == 0:
        raise RuntimeError("No frames found in the specified directories.")

    print(f"Using {n} frames.\n")

    rainy0 = load_frame_fullres(rainy_files[0])
    _, h, w = rainy0.shape

    coords = get_tile_coords(h, w, tile=TILE, rows=ROWS, cols=COLS)
    assert len(coords) == 15

    video_path = OUTPUT_DIR / "comparison_stacked_10fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, FPS, (w, h * 3))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter (mp4v).")

    times_ms = []
    print("Running overlapping + weighted-blend inference...")

    for i, (rp, cp) in enumerate(zip(rainy_files, clean_files)):
        rainy_chw = load_frame_fullres(rp)  # CPU
        clean_chw = load_frame_fullres(cp)  # CPU

        if rainy_chw.shape != clean_chw.shape:
            raise ValueError(f"Shape mismatch at frame {i}: {rainy_chw.shape} vs {clean_chw.shape}")

        tiles = []
        for (y0, x0) in coords:
            tiles.append(rainy_chw[:, y0:y0 + TILE, x0:x0 + TILE])
        tiles = torch.stack(tiles, dim=0)  # (12,3,512,512)

        inp = tiles.unsqueeze(1).to(device)  # (12,1,3,512,512)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out_tiles = model(inp)  # (12,1,3,512,512)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        times_ms.append(dt_ms)

        out_tiles = out_tiles.squeeze(1)  # (12,3,512,512)

        out_full = tiles_to_full_weighted(out_tiles, coords, h=h, w=w, tile=TILE).clamp(0, 1).cpu()

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

    avg_ms = float(np.mean(times_ms)) if times_ms else 0.0
    max_ms = float(np.max(times_ms)) if times_ms else 0.0
    print("\n✓ Inference complete.")
    print(f"✓ Saved {n} images to: {OUTPUT_DIR}")
    print(f"✓ Saved video to: {video_path}")
    print("\n" + "=" * 60)
    print(f"Inference time per frame (12 tiles batch): avg={avg_ms:.2f} ms | max={max_ms:.2f} ms")
    print(f"Stage-1 best val loss (combined): {checkpoint['val_loss']:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
