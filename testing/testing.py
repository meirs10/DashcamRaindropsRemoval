# testing/testing_tiled.py
"""
Stage 2 TEST script – tiled, same as overlapping_inference.py

- Uses scene_split.json to get test scenes.
- For each test scene and angle:
    * Loads full-res rainy & clean frames
    * Splits rainy frame into overlapping 512x512 tiles (3 x 5 grid)
    * Runs MobileNetV3UNetConvLSTMVideo on tiles (B = num_tiles, T = 1)
    * Reconstructs full-res output with Hann-weighted blending (exactly as inference)
    * Computes:
        - Full-frame MAE, MSE, PSNR between output and clean
        - Training-style CombinedVideoLoss on the tiles
- Aggregates metrics over ALL test frames.
- Saves:
    * metrics_summary.json + metrics_summary.txt
    * A few qualitative samples [rainy | output | clean].
"""

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "training"))

from model import MobileNetV3UNetConvLSTMVideo
from losses import CombinedVideoLoss

# ------------------------------------------------------------------------------------
# Paths and constants
# ------------------------------------------------------------------------------------
CLEAN_DATA = BASE / "data"
RAINY_DATA = BASE / "test_data_after_crapification"
SPLIT_FILE = BASE / "crapification" / "scene_split.json"

# Possible checkpoint locations
CHECKPOINT_PATH_ROOT = BASE / "checkpoints" / "best_stage2.pth"
CHECKPOINT_PATH_STAGE2 = BASE / "checkpoints" / "stage2" / "best_stage2.pth"

OUTPUT_DIR = BASE / "test_results" / "stage2_test_tiled"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tiling parameters – MUST match overlapping_inference.py
TILE = 512
ROWS, COLS = 3, 5   # 3 x 5 = 15 tiles

# Same max weights as train_stage_2.py
SSIM_MAX = 0.15
EDGE_MAX = 0.10
PERCEPTUAL_MAX = 0.05

ANGLES = [
    "front-forward",
    "left-backward",
    "left-forward",
    "right-backward",
    "right-forward",
]

# How many visual samples to save
SAMPLE_LIMIT = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ------------------------------------------------------------------------------------
# Helper functions – copied from overlapping_inference.py (or equivalent)
# ------------------------------------------------------------------------------------
def load_frame_fullres(path: Path) -> torch.Tensor:
    """Load image as float32 RGB in [0,1], shape (C,H,W)."""
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
    """(1, tile, tile) weight mask: high at center, low near edges."""
    w1 = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    w2 = torch.outer(w1, w1).clamp_min(eps)  # avoid exact zeros
    return w2.unsqueeze(0)  # (1,tile,tile)


def tiles_to_full_weighted(
    tiles: torch.Tensor,
    coords,
    h: int,
    w: int,
    tile: int = 512,
) -> torch.Tensor:
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
    """Convert (C,H,W) float [0,1] tensor to OpenCV BGR uint8 image."""
    chw = chw.clamp(0, 1)
    hwc = chw.permute(1, 2, 0).cpu().numpy()
    hwc_u8 = (hwc * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(hwc_u8, cv2.COLOR_RGB2BGR)


# ------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------
def fullframe_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    pred, target: (3,H,W), float in [0,1], same device
    Returns (mae, mse, psnr)
    """
    diff = pred - target
    mae = diff.abs().mean().item()
    mse = (diff ** 2).mean().item()
    eps = 1e-8
    psnr = 10.0 * math.log10(1.0 / (mse + eps))
    return mae, mse, psnr


# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    # ---------------- Scene split ----------------
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file not found: {SPLIT_FILE}")

    with open(SPLIT_FILE, "r") as f:
        split_info = json.load(f)

    test_scenes = split_info.get("test", [])
    print(f"TEST scenes from split file: {len(test_scenes)} -> {test_scenes}")

    # ---------------- Model + checkpoint ----------------
    if CHECKPOINT_PATH_ROOT.exists():
        ckpt_path = CHECKPOINT_PATH_ROOT
    elif CHECKPOINT_PATH_STAGE2.exists():
        ckpt_path = CHECKPOINT_PATH_STAGE2
    else:
        raise FileNotFoundError(
            "Checkpoint not found at either:\n"
            f"  {CHECKPOINT_PATH_ROOT}\n"
            f"  {CHECKPOINT_PATH_STAGE2}"
        )

    print(f"\nLoading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=True,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    best_val_loss = checkpoint.get("val_loss", float("nan"))

    print(
        f"✓ Loaded epoch {checkpoint.get('epoch', 'N/A')} "
        f"(val_loss={best_val_loss:.6f})\n"
    )

    # Same CombinedVideoLoss as Stage 2, but with final max weights
    criterion = CombinedVideoLoss(
        alpha=1.0,
        beta=SSIM_MAX,
        gamma=EDGE_MAX,
        delta=0.0,
        epsilon=PERCEPTUAL_MAX,
    ).to(device)

    print("Using CombinedVideoLoss for tiles with final max weights:")
    print(f"  alpha (pixel)   = 1.0")
    print(f"  beta  (SSIM)    = {SSIM_MAX}")
    print(f"  gamma (Edge)    = {EDGE_MAX}")
    print(f"  epsilon (Perc.) = {PERCEPTUAL_MAX}")
    print(f"  delta (Temporal)= 0.0\n")

    # ---------------- Global accumulators ----------------
    total_frames = 0
    sum_mae = 0.0
    sum_mse = 0.0
    sum_psnr = 0.0

    sum_total_max = 0.0   # sum over tiles
    total_tiles = 0       # how many tiles we evaluated

    sample_count = 0      # how many qualitative examples saved

    # ---------------- Iterate over test scenes and angles ----------------
    for scene_num in test_scenes:
        scene_name = f"scene_{scene_num:03d}"
        print(f"\n=== Scene {scene_name} ===")

        for angle in ANGLES:
            clean_dir = CLEAN_DATA / scene_name / "images" / angle
            rainy_dir = RAINY_DATA / scene_name / angle

            if not clean_dir.exists() or not rainy_dir.exists():
                # no such angle for this scene in test data
                continue

            clean_files = sorted(clean_dir.glob("*.jpeg"))
            rainy_files = sorted(rainy_dir.glob("*.jpeg"))
            n = min(len(clean_files), len(rainy_files))
            if n == 0:
                continue

            print(f"  Angle {angle}: {n} frames")

            for i in range(n):
                cp = clean_files[i]
                rp = rainy_files[i]

                # ----- Load full-res frames (CPU) -----
                clean_chw = load_frame_fullres(cp)   # (3,H,W), float in [0,1]
                rainy_chw = load_frame_fullres(rp)

                if clean_chw.shape != rainy_chw.shape:
                    raise ValueError(
                        f"Shape mismatch at {scene_name} {angle} frame {i}: "
                        f"{clean_chw.shape} vs {rainy_chw.shape}"
                    )

                _, H, W = rainy_chw.shape

                # ----- Build tiles exactly as in overlapping_inference.py -----
                coords = get_tile_coords(H, W, tile=TILE, rows=ROWS, cols=COLS)
                tiles_rainy = []
                tiles_clean = []
                for (y0, x0) in coords:
                    tiles_rainy.append(rainy_chw[:, y0:y0 + TILE, x0:x0 + TILE])
                    tiles_clean.append(clean_chw[:, y0:y0 + TILE, x0:x0 + TILE])

                tiles_rainy = torch.stack(tiles_rainy, dim=0)  # (N,3,512,512)
                tiles_clean = torch.stack(tiles_clean, dim=0)  # (N,3,512,512)
                N_tiles = tiles_rainy.size(0)

                # ----- Forward through model: (N,1,3,512,512) -----
                inp = tiles_rainy.unsqueeze(1).to(device)   # (N,1,3,512,512)
                target_tiles = tiles_clean.unsqueeze(1).to(device)

                with torch.no_grad():
                    out_tiles = model(inp)                 # (N,1,3,512,512)

                out_tiles = out_tiles.squeeze(1)           # (N,3,512,512)

                # ----- CombinedVideoLoss on tiles (training-style) -----
                # Criterion expects (B,T,C,H,W), so we add back T=1:
                out_seq = out_tiles.unsqueeze(1)           # (N,1,3,512,512)

                loss, loss_dict = criterion(out_seq, target_tiles)

                pixel_loss = loss_dict["pixel"]
                ssim_loss = loss_dict["ssim"]
                edge_loss = loss_dict["edge"]
                perc_loss = loss_dict["perceptual"]

                total_max_tensor = (
                    1.0 * pixel_loss
                    + SSIM_MAX * ssim_loss
                    + EDGE_MAX * edge_loss
                    + PERCEPTUAL_MAX * perc_loss
                )
                total_max = float(total_max_tensor)

                sum_total_max += total_max * N_tiles
                total_tiles += N_tiles

                # ----- Reconstruct full-res frame via Hann blending -----
                out_full = tiles_to_full_weighted(
                    out_tiles,
                    coords,
                    h=H,
                    w=W,
                    tile=TILE,
                ).clamp(0, 1)

                # ----- Full-frame metrics on CPU -----
                out_full_cpu = out_full.cpu()
                clean_cpu = clean_chw

                mae_frame, mse_frame, psnr_frame = fullframe_metrics(
                    out_full_cpu, clean_cpu
                )

                total_frames += 1
                sum_mae += mae_frame
                sum_mse += mse_frame
                sum_psnr += psnr_frame

                # ----- Save a few qualitative samples -----
                if sample_count < SAMPLE_LIMIT:
                    rainy_bgr = chw_to_bgr_uint8(rainy_chw)
                    out_bgr = chw_to_bgr_uint8(out_full_cpu)
                    clean_bgr = chw_to_bgr_uint8(clean_cpu)

                    stacked = np.concatenate([rainy_bgr, out_bgr, clean_bgr], axis=1)
                    out_path = (
                        OUTPUT_DIR
                        / f"sample_{scene_name}_{angle}_frame{i:04d}.png"
                    )
                    cv2.imwrite(str(out_path), stacked)
                    sample_count += 1

                # Some light logging
                if (total_frames % 100) == 0:
                    print(
                        f"    Frame {total_frames}: "
                        f"MAE={mae_frame:.4f}, PSNR={psnr_frame:.2f} dB, "
                        f"tile_loss={total_max:.4f}"
                    )

    if total_frames == 0:
        print("No frames processed. Check your test split and data paths.")
        return

    mean_mae = sum_mae / total_frames
    mean_mse = sum_mse / total_frames
    mean_psnr = sum_psnr / total_frames
    mean_total_max = sum_total_max / total_tiles if total_tiles > 0 else float("nan")

    print("\n" + "=" * 70)
    print("Stage 2 TILED TEST results (same pipeline as overlapping_inference.py):")
    print(f"  Num frames                : {total_frames}")
    print(f"  Num tiles (all frames)    : {total_tiles}")
    print(f"  Train-style TEST loss     : {mean_total_max:.6f}")
    print(f"  Best VAL loss (checkpoint): {best_val_loss:.6f}")
    print(f"  Mean MAE (full-frame)     : {mean_mae:.6f}")
    print(f"  Mean MSE (full-frame)     : {mean_mse:.6f}")
    print(f"  Mean PSNR (full-frame)    : {mean_psnr:.2f} dB")
    print("=" * 70)

    # ---------------- Save metrics to files ----------------
    summary = {
        "num_frames": total_frames,
        "num_tiles": total_tiles,
        "test_loss_combined": mean_total_max,
        "best_val_loss": best_val_loss,
        "mean_mae": mean_mae,
        "mean_mse": mean_mse,
        "mean_psnr_db": mean_psnr,
    }

    summary_json_path = OUTPUT_DIR / "metrics_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(summary, f, indent=4)

    summary_txt_path = OUTPUT_DIR / "metrics_summary.txt"
    with open(summary_txt_path, "w") as f:
        f.write("Stage 2 TILED TEST results (same pipeline as overlapping_inference.py)\n")
        f.write(f"Num frames             : {total_frames}\n")
        f.write(f"Num tiles              : {total_tiles}\n")
        f.write(f"Train-style TEST loss  : {mean_total_max:.6f}\n")
        f.write(f"Best VAL loss          : {best_val_loss:.6f}\n")
        f.write(f"Mean MAE               : {mean_mae:.6f}\n")
        f.write(f"Mean MSE               : {mean_mse:.6f}\n")
        f.write(f"Mean PSNR (dB)         : {mean_psnr:.2f}\n")

    print(f"\nMetrics summary saved to: {summary_json_path}")
    print(f"Text summary saved to   : {summary_txt_path}")
    print(f"Sample images saved to  : {OUTPUT_DIR} (up to {SAMPLE_LIMIT} frames)")


if __name__ == "__main__":
    main()
