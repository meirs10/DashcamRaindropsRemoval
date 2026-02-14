# testing/inference_benchmark.py
"""
Stage-2 tiled inference with overlap + feathered blending (NO padding).

Simulation of "live" performance on a recorded sequence:
- Splits each frame into 3x5 overlapping 512x512 tiles (linspace positions).
- Runs inference on all tiles as a batch (B = N_tiles, T = 1).
- Reconstructs full-res output with weighted blending:
    pixels nearer a tile center get higher weight (2D Hann window).
- Saves stacked comparison images + video.
- Reports timing and effective FPS, for comparison with a real-time
  target (e.g. 33 FPS ≈ 30.3 ms per frame).

Dynamic optimization:
- Detects GPU and *benchmarks* FP32 vs FP16 (autocast) on a sample frame.
- Uses whatever is actually faster on this machine.
"""

import json
import sys
import time
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
import torch
from torch.amp import autocast

# --------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "training"))

from model import MobileNetV3UNetConvLSTMVideo  # type: ignore

CHECKPOINT_PATH = BASE / "training" / "checkpoints" / "attention" / "latest_with_attention.pth"

RAINY_DIR = BASE / "data" / "data_crapified_test" / "scene_004" / "front-forward"
CLEAN_DIR = BASE / "data" / "data_original" / "scene_004" / "images" / "front-forward"

OUTPUT_DIR = BASE / "testing" / "test_results" / "scene_004_inference_attention"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Target "real-time" requirements (for reporting only)
TARGET_FPS = 33.0
TARGET_MS_PER_FRAME = 1000.0 / TARGET_FPS  # ~30.3 ms

# Video output FPS for visualization (unrelated to real-time timing)
VIDEO_FPS = 10

# Tiling parameters
TILE = 512
ROWS, COLS = 3, 5  # 3 x 5 = 15 tiles

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------------------------
# GPU feature detection
# --------------------------------------------------------------------
def detect_gpu_features():
    """
    Returns (has_fast_fp16, gpu_name, cc_major, cc_minor) based on *capability*.

    Heuristic:
      - has_fast_fp16 = True if compute capability >= 7.0
        (Volta/Turing/Ampere/Ada – usually RTX / Turing+ GPUs).
      - otherwise False (older GTX etc. or CPU).
    """
    if device.type != "cuda":
        print("Running on CPU – using FP32 only.")
        return False, "CPU", 0, 0

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    has_fast_fp16 = major >= 7  # 7.x and 8.x: Tensor Cores available

    print(f"CUDA GPU: {name} (cc {major}.{minor}), fast_fp16_capability={has_fast_fp16}")
    return has_fast_fp16, name, major, minor


HAS_FAST_FP16_CAP, GPU_NAME, CC_MAJOR, CC_MINOR = detect_gpu_features()


# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------
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
    """
    (1, tile, tile) weight mask: high at center, low near edges.
    Precomputed once for the device/dtype.
    """
    w1 = torch.hann_window(tile, periodic=False, device=device, dtype=dtype)
    w2 = torch.outer(w1, w1).clamp_min(eps)  # avoid exact zeros
    return w2.unsqueeze(0)  # (1,tile,tile)


def tiles_to_full_weighted(
    tiles: torch.Tensor,
    coords,
    h: int,
    w: int,
    mask: torch.Tensor,
    tile: int = 512,
) -> torch.Tensor:
    """
    Weighted overlap-add.
    tiles: (N,3,tile,tile)
    mask:  (1,tile,tile) Hann weight
    """
    acc = torch.zeros((3, h, w), device=tiles.device, dtype=tiles.dtype)
    wgt = torch.zeros((1, h, w), device=tiles.device, dtype=tiles.dtype)

    for i, (y0, x0) in enumerate(coords):
        acc[:, y0:y0 + tile, x0:x0 + tile] += tiles[i] * mask
        wgt[:, y0:y0 + tile, x0:x0 + tile] += mask

    return acc / wgt.clamp_min(1e-6)


def chw_to_bgr_uint8(chw: torch.Tensor) -> np.ndarray:
    chw = chw.clamp(0, 1)
    hwc = chw.permute(1, 2, 0).cpu().numpy()
    hwc_u8 = (hwc * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(hwc_u8, cv2.COLOR_RGB2BGR)


def stack_triplet_vertical(
    rainy_bgr: np.ndarray,
    out_bgr: np.ndarray,
    clean_bgr: np.ndarray,
) -> np.ndarray:
    return np.concatenate([rainy_bgr, out_bgr, clean_bgr], axis=0)


def put_labels(
    stacked_bgr: np.ndarray,
    w: int,
    h: int,
    dt_ms: float,
    inst_fps: float,
) -> np.ndarray:
    img = stacked_bgr
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    cv2.putText(img, "RAINY INPUT",   (20, 40),
                font, scale, (0, 0, 255), thickness)
    cv2.putText(img, "MODEL OUTPUT",  (20, h + 40),
                font, scale, (0, 255, 0), thickness)
    cv2.putText(img, "GROUND TRUTH",  (20, 2 * h + 40),
                font, scale, (255, 0, 0), thickness)

    # Timing overlay (top strip, right side)
    txt = f"{dt_ms:.1f} ms  |  {inst_fps:.1f} FPS"
    cv2.putText(
        img,
        txt,
        (w - 320, 40),
        font,
        scale,
        (255, 255, 255),
        thickness,
    )

    return img


def benchmark_forward(model: torch.nn.Module, tiles_bchw: torch.Tensor) -> bool:
    """
    Compare FP32 vs FP16 (autocast) on a sample tiles batch and
    return USE_AMP = True if FP16 is significantly faster.

    tiles_bchw: (N,3,H,W) on device.
    """
    if device.type != "cuda" or not HAS_FAST_FP16_CAP:
        print("Benchmark: no CUDA / no fast FP16 capability -> using FP32.")
        return False

    def time_mode(use_amp: bool) -> float:
        times = []
        for it in range(6):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                if use_amp:
                    with autocast("cuda"):
                        _ = model(tiles_bchw.unsqueeze(1))  # (N,1,3,H,W)
                else:
                    _ = model(tiles_bchw.unsqueeze(1))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if it > 0:  # skip first as warm-up
                times.append((t1 - t0) * 1000.0)
        return float(mean(times)) if times else float("inf")

    print("\nBenchmarking FP32 vs FP16 (autocast) on sample tiles...")
    t_fp32 = time_mode(use_amp=False)
    t_fp16 = time_mode(use_amp=True)

    print(f"  FP32 avg: {t_fp32:.2f} ms")
    print(f"  FP16 avg: {t_fp16:.2f} ms")

    # Only enable AMP if it's clearly faster (at least 10% gain)
    if t_fp16 < 0.9 * t_fp32:
        print("=> Using FP16 (autocast) for inference.\n")
        return True
    else:
        print("=> FP16 not beneficial -> using FP32.\n")
        return False


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    print("\nLoading Stage-2 checkpoint...")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=True,
    ).to(device)

    # *** IMPORTANT: load weights (fix for "green screen") ***
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(
        f"✓ Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val_loss={checkpoint['val_loss']:.6f})\n"
    )

    rainy_files = sorted(RAINY_DIR.glob("*.jpeg"))
    clean_files = sorted(CLEAN_DIR.glob("*.jpeg"))
    n = min(len(rainy_files), len(clean_files))
    rainy_files, clean_files = rainy_files[:n], clean_files[:n]
    if n == 0:
        raise RuntimeError("No frames found in the specified directories.")

    print(f"Using {n} frames from:")
    print(f"  Rainy:  {RAINY_DIR}")
    print(f"  Clean:  {CLEAN_DIR}\n")

    # Probe resolution from first frame
    rainy0 = load_frame_fullres(rainy_files[0])
    _, h, w = rainy0.shape

    coords = get_tile_coords(h, w, tile=TILE, rows=ROWS, cols=COLS)
    n_tiles = len(coords)
    print(f"Tiling: {ROWS} x {COLS} => {n_tiles} tiles per frame (tile={TILE})")
    assert n_tiles == ROWS * COLS, f"Expected {ROWS*COLS} tiles, got {n_tiles}"

    # Build sample tiles batch (for AMP benchmark)
    sample_tiles = []
    for (y0, x0) in coords:
        sample_tiles.append(rainy0[:, y0:y0 + TILE, x0:x0 + TILE])
    sample_tiles = torch.stack(sample_tiles, dim=0).to(device)  # (N,3,TILE,TILE)

    # Decide whether to use AMP based on actual timing
    USE_AMP = benchmark_forward(model, sample_tiles)

    # Precompute Hann mask (fp32) on the correct device
    hann_mask = make_hann_mask(TILE, device=device, dtype=torch.float32)

    # Setup video writer
    video_path = OUTPUT_DIR / "comparison_stacked_10fps.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, VIDEO_FPS, (w, h * 3))
    if not writer.isOpened():
        raise RuntimeError("Failed to open VideoWriter (mp4v).")

    print("Running overlapping + weighted-blend inference (timed)...")

    # Timing stats
    times_ms = []

    for i, (rp, cp) in enumerate(zip(rainy_files, clean_files)):
        rainy_chw = load_frame_fullres(rp)  # CPU
        clean_chw = load_frame_fullres(cp)  # CPU

        if rainy_chw.shape != clean_chw.shape:
            raise ValueError(
                f"Shape mismatch at frame {i}: {rainy_chw.shape} vs {clean_chw.shape}"
            )

        # Build tiles on CPU
        tiles = []
        for (y0, x0) in coords:
            tiles.append(rainy_chw[:, y0:y0 + TILE, x0:x0 + TILE])
        tiles = torch.stack(tiles, dim=0)  # (N,3,TILE,TILE)

        # Move to device
        tiles = tiles.to(device, non_blocking=True)

        # Inference timing: one batched forward over all tiles
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            if USE_AMP and device.type == "cuda":
                with autocast("cuda"):
                    inp = tiles.unsqueeze(1)      # (N,1,3,512,512)
                    out_tiles = model(inp)        # (N,1,3,512,512)
            else:
                inp = tiles.unsqueeze(1)          # (N,1,3,512,512)
                out_tiles = model(inp)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        dt_ms = (t1 - t0) * 1000.0
        times_ms.append(dt_ms)
        inst_fps = 1000.0 / dt_ms if dt_ms > 0 else float("inf")

        # Prepare output tiles for blending (fp32)
        out_tiles = out_tiles.squeeze(1)  # (N,3,TILE,TILE)
        if USE_AMP and device.type == "cuda":
            out_tiles = out_tiles.to(torch.float32)

        out_full = tiles_to_full_weighted(
            out_tiles,
            coords,
            h=h,
            w=w,
            mask=hann_mask,  # already on correct device
            tile=TILE,
        ).clamp(0, 1).cpu()

        rainy_bgr = chw_to_bgr_uint8(rainy_chw)
        out_bgr = chw_to_bgr_uint8(out_full)
        clean_bgr = chw_to_bgr_uint8(clean_chw)

        stacked = stack_triplet_vertical(rainy_bgr, out_bgr, clean_bgr)
        stacked = put_labels(stacked, w=w, h=h, dt_ms=dt_ms, inst_fps=inst_fps)

        # Save frame + video
        out_path = OUTPUT_DIR / f"comparison_frame_{i:04d}_stacked.png"
        cv2.imwrite(str(out_path), stacked)
        writer.write(stacked)

        status = (
            f"Frame {i+1}/{n}: {dt_ms:.2f} ms "
            f"({inst_fps:.1f} FPS) "
            f"{'OK' if dt_ms <= TARGET_MS_PER_FRAME else 'SLOW'}"
        )
        print(status)

    writer.release()

    # ----------------------------------------------------------------
    # Timing summary
    # ----------------------------------------------------------------
    if times_ms:
        avg_ms = float(mean(times_ms))
        med_ms = float(median(times_ms))
        max_ms = float(max(times_ms))
        eff_fps_avg = 1000.0 / avg_ms
        eff_fps_med = 1000.0 / med_ms

        print("\n✓ Inference complete.")
        print(f"✓ Saved {n} stacked images to: {OUTPUT_DIR}")
        print(f"✓ Saved video to: {video_path}")
        print("\n" + "=" * 60)
        print(
            "Timing (batch of tiles per frame, including model forward only):"
        )
        print(f"  avg latency   : {avg_ms:.2f} ms  -> {eff_fps_avg:.2f} FPS")
        print(f"  median latency: {med_ms:.2f} ms  -> {eff_fps_med:.2f} FPS")
        print(f"  max latency   : {max_ms:.2f} ms")
        print(f"  Real-time target: {TARGET_MS_PER_FRAME:.2f} ms per frame "
              f"({TARGET_FPS:.1f} FPS)")

        meets_realtime = avg_ms <= TARGET_MS_PER_FRAME
        print(f"\nMeets 33 FPS on average? {'YES' if meets_realtime else 'NO'}")
        print(f"Stage-2 best val loss (combined): {checkpoint['val_loss']:.6f}")
        print("=" * 60 + "\n")

        # Save summary as JSON
        summary = {
            "num_frames": n,
            "avg_ms_per_frame": avg_ms,
            "median_ms_per_frame": med_ms,
            "max_ms_per_frame": max_ms,
            "effective_fps_avg": eff_fps_avg,
            "effective_fps_median": eff_fps_med,
            "target_fps": TARGET_FPS,
            "target_ms_per_frame": TARGET_MS_PER_FRAME,
            "meets_realtime_avg": meets_realtime,
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_val_loss": float(checkpoint["val_loss"]),
            "gpu_name": GPU_NAME,
            "compute_capability": f"{CC_MAJOR}.{CC_MINOR}",
            "has_fast_fp16_capability": HAS_FAST_FP16_CAP,
            "use_amp": USE_AMP,
        }
        summary_path = OUTPUT_DIR / "inference_timing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"Timing summary saved to: {summary_path}")
    else:
        print("No timing data_original collected (no frames?).")


if __name__ == "__main__":
    main()
