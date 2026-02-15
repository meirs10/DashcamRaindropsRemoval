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
# Paths / Config (Kept from original)
# --------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "training"))
from training.model import MobileNetV3UNetConvLSTMVideo

CHECKPOINT_PATH = BASE / "training" / "checkpoints" / "stage2" / "best_stage2.pth"
RAINY_DIR = BASE / "data" / "data_crapified_test" / "scene_004" / "front-forward"
CLEAN_DIR = BASE / "data" / "data_original" / "scene_004" / "images" / "front-forward"
OUTPUT_DIR = BASE / "testing" / "test_results" / "scene_004_inference"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 33.0
TARGET_MS_PER_FRAME = 1000.0 / TARGET_FPS
VIDEO_FPS = 10
TILE = 512
ROWS, COLS = 3, 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------
# Optimized Helper Functions
# --------------------------------------------------------------------
def load_frame_fullres(path: Path) -> torch.Tensor:
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


def get_tile_coords(h, w, tile=512, rows=3, cols=5):
    xs = np.linspace(0, w - tile, cols).round().astype(int).tolist()
    ys = np.linspace(0, h - tile, rows).round().astype(int).tolist()
    return [(y, x) for y in ys for x in xs]


def make_hann_mask(tile, device):
    w1 = torch.hann_window(tile, periodic=False, device=device)
    return torch.outer(w1, w1).clamp_min(1e-4).unsqueeze(0)


def chw_to_bgr_uint8(chw: torch.Tensor) -> np.ndarray:
    return cv2.cvtColor((chw.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------
# Main Execution
# --------------------------------------------------------------------
def main():
    print(f"Using device: {device}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model = MobileNetV3UNetConvLSTMVideo(hidden_dim=96, out_channels=3).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rainy_files = sorted(RAINY_DIR.glob("*.jpeg"))
    clean_files = sorted(CLEAN_DIR.glob("*.jpeg"))
    n = min(len(rainy_files), len(clean_files))

    # Pre-probe dimensions
    tmp_img = cv2.imread(str(rainy_files[0]))
    h, w, _ = tmp_img.shape
    coords = get_tile_coords(h, w, TILE, ROWS, COLS)
    hann_mask = make_hann_mask(TILE, device)

    # PRE-ALLOCATE GPU BUFFERS (Critical to prevent fragmentation)
    acc_buffer = torch.zeros((3, h, w), device=device)
    wgt_buffer = torch.zeros((1, h, w), device=device)

    times_ms = []
    processed_frames = []

    print(f"\nStarting Optimized Benchmark ({n} frames)...")

    with torch.no_grad():
        for i in range(n):
            # 1. Load data (CPU)
            rainy_chw = load_frame_fullres(rainy_files[i])
            clean_chw = load_frame_fullres(clean_files[i])

            # 2. Build Tile Batch
            tiles = torch.stack([rainy_chw[:, y:y + TILE, x:x + TILE] for y, x in coords]).to(device)

            # --- START TIMING ---
            if device.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()

            # 3. Model Forward
            out_tiles = model(tiles.unsqueeze(1)).squeeze(1)

            # 4. Weighted Reconstruct (on GPU)
            acc_buffer.zero_()
            wgt_buffer.zero_()
            for idx, (y, x) in enumerate(coords):
                tile_out = out_tiles[idx]
                acc_buffer[:, y:y + TILE, x:x + TILE].add_(tile_out * hann_mask)
                wgt_buffer[:, y:y + TILE, x:x + TILE].add_(hann_mask)

            out_full = acc_buffer / wgt_buffer.clamp_min(1e-6)

            if device.type == "cuda": torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            # --- END TIMING ---

            times_ms.append(dt_ms)

            # Post-processing (Moving to CPU to free GPU ASAP)
            out_cpu = out_full.clamp(0, 1).cpu()
            processed_frames.append((rainy_chw, out_cpu, clean_chw, dt_ms))

            if i % 10 == 0:
                print(f"Frame {i}/{n}: {dt_ms:.2f}ms ({1000 / dt_ms:.1f} FPS)")

    # --------------------------------------------------------------------
    # IO Section (Save results after benchmark is done)
    # --------------------------------------------------------------------
    print("\nBenchmark finished. Saving video and images...")
    video_path = OUTPUT_DIR / "comparison_video.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, (w, h * 3))

    for i, (r_chw, o_chw, c_chw, dt) in enumerate(processed_frames):
        r_bgr = chw_to_bgr_uint8(r_chw)
        o_bgr = chw_to_bgr_uint8(o_chw)
        c_bgr = chw_to_bgr_uint8(c_chw)

        stacked = np.concatenate([r_bgr, o_bgr, c_bgr], axis=0)

        # Overlay labels
        fps = 1000.0 / dt if dt > 0 else 0
        cv2.putText(stacked, f"{dt:.1f}ms | {fps:.1f} FPS", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        writer.write(stacked)
        if i < 10:  # Only save first few as images to save space
            cv2.imwrite(str(OUTPUT_DIR / f"frame_{i:03d}.png"), stacked)

    writer.release()

    # Summary
    avg_ms = mean(times_ms)
    print("\n" + "=" * 30)
    print(f"AVERAGE LATENCY: {avg_ms:.2f} ms")
    print(f"AVERAGE FPS:     {1000 / avg_ms:.2f}")
    print(f"MEETS TARGET:    {avg_ms <= TARGET_MS_PER_FRAME}")
    print("=" * 30)


if __name__ == "__main__":
    main()