# inference.py
"""
Simple inference / visualization script for Stage 1.

- Loads best_stage1.pth.
- Runs the model on a short sequence of frames.
- Saves side-by-side comparison images (rainy | output | clean).
"""

import sys
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

OUTPUT_DIR = BASE / "test_results" / "scene_001_comparison_stage1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_frame(path: Path, size=(512, 512)) -> torch.Tensor:
    """
    For Stage 1 demo we resize input to the 512x512 training size.
    (Later you can replace this with tiling over full-resolution frames.)
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)  # (W, H) but square
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)


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
    rainy_files = sorted(list(RAINY_DIR.glob("*.jpeg")))[:8]
    clean_files = sorted(list(CLEAN_DIR.glob("*.jpeg")))[:8]

    if not rainy_files or not clean_files:
        raise RuntimeError("No frames found in the specified directories.")

    print(f"Using {len(rainy_files)} frames for visualization.\n")

    rainy_frames = [load_frame(p) for p in rainy_files]
    clean_frames = [load_frame(p) for p in clean_files]

    rainy_video = torch.stack(rainy_frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)
    clean_video = torch.stack(clean_frames).unsqueeze(0).to(device)

    # ----- Inference -----
    print("Running inference...")
    with torch.no_grad():
        output_video = model(rainy_video)
    print("✓ Inference complete.\n")

    rainy_np = (
        rainy_video.squeeze(0)
        .cpu()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    output_np = (
        output_video.squeeze(0)
        .cpu()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .numpy()
    )
    clean_np = (
        clean_video.squeeze(0)
        .cpu()
        .clamp(0, 1)
        .permute(0, 2, 3, 1)
        .numpy()
    )

    # ----- Save comparisons -----
    print("Saving comparison images...")
    for i in range(len(rainy_np)):
        concat = np.concatenate([rainy_np[i], output_np[i], clean_np[i]], axis=1)
        concat = (concat * 255).astype(np.uint8)
        concat_bgr = cv2.cvtColor(concat, cv2.COLOR_RGB2BGR)

        h, w = concat_bgr.shape[:2]
        cv2.putText(
            concat_bgr,
            "RAINY INPUT",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            concat_bgr,
            "MODEL OUTPUT",
            (w // 3 + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            concat_bgr,
            "GROUND TRUTH",
            (2 * w // 3 + 20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        out_path = OUTPUT_DIR / f"comparison_frame_{i:04d}.png"
        cv2.imwrite(str(out_path), concat_bgr)

    print(f"✓ Saved {len(rainy_np)} images to: {OUTPUT_DIR}")

    # Simple L1 sanity check
    test_l1 = torch.nn.functional.l1_loss(output_video, clean_video).item()
    print("\n" + "=" * 60)
    print(f"Test L1 loss on this sequence: {test_l1:.6f}")
    print(f"Stage-1 best val loss (combined): {checkpoint['val_loss']:.6f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
