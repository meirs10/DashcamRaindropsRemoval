# test_inference.py
import torch
from pathlib import Path
import cv2
import numpy as np
import sys

# Add training folder to path
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "training"))

from model import MobileNetV3UNetConvLSTMVideo

# Paths
CHECKPOINT_PATH = BASE / "checkpoints" / "best.pth"
RAINY_DIR = BASE / "data_after_crapification" / "scene_001" / "front-forward"
CLEAN_DIR = BASE / "data" / "scene_001" / "images" / "front-forward"
OUTPUT_DIR = BASE / "test_results" / "scene_001_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load model
print("Loading model...")
model = MobileNetV3UNetConvLSTMVideo(
    hidden_dim=96,
    out_channels=3,
    use_pretrained_encoder=True,
    freeze_encoder=True
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
print(f"✓ Validation Loss: {checkpoint['val_loss']:.6f}\n")

# Load test video (8 frames)
print("Loading test frames...")
rainy_files = sorted(list(RAINY_DIR.glob('*.jpeg')))[:8]
clean_files = sorted(list(CLEAN_DIR.glob('*.jpeg')))[:8]


def load_frame(path, size=(256, 256)):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


rainy_frames = [load_frame(f) for f in rainy_files]
clean_frames = [load_frame(f) for f in clean_files]

rainy_video = torch.stack(rainy_frames).unsqueeze(0).to(device)  # (1, 8, 3, 256, 256)
clean_video = torch.stack(clean_frames).unsqueeze(0).to(device)

print(f"✓ Loaded {len(rainy_files)} frames\n")

# Run inference
print("Running inference...")
with torch.no_grad():
    output_video = model(rainy_video)

print("✓ Inference complete!\n")

# Save comparison images
print("Saving results...")
rainy_np = rainy_video.squeeze(0).cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
output_np = output_video.squeeze(0).cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()
clean_np = clean_video.squeeze(0).cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy()

for i in range(len(rainy_np)):
    # Create side-by-side comparison
    comparison = np.concatenate([
        rainy_np[i],
        output_np[i],
        clean_np[i]
    ], axis=1)  # Horizontal concatenation

    comparison = (comparison * 255).astype(np.uint8)
    comparison_bgr = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)

    # Add labels
    h, w = comparison_bgr.shape[:2]
    cv2.putText(comparison_bgr, "RAINY INPUT", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(comparison_bgr, "MODEL OUTPUT", (w // 3 + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(comparison_bgr, "GROUND TRUTH", (2 * w // 3 + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    output_path = OUTPUT_DIR / f"comparison_frame_{i:04d}.png"
    cv2.imwrite(str(output_path), comparison_bgr)

print(f"✓ Saved {len(rainy_np)} comparison images to:\n  {OUTPUT_DIR}\n")

# Calculate test loss
test_loss = torch.nn.functional.l1_loss(output_video, clean_video).item()
print("=" * 60)
print(f"Test Loss on this video: {test_loss:.6f}")
print(f"Training Val Loss:       {checkpoint['val_loss']:.6f}")
print(f"Difference:              {abs(test_loss - checkpoint['val_loss']):.6f}")
print("=" * 60)
print("\n✓ DONE! Open the comparison images to see results!\n")