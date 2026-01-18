import torch
import torch.nn.functional as F
from pathlib import Path
import cv2
import numpy as np
from sample import MobileNetV3UNetConvLSTMVideo

# Paths from your pipeline
BASE = r"D:\Pycharm Projects\DashcamRaindropsRemoval"
OUTPUT_BASE = Path(BASE) / "data_after_crapification"

ANGLES = [
    'front-forward',
    'left-backward',
    'left-forward',
    'right-backward',
    'right-forward'
]


def load_video_sequence(scene_path, max_frames=None):
    """
    Load a sequence of images as a video tensor
    scene_path: Path to angle directory (e.g., scene_001/front-forward)
    Returns: tensor of shape (T, 3, H, W)
    """
    image_files = sorted([f for f in scene_path.glob('*.jpeg') if f.is_file()])

    if max_frames:
        image_files = image_files[:max_frames]

    frames = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Convert to tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        frames.append(img_tensor)

    # Stack into (T, C, H, W)
    video_tensor = torch.stack(frames, dim=0)
    return video_tensor


def save_video_sequence(output_tensor, output_path):
    """
    Save output tensor back to images
    output_tensor: (T, 3, H, W)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Denormalize and convert back
    frames = output_tensor.cpu().clamp(0, 1) * 255
    frames = frames.permute(0, 2, 3, 1).numpy().astype(np.uint8)

    for i, frame in enumerate(frames):
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path / f"cleaned_{i:04d}.jpeg"), frame_bgr)


# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MobileNetV3UNetConvLSTMVideo(
    hidden_dim=96,
    out_channels=3,
    use_pretrained_encoder=True
)
model = model.to(device)
model.eval()

# Example: Process scene_001, front-forward angle
scene_name = "scene_001"
angle = "front-forward"

input_dir = OUTPUT_BASE / scene_name / angle
output_dir = Path(BASE) / "cleaned_output" / scene_name / angle

print(f"Loading frames from: {input_dir}")

# Load video sequence (limit to 10 frames for testing)
video_tensor = load_video_sequence(input_dir, max_frames=10)
print(f"Loaded video shape: {video_tensor.shape}")  # (T, 3, H, W)

# Add batch dimension: (1, T, 3, H, W)
video_tensor = video_tensor.unsqueeze(0).to(device)

# Run inference
with torch.no_grad():
    cleaned_video = model(video_tensor)

print(f"Output shape: {cleaned_video.shape}")

# Remove batch dimension and save
cleaned_video = cleaned_video.squeeze(0)  # (T, 3, H, W)
save_video_sequence(cleaned_video, output_dir)

print(f"✓ Saved cleaned frames to: {output_dir}")