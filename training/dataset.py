# training/dataset.py
"""
Dataset for video rain removal training
Loads sequences of frames with scene-level split (80/10/10)
"""

import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class RainRemovalDataset(Dataset):
    """
    Dataset for loading rainy and clean video frames

    Supports scene-level split to prevent data leakage:
    - Reads split from scene_split.json if available
    - Falls back to generating split with seed=42
    """

    def __init__(self,
                 clean_base_dir,
                 rainy_base_dir,
                 num_scenes=101,
                 frames_per_clip=8,
                 consecutive_frames=True,
                 img_size=(540, 960),
                 split='train',
                 train_ratio=0.8,
                 val_ratio=0.1,
                 split_file=None):
        """
        Args:
            clean_base_dir: Path to clean images (e.g., data/)
            rainy_base_dir: Path to rainy images (e.g., data_after_crapification/)
            num_scenes: Total number of scenes (default: 101)
            frames_per_clip: Number of consecutive frames to load (default: 8)
            img_size: (height, width) to resize images to (default: 540x960)
            split: 'train', 'val', or 'test'
            train_ratio: Ratio for training set (default: 0.8 = 80%)
            val_ratio: Ratio for validation set (default: 0.1 = 10%)
                      Test ratio is automatically 1 - train_ratio - val_ratio
            split_file: Path to scene_split.json (if None, auto-detect)
        """
        self.clean_base = Path(clean_base_dir)
        self.rainy_base = Path(rainy_base_dir)
        self.frames_per_clip = frames_per_clip
        self.consecutive_frames = consecutive_frames
        self.img_size = img_size
        self.split = split

        # Camera angles
        self.angles = [
            'front-forward',
            'left-backward',
            'left-forward',
            'right-backward',
            'right-forward'
        ]

        # ===== Load or generate scene split =====
        if split_file is None:
            # Try to auto-detect scene_split.json
            split_file = Path(clean_base_dir).parent / "scene_split.json"

        if Path(split_file).exists():
            # Load split from file (preferred method)
            print(f"📋 Loading split from: {split_file}")
            with open(split_file, 'r') as f:
                split_info = json.load(f)

            selected_scenes = split_info[split]
            print(f"   Using {split} scenes from file")

        else:
            # Fallback: generate split on the fly
            print(f"⚠️  Split file not found: {split_file}")
            print(f"   Generating split with seed=42...")

            all_scenes = list(range(1, num_scenes + 1))
            random.seed(42)  # CRITICAL: Fixed seed for reproducibility
            random.shuffle(all_scenes)

            # Calculate split indices
            train_end = int(len(all_scenes) * train_ratio)
            val_end = train_end + int(len(all_scenes) * val_ratio)

            # Split scenes
            if split == 'train':
                selected_scenes = all_scenes[:train_end]
            elif split == 'val':
                selected_scenes = all_scenes[train_end:val_end]
            elif split == 'test':
                selected_scenes = all_scenes[val_end:]
            else:
                raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        print(f"   Selected {len(selected_scenes)} scenes for {split}")

        # ===== Build sample list =====
        self.samples = []

        for scene_num in selected_scenes:
            scene_name = f"scene_{scene_num:03d}"

            for angle in self.angles:
                # Paths to clean and rainy images
                clean_dir = self.clean_base / scene_name / "images" / angle
                rainy_dir = self.rainy_base / scene_name / angle

                # Check if both directories exist
                if not clean_dir.exists() or not rainy_dir.exists():
                    continue

                # Get image files
                clean_files = sorted(list(clean_dir.glob('*.jpeg')))
                rainy_files = sorted(list(rainy_dir.glob('*.jpeg')))

                # Need at least frames_per_clip frames
                if len(clean_files) >= frames_per_clip and len(rainy_files) >= frames_per_clip:
                    self.samples.append({
                        'scene': scene_name,
                        'angle': angle,
                        'clean_dir': clean_dir,
                        'rainy_dir': rainy_dir,
                        'num_frames': min(len(clean_files), len(rainy_files))
                    })

        print(f"{split.upper()} dataset: {len(self.samples)} scene/angle pairs "
              f"from {len(selected_scenes)} scenes")

    def __len__(self):
        return len(self.samples)

    def load_frame(self, path, size=None):
        """
        Load and preprocess a single frame

        Args:
            path: Path to image file
            size: (height, width) to resize to, or None to use self.img_size

        Returns:
            torch.Tensor: (C, H, W) normalized to [0, 1]
        """
        if size is None:
            size = self.img_size

        # Read image
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, (size[1], size[0]))  # cv2.resize expects (width, height)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to torch tensor (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)

        return img_tensor

    def __getitem__(self, idx):
        """
        Load a video clip (sequence of frames)

        Returns:
            rainy_video: torch.Tensor (T, C, H, W) - rainy frames
            clean_video: torch.Tensor (T, C, H, W) - clean frames
        """
        sample = self.samples[idx]

        clean_dir = sample['clean_dir']
        rainy_dir = sample['rainy_dir']
        num_frames = sample['num_frames']

        # Get all frame files
        clean_files = sorted(list(clean_dir.glob('*.jpeg')))
        rainy_files = sorted(list(rainy_dir.glob('*.jpeg')))

        # -----------------------------------------
        # Choose frame indices
        # -----------------------------------------
        if self.consecutive_frames:
            # Original behavior: a consecutive clip
            if num_frames > self.frames_per_clip:
                start_idx = random.randint(0, num_frames - self.frames_per_clip)
            else:
                start_idx = 0

            frame_indices = list(range(start_idx, start_idx + self.frames_per_clip))

        else:
            # New behavior: pick self.frames_per_clip RANDOM frames
            if num_frames >= self.frames_per_clip:
                # Sample distinct frame indices, then sort to keep temporal order
                frame_indices = sorted(
                    random.sample(range(num_frames), self.frames_per_clip)
                )
            else:
                # Fewer frames than requested: sample with replacement
                frame_indices = sorted(
                    random.choices(range(num_frames), k=self.frames_per_clip)
                )

            # -----------------------------------------
            # Load selected frames
            # -----------------------------------------
        rainy_frames = []
        clean_frames = []

        for idx in frame_indices:
            rainy_frame = self.load_frame(rainy_files[idx])
            rainy_frames.append(rainy_frame)

            clean_frame = self.load_frame(clean_files[idx])
            clean_frames.append(clean_frame)

        # Stack frames into video tensors (T, C, H, W)
        rainy_video = torch.stack(rainy_frames, dim=0)
        clean_video = torch.stack(clean_frames, dim=0)

        return rainy_video, clean_video


# ===== Utility functions =====

def get_scene_split(split_file=None, num_scenes=101, train_ratio=0.8, val_ratio=0.1):
    """
    Get scene split (train/val/test)

    Args:
        split_file: Path to scene_split.json (if None, generate split)
        num_scenes: Total number of scenes
        train_ratio: Training set ratio
        val_ratio: Validation set ratio

    Returns:
        dict with keys 'train', 'val', 'test' containing scene numbers
    """
    if split_file and Path(split_file).exists():
        with open(split_file, 'r') as f:
            return json.load(f)

    # Generate split
    all_scenes = list(range(1, num_scenes + 1))
    random.seed(42)
    random.shuffle(all_scenes)

    train_end = int(len(all_scenes) * train_ratio)
    val_end = train_end + int(len(all_scenes) * val_ratio)

    return {
        'train': sorted(all_scenes[:train_end]),
        'val': sorted(all_scenes[train_end:val_end]),
        'test': sorted(all_scenes[val_end:])
    }


# ===== Test/Debug code =====

if __name__ == "__main__":
    """Test the dataset"""
    from pathlib import Path

    BASE = Path(r"D:\Pycharm Projects\DashcamRaindropsRemoval")

    print("=" * 60)
    print("TESTING DATASET")
    print("=" * 60 + "\n")

    # Test train dataset
    print("Creating train dataset...")
    train_dataset = RainRemovalDataset(
        clean_base_dir=BASE / "data",
        rainy_base_dir=BASE / "data_after_crapification",
        num_scenes=101,
        frames_per_clip=8,
        img_size=(540, 960),
        split='train',
        train_ratio=0.8,
        val_ratio=0.1
    )

    print(f"Train dataset size: {len(train_dataset)}")

    # Load one sample
    print("\nLoading sample...")
    rainy, clean = train_dataset[0]

    print(f"Rainy video shape: {rainy.shape}")  # Should be (8, 3, 540, 960)
    print(f"Clean video shape: {clean.shape}")  # Should be (8, 3, 540, 960)
    print(f"Rainy range: [{rainy.min():.3f}, {rainy.max():.3f}]")
    print(f"Clean range: [{clean.min():.3f}, {clean.max():.3f}]")

    # Test val dataset
    print("\n" + "-" * 60)
    print("Creating val dataset...")
    val_dataset = RainRemovalDataset(
        clean_base_dir=BASE / "data",
        rainy_base_dir=BASE / "data_after_crapification",
        num_scenes=101,
        frames_per_clip=8,
        img_size=(540, 960),
        split='val',
        train_ratio=0.8,
        val_ratio=0.1
    )

    print(f"Val dataset size: {len(val_dataset)}")

    # Test test dataset
    print("\n" + "-" * 60)
    print("Creating test dataset...")
    test_dataset = RainRemovalDataset(
        clean_base_dir=BASE / "data",
        rainy_base_dir=BASE / "test_data_after_crapification",  # Different dir!
        num_scenes=101,
        frames_per_clip=8,
        img_size=(540, 960),
        split='test',
        train_ratio=0.8,
        val_ratio=0.1
    )

    print(f"Test dataset size: {len(test_dataset)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Train pairs: {len(train_dataset)}")
    print(f"Val pairs:   {len(val_dataset)}")
    print(f"Test pairs:  {len(test_dataset)}")
    print(f"Total:       {len(train_dataset) + len(val_dataset) + len(test_dataset)}")

    # Check for overlap
    train_scenes = set(s['scene'] for s in train_dataset.samples)
    val_scenes = set(s['scene'] for s in val_dataset.samples)
    test_scenes = set(s['scene'] for s in test_dataset.samples)

    overlap_train_val = train_scenes & val_scenes
    overlap_train_test = train_scenes & test_scenes
    overlap_val_test = val_scenes & test_scenes

    if not (overlap_train_val or overlap_train_test or overlap_val_test):
        print("\n✓ No scene overlap between splits!")
    else:
        print("\n❌ WARNING: Scene overlap detected!")
        if overlap_train_val:
            print(f"   Train/Val: {overlap_train_val}")
        if overlap_train_test:
            print(f"   Train/Test: {overlap_train_test}")
        if overlap_val_test:
            print(f"   Val/Test: {overlap_val_test}")

    print("=" * 60)