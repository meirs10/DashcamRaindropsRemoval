# training/dataset.py
"""
Dataset for video rain removal training.

Stage 1 configuration:

- Uses **all frames** from all videos in train and val.
- Treats each frame as an independent sample (T = 1).
- Supports random square crops (256/384/512) that are applied per sample.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RainRemovalDataset(Dataset):
    """
    Dataset for loading rainy and clean video frames.

    When per_frame=True (Stage 1):
      - Every frame in every (scene, angle) is a separate sample.
      - __getitem__ returns a single-frame "clip": (T=1, C, H, W).
    """

    def __init__(
        self,
        clean_base_dir,
        rainy_base_dir,
        num_scenes=101,
        frames_per_clip=8,
        consecutive_frames=True,
        img_size=(540, 960),
        split="train",
        train_ratio=0.8,
        val_ratio=0.1,
        split_file=None,
        per_frame: bool = False,
        random_crop: bool = False,
        crop_sizes=None,
        crop_probs=None,
    ):
        """
        Args:
            clean_base_dir: Path to clean images root.
            rainy_base_dir: Path to rainy images root.
            num_scenes: total number of scenes.
            frames_per_clip: kept for compatibility (used only when per_frame=False).
            consecutive_frames: kept for compatibility.
            img_size: (H, W) of network input after crop/resize.
            split: 'train', 'val', or 'test'.
            train_ratio, val_ratio: scene split ratios if no split_file is given.
            split_file: optional JSON with scene split.
            per_frame: if True, each frame is an independent sample (Stage 1).
            random_crop: if True, use random square crops.
            crop_sizes: list of crop side lengths (e.g. [256, 384, 512]).
            crop_probs: list of probabilities for the crop sizes.
        """
        self.clean_base = Path(clean_base_dir)
        self.rainy_base = Path(rainy_base_dir)
        self.frames_per_clip = frames_per_clip
        self.consecutive_frames = consecutive_frames
        self.img_size = img_size
        self.split = split

        self.per_frame = per_frame
        self.random_crop = random_crop
        self.crop_sizes = crop_sizes
        self.crop_probs = crop_probs

        # Camera angles present in your dataset
        self.angles = [
            "front-forward",
            "left-backward",
            "left-forward",
            "right-backward",
            "right-forward",
        ]

        # ===== Determine scene split =====
        if split_file is None:
            split_file = self.clean_base.parent / "scene_split.json"

        if Path(split_file).exists():
            print(f"📋 Loading scene split from: {split_file}")
            with open(split_file, "r") as f:
                split_info = json.load(f)
            selected_scenes = split_info[split]
        else:
            print(f"⚠️ Split file not found at {split_file}, generating a new split.")
            all_scenes = list(range(1, num_scenes + 1))
            random.seed(42)
            random.shuffle(all_scenes)

            train_end = int(len(all_scenes) * train_ratio)
            val_end = train_end + int(len(all_scenes) * val_ratio)

            if split == "train":
                selected_scenes = all_scenes[:train_end]
            elif split == "val":
                selected_scenes = all_scenes[train_end:val_end]
            elif split == "test":
                selected_scenes = all_scenes[val_end:]
            else:
                raise ValueError(f"Invalid split: {split}")

        print(f"{split.upper()} scenes: {len(selected_scenes)}")

        # ===== Build per-frame samples (Stage 1) or clip-level samples (for later) =====
        self.samples = []

        if self.per_frame:
            # --- Stage 1: every frame is a separate sample ---
            for scene_num in selected_scenes:
                scene_name = f"scene_{scene_num:03d}"

                for angle in self.angles:
                    clean_dir = self.clean_base / scene_name / "images" / angle
                    rainy_dir = self.rainy_base / scene_name / angle

                    if not clean_dir.exists() or not rainy_dir.exists():
                        continue

                    clean_files = sorted(clean_dir.glob("*.jpeg"))
                    rainy_files = sorted(rainy_dir.glob("*.jpeg"))

                    num_frames = min(len(clean_files), len(rainy_files))
                    if num_frames == 0:
                        continue

                    # Each frame index gives one sample.
                    for frame_idx in range(num_frames):
                        self.samples.append(
                            {
                                "clean_path": clean_files[frame_idx],
                                "rainy_path": rainy_files[frame_idx],
                            }
                        )

            print(
                f"Per-frame mode ON: {len(self.samples)} total frame-samples "
                f"for split '{split}'."
            )
        else:
            # --- Future use: clip-level mode (ConvLSTM with T>1) ---
            for scene_num in selected_scenes:
                scene_name = f"scene_{scene_num:03d}"

                for angle in self.angles:
                    clean_dir = self.clean_base / scene_name / "images" / angle
                    rainy_dir = self.rainy_base / scene_name / angle

                    if not clean_dir.exists() or not rainy_dir.exists():
                        continue

                    clean_files = sorted(clean_dir.glob("*.jpeg"))
                    rainy_files = sorted(rainy_dir.glob("*.jpeg"))

                    num_frames = min(len(clean_files), len(rainy_files))
                    if num_frames == 0:
                        continue

                    self.samples.append(
                        {
                            "scene": scene_name,
                            "angle": angle,
                            "clean_files": clean_files,
                            "rainy_files": rainy_files,
                            "num_frames": num_frames,
                        }
                    )

            print(
                f"Clip mode: {len(self.samples)} scene/angle clips "
                f"for split '{split}'."
            )

    def __len__(self):
        return len(self.samples)

    # ===== Image loading and cropping utilities =====
    def _random_square_crop(self, img: np.ndarray) -> np.ndarray:
        """
        img: H x W x C (RGB float or uint8)
        Returns a cropped region resized to self.img_size.
        """
        H, W, _ = img.shape
        target_h, target_w = self.img_size

        if not self.random_crop or not self.crop_sizes:
            # Fallback: direct resize
            return cv2.resize(img, (target_w, target_h))

        # choose crop size according to probabilities
        size = random.choices(self.crop_sizes, weights=self.crop_probs, k=1)[0]
        size = int(size)

        # clamp to image size
        max_side = min(H, W)
        if size > max_side:
            size = max_side

        if size <= 0:
            return cv2.resize(img, (target_w, target_h))

        y_max = H - size
        x_max = W - size
        if y_max < 0 or x_max < 0:
            return cv2.resize(img, (target_w, target_h))

        y0 = random.randint(0, max(0, y_max))
        x0 = random.randint(0, max(0, x_max))

        crop = img[y0 : y0 + size, x0 : x0 + size]
        crop = cv2.resize(crop, (target_w, target_h))
        return crop

    def _sample_square_coords(self, H, W):
        size = random.choices(self.crop_sizes, weights=self.crop_probs, k=1)[0]
        size = int(min(size, H, W))
        if size <= 0:
            return 0, 0, min(H, W)

        y_max = H - size
        x_max = W - size
        y0 = random.randint(0, max(0, y_max))
        x0 = random.randint(0, max(0, x_max))
        return y0, x0, size

    def _load_frame(self, path: Path) -> torch.Tensor:
        """
        Load one frame, apply crop/resize, normalize to [0,1], return (C,H,W) tensor.
        """
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        if self.random_crop:
            img = self._random_square_crop(img)
        else:
            h_out, w_out = self.img_size
            img = cv2.resize(img, (w_out, h_out))

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
        return img_tensor

    def __getitem__(self, idx):
        """
        Returns:
            rainy_video: (T, C, H, W)
            clean_video: (T, C, H, W)

        Stage 1 (per_frame=True): T = 1.
        """
        if self.per_frame:
            rec = self.samples[idx]

            # read both full-size images as float RGB in [0,1]
            rainy_img = cv2.imread(str(rec["rainy_path"]))
            clean_img = cv2.imread(str(rec["clean_path"]))

            rainy_img = cv2.cvtColor(rainy_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            H, W, _ = rainy_img.shape

            if self.random_crop and self.crop_sizes:
                y0, x0, size = self._sample_square_coords(H, W)
                rainy_img = rainy_img[y0:y0 + size, x0:x0 + size]
                clean_img = clean_img[y0:y0 + size, x0:x0 + size]

            # final resize to IMG_SIZE
            h_out, w_out = self.img_size
            rainy_img = cv2.resize(rainy_img, (w_out, h_out))
            clean_img = cv2.resize(clean_img, (w_out, h_out))

            rainy = torch.from_numpy(rainy_img).permute(2, 0, 1)
            clean = torch.from_numpy(clean_img).permute(2, 0, 1)

            return rainy.unsqueeze(0), clean.unsqueeze(0)  # T=1

        # --- Clip-level mode for future stages ---
        rec = self.samples[idx]
        clean_files = rec["clean_files"]
        rainy_files = rec["rainy_files"]
        num_frames = rec["num_frames"]

        # choose frame indices
        if self.consecutive_frames:
            if num_frames > self.frames_per_clip:
                start_idx = random.randint(0, num_frames - self.frames_per_clip)
            else:
                start_idx = 0
            frame_indices = list(range(start_idx, start_idx + self.frames_per_clip))
        else:
            if num_frames >= self.frames_per_clip:
                frame_indices = sorted(
                    random.sample(range(num_frames), self.frames_per_clip)
                )
            else:
                frame_indices = sorted(
                    random.choices(range(num_frames), k=self.frames_per_clip)
                )

        rainy_frames = []
        clean_frames = []
        for f_idx in frame_indices:
            rainy_frames.append(self._load_frame(rainy_files[f_idx]))
            clean_frames.append(self._load_frame(clean_files[f_idx]))

        rainy_video = torch.stack(rainy_frames, dim=0)
        clean_video = torch.stack(clean_frames, dim=0)
        return rainy_video, clean_video
