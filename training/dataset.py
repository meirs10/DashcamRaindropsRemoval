# training/dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import random


class RainRemovalDataset(Dataset):
    def __init__(self,
                 clean_base_dir,
                 rainy_base_dir,
                 num_scenes=101,
                 frames_per_clip=8,
                 img_size=(540, 960),
                 split='train',
                 train_ratio=0.8):

        self.clean_base = Path(clean_base_dir)
        self.rainy_base = Path(rainy_base_dir)
        self.frames_per_clip = frames_per_clip
        self.img_size = img_size

        self.angles = [
            'front-forward',
            'left-backward',
            'left-forward',
            'right-backward',
            'right-forward'
        ]

        self.samples = []
        for scene_num in range(1, num_scenes + 1):
            scene_name = f"scene_{scene_num:03d}"
            for angle in self.angles:
                clean_dir = self.clean_base / scene_name / "images" / angle
                rainy_dir = self.rainy_base / scene_name / angle

                if clean_dir.exists() and rainy_dir.exists():
                    clean_files = sorted(list(clean_dir.glob('*.jpeg')))
                    rainy_files = sorted(list(rainy_dir.glob('*.jpeg')))

                    if len(clean_files) >= frames_per_clip and len(rainy_files) >= frames_per_clip:
                        self.samples.append({
                            'scene': scene_name,
                            'angle': angle,
                            'clean_dir': clean_dir,
                            'rainy_dir': rainy_dir,
                            'num_frames': min(len(clean_files), len(rainy_files))
                        })

        random.seed(42)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)

        if split == 'train':
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        print(f"{split.upper()} dataset: {len(self.samples)} scene/angle pairs")

    def __len__(self):
        return len(self.samples)

    def load_frame(self, path):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img).permute(2, 0, 1)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        clean_files = sorted(list(sample['clean_dir'].glob('*.jpeg')))
        rainy_files = sorted(list(sample['rainy_dir'].glob('*.jpeg')))

        max_start = min(len(clean_files), len(rainy_files)) - self.frames_per_clip
        start_idx = random.randint(0, max_start) if max_start > 0 else 0

        clean_frames = []
        rainy_frames = []

        for i in range(self.frames_per_clip):
            frame_idx = start_idx + i
            clean_frames.append(self.load_frame(clean_files[frame_idx]))
            rainy_frames.append(self.load_frame(rainy_files[frame_idx]))

        clean_video = torch.stack(clean_frames, dim=0)
        rainy_video = torch.stack(rainy_frames, dim=0)

        return rainy_video, clean_video