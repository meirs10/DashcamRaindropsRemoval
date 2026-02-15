# determine_split.py
import random
import json
from pathlib import Path

BASE = Path(__file__).parent
SPLIT_FILE = BASE / "scene_split.json"

# Same shuffle as before (seed=42)
all_scenes = list(range(1, 102))
random.seed(42)
random.shuffle(all_scenes)

# 80/10/10 split
train_end = int(101 * 0.8)  # 80
val_end = train_end + int(101 * 0.1)  # 90

train_scenes = sorted(all_scenes[:train_end])        # 80 scenes
val_scenes = sorted(all_scenes[train_end:val_end])   # 10 scenes
test_scenes = sorted(all_scenes[val_end:])           # 11 scenes

split_info = {
    'train': train_scenes,
    'val': val_scenes,
    'test': test_scenes,
    'split_ratio': '80/10/10',
    'seed': 42
}

# Save split
with open(SPLIT_FILE, 'w') as f:
    json.dump(split_info, f, indent=2)

print("="*60)
print("SCENE SPLIT (80/10/10)")
print("="*60)
print(f"Train scenes ({len(train_scenes)}): {train_scenes}")
print(f"Val scenes ({len(val_scenes)}): {val_scenes}")
print(f"Test scenes ({len(test_scenes)}): {test_scenes}")
print(f"\nSaved to: {SPLIT_FILE}")
print("="*60)

# Verify no overlap
assert len(set(train_scenes) & set(val_scenes)) == 0
assert len(set(train_scenes) & set(test_scenes)) == 0
assert len(set(val_scenes) & set(test_scenes)) == 0
assert len(train_scenes) + len(val_scenes) + len(test_scenes) == 101
print("✓ No overlap - all scenes accounted for!\n")