# move_test_scenes_to_temp.py
import json
import shutil
from pathlib import Path
from datetime import datetime

BASE = Path(r"D:\Pycharm Projects\DashcamRaindropsRemoval")
SPLIT_FILE = BASE / "scene_split.json"
DATA_DIR = BASE / "data_after_crapification_per_frame"
BACKUP_DIR = BASE / "temp_test_scenes_backup"  # ← Backup location

# Create backup directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_DIR = BASE / f"temp_test_scenes_backup_{timestamp}"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Load split
if not SPLIT_FILE.exists():
    print("❌ scene_split.json not found!")
    print("   Run determine_split.py first.")
    exit(1)

with open(SPLIT_FILE, 'r') as f:
    split_info = json.load(f)

test_scenes = split_info['test']

print("=" * 60)
print("MOVING TEST SCENES TO BACKUP")
print("=" * 60)
print(f"Test scenes to move: {len(test_scenes)}")
print(f"Scenes: {test_scenes}")
print(f"From: {DATA_DIR}")
print(f"To:   {BACKUP_DIR}")
print("=" * 60 + "\n")

# Ask for confirmation
response = input("Continue? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("Cancelled.")
    exit(0)

moved_count = 0
not_found = []

for scene_num in test_scenes:
    scene_name = f"scene_{scene_num:03d}"
    source_dir = DATA_DIR / scene_name
    dest_dir = BACKUP_DIR / scene_name

    if source_dir.exists():
        print(f"Moving {scene_name}...")
        shutil.move(str(source_dir), str(dest_dir))
        moved_count += 1
    else:
        print(f"⚠️  {scene_name} not found (skipping)")
        not_found.append(scene_name)

print("\n" + "=" * 60)
print("MOVE COMPLETE")
print("=" * 60)
print(f"Moved: {moved_count}/{len(test_scenes)} scenes")

if not_found:
    print(f"Not found: {not_found}")

print(f"\n✓ Test scenes backed up to:")
print(f"  {BACKUP_DIR}")
print(f"\n✓ data_after_crapification_per_frame now contains only train+val scenes")
print(f"\nTo restore test scenes later:")
print(f"  Simply move them back from {BACKUP_DIR}")
print("=" * 60)

# Save backup info
backup_info = {
    'timestamp': timestamp,
    'moved_scenes': test_scenes,
    'moved_count': moved_count,
    'not_found': not_found,
    'backup_location': str(BACKUP_DIR),
    'original_location': str(DATA_DIR)
}

info_file = BACKUP_DIR / "backup_info.json"
with open(info_file, 'w') as f:
    json.dump(backup_info, f, indent=2)

print(f"\n✓ Backup info saved to: {info_file}")