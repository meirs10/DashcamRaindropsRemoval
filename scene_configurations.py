import json
import random
import os
from pathlib import Path


def generate_scene_configurations(num_scenes=101, base_dir=".", seed=42):
    """
    Generate diverse intensity configurations for all scenes.

    First 15 scenes: Heavy everything (already processed)
    Remaining scenes: Diverse combinations

    Returns:
        Dictionary mapping scene numbers to intensity configs
    """
    random.seed(seed)

    configs = {}

    # ==========================================
    # Scenes 1-15: HEAVY EVERYTHING (already done)
    # ==========================================
    for scene_num in range(1, 16):
        configs[scene_num] = {
            'fog': 'heavy',
            'rain': 'heavy',
            'droplets': 'heavy',
            'group': 'baseline'
        }

    # ==========================================
    # Remaining scenes: DIVERSE
    # ==========================================
    remaining = list(range(16, num_scenes + 1))
    random.shuffle(remaining)

    # Split into groups
    group_size = len(remaining) // 4

    # GROUP 1: Droplet-focused variations (~21 scenes)
    group1 = remaining[:group_size]
    droplet_configs = [
        {'fog': 'light', 'rain': 'none', 'droplets': 'heavy', 'group': 'droplet-focused'},
        {'fog': 'medium', 'rain': 'light', 'droplets': 'medium', 'group': 'droplet-focused'},
        {'fog': 'none', 'rain': 'medium', 'droplets': 'light', 'group': 'droplet-focused'},
        {'fog': 'none', 'rain': 'none', 'droplets': 'extreme', 'group': 'droplet-focused'},
        {'fog': 'light', 'rain': 'heavy', 'droplets': 'heavy', 'group': 'droplet-focused'},
    ]
    for i, scene_num in enumerate(group1):
        configs[scene_num] = droplet_configs[i % len(droplet_configs)].copy()

    # GROUP 2: Environmental variations (~21 scenes)
    group2 = remaining[group_size:group_size * 2]
    env_configs = [
        {'fog': 'heavy', 'rain': 'heavy', 'droplets': 'light', 'group': 'environmental'},
        {'fog': 'heavy', 'rain': 'none', 'droplets': 'light', 'group': 'environmental'},
        {'fog': 'light', 'rain': 'heavy', 'droplets': 'medium', 'group': 'environmental'},
        {'fog': 'none', 'rain': 'heavy', 'droplets': 'heavy', 'group': 'environmental'},
        {'fog': 'heavy', 'rain': 'medium', 'droplets': 'light', 'group': 'environmental'},
    ]
    for i, scene_num in enumerate(group2):
        configs[scene_num] = env_configs[i % len(env_configs)].copy()

    # GROUP 3: Mixed severity (~21 scenes)
    group3 = remaining[group_size * 2:group_size * 3]
    mixed_configs = [
        {'fog': 'medium', 'rain': 'medium', 'droplets': 'medium', 'group': 'mixed-severity'},
        {'fog': 'light', 'rain': 'light', 'droplets': 'light', 'group': 'mixed-severity'},
        {'fog': 'heavy', 'rain': 'light', 'droplets': 'medium', 'group': 'mixed-severity'},
        {'fog': 'light', 'rain': 'heavy', 'droplets': 'light', 'group': 'mixed-severity'},
        {'fog': 'medium', 'rain': 'light', 'droplets': 'heavy', 'group': 'mixed-severity'},
        {'fog': 'none', 'rain': 'light', 'droplets': 'medium', 'group': 'mixed-severity'},
    ]
    for i, scene_num in enumerate(group3):
        configs[scene_num] = mixed_configs[i % len(mixed_configs)].copy()

    # GROUP 4: Edge cases (~23 scenes)
    group4 = remaining[group_size * 3:]
    edge_configs = [
        {'fog': 'heavy', 'rain': 'none', 'droplets': 'light', 'group': 'edge-case'},
        {'fog': 'none', 'rain': 'heavy', 'droplets': 'light', 'group': 'edge-case'},
        {'fog': 'none', 'rain': 'none', 'droplets': 'heavy', 'group': 'edge-case'},
        {'fog': 'light', 'rain': 'none', 'droplets': 'light', 'group': 'edge-case'},
        {'fog': 'none', 'rain': 'light', 'droplets': 'light', 'group': 'edge-case'},
        {'fog': 'none', 'rain': 'none', 'droplets': 'light', 'group': 'edge-case'},
        {'fog': 'medium', 'rain': 'none', 'droplets': 'heavy', 'group': 'edge-case'},
        {'fog': 'none', 'rain': 'medium', 'droplets': 'heavy', 'group': 'edge-case'},
    ]
    for i, scene_num in enumerate(group4):
        configs[scene_num] = edge_configs[i % len(edge_configs)].copy()

    return configs


def save_configurations(configs, filepath):
    """Save configurations to JSON file"""
    # Convert int keys to strings for JSON
    json_configs = {f"scene_{k:03d}": v for k, v in configs.items()}

    with open(filepath, 'w') as f:
        json.dump(json_configs, f, indent=2)

    print(f"✓ Configurations saved to: {filepath}")


def load_configurations(filepath):
    """Load configurations from JSON file"""
    if not os.path.exists(filepath):
        return None

    with open(filepath, 'r') as f:
        json_configs = json.load(f)

    # Convert string keys back to scene numbers
    configs = {}
    for scene_name, config in json_configs.items():
        scene_num = int(scene_name.split('_')[1])
        configs[scene_num] = config

    return configs


def print_configuration_summary(configs):
    """Print summary of configuration distribution"""
    print("\n" + "=" * 60)
    print("SCENE CONFIGURATION SUMMARY")
    print("=" * 60)

    groups = {}
    for scene_num, config in configs.items():
        group = config['group']
        if group not in groups:
            groups[group] = []
        groups[group].append(scene_num)

    for group, scenes in sorted(groups.items()):
        print(f"\n{group.upper()}: {len(scenes)} scenes")
        if len(scenes) <= 10:
            print(f"  Scenes: {scenes}")

    # Count intensity distributions
    fog_counts = {'none': 0, 'light': 0, 'medium': 0, 'heavy': 0}
    rain_counts = {'none': 0, 'light': 0, 'medium': 0, 'heavy': 0}
    droplet_counts = {'light': 0, 'medium': 0, 'heavy': 0, 'extreme': 0}

    for config in configs.values():
        fog_counts[config['fog']] += 1
        rain_counts[config['rain']] += 1
        droplet_counts[config['droplets']] += 1

    print("\n" + "-" * 60)
    print("INTENSITY DISTRIBUTION:")
    print(f"Fog:      {fog_counts}")
    print(f"Rain:     {rain_counts}")
    print(f"Droplets: {droplet_counts}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Generate configurations
    configs = generate_scene_configurations(num_scenes=101, seed=42)

    # Save to file
    base_dir = Path(r"D:\Pycharm Projects\DashcamRaindropsRemoval")
    config_file = base_dir / "scene_intensity_configs.json"
    save_configurations(configs, config_file)

    # Print summary
    print_configuration_summary(configs)