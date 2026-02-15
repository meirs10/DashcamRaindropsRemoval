# plot_results.py
import torch
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).parent.parent
CHECKPOINT_PATH = BASE / "checkpoints" / "latest_convlstm.pth"

checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

train_losses = checkpoint['train_losses']
val_losses = checkpoint['val_losses']
epochs = range(1, len(train_losses) + 1)

print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Total Epochs:        {checkpoint['epoch']}")
print(f"Final Train Loss:    {checkpoint['train_loss']:.6f}")
print(f"Final Val Loss:      {checkpoint['val_loss']:.6f}")
print(f"Best Val Loss:       {min(val_losses):.6f} at Epoch {val_losses.index(min(val_losses))+1}")
print(f"Train/Val Gap:       {abs(checkpoint['train_loss'] - checkpoint['val_loss']):.6f}")
print("="*60 + "\n")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curve
ax1.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=2, linewidth=2)
ax1.plot(epochs, val_losses, label='Val Loss', marker='s', markersize=2, linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Total Loss', fontsize=12)
ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Train/Val gap
gap = [abs(t - v) for t, v in zip(train_losses, val_losses)]
ax2.plot(epochs, gap, color='purple', marker='o', markersize=2, linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Train-Val Gap', fontsize=12)
ax2.set_title('Overfitting Monitor (Lower = Better)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0.02, color='orange', linestyle='--', label='Acceptable Gap (0.02)')
ax2.legend()

plt.tight_layout()
output_path = BASE / "checkpoints" / "training_results.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved training curve to: {output_path}\n")
plt.show()