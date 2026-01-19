# training/train.py
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "training"))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # FIXED
import time

from model import MobileNetV3UNetConvLSTMVideo
from dataset import RainRemovalDataset
from losses import CombinedVideoLoss

# Paths
CLEAN_DATA = BASE / "data"
RAINY_DATA = BASE / "data_after_crapification"
CHECKPOINT_DIR = BASE / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Hyperparameters
BATCH_SIZE = 1
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
FRAMES_PER_CLIP = 5
IMG_SIZE = (540, 960)  # Half resolution (16:9)
NUM_WORKERS = 4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def main():
    # Datasets
    print("\nCreating datasets...")
    train_dataset = RainRemovalDataset(
        clean_base_dir=CLEAN_DATA,
        rainy_base_dir=RAINY_DATA,
        num_scenes=101,
        frames_per_clip=FRAMES_PER_CLIP,
        img_size=IMG_SIZE,
        split='train',
        train_ratio=0.8
    )

    val_dataset = RainRemovalDataset(
        clean_base_dir=CLEAN_DATA,
        rainy_base_dir=RAINY_DATA,
        num_scenes=101,
        frames_per_clip=FRAMES_PER_CLIP,
        img_size=IMG_SIZE,
        split='val',
        train_ratio=0.8
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print("\nInitializing model...")
    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=True
    )
    model = model.to(device)

    # Initialize lazy layers
    print("Initializing lazy layers...")
    with torch.no_grad():
        dummy_input = torch.randn(1, FRAMES_PER_CLIP, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
        _ = model(dummy_input)
        del dummy_input
    print("✓ Lazy layers initialized")

    model.print_param_summary()

    # Loss function - FIXED!
    criterion = CombinedVideoLoss(
        alpha=1.0,  # Charbonnier (pixel fidelity)
        beta=0.2,  # SSIM (structural similarity)
        gamma=0.1,  # Edge preservation
        delta=0.5,  # Temporal consistency
        epsilon=0.1  # Perceptual (VGG-based)
    ).to(device)  # ← CRITICAL FIX!

    print("\n✓ Using Combined Loss:")
    print(f"  α (Charbonnier): {criterion.alpha}")
    print(f"  β (SSIM):        {criterion.beta}")
    print(f"  γ (Edge):        {criterion.gamma}")
    print(f"  δ (Temporal):    {criterion.delta}")
    print(f"  ε (Perceptual):  {criterion.epsilon}")

    # Optimizer and scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Mixed precision scaler - FIXED!
    scaler = GradScaler('cuda')

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60 + "\n")

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0

        for batch_idx, (rainy, clean) in enumerate(train_loader):
            rainy = rainy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass - FIXED!
            with autocast('cuda'):
                output = model(rainy)
                loss, loss_dict = criterion(output, clean)

            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # Log detailed losses every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Batch [{batch_idx + 1}/{len(train_loader)}]")
                print(f"  Total: {loss_dict['total']:.4f} | "
                      f"Pixel: {loss_dict['pixel']:.4f} | "
                      f"SSIM: {loss_dict['ssim']:.4f} | "
                      f"Edge: {loss_dict['edge']:.4f} | "
                      f"Temp: {loss_dict['temporal']:.4f} | "
                      f"Perc: {loss_dict['perceptual']:.4f}")

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for rainy, clean in val_loader:
                rainy = rainy.to(device)
                clean = clean.to(device)

                with autocast('cuda'):  # FIXED!
                    output = model(rainy)
                    loss, _ = criterion(output, clean)

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print(f"\n{'=' * 60}")
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Complete")
        print(f"{'=' * 60}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Time:       {epoch_time:.1f}s")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'=' * 60}\n")

        # Save checkpoints
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
        }

        torch.save(checkpoint, CHECKPOINT_DIR / 'latest.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, CHECKPOINT_DIR / 'best.pth')
            print(f"✓ New best model saved! Val Loss: {val_loss:.4f}\n")

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, CHECKPOINT_DIR / f'epoch_{epoch + 1}.pth')

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")


if __name__ == '__main__':
    main()