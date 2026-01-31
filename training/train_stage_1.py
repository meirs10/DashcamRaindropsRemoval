# training/train_stage_1.py
"""
Stage 1 training script.

Key properties:
- Encoder fully frozen (including BatchNorm running stats).
- Temporal and perceptual losses OFF at the start (delta = epsilon = 0).
- Per-frame training: every frame from every video is one sample (T = 1).
- Random square crops with multi-scale (256, 384, 512) for both train and val.
- Max epochs = 15, ReduceLROnPlateau + early stopping.
- Perceptual loss enabled with epsilon = 0.05 starting from epoch 7.
- Checkpoints saved so Stage 2 can resume from best_stage1.pth.
"""

import sys
import time
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from model import MobileNetV3UNetConvLSTMVideo
from dataset import RainRemovalDataset
from losses import CombinedVideoLoss

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "training"))

# Paths
CLEAN_DATA = BASE / "data"
RAINY_DATA = BASE / "data_after_crapification_per_frame"
CHECKPOINT_DIR = BASE / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Resume from checkpoint flag
RESUME_TRAINING = False
RESUME_PATH = CHECKPOINT_DIR / "latest_stage1.pth"

# Stage-1 Hyperparameters
BATCH_SIZE = 64
MAX_EPOCHS = 50
LEARNING_RATE = 5e-5           # Reduced LR (many more updates now)
FRAMES_PER_CLIP = 1           # T=1 so ConvLSTM has no temporal effect
IMG_SIZE = (512, 512)          # Network input size (after crop/resize)
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 16
SCHEDULER_PATIENCE = 7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def main():
    # ==================== DATASETS ====================
    print("\nCreating datasets (Stage 1, per-frame)...")

    # Train: every frame is a sample, random multi-scale square crops
    train_dataset = RainRemovalDataset(
        clean_base_dir=CLEAN_DATA,
        rainy_base_dir=RAINY_DATA,
        num_scenes=101,
        frames_per_clip=FRAMES_PER_CLIP,
        consecutive_frames=True,  # ignored in per_frame mode
        img_size=IMG_SIZE,
        split="train",
        train_ratio=0.8,
        val_ratio=0.1,
        per_frame=True,
        random_crop=True,
        crop_sizes=[256, 384, 512],
        crop_probs=[0.15, 0.25, 0.60],
    )

    # Val: same per-frame logic + same crop behaviour for consistent distribution
    val_dataset = RainRemovalDataset(
        clean_base_dir=CLEAN_DATA,
        rainy_base_dir=RAINY_DATA,
        num_scenes=101,
        frames_per_clip=FRAMES_PER_CLIP,
        consecutive_frames=True,
        img_size=IMG_SIZE,
        split="val",
        train_ratio=0.8,
        val_ratio=0.1,
        per_frame=True,
        random_crop=True,
        crop_sizes=[256, 384, 512],
        crop_probs=[0.15, 0.25, 0.60],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples:  {len(train_dataset)}  -> batches: {len(train_loader)}")
    print(f"Val samples:    {len(val_dataset)}    -> batches: {len(val_loader)}")

    # ==================== MODEL ====================
    print("\nInitializing model...")
    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=True,  # encoder weights + BN stats frozen
    ).to(device)

    # Initialize lazy layers with dummy input (T=1)
    print("Initializing lazy layers...")
    with torch.no_grad():
        dummy = torch.randn(1, FRAMES_PER_CLIP, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
        _ = model(dummy)
        del dummy
    print("✓ Lazy layers initialized\n")

    model.print_param_summary()

    # ==================== LOSS ====================
    # Stage 1: temporal and perceptual OFF initially
    criterion = CombinedVideoLoss(
        alpha=1.0,   # pixel (Charbonnier)
        beta=0.0,    # SSIM
        gamma=0.0,   # Edge
        delta=0.0,   # Temporal OFF
        epsilon=0.0  # Perceptual OFF initially
    ).to(device)

    print("Using CombinedVideoLoss (Stage 1 warmup):")
    print(f"  alpha (pixel):      {criterion.alpha}")
    print(f"  beta  (SSIM):       {criterion.beta}")
    print(f"  gamma (edge):       {criterion.gamma}")
    print(f"  delta (temporal):   {criterion.delta}")
    print(f"  epsilon (percept.): {criterion.epsilon}\n")

    # ==================== OPTIMIZER & SCHEDULER ====================
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=SCHEDULER_PATIENCE,
        threshold=1e-4,
        threshold_mode="rel",
    )

    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    patience_counter = 0
    start_epoch = 0  # index in range(...)

    if RESUME_TRAINING and RESUME_PATH.exists():
        print(f"\n>>> Resuming from checkpoint: {RESUME_PATH}\n")
        checkpoint = torch.load(RESUME_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # epoch in checkpoint is 1-based (epoch + 1), so we can resume from that index
        start_epoch = checkpoint["epoch"]  # next epoch index in range()
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])

        if val_losses:
            best_val_loss = min(val_losses)
        else:
            best_val_loss = checkpoint.get("val_loss", float("inf"))

        # optionally reset patience so early stopping doesn’t instantly trigger
        patience_counter = 0

        print(
            f"Resumed at epoch={checkpoint['epoch']} "
            f"(best_val_loss={best_val_loss:.6f})\n"
        )

    print("=" * 60)
    print("STARTING STAGE 1 TRAINING (PER-FRAME)")
    print("=" * 60)

    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start = time.time()

        # -------------------- TRAIN --------------------
        model.train()
        running_train_loss = 0.0

        for batch_idx, (rainy, clean) in enumerate(train_loader):
            rainy = rainy.to(device)   # (B, 1, 3, H, W)
            clean = clean.to(device)   # (B, 1, 3, H, W)

            optimizer.zero_grad()

            with autocast("cuda"):
                output = model(rainy)
                loss, loss_dict = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{MAX_EPOCHS}] "
                    f"Batch [{batch_idx + 1}/{len(train_loader)}]"
                )
                print(
                    f"  Total: {loss_dict['total']:.4f} | "
                    f"Pixel: {loss_dict['pixel']:.4f} | "
                    f"SSIM: {loss_dict['ssim']:.4f} | "
                    f"Edge: {loss_dict['edge']:.4f} | "
                    f"Temp: {loss_dict['temporal']:.4f} | "
                    f"Perc: {loss_dict['perceptual']:.4f}"
                )

        train_loss = running_train_loss / len(train_loader)
        train_losses.append(train_loss)

        # -------------------- VALIDATION --------------------
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for rainy, clean in val_loader:
                rainy = rainy.to(device)
                clean = clean.to(device)

                with autocast("cuda"):
                    output = model(rainy)
                    loss, _ = criterion(output, clean)

                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        print("\n" + "-" * 60)
        print(f"Epoch [{epoch + 1}/{MAX_EPOCHS}] completed")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Time:       {epoch_time:.1f}s")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"epsilon:    {criterion.epsilon:.4f}")
        print("-" * 60 + "\n")

        # -------------------- CHECKPOINTING --------------------
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        # latest
        torch.save(checkpoint, CHECKPOINT_DIR / "latest_stage1.pth")

        # best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, CHECKPOINT_DIR / "best_stage1.pth")
            print(f"✓ New best Stage-1 model saved (val_loss={val_loss:.6f})\n")
        else:
            patience_counter += 1

        # Occasional extra snapshot
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, CHECKPOINT_DIR / f"stage1_epoch_{epoch + 1}.pth")

        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered (Stage 1).")
            break

    print("=" * 60)
    print("STAGE 1 TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("Use best_stage1.pth as the starting point for Stage 2.")
    print("=" * 60)


if __name__ == "__main__":
    main()
