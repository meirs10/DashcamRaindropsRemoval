# training/train_stage_1.py
"""
Stage 1 training script.

Key properties:
- Encoder fully frozen (including BatchNorm running stats).
- Temporal and perceptual losses OFF at the start (delta = epsilon = 0).
- Per-frame training: every frame from every video is one sample (T = 1).
- Random square crops with multi-scale (256, 384, 512) for both train and val.
- Max epochs = 60
- Epochs 1–10: pixel-only loss
- Epochs 11–20: gradually add SSIM, Edge, Perceptual (linear ramp)
- CosineAnnealingLR until epoch 20, then ReduceLROnPlateau (patience=5)
- Early stopping patience = 12
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
RESUME_TRAINING = True
RESUME_PATH = CHECKPOINT_DIR / "latest_stage1.pth"

# Stage-1 Hyperparameters
BATCH_SIZE = 64
MAX_EPOCHS = 60
LEARNING_RATE = 5e-5           # Reduced LR
FRAMES_PER_CLIP = 1           # T=1 so ConvLSTM has no temporal effect
IMG_SIZE = (512, 512)          # Network input size (after crop/resize)
NUM_WORKERS = 4

# Schedulers + curriculum hyperparams
COSINE_EPOCHS = 20            # use cosine annealing for epochs 1–20
COSINE_LAST_LR = 2e-5
PLATEAU_PATIENCE = 5          # ReduceLROnPlateau patience after epoch 20
EARLY_STOPPING_PATIENCE = 12  # early stopping patience

# Loss ramping: epochs
PIXEL_ONLY_EPOCHS = 10        # 1–10: pixel only
RAMP_END_EPOCH = 20           # at epoch 20: full SSIM/Edge/Perceptual weights

# Target (max) weights for non-pixel losses (you can tweak these)
TARGET_BETA_SSIM = 0.15       # final SSIM weight
TARGET_GAMMA_EDGE = 0.1      # final Edge weight
TARGET_EPSILON_PERC = 0.05    # final Perceptual weight

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
        crop_sizes=[256, 384, 512, 768],
        crop_probs=[0.10, 0.20, 0.60, 0.10],
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
        crop_sizes=[256, 384, 512, 768],
        crop_probs=[0.10, 0.20, 0.60, 0.10],
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
    # Create criterion with target (max) non-pixel weights,
    # but we'll override them dynamically per epoch.
    criterion = CombinedVideoLoss(
        alpha=1.0,                        # pixel (Charbonnier)
        beta=TARGET_BETA_SSIM,            # SSIM (target)
        gamma=TARGET_GAMMA_EDGE,          # Edge (target)
        delta=0.0,                        # Temporal OFF in Stage 1
        epsilon=TARGET_EPSILON_PERC,      # Perceptual (target)
    ).to(device)

    # Store targets and start from pixel-only
    base_beta = criterion.beta
    base_gamma = criterion.gamma
    base_epsilon = criterion.epsilon

    criterion.beta = 0.0
    criterion.gamma = 0.0
    criterion.epsilon = 0.0

    print("Using CombinedVideoLoss with curriculum:")
    print(f"  alpha (pixel):      {criterion.alpha}")
    print(f"  target beta (SSIM): {base_beta}")
    print(f"  target gamma(edge): {base_gamma}")
    print(f"  target epsilon(percept.): {base_epsilon}\n")

    # ==================== OPTIMIZER & SCHEDULERS ====================
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # Cosine annealing for first COSINE_EPOCHS epochs
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=COSINE_EPOCHS,
        eta_min=COSINE_LAST_LR,
    )

    # After COSINE_EPOCHS, switch to ReduceLROnPlateau
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=PLATEAU_PATIENCE,
        threshold=1e-4,
        threshold_mode="rel",
    )

    scaler = GradScaler("cuda")

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    patience_counter = 0
    start_epoch = 0  # index in range(...)

    # ==================== RESUME LOGIC ====================
    if RESUME_TRAINING and RESUME_PATH.exists():
        print(f"\n>>> Resuming from checkpoint: {RESUME_PATH}\n")
        checkpoint = torch.load(RESUME_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load schedulers if present (backward compatible)
        if "cosine_scheduler_state_dict" in checkpoint:
            cosine_scheduler.load_state_dict(checkpoint["cosine_scheduler_state_dict"])
        if "plateau_scheduler_state_dict" in checkpoint:
            plateau_scheduler.load_state_dict(checkpoint["plateau_scheduler_state_dict"])

        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"]  # next epoch index in range()
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])

        if val_losses:
            best_val_loss = min(val_losses)
        else:
            best_val_loss = checkpoint.get("val_loss", float("inf"))

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
        epoch_num = epoch + 1  # human-readable epoch index starting at 1

        # --------- UPDATE LOSS WEIGHTS (CURRICULUM) ----------
        if epoch_num <= PIXEL_ONLY_EPOCHS:
            # pixel only
            ramp_factor = 0.0
        elif epoch_num <= RAMP_END_EPOCH:
            # linear ramp from 0 -> 1 across epochs (PIXEL_ONLY_EPOCHS+1 ... RAMP_END_EPOCH)
            ramp_span = RAMP_END_EPOCH - PIXEL_ONLY_EPOCHS
            ramp_factor = (epoch_num - PIXEL_ONLY_EPOCHS) / ramp_span
        else:
            ramp_factor = 1.0

        if epoch_num == RAMP_END_EPOCH + 1:
            print("Final loss reached, resetting best val loss")
            best_val_loss = float("inf")

        criterion.beta = base_beta * ramp_factor
        criterion.gamma = base_gamma * ramp_factor
        criterion.epsilon = base_epsilon * ramp_factor

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
                    f"Epoch [{epoch_num}/{MAX_EPOCHS}] "
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

        # -------------------- SCHEDULERS --------------------
        # Cosine annealing for first COSINE_EPOCHS epochs, then ReduceLROnPlateau
        if epoch_num <= COSINE_EPOCHS:
            cosine_scheduler.step()
        else:
            plateau_scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        print("\n" + "-" * 60)
        print(f"Epoch [{epoch_num}/{MAX_EPOCHS}] completed")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Time:       {epoch_time:.1f}s")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(
            f"Loss weights -> "
            f"alpha: {criterion.alpha:.3f} | "
            f"beta (SSIM): {criterion.beta:.3f} | "
            f"gamma (Edge): {criterion.gamma:.3f} | "
            f"epsilon (Perc): {criterion.epsilon:.3f}"
        )
        print("-" * 60 + "\n")

        # -------------------- CHECKPOINTING --------------------
        checkpoint = {
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "cosine_scheduler_state_dict": cosine_scheduler.state_dict(),
            "plateau_scheduler_state_dict": plateau_scheduler.state_dict(),
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
        elif epoch_num > RAMP_END_EPOCH:
            patience_counter += 1

        # Occasional extra snapshot
        if (epoch_num) % 5 == 0:
            torch.save(checkpoint, CHECKPOINT_DIR / f"stage1_epoch_{epoch_num}.pth")

        # Early stopping (global, but practically kicks in in the later phase)
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
