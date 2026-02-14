# training/train_stage_2.py
"""
Stage 2 fine-tuning script (multi-stage sharpening: SSIM + Edge + Perceptual).

Goal:
- Start from best_stage1.pth (trained with pixel loss).
- Keep encoder frozen.
- Gradually introduce:
    * SSIM loss (early)
    * Edge loss (mid)
    * Perceptual loss (late)

Schedules (1-based epoch numbering):
- Total epochs: 35
- SSIM:
    - Gradually increase in epochs 1–5
    - MAX_SSIM reached at epoch 6
    - Epoch 6+ trained with MAX_SSIM
- Edge:
    - Gradually increase in epochs 11–15
    - MAX_EDGE reached at epoch 16
    - Epoch 16+ trained with MAX_EDGE
- Perceptual:
    - Gradually increase in epochs 21–25
    - MAX_PERCEPTUAL reached at epoch 26
    - Epoch 26+ trained with MAX_PERCEPTUAL

Key properties:
- Per-frame training (T = 1), same dataset setup as Stage 1.
- Random multi-scale crops (256, 384, 512) for train and val.
- Charbonnier loss always ON (alpha > 0).
- Temporal loss OFF (delta = 0.0).

Losses used for checkpointing / curves:
- For train & val, we always record a metric with the *final* max weights:
    total_max = pixel + MAX_SSIM * ssim + MAX_EDGE * edge + MAX_PERCEPTUAL * perceptual
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

# Paths
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "training"))
CLEAN_DATA = BASE / "data_original"
RAINY_DATA = BASE / "data_after_crapification_per_frame"
CHECKPOINT_DIR = BASE / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

BEST_STAGE1_PATH = CHECKPOINT_DIR / "stage1" / "best_stage1.pth"

# Fine-tuning Hyperparameters (Stage 2)
BATCH_SIZE = 64
MAX_EPOCHS = 35
LEARNING_RATE = 2e-5          # smaller LR for fine-tuning
FRAMES_PER_CLIP = 1
IMG_SIZE = (512, 512)
NUM_WORKERS = 4

# Max weights for losses (configurable)
SSIM_MAX = 0.15          # beta_max
EDGE_MAX = 0.1          # gamma_max
PERCEPTUAL_MAX = 0.05    # epsilon_max

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def ramp_weight(epoch_idx: int, start_epoch: int, end_epoch: int, max_value: float) -> float:
    """
    Generic linear ramp for a weight.

    Epoch indexing here:
    - epoch_idx is 0-based
    - we convert to e = epoch_idx + 1 (1-based)

    Behavior:
    - For e <= start_epoch: 0
    - For start_epoch < e <= end_epoch: linearly increase from 0 -> <max_value
    - For e >= end_epoch + 1: max_value

    Example (start=1, end=5):
    - e=1 -> 0
    - e=2..5 -> ramp
    - e>=6 -> max_value
    """
    e = epoch_idx + 1  # 1-based
    if e < start_epoch:
        return 0.0
    if e > end_epoch:
        return max_value
    # Linear ramp between (start_epoch, end_epoch)
    n_increasement_epochs = (end_epoch + 1) - start_epoch
    curr_increasement_epoch = (e + 1) - start_epoch
    progress = curr_increasement_epoch / float(n_increasement_epochs + 1)
    progress = min(1.0, progress)
    return max_value * progress


def get_current_ssim_weight(epoch_idx: int) -> float:
    """
    SSIM schedule:
    - Gradual in epochs 1–5
    - MAX at epoch 6+
    """
    return ramp_weight(epoch_idx, start_epoch=1, end_epoch=5, max_value=SSIM_MAX)


def get_current_edge_weight(epoch_idx: int) -> float:
    """
    Edge schedule:
    - Gradual in epochs 11–15
    - MAX at epoch 16+
    """
    return ramp_weight(epoch_idx, start_epoch=11, end_epoch=15, max_value=EDGE_MAX)


def get_current_perceptual_weight(epoch_idx: int) -> float:
    """
    Perceptual schedule:
    - Gradual in epochs 21–25
    - MAX at epoch 26+
    """
    return ramp_weight(epoch_idx, start_epoch=21, end_epoch=25, max_value=PERCEPTUAL_MAX)


def main():
    # ==================== DATASETS ====================
    print("\nCreating datasets (Stage 2, per-frame)...")

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
    print("\nInitializing model for Stage 2 (fine-tuning from best_stage1)...")

    if not BEST_STAGE1_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {BEST_STAGE1_PATH}. "
            f"Train Stage 1 first to produce best_stage1.pth."
        )

    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=True,  # keep encoder frozen for sharpening
    ).to(device)

    # Initialize lazy layers
    print("Initializing lazy layers...")
    with torch.no_grad():
        dummy = torch.randn(1, FRAMES_PER_CLIP, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
        _ = model(dummy)
        del dummy
    print("✓ Lazy layers initialized")

    # Load Stage-1 checkpoint weights
    print(f"Loading weights from: {BEST_STAGE1_PATH}")
    ckpt = torch.load(BEST_STAGE1_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print("✓ Weights loaded from Stage 1\n")

    model.print_param_summary()

    # ==================== LOSS ====================
    # Start with only pixel loss; other weights will be updated every epoch.
    criterion = CombinedVideoLoss(
        alpha=1.0,   # pixel (Charbonnier)
        beta=0.0,    # SSIM (will ramp to SSIM_MAX)
        gamma=0.0,   # Edge (will ramp to EDGE_MAX)
        delta=0.0,   # Temporal OFF
        epsilon=0.0  # Perceptual (will ramp to PERCEPTUAL_MAX)
    ).to(device)

    print("Using CombinedVideoLoss (Stage 2 fine-tuning):")
    print(f"  alpha (pixel):        {criterion.alpha}")
    print(f"  beta  (SSIM)  max:    {SSIM_MAX}")
    print(f"  gamma (edge)  max:    {EDGE_MAX}")
    print(f"  epsilon (percept.) max:{PERCEPTUAL_MAX}")
    print(f"  delta (temporal):     {criterion.delta}\n")

    # ==================== OPTIMIZER & SCHEDULER ====================
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    # Cosine annealing over the full 35 epochs – common for fine-tuning
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,
        eta_min=5e-6,
    )

    scaler = GradScaler("cuda")

    # We track "best" using the fixed max-weight metric:
    # total_max = pixel + SSIM_MAX * ssim + EDGE_MAX * edge + PERCEPTUAL_MAX * perceptual
    best_val_loss = float("inf")
    best_epoch = 0

    train_losses = []
    val_losses = []

    print("=" * 60)
    print("STARTING STAGE 2 FINE-TUNING (PIXEL + SSIM + EDGE + PERCEPTUAL)")
    print("=" * 60)

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        epoch_num = epoch + 1

        # ----- Update loss weights for this epoch -----
        current_beta = get_current_ssim_weight(epoch)
        current_gamma = get_current_edge_weight(epoch)
        current_epsilon = get_current_perceptual_weight(epoch)

        criterion.beta = current_beta
        criterion.gamma = current_gamma
        criterion.epsilon = current_epsilon

        print(
            f"\nEpoch {epoch_num}/{MAX_EPOCHS} "
            f"(beta/SSIM={current_beta:.4f}, "
            f"gamma/Edge={current_gamma:.4f}, "
            f"epsilon/Perc={current_epsilon:.4f})"
        )

        # -------------------- TRAIN --------------------
        model.train()
        running_train_loss_max = 0.0

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

            # --- Metric with final max weights (for curves/checkpoints) ---
            pixel_loss = loss_dict["pixel"]
            ssim_loss = loss_dict["ssim"]
            edge_loss = loss_dict["edge"]
            perc_loss = loss_dict["perceptual"]

            total_max = (
                1.0 * pixel_loss
                + SSIM_MAX * ssim_loss
                + EDGE_MAX * edge_loss
                + PERCEPTUAL_MAX * perc_loss
            )

            running_train_loss_max += float(total_max)

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                    f"Total(curr): {loss_dict['total']:.4f} | "
                    f"Pixel: {pixel_loss:.4f} | "
                    f"SSIM: {ssim_loss:.4f} | "
                    f"Edge: {edge_loss:.4f} | "
                    f"Perc: {perc_loss:.4f}"
                )

        train_loss = running_train_loss_max / len(train_loader)
        train_losses.append(train_loss)

        # -------------------- VALIDATION --------------------
        model.eval()
        running_val_loss_max = 0.0

        with torch.no_grad():
            for rainy, clean in val_loader:
                rainy = rainy.to(device)
                clean = clean.to(device)

                with autocast("cuda"):
                    output = model(rainy)
                    loss, loss_dict = criterion(output, clean)

                # Same fixed max-weight metric as in train
                pixel_loss = loss_dict["pixel"]
                ssim_loss = loss_dict["ssim"]
                edge_loss = loss_dict["edge"]
                perc_loss = loss_dict["perceptual"]

                total_max = (
                    1.0 * pixel_loss
                    + SSIM_MAX * ssim_loss
                    + EDGE_MAX * edge_loss
                    + PERCEPTUAL_MAX * perc_loss
                )

                running_val_loss_max += float(total_max)

        val_loss = running_val_loss_max / len(val_loader)
        val_losses.append(val_loss)

        # Step LR scheduler (epoch-wise)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print("\n" + "-" * 60)
        print(f"Epoch [{epoch_num}/{MAX_EPOCHS}] completed")
        print(f"Train Loss (max-weight metric): {train_loss:.6f}")
        print(f"Val   Loss (max-weight metric): {val_loss:.6f}")
        print(f"Time:       {epoch_time:.1f}s")
        print(f"LR:         {current_lr:.6f}")
        print(
            f"Weights -> alpha:1.0  beta/SSIM:{criterion.beta:.4f}  "
            f"gamma/Edge:{criterion.gamma:.4f}  epsilon/Perc:{criterion.epsilon:.4f}"
        )
        print("-" * 60 + "\n")

        # -------------------- CHECKPOINTING --------------------
        checkpoint = {
            "epoch": epoch_num,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            # Final-weights loss metrics:
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            # Optional extra info:
            "beta_ssim": criterion.beta,
            "gamma_edge": criterion.gamma,
            "epsilon_perceptual": criterion.epsilon,
            "SSIM_MAX": SSIM_MAX,
            "EDGE_MAX": EDGE_MAX,
            "PERCEPTUAL_MAX": PERCEPTUAL_MAX,
        }

        # latest Stage 2
        torch.save(checkpoint, CHECKPOINT_DIR / "latest_stage2.pth")

        # best Stage 2 – based on max-weight val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch_num
            torch.save(checkpoint, CHECKPOINT_DIR / "best_stage2.pth")
            print(f"✓ New best Stage-2 model saved "
                  f"(val_loss={val_loss:.6f} at epoch {epoch_num})\n")

        # extra snapshots (every 5 epochs)
        if epoch_num % 5 == 0:
            torch.save(
                checkpoint,
                CHECKPOINT_DIR / f"stage2_epoch_{epoch_num}.pth"
            )

    print("=" * 60)
    print("STAGE 2 FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss (max-weight metric): {best_val_loss:.6f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print("Use best_stage2.pth for inference.")
    print("=" * 60)


if __name__ == "__main__":
    main()
