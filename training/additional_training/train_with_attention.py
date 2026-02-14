# training/train_with_attention.py
"""
Stage 2.5 training script – temporal + attention fine-tuning.

Key properties:
- Starts from Stage 2 checkpoint: checkpoints/best_stage2.pth
- Uses video clips (per_frame = False) with temporal loss enabled.
- 512x512 crops only (no multi-scale).
- New crapification is generated at the START and every 3 epochs via
  train_crapification_pipeline.main().
- Loss weights (initial):
    alpha  = 1.0   (pixel / Charbonnier)
    beta   = 0.15  (SSIM)
    gamma  = 0.10  (Edge)
    delta  = 0.0   (Temporal, will be ramped)
    epsilon= 0.05  (Perceptual)
- Temporal loss weight delta is increased linearly from epoch 1 to 10
  up to a max of 0.5, then kept constant.
- Uses CosineAnnealingLR for 15 epochs.
- Trains for at most 15 epochs.
- Saves checkpoints:
    checkpoints/latest_with_attention.pth
    checkpoints/best_with_attention.pth
"""
import gc
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

BASE = Path(__file__).parent.parent
# Make project root and training package importable
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "training"))

from model import MobileNetV3UNetConvLSTMVideo
from dataset import RainRemovalDataset
from losses import CombinedVideoLoss
from crapification.per_video_crapification_pipeline import main as run_crapification


# Paths
CLEAN_DATA = BASE / "data"
RAINY_DATA = BASE / "data_after_crapification_per_video"  # video-based crapified data
CHECKPOINT_DIR = BASE / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

STAGE2_START_CKPT = CHECKPOINT_DIR / "stage2" / "best_stage2.pth"

LATEST_ATTENTION_CKPT = CHECKPOINT_DIR / "latest_with_attention.pth"
BEST_ATTENTION_CKPT = CHECKPOINT_DIR / "best_with_attention.pth"

# Hyperparameters
BATCH_SIZE = 1
MAX_EPOCHS = 15
LEARNING_RATE = 5e-5
FRAMES_PER_CLIP = 128        # T > 1 for temporal training
IMG_SIZE = (512, 512)
NUM_WORKERS = 4

# Temporal loss ramp
DELTA_MAX = 0.5
DELTA_RAMP_END_EPOCH = 10  # (1..10 inclusive) – in 1-based indexing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_dataloaders():
    """
    Helper to (re)create train/val datasets and loaders.
    Called initially and after each new crapification.
    """

    print("\nCreating datasets (video mode, 512x512 crops only)...")

    # Train: video clips, consecutive frames, 512x512 crops only
    train_dataset = RainRemovalDataset(
        clean_base_dir=CLEAN_DATA,
        rainy_base_dir=RAINY_DATA,
        num_scenes=101,
        frames_per_clip=FRAMES_PER_CLIP,
        consecutive_frames=True,
        img_size=IMG_SIZE,
        split="train",
        train_ratio=0.8,
        val_ratio=0.1,
        per_frame=False,          # <-- video-based
        random_crop=True,
        crop_sizes=[256, 384, 512],
        crop_probs=[0.15, 0.25, 0.60],
    )

    # Val: same config for consistent distribution
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
        per_frame=False,
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

    print(f"Train clips: {len(train_dataset)} -> batches: {len(train_loader)}")
    print(f"Val clips:   {len(val_dataset)}   -> batches: {len(val_loader)}\n")

    return train_loader, val_loader


def compute_delta_for_epoch(epoch_index: int) -> float:
    """
    Compute temporal loss weight delta for a given epoch index (0-based).
    Requirement:
      - Increase delta linearly from epoch 1 to 10 (1-based) up to max 0.5
      - Then keep at 0.5

    epoch_index: 0-based (0..MAX_EPOCHS-1)
    """
    epoch_1_based = epoch_index + 1

    if epoch_1_based <= DELTA_RAMP_END_EPOCH:
        # Linear ramp: epoch=1 -> 0.05 ... epoch=10 -> 0.5
        return DELTA_MAX * (epoch_1_based / DELTA_RAMP_END_EPOCH)
    else:
        return DELTA_MAX


def main():
    # ==================== INITIAL CRAPIFICATION ====================
    print("=" * 60)

    # ==================== DATASETS / LOADERS ====================
    train_loader, val_loader = create_dataloaders()

    # ==================== MODEL ====================
    print("Initializing model (with attention, Stage 2 weights)...")
    model = MobileNetV3UNetConvLSTMVideo(
        hidden_dim=96,
        out_channels=3,
        use_pretrained_encoder=True,
        freeze_encoder=False,  # Stage 2 / attention fine-tuning: encoder unfrozen
    ).to(device)

    # Initialize lazy layers with dummy input (T = FRAMES_PER_CLIP)
    print("Initializing lazy layers with dummy input...")
    with torch.no_grad():
        dummy = torch.randn(1, FRAMES_PER_CLIP, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device)
        _ = model(dummy)
        del dummy
    print("✓ Lazy layers initialized\n")

    if hasattr(model, "print_param_summary"):
        model.print_param_summary()

    # Load Stage 2 starting checkpoint
    if STAGE2_START_CKPT.exists():
        print(f"\nLoading Stage 2 starting weights from: {STAGE2_START_CKPT}")
        ckpt = torch.load(STAGE2_START_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        print("✓ Stage 2 weights loaded\n")
    else:
        print(f"\nWARNING: Stage 2 checkpoint not found at {STAGE2_START_CKPT}")
        print("Starting from current model initialization instead.\n")

    # ==================== LOSS ====================
    criterion = CombinedVideoLoss(
        alpha=1.0,   # pixel (Charbonnier)
        beta=0.15,   # SSIM
        gamma=0.10,  # Edge
        delta=0.0,   # Temporal (will be ramped)
        epsilon=0.05  # Perceptual
    ).to(device)

    print("Using CombinedVideoLoss (video training with temporal ramp):")
    print(f"  alpha (pixel):      {criterion.alpha}")
    print(f"  beta  (SSIM):       {criterion.beta}")
    print(f"  gamma (edge):       {criterion.gamma}")
    print(f"  delta (temporal):   {criterion.delta}")
    print(f"  epsilon (percept.): {criterion.epsilon}\n")

    # ==================== OPTIMIZER & SCHEDULER (COSINE ANNEALING) ====================
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS,  # full cosine schedule over 15 epochs
        eta_min=0.0,       # you can set a floor if you want, e.g. 1e-6
    )

    scaler = GradScaler("cuda" if device.type == "cuda" else "cpu")

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    start_epoch = 0  # 0-based index into range()

    # Optional resume from attention checkpoint (if you enable it manually)
    RESUME_TRAINING = False
    RESUME_PATH = LATEST_ATTENTION_CKPT

    if RESUME_TRAINING and RESUME_PATH.exists():
        print(f"\n>>> Resuming with-attention training from: {RESUME_PATH}\n")
        checkpoint = torch.load(RESUME_PATH, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"]
        train_losses = checkpoint.get("train_losses", [])
        val_losses = checkpoint.get("val_losses", [])

        if val_losses:
            best_val_loss = min(val_losses)
        else:
            best_val_loss = checkpoint.get("val_loss", float("inf"))

        print(
            f"Resumed at epoch={checkpoint['epoch']} "
            f"(best_val_loss={best_val_loss:.6f})\n"
        )

    print("=" * 60)
    print("STARTING TRAINING WITH ATTENTION (VIDEO + TEMPORAL LOSS)")
    print("=" * 60)

    for epoch in range(start_epoch, MAX_EPOCHS):
        epoch_start = time.time()

        # -------------------- TEMPORAL WEIGHT RAMP --------------------
        new_delta = compute_delta_for_epoch(epoch)
        criterion.delta = new_delta

        print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS} – setting temporal delta = {criterion.delta:.4f}")

        # -------------------- PERIODIC CRAPIFICATION --------------------
        # At the beginning (already done) and every 3 epochs
        if epoch != 0 and (epoch % 3 == 0):
            print("=" * 60)
            print(f"Epoch {epoch + 1}: regenerating crapification...")
            print("=" * 60)
            run_crapification()
            print("✓ Crapification regenerated. Rebuilding dataloaders...")
            train_loader, val_loader = create_dataloaders()

        # -------------------- TRAIN --------------------
        model.train()
        running_train_loss = 0.0

        for batch_idx, (rainy, clean) in enumerate(train_loader):
            rainy = rainy.to(device)   # (B, T, 3, H, W)
            clean = clean.to(device)   # (B, T, 3, H, W)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                output = model(rainy)
                loss, loss_dict = criterion(output, clean)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
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

                with autocast(device_type=device.type):
                    output = model(rainy)
                    loss, _ = criterion(output, clean)

                running_val_loss += loss.item()

        val_loss = running_val_loss / len(val_loader)
        val_losses.append(val_loss)

        # Step cosine scheduler once per epoch (independent of loss)
        scheduler.step()

        epoch_time = time.time() - epoch_start

        print("\n" + "-" * 60)
        print(f"Epoch [{epoch + 1}/{MAX_EPOCHS}] completed")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"Time:       {epoch_time:.1f}s")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"delta (temp): {criterion.delta:.4f}")
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

        # latest attention checkpoint
        torch.save(checkpoint, LATEST_ATTENTION_CKPT)

        # best attention checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, BEST_ATTENTION_CKPT)
            print(f"✓ New BEST with-attention model saved (val_loss={val_loss:.6f})\n")

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("=" * 60)
    print("WITH-ATTENTION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"  Latest: {LATEST_ATTENTION_CKPT.name}")
    print(f"  Best:   {BEST_ATTENTION_CKPT.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
