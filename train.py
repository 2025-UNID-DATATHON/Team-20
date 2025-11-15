"""
train.py
Training pipeline with config-based setup
"""
import os
import random
import argparse
from typing import Dict, Any

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from preprocess import make_loader, Vocab
from model import build_model, freeze_encoder, get_trainable_params, giou_loss, compute_iou, compute_giou


# ===== Utilities =====
def seed_everything(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model: torch.nn.Module,
    vocab: Vocab,
    config: Dict[str, Any],
    save_path: str
):
    """Save model checkpoint with vocab and config."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocab_itos": vocab.itos,
        "config": config,
    }, save_path)
    print(f"[Saved] {save_path}")


# ===== Validation Function =====
def validate(model, val_dl, device, use_amp=False, use_giou=True):
    """Validation loop to compute average loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    ious = []
    gious = []

    with torch.no_grad():
        for imgs, ids, lens, targets, meta in val_dl:
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)

            # Filter out samples without target
            valid_idx = [i for i, t in enumerate(targets) if t is not None]
            if not valid_idx:
                continue

            imgs = imgs[valid_idx]
            ids = ids[valid_idx]
            lens = lens[valid_idx]
            t = torch.stack([targets[i] for i in valid_idx], dim=0).to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(imgs, ids, lens)
                if use_giou:
                    loss = giou_loss(pred, t, reduction="mean")
                else:
                    loss = F.smooth_l1_loss(pred, t, reduction="mean")

                # Compute IoU and GIoU metrics
                iou = compute_iou(pred, t)
                giou = compute_giou(pred, t)
                ious.extend(iou.cpu().numpy().tolist())
                gious.extend(giou.cpu().numpy().tolist())

            total_loss += loss.item() * len(valid_idx)
            total_samples += len(valid_idx)

    model.train()
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    metrics = {
        'loss': avg_loss,
        'mean_iou': np.mean(ious) if ious else 0.0,
        'mean_giou': np.mean(gious) if gious else 0.0,
        'iou@0.5': np.mean([iou >= 0.5 for iou in ious]) if ious else 0.0,
        'iou@0.75': np.mean([iou >= 0.75 for iou in ious]) if ious else 0.0,
    }
    return metrics


# ===== Training Loop =====
def train_loop(config: Dict[str, Any], args):
    """Main training loop with validation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Check if multiple GPUs available
    num_gpus = torch.cuda.device_count()
    print(f"[GPU] {num_gpus} GPU(s) available")
    if num_gpus > 0:
        print(f"[GPU] Using {torch.cuda.get_device_name(0)}")

    # Data
    print("[Data] Building dataset and vocab...")

    # Check if using multiple directories (report + press)
    if "train_json_dirs" in config["data"]:
        # Multiple directories
        result = make_loader(
            json_dirs=config["data"]["train_json_dirs"],
            jpg_dirs=config["data"]["train_jpg_dirs"],
            vocab=None,
            build_vocab=True,
            batch_size=config["training"]["batch_size"],
            img_size=config["data"]["img_size"],
            num_workers=config["data"]["num_workers"],
            shuffle=True
        )
        train_ds, train_dl, vocab = result
    else:
        # Single directory (backward compatibility)
        train_ds, train_dl = make_loader(
            json_dir=config["data"]["train_json_dir"],
            jpg_dir=config["data"]["train_jpg_dir"],
            vocab=None,
            build_vocab=True,
            batch_size=config["training"]["batch_size"],
            img_size=config["data"]["img_size"],
            num_workers=config["data"]["num_workers"],
            shuffle=True
        )

        # Resolve vocab (handle Subset wrapper)
        if isinstance(train_ds, Subset):
            vocab = train_ds.dataset.vocab
        else:
            vocab = train_ds.vocab

    print(f"[Vocab] Size: {len(vocab)}")
    print(f"[Data] Train samples: {len(train_ds)}")

    # Validation data
    val_dl = None
    if "val_json_dirs" in config["data"]:
        # Multiple directories
        print("[Data] Building validation dataset...")
        val_ds, val_dl = make_loader(
            json_dirs=config["data"]["val_json_dirs"],
            jpg_dirs=config["data"]["val_jpg_dirs"],
            vocab=vocab,
            build_vocab=False,
            batch_size=config["training"]["batch_size"],
            img_size=config["data"]["img_size"],
            num_workers=config["data"]["num_workers"],
            shuffle=False
        )
        print(f"[Data] Validation samples: {len(val_ds)}")
    elif "val_json_dir" in config["data"]:
        # Single directory
        print("[Data] Building validation dataset...")
        val_ds, val_dl = make_loader(
            json_dir=config["data"]["val_json_dir"],
            jpg_dir=config["data"]["val_jpg_dir"],
            vocab=vocab,
            build_vocab=False,
            batch_size=config["training"]["batch_size"],
            img_size=config["data"]["img_size"],
            num_workers=config["data"]["num_workers"],
            shuffle=False
        )
        print(f"[Data] Validation samples: {len(val_ds)}")

    # Model
    print("[Model] Building model...")
    model = build_model(config["model"], vocab_size=len(vocab))
    model = model.to(device)

    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[GPU] Using DataParallel with {num_gpus} GPUs")

    # Freeze encoders if specified
    if config["model"].get("freeze_vision_encoder", False):
        freeze_encoder(model, "vision")
    if config["model"].get("freeze_text_encoder", False):
        freeze_encoder(model, "text")

    # Optimizer
    print("[Optimizer] Setting up optimizer...")
    param_groups = get_trainable_params(model)

    # Build parameter groups with different learning rates
    optimizer_params = []
    lr = config["training"]["learning_rate"]
    adapter_lr_scale = config["training"].get("adapter_lr_scale", 1.0)

    if param_groups["adapter"]:
        optimizer_params.append({
            "params": param_groups["adapter"],
            "lr": lr * adapter_lr_scale,
            "name": "adapter"
        })
    if param_groups["fusion"]:
        optimizer_params.append({
            "params": param_groups["fusion"],
            "lr": lr,
            "name": "fusion"
        })
    if param_groups["head"]:
        optimizer_params.append({
            "params": param_groups["head"],
            "lr": lr,
            "name": "head"
        })
    if param_groups["other"]:
        optimizer_params.append({
            "params": param_groups["other"],
            "lr": lr,
            "name": "other"
        })

    if not optimizer_params:
        raise RuntimeError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=lr,
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"]
    )

    # Mixed precision
    use_amp = config["training"].get("use_amp", True) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Gradient accumulation
    grad_accum_steps = config["training"].get("gradient_accumulation_steps", 1)
    grad_clip = config["training"].get("grad_clip", None)
    use_giou = config["training"].get("use_giou_loss", True)
    print(f"[Training] Gradient accumulation steps: {grad_accum_steps}")
    print(f"[Training] Using GIoU loss: {use_giou}")
    if grad_clip:
        print(f"[Training] Gradient clipping: {grad_clip}")

    # Training
    total_samples = len(train_ds)
    best_loss = float('inf')
    best_val_loss = float('inf')
    early_stopping_patience = config["training"].get("early_stopping_patience", None)
    patience_counter = 0

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (imgs, ids, lens, targets, meta) in enumerate(train_dl):
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)

            # Filter out samples without target
            valid_idx = [i for i, t in enumerate(targets) if t is not None]
            if not valid_idx:
                continue

            imgs = imgs[valid_idx]
            ids = ids[valid_idx]
            lens = lens[valid_idx]
            t = torch.stack([targets[i] for i in valid_idx], dim=0).to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(imgs, ids, lens)
                if use_giou:
                    loss = giou_loss(pred, t, reduction="mean")
                else:
                    loss = F.smooth_l1_loss(pred, t, reduction="mean")
                loss = loss / grad_accum_steps  # Scale loss for gradient accumulation

            scaler.scale(loss).backward()

            # Gradient accumulation: update every N steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                if grad_clip:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * len(valid_idx) * grad_accum_steps

        scheduler.step()

        avg_loss = running_loss / total_samples
        lr_current = scheduler.get_last_lr()[0]

        # Validation
        val_metrics = None
        if val_dl is not None and epoch % config["training"].get("eval_interval", 1) == 0:
            val_metrics = validate(model, val_dl, device, use_amp, use_giou)
            print(f"[Epoch {epoch}/{config['training']['epochs']}] "
                  f"train_loss={avg_loss:.4f} val_loss={val_metrics['loss']:.4f} "
                  f"val_mIoU={val_metrics['mean_iou']:.4f} val_mGIoU={val_metrics['mean_giou']:.4f} "
                  f"IoU@0.5={val_metrics['iou@0.5']:.4f} lr={lr_current:.6f}")
        else:
            print(f"[Epoch {epoch}/{config['training']['epochs']}] "
                  f"train_loss={avg_loss:.4f} lr={lr_current:.6f}")

        # Save best model based on validation loss if available, otherwise training loss
        if val_metrics is not None:
            val_loss = val_metrics['loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = config["training"]["save_ckpt"].replace(".pth", "_best.pth")
                model_to_save = model.module if num_gpus > 1 else model
                save_checkpoint(model_to_save, vocab, config, save_path)
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = config["training"]["save_ckpt"].replace(".pth", "_best.pth")
                model_to_save = model.module if num_gpus > 1 else model
                save_checkpoint(model_to_save, vocab, config, save_path)

        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"[Early Stopping] No improvement for {early_stopping_patience} epochs. Stopping training.")
            break

        # Save checkpoint at intervals
        if epoch % config["training"].get("save_interval", 5) == 0:
            save_path = config["training"]["save_ckpt"].replace(".pth", f"_epoch{epoch}.pth")
            model_to_save = model.module if num_gpus > 1 else model
            save_checkpoint(model_to_save, vocab, config, save_path)

    # Save final model
    model_to_save = model.module if num_gpus > 1 else model
    save_checkpoint(model_to_save, vocab, config, config["training"]["save_ckpt"])
    if val_metrics is not None:
        print(f"[Training] Completed. Best val_loss: {best_val_loss:.4f}")
    else:
        print(f"[Training] Completed. Best train_loss: {best_loss:.4f}")


# ===== Main =====
def main():
    parser = argparse.ArgumentParser(description="Train document layout detector")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set seed
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    seed_everything(seed)
    print(f"[Seed] {seed}")

    # Train
    train_loop(config, args)


if __name__ == "__main__":
    main()
