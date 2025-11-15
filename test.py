"""
test.py
Evaluation and Prediction pipeline
"""
import os
import argparse
import zipfile
from typing import Dict, Any

import yaml
import numpy as np
import pandas as pd
import torch

from preprocess import make_loader, Vocab
from model import build_model


# ===== Utilities =====
def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_checkpoint(ckpt_path: str, device: torch.device):
    """Load model checkpoint and return model, vocab, config."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Restore vocab
    vocab = Vocab()
    vocab.itos = ckpt["vocab_itos"]
    vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}

    # Restore config
    config = ckpt.get("config", {})

    # Build model
    model = build_model(config.get("model", {}), vocab_size=len(vocab))
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    return model, vocab, config


def iou_xywh_pixel(pred_xywh, gt_xywh):
    """Calculate IoU between two bboxes in pixel coordinates (x, y, w, h)."""
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh

    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh

    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter if (pw * ph + gw * gh - inter) > 0 else 1e-6

    return inter / union


# ===== Evaluation Loop =====
def evaluate_loop(config: Dict[str, Any], args):
    """Evaluation loop with ground truth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load checkpoint
    print(f"[Checkpoint] Loading from {args.ckpt}...")
    model, vocab, ckpt_config = load_checkpoint(args.ckpt, device)

    # Use checkpoint config for data settings
    data_config = ckpt_config.get("data", config.get("data", {}))
    img_size = data_config.get("img_size", 512)
    batch_size = config.get("eval", {}).get("batch_size", 8)
    num_workers = data_config.get("num_workers", 2)

    # Data
    print("[Data] Loading evaluation dataset...")
    eval_ds, eval_dl = make_loader(
        json_dir=args.json_dir or config["eval"]["json_dir"],
        jpg_dir=args.jpg_dir or config["eval"].get("jpg_dir"),
        vocab=vocab,
        build_vocab=False,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        shuffle=False
    )
    print(f"[Data] Eval samples: {len(eval_ds)}")

    # Inference
    rows = []
    ious = []
    print(f"[Inference] Starting evaluation for {len(eval_ds)} samples...")
    print(f"[Inference] Batch size: {batch_size}")
    total_batches = len(eval_dl)

    with torch.no_grad():
        for batch_idx, (imgs, ids, lens, targets, meta) in enumerate(eval_dl):
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)

            pred = model(imgs, ids, lens)  # (B, 4) normalized

            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]

                # Convert to pixel coordinates
                x = (cx - nw / 2.0) * W
                y = (cy - nh / 2.0) * H
                w = nw * W
                h = nh * H

                rows.append({
                    "query_id": meta[i]["query_id"],
                    "query_text": meta[i]["query_text"],
                    "class_name": meta[i]["class_name"],
                    "pred_x": x,
                    "pred_y": y,
                    "pred_w": w,
                    "pred_h": h
                })

                # Calculate IoU if GT available
                if targets[i] is not None:
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gx = (gt[0] - gt[2] / 2.0) * W
                    gy = (gt[1] - gt[3] / 2.0) * H
                    gw = gt[2] * W
                    gh = gt[3] * H
                    iou = iou_xywh_pixel([x, y, w, h], [gx, gy, gw, gh])
                    ious.append(iou)

                    rows[-1]["gt_x"] = gx
                    rows[-1]["gt_y"] = gy
                    rows[-1]["gt_w"] = gw
                    rows[-1]["gt_h"] = gh
                    rows[-1]["iou"] = iou

            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                processed = len(rows)
                avg_iou = np.mean(ious) if ious else 0.0
                print(f"[Progress] Batch {batch_idx + 1}/{total_batches} | Processed {processed}/{len(eval_ds)} samples | Avg IoU: {avg_iou:.4f}")

    # Save results
    out_csv = args.out_csv or config["eval"]["out_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out_csv}")

    # Print metrics
    if ious:
        mean_iou = np.mean(ious)
        print(f"[Eval] mIoU: {mean_iou:.4f}")
        print(f"[Eval] IoU@0.5: {np.mean([iou >= 0.5 for iou in ious]):.4f}")
        print(f"[Eval] IoU@0.75: {np.mean([iou >= 0.75 for iou in ious]):.4f}")
    else:
        print("[Eval] No ground truth found; metrics not computed.")


# ===== Prediction Loop =====
def predict_loop(config: Dict[str, Any], args):
    """Prediction loop without ground truth."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Load checkpoint
    print(f"[Checkpoint] Loading from {args.ckpt}...")
    model, vocab, ckpt_config = load_checkpoint(args.ckpt, device)

    # Use checkpoint config for data settings
    data_config = ckpt_config.get("data", config.get("data", {}))
    img_size = data_config.get("img_size", 512)
    batch_size = config.get("predict", {}).get("batch_size", 8)
    num_workers = data_config.get("num_workers", 2)

    # Data
    print("[Data] Loading test dataset...")
    test_ds, test_dl = make_loader(
        json_dir=args.json_dir or config["predict"]["json_dir"],
        jpg_dir=args.jpg_dir or config["predict"].get("jpg_dir"),
        vocab=vocab,
        build_vocab=False,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=num_workers,
        shuffle=False
    )
    print(f"[Data] Test samples: {len(test_ds)}")

    # Inference
    rows = []
    print(f"[Inference] Starting prediction for {len(test_ds)} samples...")
    print(f"[Inference] Batch size: {batch_size}")
    total_batches = len(test_dl)

    with torch.no_grad():
        for batch_idx, (imgs, ids, lens, targets, meta) in enumerate(test_dl):
            imgs = imgs.to(device)
            ids = ids.to(device)
            lens = lens.to(device)

            pred = model(imgs, ids, lens)  # (B, 4) normalized

            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]

                # Convert to pixel coordinates
                x = (cx - nw / 2.0) * W
                y = (cy - nh / 2.0) * H
                w = nw * W
                h = nh * H

                rows.append({
                    "query_id": meta[i]["query_id"],
                    "query_text": meta[i]["query_text"],
                    "pred_x": x,
                    "pred_y": y,
                    "pred_w": w,
                    "pred_h": h
                })

            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                processed = len(rows)
                print(f"[Progress] Batch {batch_idx + 1}/{total_batches} | Processed {processed}/{len(test_ds)} samples")

    # Save results
    out_csv = args.out_csv or config["predict"]["out_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out_csv}")


# ===== Submission Zip =====
def zip_submission(csv_path: str, zip_path: str):
    """Create submission zip file."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        arcname = os.path.basename(csv_path)
        zf.write(csv_path, arcname=arcname)

    print(f"[Submission] Zipped {csv_path} -> {zip_path}")


# ===== Main =====
def main():
    parser = argparse.ArgumentParser(description="Evaluate or predict with document layout detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_common_args(p):
        p.add_argument("--config", type=str, required=True, help="Path to config YAML file")
        p.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
        p.add_argument("--json_dir", type=str, default=None, help="JSON directory (overrides config)")
        p.add_argument("--jpg_dir", type=str, default=None, help="JPG directory (overrides config)")
        p.add_argument("--out_csv", type=str, default=None, help="Output CSV path (overrides config)")

    # Eval
    eval_parser = subparsers.add_parser("eval", help="Evaluate with ground truth")
    add_common_args(eval_parser)

    # Predict
    predict_parser = subparsers.add_parser("predict", help="Predict without ground truth")
    add_common_args(predict_parser)

    # Zip
    zip_parser = subparsers.add_parser("zip", help="Create submission zip")
    zip_parser.add_argument("--csv", type=str, required=True, help="CSV file to zip")
    zip_parser.add_argument("--out_zip", type=str, required=True, help="Output zip path")

    args = parser.parse_args()

    if args.command in ["eval", "predict"]:
        config = load_config(args.config)

        if args.command == "eval":
            evaluate_loop(config, args)
        elif args.command == "predict":
            predict_loop(config, args)

    elif args.command == "zip":
        zip_submission(args.csv, args.out_zip)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
