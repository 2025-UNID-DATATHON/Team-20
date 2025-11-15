# test.py
import os
import io
import json
import zipfile
import argparse
from glob import glob
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch

# 다른 모듈에서 임포트
from preprocess import CFG, make_loader, seed_everything, get_processor
from model import CrossAttnVLM

# --- 평가/예측 유틸리티 ---

def iou_xywh_pixel(pred_xywh, gt_xywh):
    px, py, pw, ph = pred_xywh
    gx, gy, gw, gh = gt_xywh
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = pw * ph + gw * gh - inter if (pw * ph + gw * gh - inter) > 0 else 1e-6
    return inter / union

def _load_model_from_ckpt(ckpt_path: str, device: torch.device):
    global PROCESSOR, CFG
    
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model_name = ckpt.get("model_name", CFG.MODEL_NAME)
    img_size = ckpt.get("img_size", CFG.IMG_SIZE)
    dim = ckpt["dim"]
    
    num_fusion = ckpt.get("num_fusion_layers", CFG.NUM_FUSION_LAYERS)
    num_decoder = ckpt.get("num_decoder_layers", CFG.NUM_DECODER_LAYERS)

    # CFG와 PROCESSOR를 체크포인트 기준으로 업데이트
    CFG.IMG_SIZE = img_size
    PROCESSOR = get_processor(model_name)

    model = CrossAttnVLM(model_name=model_name,
                         dim=dim,
                         num_fusion_layers=num_fusion,
                         num_decoder_layers=num_decoder).to(device)
    
    # 새 아키텍처로 저장된 체크포인트는 strict=True로 로드
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    
    return model, PROCESSOR, img_size

# --- 루프 정의 ---

def evaluate_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, processor, img_size = _load_model_from_ckpt(args.ckpt, device)
    
    ds, dl = make_loader(args.json_dir, args.jpg_dir,
                         batch_size=args.batch_size, img_size=img_size,
                         num_workers=args.num_workers, shuffle=False,
                         is_train=False) # is_train=False (GT가 없어도 평가)

    rows = []; ious = []
    with torch.no_grad():
        for imgs, ids, mask, targets, meta in tqdm(dl, desc="Evaluating"):
            imgs = imgs.to(device); ids = ids.to(device); mask = mask.to(device)
            pred = model(imgs, ids, mask)
            
            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W; y = (cy - nh / 2.0) * H
                w = nw * W; h = nh * H
                rows.append({
                    "query_id": meta[i]["query_id"], "query_text": meta[i]["query_text"],
                    "pred_x": x, "pred_y": y, "pred_w": w, "pred_h": h
                })
                # targets (GT)가 있는 경우에만 mIoU 계산
                if targets[i] is not None:
                    gt = [float(v) for v in targets[i].numpy().tolist()]
                    gx = (gt[0] - gt[2] / 2.0) * W; gy = (gt[1] - gt[3] / 2.0) * H
                    gw = gt[2] * W; gh = gt[3] * H
                    ious.append(iou_xywh_pixel([x, y, w, h], [gx, gy, gw, gh]))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.out_csv}")
    if ious:
        print(f"[Eval] mIoU={float(np.mean(ious)):.4f}")
    else:
        print("[Eval] No GT found; mIoU not computed.")

def predict_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor, img_size = _load_model_from_ckpt(args.ckpt, device)
    
    ds, dl = make_loader(args.json_dir, args.jpg_dir,
                         batch_size=args.batch_size, img_size=img_size,
                         num_workers=args.num_workers, shuffle=False,
                         is_train=False)

    rows = []
    with torch.no_grad():
        for imgs, ids, mask, targets, meta in tqdm(dl, desc="Predicting"):
            imgs = imgs.to(device); ids = ids.to(device); mask = mask.to(device)
            pred = model(imgs, ids, mask)
            for i in range(imgs.size(0)):
                W, H = meta[i]["orig_size"]
                cx, cy, nw, nh = [float(v) for v in pred[i].cpu().numpy().tolist()]
                x = (cx - nw / 2.0) * W; y = (cy - nh / 2.0) * H
                w = nw * W; h = nh * H
                rows.append({
                    "query_id": meta[i]["query_id"], "query_text": meta[i]["query_text"],
                    "pred_x": x, "pred_y": y, "pred_w": w, "pred_h": h
                })

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df = pd.DataFrame(rows, columns=["query_id", "query_text", "pred_x", "pred_y", "pred_w", "pred_h"])
    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {args.out_csv}")

def zip_submission(csv_path: str, zip_path: str):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        arcname = os.path.basename(csv_path)
        zf.write(csv_path, arcname=arcname)
    print(f"[Submission] Zipped {csv_path} → {zip_path}")

# --- 실행 로직 ---

def get_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--json_dir", type=str, default=CFG.JSON_DIR)
        p.add_argument("--jpg_dir", type=str, default=CFG.JPG_DIR)
        p.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
        # img_size, dim 등은 ckpt에서 로드하므로 test 시점에는 필요 없음
        p.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)

    # eval
    p_eval = sub.add_parser("eval")
    add_common(p_eval)
    p_eval.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    p_eval.add_argument("--out_csv", type=str, default=CFG.EVAL_CSV)

    # predict
    p_pred = sub.add_parser("predict")
    add_common(p_pred)
    p_pred.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint.")
    p_pred.add_argument("--out_csv", type=str, default=CFG.PRED_CSV)

    # zip
    p_zip = sub.add_parser("zip")
    p_zip.add_argument("--csv", type=str, required=True, help="CSV file to zip.")
    p_zip.add_argument("--out_zip", type=str, default=CFG.SUBMISSION_ZIP)

    return ap.parse_args()

def main():
    seed_everything(CFG.SEED)
    args = get_args()

    # CFG 전역 변수 업데이트
    CFG.BATCH_SIZE = args.batch_size
    CFG.JSON_DIR = args.json_dir
    CFG.JPG_DIR = args.jpg_dir

    if args.cmd == "eval":
        evaluate_loop(args)
    elif args.cmd == "predict":
        predict_loop(args)
    elif args.cmd == "zip":
        zip_submission(args.csv, args.out_zip)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")

if __name__ == "__main__":
    main()