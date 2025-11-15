# train.py
import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

# 다른 모듈에서 임포트
from preprocess import CFG, make_loader, seed_everything
from model import CrossAttnVLM

# --- 손실 함수 (train.py에서만 사용) ---

def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def generalized_box_iou_loss(pred_norm_cxcywh: torch.Tensor,
                             target_norm_cxcywh: torch.Tensor) -> torch.Tensor:
    pred_boxes = box_cxcywh_to_xyxy(pred_norm_cxcywh)
    target_boxes = box_cxcywh_to_xyxy(target_norm_cxcywh)
    x1, y1, x2, y2 = pred_boxes.unbind(-1)
    x1g, y1g, x2g, y2g = target_boxes.unbind(-1)
    
    ix1 = torch.max(x1, x1g)
    iy1 = torch.max(y1, y1g)
    ix2 = torch.min(x2, x2g)
    iy2 = torch.min(y2, y2g)
    inter_area = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    
    pred_area = (x2 - x1) * (y2 - y1)
    target_area = (x2g - x1g) * (y2g - y1g)
    union_area = pred_area + target_area - inter_area + 1e-6
    
    iou = inter_area / union_area
    
    cx1 = torch.min(x1, x1g)
    cy1 = torch.min(y1, y1g)
    cx2 = torch.max(x2, x2g)
    cy2 = torch.max(y2, y2g)
    c_area = (cx2 - cx1) * (cy2 - cy1) + 1e-6
    
    giou = iou - (c_area - union_area) / c_area
    loss = 1.0 - giou
    return loss.mean()

# --- 학습 루프 ---

def train_loop(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds, train_dl = make_loader(args.json_dir, args.jpg_dir,
                                     batch_size=args.batch_size, img_size=args.img_size,
                                     num_workers=args.num_workers, shuffle=True,
                                     is_train=True)

    model = CrossAttnVLM(model_name=CFG.MODEL_NAME,
                         dim=args.dim,
                         num_fusion_layers=args.num_fusion_layers,
                         num_decoder_layers=args.num_decoder_layers).to(device)
    
    if args.resume_ckpt:
        if os.path.exists(args.resume_ckpt):
            print(f"Loading weights from: {args.resume_ckpt}")
            ckpt = torch.load(args.resume_ckpt, map_location=device)
            try:
                model.load_state_dict(ckpt["model_state"], strict=False)
                print("Successfully loaded model weights (strict=False).")
                print("Note: Fusion/Decoder weights may be re-initialized if architecture changed.")
            except Exception as e:
                print(f"Error loading state_dict: {e}")
                print("Warning: Could not load weights. Training from scratch.")
        else:
            print(f"Warning: --resume_ckpt path not found, training from scratch: {args.resume_ckpt}")
    else:
        print("Training from scratch (no --resume_ckpt provided).")

    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("txt.model.") or name.startswith("img.model."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    print(f"Backbone params: {len(backbone_params)}, Head params: {len(head_params)}")

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params, 'lr': args.lr}          
    ], lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    total_samples = len(train_ds)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        print(f'Current Epoch : {epoch}')
        for idx, (imgs, ids, mask, targets, meta) in enumerate(tqdm(train_dl, desc=f"Epoch {epoch}")):
            imgs = imgs.to(device); ids = ids.to(device); mask = mask.to(device)
            t = torch.stack([tar for tar in targets if tar is not None], dim=0).to(device)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                pred = model(imgs, ids, mask)
                loss_l1 = F.smooth_l1_loss(pred, t, reduction="mean")
                loss_giou = generalized_box_iou_loss(pred, t)
                loss = loss_l1 * 5.0 + loss_giou * 2.0
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * imgs.size(0)
            
            if idx % 50 == 0:
                avg = running / ((idx + 1) * args.batch_size)
                current_lr_head = optimizer.param_groups[1]['lr']
                current_lr_bb = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch}/{args.epochs} Step {idx}] loss={avg:.4f} lr_head={current_lr_head:.6f} lr_bb={current_lr_bb:.6f}")

                # 50 스텝마다 체크포인트 저장
                ckpt_path = f'./outputs/ckpt/detr_fusion_{epoch}_{idx}.pth'
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                
                torch.save({
                    "model_state": model.state_dict(),
                    "model_name": CFG.MODEL_NAME,
                    "dim": args.dim,
                    "img_size": args.img_size,
                    "num_fusion_layers": args.num_fusion_layers,
                    "num_decoder_layers": args.num_decoder_layers
                }, ckpt_path)

        scheduler.step()
        avg = running / total_samples
        print(f"[Epoch {epoch}/{args.epochs}] loss={avg:.4f}")

    # --- 최종 모델 저장 ---
    os.makedirs(os.path.dirname(args.save_ckpt), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "model_name": CFG.MODEL_NAME,
        "dim": args.dim,
        "img_size": args.img_size,
        "num_fusion_layers": args.num_fusion_layers,
        "num_decoder_layers": args.num_decoder_layers
    }, args.save_ckpt)
    print(f"[Saved] {args.save_ckpt}")

# --- 실행 로직 ---

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", type=str, default=CFG.JSON_DIR)
    ap.add_argument("--jpg_dir", type=str, default=CFG.JPG_DIR)
    ap.add_argument("--batch_size", type=int, default=CFG.BATCH_SIZE)
    ap.add_argument("--img_size", type=int, default=CFG.IMG_SIZE)
    ap.add_argument("--dim", type=int, default=CFG.DIM)
    ap.add_argument("--num_workers", type=int, default=CFG.NUM_WORKERS)
    
    ap.add_argument("--epochs", type=int, default=CFG.EPOCHS)
    ap.add_argument("--lr", type=float, default=CFG.LEARNING_RATE, help="Head learning rate")
    ap.add_argument("--backbone_lr", type=float, default=CFG.BACKBONE_LR, help="Backbone learning rate")
    ap.add_argument("--save_ckpt", type=str, default=CFG.CKPT_PATH)
    ap.add_argument("--resume_ckpt", type=str, default=None, help="Path to checkpoint to resume training from.")
    ap.add_argument("--num_fusion_layers", type=int, default=CFG.NUM_FUSION_LAYERS, help="Number of bi-attention fusion layers.")
    ap.add_argument("--num_decoder_layers", type=int, default=CFG.NUM_DECODER_LAYERS, help="Number of decoder layers.")
    
    return ap.parse_args()

def main():
    seed_everything(CFG.SEED)
    args = get_args()

    # CFG 전역 변수 업데이트
    CFG.IMG_SIZE = args.img_size
    CFG.DIM = args.dim
    CFG.BATCH_SIZE = args.batch_size
    CFG.JSON_DIR = args.json_dir
    CFG.JPG_DIR = args.jpg_dir
    CFG.CKPT_PATH = args.save_ckpt
    CFG.LEARNING_RATE = args.lr
    CFG.BACKBONE_LR = args.backbone_lr
    CFG.NUM_FUSION_LAYERS = args.num_fusion_layers
    CFG.NUM_DECODER_LAYERS = args.num_decoder_layers
    
    train_loop(args)

if __name__ == "__main__":
    main()