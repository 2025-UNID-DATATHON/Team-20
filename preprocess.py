# preprocess.py
import os
import io
import json
import math
import time
import random
from glob import glob
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import AutoProcessor
    _BACKBONE_OK = True
except Exception:
    _BACKBONE_OK = False
    print("Failed to import 'transformers'. Please run: pip install transformers")

class CFG:
    # Core
    IMG_SIZE: int = 224
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.000008      # Head 부분의 학습률
    BACKBONE_LR: float = 0.000001      # Backbone (CLIP) 부분의 학습률
    BATCH_SIZE: int = 256
    SEED: int = 42
    DIM: int = 256
    NUM_WORKERS: int = 32
    
    # 모델 아키텍처 하이퍼파라미터
    NUM_FUSION_LAYERS: int = 6       # Bi-Attention Fusion 레이어 수
    NUM_DECODER_LAYERS: int = 6     # 최종 디코더 레이어 수
    
    MODEL_NAME: str = "openai/clip-vit-base-patch32"

    # Paths
    JSON_DIR: str = "./data/train_valid/train/press_json"
    JPG_DIR: str = "./data/train_valid/train/press_jpg"
    CKPT_PATH: str = "./outputs/ckpt/detr_fusion_vlm.pth"
    EVAL_CSV: str = "./outputs/preds/eval_pred.csv"
    PRED_CSV: str = "./outputs/preds/test_pred.csv"
    SUBMISSION_ZIP: str = "./outputs/submission.zip"

PROCESSOR = None

def get_processor(model_name=CFG.MODEL_NAME):
    global PROCESSOR
    if PROCESSOR is None:
        if not _BACKBONE_OK:
            raise ImportError("transformers 라이브러리를 import할 수 없습니다.")
        PROCESSOR = AutoProcessor.from_pretrained(model_name, use_fast=True)
    return PROCESSOR

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_jsons(json_dir: str) -> List[str]:
    if os.path.isdir(json_dir):
        return sorted(glob(os.path.join(json_dir, "*.json")))
    raise FileNotFoundError(f"json_dir not found: {json_dir}")

def read_json(path: str):
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[WARN] Invalid JSON, skip: {path} ({e})")
        return None

def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str = None) -> str:
    src = data.get("source_data_info", {})
    jpg_name = src.get("source_data_name_jpg", None)
    if jpg_dir and jpg_name:
        path = os.path.join(jpg_dir, jpg_name)
        if os.path.exists(path):
            return path
    if jpg_name:
        maybe = json_path.replace(os.sep + "json" + os.sep, os.sep + "jpg" + os.sep)
        maybe = os.path.join(os.path.dirname(maybe), jpg_name) if os.path.isdir(os.path.dirname(maybe)) else maybe
        if os.path.exists(maybe):
            return path
    base = os.path.splitext(os.path.basename(json_path))[0]
    sibling = os.path.join(os.path.dirname(json_path), base.replace("MI3", "MI2") + ".jpg")
    if os.path.exists(sibling):
        return sibling
    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_dir={jpg_dir})")

def is_visual_ann(a: dict) -> bool:
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(k in cname for k in ["표", "차트", "그래프", "chart", "table"])
    return has_q and looks_visual

class UniDSet(Dataset):
    def __init__(self, json_files: List[str], jpg_dir: str = None,
                 resize_to: Tuple[int, int] = (CFG.IMG_SIZE, CFG.IMG_SIZE)):
        self.processor = get_processor()
        self.resize_to = resize_to
        self.items = []
        for jf in json_files:
            # print(jf) # 너무 많은 로그를 출력하므로 주석 처리
            data = read_json(jf)
            if data is None:
                continue  # 깨진 파일은 그냥 스킵
            ann = data.get("learning_data_info", {}).get("annotation", [])
            img_path = get_image_path(jf, data, jpg_dir=jpg_dir)
            for a in ann:
                if not is_visual_ann(a):
                    continue
                qid = a.get("instance_id", "")
                qtxt = str(a.get("visual_instruction", "")).strip()
                bbox = a.get("bounding_box", None)
                cname = a.get("class_name", "")
                self.items.append({
                    "json": jf, "img": img_path,
                    "query_id": qid, "query": qtxt,
                    "bbox": bbox, "class_name": cname,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img = Image.open(it["img"]).convert("RGB")
        W, H = img.size

        img_processed = self.processor(
            images=img,
            return_tensors="pt"
        )
        img_t = img_processed["pixel_values"].squeeze(0)

        inputs = self.processor.tokenizer(
            it["query"],
            padding='do_not_pad',
            truncation=True,
            max_length=40,
        )
        ids = inputs["input_ids"]
        length = max(1, len(ids))

        sample: Dict[str, Any] = {
            "image": img_t,
            "query_ids": torch.tensor(ids, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
            "query_text": it["query"],
            "query_id": it["query_id"],
            "orig_size": (W, H),
            "class_name": it["class_name"],
        }
        
        if it["bbox"] is not None and isinstance(it["bbox"], (list, tuple)) and len(it["bbox"]) == 4:
            x, y, w, h = it["bbox"]
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            target = torch.tensor([cx, cy, nw, nh], dtype=torch.float32)
        else:
            target = None
        sample["target"] = target
        return sample


def collate_fn(batch: List[Dict[str, Any]]):
    max_len = max(max(1, int(b["length"])) for b in batch)
    B = len(batch)
    pad_token_id = get_processor().tokenizer.pad_token_id
    ids = torch.full((B, max_len), pad_token_id, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.long)
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    targets = []
    meta = []
    for i, b in enumerate(batch):
        l = max(1, int(b["length"]))
        ids[i, :l] = b["query_ids"][:l]
        mask[i, :l] = 1
        targets.append(b["target"])
        meta.append({
            "query_id": b["query_id"],
            "query_text": b["query_text"],
            "orig_size": b["orig_size"],
            "class_name": b["class_name"],
        })
    return imgs, ids, mask, targets, meta

def make_loader(json_dir: str, jpg_dir: str,
                batch_size: int = CFG.BATCH_SIZE, img_size: int = CFG.IMG_SIZE,
                num_workers: int = CFG.NUM_WORKERS, shuffle: bool = False,
                is_train: bool = False):
    
    json_files = find_jsons(json_dir)
    ds = UniDSet(json_files, jpg_dir=jpg_dir,
                 resize_to=(img_size, img_size))
    
    if is_train:
        print("Filtering supervised indices (optimized)...")
        sup_idx = []
        # TQDM을 iteritems() 대신 enumerate(ds.items)에 사용
        for i, item in enumerate(tqdm(ds.items, desc="Filtering items")):
            bbox = item.get("bbox")
            if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                sup_idx.append(i)
        
        if len(sup_idx) == 0:
            raise RuntimeError("No supervised samples (no bboxes) in given json_dir.")
        print(f"Found {len(sup_idx)} supervised samples out of {len(ds.items)} total items.")
        ds = torch.utils.data.Subset(ds, sup_idx)
    
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, collate_fn=collate_fn)
    return ds, dl