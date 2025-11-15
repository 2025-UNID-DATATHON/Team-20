"""
preprocess.py
Data preprocessing utilities: Vocab, Dataset, Collate function
"""
import os
import json
from glob import glob
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# ===== Helper Functions =====
def find_jsons(json_dir: str) -> List[str]:
    if os.path.isdir(json_dir):
        return sorted(glob(os.path.join(json_dir, "*.json")))
    raise FileNotFoundError(f"json_dir not found: {json_dir}")


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_image_path(json_path: str, data: Dict[str, Any], jpg_dir: str = None) -> str:
    """Resolve image path from JSON metadata."""
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
            return maybe

    base = os.path.splitext(os.path.basename(json_path))[0]
    sibling = os.path.join(os.path.dirname(json_path), base.replace("MI3", "MI2") + ".jpg")
    if os.path.exists(sibling):
        return sibling

    raise FileNotFoundError(f"Could not resolve JPG for {json_path} (jpg_dir={jpg_dir})")


def simple_tokenize(s: str) -> List[str]:
    """Simple tokenizer for Korean/English mixed text."""
    s = (s or "")
    s = s.replace("##", " ").replace(",", " ").replace("(", " ").replace(")", " ")
    s = s.replace(":", " ").replace("?", " ").replace("!", " ").replace("·", " ")
    return [t for t in s.strip().split() if t]


def is_visual_ann(a: dict) -> bool:
    """Filter visual elements (table/chart) with non-empty query."""
    cid = str(a.get("class_id", "") or "")
    cname = str(a.get("class_name", "") or "")
    has_q = bool(str(a.get("visual_instruction", "") or "").strip())
    looks_visual = cid.startswith("V") or any(k in cname for k in ["표", "차트", "그래프", "chart", "table"])
    return has_q and looks_visual


# ===== Vocabulary =====
class Vocab:
    """Simple vocabulary builder for text encoding."""
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.freq: Dict[str, int] = {}
        self.itos: List[str] = ["<pad>", "<unk>"]
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.itos)}

    def build(self, texts: List[str]):
        """Build vocab from list of text strings."""
        for s in texts:
            for tok in simple_tokenize(s):
                self.freq[tok] = self.freq.get(tok, 0) + 1

        for tok, f in sorted(self.freq.items(), key=lambda x: (-x[1], x[0])):
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, s: str, max_len: int = 40) -> List[int]:
        """Encode text to token ids."""
        toks = simple_tokenize(s)[:max_len]
        if not toks:
            return [1]  # <unk>
        return [self.stoi.get(t, 1) for t in toks]

    def __len__(self):
        return len(self.itos)


# ===== Dataset =====
class UniDSet(Dataset):
    """Document Layout Dataset for visual element detection."""
    def __init__(
        self,
        json_files: List[str],
        jpg_dir: str = None,
        vocab: Vocab = None,
        build_vocab: bool = False,
        resize_to: Tuple[int, int] = (512, 512),
        transform=None
    ):
        self.items = []
        for jf in json_files:
            data = read_json(jf)
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
                    "json": jf,
                    "img": img_path,
                    "query_id": qid,
                    "query": qtxt,
                    "bbox": bbox,
                    "class_name": cname,
                })

        self.vocab = vocab if vocab is not None else Vocab(min_freq=1)
        if build_vocab:
            self.vocab.build([it["query"] for it in self.items])

        self.resize_to = resize_to
        self.transform = transform

        try:
            from torchvision import transforms as T
            if self.transform is None:
                self.transform = T.Compose([
                    T.Resize(resize_to),
                    T.ToTensor()
                ])
            self._use_torchvision = True
        except ImportError:
            self._use_torchvision = False

    def __len__(self):
        return len(self.items)

    @staticmethod
    def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
        """Manual PIL to tensor conversion (fallback)."""
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        it = self.items[idx]
        img = Image.open(it["img"]).convert("RGB")
        W, H = img.size

        if self._use_torchvision and self.transform is not None:
            img_t = self.transform(img)
        else:
            img = img.resize(self.resize_to, Image.BILINEAR)
            img_t = self._pil_to_tensor(img)

        ids = self.vocab.encode(it["query"], max_len=40)
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


# ===== Collate Function =====
def collate_fn(batch: List[Dict[str, Any]]):
    """Collate function for variable-length text queries."""
    max_len = max(max(1, int(b["length"])) for b in batch)
    B = len(batch)

    ids = torch.zeros(B, max_len, dtype=torch.long)
    lens = torch.zeros(B, dtype=torch.long)
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    targets = []
    meta = []

    for i, b in enumerate(batch):
        l = max(1, int(b["length"]))
        ids[i, :l] = b["query_ids"][:l]
        lens[i] = l
        targets.append(b["target"])
        meta.append({
            "query_id": b["query_id"],
            "query_text": b["query_text"],
            "orig_size": b["orig_size"],
            "class_name": b["class_name"],
        })

    return imgs, ids, lens, targets, meta


# ===== DataLoader Builder =====
def make_loader(
    json_dir: str,
    jpg_dir: str,
    vocab: Vocab = None,
    build_vocab: bool = False,
    batch_size: int = 8,
    img_size: int = 512,
    num_workers: int = 2,
    shuffle: bool = False,
    transform=None
):
    """Build Dataset and DataLoader."""
    from torch.utils.data import DataLoader, Subset

    json_files = find_jsons(json_dir)
    ds = UniDSet(
        json_files,
        jpg_dir=jpg_dir,
        vocab=vocab,
        build_vocab=build_vocab,
        resize_to=(img_size, img_size),
        transform=transform
    )

    if build_vocab:
        sup_idx = [i for i in range(len(ds)) if ds[i]["target"] is not None]
        if len(sup_idx) == 0:
            raise RuntimeError("No supervised samples (no bboxes) in given json_dir.")
        ds = Subset(ds, sup_idx)

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return ds, dl
