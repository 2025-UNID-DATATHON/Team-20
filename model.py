"""
model.py
Modular model components: Encoders, Adapters, Heads, and Model Factory
"""
import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== Loss Functions =====
def giou_loss(pred_boxes, target_boxes, reduction='mean'):
    """
    Compute GIoU (Generalized Intersection over Union) loss.

    Args:
        pred_boxes: (B, 4) normalized boxes [cx, cy, w, h]
        target_boxes: (B, 4) normalized boxes [cx, cy, w, h]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        GIoU loss (1 - GIoU)
    """
    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-7)

    # Smallest enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose_area = enclose_w * enclose_h

    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)

    # GIoU loss
    loss = 1 - giou

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def compute_iou(pred_boxes, target_boxes):
    """
    Compute IoU between predicted and target boxes.

    Args:
        pred_boxes: (B, 4) normalized boxes [cx, cy, w, h]
        target_boxes: (B, 4) normalized boxes [cx, cy, w, h]

    Returns:
        IoU values (B,)
    """
    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-7)

    return iou


def compute_giou(pred_boxes, target_boxes):
    """
    Compute GIoU between predicted and target boxes.

    Args:
        pred_boxes: (B, 4) normalized boxes [cx, cy, w, h]
        target_boxes: (B, 4) normalized boxes [cx, cy, w, h]

    Returns:
        GIoU values (B,)
    """
    # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection area
    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # Union area
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union_area = pred_area + target_area - inter_area

    # IoU
    iou = inter_area / (union_area + 1e-7)

    # Smallest enclosing box
    enclose_x1 = torch.min(pred_x1, target_x1)
    enclose_y1 = torch.min(pred_y1, target_y1)
    enclose_x2 = torch.max(pred_x2, target_x2)
    enclose_y2 = torch.max(pred_y2, target_y2)

    enclose_w = (enclose_x2 - enclose_x1).clamp(min=0)
    enclose_h = (enclose_y2 - enclose_y1).clamp(min=0)
    enclose_area = enclose_w * enclose_h

    # GIoU
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-7)

    return giou


# ===== Text Encoders =====
class TextEncoderBase(nn.Module):
    """Base class for text encoders."""
    def __init__(self, vocab_size: int, emb_dim: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GRUTextEncoder(TextEncoderBase):
    """GRU-based text encoder."""
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden: int = 256, **kwargs):
        super().__init__(vocab_size, emb_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden * 2, emb_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)  # (B, L, E)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, h = self.gru(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, 2*hidden)
        q = self.proj(h_cat)  # (B, D)
        return q


class LSTMTextEncoder(TextEncoderBase):
    """LSTM-based text encoder (alternative)."""
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden: int = 256, **kwargs):
        super().__init__(vocab_size, emb_dim)
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Linear(hidden * 2, emb_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, (h, c) = self.lstm(packed)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)
        q = self.proj(h_cat)
        return q


class BERTTextEncoder(TextEncoderBase):
    """BERT-based text encoder (multilingual)."""
    def __init__(self, vocab_size: int, emb_dim: int = 256, freeze_bert: bool = True,
                 model_name: str = 'bert-base-multilingual-cased', **kwargs):
        super().__init__(vocab_size, emb_dim)
        self.freeze_bert = freeze_bert
        self.model_name = model_name

        try:
            from transformers import BertModel
            # Use BERT pretrained model (multilingual supports Korean)
            self.bert = BertModel.from_pretrained(model_name)
            self.bert_dim = 768

            if freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
                print(f"[BERT] Frozen pretrained weights ({model_name})")
            else:
                print(f"[BERT] Fine-tuning enabled ({model_name})")

            self.proj = nn.Linear(self.bert_dim, emb_dim)

        except Exception as e:
            print(f"Warning: BERT loading failed ({e}). Using simple embedding fallback.")
            self.bert = None
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.bert is not None:
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = (tokens != 0).long()

            # BERT forward
            with torch.set_grad_enabled(not self.freeze_bert):
                outputs = self.bert(input_ids=tokens, attention_mask=attention_mask)

            # Use [CLS] token representation
            cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            q = self.proj(cls_output)  # (B, D)

        else:
            # Fallback to simple embedding
            x = self.emb(tokens)  # (B, L, D)
            # Mean pooling
            mask = (tokens != 0).unsqueeze(-1).float()
            x_masked = x * mask
            q = x_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            q = self.proj(q)

        return q


# ===== Vision Encoders =====
class VisionEncoderBase(nn.Module):
    """Base class for vision encoders."""
    def __init__(self, out_dim: int, **kwargs):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TinyCNN(VisionEncoderBase):
    """Lightweight CNN backbone (fallback)."""
    def __init__(self, out_dim: int = 256, **kwargs):
        super().__init__(out_dim)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (B, D, H', W')


class ResNetVisionEncoder(VisionEncoderBase):
    """ResNet-based vision encoder with ImageNet pretraining."""
    def __init__(self, out_dim: int = 256, pretrained: bool = True, backbone: str = "resnet18", **kwargs):
        super().__init__(out_dim)
        try:
            if backbone == "resnet18":
                from torchvision.models import resnet18, ResNet18_Weights
                weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                m = resnet18(weights=weights)
                self.in_channels = 512
            elif backbone == "resnet50":
                from torchvision.models import resnet50, ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                m = resnet50(weights=weights)
                self.in_channels = 2048
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

            layers = list(m.children())[:-2]  # Remove FC and avgpool
            self.backbone = nn.Sequential(*layers)
            self.proj = nn.Conv2d(self.in_channels, out_dim, 1)
        except Exception as e:
            print(f"Warning: ResNet backbone failed ({e}). Using TinyCNN fallback.")
            self.backbone = TinyCNN(out_dim)
            self.proj = nn.Identity()

    def forward(self, x):
        f = self.backbone(x)
        f = self.proj(f)
        return f


class SwinVisionEncoder(VisionEncoderBase):
    """Swin Transformer-based vision encoder."""
    def __init__(self, out_dim: int = 256, pretrained: bool = True, backbone: str = "swin_tiny", **kwargs):
        super().__init__(out_dim)
        try:
            if backbone == "swin_tiny":
                from torchvision.models import swin_t, Swin_T_Weights
                weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
                m = swin_t(weights=weights)
                self.in_channels = 768
            elif backbone == "swin_small":
                from torchvision.models import swin_s, Swin_S_Weights
                weights = Swin_S_Weights.IMAGENET1K_V1 if pretrained else None
                m = swin_s(weights=weights)
                self.in_channels = 768
            else:
                raise ValueError(f"Unsupported Swin backbone: {backbone}")

            self.backbone = nn.Sequential(*list(m.children())[:-2])
            self.proj = nn.Conv2d(self.in_channels, out_dim, 1)
        except Exception as e:
            print(f"Warning: Swin backbone failed ({e}). Using TinyCNN fallback.")
            self.backbone = TinyCNN(out_dim)
            self.proj = nn.Identity()

    def forward(self, x):
        f = self.backbone(x)
        if f.dim() == 3:  # (B, L, C) -> (B, C, H, W)
            B, L, C = f.shape
            H = W = int(L ** 0.5)
            f = f.transpose(1, 2).reshape(B, C, H, W)
        f = self.proj(f)
        return f


# ===== Adapters =====
class LayoutAdapter(nn.Module):
    """Adapter module for document layout adaptation."""
    def __init__(self, dim: int = 256, reduction: int = 4, **kwargs):
        super().__init__()
        hidden = max(dim // reduction, 64)
        self.down = nn.Linear(dim, hidden)
        self.up = nn.Linear(hidden, dim)
        self.act = nn.ReLU(inplace=True)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, D, H, W) or (B, D)
        """
        orig_shape = x.shape
        if x.dim() == 4:  # Conv feature map
            B, D, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B * H * W, D)  # (B*H*W, D)

        residual = x
        x = self.down(x)
        x = self.act(x)
        x = self.up(x)
        x = residual + self.scale * x

        if len(orig_shape) == 4:
            B, D, H, W = orig_shape
            x = x.reshape(B, H, W, D).permute(0, 3, 1, 2)

        return x


# ===== Fusion Modules =====
class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for vision-text features."""
    def __init__(self, dim: int = 256, num_heads: int = 8, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        """
        q_vec: (B, D) - text query
        fmap: (B, D, H, W) - vision feature map
        Returns: (B, D) - fused feature
        """
        B, D, H, W = fmap.shape

        Q = self.q_proj(q_vec).reshape(B, self.num_heads, self.head_dim)  # (B, nh, hd)
        K = self.k_proj(fmap).reshape(B, self.num_heads, self.head_dim, H * W)  # (B, nh, hd, HW)
        V = self.v_proj(fmap).reshape(B, self.num_heads, self.head_dim, H * W)  # (B, nh, hd, HW)

        attn = torch.einsum("bhd,bdhw->bhw", Q, K) / math.sqrt(self.head_dim)  # (B, nh, HW)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bhw,bdhw->bhd", attn, V)  # (B, nh, hd)
        out = out.reshape(B, D)
        out = self.out_proj(out)

        return out


class SimpleFusion(nn.Module):
    """Simple late fusion (element-wise product + projection)."""
    def __init__(self, dim: int = 256, **kwargs):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, q_vec: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        B, D, H, W = fmap.shape
        q = self.q_proj(q_vec)  # (B, D)
        K = self.k_proj(fmap)  # (B, D, H, W)
        V = self.v_proj(fmap)

        Kf = K.flatten(2).transpose(1, 2)  # (B, HW, D)
        Vf = V.flatten(2).transpose(1, 2)
        q = q.unsqueeze(1)  # (B, 1, D)

        attn = torch.matmul(q, Kf.transpose(1, 2)) / math.sqrt(D)
        attn = torch.softmax(attn, dim=-1)
        ctx = torch.matmul(attn, Vf).squeeze(1)

        return ctx


# ===== GroundingDINO Components =====
class DeformableAttention(nn.Module):
    """Simplified Deformable Attention for GroundingDINO."""
    def __init__(self, dim: int = 256, num_heads: int = 8, num_points: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(dim, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(dim, num_heads * num_points)
        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, Nq, D) query features
            reference_points: (B, Nq, 2) normalized reference points
            value: (B, D, H, W) vision feature map - MUST be already projected to dim D
        """
        B, Nq, D = query.shape
        _, C, H, W = value.shape

        # Ensure value has correct dimension
        if C != D:
            raise ValueError(f"DeformableAttention expects value dim {D}, got {C}. "
                           f"Make sure vision features are projected before passing to decoder.")

        offsets = self.sampling_offsets(query).view(B, Nq, self.num_heads, self.num_points, 2)
        attn_weights = self.attention_weights(query).view(B, Nq, self.num_heads, self.num_points)
        attn_weights = F.softmax(attn_weights, dim=-1)

        ref = reference_points.unsqueeze(2).unsqueeze(3)
        sampling_locations = (ref + offsets * 0.1).clamp(0, 1)
        sampling_grid = sampling_locations * 2.0 - 1.0

        value_flat = value.flatten(2).transpose(1, 2)  # (B, H*W, D)
        value_proj = self.value_proj(value_flat)
        value_map = value_proj.transpose(1, 2).reshape(B, D, H, W)

        sampled_features = []
        for h in range(self.num_heads):
            grid = sampling_grid[:, :, h, :, :].reshape(B, Nq * self.num_points, 1, 2)
            sampled = F.grid_sample(value_map, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            sampled = sampled.squeeze(-1).transpose(1, 2).view(B, Nq, self.num_points, D)
            sampled_features.append(sampled)

        # sampled_features: list of (B, Nq, num_points, D), len=num_heads
        sampled_features = torch.stack(sampled_features, dim=2)  # (B, Nq, num_heads, num_points, D)
        # attn_weights: (B, Nq, num_heads, num_points)
        # Weighted sum over points
        output = (sampled_features * attn_weights.unsqueeze(-1)).sum(dim=3)  # (B, Nq, num_heads, D)
        # Average over heads instead of concat
        output = output.mean(dim=2)  # (B, Nq, D)
        return self.output_proj(output)


class FeatureEnhancer(nn.Module):
    """Vision-Text bi-directional cross-attention."""
    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.v2t_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.v2t_norm = nn.LayerNorm(dim)
        self.t2v_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.t2v_norm = nn.LayerNorm(dim)

        self.ffn_v = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.ffn_t = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.norm_ffn_v = nn.LayerNorm(dim)
        self.norm_ffn_t = nn.LayerNorm(dim)

    def forward(self, vision_feat: torch.Tensor, text_feat: torch.Tensor, text_mask=None):
        v_enh, _ = self.t2v_attn(vision_feat, text_feat, text_feat, key_padding_mask=text_mask)
        vision_feat = self.t2v_norm(vision_feat + v_enh)
        vision_feat = self.norm_ffn_v(vision_feat + self.ffn_v(vision_feat))

        t_enh, _ = self.v2t_attn(text_feat, vision_feat, vision_feat)
        text_feat = self.v2t_norm(text_feat + t_enh)
        text_feat = self.norm_ffn_t(text_feat + self.ffn_t(text_feat))
        return vision_feat, text_feat


class LanguageGuidedQuerySelector(nn.Module):
    """Initialize queries from text features."""
    def __init__(self, dim: int = 256, num_queries: int = 100):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, dim)
        self.text_to_query = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ref_point_head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, 2), nn.Sigmoid())

    def forward(self, text_feat: torch.Tensor, text_mask=None):
        B = text_feat.size(0)
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        enh_q, _ = self.text_to_query(queries, text_feat, text_feat, key_padding_mask=text_mask)
        queries = self.norm(queries + enh_q)
        reference_points = self.ref_point_head(queries)
        return queries, reference_points


class CrossModalityDecoderLayer(nn.Module):
    """GroundingDINO decoder layer."""
    def __init__(self, dim: int = 256, num_heads: int = 8, num_points: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.cross_attn_vision = DeformableAttention(dim, num_heads, num_points)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn_text = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(True), nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.norm4 = nn.LayerNorm(dim)

    def forward(self, queries, reference_points, vision_feat, text_feat, text_mask=None):
        q2, _ = self.self_attn(queries, queries, queries)
        queries = self.norm1(queries + q2)
        q2 = self.cross_attn_vision(queries, reference_points, vision_feat)
        queries = self.norm2(queries + q2)
        q2, _ = self.cross_attn_text(queries, text_feat, text_feat, key_padding_mask=text_mask)
        queries = self.norm3(queries + q2)
        queries = self.norm4(queries + self.ffn(queries))
        return queries, reference_points


class GroundingDINOHead(nn.Module):
    """GroundingDINO detection head with bbox + similarity."""
    def __init__(self, dim: int = 256, num_queries: int = 100):
        super().__init__()
        self.dim = dim
        self.bbox_embed = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, 4), nn.Sigmoid())
        self.query_proj = nn.Linear(dim, dim)
        self.text_proj = nn.Linear(dim, dim)

    def forward(self, queries: torch.Tensor, text_feat: torch.Tensor):
        bbox = self.bbox_embed(queries)
        query_embed = self.query_proj(queries)
        if text_feat.dim() == 3:
            text_embed = self.text_proj(text_feat)
            logits = torch.bmm(query_embed, text_embed.transpose(1, 2)) / math.sqrt(self.dim)
        else:
            text_embed = self.text_proj(text_feat)
            logits = torch.bmm(query_embed, text_embed.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.dim)
        return bbox, logits


class GroundingDINOLocalization(nn.Module):
    """Full GroundingDINO localization module."""
    def __init__(self, dim: int = 256, num_queries: int = 100, num_decoder_layers: int = 6,
                 num_heads: int = 8, num_points: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_queries = num_queries
        self.feature_enhancer = FeatureEnhancer(dim, num_heads, dropout)
        self.query_selector = LanguageGuidedQuerySelector(dim, num_queries)
        self.decoder_layers = nn.ModuleList([
            CrossModalityDecoderLayer(dim, num_heads, num_points, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.head = GroundingDINOHead(dim, num_queries)

    def forward(self, vision_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        B, D, H, W = vision_feat.shape

        text_seq = text_feat.unsqueeze(1)  # (B, 1, D)
        vision_seq = vision_feat.flatten(2).transpose(1, 2)  # (B, H*W, D)

        # Skip FeatureEnhancer to avoid dimension issues - use vision_feat directly
        vision_enhanced = vision_seq
        text_enhanced = text_seq

        # Reshape back to feature map
        vision_feat_enhanced = vision_enhanced.transpose(1, 2).reshape(B, D, H, W)

        queries, reference_points = self.query_selector(text_enhanced)
        for layer in self.decoder_layers:
            queries, reference_points = layer(queries, reference_points, vision_feat_enhanced, text_enhanced)

        bbox_preds, logits = self.head(queries, text_feat)
        best_idx = logits.argmax(dim=1)
        best_bbox = torch.gather(bbox_preds, 1, best_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
        return best_bbox


# ===== Detection Heads =====
class BBoxHead(nn.Module):
    """Simple bbox regression head."""
    def __init__(self, dim: int = 256, hidden: int = 256, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.net(x)
        pred = torch.sigmoid(pred)  # Normalize to [0, 1]
        return pred


# ===== Complete Models =====
class GroundingDocDetector(nn.Module):
    """Modular Grounding-based Document Detector."""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        text_encoder: str = "gru",
        vision_encoder: str = "resnet18",
        fusion: str = "simple",
        use_adapter: bool = True,
        adapter_config: Optional[Dict[str, Any]] = None,
        pretrained_backbone: bool = True,
        **kwargs
    ):
        super().__init__()

        # Text Encoder
        if text_encoder == "gru":
            self.txt = GRUTextEncoder(vocab_size, dim, dim)
        elif text_encoder == "lstm":
            self.txt = LSTMTextEncoder(vocab_size, dim, dim)
        elif text_encoder == "bert":
            bert_model_name = kwargs.get('bert_model', 'bert-base-multilingual-cased')
            self.txt = BERTTextEncoder(vocab_size, dim, freeze_bert=kwargs.get("freeze_bert", True), model_name=bert_model_name)
        elif text_encoder == "kobert":
            bert_model_name = kwargs.get('bert_model', 'skt/kobert-base-v1')
            self.txt = BERTTextEncoder(vocab_size, dim, freeze_bert=kwargs.get("freeze_bert", True), model_name=bert_model_name)
        else:
            raise ValueError(f"Unknown text_encoder: {text_encoder}")

        # Vision Encoder
        if vision_encoder == "resnet18":
            self.img = ResNetVisionEncoder(dim, pretrained_backbone, "resnet18")
        elif vision_encoder == "resnet50":
            self.img = ResNetVisionEncoder(dim, pretrained_backbone, "resnet50")
        elif vision_encoder == "swin_tiny":
            self.img = SwinVisionEncoder(dim, pretrained_backbone, "swin_tiny")
        elif vision_encoder == "tiny_cnn":
            self.img = TinyCNN(dim)
        else:
            raise ValueError(f"Unknown vision_encoder: {vision_encoder}")

        # Adapter (optional)
        self.use_adapter = use_adapter
        if use_adapter:
            adapter_config = adapter_config or {}
            self.adapter = LayoutAdapter(dim, **adapter_config)
        else:
            self.adapter = nn.Identity()

        # Fusion Module
        if fusion == "simple":
            self.fusion = SimpleFusion(dim)
        elif fusion == "cross_attn":
            self.fusion = CrossAttentionFusion(dim, num_heads=8)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # Detection Head
        self.head = BBoxHead(dim)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        q = self.txt(tokens, lengths)  # (B, D)
        fmap = self.img(images)  # (B, D, H', W')

        if self.use_adapter:
            fmap = self.adapter(fmap)

        ctx = self.fusion(q, fmap)  # (B, D)
        pred = self.head(ctx)  # (B, 4)

        return pred


class GroundingDINODocDetector(nn.Module):
    """GroundingDINO-style Document Detector with query-conditioned localization."""
    def __init__(
        self,
        vocab_size: int,
        dim: int = 256,
        text_encoder: str = "gru",
        vision_encoder: str = "resnet18",
        use_adapter: bool = True,
        adapter_config: Optional[Dict[str, Any]] = None,
        pretrained_backbone: bool = True,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_points: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        # Text Encoder
        if text_encoder == "gru":
            self.txt = GRUTextEncoder(vocab_size, dim, dim)
        elif text_encoder == "lstm":
            self.txt = LSTMTextEncoder(vocab_size, dim, dim)
        elif text_encoder == "bert":
            bert_model_name = kwargs.get('bert_model', 'bert-base-multilingual-cased')
            self.txt = BERTTextEncoder(vocab_size, dim, freeze_bert=kwargs.get("freeze_bert", True), model_name=bert_model_name)
        elif text_encoder == "kobert":
            bert_model_name = kwargs.get('bert_model', 'skt/kobert-base-v1')
            self.txt = BERTTextEncoder(vocab_size, dim, freeze_bert=kwargs.get("freeze_bert", True), model_name=bert_model_name)
        else:
            raise ValueError(f"Unknown text_encoder: {text_encoder}")

        # Vision Encoder
        if vision_encoder == "resnet18":
            self.img = ResNetVisionEncoder(dim, pretrained_backbone, "resnet18")
        elif vision_encoder == "resnet50":
            self.img = ResNetVisionEncoder(dim, pretrained_backbone, "resnet50")
        elif vision_encoder == "swin_tiny":
            self.img = SwinVisionEncoder(dim, pretrained_backbone, "swin_tiny")
        elif vision_encoder == "tiny_cnn":
            self.img = TinyCNN(dim)
        else:
            raise ValueError(f"Unknown vision_encoder: {vision_encoder}")

        # Adapter (optional)
        self.use_adapter = use_adapter
        if use_adapter:
            adapter_config = adapter_config or {}
            self.adapter = LayoutAdapter(dim, **adapter_config)
        else:
            self.adapter = nn.Identity()

        # GroundingDINO Localization Module
        self.grounding_head = GroundingDINOLocalization(
            dim=dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_points=num_points,
            dropout=dropout
        )

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Text encoding: (B, D)
        q = self.txt(tokens, lengths)

        # Vision encoding: (B, D, H, W)
        fmap = self.img(images)

        # Adapter (optional)
        if self.use_adapter:
            fmap = self.adapter(fmap)

        # GroundingDINO localization
        pred = self.grounding_head(fmap, q)  # (B, 4)

        return pred


# ===== Model Factory =====
def build_model(config: Dict[str, Any], vocab_size: int) -> nn.Module:
    """Factory function to build model from config."""
    model_type = config.get("model_type", "grounding_doc_detector")

    if model_type == "grounding_doc_detector":
        return GroundingDocDetector(
            vocab_size=vocab_size,
            dim=config.get("dim", 256),
            text_encoder=config.get("text_encoder", "gru"),
            vision_encoder=config.get("vision_encoder", "resnet18"),
            fusion=config.get("fusion", "simple"),
            use_adapter=config.get("use_adapter", True),
            adapter_config=config.get("adapter_config", {}),
            pretrained_backbone=config.get("pretrained_backbone", True),
            bert_model=config.get("bert_model", "bert-base-multilingual-cased"),
            freeze_bert=config.get("freeze_bert", True),
        )
    elif model_type == "grounding_dino":
        return GroundingDINODocDetector(
            vocab_size=vocab_size,
            dim=config.get("dim", 256),
            text_encoder=config.get("text_encoder", "gru"),
            vision_encoder=config.get("vision_encoder", "resnet18"),
            use_adapter=config.get("use_adapter", True),
            adapter_config=config.get("adapter_config", {}),
            pretrained_backbone=config.get("pretrained_backbone", True),
            num_queries=config.get("num_queries", 100),
            num_decoder_layers=config.get("num_decoder_layers", 6),
            num_heads=config.get("num_heads", 8),
            num_points=config.get("num_points", 4),
            dropout=config.get("dropout", 0.1),
            bert_model=config.get("bert_model", "bert-base-multilingual-cased"),
            freeze_bert=config.get("freeze_bert", True),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ===== Freezing Utilities =====
def freeze_encoder(model: nn.Module, encoder_type: str = "vision"):
    """Freeze encoder parameters."""
    if encoder_type == "vision":
        for param in model.img.parameters():
            param.requires_grad = False
        print("[Freeze] Vision encoder frozen.")
    elif encoder_type == "text":
        for param in model.txt.parameters():
            param.requires_grad = False
        print("[Freeze] Text encoder frozen.")
    elif encoder_type == "both":
        freeze_encoder(model, "vision")
        freeze_encoder(model, "text")


def get_trainable_params(model: nn.Module):
    """Get trainable parameter groups."""
    adapter_params = []
    fusion_params = []
    head_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "adapter" in name:
            adapter_params.append(param)
        elif "fusion" in name:
            fusion_params.append(param)
        elif "head" in name:
            head_params.append(param)
        else:
            other_params.append(param)

    return {
        "adapter": adapter_params,
        "fusion": fusion_params,
        "head": head_params,
        "other": other_params,
    }
