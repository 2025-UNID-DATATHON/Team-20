# model.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from transformers import CLIPVisionModel, CLIPTextModel
except Exception:
    print("Failed to import 'transformers'. Please run: pip install transformers")
    
# preprocess.py에서 CFG를 임포트합니다.
from preprocess import CFG

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = CFG.MODEL_NAME, out_dim: int = CFG.DIM):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.proj = nn.Linear(self.model.config.hidden_size, out_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        token_features = outputs.last_hidden_state
        projected_tokens = self.proj(token_features)
        return projected_tokens


class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = CFG.MODEL_NAME, out_dim: int = CFG.DIM):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained(model_name)
        vision_dim = self.model.config.hidden_size
        self.proj = nn.Conv2d(vision_dim, out_dim, kernel_size=1)
        self.patch_size = self.model.config.patch_size
        self.grid_size = CFG.IMG_SIZE // self.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        patch_features = outputs.last_hidden_state
        patch_features = patch_features[:, 1:, :]
        B, L, D = patch_features.shape
        H = W = self.grid_size
        if L != H * W:
            raise ValueError(f"Feature map size mismatch. Expected {H*W} patches, got {L}")
        fmap = patch_features.reshape(B, H, W, D)
        fmap = fmap.permute(0, 3, 1, 2)
        fmap = self.proj(fmap)
        return fmap

class BiAttentionFusion(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        self.sa_img_layers = nn.ModuleList()
        self.sa_txt_layers = nn.ModuleList()
        
        self.ca_img_on_txt_layers = nn.ModuleList()
        self.ca_txt_on_img_layers = nn.ModuleList()
        
        self.ffn_img_layers = nn.ModuleList()
        self.ffn_txt_layers = nn.ModuleList()

        # Norms
        self.norm_sa_img = nn.ModuleList()
        self.norm_sa_txt = nn.ModuleList()
        self.norm_ca_img = nn.ModuleList()
        self.norm_ca_txt = nn.ModuleList()
        self.norm_ffn_img = nn.ModuleList()
        self.norm_ffn_txt = nn.ModuleList()

        # Dropouts
        self.drop_sa_img = nn.ModuleList()
        self.drop_sa_txt = nn.ModuleList()
        self.drop_ca_img = nn.ModuleList()
        self.drop_ca_txt = nn.ModuleList()
        self.drop_ffn_img = nn.ModuleList()
        self.drop_ffn_txt = nn.ModuleList()

        for _ in range(num_layers):
            self.sa_img_layers.append(nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True))
            self.sa_txt_layers.append(nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True))
            
            self.ca_img_on_txt_layers.append(nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True))
            self.ca_txt_on_img_layers.append(nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True))
            
            ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(dim * 4, dim)
            )
            self.ffn_img_layers.append(copy.deepcopy(ffn))
            self.ffn_txt_layers.append(copy.deepcopy(ffn))
            
            self.norm_sa_img.append(nn.LayerNorm(dim))
            self.norm_sa_txt.append(nn.LayerNorm(dim))
            self.norm_ca_img.append(nn.LayerNorm(dim))
            self.norm_ca_txt.append(nn.LayerNorm(dim))
            self.norm_ffn_img.append(nn.LayerNorm(dim))
            self.norm_ffn_txt.append(nn.LayerNorm(dim))
            
            self.drop_sa_img.append(nn.Dropout(0.1))
            self.drop_sa_txt.append(nn.Dropout(0.1))
            self.drop_ca_img.append(nn.Dropout(0.1))
            self.drop_ca_txt.append(nn.Dropout(0.1))
            self.drop_ffn_img.append(nn.Dropout(0.1))
            self.drop_ffn_txt.append(nn.Dropout(0.1))

    def forward(self, img_feat: torch.Tensor, txt_feat: torch.Tensor,
                  txt_pad_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        img_feat_fused = img_feat
        txt_feat_fused = txt_feat
        
        for i in range(self.num_layers):
            
            # --- 1. Self-Attention (SA) ---
            img_sa = self.sa_img_layers[i](
                query=img_feat_fused, key=img_feat_fused, value=img_feat_fused,
                need_weights=False
            )[0]
            img_feat_fused = self.norm_sa_img[i](img_feat_fused + self.drop_sa_img[i](img_sa))
            
            txt_sa = self.sa_txt_layers[i](
                query=txt_feat_fused, key=txt_feat_fused, value=txt_feat_fused,
                key_padding_mask=txt_pad_mask,
                need_weights=False
            )[0]
            txt_feat_fused = self.norm_sa_txt[i](txt_feat_fused + self.drop_sa_txt[i](txt_sa))

            # --- 2. Bi-Attention (Cross-Attention) ---
            img_ca = self.ca_img_on_txt_layers[i](
                query=img_feat_fused,
                key=txt_feat_fused,
                value=txt_feat_fused,
                key_padding_mask=txt_pad_mask,
                need_weights=False
            )[0]
            temp_img = self.norm_ca_img[i](img_feat_fused + self.drop_ca_img[i](img_ca))
            
            txt_ca = self.ca_txt_on_img_layers[i](
                query=txt_feat_fused,
                key=img_feat_fused,
                value=img_feat_fused,
                need_weights=False
            )[0]
            temp_txt = self.norm_ca_txt[i](txt_feat_fused + self.drop_ca_txt[i](txt_ca))
            
            img_feat_fused = temp_img
            txt_feat_fused = temp_txt

            # --- 3. Feed-Forward Network (FFN) ---
            img_ffn = self.ffn_img_layers[i](img_feat_fused)
            img_feat_fused = self.norm_ffn_img[i](img_feat_fused + self.drop_ffn_img[i](img_ffn))
            
            txt_ffn = self.ffn_txt_layers[i](txt_feat_fused)
            txt_feat_fused = self.norm_ffn_txt[i](txt_feat_fused + self.drop_ffn_txt[i](txt_ffn))
        
        return img_feat_fused, txt_feat_fused

class CrossAttnVLM(nn.Module):
    def __init__(self, model_name: str = CFG.MODEL_NAME, dim: int = CFG.DIM,
                 num_heads: int = 8,
                 num_fusion_layers: int = CFG.NUM_FUSION_LAYERS,
                 num_decoder_layers: int = CFG.NUM_DECODER_LAYERS):
        super().__init__()
        self.txt = TextEncoder(model_name=model_name, out_dim=dim)
        self.img = ImageEncoder(model_name=model_name, out_dim=dim)
        self.dim = dim

        # 1. Learnable Object Query
        self.num_queries = 1
        self.obj_queries = nn.Embedding(self.num_queries, dim)

        # 2. Bi-Attention Fusion Module
        self.fusion_module = BiAttentionFusion(
            dim=dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers
        )

        # 3. Granular Decoder (SA -> CA(Vis) -> CA(Txt) -> FFN)
        self.num_decoder_layers = num_decoder_layers
        
        self.decoder_self_attns = nn.ModuleList()
        self.decoder_cross_attns_vis = nn.ModuleList()
        self.decoder_cross_attns_txt = nn.ModuleList()
        self.decoder_ffns = nn.ModuleList()
        
        self.decoder_norm1s = nn.ModuleList() # After SA
        self.decoder_norm2s = nn.ModuleList() # After CA-Vis
        self.decoder_norm3s = nn.ModuleList() # After CA-Txt
        self.decoder_norm4s = nn.ModuleList() # After FFN
        
        self.decoder_dropout1s = nn.ModuleList()
        self.decoder_dropout2s = nn.ModuleList()
        self.decoder_dropout3s = nn.ModuleList()
        self.decoder_dropout4s = nn.ModuleList()

        for _ in range(num_decoder_layers):
            self.decoder_self_attns.append(
                nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
            )
            self.decoder_cross_attns_vis.append(
                nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
            )
            self.decoder_cross_attns_txt.append(
                nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
            )
            self.decoder_ffns.append(
                nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 4, dim)
                )
            )
            
            self.decoder_norm1s.append(nn.LayerNorm(dim))
            self.decoder_norm2s.append(nn.LayerNorm(dim))
            self.decoder_norm3s.append(nn.LayerNorm(dim))
            self.decoder_norm4s.append(nn.LayerNorm(dim))
            
            self.decoder_dropout1s.append(nn.Dropout(0.1))
            self.decoder_dropout2s.append(nn.Dropout(0.1))
            self.decoder_dropout3s.append(nn.Dropout(0.1))
            self.decoder_dropout4s.append(nn.Dropout(0.1))
        
        # 4. BBox 예측 헤드
        self.bbox_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, 4)
        )

    def forward(self, images: torch.Tensor, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B = images.size(0)

        # 1. 텍스트 특징 추출
        text_features = self.txt(tokens, attention_mask)
        text_pad_mask = (attention_mask == 0) # (True: pad)

        # 2. 이미지 특징 추출
        fmap = self.img(images)
        img_features = fmap.flatten(2).permute(0, 2, 1)

        # 3. Bi-Attention Fusion
        fused_img_feat, fused_text_feat = self.fusion_module(
            img_features, text_features, text_pad_mask
        )

        # 4. Learnable Object Query 준비
        obj_q = self.obj_queries.weight.unsqueeze(0).repeat(B, 1, 1)

        # 5. N-Layer Decoder (SA -> CA(Vis) -> CA(Txt) -> FFN)
        for i in range(self.num_decoder_layers):
            
            # 1. Self-Attention (on Object Query)
            q_sa = self.decoder_self_attns[i](
                query=obj_q, key=obj_q, value=obj_q,
                need_weights=False
            )[0]
            obj_q = self.decoder_norm1s[i](obj_q + self.decoder_dropout1s[i](q_sa))

            # 2. Cross-Attention (Vision)
            q_ca_vis = self.decoder_cross_attns_vis[i](
                query=obj_q, key=fused_img_feat, value=fused_img_feat,
                need_weights=False
            )[0]
            obj_q = self.decoder_norm2s[i](obj_q + self.decoder_dropout2s[i](q_ca_vis))

            # 3. Cross-Attention (Text)
            q_ca_txt = self.decoder_cross_attns_txt[i](
                query=obj_q, key=fused_text_feat, value=fused_text_feat,
                key_padding_mask=text_pad_mask,
                need_weights=False
            )[0]
            obj_q = self.decoder_norm3s[i](obj_q + self.decoder_dropout3s[i](q_ca_txt))
            
            # 4. FFN (Feed-Forward Network)
            q_ffn = self.decoder_ffns[i](obj_q)
            obj_q = self.decoder_norm4s[i](obj_q + self.decoder_dropout4s[i](q_ffn))

        # 6. BBox 예측
        final_q_squeezed = obj_q.squeeze(1)
        pred = self.bbox_head(final_q_squeezed)
        pred_norm = torch.sigmoid(pred)
        
        return pred_norm