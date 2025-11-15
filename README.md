# UNI-D Datathon Team 20
##  Cross-Attention VLM for Visual Grounding

This project is a Vision-Language Model (VLM) that takes image and text queries (Visual Instruction) as input and predicts the **bounding box** of the object specified by the text.

## Model Description

This model aims to "ground" text queries to specific regions within an image.

**Key Features:**

* **Backbone:** Uses the image encoder and text encoder from `openai/clip-vit-base-patch32` to extract features from each modality.
* **Feature Fusion:** Performs deep bidirectional Cross-Attention between image patches and text tokens using the `BiAttentionFusion` module (a custom Transformer block).
* **Decoder:** Sequentially attentively processes the fused image and text features using a single learnable `Object Query`. (Self-Attention -> Cross-Attention(Vision) -> Cross-Attention(Text) -> FFN)
* **Output:** Finally, predicts the bounding box (`cx, cy, w, h`) from the refined single query embedding.

## Model Architecture

## Setup

### 1. Virtual Environment Creation (conda)

```bash
conda create -n vlm_grounding python=3.10
conda activate vlm_grounding
```

### 2. Installing Required Libraries
```bash
pip install -r requirements.txt
```

## Training
Train the model using the train.py script.
```bash
python train.py \
    --json_dir ./data/train/json \
    --jpg_dir ./data/train/jpg \
    --save_ckpt ./outputs/ckpt/my_model_v1.pth \
    --epochs 20 \
    --batch_size 128 \
    --lr 8e-6 \
    --backbone_lr 1e-6 \
    --num_fusion_layers 6 \
    --num_decoder_layers 6
```
## Testing & Prediction
The test.py script supports three subcommands: eval, predict, and zip.

### 1. Eval
Perform predictions on the Valid/Test dataset and calculate mIoU when Ground Truth is available.
```bash
python test.py eval \
    --json_dir ./data/valid/json \
    --jpg_dir ./data/valid/jpg \
    --ckpt ./outputs/ckpt/my_model_v1.pth \
    --out_csv ./outputs/preds/eval_results.csv \
    --batch_size 256
```

### 2. Predict
For datasets without ground truth (e.g., for test server submissions), it performs predictions only and saves the results to CSV.

```bash
python test.py predict \
    --json_dir ./data/test/query \
    --jpg_dir ./data/test/images \
    --ckpt ./outputs/ckpt/my_model_v1.pth \
    --out_csv ./outputs/preds/test_submission.csv \
    --batch_size 256
```
### 3. Zip Submission
Compress the predicted CSV file into a zip file for submission.

```bash
python test.py zip \
    --csv ./outputs/preds/test_submission.csv \
    --out_zip ./outputs/submission.zip
```

# File Structure
preprocess.py: CFG configuration, UniDSet (Dataset), make_loader (DataLoader), and preprocessing utilities.

model.py: CrossAttnVLM model architecture and submodule definitions (e.g., BiAttentionFusion).

train.py: Model training script. (python train.py ...)

test.py: Model evaluation, prediction, and zip generation script. (python test.py [eval|predict|zip] ...)