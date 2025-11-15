#!/bin/bash

# Submission Generation Script
# Model: GroundingDINO (BERT + ResNet18)
# Checkpoint: grounding_dino_gpu_max_best.pth

set -e  # Exit on error

WORK_DIR="/home/work/CUAI_DATA/황민아/Team-20"
cd $WORK_DIR

CKPT="checkpoints/grounding_dino_gpu_max_best.pth"
CONFIG="configs/predict_submission.yaml"
JSON_DIR="/home/work/CUAI_DATA/황민아/Team-20/open/test/query"
JPG_DIR="/home/work/CUAI_DATA/황민아/Team-20/open/test/images"
OUT_CSV="outputs/submission.csv"
OUT_ZIP="outputs/submission.zip"

echo "================================"
echo "Submission File Generation"
echo "================================"
echo "Checkpoint: $CKPT"
echo "Test JSON: $JSON_DIR"
echo "Test JPG: $JPG_DIR"
echo "Output CSV: $OUT_CSV"
echo "Output ZIP: $OUT_ZIP"
echo "================================"

# Check checkpoint exists
if [ ! -f "$CKPT" ]; then
    echo "Error: Checkpoint file not found: $CKPT"
    exit 1
fi

# Check test data exists
if [ ! -d "$JSON_DIR" ]; then
    echo "Error: Test JSON directory not found: $JSON_DIR"
    exit 1
fi

if [ ! -d "$JPG_DIR" ]; then
    echo "Error: Test image directory not found: $JPG_DIR"
    exit 1
fi

# Create output directory
mkdir -p outputs

# Run prediction
echo ""
echo "[1/2] Running prediction..."
python test.py predict \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --json_dir "$JSON_DIR" \
    --jpg_dir "$JPG_DIR" \
    --out_csv "$OUT_CSV"

# Check if CSV was created
if [ ! -f "$OUT_CSV" ]; then
    echo "Error: Prediction CSV not created"
    exit 1
fi

echo ""
echo "[2/2] Creating submission zip..."
python test.py zip \
    --csv "$OUT_CSV" \
    --out_zip "$OUT_ZIP"

# Check if ZIP was created
if [ ! -f "$OUT_ZIP" ]; then
    echo "Error: Submission ZIP not created"
    exit 1
fi

echo ""
echo "================================"
echo "Submission file created successfully!"
echo "================================"
echo "CSV: $OUT_CSV"
echo "ZIP: $OUT_ZIP"
echo ""
echo "You can now submit: $OUT_ZIP"
echo "================================"
