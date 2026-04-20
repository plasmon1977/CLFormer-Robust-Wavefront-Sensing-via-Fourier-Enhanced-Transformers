#!/bin/bash
eval "$(~/anaconda3/bin/conda shell.bash hook)"

# ========== Hyperparameters ==========
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Update these paths to point to your dataset and where you want to save checkpoints
TRAIN_DIR="/path/to/CLFormer/train"
VAL_DIR="/path/to/CLFormer/val"
TEST_DIR="/path/to/CLFormer/test"
CHECKPOINT_DIR="/path/to/CLFormer/checkpoints"

INPUT_SIZE=112
PATCH_SIZE=8
FSAS_PATCH_SIZE=8
CBAM_REDUCTION=16
CBAM_KERNEL_SIZE=7
CA_REDUCTION=32
NUM_IMAGES=5
NUM_COEFFICIENTS=77
VARIANT="small"
DROPOUT=0.1
BATCH_SIZE=32
EPOCHS=200
LR=1e-4
WEIGHT_DECAY=0.05
MAX_GRAD_NORM=1.0
NUM_WORKERS=8
USE_AMP=""
# USE_AMP="--use_amp"   # uncomment to enable AMP
PATIENCE=200
# =====================================

python CLFormer/train/train.py \
    --train_dir "$TRAIN_DIR" \
    --val_dir "$VAL_DIR" \
    --test_dir "$TEST_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --input_size "$INPUT_SIZE" \
    --patch_size "$PATCH_SIZE" \
    --fsas_patch_size "$FSAS_PATCH_SIZE" \
    --cbam_reduction "$CBAM_REDUCTION" \
    --cbam_kernel_size "$CBAM_KERNEL_SIZE" \
    --ca_reduction "$CA_REDUCTION" \
    --num_images "$NUM_IMAGES" \
    --num_coefficients "$NUM_COEFFICIENTS" \
    --variant "$VARIANT" \
    --dropout "$DROPOUT" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --num_workers "$NUM_WORKERS" \
    $USE_AMP \
    --patience "$PATIENCE"
