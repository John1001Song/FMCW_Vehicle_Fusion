#!/bin/bash

# Run middle fusion with only rear data
echo "Running model with rear data..."
python3 middle_fusion_updated.py \
    --input_file ./data/rear/inputs.npy \
    --label_file ./data/rear/labels.npy \
    --train_indices_file ./data/rear/train_indices.npy \
    --val_indices_file ./data/rear/val_indices.npy \
    --save_model_dir ./results/ \
    > logging_rear.txt

# Run middle fusion with only side data
echo "Running model with side data..."
python3 middle_fusion_updated.py \
    --input_file ./data/side/inputs.npy \
    --label_file ./data/side/labels.npy \
    --train_indices_file ./data/side/train_indices.npy \
    --val_indices_file ./data/side/val_indices.npy \
    --save_model_dir ./results/ \
    > logging_side.txt

# Run middle fusion with combined data
echo "Running model with combined data..."
python3 middle_fusion_updated.py \
    --input_file ./data/combined/inputs.npy \
    --label_file ./data/combined/labels.npy \
    --train_indices_file ./data/combined/train_indices.npy \
    --val_indices_file ./data/combined/val_indices.npy \
    --save_model_dir ./results/ \
    > logging_combined.txt
