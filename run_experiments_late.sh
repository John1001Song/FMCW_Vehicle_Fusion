#!/usr/bin/env bash

# Run fusion training with relative paths for rear + side datasets.

python RaDE-net6_w_late_fusion.py \
    --rear_train_inputs ./data/late_fusion/rear/inputs.npy \
    --rear_train_labels ./data/late_fusion/rear/labels.npy \
    --rear_train_indices ./data/late_fusion/rear/train_indices.npy \
    --rear_val_inputs ./data/late_fusion/rear/inputs.npy \
    --rear_val_labels ./data/late_fusion/rear/labels.npy \
    --rear_val_indices ./data/late_fusion/rear/val_indices.npy \
    \
    --side_train_inputs ./data/late_fusion/side/inputs.npy \
    --side_train_labels ./data/late_fusion/side/labels.npy \
    --side_train_indices ./data/late_fusion/side/train_indices.npy \
    --side_val_inputs ./data/late_fusion/side/inputs.npy \
    --side_val_labels ./data/late_fusion/side/labels.npy \
    --side_val_indices ./data/late_fusion/side/val_indices.npy \
    \
    --log_file ./results/fusion_log.txt \
    --num_workers 4 \
    --batch_size 8 \
    --epochs 50
