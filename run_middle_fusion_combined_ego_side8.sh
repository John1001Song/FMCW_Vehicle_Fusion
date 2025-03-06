#!/bin/bash
# run_train.sh
python RaDE-net8.py \
    --input_file ./data/combined_ego_side/inputs.npy \
    --label_file ./data/combined_ego_side/labels.npy \
    --train_indices_file ./data/combined_ego_side/train_indices.npy \
    --val_indices_file ./data/combined_ego_side/val_indices.npy \
    --test_indices_file ./data/combined_ego_side/test_indices.npy \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --epochs 600 \
    --loss_weights 0.5 0.10 0.2 0.20 \
    --log_file ./results/middle_fusion_combined_ego_side8_log.txt \
    --num_workers 4 \
    --save_model_dir ./results/combined_ego_side8/
