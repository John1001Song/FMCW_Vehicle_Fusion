#!/bin/bash
# run_train.sh
python RaDE-net8.py \
    --input_file ./data/combined_ego_rear/inputs.npy \
    --label_file ./data/combined_ego_rear/labels.npy \
    --train_indices_file ./data/combined_ego_rear/train_indices.npy \
    --val_indices_file ./data/combined_ego_rear/val_indices.npy \
    --test_indices_file ./data/combined_ego_rear/test_indices.npy \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --epochs 600 \
    --loss_weights 0.5 0.10 0.2 0.20 \
    --log_file ./results/middle_fusion_combined_ego_rear8_log.txt \
    --num_workers 4 \
    --save_model_dir ./results/combined_ego_rear8/
