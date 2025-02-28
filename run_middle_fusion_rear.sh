#!/bin/bash
# run_train.sh
python middle_fusion_model.py \
    --input_file ./data/rear/inputs.npy \
    --label_file ./data/rear/labels.npy \
    --train_indices_file ./data/rear/train_indices.npy \
    --val_indices_file ./data/rear/val_indices.npy \
    --test_indices_file ./data/rear/test_indices.npy \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --epochs 300 \
    --loss_weights 0.60 0.10 0.30 \
    --log_file ./results/middle_fusion_rear_log.txt \
    --num_workers 4 \
    --save_model_dir ./results/
