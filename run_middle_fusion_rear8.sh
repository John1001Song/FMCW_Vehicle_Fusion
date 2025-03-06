#!/bin/bash
# run_train.sh

python RaDE-net8.py \
    --input_file ./data/rear/inputs.npy \
    --label_file ./data/rear/labels.npy \
    --train_indices_file ./data/rear/train_indices.npy \
    --val_indices_file ./data/rear/val_indices.npy \
    --test_indices_file ./data/rear/test_indices.npy \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --epochs 600 \
    --loss_weights 0.5 0.1 0.2 0.2 \
    --log_file ./results/middle_fusion_rear8_log.txt \
    --num_workers 4 \
    --save_model_dir ./results/rear8/
