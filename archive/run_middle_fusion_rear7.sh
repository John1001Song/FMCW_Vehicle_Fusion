#!/bin/bash
# run_train.sh
python RaDE-net7_gpu.py \
    --input_file ./data/rear/inputs.npy \
    --label_file ./data/rear/labels.npy \
    --train_indices_file ./data/rear/train_indices.npy \
    --val_indices_file ./data/rear/val_indices.npy \
    --test_indices_file ./data/rear/test_indices.npy \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --epochs 300 \
    --loss_weights 0.55 0.10 0.25 0.10 \
    --log_file ./results/middle_fusion_rear7_log.txt \
    --num_workers 4 \
    --save_model_dir ./results/rear7/
