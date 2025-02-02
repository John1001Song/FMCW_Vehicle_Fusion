#definitions
raw data: radar signals captured by our device.
middle fusion: some features are visible.
raw data is already processed, in human perspective we can already see what it is.


# Current Experiments

# Planned Experiments

# Completed Experiments
- model.py on all 3(rear, side and combined)
  - rear: performed best
  - side: worst performing
  - combined: second
  - combined: looking at both rear and side data.
- model_with_fusion.py
-   you train model.py on rear dataset
-   you train model.py on side dataset
-   let both model' predict on the test dataset
-   and then combine their predictions in the fusion layer. (post  evaluation stage)
-   combining does not meet pooling, but creating a more holistic partial 3D view from two partial 3D views.

## Late Fusion cmd 
example:

python late_fusion.py \
    --rear_train_inputs /home/jinyues/pointillism/pointillism/rear/inputs.npy \
    --rear_train_labels /home/jinyues/pointillism/pointillism/rear/labels.npy \
    --rear_train_indices /home/jinyues/pointillism/pointillism/rear/train_indices.npy \
    --rear_val_inputs /home/jinyues/pointillism/pointillism/rear/inputs.npy \
    --rear_val_labels /home/jinyues/pointillism/pointillism/rear/labels.npy \
    --rear_val_indices /home/jinyues/pointillism/pointillism/rear/val_indices.npy \
    \
    --side_train_inputs /home/jinyues/pointillism/pointillism/side/inputs.npy \
    --side_train_labels /home/jinyues/pointillism/pointillism/side/labels.npy \
    --side_train_indices /home/jinyues/pointillism/pointillism/side/train_indices.npy \
    --side_val_inputs /home/jinyues/pointillism/pointillism/side/inputs.npy \
    --side_val_labels /home/jinyues/pointillism/pointillism/side/labels.npy \
    --side_val_indices /home/jinyues/pointillism/pointillism/side/val_indices.npy \
    \
    --log_file ./results/fusion_log.txt \
    --num_workers 4 \
    --batch_size 8 \
    --epochs 500


# Jayneel AR
- look and visualize the radar data from the npy files
