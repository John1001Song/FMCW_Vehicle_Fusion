# Project Overview

## Definitions
- **Raw Data**: Radar signals captured by our device.
### middle fusion 
- we have two views- rear and side.
- we create two ego based views ego_rear and ego_side which will have added data points from the other view.
- the only combination is this addition or contatentation of views.
- We train the model on rear vs ego_rear or side vs ego_side alone, and test on rear vs ego_rear and side vs ego_side respectively.

### late fusion
- train model1 on rear only
- train model2 on side only
- do not use combined view features, we combine the predicted bounding box from model1 and model2
- tested on side vs rear

---
## Folder Structure
```
/data
├── combined_ego_rear_npy/  
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── test_features.npy
│   ├── test_labels.npy
│
├── combined_ego_side_npy/  
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── test_features.npy
│   ├── test_labels.npy
│
├── rear_npy/  
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── test_features.npy
│   ├── test_labels.npy
│
├── side_npy/  
│   ├── train_features.npy
│   ├── train_labels.npy
│   ├── val_features.npy
│   ├── val_labels.npy
│   ├── test_features.npy
│   ├── test_labels.npy
│
├── raw_data/  # Contains two subdirectories
│   ├── Nov_16_Processed_with_Combined
│   ├── Nov_17_Processed_with_Combined

/results/  # Directory to log results and model runs
/archive/  # Archived files for future reference
```

### Notes:
- The **rear view folder** in both fusions has similar content, but the way **Jinyue** sliced them differs.
- The following need to be identified:
  - Middle Fusion: **[Need to find]**
  - Late Fusion: **[Need to find]**

---

## Scripts
- **`late_fusion_data_generator_rear_side_matched.py`**: Generates `.npy` files, taking `.csv` files as input.
- **`late_fusion.py`**: Runs the late fusion experiment.
- **`middle_fusion_model.py`**: Model used in middle fusion based on JV's best model parameters and code. 
- **`run_middle_fusion_rear.sh`**: Shell script for running middle fusion experiments on rear view only.
- **`run_experiments_late.sh`**: (To be created) Shell script for running late fusion experiments.
- **`npy_info.py`**: Display all npy shpaes in data folder 

---

## Experiments

### Current Experiments
- **Ongoing development and model testing.**

### Planned Experiments
- **Jinyue AR**:
  - Slice the **rear** and **side** views for late fusion to have **train, val, test** datasets.
  - Ensure parameters match those of **middle fusion**.
  - Command to run `late_fusion_data_generator_rear_side_matched.py`.
  - File to generate `.npy` files for middle fusion.
  - Clean commands for late fusion and add them to `run_experiments_late.sh`.

### Completed Experiments
- **`model.py` on all three views (rear, side, combined)**
  - **Rear View**: Best performing.
  - **Side View**: Worst performing.
  - **Combined View**: Second best.
- **`model_with_fusion.py`**
  - Train `model.py` on **rear** dataset.
  - Train `model.py` on **side** dataset.
  - Let both models predict on the test dataset.
  - Combine their predictions in the **fusion layer** (post-evaluation stage).
  - **Note**: Combining does not mean pooling but creating a more holistic partial **3D view** from two partial 3D views.

---

## Commands
### Late Fusion
```
# To run late_fusion.py
[Jinyue]
```

---

## Action Items

### Jayneel AR
- Look and visualize the radar data from the `.npy` files.
- Confirm with **Hansol** that the data is logically correct.
- Check if **Hansol** has any additional code used for processing.
- Obtain **Jinyue's** notes for ground truth.

### Questions for Hansol
- How was the **combined view dataset** created?
