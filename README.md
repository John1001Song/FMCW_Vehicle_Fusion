# Project Overview

## Definitions
- **Raw Data**: Radar signals captured by our device.
### middle fusion 
- we have two views- rear and side.
- we create two ego based views ego_rear and ego_side which will have added data points from the other view.
- the only combination is this addition or contatentation of views.
- We train the model on rear vs ego_rear or side vs ego_side alone, and test on rear vs ego_rear and side vs ego_side respectively.

### late fusion
- load pre-trained rear_model and side_model
- fine-tune the decision layer network
- do not use combined view features, we combine the predicted bounding box from rear_model and side_model
- tested on ego:side vs ego:rear

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
- **`RaDE-net8.py`**: Middle fusion model in version 8 with new sub-network attached with attention module on feature Intensity. 
- **`run_middle_fusion_rear8.sh`**: Shell script for running the middle fusion model verison 8 on rear view only.
- **`run_middle_fusion_side8.sh`**: Shell script for running the middle fusion model verison 8 on side view only.
- **`run_middle_fusion_combined_ego_rear8.sh`**: Shell script for running the middle fusion model verison 8 on rear view enhanced by side view.
- **`run_middle_fusion_combined_ego_side8.sh`**: Shell script for running the middle fusion model verison 8 on side view enhanced by rear view.
- **`run_experiments_late.sh`**: (To be created) Shell script for running late fusion experiments.
- **`npy_info.py`**: Display all npy shpaes in data folder 

---

## Experiments

### Current Experiments
- **Ongoing development and model testing.**

### Planned Experiments
- **Jinyue AR**:
  - Add latest dataset generation code for `.npy files`. 
  - Command to run `late_fusion_data_generator_rear_side_matched.py`.

### Completed Experiments
- File to generate `.npy` files for middle fusion.
- Clean commands for late fusion and add them to `run_experiments_late.sh`.
- Slice the **rear** and **side** views for late fusion to have **train, val, test** datasets.
- Ensure parameters match those of **middle fusion**.

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
