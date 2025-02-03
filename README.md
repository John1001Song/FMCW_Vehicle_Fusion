# Project Overview

## Definitions
- **Raw Data**: Radar signals captured by our device.
- **Middle Fusion**: Some features are visible. The raw data is already processed, and from a human perspective, we can already see what it is.

---

## Folder Structure
```
/data
├── middle_fusion/  # Contains three subdirectories
│   ├── rear_view/
│   ├── side_view/
│   ├── combined/  # A superposition of rear and side views
│
├── late_fusion/  # Contains two subdirectories
│   ├── rear_view/
│   ├── side_view/
│
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
- **`middle_fusion.py`**: Jinyue will integrate the middle fusion model used for late fusion.
- **`middle_fusion_JV_best.py`**: JV’s slight improvement on JS’s implementation, but not used for late fusion.
- **`run_experiments.sh`**: Shell script for running middle fusion experiments.
- **`run_experiments_late.sh`**: (To be created) Shell script for running late fusion experiments.

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