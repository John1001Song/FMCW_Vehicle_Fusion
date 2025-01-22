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
-   

#Jayneel AR
- look and visualize the radar data from the npy files
