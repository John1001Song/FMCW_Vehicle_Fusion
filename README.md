# definitions
raw data: radar signals captured by our device.
middle fusion: some features are visible.
raw data is already processed, in human perspective we can already see what it is.

# Folder Structure
 /data : contains two subdirectories, one for each experiment(middle and late fusion)
    late fusion: has two- rear and side view
    middle fusion: has three- rear view, side view and combined(which is a super position of rear and side views)
    the rear view folder in
    - content is similiar.  how Jinyue sliced them is different. 
    middle fusion: [Need to find]
    late fusion: [Need to find]
late_fusion_data_generator_rear_side_matched.py: generates npy file, takes csv files
late_fusion.py: runs the late fusione experiment. command below
middle_fusion.py: Jinyue will add the middle fusion model he is using for late fusion.
middle_fusion_JV_best.py: JV's slight improvement on JS's implementation, but not used for late fusion.
run_experiments.sh: shell script for running middle fusion

results/ : directory to log results, model runs.
archive/ : archived files for future reference, just in case.
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

# Commands
# late_fusion.py
[Jinyue]

# Jayneel AR
- look and visualize the radar data from the npy files

# Jinyue AR
- slice the rear, side view for late fusion to have train, val, test(keep the same paramers as middle fusion)
- command to run late_fusion_data_generator_rear_side_matched.py
- file to generate npy files for middle fusion.
- clean commands for late fusion, add it as a .sh file and name it: run_experiments_late.sh
# to ask Hansol
- how the combined view dataset was made?
