#!/bin/bash

# Define the list of resolutions, group names, and prediction types
resolutions=(128)
group_names=('nurbs')
predict_types=('velocity_x')
expand_factor=(3 4)

# Directory where the result files are stored
RESULTS_DIR="results"

# Loop through all combinations of resolution, group_name, and predict
for resolution in "${resolutions[@]}"; do
  for group_name in "${group_names[@]}"; do
    for predict in "${predict_types[@]}"; do
      for expand_factor in "${expand_factor[@]}"; do
        # Construct the result file name based on resolution, group_name, and predict type
        result_file="${RESULTS_DIR}/training_results_${resolution}x${resolution}_${group_name}_${predict}_expand${expand_factor}.csv"

        # Check if the result file already exists
        if [ -f "$result_file" ]; then
          echo "Skipping already completed test for expand factor $expand_factor"
        else
          echo "Running training for expand factor $expand_factor"
          
          # Run the training script with the current resolution, group_name, and predict type
          CUDA_VISIBLE_DEVICES=3 python train.py --resolution $resolution --group_name $group_name --predict $predict --expand_factor $expand_factor
          
        fi

        # Optional: Add a wait or sleep here if needed
        # sleep 1
      done
    done
  done
done
