#!/bin/bash

# Define the list of resolutions, group names, and prediction types
resolutions=(128 256)
group_names=('nurbs' 'harmonics' 'skelneton')
predict_types=('velocity_x' 'velocity_y' 'pressure')

# Directory where the result files are stored
RESULTS_DIR="results"

# Loop through all combinations of resolution, group_name, and predict
for resolution in "${resolutions[@]}"; do
  for group_name in "${group_names[@]}"; do
    for predict in "${predict_types[@]}"; do
      # Construct the result file name based on resolution, group_name, and predict type
      result_file="${RESULTS_DIR}/training_results_${resolution}x${resolution}_${group_name}_${predict}.csv"

      # Check if the result file already exists
      if [ -f "$result_file" ]; then
        echo "Skipping already completed test for resolution $resolution, group $group_name, and prediction type $predict"
      else
        echo "Running training for resolution $resolution, group $group_name, and prediction type $predict"
        
        # Run the training script with the current resolution, group_name, and predict type
        CUDA_VISIBLE_DEVICES=3 python train.py --resolution $resolution --group_name $group_name --predict $predict --expand_factor 1
        
      fi

      # Optional: Add a wait or sleep here if needed
      # sleep 1
    done
  done
done
