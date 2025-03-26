#!/bin/bash

# Define the list of resolutions, group names, and prediction types
#resolutions=128
#expand_factor=3
group_names=('nurbs' 'harmonics' 'skelneton')
latent_shape=('square' 'ring')

# Directory where the result files are stored
RESULTS_DIR="results"

# Loop through all combinations of resolution, group_name, and predict
for latent_shape in "${latent_shape[@]}"; do
    for group_name in "${group_names[@]}"; do
    
        # Construct the result file name based on resolution, group_name, and predict type
        #result_file="${RESULTS_DIR}/otno_3fileds_512_${group_name}_${latent_shape}.csv"

        # Check if the result file already exists
        #if [ -f "$result_file" ]; then
        #echo "Skipping already completed test for resolution 512, group $group_name, latent shape $latent_shape"
        #else
        echo "Running training for resolution 512, group $group_name, and latent shape $latent_shape"
        
        # Run the training script with the current resolution, group_name, and predict type
        CUDA_VISIBLE_DEVICES=1 python ot_train_fullspace.py --resolution 512 --group_name $group_name --expand_factor 3 --latent_shape $latent_shape
        
        #fi

    done
done
