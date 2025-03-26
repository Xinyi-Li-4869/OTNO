'''import random
import torch
import numpy as np
resolution = 512
latent_shape = 'square'
expand_factor = 3

data = {'inputs': [], 'outs': [], 'indices_decoder': []}

# Loop over the names and load the corresponding data
for name in ['nurbs', 'harmonics', 'skelneton']:
    data_path = f'/home/xinyili/OTNO/flowbench/otno/ot-data/LDC_NS_2D_{resolution}_{name}_{latent_shape}_expand{expand_factor}_reg1e-6_combined.pt'
    group_data = torch.load(data_path)
    
    # Flatten the list of tensors in group_data['inputs'] and append them to data['inputs']
    for input_tensor in group_data['inputs']:
        data['inputs'].append(input_tensor)  # Flatten the list

    # Flatten the list of tensors in group_data['outs'] and append them to data['outs']
    for output_tensor in group_data['outs']:
        data['outs'].append(output_tensor)  # Flatten the list

    # Flatten the list of tensors in group_data['indices_decoder'] and append them to data['indices_decoder']
    for index_tensor in group_data['indices_decoder']:
        data['indices_decoder'].append(index_tensor)  # Flatten the list

# Shuffle the data consistently (zipping, shuffling, then unzipping)
combined_data = list(zip(data['inputs'], data['outs'], data['indices_decoder']))
random.shuffle(combined_data)

# Unzip the shuffled data back into separate tensors
data['inputs'], data['outs'], data['indices_decoder'] = zip(*combined_data)

print(len(data['inputs']), data['inputs'][0].shape)

torch.save(data, f'/home/xinyili/OTNO/flowbench/otno/ot-data/LDC_NS_2D_{resolution}_allgroups_{latent_shape}_expand{expand_factor}_reg1e-6_combined.pt')'''

import random
import torch
import numpy as np
resolution = 512
latent_shape = 'square'
expand_factor = 3

inputs_original = []

# Loop over the names and load the corresponding data
for name in ['nurbs', 'harmonics', 'skelneton']:
    data_path = f'/data/xinyili/datasets/flowbench/ot-data/LDC_NS_2D_{resolution}_{name}_{latent_shape}_expand{expand_factor}_reg1e-6_combined.pt'
    group_data = torch.load(data_path)
    inputs_original.extend(group_data['inputs'])

data_shuffled = torch.load(f'/data/xinyili/datasets/flowbench/ot-data/LDC_NS_2D_fullspace_{resolution}_allgroups_{latent_shape}_expand{expand_factor}_reg1e-6_combined.pt')
inputs_shuffled = data_shuffled['inputs']

indices_mapping = []
for x in inputs_shuffled:
    best_match = None
    best_index = None
    min_diff = float("inf")
    
    for i, orig in enumerate(inputs_original):
        if orig.shape == x.shape:  # Only compare tensors of the same shape
            diff = torch.norm(x - orig).item()  # Compute similarity
            if diff < min_diff:
                min_diff = diff
                best_match = orig
                best_index = i
    
    if best_index is not None:
        indices_mapping.append(best_index)
    else:
        print(f"Warning: No match found for a shuffled input of shape {x.shape}!")


with open("combine_indices.txt", "w") as f:
    for item in indices_mapping:
        f.write(f"{item}\n")  # Write each element on a new line