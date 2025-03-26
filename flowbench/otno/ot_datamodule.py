import numpy as np
import torch
import open3d as o3d
from ot.bregman import empirical_sinkhorn2_geomloss
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
import os

def square(n):
    x = torch.linspace(0, 2, n)
    y = torch.linspace(0, 2, n)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    return torch.stack([grid_x, grid_y], dim=2)

def ring(n, r_min=0.5, r_max=1.0):
    # Generate theta values evenly spaced around the circle
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Generate radial distances evenly spaced between r_min and r_max
    r = np.linspace(r_min, r_max, n)
    
    # Create meshgrid for the theta and radial distances
    R, Theta = np.meshgrid(r, theta, indexing='ij')
    
    # Convert polar coordinates (R, Theta) to Cartesian coordinates (x, y)
    x = R * np.cos(Theta)
    y = R * np.sin(Theta)
    
    # Stack the x and y coordinates into a single tensor and return
    grid = np.stack((x, y), axis=-1)
    
    return torch.tensor(grid, dtype=torch.float32)

def OT_mean_closest(lazy_tensor, vertices, device, batchsize=2000):
    num_rows, num_cols = lazy_tensor.shape
    all_transports, all_enc_indices = [], []

    # Iterate through columns in batches
    for batch_start in range(0, num_rows, batchsize):
        batch_end = min(batch_start + batchsize, num_rows)
        row_values = lazy_tensor[batch_start:batch_end, :].clone().detach().to(device)  # Evaluate the batch of columns and detach from graph
        row_norms = torch.norm(row_values, p=1, dim=1, keepdim=True)
        batch_OTforward = row_values / row_norms
        
        batch_transport = torch.mm(batch_OTforward, vertices)
        distances = torch.cdist(batch_transport, vertices)
        min_indices = torch.argmin(distances, dim=1)
        all_enc_indices.append(min_indices)
        all_transports.append(batch_transport)

    # Concatenate all transports across batches
    all_enc_indices = torch.cat(all_enc_indices)
    all_transports = torch.cat(all_transports)
    return all_enc_indices, all_transports

def compute_min_indices_batched(transport, centroids, batch_size=2000):
    num_centroids = centroids.size(0)
    
    # Initialize arrays to hold the minimum distances and the corresponding indices
    min_distances = torch.full((num_centroids,), float('inf'), device=transport.device)
    indices_decoder = torch.zeros((num_centroids,), dtype=torch.long, device=transport.device)

    # Iterate through transport points in batches
    for i in range(0, transport.size(0), batch_size):
        end_idx = min(i + batch_size, transport.size(0))
        
        # Calculate distances between a batch of transport points and all centroids
        distances = torch.cdist(transport[i:end_idx], centroids)

        # For each centroid, find the minimal distance and corresponding index in this batch
        batch_min_distances, batch_min_indices = torch.min(distances, dim=0)

        # Update global minimum distances and indices where new smaller distances are found
        mask = batch_min_distances < min_distances
        min_distances[mask] = batch_min_distances[mask]
        indices_decoder[mask] = batch_min_indices[mask] + i  # Add the offset of the current batch

    return indices_decoder

def plot_points(points, filename='plot.png'):
    """
    Plot points from a (2, m) tensor and save the figure.

    Args:
    points (torch.Tensor): A (2, m) tensor where each column represents x and y coordinates.
    filename (str): Filename to save the plot to.
    """
    if points.shape[0] != 2:
        raise ValueError("Expected points tensor of shape (2, m)")

    # Convert points to numpy if not already in that format
    points_np = points.numpy()

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.scatter(points_np[0], points_np[1], c='blue', edgecolor='k', s=20)
    plt.title("Plot of Points")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.axis('equal')  # Set equal scaling by changing axis limits.

    # Save the plot
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory
    print(f'Plot saved as {filename}')

def plot_sdf(points, sdf_values, filename='sdf_plot.png'):
    """
    Plot SDF values for points given as a (2, m) tensor and save the figure.

    Args:
    points (torch.Tensor): A (2, m) tensor where each column represents x and y coordinates.
    sdf_values (torch.Tensor): A (m) tensor representing the SDF values at the corresponding points.
    filename (str): Filename to save the plot to.
    """

    # Convert tensors to numpy arrays
    points_np = points.numpy()
    sdf_values_np = sdf_values.numpy().flatten()  # Flatten for compatibility with scatter
    sdf_values_np = sdf_values_np/np.max(sdf_values_np)

    # Create the plot
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(points_np[0], points_np[1], c=sdf_values_np, cmap='viridis')
    plt.colorbar(scatter, label='SDF Value')
    plt.title("SDF Values on Points")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.axis('equal')  # Set equal scaling by changing axis limits.

    # Save the plot
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory
    print(f'Plot saved as {filename}')

def plot_press(points, press_values, filename='press'):
    """
    Plot SDF values for points given as a (2, m) tensor and save the figure.

    Args:
    points (torch.Tensor): A (2, m) tensor where each column represents x and y coordinates.
    sdf_values (torch.Tensor): A (m) tensor representing the SDF values at the corresponding points.
    filename (str): Filename to save the plot to.
    """

    # Convert tensors to numpy arrays
    points_np = points.numpy()
    values = press_values.numpy().flatten()  # Flatten for compatibility with scatter
    # Calculate normalization factors
    min_val = values.min()
    max_val = values.max()
    range_values = max_val - min_val
    mid_values = (max_val + min_val) / 2.0
    normalized_values = (values - mid_values) / range_values  # Normalize values

    # Create the plot
    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(points_np[0], points_np[1], c=normalized_values, cmap='viridis')
    plt.colorbar(scatter, label=filename)
    plt.title(filename + " on Points")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    plt.axis('equal')  # Set equal scaling by changing axis limits.

    # Save the plot
    plt.savefig("flowbench/otno/images/" + filename + ".png")
    plt.close()  # Close the figure to free memory
    print(f'Plot saved as {filename}')

###### configs ######
expand_factor = 3
resolution = 512
group_name = 'harmonics'
generate_grid = ring
device = torch.device("cuda")
reg=1e-06
save_path = 'ot-data/LDC_NS_2D_boundary_' + str(resolution) + '_' + group_name + '_ring_expand'+str(expand_factor) +'_reg1e-6_combined.pt'
print(save_path)
###### load data ######
data = np.load('/data/xinyili/datasets/flowbench/LDC_NS_2D/' + str(resolution) + 'x' + str(resolution) + '/' + group_name + '_lid_driven_cavity_X.npz')
outputs = np.load('/data/xinyili/datasets/flowbench/LDC_NS_2D/' + str(resolution) + 'x' + str(resolution) + '/' + group_name + '_lid_driven_cavity_Y.npz')
#print(data['data'].shape, outputs['data'].shape)
#sys.exit(0)
sdf_data = data['data'][:,1,:,:]  # sdf
scale = data['data'][:,:,0,0]
output_data = outputs['data'][:,0:3,:,:] # velocity_x, velocity_y, press
del data, outputs

###### initialize ######
#all_points, all_sdfs, all_trans_points, all_trans_sdfs, all_outs, all_indices_encoder, all_indices_decoder = [], [], [], [], [], [], []
all_inputs, all_outs, all_indices_decoder = [], [], []
non_surjective = 0
tt = default_timer()

###### loop over shapes ######
for i in range(1000):
    t0 = default_timer()
    out = output_data[i,:,:,:]
    sdf = sdf_data[i,:,:]
    grid = square(resolution).reshape(-1,2)
    out_flattened = torch.from_numpy(out).reshape(3,-1).T
    sdf_flattened = torch.from_numpy(sdf).flatten()
    #plot_press(grid.T, out_flattened, 'velocity_x_full')

    indices = torch.nonzero((sdf_flattened >= 0) & (sdf_flattened < 0.2)).squeeze()
    out = out_flattened[indices]
    target = grid[indices].to(device)
    target_sdf = sdf_flattened[indices].to(device)
    #plot_points(target)
    #plot_sdf(target, target_sdf, 'flowbench/otno/images/target_sdf.png')
    #plot_press(target.T, out[:,2], 'pressure')
    #sys.exit(0)
    
    n_t = len(target)
    n_s = int(np.sqrt(n_t*expand_factor))
    source = generate_grid(n_s).reshape(-1,2).to(device)
    # OT
    _, log = empirical_sinkhorn2_geomloss(X_s=source.to(dtype=torch.float32), X_t=target.to(dtype=torch.float32), reg=reg, log=True) # utilize weighted Sinkhorn a=a.to(device), b=b.to(device),
    if resolution<256:
        gamma = log['lazy_plan'][:].detach() # convert the lazy tensor to torch.tensor (dense)

        # normalize the OT plan matrix by column
        row_norms = torch.norm(gamma, p=1, dim=1, keepdim=True)
        gamma_encoder = gamma / row_norms

        # transport target to source
        transport = torch.mm(gamma_encoder, target.to(dtype=torch.float32,device=device))

        # encoder: target -> source & decoder: source -> target
        distances = torch.cdist(transport, target.to(dtype=torch.float32,device=device))
        indices_encoder = torch.argmin(distances, dim=1)
        indices_decoder = torch.argmin(distances, dim=0)
        #plot_sdf(source.T, target_sdf[indices_encoder].cpu(), 'flowbench/otno/images/transports_sdf_ring.png')
    else:
        indices_encoder, transport = OT_mean_closest(log['lazy_plan'], target, device)
        indices_decoder = compute_min_indices_batched(transport, target)    
    
    # reset the transport as the closest point in target
    transport = target[indices_encoder].reshape(n_s,n_s,2)
    transport_sdf = target_sdf[indices_encoder].reshape(n_s,n_s,1)
    
    unique = len(torch.unique(indices_encoder))
    if unique!=len(target):
        non_surjective += 1
    #represent_points = transport.reshape(-1,2)[indices_decoder]
    #represent_sdf = transport_sdf.reshape(-1,1)[indices_decoder]
    #plot_sdf(represent_points.T.cpu(), represent_sdf.cpu(), 'flowbench/otno/images/represent_sdf_ring.png')
    #sys.exit(0)
    ones = torch.ones_like(transport_sdf.to(device))
    re = scale[i,0]
    mask = scale[i,2]
    input = torch.cat([source.reshape(n_s,n_s,2), transport, transport_sdf, ones*re, ones*mask], dim=2)
    print(i+1, default_timer()-t0, n_s, n_t, unique)
    #print(input.shape, out.shape, indices_decoder.shape)
    
    #all_points.append(target.to(dtype=torch.float32))
    #all_sdfs.append(target_sdf.to(dtype=torch.float32))
    #all_trans_points.append(transport.to(dtype=torch.float32))
    #all_trans_sdfs.append(transport_sdf.to(dtype=torch.float32))
    all_inputs.append(input.cpu().to(dtype=torch.float32))
    all_outs.append(out.cpu().to(dtype=torch.float32))
    #all_indices_encoder.append(indices_encoder)
    all_indices_decoder.append(indices_decoder.cpu())
    #print(input.shape, out.shape, indices_decoder.shape)
    #sys.exit(0)

torch.save({
        'inputs': all_inputs,
        'outs': all_outs,
        #'points': all_points,
        #'sdfs': all_sdfs,
        #'indices_encoder': all_indices_encoder,
        'indices_decoder': all_indices_decoder,
        #'trans_points': all_trans_points,
        #'trans_sdfs': all_trans_sdfs
        }, save_path)

print(non_surjective, default_timer()-tt)

#LDC_NS_2D_128_skelneton_expand3_reg1e-6 200s
#LDC_NS_2D_128_skelneton_expand2_reg1e-6 230.73352134902962
#128_nurbs_ring_expand3 20 270.46842645900324
#128bcombined square nurbs 980 220.5013703550212
#512 nurbs square 1000 17729.50675213989
#512 harmonics square 990 19498.212556011975
#512 skelneton ring 1000 1748.1444262778386
#512 harmonics ring 1000 1211.5715355048887