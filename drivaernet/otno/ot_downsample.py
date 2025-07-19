import ot
from ot.bregman import empirical_sinkhorn2_geomloss
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
import pyvista as pv
import pandas as pd
from timeit import default_timer
import open3d as o3d


def torus_grid(n_s_sqrt):
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    # Create a grid using meshgrid
    X, Y = torch.meshgrid(theta, phi, indexing='ij')
    points = torch.stack((X, Y)).reshape((2, -1)).T

    r = 1.0
    R = 1.5
    y = (R + r * torch.cos(points[:, 0])) * torch.cos(points[:, 1])
    z = (R + r * torch.cos(points[:, 0])) * torch.sin(points[:, 1])
    x = r * torch.sin(points[:, 0])
    
    return torch.stack((x, y, z), axis=1)

def torus_grid_with_normalized_density(n_s_sqrt, device):
    theta = torch.linspace(0, 2 * torch.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * torch.pi, n_s_sqrt + 1)[:-1]
    # Create a grid using meshgrid
    X, Y = torch.meshgrid(theta, phi, indexing='ij')
    points = torch.stack((X, Y)).reshape((2, -1)).T

    r = 0.5
    R = 1.0
    y = (R + r * torch.cos(points[:, 0])) * torch.cos(points[:, 1])
    z = (R + r * torch.cos(points[:, 0])) * torch.sin(points[:, 1])
    x = r * torch.sin(points[:, 0])
    grid = torch.stack((x, y, z), axis=1)
    
    # Calculate distances in the theta direction (along rows)
    theta_distances = R + r * torch.cos(X)
    theta_segment_lengths = theta_distances * (2 * torch.pi / n_s_sqrt)

    # Calculate distances in the phi direction (along columns)
    phi_segment_lengths = r * (2 * torch.pi / n_s_sqrt)

    # Calculate local area approximations
    local_areas = theta_segment_lengths * phi_segment_lengths
    total_area = local_areas.sum()
    
    # Normalize the density vector so that its sum is 1
    normalized_density_vector = local_areas.reshape(-1, ) / total_area
    
    return grid.to(device), normalized_density_vector.to(device)

def load_and_filter_csv_to_tensor(csv_file_path, design_ids):
    data = pd.read_csv(csv_file_path, index_col='Design')
    value_mapping = data['Average Cd'].to_dict()
    valid_ids = [id for id in design_ids if id in value_mapping]
    values = [value_mapping[id] for id in valid_ids]
    values_tensor = torch.tensor(values, dtype=torch.float32)
    return values_tensor, valid_ids

def OT_data_processor(subset_name, config, grid, device):
    all_downsamples, all_transports, all_downsample_normals, all_indices_encoder, all_indices_decoder = [], [], [], [], []
    non_surjective = 0
    file_path = os.path.join(config['subset_dir'], f"{subset_name}.txt")
    with open(file_path, 'r') as file:
        design_ids = [line.strip() for line in file.readlines()] 

    all_aero_coeff, valid_ids = load_and_filter_csv_to_tensor(config['aero_coeff'], design_ids) # load Cd

    for k,design_id in enumerate(valid_ids):
        
        tk = default_timer()
        file_name = f"{design_id}.stl"
        full_path = os.path.join(config['dataset_path'], file_name)
        try:
            pv_mesh = pv.read(full_path)
        except FileNotFoundError:
            print(f"File not found: {full_path}")
            continue
        vertices = np.asarray(pv_mesh.points)

        # Convert numpy array to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)

        # Voxel downsampling
        down_pcd = pcd.voxel_down_sample(config['voxel_size'])
        down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=config['voxel_size'] * 2, max_nn=30))

        # Extract downsampled points (target) and normals
        downsampled_points_np = np.asarray(down_pcd.points)
        downsampled_normals_np = np.asarray(down_pcd.normals)

        # Convert back to tensor if needed
        downsampled_points = torch.tensor(downsampled_points_np, dtype=torch.float32)
        downsampled_normals = torch.tensor(downsampled_normals_np, dtype=torch.float32)
        
        all_downsamples.append(downsampled_points)
        all_downsample_normals.append(downsampled_normals)
        
        ''' OT '''
        downsampled_points = downsampled_points.to(device)
        n_s_sqrt = int(torch.sqrt(torch.tensor(config['expand_factor'])) * torch.ceil(torch.sqrt(torch.tensor(len(downsampled_points)))))
        source, a = grid(n_s_sqrt, device)
        
        _, log = empirical_sinkhorn2_geomloss(X_s=source.to(dtype=torch.float32), X_t=downsampled_points.to(dtype=torch.float32), reg=config['reg'], log=True) #a=a.to(dtype=torch.float32), b=b.to(device)
        gamma = log['lazy_plan'][:].detach() # convert the lazy tensor to torch.tensor (dense)
       
        # normalize the OT plan matrix by column
        row_norms = torch.norm(gamma, p=1, dim=1, keepdim=True)
        gamma_encoder = gamma / row_norms

        # transport target to source
        transport = torch.mm(gamma_encoder, downsampled_points)

        # encoder: target -> source
        distances = torch.cdist(transport, downsampled_points)
        indices_encoder = torch.argmin(distances, dim=1) # find the closest point in "target" (car vertices) to each point in "transport" (latent grids)
      
        # reset the transport as the closest point in target
        transport = downsampled_points[indices_encoder]
        transport = transport.reshape(n_s_sqrt,n_s_sqrt,3).permute(2,0,1)
      
        #judge whether all points on car surface are used
        unique = len(torch.unique(indices_encoder))
        if unique!=len(downsampled_points):
            non_surjective += 1  

        # decoder: source -> target
        indices_decoder = torch.argmin(distances, dim=0) # # find the closest point in "transport" (latent grids) to each point in "target" (car vertices)
        
        all_indices_encoder.append(indices_encoder.cpu())
        all_indices_decoder.append(indices_decoder.cpu())
        all_transports.append(transport.to(dtype=torch.float32).cpu())
        print(k, default_timer()-tk, len(torch.unique(indices_encoder)), len(downsampled_points))
        
        
    torch.save({
        'Cd': all_aero_coeff,
        'points': all_downsamples,
        'normals': all_downsample_normals,
        'indices_encoder': all_indices_encoder,
        'indices_decoder': all_indices_decoder,
        'transports': all_transports
        }, os.path.join(save_path, f"{subset_name}.pt"))

    return non_surjective
    

    
if __name__ == '__main__':
    tt1 = default_timer()
    config = {
        'aero_coeff': '/your_path_to_drivaernet_dataset/AeroCoefficient_DrivAerNet_Filtered.csv',
        'subset_dir': '/your_path/train_val_test_splits',
        'dataset_path':  '/your_path_to_drivaernet_dataset/DrivAerNet_STLs_Combined',
        'expand_factor': 3.0,
        'reg': 1e-06,   
        'voxel_size': 0.05
    }
    save_path = '/your_save_path/STL200k_torusXpole_meanOTidx_voxelsize' + str(config['voxel_size']) + '_reg' + str(config['reg']) + '_expand' + str(config['expand_factor'])
    print(save_path)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    non_surjective = 0   
    for subset_name in ['train_design_ids', 'test_design_ids', 'val_design_ids']:
        non_surjective += OT_data_processor(subset_name=subset_name, config=config, grid=torus_grid_with_normalized_density, device=device)

    tt2 = default_timer()
    print(f"Total time: {tt2-tt1:.2f} seconds.")
    print(f"Non surjective plans: {non_surjective}")
