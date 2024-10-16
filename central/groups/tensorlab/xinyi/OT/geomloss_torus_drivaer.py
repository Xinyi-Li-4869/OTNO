import debugpy

# Listen for incoming connections from the debugger on a specified port
debugpy.listen(('0.0.0.0', 5678))  # 0.0.0.0 allows connections from any IP
print("Debugger is waiting for connection on port 5678...")
debugpy.wait_for_client()  # Blocks execution until the debugger is attached

import ot
from ot.bregman import empirical_sinkhorn2_geomloss
import numpy as np
#np.__config__.show()  # Show linked BLAS/LAPACK libraries
import matplotlib.pyplot as plt
import torch
import os
import pyvista as pv
from timeit import default_timer
from pykeops.torch import LazyTensor

def save_chunk(data, chunk_id, directory):
    file_path = os.path.join(directory, f"chunk_{chunk_id}.pt")
    torch.save(data, file_path)

def load_chunk(chunk_id, directory):
    file_path = os.path.join(directory, f"chunk_{chunk_id}.pt")
    return torch.load(file_path)
#TODO: topk_row
def topk_row_normalized_sparse_batch(lazy_tensor, k, batchsize, device):
    """
    Returns a sparse tensor with the same shape as the input lazy_tensor where each row contains the
    k largest elements normalized by the sum of these elements, with other elements set to zero.
    If a row has fewer than k non-zero elements, all are retained.
    
    Parameters:
    - lazy_tensor (LazyTensor): The LazyTensor object.
    - k (int): Number of top elements to keep in each row.
    - batchsize (int): Number of rows to process in each batch.
    - device (torch.device): The device on which to perform the computation.

    Returns:
    - torch.sparse.Tensor: A PyTorch sparse tensor with the same shape, where non-top-k elements in each row are zero.
    """
    num_rows, num_cols = lazy_tensor.shape
    all_indices = []
    all_values = []

    # Iterate through rows in batches
    for batch_start in range(0, num_rows, batchsize):
        batch_end = min(batch_start + batchsize, num_rows)
        row_values = lazy_tensor[batch_start:batch_end, :].clone().detach().to(device)  # Evaluate the batch of rows and detach from graph
        
        top_k_values, top_k_indices = torch.topk(row_values, k=k, dim=1)  # Get top k values and indices for each row in the batch
        
        # Compute row indices for this batch
        row_indices = torch.arange(batch_start, batch_end, device=device).repeat(k, 1)
        
        # Flatten and combine the indices
        batch_indices = torch.stack([row_indices, top_k_indices.t()], dim=0).view(2,-1)
        all_indices.append(batch_indices)
        all_values.append(top_k_values.view(-1))

    # Concatenate all indices and values across batches
    all_indices = torch.cat(all_indices, dim=1)
    all_values = torch.cat(all_values)

    # Create the sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(all_indices, all_values, (num_rows, num_cols), dtype=torch.double, device=device)

    # Normalize the rows
    row_sums = torch.sparse.sum(sparse_tensor, dim=1).to_dense()
    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor.values().div_(row_sums[sparse_tensor.indices()[0]])

    return sparse_tensor

def topk_column_normalized_sparse_batch(lazy_tensor, k, batchsize, device):
    """
    Returns a sparse tensor with the same shape as the input lazy_tensor where each column contains the
    k largest elements normalized by the sum of these elements, with other elements set to zero.
    If a column has fewer than k non-zero elements, all are retained.
    
    Parameters:
    - lazy_tensor (LazyTensor): The LazyTensor object.
    - k (int): Number of top elements to keep in each column.
    - batchsize (int): Number of columns to process in each batch.
    - device (torch.device): The device on which to perform the computation.

    Returns:
    - torch.sparse.Tensor: A PyTorch sparse tensor with the same shape, where non-top-k elements in each column are zero.
    """
    num_rows, num_cols = lazy_tensor.shape
    all_indices = []
    all_values = []

    # Iterate through columns in batches
    for batch_start in range(0, num_cols, batchsize):
        batch_end = min(batch_start + batchsize, num_cols)
        col_values = lazy_tensor[:, batch_start:batch_end].clone().detach().to(device)  # Evaluate the batch of columns and detach from graph
        
        top_k_values, top_k_indices = torch.topk(col_values, k=k, dim=0)  # Get top k values and indices for each column in the batch
        
        # Compute column indices for this batch
        col_indices = torch.arange(batch_start, batch_end, device=device).repeat(k, 1)
        
        # Flatten and combine the indices
        batch_indices = torch.stack([top_k_indices, col_indices], dim=0).view(2, -1)
        all_indices.append(batch_indices)
        all_values.append(top_k_values.view(-1))

    # Concatenate all indices and values across batches
    all_indices = torch.cat(all_indices, dim=1)
    all_values = torch.cat(all_values)

    # Create the sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(all_indices, all_values, (num_rows, num_cols), dtype=torch.double, device=device)

    # Normalize the columns
    column_sums = torch.sparse.sum(sparse_tensor, dim=0).to_dense()
    sparse_tensor = sparse_tensor.coalesce()
    sparse_tensor.values().div_(column_sums[sparse_tensor.indices()[1]])

    return sparse_tensor

def compute_transport_sparse(gamma, target):
    """
    Computes the transport matrix using a sparse representation of gamma.
    
    Parameters:
    - gamma (torch.sparse.Tensor): The sparse gamma matrix from topk_column_normalized_sparse.
    - target (torch.Tensor): The target points (n_t, 3).

    Returns:
    - np.ndarray: The transport matrix (n_s, 3).
    """
    n_s = gamma.shape[0]
    transport = torch.zeros((n_s, 3), dtype=target.dtype, device=target.device)
    
    # Coalesce the sparse matrix to ensure it is in a canonical format
    gamma = gamma.coalesce()
    indices = gamma.indices()
    values = gamma.values()
    
    # For each source index, find the target index with the maximum contribution
    current_row = None
    max_val = -torch.inf
    max_idx = -1
    
    for idx, (row, col) in enumerate(indices.t()):
        val = values[idx]
        if row != current_row:
            if current_row is not None:
                # Assign the max index target to the transport matrix
                transport[current_row] = target[max_idx]
            # Reset for the new row
            max_val = val
            max_idx = col
            current_row = row
        elif val > max_val:
            max_val = val
            max_idx = col
            
    # Assign the last row's max index target
    if current_row is not None:
        transport[current_row] = target[max_idx]
    
    return transport

tt1 = default_timer()
#path = '/home/nikola/HDD/car-cfd/data/'
path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/drivaer-net-dataset'

device = torch.device('cuda')
N = 611
expand_factor = 2.0
topk = 2
batchsize = 1500 #500 OOM P100; 200 P100 12min; 1000 H100 2min; 2000 H100 105s; 3000 OOM H100;
reg=0.0000001 #413460: 1e-7 125s(H100) 413415; 1e-6 10min(P100) 105s(H100) 413389; 1e-3 4min(P100) 47s(H100) 387336; 1e-1 1min(P100) 355270;
    
save_path = '/central/groups/tensorlab/xinyi/OT/featureDrivAer_torus_geomloss_reg' + str(reg) + '_topk' + str(topk) + '_expand' + str(expand_factor)
os.makedirs(save_path, exist_ok=True)
k=0
index=1

while k<N:
    t1 = default_timer()
    file_name = f"DrivAer_F_D_WM_WW_{str(index).zfill(4)}.vtk"
    full_path = os.path.join(path, file_name)
            
    if not os.path.exists(full_path): 
        index +=1
        continue 

    mesh = pv.read(full_path)
    # Compute normals to the mesh
    mesh_with_normals = mesh.compute_normals()
    normals = mesh_with_normals.point_normals
    curvature_gaussian = mesh.curvature(curv_type='gaussian')
    curvature_mean = mesh.curvature(curv_type='mean')

    target = torch.tensor(mesh.points,dtype=torch.float32) #(n_in,3) n_in=413460
    pressure = torch.tensor(mesh.point_data['p'],dtype=torch.float32) #(3586,)
    n_t = target.shape[0]
    print(n_t)
    if pressure.shape[0] != n_t:
        print(f"Error in dataset {index}: Expected {n_t} pressure data points, but got {pressure.shape[0]}")
        continue

    n_s_sqrt = int(torch.sqrt(torch.tensor(expand_factor)) * torch.ceil(torch.sqrt(torch.tensor(n_t))))
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    n_s = n_s_sqrt ** 2
    print(n_s)

    # Create a grid using meshgrid
    X, Y = torch.meshgrid(theta, phi, indexing='ij')
    points = torch.stack((X, Y)).reshape((2, -1)).T

    r = 1.0
    R = 1.5
    x = (R + r * torch.cos(points[:, 0])) * torch.cos(points[:, 1])
    y = (R + r * torch.cos(points[:, 0])) * torch.sin(points[:, 1])
    z = r * torch.sin(points[:, 0])

    source = torch.stack((x, y, z), axis=1)

    ts1 = default_timer()
    value, log = empirical_sinkhorn2_geomloss(source.to(dtype=torch.double, device=device), target.to(dtype=torch.double, device=device), reg, log=True) 
    ts2 = default_timer()
    print(f"Sinkhorn time: {ts2-ts1:.2f} seconds.")
    gamma = log['lazy_plan']
    
    gamma = topk_row_normalized_sparse_batch(gamma, topk, batchsize, device)
    ts3 = default_timer()
    print(f"Normalize time: {ts3-ts2:.2f} seconds.")
    indices = gamma.indices()    
    # Use the indices to count the occurrences of each row index
    unique_indices, counts = indices[1].unique(return_counts=True)   
    # Find the maximum count among the rows
    max_count = counts.max().item() 
    print(max_count)
    for repeats in range(max_count):
        count = (counts == repeats+1).sum().item()
        if count!=0:
            print(count, repeats+1)

    #transport = compute_transport_sparse(gamma, target)
    transport = torch.sparse.mm(gamma, target.to(dtype=torch.double, device=device))
    transport = transport.to_dense()
    ts4 = default_timer()
    print(f"Compute transport time: {ts4-ts3:.2f} seconds.")

    unique = torch.unique(transport, dim=0).shape[0]
    print(f"Compute unique time: {default_timer()-ts4:.2f} seconds.")

    transport = transport.T.reshape((3,n_s_sqrt,n_s_sqrt))

    print(unique, default_timer()-t1, k+1, index) #k+1:148 index:150

    data = {
        'points': target.to(dtype=torch.float32), 
        'pressures': pressure.to(dtype=torch.float32), 
        'couplings': gamma.to(dtype=torch.float32).to_sparse(),
        'normals': torch.tensor(normals,dtype=torch.float32),
        'curvatures': torch.tensor(curvature_gaussian,dtype=torch.float32)
        }
    save_chunk(data, k+1, save_path)

    k+=1
    index+=1

tt2 = default_timer()
print(f"Total time: {tt2-tt1:.2f} seconds.")
'''
T = 60
movement = np.zeros((T,n_s,3))

for j in range(n_s):
    tx = np.linspace(source[j,0], coupling[j,0], T).reshape((T,1))
    ty = np.linspace(source[j,1], coupling[j,1], T).reshape((T,1))
    tz = np.linspace(source[j,2], coupling[j,2], T).reshape((T,1))
    movement[:,j,:] = np.concatenate((tx,ty,tz), axis=1)


plotter = pv.Plotter()
plotter.open_movie('torus_to_car.mp4', framerate=20)
plotter.add_points(source, render_points_as_spheres=True, point_size=5, name='points')
plotter.camera_position = 'xy'
plotter.camera.azimuth= 135
plotter.camera.elevation= 30
#plotter.camera.zoom(0.9)

plotter.write_frame()
for j in range(1,T):
    plotter.add_points(movement[j,...], render_points_as_spheres=True, point_size=5, name='points')
    plotter.write_frame()

plotter.close()
'''