'''import debugpy

# Listen for incoming connections from the debugger on a specified port
debugpy.listen(('0.0.0.0', 5678))  # 0.0.0.0 allows connections from any IP
print("Debugger is waiting for connection on port 5678...")
debugpy.wait_for_client()  # Blocks execution until the debugger is attached'''
import sys
import ot
from ot.bregman import empirical_sinkhorn2_geomloss
import numpy as np
#np.__config__.show()  # Show linked BLAS/LAPACK libraries
#import pyvista as pv
import open3d as o3d
import matplotlib.pyplot as plt
import torch
from timeit import default_timer
from pykeops.torch import LazyTensor

def plot(x_in, out, path):
    x = x_in[:, 0]  # X coordinates
    y = x_in[:, 1]  # Y coordinates
    z = x_in[:, 2]  # Z coordinates
    values = out  # Values used for color mapping

    # Normalize the values for consistent scaling across different datasets
    range_values = values.max() - values.min()
    mid_values = (values.max() + values.min()) / 2.0
    values = (values - mid_values) / range_values  # Normalize values

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Sort points by depth to improve the visual effect of transparency
    order = np.argsort(z)
    x = x[order]
    y = y[order]
    z = z[order]
    values = values[order]

    # Create a scatter plot with smaller points and semi-transparency
    scatter = ax.scatter(x, y, z, c=values, cmap='viridis', alpha=0.6, s=15)

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Remove all axis labels and ticks
    #ax.set_axis_off()

    # Remove grid and box (by setting visibility to False)
    #ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.title('Random sampling')

    # Save the plot to a file without displaying it
    plt.savefig('/central/groups/tensorlab/xinyi/OT/images/' + path + '.png', format='png')
    plt.close()  # Close the plot window

    # Plot for input data
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(x, y, z, color='blue', alpha=0.2, s=15)  # Plotting as blue points
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    #ax.set_axis_off()
    #ax.grid(False)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.title('Random sampling')
    plt.savefig(f'/central/groups/tensorlab/xinyi/GINO-OT/image/{path}_mesh.png', format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

def topk_column_normalized(lazy_tensor, k, device):
    """
    Returns a tensor with the same shape as the input lazy_tensor where each column contains the
    k largest elements normalized by the sum of these elements, with other elements set to zero.
    
    Parameters:
    - lazy_tensor (LazyTensor): The LazyTensor object.
    - k (int): Number of top elements to keep in each column.

    Returns:
    - torch.Tensor: A PyTorch tensor with the same shape, where non-top-k elements in each column are zero.
    """
    num_rows = lazy_tensor.shape[0]
    num_cols = lazy_tensor.shape[1]
    full_matrix = torch.zeros((num_rows, num_cols), dtype=torch.double, device=device)  # Initialize a tensor with zeros of the same shape

    for j in range(num_cols):
        col_values = lazy_tensor[:, j].clone().detach()   # Evaluate the entire column j and detach from graph
        if col_values.ndim > 1:
            col_values = col_values.squeeze()  # Ensure it's a 1D tensor
        
        # Get the indices of the top k values in the column
        top_k_values, indices = torch.topk(col_values, k)
        
        # Place top k values in their respective positions in the full matrix
        full_matrix[indices, j] = top_k_values

    # Normalize the columns where the top k values are placed
    column_sums = full_matrix.sum(dim=0, keepdim=True)
    full_matrix = (full_matrix / column_sums).clone().detach()  # Normalize to make the sum of each column 1 and detach

    return full_matrix

def get_triangle_centroids(vertices: torch.Tensor, triangles: torch.Tensor):
    A, B, C = (
        vertices[triangles[:, 0]],
        vertices[triangles[:, 1]],
        vertices[triangles[:, 2]],
    )
    centroids = (A + B + C) / 3
    areas = torch.sqrt(torch.sum(torch.cross(B - A, C - A) ** 2, dim=1)) / 2
    return centroids, areas

def torus_grid(n_s_sqrt):
    theta = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    phi = torch.linspace(0, 2 * np.pi, n_s_sqrt + 1)[:-1]
    # Create a grid using meshgrid
    X, Y = torch.meshgrid(theta, phi, indexing='ij')
    points = torch.stack((X, Y)).reshape((2, -1)).T

    r = 1.0
    R = 1.5
    x = (R + r * torch.cos(points[:, 0])) * torch.cos(points[:, 1])
    y = (R + r * torch.cos(points[:, 0])) * torch.sin(points[:, 1])
    z = r * torch.sin(points[:, 0])
    
    return torch.stack((z, x, y), axis=1)

def ot_data_processor(data_path, save_name, N, expand_factor, reg, topk, device):
    tt1 = default_timer()
    
    all_points = []
    all_transports = []
    all_presures = []
    all_couplings = []
    all_normals = []
    all_velocity = []
    all_areas = []

    for k in range(N):
        t1 = default_timer()

        info = torch.load(data_path + 'info_' + str(k+1).zfill(3) + '.pt')
        velocity = info['velocity']
        all_velocity.append(torch.tensor(velocity, dtype=torch.float32))

        mesh = o3d.io.read_triangle_mesh(data_path + 'mesh_' + str(k+1).zfill(3) + '.ply')
        vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
        triangles = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long)

        # Calculate the centroids and areas of the triangles
        centroids, areas = get_triangle_centroids(vertices, triangles)
        all_points.append(centroids.to(dtype=torch.float32))
        all_areas.append(areas.to(dtype=torch.float32))

        pressure = np.load(data_path + 'press_' + str(k+1).zfill(3) + '.npy') 
        all_presures.append(torch.tensor(pressure, dtype=torch.float32))
        plot(centroids, pressure, 'ahmed')

        if len(pressure)!=len(centroids):
            print(f"{k}th mesh has problem.")

        n_t = len(centroids)
        n_s_sqrt = int(np.sqrt(expand_factor)*np.ceil(np.sqrt(n_t)))
        n_s = n_s_sqrt**2
        source = torus_grid(n_s_sqrt)
        
        value, log = empirical_sinkhorn2_geomloss(source.to(dtype=torch.double, device=device), centroids.to(dtype=torch.double, device=device), reg, log=True)
        gamma = log['lazy_plan']
        gamma = topk_column_normalized(gamma, topk, device)
        all_couplings.append(gamma.to_sparse().to(dtype=torch.float32))

        non_zero_counts = torch.count_nonzero(gamma, dim=1)
        unique, counts = non_zero_counts.unique(return_counts=True)

        # Calculate how many rows have each possible count of non-zero entries
        print("Non-zero counts per column:")
        for count, num_rows in zip(unique, counts):
            print(num_rows.item(), count.item())

        transport = torch.sparse.mm(gamma, centroids.to(dtype=torch.double, device=device))
        press = torch.sparse.mm(gamma, pressure.unsqueeze(1).to(dtype=torch.double, device=device))
        plot(transport.cpu(), press.cpu())
        transport = transport.T.reshape((3,n_s_sqrt,n_s_sqrt))
        all_transports.append(transport)


        print(n_t, default_timer()-t1, k+1)

    save_path = '/central/groups/tensorlab/xinyi/OT/ot-data/ahmed_torus' + '_reg' + str(reg) + '_columntopk' + str(topk) + '_expand' + str(expand_factor) + save_name +'.pt' 
    torch.save({'points': torch.from_numpy(all_points), 'transports': torch.from_numpy(all_transports), 'pressures': torch.from_numpy(all_presures), 'couplings': all_couplings}, save_path)
    tt2 = default_timer()
    print(f"Total time: {tt2-tt1:.2f} seconds.") #510.57 seconds.

device = torch.device('cuda')
train_path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/ahmed-body-dataset/train/'
test_path = '/central/groups/tensorlab/xinyi/GINO-main/datasets/ahmed-body-dataset/test/'
ot_data_processor(test_path, 'test', 51, 1.5, 1e-06, 1, device)